#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import numpy.linalg as LA
import shutil
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import math
import random

###############################################################################
# MODEL DEFINITIONS FROM THREE_STAGE_TRANSMITTANCE.PY
###############################################################################
class Shape2FilterModel(nn.Module):
    def __init__(self, d_in=3, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, 11 * 100)
        )
    
    def forward(self, shape_4x3):
        bsz = shape_4x3.size(0)
        presence = shape_4x3[:, :, 0]
        key_padding_mask = (presence < 0.5)
        x_proj = self.input_proj(shape_4x3)
        x_enc = self.encoder(x_proj, src_key_padding_mask=key_padding_mask)
        pres_sum = presence.sum(dim=1, keepdim=True) + 1e-8
        x_enc_w = x_enc * presence.unsqueeze(-1)
        shape_emb = x_enc_w.sum(dim=1) / pres_sum
        out_flat = self.mlp(shape_emb)
        out_2d = out_flat.view(bsz, 11, 100)
        # Apply sigmoid activation to constrain output to [0, 1] range
        out_2d = torch.sigmoid(out_2d)
        return out_2d

class Filter2ShapeVarLen(nn.Module):
    def __init__(self, d_in=100, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.row_preproc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 12)  # outputs: presence (4) and (x,y) for 4 points
        )
    def forward(self, spec_11x100):
        bsz = spec_11x100.size(0)
        x_r = spec_11x100.view(-1, spec_11x100.size(2))
        x_pre = self.row_preproc(x_r)
        x_pre = x_pre.view(bsz, -1, x_pre.size(-1))
        x_enc = self.encoder(x_pre)
        x_agg = x_enc.mean(dim=1)
        out_12 = self.mlp(x_agg)
        out_4x3 = out_12.view(bsz, 4, 3)
        presence_logits = out_4x3[:, :, 0]
        xy_raw = out_4x3[:, :, 1:]
        presence_list = []
        for i in range(4):
            if i == 0:
                presence_list.append(torch.ones(bsz, device=out_4x3.device, dtype=torch.float32))
            else:
                prob_i = torch.sigmoid(presence_logits[:, i]).clamp(1e-6, 1 - 1e-6)
                prob_chain = prob_i * presence_list[i - 1]
                ste_i = (prob_chain > 0.5).float() + prob_chain - prob_chain.detach()
                presence_list.append(ste_i)
        presence_stack = torch.stack(presence_list, dim=1)
        xy_bounded = torch.sigmoid(xy_raw)
        xy_final = xy_bounded * presence_stack.unsqueeze(-1)
        final_shape = torch.cat([presence_stack.unsqueeze(-1), xy_final], dim=-1)
        return final_shape

class Filter2ShapeFrozen(nn.Module):
    def __init__(self, filter2shape_net, shape2filter_frozen, no_grad_frozen=True):
        """
        no_grad_frozen: if True, the frozen shape2filter network is computed in a no_grad block.
                         For Stage C training, set this to False so gradients can flow.
        """
        super().__init__()
        self.filter2shape = filter2shape_net
        self.shape2filter_frozen = shape2filter_frozen
        self.no_grad_frozen = no_grad_frozen
        for p in self.filter2shape.parameters():
            p.requires_grad = False
        for p in self.shape2filter_frozen.parameters():
            p.requires_grad = False
    def forward(self, spec_input):
        shape_pred = self.filter2shape(spec_input)
        if self.no_grad_frozen:
            with torch.no_grad():
                spec_chain = self.shape2filter_frozen(shape_pred)
        else:
            spec_chain = self.shape2filter_frozen(shape_pred)
        return shape_pred, spec_chain

###############################################################################
# UTILITY FUNCTIONS
###############################################################################
def calculate_condition_number(filters):
    """
    Calculate condition number of the spectral filters matrix.
    
    Parameters:
    filters: Tensor of shape [11, 100] representing the spectral filters
    
    Returns:
    float: Condition number
    """
    # Convert to numpy for condition number calculation
    filters_np = filters.detach().cpu().numpy()
    
    # Use singular value decomposition to calculate condition number
    u, s, vh = LA.svd(filters_np)
    
    # Condition number is the ratio of largest to smallest singular value
    # Add small epsilon to prevent division by zero
    condition_number = s[0] / (s[-1] + 1e-10)
    
    return condition_number

def replicate_c4(points):
    """Replicate points with C4 symmetry"""
    c4 = []
    for (x, y) in points:
        c4.append([x, y])       # Q1: original
        c4.append([-y, x])      # Q2: rotate 90°
        c4.append([-x, -y])     # Q3: rotate 180°
        c4.append([y, -x])      # Q4: rotate 270°
    return np.array(c4, dtype=np.float32)

def sort_points_by_angle(points):
    """Sort points by angle from center for polygon drawing"""
    if len(points) < 3:
        return points
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    idx = np.argsort(angles)
    return points[idx]

def plot_shape_with_c4(shape, title, save_path):
    """Plot shape with C4 symmetry replication in a minimal academic style"""
    plt.figure(figsize=(5, 5))
    plt.xlim(-0.7, 0.7)  # Fixed limits as requested
    plt.ylim(-0.7, 0.7)
    
    # Extract active points
    presence = shape[:, 0] > 0.5
    active_points = shape[presence, 1:3]
    
    # Plot original Q1 points
    plt.scatter(shape[presence, 1], shape[presence, 2], color='red', s=50)
    
    # Apply C4 symmetry and plot the polygon
    if len(active_points) > 0:
        c4_points = replicate_c4(active_points)
        sorted_points = sort_points_by_angle(c4_points)
        
        # If we have enough points for a polygon
        if len(sorted_points) >= 3:
            # Close the polygon
            polygon = np.vstack([sorted_points, sorted_points[0]])
            plt.plot(polygon[:, 0], polygon[:, 1], 'k-', linewidth=1.5)
            plt.fill(polygon[:, 0], polygon[:, 1], 'lightblue', alpha=0.5)
        else:
            # Just plot the points
            plt.scatter(c4_points[:, 0], c4_points[:, 1], color='blue', alpha=0.4, s=30)
    
    plt.title(title, fontsize=12)
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Function to generate a fixed initial filter for use across all experiments
def generate_initial_filter(device=None):
    """
    Generate a fixed initial filter (11x100) to be used across all experiments.
    
    Parameters:
    device: torch.device to place the tensor on
    
    Returns:
    torch.Tensor: Initial filter of shape [11, 100]
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate initial filter with a specific pattern
    # Using a balanced pattern with some structure
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a base 11x100 filter
    filter_params = torch.zeros(11, 100, device=device)
    
    # Add some structured patterns for different rows
    for i in range(11):
        # Create different patterns for different rows
        if i % 3 == 0:  
            # High frequency pattern
            for j in range(100):
                filter_params[i, j] = 0.5 + 0.4 * np.sin(j/5 + i/3)
        elif i % 3 == 1:  
            # Medium frequency pattern
            for j in range(100):
                filter_params[i, j] = 0.5 + 0.4 * np.sin(j/10 + i/2)
        else:  
            # Low frequency pattern
            for j in range(100):
                filter_params[i, j] = 0.5 + 0.4 * np.sin(j/20 + i)
    
    # Add some random noise for diversity
    filter_params += 0.1 * torch.randn_like(filter_params)
    
    # Ensure all values are reasonable (avoiding extremely large values)
    filter_params = torch.clamp(filter_params, -2.0, 2.0)
    
    return filter_params

###############################################################################
# DATA LOADING FUNCTIONS
###############################################################################
def load_aviris_swir_data(swir_base_path="AVIRIS_SWIR_INTP", tile_size=100, cache_file=None, use_cache=False, folder_patterns="all"):
    """
    Load pre-processed AVIRIS_SWIR_INTP hyperspectral data from selected subfolders and crop it into tiles
    
    Parameters:
    swir_base_path: Base path to the AVIRIS_SWIR_INTP directory
    tile_size: Size of the tiles (square)
    cache_file: Path to the cache file to save/load the processed data
    use_cache: If True, try to load from cache file first
    folder_patterns: Comma-separated list of folder name patterns to include, or 'all' for all folders
    
    Returns:
    torch.Tensor of shape [num_tiles, tile_size, tile_size, num_bands]
    """
    import torch
    import numpy as np
    import os
    from tqdm import tqdm
    
    # If use_cache is True and cache_file exists, try to load from cache
    if use_cache and cache_file and os.path.exists(cache_file):
        print(f"Loading processed data from cache: {cache_file}")
        try:
            tiles_tensor = torch.load(cache_file)
            print(f"Loaded data from cache: {tiles_tensor.shape}")
            return tiles_tensor
        except Exception as e:
            print(f"Error loading cache file: {str(e)}")
            print("Falling back to processing raw data.")
    
    # Validate base path
    if not os.path.exists(swir_base_path) or not os.path.isdir(swir_base_path):
        raise FileNotFoundError(f"AVIRIS_SWIR_INTP directory not found at: {os.path.abspath(swir_base_path)}")
    
    print(f"Looking for AVIRIS_SWIR_INTP data in: {os.path.abspath(swir_base_path)}")
    
    # Get list of all subfolders
    all_subfolders = [f for f in os.listdir(swir_base_path) 
                    if os.path.isdir(os.path.join(swir_base_path, f))]
    
    if not all_subfolders:
        raise FileNotFoundError(f"No subfolders found in AVIRIS_SWIR_INTP directory: {os.path.abspath(swir_base_path)}")
    
    # Filter subfolders based on patterns
    if folder_patterns.lower() == "all":
        subfolders = all_subfolders
    else:
        patterns = [p.strip() for p in folder_patterns.split(',')]
        subfolders = []
        for subfolder in all_subfolders:
            if any(pattern in subfolder for pattern in patterns):
                subfolders.append(subfolder)
    
    if not subfolders:
        raise FileNotFoundError(f"No matching subfolders found for patterns: {folder_patterns}")
    
    print(f"Found {len(subfolders)} matching subfolders: {', '.join(subfolders)}")
    
    # List to store all loaded data tensors
    all_data_list = []
    
    # Process each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(swir_base_path, subfolder)
        print(f"\nProcessing subfolder: {subfolder}")
        
        # Try to find torch data first
        torch_path = os.path.join(subfolder_path, "torch", "aviris_swir.pt")
        numpy_path = os.path.join(subfolder_path, "numpy", "aviris_swir.npy")
        
        data = None
        
        # Try to load data from torch file
        if os.path.exists(torch_path):
            print(f"Loading PyTorch data from: {torch_path}")
            try:
                data = torch.load(torch_path)
                print(f"Loaded PyTorch data of shape: {data.shape}")
            except Exception as e:
                print(f"Error loading PyTorch file: {str(e)}")
                data = None
        
        # If torch data not available, try numpy
        if data is None and os.path.exists(numpy_path):
            print(f"Loading NumPy data from: {numpy_path}")
            try:
                data = torch.from_numpy(np.load(numpy_path))
                print(f"Loaded NumPy data of shape: {data.shape}")
            except Exception as e:
                print(f"Error loading NumPy file: {str(e)}")
                data = None
        
        # If neither torch nor numpy data available, try ENVI format
        if data is None:
            # Look for .hdr files
            hdr_files = [f for f in os.listdir(subfolder_path) if f.endswith('.hdr')]
            if hdr_files:
                try:
                    import spectral
                    hdr_path = os.path.join(subfolder_path, hdr_files[0])
                    print(f"Loading ENVI data from: {hdr_path}")
                    img = spectral.open_image(hdr_path)
                    data = torch.tensor(img.load())
                    print(f"Loaded ENVI data of shape: {data.shape}")
                except Exception as e:
                    print(f"Error loading ENVI file: {str(e)}")
                    data = None
        
        # If we couldn't load data from this subfolder, skip it
        if data is None:
            print(f"Could not load data from subfolder: {subfolder}")
            continue
        
        # Handle any no-data values in the data
        # Replace -9999 (common no-data value) with 0
        if torch.min(data) < -1000:
            data = torch.where(data < -1000, torch.zeros_like(data), data)
            print(f"Replaced no-data values. New min value: {torch.min(data)}")
        
        # Normalize the data to [0, 1] range
        data_min = torch.min(data)
        data_max = torch.max(data)
        data = (data - data_min) / (data_max - data_min)
        print(f"Data normalized to range [0, 1]. Min: {torch.min(data)}, Max: {torch.max(data)}")
        
        # Make sure data is in the right shape [height, width, bands]
        if len(data.shape) == 3:
            # Check if first dimension is spectral bands
            if data.shape[0] == 100:  # If first dimension is 100, it's [bands, height, width]
                data = data.permute(1, 2, 0)
                print(f"Rearranged data to shape: {data.shape}")
        
        # Calculate how many tiles we can extract
        h_tiles = data.shape[0] // tile_size
        w_tiles = data.shape[1] // tile_size
        total_tiles = h_tiles * w_tiles
        
        print(f"Creating {h_tiles}×{w_tiles} = {total_tiles} tiles of size {tile_size}×{tile_size}")
        
        # Create the tiles
        subfolder_tiles = []
        for i in range(h_tiles):
            for j in range(w_tiles):
                h_start = i * tile_size
                h_end = (i + 1) * tile_size
                w_start = j * tile_size
                w_end = (j + 1) * tile_size
                
                tile = data[h_start:h_end, w_start:w_end, :]
                subfolder_tiles.append(tile)
        
        # Convert to tensor and add to list
        if subfolder_tiles:
            subfolder_tensor = torch.stack(subfolder_tiles)
            print(f"Created tensor of shape {subfolder_tensor.shape} from subfolder {subfolder}")
            all_data_list.append(subfolder_tensor)
    
    # Combine data from all subfolders
    if not all_data_list:
        raise FileNotFoundError("Could not load data from any subfolder")
    
    # If we have multiple data tensors, concatenate them
    if len(all_data_list) > 1:
        # Check if all tensors have the same number of spectral bands
        num_bands = all_data_list[0].shape[-1]
        all_same_bands = all(tensor.shape[-1] == num_bands for tensor in all_data_list)
        
        if not all_same_bands:
            print("Warning: Data from different subfolders have different numbers of spectral bands.")
            print("Using only the spectral bands from the first subfolder for consistency.")
            # Only use the first subfolder
            tiles_tensor = all_data_list[0]
        else:
            # Concatenate along the first dimension (batch)
            tiles_tensor = torch.cat(all_data_list, dim=0)
            print(f"Combined data from {len(all_data_list)} subfolders: {tiles_tensor.shape}")
    else:
        # Only one subfolder was processed
        tiles_tensor = all_data_list[0]
    
    # Shuffle the tiles to mix data from different subfolders
    indices = torch.randperm(tiles_tensor.shape[0])
    tiles_tensor = tiles_tensor[indices]
    
    print(f"Final tiles tensor shape: {tiles_tensor.shape}")
    
    # If cache_file is provided, save the processed data
    if cache_file:
        print(f"Saving processed data to cache: {cache_file}")
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        try:
            torch.save(tiles_tensor, cache_file)
            print(f"Data saved to cache successfully.")
        except Exception as e:
            print(f"Error saving to cache: {str(e)}")
    
    return tiles_tensor

###############################################################################
# DECODER MODEL
###############################################################################
class DecoderCNN5Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(11, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 100, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.decoder(x)

###############################################################################
# MAIN AUTOENCODER MODEL WITH FILTER2SHAPE2FILTER ARCHITECTURE
###############################################################################
class HyperspectralAutoencoder(nn.Module):
    def __init__(self, shape2filter_path, filter2shape_path, target_snr=None, initial_filter_params=None):
        super().__init__()

        # Target SNR (in dB) as model parameter
        self.target_snr = target_snr
        
        # Device for model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the pretrained models
        self.shape2filter = Shape2FilterModel()
        self.shape2filter.load_state_dict(torch.load(shape2filter_path, map_location=self.device))
        self.shape2filter = self.shape2filter.to(self.device)  # Add this line
        
        self.filter2shape = Filter2ShapeVarLen()
        self.filter2shape.load_state_dict(torch.load(filter2shape_path, map_location=self.device))
        self.filter2shape = self.filter2shape.to(self.device)  # Add this line
        
        # Freeze both models
        for param in self.shape2filter.parameters():
            param.requires_grad = False
            
        for param in self.filter2shape.parameters():
            param.requires_grad = False
        
        # Create the filter2shape2filter pipeline
        self.pipeline = Filter2ShapeFrozen(self.filter2shape, self.shape2filter, no_grad_frozen=True)
        
        # # Initialize learnable filter parameters (11 x 100)
        # # Use provided initial parameters if available, otherwise generate new ones
        # if initial_filter_params is not None:
        #     self.filter_params = nn.Parameter(initial_filter_params.clone().to(self.device))
        # else:
        #     # Default to random initialization
        #     self.filter_params = nn.Parameter(torch.rand(11, 100, device=self.device))

        # Initialize learnable filter parameters (11 x 100)
        # Use provided initial parameters if available, otherwise generate new ones
        if initial_filter_params is not None:
            # Pass through pipeline to get reconstructed filters
            _, recon_filters = self.pipeline(initial_filter_params.unsqueeze(0).clone().to(self.device))
            self.filter_params = nn.Parameter(recon_filters[0])  # Use [0] to remove batch dimension
        else:
            # Default to random initialization
            init_random = torch.rand(1, 11, 100, device=self.device)
            _, recon_filters = self.pipeline(init_random)
            self.filter_params = nn.Parameter(recon_filters[0])  # Use [0] to remove batch dimension
        
        # Decoder: 3-layer convolutional network
        self.decoder = nn.Sequential(
            nn.Conv2d(11, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 100, kernel_size=3, padding=1)
        )
    
    def get_current_filter(self):
        """Return the current learnable filter parameters"""
        return self.filter_params
    
    def get_current_shape(self):
        """Get the current shape from filter2shape model"""
        filter = self.get_current_filter().unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            shape = self.filter2shape(filter)[0]  # Remove batch dimension
        return shape
    
    def get_reconstructed_filter(self):
        """Get the reconstructed filter from the full pipeline"""
        filter = self.get_current_filter().unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            _, recon_filter = self.pipeline(filter)
        return recon_filter[0]  # Remove batch dimension

    def add_noise(self, tensor):
        """
        Add Gaussian white noise to tensor, with noise level determined by self.target_snr.
        Signal power calculation uses detach() to prevent noise calculation from participating in gradient propagation.
        """
        if self.target_snr is None:
            return tensor
            
        # Calculate signal power (mean square), using detach() to separate calculation graph
        signal_power = tensor.detach().pow(2).mean()
        # Convert signal power to decibels
        signal_power_db = 10 * torch.log10(signal_power)
        # Calculate noise power (decibels)
        noise_power_db = signal_power_db - self.target_snr
        # Convert noise power back to linear scale
        noise_power = 10 ** (noise_power_db / 10)
        # Generate Gaussian white noise with the same shape as input tensor
        noise = torch.randn_like(tensor) * torch.sqrt(noise_power)
        # Return tensor with added noise
        return tensor + noise
    
    def forward(self, x):
        """
        Forward pass of the autoencoder
        Input: x - Hyperspectral data of shape [batch_size, height, width, 100]
        """
        # Get dimensions
        batch_size, height, width, spectral_bands = x.shape
        
        # Get current filter
        filter = self.get_current_filter()  # Shape: [11, 100]
        
        # Convert input from [B,H,W,C] to [B,C,H,W] format for PyTorch convolution
        x_channels_first = x.permute(0, 3, 1, 2)
        
        # Normalize filter for filtering
        filter_normalized = filter / 50.0  # Shape: [11, 100]
        
        # Use efficient tensor operations for spectral filtering
        # Einstein summation: 'bchw,oc->bohw'
        # This performs the weighted sum across spectral dimension for each output band
        encoded_channels_first = torch.einsum('bchw,oc->bohw', x_channels_first, filter_normalized)

        # ----------------- Add noise ------------------
        if self.target_snr is not None:
            encoded_channels_first = self.add_noise(encoded_channels_first)
        
        # Convert encoded data back to channels-last format [B,H,W,C]
        encoded = encoded_channels_first.permute(0, 2, 3, 1)
        
        # Decode: use the CNN decoder to expand from 11 to 100 bands
        decoded_channels_first = self.decoder(encoded_channels_first)
        
        # Convert back to original format [B,H,W,C]
        decoded = decoded_channels_first.permute(0, 2, 3, 1)
        
        return decoded, encoded

###############################################################################
# TRAINING FUNCTIONS
###############################################################################
def train_with_noise_level(shape2filter_path, filter2shape_path, output_dir, noise_level, 
                          initial_filter_params=None, batch_size=10, num_epochs=500, learning_rate=0.001, 
                          cache_file=None, use_cache=False, folder_patterns="all",
                          train_data=None, test_data=None):
    """
    Train and visualize the hyperspectral autoencoder with a specific noise level
    using filter2shape2filter architecture
    
    Parameters:
    shape2filter_path: Path to the pretrained shape2filter model
    filter2shape_path: Path to the pretrained filter2shape model
    output_dir: Directory to save outputs
    noise_level: SNR level in dB to apply during training
    initial_filter_params: Initial filter parameters (11x100)
    batch_size: Batch size for training
    num_epochs: Number of training epochs
    learning_rate: Learning rate for optimizer
    cache_file: Path to cache file for storing processed data
    use_cache: If True, try to load from cache file first
    folder_patterns: Comma-separated list of folder name patterns to include, or 'all' for all folders
    train_data: Optional training data, if None will be loaded
    test_data: Optional testing data, if None will be loaded
    
    Returns:
    tuple: (initial_shape_path, least_mse_shape_path, lowest_cn_shape_path, final_shape_path, output_dir)
    """
    # Create output directory with noise level
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    noise_dir = f"noise_{noise_level}dB"
    output_dir = os.path.join(output_dir, noise_dir + f"_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}/")
    
    # Create subfolder for intermediate visualizations
    viz_dir = os.path.join(output_dir, "intermediate_viz")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load or use provided data
    if train_data is None:
        data = load_aviris_swir_data(swir_base_path="AVIRIS_SWIR_INTP", tile_size=100, 
                                    cache_file=cache_file, use_cache=use_cache, folder_patterns=folder_patterns)
        data = data.to(device)
    else:
        data = train_data.to(device)
    
    print("Training data shape:", data.shape)
    
    # Create dataset and dataloader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size//2, shuffle=True)
    
    # If no initial filter parameters were provided, generate them
    if initial_filter_params is None:
        initial_filter_params = generate_initial_filter(device)
    
    # Initialize model with specified noise level and initial filter parameters
    model = HyperspectralAutoencoder(shape2filter_path, filter2shape_path, 
                                   target_snr=noise_level, initial_filter_params=initial_filter_params).to(device)
    
    print(f"Model initialized with noise level: {noise_level} dB and shared initial filter parameters")
    
    # Get initial filter, shape, and filters for visualization
    initial_filter = model.get_current_filter().detach().cpu().numpy()
    initial_shape = model.get_current_shape().detach().cpu().numpy()
    
    # Calculate initial condition number
    initial_condition_number = calculate_condition_number(model.get_reconstructed_filter().detach().cpu())
    print(f"Initial condition number: {initial_condition_number:.4f}")
    
    # Save initial shape with C4 symmetry
    initial_shape_path = f"{output_dir}/initial_shape.png"
    plot_shape_with_c4(initial_shape, f"Initial Shape", initial_shape_path)
    print(f"Initial shape saved to: {os.path.abspath(initial_shape_path)}")
    
    # Save initial filter
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(initial_filter[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Initial Spectral Parameters (SNR: {noise_level} dB)")
    plt.legend()
    initial_filter_path = f"{output_dir}/initial_filter.png"
    plt.savefig(initial_filter_path)
    plt.close()
    
    # Also save the reconstructed filter from the pipeline
    initial_recon_filter = model.get_reconstructed_filter().detach().cpu().numpy()
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(initial_recon_filter[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Initial Reconstructed Filter (SNR: {noise_level} dB)")
    plt.legend()
    initial_recon_filter_path = f"{output_dir}/initial_recon_filter.png"
    plt.savefig(initial_recon_filter_path)
    plt.close()
    
    # Before training, get reconstruction of sample from train set
    sample_idx = min(5, len(data) - 1)  # Make sure index is within range
    sample_tensor = data[sample_idx:sample_idx+1]  # Keep as tensor with batch dimension
    
    # If test data is provided, also get test sample
    test_sample_tensor = None
    if test_data is not None:
        test_sample_idx = min(5, len(test_data) - 1)
        test_sample_tensor = test_data[test_sample_idx:test_sample_idx+1].to(device)
    
    with torch.no_grad():
        initial_recon, encoded = model(sample_tensor)
        # Convert to numpy for visualization
        initial_recon_np = initial_recon.detach().cpu().numpy()[0]
        encoded_np = encoded.detach().cpu().numpy()[0]
        sample_np = sample_tensor.detach().cpu().numpy()[0]
        
        # Calculate initial MSE for train
        initial_train_mse = ((initial_recon - sample_tensor) ** 2).mean().item()
        print(f"Initial Train MSE: {initial_train_mse:.6f}")
        
        # Calculate initial MSE for test if available
        initial_test_mse = None
        if test_sample_tensor is not None:
            test_recon, _ = model(test_sample_tensor)
            initial_test_mse = ((test_recon - test_sample_tensor) ** 2).mean().item()
            print(f"Initial Test MSE: {initial_test_mse:.6f}")
    
    # Training loop
    losses = []
    condition_numbers = [initial_condition_number]
    train_mse_values = [initial_train_mse]
    test_mse_values = [initial_test_mse] if initial_test_mse is not None else []
    
    # Add variables to track the lowest condition number and corresponding shape
    lowest_condition_number = float('inf')
    lowest_cn_shape = None
    lowest_cn_filter = None
    lowest_cn_epoch = -1
    
    # Add variables to track the lowest MSE and corresponding shape
    lowest_train_mse = float('inf')
    lowest_mse_shape = None
    lowest_mse_filter = None
    lowest_mse_epoch = -1
    
    print(f"Starting training with SNR: {noise_level} dB...")
    
    # Create optimizer for just the filter parameters
    optimizer = optim.Adam([model.filter_params], lr=learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = batch[0]
            
            # Forward pass
            recon, _ = model(x)
            
            # Calculate loss
            loss = criterion(recon, x)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss for epoch
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Calculate condition number for current epoch
        current_filter = model.get_reconstructed_filter().detach().cpu()
        current_condition_number = calculate_condition_number(current_filter)
        condition_numbers.append(current_condition_number)
        
        # Get current shape for visualization
        current_shape = model.get_current_shape().detach().cpu().numpy()
        current_filter_raw = model.get_current_filter().detach().cpu()
        
        # Check if this is the lowest condition number so far
        if current_condition_number < lowest_condition_number:
            lowest_condition_number = current_condition_number
            lowest_cn_shape = current_shape.copy()
            lowest_cn_filter = current_filter.clone()
            lowest_cn_epoch = epoch
            print(f"New lowest condition number: {lowest_condition_number:.4f} at epoch {epoch+1}")
        
        # Calculate current MSE for train
        with torch.no_grad():
            current_train_recon, _ = model(sample_tensor)
            current_train_mse = ((current_train_recon - sample_tensor) ** 2).mean().item()
            train_mse_values.append(current_train_mse)
            
            # Check if this is the lowest MSE so far
            if current_train_mse < lowest_train_mse:
                lowest_train_mse = current_train_mse
                lowest_mse_shape = current_shape.copy()
                lowest_mse_filter = current_filter_raw.clone()
                lowest_mse_epoch = epoch
                print(f"New lowest train MSE: {lowest_train_mse:.6f} at epoch {epoch+1}")
            
            # Calculate current MSE for test if available
            if test_sample_tensor is not None:
                current_test_recon, _ = model(test_sample_tensor)
                current_test_mse = ((current_test_recon - test_sample_tensor) ** 2).mean().item()
                test_mse_values.append(current_test_mse)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Condition Number: {current_condition_number:.4f}, "
                      f"Train MSE: {current_train_mse:.6f}, Test MSE: {current_test_mse:.6f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Condition Number: {current_condition_number:.4f}, "
                      f"Train MSE: {current_train_mse:.6f}")
        
        # Save intermediate shapes and visualizations
        if (epoch+1) % (num_epochs // 10) == 0 or epoch == 0:
            # Save shape visualization to intermediate directory
            plot_shape_with_c4(
                current_shape, 
                f"Shape at Epoch {epoch+1}", 
                f"{viz_dir}/shape_epoch_{epoch+1}.png"
            )
            
            # Save filter visualization
            plt.figure(figsize=(10, 6))
            for i in range(11):
                plt.plot(current_filter.numpy()[i], label=f'Filter {i}' if i % 3 == 0 else None)
            plt.grid(True)
            plt.xlabel("Wavelength Index")
            plt.ylabel("Filter Value")
            plt.title(f"Filter at Epoch {epoch+1}")
            plt.legend()
            plt.savefig(f"{viz_dir}/filter_epoch_{epoch+1}.png")
            plt.close()
    
    # Get final filter and shape
    final_filter = model.get_current_filter().detach().cpu()
    final_shape = model.get_current_shape().detach().cpu().numpy()
    
    # Calculate final condition number
    final_condition_number = calculate_condition_number(model.get_reconstructed_filter().detach().cpu())
    print(f"Final condition number: {final_condition_number:.4f}")
    
    # Save final shape
    final_shape_path = f"{output_dir}/final_shape.png"
    plot_shape_with_c4(final_shape, f"Final Shape", final_shape_path)
    print(f"Final shape saved to: {os.path.abspath(final_shape_path)}")
    
    # Save final filter
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(final_filter.numpy()[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Final Spectral Parameters (SNR: {noise_level} dB)")
    plt.legend()
    final_filter_path = f"{output_dir}/final_filter.png"
    plt.savefig(final_filter_path)
    plt.close()
    
    # Also save the reconstructed filter from the pipeline
    final_recon_filter = model.get_reconstructed_filter().detach().cpu().numpy()
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(final_recon_filter[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Final Reconstructed Filter (SNR: {noise_level} dB)")
    plt.legend()
    final_recon_filter_path = f"{output_dir}/final_recon_filter.png"
    plt.savefig(final_recon_filter_path)
    plt.close()
    
    # After training, get reconstruction of the same sample
    with torch.no_grad():
        final_train_recon, _ = model(sample_tensor)
        final_train_mse = ((final_train_recon - sample_tensor) ** 2).mean().item()
        print(f"Final Train MSE: {final_train_mse:.6f}")
        
        final_test_mse = None
        if test_sample_tensor is not None:
            final_test_recon, _ = model(test_sample_tensor)
            final_test_mse = ((final_test_recon - test_sample_tensor) ** 2).mean().item()
            print(f"Final Test MSE: {final_test_mse:.6f}")
    
    # Create a directory for the lowest condition number results
    lowest_cn_dir = os.path.join(output_dir, "lowest_cn")
    os.makedirs(lowest_cn_dir, exist_ok=True)
    
    # Save the lowest condition number shape
    lowest_cn_shape_path = f"{lowest_cn_dir}/shape.png"
    plot_shape_with_c4(lowest_cn_shape, f"Shape with Lowest CN: {lowest_condition_number:.4f}", lowest_cn_shape_path)
    
    # Save the lowest condition number filter
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(lowest_cn_filter.numpy()[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Spectral Parameters with Lowest Condition Number: {lowest_condition_number:.4f}")
    plt.legend()
    lowest_cn_filter_path = f"{lowest_cn_dir}/filter.png"
    plt.savefig(lowest_cn_filter_path)
    plt.close()
    
    # Create a directory for the lowest MSE results
    lowest_mse_dir = os.path.join(output_dir, "lowest_mse")
    os.makedirs(lowest_mse_dir, exist_ok=True)
    
    # Save the lowest MSE shape
    lowest_mse_shape_path = f"{lowest_mse_dir}/shape.png"
    plot_shape_with_c4(lowest_mse_shape, f"Shape with Lowest MSE: {lowest_train_mse:.6f}", lowest_mse_shape_path)
    
    # Save the lowest MSE filter
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(lowest_mse_filter.numpy()[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Spectral Parameters with Lowest MSE: {lowest_train_mse:.6f}")
    plt.legend()
    lowest_mse_filter_path = f"{lowest_mse_dir}/filter.png"
    plt.savefig(lowest_mse_filter_path)
    plt.close()
    
    # Save the lowest condition number shape as numpy file
    lowest_cn_shape_npy_path = f"{lowest_cn_dir}/shape.npy"
    np.save(lowest_cn_shape_npy_path, lowest_cn_shape)
    
    # Save the lowest condition number filter as numpy file
    lowest_cn_filter_npy_path = f"{lowest_cn_dir}/filter.npy"
    np.save(lowest_cn_filter_npy_path, lowest_cn_filter.numpy())
    
    # Save the lowest MSE shape as numpy file
    lowest_mse_shape_npy_path = f"{lowest_mse_dir}/shape.npy"
    np.save(lowest_mse_shape_npy_path, lowest_mse_shape)
    
    # Save the lowest MSE filter as numpy file
    lowest_mse_filter_npy_path = f"{lowest_mse_dir}/filter.npy"
    np.save(lowest_mse_filter_npy_path, lowest_mse_filter.numpy())
    
    # Plot condition number during training
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs+1), condition_numbers, 'r-')
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Condition Number")
    plt.title(f"Filter Matrix Condition Number During Training (SNR: {noise_level} dB)")
    plt.axhline(y=lowest_condition_number, color='g', linestyle='--', 
                label=f'Lowest CN: {lowest_condition_number:.4f} (Epoch {lowest_cn_epoch+1})')
    plt.legend()
    condition_plot_path = f"{output_dir}/condition_number.png"
    plt.savefig(condition_plot_path)
    plt.close()
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Loss (SNR: {noise_level} dB)")
    loss_plot_path = f"{output_dir}/training_loss.png"
    plt.savefig(loss_plot_path)
    plt.close()
    
    # Plot MSE values
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs+1), train_mse_values, 'b-', label='Train MSE')
    plt.axhline(y=lowest_train_mse, color='c', linestyle='--', 
              label=f'Lowest MSE: {lowest_train_mse:.6f} (Epoch {lowest_mse_epoch+1})')
    if test_mse_values:
        plt.plot(range(num_epochs+1), test_mse_values, 'r--', label='Test MSE')
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"MSE of Reconstruction and Input (SNR: {noise_level} dB)")
    plt.legend()
    mse_plot_path = f"{output_dir}/mse_values.png"
    plt.savefig(mse_plot_path)
    plt.close()
    
    # Save model parameters
    model_save_path = f"{output_dir}/model_state.pt"
    torch.save({
        'filter_params': model.filter_params.detach().cpu(),
        'decoder_state_dict': model.decoder.state_dict()
    }, model_save_path)
    
    # Also save numerical data
    np.save(f"{output_dir}/initial_shape.npy", initial_shape)
    np.save(f"{output_dir}/final_shape.npy", final_shape)
    np.save(f"{output_dir}/initial_filter.npy", initial_filter)
    np.save(f"{output_dir}/final_filter.npy", final_filter.numpy())
    np.save(f"{output_dir}/condition_numbers.npy", np.array(condition_numbers))
    np.save(f"{output_dir}/losses.npy", np.array(losses))
    np.save(f"{output_dir}/train_mse_values.npy", np.array(train_mse_values))
    if test_mse_values:
        np.save(f"{output_dir}/test_mse_values.npy", np.array(test_mse_values))
    
    # Create log file to save training parameters
    with open(f"{output_dir}/training_params.txt", "w") as f:
        f.write(f"Noise level (SNR): {noise_level} dB\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Cache file: {cache_file}\n")
        f.write(f"Used cache: {use_cache}\n")
        f.write(f"Folder patterns: {folder_patterns}\n")
        f.write("\n")
        
        # Save condition number information
        f.write(f"Initial condition number: {initial_condition_number:.4f}\n")
        f.write(f"Final condition number: {final_condition_number:.4f}\n")
        f.write(f"Lowest condition number: {lowest_condition_number:.4f} at epoch {lowest_cn_epoch+1}\n")
        f.write(f"Condition number change: {final_condition_number - initial_condition_number:.4f}\n\n")
        
        # Save MSE information
        f.write(f"Initial Train MSE: {initial_train_mse:.6f}\n")
        f.write(f"Final Train MSE: {final_train_mse:.6f}\n")
        f.write(f"Lowest Train MSE: {lowest_train_mse:.6f} at epoch {lowest_mse_epoch+1}\n")
        f.write(f"Train MSE improvement: {initial_train_mse - final_train_mse:.6f} ({(1 - final_train_mse/initial_train_mse) * 100:.2f}%)\n")
        
        if initial_test_mse is not None and final_test_mse is not None:
            f.write(f"Initial Test MSE: {initial_test_mse:.6f}\n")
            f.write(f"Final Test MSE: {final_test_mse:.6f}\n")
            f.write(f"Test MSE improvement: {initial_test_mse - final_test_mse:.6f} ({(1 - final_test_mse/initial_test_mse) * 100:.2f}%)\n")
    
    print(f"Training with SNR {noise_level} dB completed. All results saved to {output_dir}/")
    
    # Return paths to all shapes for later training
    initial_shape_npy_path = f"{output_dir}/initial_shape.npy"
    final_shape_npy_path = f"{output_dir}/final_shape.npy"
    lowest_cn_shape_npy_path = f"{lowest_cn_dir}/shape.npy"
    lowest_mse_shape_npy_path = f"{lowest_mse_dir}/shape.npy"
    
    return initial_shape_npy_path, lowest_mse_shape_npy_path, lowest_cn_shape_npy_path, final_shape_npy_path, output_dir

def train_decoder_with_comparison(shape2filter_path, filter2shape_path, shapes_dict, output_dir, noise_level, 
                                 num_epochs=100, batch_size=10, learning_rate=0.001, 
                                 cache_file=None, use_cache=False, folder_patterns="all", 
                                 train_data=None, test_data=None):
    """
    Train decoders for all shapes (initial, lowest MSE, lowest CN, final) with train/test split
    
    Parameters:
    shape2filter_path: Path to the pretrained shape2filter model
    filter2shape_path: Path to the pretrained filter2shape model
    shapes_dict: Dictionary with paths to 'initial', 'lowest_mse', 'lowest_cn', and 'final' shapes
    output_dir: Directory to save outputs
    noise_level: SNR level in dB
    num_epochs: Number of training epochs
    batch_size: Batch size for training
    learning_rate: Learning rate for optimizer
    cache_file: Path to cache file for processed data
    use_cache: If True, try to load from cache file first
    folder_patterns: Comma-separated list of folder name patterns to include, or 'all' for all folders
    train_data: Optional training data tensor
    test_data: Optional testing data tensor
    
    Returns:
    dict: Dictionary of final MSE values for each shape
    """
    # Create output directory
    decoder_dir = os.path.join(output_dir, f"decoder_comparison_noise_{noise_level}dB")
    os.makedirs(decoder_dir, exist_ok=True)
    print(f"Saving decoder comparison results to {decoder_dir}/")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load shapes
    shapes = {}
    for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']:
        # Check if shape exists in dictionary
        if shape_type in shapes_dict and os.path.exists(shapes_dict[shape_type]):
            shapes[shape_type] = np.load(shapes_dict[shape_type])
            print(f"Loaded {shape_type} shape from: {shapes_dict[shape_type]}")
        else:
            print(f"Warning: {shape_type} shape not found in dictionary or file doesn't exist")
    
    # If no shapes were loaded successfully, raise an error
    if not shapes:
        raise ValueError("No valid shapes were found in the provided dictionary")
    
    # Load data if not provided
    if train_data is None or test_data is None:
        data = load_aviris_swir_data(swir_base_path="AVIRIS_SWIR_INTP", tile_size=100, 
                                     cache_file=cache_file, use_cache=use_cache, folder_patterns=folder_patterns)
        
        # Split data into training and testing sets (80% train, 20% test)
        num_samples = data.shape[0]
        indices = torch.randperm(num_samples)
        train_size = int(0.8 * num_samples)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_data = data[train_indices].to(device)
        test_data = data[test_indices].to(device)
        
        print(f"Data split into {train_data.shape[0]} training and {test_data.shape[0]} testing samples")
    else:
        train_data = train_data.to(device)
        test_data = test_data.to(device)
    
    # Create dataloaders
    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size//2, shuffle=True)
    
    # Sample for evaluation
    train_sample_idx = min(5, len(train_data) - 1)
    train_sample = train_data[train_sample_idx:train_sample_idx+1]
    
    test_sample_idx = min(5, len(test_data) - 1)
    test_sample = test_data[test_sample_idx:test_sample_idx+1]
    
    # Dictionary to store MSE values for each shape type
    mse_values = {shape_type: {'train': [], 'test': []} for shape_type in shapes}
    
    # Dictionary to store models and optimizers
    models = {}
    optimizers = {}
    
    # Dictionaries to track best MSE values for each shape type
    best_mse_values = {shape_type: {'train': float('inf'), 'test': float('inf')} for shape_type in shapes}
    best_decoders = {shape_type: None for shape_type in shapes}
    best_epoch = {shape_type: -1 for shape_type in shapes}
    
    # Initialize models for each shape type
    for shape_type in shapes:
        # Initialize model
        models[shape_type] = HyperspectralAutoencoder(shape2filter_path, filter2shape_path, target_snr=noise_level).to(device)
        
        # Create a fixed shape tensor
        saved_shape_tensor = torch.tensor(shapes[shape_type], dtype=torch.float32, device=device)
        
        # Replace the get_current_shape method to use the fixed shape
        original_get_shape = models[shape_type].get_current_shape
        models[shape_type].get_current_shape = lambda saved_tensor=saved_shape_tensor: saved_tensor
        
        # Create a new decoder CNN
        new_decoder = DecoderCNN5Layer().to(device)
        
        # Replace the original decoder with the new one
        models[shape_type].decoder = new_decoder.decoder
        
        # Create optimizer for the decoder parameters only
        # Freeze the filter parameters (we're only training the decoder)
        models[shape_type].filter_params.requires_grad = False
        optimizers[shape_type] = optim.Adam(models[shape_type].decoder.parameters(), lr=learning_rate)
        
        # Get initial MSE with the new decoder (before training)
        with torch.no_grad():
            initial_train_recon, _ = models[shape_type](train_sample)
            initial_train_mse = ((initial_train_recon - train_sample) ** 2).mean().item()
            mse_values[shape_type]['train'].append(initial_train_mse)
            
            initial_test_recon, _ = models[shape_type](test_sample)
            initial_test_mse = ((initial_test_recon - test_sample) ** 2).mean().item()
            mse_values[shape_type]['test'].append(initial_test_mse)
            
            print(f"Initial {shape_type} shape - Train MSE: {initial_train_mse:.6f}, Test MSE: {initial_test_mse:.6f}")
    
    # Training criterion
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        # Train each model for one epoch
        for shape_type in shapes:
            epoch_loss = 0.0
            num_batches = 0
            
            # Set model to training mode
            models[shape_type].train()
            
            for batch in train_dataloader:
                x = batch[0]
                
                # Forward pass
                recon, _ = models[shape_type](x)
                
                # Calculate loss
                loss = criterion(recon, x)
                
                # Backward pass and optimize
                optimizers[shape_type].zero_grad()
                loss.backward()
                optimizers[shape_type].step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Calculate average loss for epoch
            avg_loss = epoch_loss / num_batches
            
            # Evaluate on train and test samples
            models[shape_type].eval()
            with torch.no_grad():
                # Train sample
                train_recon, _ = models[shape_type](train_sample)
                train_mse = ((train_recon - train_sample) ** 2).mean().item()
                mse_values[shape_type]['train'].append(train_mse)
                
                # Test sample
                test_recon, _ = models[shape_type](test_sample)
                test_mse = ((test_recon - test_sample) ** 2).mean().item()
                mse_values[shape_type]['test'].append(test_mse)
                
                # Check if this is the best test MSE so far
                if test_mse < best_mse_values[shape_type]['test']:
                    best_mse_values[shape_type]['test'] = test_mse
                    best_mse_values[shape_type]['train'] = train_mse
                    best_decoders[shape_type] = models[shape_type].decoder.state_dict()
                    best_epoch[shape_type] = epoch
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}:")
            for shape_type in shapes:
                print(f"  {shape_type.capitalize()} shape - Train MSE: {mse_values[shape_type]['train'][-1]:.6f}, "
                      f"Test MSE: {mse_values[shape_type]['test'][-1]:.6f}")
    
    # Create directories for plots
    plots_dir = os.path.join(decoder_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Plot combined MSE curves for all shape types - Final values
    plt.figure(figsize=(12, 8))
    
    # Training MSE - Final
    plt.subplot(1, 2, 1)
    for shape_type in shapes:
        plt.plot(range(num_epochs+1), mse_values[shape_type]['train'], 
                 label=f"{shape_type.capitalize()} Shape")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"Training MSE (SNR: {noise_level} dB)")
    plt.legend()
    
    # Testing MSE - Final
    plt.subplot(1, 2, 2)
    for shape_type in shapes:
        plt.plot(range(num_epochs+1), mse_values[shape_type]['test'], 
                 label=f"{shape_type.capitalize()} Shape")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"Testing MSE (SNR: {noise_level} dB)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/shape_comparison_final_mse.png", dpi=300)
    plt.close()
    
    # Create bar chart comparison of final and best MSE values
    plt.figure(figsize=(14, 10))
    
    # Final train MSE comparison
    plt.subplot(2, 2, 1)
    shape_types = list(shapes.keys())
    final_train_mse = [mse_values[shape_type]['train'][-1] for shape_type in shape_types]
    x_pos = range(len(shape_types))
    plt.bar(x_pos, final_train_mse)
    plt.xticks(x_pos, [s.capitalize() for s in shape_types], rotation=45)
    plt.ylabel('MSE')
    plt.title('Final Train MSE by Shape Type')
    plt.grid(axis='y')
    
    # Final test MSE comparison
    plt.subplot(2, 2, 2)
    final_test_mse = [mse_values[shape_type]['test'][-1] for shape_type in shape_types]
    plt.bar(x_pos, final_test_mse)
    plt.xticks(x_pos, [s.capitalize() for s in shape_types], rotation=45)
    plt.ylabel('MSE')
    plt.title('Final Test MSE by Shape Type')
    plt.grid(axis='y')
    
    # Best train MSE comparison
    plt.subplot(2, 2, 3)
    best_train_mse = [best_mse_values[shape_type]['train'] for shape_type in shape_types]
    plt.bar(x_pos, best_train_mse)
    plt.xticks(x_pos, [s.capitalize() for s in shape_types], rotation=45)
    plt.ylabel('MSE')
    plt.title('Best Train MSE by Shape Type')
    plt.grid(axis='y')
    
    # Best test MSE comparison
    plt.subplot(2, 2, 4)
    best_test_mse = [best_mse_values[shape_type]['test'] for shape_type in shape_types]
    plt.bar(x_pos, best_test_mse)
    plt.xticks(x_pos, [s.capitalize() for s in shape_types], rotation=45)
    plt.ylabel('MSE')
    plt.title('Best Test MSE by Shape Type')
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/shape_comparison_bar_charts.png", dpi=300)
    plt.close()
    
    # Save the MSE values
    for shape_type in shapes:
        np.save(f"{decoder_dir}/{shape_type}_train_mse.npy", np.array(mse_values[shape_type]['train']))
        np.save(f"{decoder_dir}/{shape_type}_test_mse.npy", np.array(mse_values[shape_type]['test']))
    
    # Calculate final MSE values
    final_mse_values = {}
    best_mse_epoch_values = {}
    
    for shape_type in shapes:
        final_mse_values[shape_type] = {
            'train': mse_values[shape_type]['train'][-1],
            'test': mse_values[shape_type]['test'][-1]
        }
        
        best_mse_epoch_values[shape_type] = {
            'train': best_mse_values[shape_type]['train'],
            'test': best_mse_values[shape_type]['test'],
            'epoch': best_epoch[shape_type]
        }
        
        # Save final model
        torch.save(models[shape_type].decoder.state_dict(), f"{decoder_dir}/{shape_type}_final_decoder_state.pt")
        
        # Save best model
        if best_decoders[shape_type] is not None:
            torch.save(best_decoders[shape_type], f"{decoder_dir}/{shape_type}_best_decoder_state.pt")
    
    # Create log file to save results
    with open(f"{decoder_dir}/decoder_comparison_results.txt", "w") as f:
        f.write(f"Noise level (SNR): {noise_level} dB\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Training samples: {train_data.shape[0]}\n")
        f.write(f"Testing samples: {test_data.shape[0]}\n\n")
        
        f.write("Initial MSE values:\n")
        for shape_type in shapes:
            f.write(f"  {shape_type.capitalize()} shape - Train: {mse_values[shape_type]['train'][0]:.6f}, "
                    f"Test: {mse_values[shape_type]['test'][0]:.6f}\n")
        
        f.write("\nFinal MSE values:\n")
        for shape_type in shapes:
            f.write(f"  {shape_type.capitalize()} shape - Train: {mse_values[shape_type]['train'][-1]:.6f}, "
                    f"Test: {mse_values[shape_type]['test'][-1]:.6f}\n")
        
        f.write("\nBest MSE values:\n")
        for shape_type in shapes:
            f.write(f"  {shape_type.capitalize()} shape - Train: {best_mse_values[shape_type]['train']:.6f}, "
                    f"Test: {best_mse_values[shape_type]['test']:.6f} at epoch {best_epoch[shape_type]+1}\n")
        
        f.write("\nMSE improvements (initial to final):\n")
        for shape_type in shapes:
            train_improvement = mse_values[shape_type]['train'][0] - mse_values[shape_type]['train'][-1]
            train_percent = (1 - mse_values[shape_type]['train'][-1] / mse_values[shape_type]['train'][0]) * 100
            
            test_improvement = mse_values[shape_type]['test'][0] - mse_values[shape_type]['test'][-1]
            test_percent = (1 - mse_values[shape_type]['test'][-1] / mse_values[shape_type]['test'][0]) * 100
            
            f.write(f"  {shape_type.capitalize()} shape - Train: {train_improvement:.6f} ({train_percent:.2f}%), "
                    f"Test: {test_improvement:.6f} ({test_percent:.2f}%)\n")
            
        f.write("\nMSE improvements (initial to best):\n")
        for shape_type in shapes:
            train_improvement = mse_values[shape_type]['train'][0] - best_mse_values[shape_type]['train']
            train_percent = (1 - best_mse_values[shape_type]['train'] / mse_values[shape_type]['train'][0]) * 100
            
            test_improvement = mse_values[shape_type]['test'][0] - best_mse_values[shape_type]['test']
            test_percent = (1 - best_mse_values[shape_type]['test'] / mse_values[shape_type]['test'][0]) * 100
            
            f.write(f"  {shape_type.capitalize()} shape - Train: {train_improvement:.6f} ({train_percent:.2f}%), "
                    f"Test: {test_improvement:.6f} ({test_percent:.2f}%)\n")
    
    print(f"Decoder comparison completed. Results saved to: {decoder_dir}/")
    
    return {
        'final_mse': final_mse_values,
        'best_mse': best_mse_epoch_values
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run hyperspectral autoencoder noise experiment with filter2shape2filter architecture.")
    parser.add_argument("--cache", type=str, default="cache/aviris_tiles_swir.pt", help="Path to cache file for processed data")
    parser.add_argument("--use-cache", action="store_true", help="Use cached data if available")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (e.g., 0, 1, 2, 3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs for full training")
    parser.add_argument("--decoder-epochs", type=int, default=100, help="Number of epochs for decoder-only training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("-f", "--folders", type=str, default="all", 
                       help="Comma-separated list of folder name patterns to include, or 'all' for all folders")
    
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")
    
    # Modify cache filename based on folders
    folder_patterns = args.folders
    if folder_patterns.lower() == "all":
        folder_suffix = "all"
    else:
        folder_suffix = folder_patterns.replace(",", "_")
    
    # Update the cache path to include folder suffix
    cache_path = args.cache
    if ".pt" in cache_path:
        cache_path = cache_path.replace(".pt", f"_{folder_suffix}.pt")
    else:
        cache_path = f"{cache_path}_{folder_suffix}.pt"
    
    print(f"Using cache path: {cache_path}")
    
    # Check if the model paths exist
    shape2filter_path = "outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt"
    filter2shape_path = "outputs_three_stage_20250322_145925/stageC/spec2shape_stageC.pt"
    
    for model_path in [shape2filter_path, filter2shape_path]:
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            print("Searching for model file in current directory...")
            
            # Try to find the model file in the current directory structure
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.pt'):
                        if "shape2spec" in file and model_path == shape2filter_path:
                            shape2filter_path = os.path.join(root, file)
                            print(f"Using shape2filter model file: {shape2filter_path}")
                        elif "spec2shape" in file and model_path == filter2shape_path:
                            filter2shape_path = os.path.join(root, file)
                            print(f"Using filter2shape model file: {filter2shape_path}")
    
    # Create base output directory with folder patterns
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"noise_experiment_filter2shape2filter_{folder_suffix}_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create cache directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    
    # Set device for generating initial parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate a unified initial filter to use across all experiments
    initial_filter_params = generate_initial_filter(device)
    
    # Save the initial filter for reference
    np.save(f"{base_output_dir}/unified_initial_filter.npy", initial_filter_params.detach().cpu().numpy())
    
    # Load and split data first
    print("Loading and splitting data into train/test sets...")
    data = load_aviris_swir_data(swir_base_path="AVIRIS_SWIR_INTP", tile_size=100, 
                                 cache_file=cache_path, use_cache=args.use_cache, folder_patterns=folder_patterns)
    
    # Split data into training and testing sets (80% train, 20% test)
    num_samples = data.shape[0]
    indices = torch.randperm(num_samples)
    train_size = int(0.8 * num_samples)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_data = data[train_indices]
    test_data = data[test_indices]
    
    print(f"Data split into {train_data.shape[0]} training and {test_data.shape[0]} testing samples")
    
    # Define noise levels to test (in dB)
    noise_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    # noise_levels = [0, 20]  # Uncomment for quick testing
    
    # Dictionary to store paths to shapes for each noise level
    shape_paths = {}
    
    # Run full training for each noise level
    for noise_level in noise_levels:
        print(f"\n{'='*50}")
        print(f"Running full training with noise level: {noise_level} dB")
        print(f"{'='*50}\n")
        
        initial_shape_path, lowest_mse_shape_path, lowest_cn_shape_path, final_shape_path, output_dir = train_with_noise_level(
            shape2filter_path=shape2filter_path,
            filter2shape_path=filter2shape_path,
            output_dir=base_output_dir,
            noise_level=noise_level,
            initial_filter_params=initial_filter_params,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            cache_file=cache_path,
            use_cache=args.use_cache,
            folder_patterns=folder_patterns,
            train_data=train_data,
            test_data=test_data
        )
        
        # Store paths for later use
        shape_paths[noise_level] = {
            'initial': initial_shape_path,
            'lowest_mse': lowest_mse_shape_path,
            'lowest_cn': lowest_cn_shape_path,
            'final': final_shape_path,
            'output_dir': output_dir
        }
    
    # Create directory for decoder-only results
    decoder_output_dir = os.path.join(base_output_dir, "decoder_comparison_results")
    os.makedirs(decoder_output_dir, exist_ok=True)
    
    # Run decoder training with comparison for each noise level
    mse_results = {}
    for noise_level in noise_levels:
        print(f"\n{'='*50}")
        print(f"Running decoder comparison training for noise level: {noise_level} dB")
        print(f"{'='*50}\n")
        
        # Get paths to shapes
        paths = shape_paths[noise_level]
        
        # Train decoder with comparison of all shapes
        mse_results[noise_level] = train_decoder_with_comparison(
            shape2filter_path=shape2filter_path,
            filter2shape_path=filter2shape_path,
            shapes_dict=paths,
            output_dir=decoder_output_dir,
            noise_level=noise_level,
            num_epochs=args.decoder_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            cache_file=cache_path,
            use_cache=args.use_cache,
            folder_patterns=folder_patterns,
            train_data=train_data,
            test_data=test_data
        )
    
    # Collect results for all noise levels - both final and best MSE
    final_train_results = {shape_type: [] for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']}
    final_test_results = {shape_type: [] for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']}
    best_train_results = {shape_type: [] for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']}
    best_test_results = {shape_type: [] for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']}
    
    for noise_level in noise_levels:
        for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']:
            # Check if shape type exists in the results
            if shape_type in mse_results[noise_level]['final_mse']:
                final_train_results[shape_type].append(mse_results[noise_level]['final_mse'][shape_type]['train'])
                final_test_results[shape_type].append(mse_results[noise_level]['final_mse'][shape_type]['test'])
                best_train_results[shape_type].append(mse_results[noise_level]['best_mse'][shape_type]['train'])
                best_test_results[shape_type].append(mse_results[noise_level]['best_mse'][shape_type]['test'])
    
    # Create summary plots directory
    summary_dir = os.path.join(base_output_dir, "summary_plots")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Plot final MSE comparison across noise levels
    plt.figure(figsize=(12, 10))
    
    # Final Training MSE
    plt.subplot(2, 1, 1)
    for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']:
        if shape_type in final_train_results and len(final_train_results[shape_type]) == len(noise_levels):
            plt.plot(noise_levels, final_train_results[shape_type], 'o-', 
                    label=f"{shape_type.capitalize()} Shape")
    plt.grid(True)
    plt.xlabel("Noise Level (SNR in dB)")
    plt.ylabel("MSE after Training")
    plt.title("Final Training MSE Comparison Across Noise Levels")
    plt.legend()
    plt.gca().invert_xaxis()  # Higher SNR means less noise
    
    # Final Testing MSE
    plt.subplot(2, 1, 2)
    for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']:
        if shape_type in final_test_results and len(final_test_results[shape_type]) == len(noise_levels):
            plt.plot(noise_levels, final_test_results[shape_type], 'o-', 
                    label=f"{shape_type.capitalize()} Shape")
    plt.grid(True)
    plt.xlabel("Noise Level (SNR in dB)")
    plt.ylabel("MSE after Training")
    plt.title("Final Testing MSE Comparison Across Noise Levels")
    plt.legend()
    plt.gca().invert_xaxis()  # Higher SNR means less noise
    
    plt.tight_layout()
    final_comparison_path = os.path.join(summary_dir, "noise_level_final_mse_comparison.png")
    plt.savefig(final_comparison_path, dpi=300)
    plt.close()
    
    # Plot best MSE comparison across noise levels
    plt.figure(figsize=(12, 10))
    
    # Best Training MSE
    plt.subplot(2, 1, 1)
    for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']:
        if shape_type in best_train_results and len(best_train_results[shape_type]) == len(noise_levels):
            plt.plot(noise_levels, best_train_results[shape_type], 'o-', 
                    label=f"{shape_type.capitalize()} Shape")
    plt.grid(True)
    plt.xlabel("Noise Level (SNR in dB)")
    plt.ylabel("Best MSE During Training")
    plt.title("Best Training MSE Comparison Across Noise Levels")
    plt.legend()
    plt.gca().invert_xaxis()  # Higher SNR means less noise
    
    # Best Testing MSE
    plt.subplot(2, 1, 2)
    for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']:
        if shape_type in best_test_results and len(best_test_results[shape_type]) == len(noise_levels):
            plt.plot(noise_levels, best_test_results[shape_type], 'o-', 
                    label=f"{shape_type.capitalize()} Shape")
    plt.grid(True)
    plt.xlabel("Noise Level (SNR in dB)")
    plt.ylabel("Best MSE During Training")
    plt.title("Best Testing MSE Comparison Across Noise Levels")
    plt.legend()
    plt.gca().invert_xaxis()  # Higher SNR means less noise
    
    plt.tight_layout()
    best_comparison_path = os.path.join(summary_dir, "noise_level_best_mse_comparison.png")
    plt.savefig(best_comparison_path, dpi=300)
    plt.close()
    
    # Save the numerical results
    np.savez(os.path.join(summary_dir, "all_noise_shape_final_mse_results.npz"),
             noise_levels=noise_levels,
             final_train_results=final_train_results,
             final_test_results=final_test_results)
    
    np.savez(os.path.join(summary_dir, "all_noise_shape_best_mse_results.npz"),
             noise_levels=noise_levels,
             best_train_results=best_train_results,
             best_test_results=best_test_results)
    
    print(f"\nAll experiments completed! Results saved to: {os.path.abspath(base_output_dir)}")

if __name__ == "__main__":
    main()