import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import argparse
import numpy.linalg as LA
import shutil
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Function to create a drastic initial shape with higher variance
def create_drastic_initial_shape():
    """
    Create a drastic initial shape with larger variance.
    Returns a tuple of (raw_params, raw_shift) to be used for initialization.
    """
    # Use higher values for presence (first column) to make vertices more definitively present/absent
    # Use higher magnitude values for radius (second column) to create more variance
    raw_params = torch.tensor([
        [2.0, 1.5],    # First vertex always present (high value) with large radius
        [2.0, -1.2],  # Second vertex likely absent with different radius if present
        [2.0, 0.8],    # Third vertex likely present with medium-large radius
        [2.0, 2.0]    # Fourth vertex likely absent with very large radius if present
    ], dtype=torch.float32)
    
    # Use a non-zero shift for more asymmetry
    raw_shift = torch.tensor([1.5], dtype=torch.float32)
    
    return raw_params, raw_shift

# Function to calculate condition number of spectral filters
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

# Function to load AVIRIS_SWIR_INTP data from selected subfolders
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

# Function to replicate points with C4 symmetry
def replicate_c4(points):
    """Replicate points with C4 symmetry"""
    c4 = []
    for (x, y) in points:
        c4.append([x, y])       # Q1: original
        c4.append([-y, x])      # Q2: rotate 90°
        c4.append([-x, -y])     # Q3: rotate 180°
        c4.append([y, -x])      # Q4: rotate 270°
    return np.array(c4, dtype=np.float32)

# Function to sort points by angle for polygon drawing
def sort_points_by_angle(points):
    """Sort points by angle from center for polygon drawing"""
    if len(points) < 3:
        return points
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    idx = np.argsort(angles)
    return points[idx]

# Updated function to plot shape with C4 symmetry in a more academic style
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
    # plt.axis('off')  # Hide the axes for a cleaner look
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Define the ShapeToSpectraModel class (frozen)
class ShapeToSpectraModel(nn.Module):
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

        # Add this line to apply sigmoid activation:
        out_2d = torch.sigmoid(out_2d)
        
        return out_2d

# Shape parameterization functions
def straight_through_threshold(x, thresh=0.5):
    """
    Forward: threshold x at thresh (returns 1.0 where x>=thresh, 0 otherwise).
    Backward: gradients flow as if the operation were the identity.
    """
    y = (x >= thresh).float()
    return x + (y - x).detach()

def differentiable_legal_shape(raw_params, raw_shift):
    """
    Input:
      raw_params: a 4×2 tensor, each row: [raw_v_pres, raw_radius]
      raw_shift: a scalar tensor representing the raw angle shift.
    
    Output:
      A 4×3 matrix with column 0: binary vertex presence, column 1: x, and column 2: y.
    """
    device = raw_params.device
    dtype = raw_params.dtype

    # --- Vertex Presence ---
    # Apply sigmoid to raw_v_pres (column 0)
    v_pres_prob = torch.sigmoid(raw_params[:, 0])
    # Compute cumulative product: once a value is low, later ones get suppressed.
    v_pres_cum = torch.cumprod(v_pres_prob, dim=0)
    # Force the first vertex to be present:
    v_pres_cum = torch.cat([torch.ones(1, device=device, dtype=dtype), v_pres_cum[1:]], dim=0)
    # Apply straight-through threshold at 0.5:
    v_pres_bin = straight_through_threshold(v_pres_cum, thresh=0.5)

    # --- Count Valid Vertices ---
    n = v_pres_bin.sum()  # differentiable count (should be 4 if all are active)

    # --- Cumulative Indices for Valid Vertices ---
    idx = torch.cumsum(v_pres_bin, dim=0) - 1.0  # indices: first valid gets 0, second gets 1, etc.

    # --- Angle Assignment ---
    # Spacing s = π/(2*n) (avoid division by zero)
    s_spacing = math.pi / (2.0 * torch.clamp(n, min=1.0))
    base_angles = idx * s_spacing  # base angles for each vertex
    # Use the raw_shift parameter to compute delta (shift) in [0, s]
    delta = s_spacing * torch.sigmoid(raw_shift)
    # Final angles for active vertices
    angles = (base_angles + delta) * v_pres_bin

    # --- Radius Mapping ---
    # Map raw_radius (column 1) via sigmoid then linearly to [0.05, 0.65]
    radius = 0.05 + 0.6 * torch.sigmoid(raw_params[:, 1])
    radius = radius * v_pres_bin  # zero out inactive vertices

    # --- Cartesian Coordinates ---
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    coordinates = torch.stack([x, y], dim=1)

    # --- Final Output: 4×3 matrix ---
    # Column 0: binary vertex presence, Column 1 and 2: x and y coordinates.
    final_shape = torch.cat([v_pres_bin.unsqueeze(1), coordinates], dim=1)
    
    return final_shape

# Define the 5-layer CNN decoder for the second stage
class DecoderCNN5Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            # # Layer 1: 11 -> 24
            # nn.Conv2d(11, 24, kernel_size=3, padding=1),
            # nn.ReLU(),
            # # Layer 2: 24 -> 48
            # nn.Conv2d(24, 48, kernel_size=3, padding=1),
            # nn.ReLU(),
            # # Layer 3: 48 -> 72
            # nn.Conv2d(48, 72, kernel_size=3, padding=1),
            # nn.ReLU(),
            # # Layer 4: 72 -> 96
            # nn.Conv2d(72, 96, kernel_size=3, padding=1),
            # nn.ReLU(),
            # # Layer 5: 96 -> 100
            # nn.Conv2d(96, 100, kernel_size=3, padding=1)
            nn.Conv2d(11, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 100, kernel_size=3, padding=1)
            # nn.Conv2d(11, 32, kernel_size=1, padding="same"),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=1, padding="same"),
            # nn.ReLU(),
            # nn.Conv2d(64, 100, kernel_size=1, padding="same")
        )
    
    def forward(self, x):
        return self.decoder(x)

# Hyperspectral Autoencoder that uses shape2spectrum as filters
class HyperspectralAutoencoder(nn.Module):
    def __init__(self, shape2spec_model_path, target_snr=None, use_drastic_shape=False):
        super().__init__()

        # Target SNR (in dB) as model parameter
        self.target_snr = target_snr
        
        # Load the pretrained shape2spectrum model
        self.shape2spec = ShapeToSpectraModel()
        self.shape2spec.load_state_dict(torch.load(shape2spec_model_path, map_location='cpu'))
        
        # Freeze the shape2spectrum model
        for param in self.shape2spec.parameters():
            param.requires_grad = False
        
        # Shape parameters (learnable)
        if use_drastic_shape:
            # Use drastic initial shape with larger variance
            drastic_raw_params, drastic_raw_shift = create_drastic_initial_shape()
            self.raw_params = nn.Parameter(drastic_raw_params)
            self.raw_shift = nn.Parameter(drastic_raw_shift)
        else:
            # Initialize with values that will produce a reasonable initial shape
            self.raw_params = nn.Parameter(torch.tensor([
                [2.0, 0],  # First vertex active with medium radius
                [2.0, 0],  # 50/50 chance for second vertex
                [2.0, 0],  # 50/50 chance for third vertex
                [2.0, 0]   # 50/50 chance for fourth vertex
            ], dtype=torch.float32))
            self.raw_shift = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        
        # Decoder: 3-layer convolutional network for first stage
        self.decoder = nn.Sequential(
            # nn.Conv2d(11, 100, kernel_size=3, padding=1),
            nn.Conv2d(11, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 100, kernel_size=3, padding=1)
            # nn.Conv2d(11, 32, kernel_size=1, padding="same"),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=1, padding="same"),
            # nn.ReLU(),
            # nn.Conv2d(64, 100, kernel_size=1, padding="same")
            # Layer 1: 11 -> 24
            # nn.Conv2d(11, 24, kernel_size=3, padding=1),
            # nn.ReLU(),
            # # Layer 2: 24 -> 48
            # nn.Conv2d(24, 48, kernel_size=3, padding=1),
            # nn.ReLU(),
            # # Layer 3: 48 -> 72
            # nn.Conv2d(48, 72, kernel_size=3, padding=1),
            # nn.ReLU(),
            # # Layer 4: 72 -> 96
            # nn.Conv2d(72, 96, kernel_size=3, padding=1),
            # nn.ReLU(),
            # # Layer 5: 96 -> 100
            # nn.Conv2d(96, 100, kernel_size=3, padding=1)
        )
    
    def get_current_shape(self):
        """Return the current shape tensor based on learnable parameters"""
        return differentiable_legal_shape(self.raw_params, self.raw_shift)
    
    def get_current_filters(self):
        """Get the current spectral filters from the shape2spec model"""
        shape = self.get_current_shape().unsqueeze(0)  # Add batch dimension
        # When freezing decoder, we still need to allow gradients to flow back to shape parameters
        filters = self.shape2spec(shape)[0]  # Remove batch dimension: 11 x 100
        return filters

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
        
        # Get spectral filters from shape2spec
        filters = self.get_current_filters()  # Shape: [11, 100]
        
        # Convert input from [B,H,W,C] to [B,C,H,W] format for PyTorch convolution
        x_channels_first = x.permute(0, 3, 1, 2)
        
        # Normalize filters
        # filters_normalized = filters / 100.0  # Shape: [11, 100]
        filters_normalized = filters / 50.0  # Shape: [11, 100]
        
        # Use efficient tensor operations for spectral filtering
        # Einstein summation: 'bchw,oc->bohw'
        # This performs the weighted sum across spectral dimension for each output band
        encoded_channels_first = torch.einsum('bchw,oc->bohw', x_channels_first, filters_normalized)

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

# Function for full training with shape optimization
def train_with_noise_level(model_path, output_dir, noise_level, batch_size=10, num_epochs=500, 
                           learning_rate=0.001, cache_file=None, use_cache=False, folder_patterns="all",
                           use_drastic_shape=True, train_data=None, test_data=None):
    """
    Train and visualize the hyperspectral autoencoder with a specific noise level
    
    Parameters:
    model_path: Path to the pretrained shape2spec model
    output_dir: Directory to save outputs
    noise_level: SNR level in dB to apply during training
    batch_size: Batch size for training
    num_epochs: Number of training epochs
    learning_rate: Learning rate for optimizer
    cache_file: Path to cache file for storing processed data
    use_cache: If True, try to load from cache file first
    folder_patterns: Comma-separated list of folder name patterns to include, or 'all' for all folders
    use_drastic_shape: Whether to use the drastic initial shape
    train_data: Optional training data, if None will be loaded
    test_data: Optional testing data, if None will be loaded
    
    Returns:
    tuple: (initial_shape_path, final_shape_path, lowest_cn_shape_path, lowest_test_mse_shape_path, output_dir)
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
    
    # Initialize model with specified noise level and drastic shape
    model = HyperspectralAutoencoder(model_path, target_snr=noise_level, use_drastic_shape=use_drastic_shape).to(device)
    
    print(f"Model initialized with noise level: {noise_level} dB and {'drastic' if use_drastic_shape else 'standard'} initial shape")
    
    # Get initial shape and filters for visualization
    initial_shape = model.get_current_shape().detach().cpu().numpy()
    initial_filters = model.get_current_filters().detach().cpu()
    
    # Calculate initial condition number
    initial_condition_number = calculate_condition_number(initial_filters)
    print(f"Initial condition number: {initial_condition_number:.4f}")
    
    # Save initial shape with C4 symmetry
    initial_shape_path = f"{output_dir}/initial_shape.png"
    plot_shape_with_c4(initial_shape, f"Initial Shape", initial_shape_path)
    print(f"Initial shape saved to: {os.path.abspath(initial_shape_path)}")
    
    # Save initial filters
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(initial_filters.numpy()[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Initial Spectral Filters (SNR: {noise_level} dB)")
    plt.legend()
    initial_filters_path = f"{output_dir}/initial_filters.png"
    plt.savefig(initial_filters_path)
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
    lowest_cn_filters = None
    lowest_cn_epoch = -1
    
    # NEW: Add variables to track the lowest test MSE and corresponding shape
    lowest_test_mse = float('inf') if test_sample_tensor is not None else None
    lowest_test_mse_shape = None
    lowest_test_mse_filters = None
    lowest_test_mse_epoch = -1
    
    print(f"Starting training with SNR: {noise_level} dB...")
    
    # Create optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
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
        current_filters = model.get_current_filters().detach().cpu()
        current_condition_number = calculate_condition_number(current_filters)
        condition_numbers.append(current_condition_number)
        
        # Check if this is the lowest condition number so far
        if current_condition_number < lowest_condition_number:
            lowest_condition_number = current_condition_number
            lowest_cn_shape = model.get_current_shape().detach().cpu().numpy()
            lowest_cn_filters = current_filters.clone()
            lowest_cn_epoch = epoch
            print(f"New lowest condition number: {lowest_condition_number:.4f} at epoch {epoch+1}")
        
        # Calculate current MSE for train
        with torch.no_grad():
            current_train_recon, _ = model(sample_tensor)
            current_train_mse = ((current_train_recon - sample_tensor) ** 2).mean().item()
            train_mse_values.append(current_train_mse)
            
            # Calculate current MSE for test if available
            if test_sample_tensor is not None:
                current_test_recon, _ = model(test_sample_tensor)
                current_test_mse = ((current_test_recon - test_sample_tensor) ** 2).mean().item()
                test_mse_values.append(current_test_mse)
                
                # NEW: Check if this is the lowest test MSE so far
                if current_test_mse < lowest_test_mse:
                    lowest_test_mse = current_test_mse
                    lowest_test_mse_shape = model.get_current_shape().detach().cpu().numpy()
                    lowest_test_mse_filters = current_filters.clone()
                    lowest_test_mse_epoch = epoch
                    print(f"New lowest test MSE: {lowest_test_mse:.6f} at epoch {epoch+1}")
                
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Condition Number: {current_condition_number:.4f}, "
                      f"Train MSE: {current_train_mse:.6f}, Test MSE: {current_test_mse:.6f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Condition Number: {current_condition_number:.4f}, "
                      f"Train MSE: {current_train_mse:.6f}")
        
        # Save intermediate shapes and visualizations
        if (epoch+1) % 50 == 0 or epoch == 0:
            # Get current shape
            current_shape = model.get_current_shape().detach().cpu().numpy()
            
            # Save shape visualization to intermediate directory
            plot_shape_with_c4(
                current_shape, 
                f"Shape at Epoch {epoch+1}", 
                f"{viz_dir}/shape_epoch_{epoch+1}.png"
            )
    
    # Get final shape and filters
    final_shape = model.get_current_shape().detach().cpu().numpy()
    final_filters = model.get_current_filters().detach().cpu()
    
    # Calculate final condition number
    final_condition_number = calculate_condition_number(final_filters)
    print(f"Final condition number: {final_condition_number:.4f}")
    
    # Save final shape
    final_shape_path = f"{output_dir}/final_shape.png"
    plot_shape_with_c4(final_shape, f"Final Shape", final_shape_path)
    print(f"Final shape saved to: {os.path.abspath(final_shape_path)}")
    
    # Save final filters
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(final_filters.numpy()[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Final Spectral Filters (SNR: {noise_level} dB)")
    plt.legend()
    final_filters_path = f"{output_dir}/final_filters.png"
    plt.savefig(final_filters_path)
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
    
    # Save the lowest condition number filters
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(lowest_cn_filters.numpy()[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Spectral Filters with Lowest Condition Number: {lowest_condition_number:.4f}")
    plt.legend()
    lowest_cn_filters_path = f"{lowest_cn_dir}/filters.png"
    plt.savefig(lowest_cn_filters_path)
    plt.close()
    
    # Save the lowest condition number shape as numpy file
    lowest_cn_shape_npy_path = f"{lowest_cn_dir}/shape.npy"
    np.save(lowest_cn_shape_npy_path, lowest_cn_shape)
    
    # NEW: Create a directory for the lowest test MSE results if test data was available
    lowest_test_mse_shape_npy_path = None
    if test_sample_tensor is not None and lowest_test_mse_shape is not None:
        lowest_test_mse_dir = os.path.join(output_dir, "lowest_test_mse")
        os.makedirs(lowest_test_mse_dir, exist_ok=True)
        
        # Save the lowest test MSE shape
        lowest_test_mse_shape_path = f"{lowest_test_mse_dir}/shape.png"
        plot_shape_with_c4(lowest_test_mse_shape, f"Shape with Lowest Test MSE: {lowest_test_mse:.6f}", lowest_test_mse_shape_path)
        
        # Save the lowest test MSE filters
        plt.figure(figsize=(12, 8))
        for i in range(11):
            plt.plot(lowest_test_mse_filters.numpy()[i], label=f'Filter {i}' if i % 3 == 0 else None)
        plt.grid(True)
        plt.xlabel("Wavelength Index")
        plt.ylabel("Filter Value")
        plt.title(f"Spectral Filters with Lowest Test MSE: {lowest_test_mse:.6f}")
        plt.legend()
        lowest_test_mse_filters_path = f"{lowest_test_mse_dir}/filters.png"
        plt.savefig(lowest_test_mse_filters_path)
        plt.close()
        
        # Save the lowest test MSE shape as numpy file
        lowest_test_mse_shape_npy_path = f"{lowest_test_mse_dir}/shape.npy"
        np.save(lowest_test_mse_shape_npy_path, lowest_test_mse_shape)
    
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
    if test_mse_values:
        plt.plot(range(num_epochs+1), test_mse_values, 'r--', label='Test MSE')
        # NEW: Mark the lowest test MSE
        if lowest_test_mse_epoch >= 0:
            plt.axvline(x=lowest_test_mse_epoch+1, color='g', linestyle='--', 
                      label=f'Lowest Test MSE: {lowest_test_mse:.6f} (Epoch {lowest_test_mse_epoch+1})')
        plt.legend()
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"MSE of Reconstruction and Input (SNR: {noise_level} dB)")
    mse_plot_path = f"{output_dir}/mse_values.png"
    plt.savefig(mse_plot_path)
    plt.close()
    
    # Save model parameters
    model_save_path = f"{output_dir}/model_state.pt"
    torch.save({
        'raw_params': model.raw_params.detach().cpu(),
        'raw_shift': model.raw_shift.detach().cpu(),
        'decoder_state_dict': model.decoder.state_dict()
    }, model_save_path)
    
    # Also save numerical data
    np.save(f"{output_dir}/initial_shape.npy", initial_shape)
    np.save(f"{output_dir}/final_shape.npy", final_shape)
    np.save(f"{output_dir}/initial_filters.npy", initial_filters.numpy())
    np.save(f"{output_dir}/final_filters.npy", final_filters.numpy())
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
        f.write(f"Used drastic shape: {use_drastic_shape}\n")
        f.write("\n")
        
        # Save initial and final shape parameters
        f.write("Initial shape parameters:\n")
        f.write(f"raw_params:\n{model.raw_params.detach().cpu().numpy()}\n")
        f.write(f"raw_shift: {model.raw_shift.item()}\n\n")
        
        f.write("Final shape parameters after training:\n")
        f.write(f"raw_params:\n{model.raw_params.detach().cpu().numpy()}\n")
        f.write(f"raw_shift: {model.raw_shift.item()}\n\n")
        
        # Save condition number information
        f.write(f"Initial condition number: {initial_condition_number:.4f}\n")
        f.write(f"Final condition number: {final_condition_number:.4f}\n")
        f.write(f"Lowest condition number: {lowest_condition_number:.4f} at epoch {lowest_cn_epoch+1}\n")
        f.write(f"Condition number change: {final_condition_number - initial_condition_number:.4f}\n\n")
        
        # Save MSE information
        f.write(f"Initial Train MSE: {initial_train_mse:.6f}\n")
        f.write(f"Final Train MSE: {final_train_mse:.6f}\n")
        f.write(f"Train MSE improvement: {initial_train_mse - final_train_mse:.6f} ({(1 - final_train_mse/initial_train_mse) * 100:.2f}%)\n")
        
        if initial_test_mse is not None and final_test_mse is not None:
            f.write(f"Initial Test MSE: {initial_test_mse:.6f}\n")
            f.write(f"Final Test MSE: {final_test_mse:.6f}\n")
            f.write(f"Test MSE improvement: {initial_test_mse - final_test_mse:.6f} ({(1 - final_test_mse/initial_test_mse) * 100:.2f}%)\n")
            # NEW: Add information about lowest test MSE
            if lowest_test_mse_epoch >= 0:
                f.write(f"Lowest Test MSE: {lowest_test_mse:.6f} at epoch {lowest_test_mse_epoch+1}\n")
                f.write(f"Test MSE improvement (lowest vs initial): {initial_test_mse - lowest_test_mse:.6f} ({(1 - lowest_test_mse/initial_test_mse) * 100:.2f}%)\n")
    
    print(f"Training with SNR {noise_level} dB completed. All results saved to {output_dir}/")
    
    # Return paths to all shapes for later training
    initial_shape_npy_path = f"{output_dir}/initial_shape.npy"
    final_shape_npy_path = f"{output_dir}/final_shape.npy"
    lowest_cn_shape_npy_path = f"{lowest_cn_dir}/shape.npy"
    
    return initial_shape_npy_path, final_shape_npy_path, lowest_cn_shape_npy_path, lowest_test_mse_shape_npy_path, output_dir

# Function to train the decoder with train/test split and compare all shapes
def train_decoder_with_comparison(model_path, shapes_dict, output_dir, noise_level, num_epochs=100, 
                                 batch_size=10, learning_rate=0.001, cache_file=None, use_cache=False, 
                                 folder_patterns="all", train_data=None, test_data=None):
    """
    Train decoders for all shapes together (initial, lowest CN, final, lowest test MSE) with train/test split
    
    Parameters:
    model_path: Path to the pretrained shape2spec model
    shapes_dict: Dictionary with paths to 'initial', 'lowest_cn', 'final', and optionally 'lowest_test_mse' shapes
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
    dict: Dictionary containing both final and lowest MSE values for each shape
    """
    # Create output directory
    decoder_dir = os.path.join(output_dir, f"decoder_comparison_noise_{noise_level}dB")
    os.makedirs(decoder_dir, exist_ok=True)
    print(f"Saving decoder comparison results to {decoder_dir}/")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load shapes
    shapes = {}
    shape_types = ['initial', 'lowest_cn', 'final']
    
    # Add lowest_test_mse shape if available
    if 'lowest_test_mse' in shapes_dict and shapes_dict['lowest_test_mse'] is not None:
        shape_types.append('lowest_test_mse')
    
    for shape_type in shape_types:
        if shape_type in shapes_dict and shapes_dict[shape_type] is not None:
            shapes[shape_type] = np.load(shapes_dict[shape_type])
            print(f"Loaded {shape_type} shape from: {shapes_dict[shape_type]}")
    
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
    mse_values = {
        shape_type: {'train': [], 'test': []} for shape_type in shapes
    }
    
    # NEW: Dictionary to store lowest MSE values for each shape type
    lowest_mse_values = {
        shape_type: {'train': {'value': float('inf'), 'epoch': -1}, 
                     'test': {'value': float('inf'), 'epoch': -1}}
        for shape_type in shapes
    }
    
    # Dictionary to store models and optimizers
    models = {}
    optimizers = {}
    
    # Initialize models for each shape type
    for shape_type in shapes:
        # Initialize model
        models[shape_type] = HyperspectralAutoencoder(model_path, target_snr=noise_level).to(device)
        
        # Set shape to the loaded fixed shape
        with torch.no_grad():
            saved_shape_tensor = torch.tensor(shapes[shape_type], dtype=torch.float32, device=device)
            
            # Replace the get_current_shape method
            original_get_shape = models[shape_type].get_current_shape
            models[shape_type].get_current_shape = lambda saved_tensor=saved_shape_tensor: saved_tensor
            
            # Freeze shape parameters
            models[shape_type].raw_params.requires_grad = False
            models[shape_type].raw_shift.requires_grad = False
        
        # Create a new 5-layer decoder
        new_decoder = DecoderCNN5Layer().to(device)
        
        # Replace the original decoder with the new decoder
        models[shape_type].decoder = new_decoder.decoder
        
        # Create optimizer for the decoder parameters only
        optimizers[shape_type] = optim.Adam(models[shape_type].decoder.parameters(), lr=learning_rate)
        
        # Get initial MSE with the new decoder (before training)
        with torch.no_grad():
            initial_train_recon, _ = models[shape_type](train_sample)
            initial_train_mse = ((initial_train_recon - train_sample) ** 2).mean().item()
            mse_values[shape_type]['train'].append(initial_train_mse)
            
            # Update lowest train MSE
            if initial_train_mse < lowest_mse_values[shape_type]['train']['value']:
                lowest_mse_values[shape_type]['train']['value'] = initial_train_mse
                lowest_mse_values[shape_type]['train']['epoch'] = 0
            
            initial_test_recon, _ = models[shape_type](test_sample)
            initial_test_mse = ((initial_test_recon - test_sample) ** 2).mean().item()
            mse_values[shape_type]['test'].append(initial_test_mse)
            
            # Update lowest test MSE
            if initial_test_mse < lowest_mse_values[shape_type]['test']['value']:
                lowest_mse_values[shape_type]['test']['value'] = initial_test_mse
                lowest_mse_values[shape_type]['test']['epoch'] = 0
            
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
                
                # Update lowest train MSE
                if train_mse < lowest_mse_values[shape_type]['train']['value']:
                    lowest_mse_values[shape_type]['train']['value'] = train_mse
                    lowest_mse_values[shape_type]['train']['epoch'] = epoch + 1
                
                # Test sample
                test_recon, _ = models[shape_type](test_sample)
                test_mse = ((test_recon - test_sample) ** 2).mean().item()
                mse_values[shape_type]['test'].append(test_mse)
                
                # Update lowest test MSE
                if test_mse < lowest_mse_values[shape_type]['test']['value']:
                    lowest_mse_values[shape_type]['test']['value'] = test_mse
                    lowest_mse_values[shape_type]['test']['epoch'] = epoch + 1
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}:")
            for shape_type in shapes:
                print(f"  {shape_type.capitalize()} shape - Train MSE: {mse_values[shape_type]['train'][-1]:.6f}, "
                      f"Test MSE: {mse_values[shape_type]['test'][-1]:.6f}")
    
    # Plot combined MSE curves for all shape types
    plt.figure(figsize=(12, 8))
    
    # Training MSE
    plt.subplot(1, 2, 1)
    for shape_type in shapes:
        plt.plot(range(num_epochs+1), mse_values[shape_type]['train'], 
                 label=f"{shape_type.capitalize()} Shape")
        # Mark the lowest MSE point
        min_epoch = lowest_mse_values[shape_type]['train']['epoch']
        min_value = lowest_mse_values[shape_type]['train']['value']
        plt.scatter([min_epoch], [min_value], marker='o', s=100, 
                    edgecolors='black', label=f"{shape_type.capitalize()} Lowest")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"Training MSE (SNR: {noise_level} dB)")
    plt.legend()
    
    # Testing MSE
    plt.subplot(1, 2, 2)
    for shape_type in shapes:
        plt.plot(range(num_epochs+1), mse_values[shape_type]['test'], 
                 label=f"{shape_type.capitalize()} Shape")
        # Mark the lowest MSE point
        min_epoch = lowest_mse_values[shape_type]['test']['epoch']
        min_value = lowest_mse_values[shape_type]['test']['value']
        plt.scatter([min_epoch], [min_value], marker='o', s=100, 
                    edgecolors='black', label=f"{shape_type.capitalize()} Lowest")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"Testing MSE (SNR: {noise_level} dB)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{decoder_dir}/shape_comparison_mse.png", dpi=300)
    plt.close()
    
    # Save the MSE values
    for shape_type in shapes:
        np.save(f"{decoder_dir}/{shape_type}_train_mse.npy", np.array(mse_values[shape_type]['train']))
        np.save(f"{decoder_dir}/{shape_type}_test_mse.npy", np.array(mse_values[shape_type]['test']))
    
    # Calculate final MSE values
    final_mse_values = {}
    for shape_type in shapes:
        final_mse_values[shape_type] = {
            'train': mse_values[shape_type]['train'][-1],
            'test': mse_values[shape_type]['test'][-1]
        }
        
        # Save model
        torch.save(models[shape_type].decoder.state_dict(), f"{decoder_dir}/{shape_type}_decoder_state.pt")
    
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
        
        # NEW: Add lowest MSE values
        f.write("\nLowest MSE values:\n")
        for shape_type in shapes:
            train_info = lowest_mse_values[shape_type]['train']
            test_info = lowest_mse_values[shape_type]['test']
            f.write(f"  {shape_type.capitalize()} shape - Train: {train_info['value']:.6f} (Epoch {train_info['epoch']}), "
                    f"Test: {test_info['value']:.6f} (Epoch {test_info['epoch']})\n")
        
        f.write("\nMSE improvements (from initial to final):\n")
        for shape_type in shapes:
            train_improvement = mse_values[shape_type]['train'][0] - mse_values[shape_type]['train'][-1]
            train_percent = (1 - mse_values[shape_type]['train'][-1] / mse_values[shape_type]['train'][0]) * 100
            
            test_improvement = mse_values[shape_type]['test'][0] - mse_values[shape_type]['test'][-1]
            test_percent = (1 - mse_values[shape_type]['test'][-1] / mse_values[shape_type]['test'][0]) * 100
            
            f.write(f"  {shape_type.capitalize()} shape - Train: {train_improvement:.6f} ({train_percent:.2f}%), "
                    f"Test: {test_improvement:.6f} ({test_percent:.2f}%)\n")
        
        # NEW: Add improvements from initial to lowest MSE
        f.write("\nMSE improvements (from initial to lowest):\n")
        for shape_type in shapes:
            train_improvement = mse_values[shape_type]['train'][0] - lowest_mse_values[shape_type]['train']['value']
            train_percent = (1 - lowest_mse_values[shape_type]['train']['value'] / mse_values[shape_type]['train'][0]) * 100
            
            test_improvement = mse_values[shape_type]['test'][0] - lowest_mse_values[shape_type]['test']['value']
            test_percent = (1 - lowest_mse_values[shape_type]['test']['value'] / mse_values[shape_type]['test'][0]) * 100
            
            f.write(f"  {shape_type.capitalize()} shape - Train: {train_improvement:.6f} ({train_percent:.2f}%), "
                    f"Test: {test_improvement:.6f} ({test_percent:.2f}%)\n")
    
    print(f"Decoder comparison completed. Results saved to: {decoder_dir}/")
    
    # Return both final MSE values and lowest MSE values
    return {
        'final': final_mse_values,
        'lowest': {shape_type: {'train': info['train']['value'], 'test': info['test']['value']} 
                  for shape_type, info in lowest_mse_values.items()}
    }

# Function to plot a comparison of the drastic initial shape vs. conventional shape
def plot_drastic_shape_comparison(output_dir):
    """
    Generate and plot a comparison between the drastic initial shape and conventional shape
    
    Parameters:
    output_dir: Directory to save the comparison plot
    """
    # Create drastic shape
    drastic_raw_params, drastic_raw_shift = create_drastic_initial_shape()
    
    # Create conventional shape
    conventional_raw_params = torch.tensor([
        [2.0, 0],  # First vertex active with medium radius
        [2.0, 0],  # 50/50 chance for second vertex
        [2.0, 0],  # 50/50 chance for third vertex
        [2.0, 0]   # 50/50 chance for fourth vertex
    ], dtype=torch.float32)
    conventional_raw_shift = torch.tensor([0.0], dtype=torch.float32)
    
    # Generate the shapes
    drastic_shape = differentiable_legal_shape(drastic_raw_params, drastic_raw_shift).numpy()
    conventional_shape = differentiable_legal_shape(conventional_raw_params, conventional_raw_shift).numpy()
    
    # Create figure for comparison
    plt.figure(figsize=(10, 5))
    
    # Plot conventional shape
    plt.subplot(1, 2, 1)
    plt.xlim(-0.7, 0.7)
    plt.ylim(-0.7, 0.7)
    
    # Extract active points for conventional
    presence = conventional_shape[:, 0] > 0.5
    active_points = conventional_shape[presence, 1:3]
    
    # Plot original Q1 points
    plt.scatter(conventional_shape[presence, 1], conventional_shape[presence, 2], color='red', s=50)
    
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
    
    plt.title("Conventional Initial Shape", fontsize=12)
    plt.axis('equal')
    
    # Plot drastic shape
    plt.subplot(1, 2, 2)
    plt.xlim(-0.7, 0.7)
    plt.ylim(-0.7, 0.7)
    
    # Extract active points for drastic
    presence = drastic_shape[:, 0] > 0.5
    active_points = drastic_shape[presence, 1:3]
    
    # Plot original Q1 points
    plt.scatter(drastic_shape[presence, 1], drastic_shape[presence, 2], color='red', s=50)
    
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
    
    plt.title("Drastic Initial Shape", fontsize=12)
    plt.axis('equal')
    
    plt.tight_layout()
    shape_comparison_path = os.path.join(output_dir, "drastic_vs_conventional_shape.png")
    plt.savefig(shape_comparison_path, dpi=300)
    plt.close()
    
    print(f"Shape comparison saved to: {shape_comparison_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run hyperspectral autoencoder noise experiment on SWIR data.")
    parser.add_argument("--cache", type=str, default="cache/aviris_tiles_swir.pt", help="Path to cache file for processed data")
    parser.add_argument("--use-cache", action="store_true", help="Use cached data if available")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (e.g., 0, 1, 2, 3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs for full training")
    parser.add_argument("--decoder-epochs", type=int, default=100, help="Number of epochs for decoder-only training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("-f", "--folders", type=str, default="all", 
                       help="Comma-separated list of folder name patterns to include, or 'all' for all folders")
    parser.add_argument("--use-drastic-shape", action="store_true", default=True, 
                       help="Use drastic initial shape with higher variance")
    
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
    
    # Check if the model path exists
    model_path = "outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt"
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("Searching for model file in current directory...")
        
        # Try to find the model file in the current directory structure
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.pt'):
                    print(f"Found model file: {os.path.join(root, file)}")
                    if "shape2spec" in file:
                        model_path = os.path.join(root, file)
                        print(f"Using model file: {model_path}")
                        break
    
    # Create base output directory with folder patterns
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"noise_experiment_swir_{folder_suffix}_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create cache directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    
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
    
    # Generate and save a comparison of the drastic initial shape vs conventional shape
    print("Generating shape comparison...")
    plot_drastic_shape_comparison(base_output_dir)
    
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
        
        initial_shape_path, final_shape_path, lowest_cn_shape_path, lowest_test_mse_shape_path, output_dir = train_with_noise_level(
            model_path=model_path,
            output_dir=base_output_dir,
            noise_level=noise_level,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            cache_file=cache_path,
            use_cache=args.use_cache,
            folder_patterns=folder_patterns,
            use_drastic_shape=args.use_drastic_shape,
            train_data=train_data,
            test_data=test_data
        )
        
        # Store paths for later use
        shape_paths[noise_level] = {
            'initial': initial_shape_path,
            'final': final_shape_path,
            'lowest_cn': lowest_cn_shape_path,
            'lowest_test_mse': lowest_test_mse_shape_path,
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
            model_path=model_path,
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
    
    # Collect results for all noise levels - both final and lowest MSE
    # Final MSE
    final_train_results = {shape_type: [] for shape_type in ['initial', 'lowest_cn', 'final', 'lowest_test_mse']}
    final_test_results = {shape_type: [] for shape_type in ['initial', 'lowest_cn', 'final', 'lowest_test_mse']}
    
    # Lowest MSE
    lowest_train_results = {shape_type: [] for shape_type in ['initial', 'lowest_cn', 'final', 'lowest_test_mse']}
    lowest_test_results = {shape_type: [] for shape_type in ['initial', 'lowest_cn', 'final', 'lowest_test_mse']}
    
    for noise_level in noise_levels:
        for shape_type in ['initial', 'lowest_cn', 'final', 'lowest_test_mse']:
            # Skip if shape_type doesn't exist for this noise level
            if shape_type not in mse_results[noise_level]['final']:
                continue
                
            # Final MSE
            final_train_results[shape_type].append(mse_results[noise_level]['final'][shape_type]['train'])
            final_test_results[shape_type].append(mse_results[noise_level]['final'][shape_type]['test'])
            
            # Lowest MSE
            lowest_train_results[shape_type].append(mse_results[noise_level]['lowest'][shape_type]['train'])
            lowest_test_results[shape_type].append(mse_results[noise_level]['lowest'][shape_type]['test'])
    
    # Plot noise level comparison for all shapes - FINAL MSE
    plt.figure(figsize=(16, 12))
    
    # Final Training MSE
    plt.subplot(2, 2, 1)
    for shape_type in ['initial', 'lowest_cn', 'final', 'lowest_test_mse']:
        # Only plot if we have data for this shape type
        if final_train_results[shape_type]:
            plt.plot(noise_levels[:len(final_train_results[shape_type])], final_train_results[shape_type], 'o-', 
                    label=f"{shape_type.capitalize()} Shape")
    plt.grid(True)
    plt.xlabel("Noise Level (SNR in dB)")
    plt.ylabel("Final MSE after Training")
    plt.title("Final Training MSE Comparison Across Noise Levels")
    plt.legend()
    plt.gca().invert_xaxis()  # Higher SNR means less noise
    
    # Final Testing MSE
    plt.subplot(2, 2, 2)
    for shape_type in ['initial', 'lowest_cn', 'final', 'lowest_test_mse']:
        # Only plot if we have data for this shape type
        if final_test_results[shape_type]:
            plt.plot(noise_levels[:len(final_test_results[shape_type])], final_test_results[shape_type], 'o-', 
                    label=f"{shape_type.capitalize()} Shape")
    plt.grid(True)
    plt.xlabel("Noise Level (SNR in dB)")
    plt.ylabel("Final MSE after Training")
    plt.title("Final Testing MSE Comparison Across Noise Levels")
    plt.legend()
    plt.gca().invert_xaxis()  # Higher SNR means less noise
    
    # Lowest Training MSE
    plt.subplot(2, 2, 3)
    for shape_type in ['initial', 'lowest_cn', 'final', 'lowest_test_mse']:
        # Only plot if we have data for this shape type
        if lowest_train_results[shape_type]:
            plt.plot(noise_levels[:len(lowest_train_results[shape_type])], lowest_train_results[shape_type], 'o-', 
                    label=f"{shape_type.capitalize()} Shape")
    plt.grid(True)
    plt.xlabel("Noise Level (SNR in dB)")
    plt.ylabel("Lowest MSE during Training")
    plt.title("Lowest Training MSE Comparison Across Noise Levels")
    plt.legend()
    plt.gca().invert_xaxis()  # Higher SNR means less noise
    
    # Lowest Testing MSE
    plt.subplot(2, 2, 4)
    for shape_type in ['initial', 'lowest_cn', 'final', 'lowest_test_mse']:
        # Only plot if we have data for this shape type
        if lowest_test_results[shape_type]:
            plt.plot(noise_levels[:len(lowest_test_results[shape_type])], lowest_test_results[shape_type], 'o-', 
                    label=f"{shape_type.capitalize()} Shape")
    plt.grid(True)
    plt.xlabel("Noise Level (SNR in dB)")
    plt.ylabel("Lowest MSE during Training")
    plt.title("Lowest Testing MSE Comparison Across Noise Levels")
    plt.legend()
    plt.gca().invert_xaxis()  # Higher SNR means less noise
    
    plt.tight_layout()
    comparison_path = os.path.join(base_output_dir, "noise_level_shape_mse_comparison.png")
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    
    # Save the numerical results
    np.savez(os.path.join(base_output_dir, "all_noise_shape_comparison_results.npz"),
             noise_levels=noise_levels,
             final_train_results=final_train_results,
             final_test_results=final_test_results,
             lowest_train_results=lowest_train_results,
             lowest_test_results=lowest_test_results)
    
    print(f"\nAll experiments completed! Results saved to: {os.path.abspath(base_output_dir)}")

if __name__ == "__main__":
    main()