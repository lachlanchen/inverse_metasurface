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

# Function to load AVIRIS_SWIR data from all available subfolders
def load_aviris_swir_data(swir_base_path="AVIRIS_SWIR", tile_size=100, cache_file=None, use_cache=False):
    """
    Load pre-processed AVIRIS_SWIR hyperspectral data from all available subfolders and crop it into tiles
    
    Parameters:
    swir_base_path: Base path to the AVIRIS_SWIR directory
    tile_size: Size of the tiles (square)
    cache_file: Path to the cache file to save/load the processed data
    use_cache: If True, try to load from cache file first
    
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
        raise FileNotFoundError(f"AVIRIS_SWIR directory not found at: {os.path.abspath(swir_base_path)}")
    
    print(f"Looking for AVIRIS_SWIR data in: {os.path.abspath(swir_base_path)}")
    
    # Get list of subfolders
    subfolders = [f for f in os.listdir(swir_base_path) 
                 if os.path.isdir(os.path.join(swir_base_path, f))]
    
    if not subfolders:
        raise FileNotFoundError(f"No subfolders found in AVIRIS_SWIR directory: {os.path.abspath(swir_base_path)}")
    
    print(f"Found {len(subfolders)} subfolders: {', '.join(subfolders)}")
    
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

# Function to plot shape with C4 symmetry
def plot_shape_with_c4(shape, title, save_path):
    """Plot shape with C4 symmetry replication"""
    plt.figure(figsize=(8, 8))
    plt.xlim(-0.7, 0.7)
    plt.ylim(-0.7, 0.7)
    plt.grid(True)
    
    # Extract active points
    presence = shape[:, 0] > 0.5
    active_points = shape[presence, 1:3]
    
    # Plot original Q1 points
    plt.scatter(shape[presence, 1], shape[presence, 2], color='red', s=100, label='Q1 Control Points')
    
    # Apply C4 symmetry and plot the polygon
    if len(active_points) > 0:
        c4_points = replicate_c4(active_points)
        sorted_points = sort_points_by_angle(c4_points)
        
        # If we have enough points for a polygon
        if len(sorted_points) >= 3:
            # Close the polygon
            polygon = np.vstack([sorted_points, sorted_points[0]])
            plt.plot(polygon[:, 0], polygon[:, 1], 'g-', linewidth=2)
            plt.fill(polygon[:, 0], polygon[:, 1], 'g', alpha=0.4)
        else:
            # Just plot the points
            plt.scatter(c4_points[:, 0], c4_points[:, 1], color='green', alpha=0.4)
    
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    plt.savefig(save_path, dpi=300)
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

def differentiable_legal_shape_flexible(raw_params, raw_shift):
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

def differentiable_legal_shape(raw_params, raw_shift):
    """
    Input:
      raw_params: a 4×2 tensor, each row: [raw_v_pres (unused now), raw_radius]
      raw_shift: a scalar tensor representing the raw angle shift.
    
    Output:
      A 4×3 matrix with column 0: binary vertex presence (always 1), column 1: x, and column 2: y.
    """
    device = raw_params.device
    dtype = raw_params.dtype
    
    # --- Vertex Presence ---
    # All vertices are always present (always 1)
    v_pres_bin = torch.ones(4, device=device, dtype=dtype)
    
    # --- Fixed indices for 4 vertices ---
    idx = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device, dtype=dtype)
    
    # --- Angle Assignment ---
    # Fixed spacing for 4 points in first quadrant (π/2 divided by 4)
    s_spacing = math.pi / 8.0
    base_angles = idx * s_spacing  # base angles for each vertex
    
    # Use the raw_shift parameter to compute delta (shift) in [0, s]
    delta = s_spacing * torch.sigmoid(raw_shift)
    
    # Final angles for all vertices
    angles = base_angles + delta
    
    # --- Radius Mapping ---
    # Map raw_radius (column 1) via sigmoid then linearly to [0.05, 0.65]
    radius = 0.05 + 0.6 * torch.sigmoid(raw_params[:, 1])
    
    # --- Cartesian Coordinates ---
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    coordinates = torch.stack([x, y], dim=1)
    
    # --- Final Output: 4×3 matrix ---
    # Column 0: binary vertex presence (all 1), Column 1 and 2: x and y coordinates.
    final_shape = torch.cat([v_pres_bin.unsqueeze(1), coordinates], dim=1)
    
    return final_shape

# Hyperspectral Autoencoder that uses shape2spectrum as filters
class HyperspectralAutoencoder(nn.Module):
    def __init__(self, shape2spec_model_path, target_snr=None):
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
        # Initialize with values that will produce a reasonable initial shape
        self.raw_params = nn.Parameter(torch.tensor([
            [2.0, 0.0],  # First vertex active with medium radius
            [2.0, 0.0],  # 50/50 chance for second vertex
            [2.0, 0.0],  # 50/50 chance for third vertex
            [2.0, 0.0]   # 50/50 chance for fourth vertex
        ], dtype=torch.float32))
        
        self.raw_shift = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        
        # Decoder: Simple convolutional network to reconstruct 100 bands from 11
        self.decoder = nn.Sequential(
            # nn.Conv2d(11, 100, kernel_size=3, padding=1),
            # nn.Conv2d(11, 100, kernel_size=1, padding="same"),
            nn.Conv2d(11, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 100, kernel_size=3, padding=1)
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
        filters_normalized = filters / 100.0  # Shape: [11, 100]
        
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
                           learning_rate=0.001, cache_file=None, use_cache=False):
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
    
    Returns:
    tuple: (initial_shape_path, final_shape_path, output_dir)
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
    
    # Load AVIRIS_SWIR data with caching support
    data = load_aviris_swir_data(swir_base_path="AVIRIS_SWIR", tile_size=100, cache_file=cache_file, use_cache=use_cache)
    data = data.to(device)
    
    print("Data shape:", data.shape)
    
    # Create dataset and dataloader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size//2, shuffle=True)
    
    # Initialize model with specified noise level
    model = HyperspectralAutoencoder(model_path, target_snr=noise_level).to(device)
    
    print(f"Model initialized with noise level: {noise_level} dB")
    
    # Get initial shape and filters for visualization
    initial_shape = model.get_current_shape().detach().cpu().numpy()
    initial_filters = model.get_current_filters().detach().cpu()
    
    # Calculate initial condition number
    initial_condition_number = calculate_condition_number(initial_filters)
    print(f"Initial condition number: {initial_condition_number:.4f}")
    
    # Save initial shape with C4 symmetry
    initial_shape_path = f"{output_dir}/initial_shape.png"
    plot_shape_with_c4(initial_shape, f"Initial Shape (SNR: {noise_level} dB)", initial_shape_path)
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
    
    # Before training, get reconstruction of sample
    sample_idx = min(5, len(data) - 1)  # Make sure index is within range
    sample_tensor = data[sample_idx:sample_idx+1]  # Keep as tensor with batch dimension
    
    with torch.no_grad():
        initial_recon, encoded = model(sample_tensor)
        # Convert to numpy for visualization
        initial_recon_np = initial_recon.detach().cpu().numpy()[0]
        encoded_np = encoded.detach().cpu().numpy()[0]
        sample_np = sample_tensor.detach().cpu().numpy()[0]
        
        # Calculate initial MSE
        initial_mse = ((initial_recon - sample_tensor) ** 2).mean().item()
        print(f"Initial MSE: {initial_mse:.6f}")
    
    # Training loop
    losses = []
    condition_numbers = [initial_condition_number]
    mse_values = [initial_mse]
    
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
        
        # Calculate current MSE
        with torch.no_grad():
            current_recon, _ = model(sample_tensor)
            current_mse = ((current_recon - sample_tensor) ** 2).mean().item()
            mse_values.append(current_mse)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Condition Number: {current_condition_number:.4f}, MSE: {current_mse:.6f}")
        
        # Save intermediate shapes and visualizations
        if (epoch+1) % 50 == 0 or epoch == 0:
            # Get current shape
            current_shape = model.get_current_shape().detach().cpu().numpy()
            
            # Save shape visualization to intermediate directory
            plot_shape_with_c4(
                current_shape, 
                f"Shape at Epoch {epoch+1} (SNR: {noise_level} dB)", 
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
    plot_shape_with_c4(final_shape, f"Final Shape (SNR: {noise_level} dB)", final_shape_path)
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
        final_recon, encoded = model(sample_tensor)
        final_recon_np = final_recon.detach().cpu().numpy()[0]
        encoded_np = encoded.detach().cpu().numpy()[0]
        
        # Calculate final MSE
        final_mse = ((final_recon - sample_tensor) ** 2).mean().item()
        print(f"Final MSE: {final_mse:.6f}")
    
    # Plot condition number during training
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs+1), condition_numbers, 'r-')
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Condition Number")
    plt.title(f"Filter Matrix Condition Number During Training (SNR: {noise_level} dB)")
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
    plt.plot(range(num_epochs+1), mse_values)
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
    np.save(f"{output_dir}/mse_values.npy", np.array(mse_values))
    
    # Create log file to save training parameters
    with open(f"{output_dir}/training_params.txt", "w") as f:
        f.write(f"Noise level (SNR): {noise_level} dB\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Cache file: {cache_file}\n")
        f.write(f"Used cache: {use_cache}\n")
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
        f.write(f"Condition number change: {final_condition_number - initial_condition_number:.4f}\n\n")
        
        # Save MSE information
        f.write(f"Initial MSE: {initial_mse:.6f}\n")
        f.write(f"Final MSE: {final_mse:.6f}\n")
        f.write(f"MSE improvement: {initial_mse - final_mse:.6f} ({(1 - final_mse/initial_mse) * 100:.2f}%)\n")
    
    print(f"Training with SNR {noise_level} dB completed. All results saved to {output_dir}/")
    
    # Return paths to the initial and final shapes for later training
    initial_shape_npy_path = f"{output_dir}/initial_shape.npy"
    final_shape_npy_path = f"{output_dir}/final_shape.npy"
    return initial_shape_npy_path, final_shape_npy_path, output_dir

# Function to train the decoder only with a fixed shape
def train_decoder_only(model_path, shape_path, output_dir, noise_level, num_epochs=100, 
                      batch_size=10, learning_rate=0.001, cache_file=None, use_cache=False):
    """
    Train only the decoder with a fixed shape
    
    Parameters:
    model_path: Path to the pretrained shape2spec model
    shape_path: Path to the saved shape numpy file
    output_dir: Directory to save outputs
    noise_level: SNR level in dB
    num_epochs: Number of training epochs
    batch_size: Batch size for training
    learning_rate: Learning rate for optimizer
    cache_file: Path to cache file for processed data
    use_cache: If True, try to load from cache file first
    
    Returns:
    float: Final MSE after training
    """
    # Create output directory
    shape_type = "initial" if "initial" in shape_path else "final"
    decoder_dir = os.path.join(output_dir, f"decoder_only_{shape_type}_noise_{noise_level}dB")
    os.makedirs(decoder_dir, exist_ok=True)
    print(f"Saving decoder-only results to {decoder_dir}/")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved shape
    saved_shape = np.load(shape_path)
    print(f"Loaded {shape_type} shape from: {shape_path}")
    
    # Load AVIRIS_SWIR data
    data = load_aviris_swir_data(swir_base_path="AVIRIS_SWIR", tile_size=100, cache_file=cache_file, use_cache=use_cache)
    data = data.to(device)
    
    # Create dataset and dataloader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size//2, shuffle=True)
    
    # Initialize model
    model = HyperspectralAutoencoder(model_path, target_snr=noise_level).to(device)
    
    # Set shape to the loaded fixed shape
    with torch.no_grad():
        # Create a template shape to get the right device
        template_shape = model.get_current_shape()
        
        # Reset raw params to initialize a new shape
        saved_shape_tensor = torch.tensor(saved_shape, dtype=torch.float32, device=device)
        
        # Set up a custom shape model
        class CustomShapeModel(nn.Module):
            def __init__(self, fixed_shape):
                super().__init__()
                self.fixed_shape = fixed_shape
                
            def forward(self, x):
                batch_size = x.size(0)
                # Repeat the fixed shape for each item in the batch
                expanded_shape = self.fixed_shape.unsqueeze(0).expand(batch_size, -1, -1)
                return expanded_shape
        
        # Replace the get_current_shape method
        original_get_shape = model.get_current_shape
        model.get_current_shape = lambda: saved_shape_tensor
        
        # Freeze shape parameters
        model.raw_params.requires_grad = False
        model.raw_shift.requires_grad = False
    
    # Ensure only decoder parameters are trainable
    trainable_params = []
    for name, param in model.named_parameters():
        if 'decoder' in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    
    print(f"Training only decoder with fixed {shape_type} shape, SNR: {noise_level} dB")
    print(f"Number of trainable parameters: {len(trainable_params)}")
    
    # Sample for evaluation
    sample_idx = min(5, len(data) - 1)
    sample_tensor = data[sample_idx:sample_idx+1]
    
    # Get initial MSE
    with torch.no_grad():
        initial_recon, _ = model(sample_tensor)
        initial_mse = ((initial_recon - sample_tensor) ** 2).mean().item()
        print(f"Initial MSE with fixed {shape_type} shape: {initial_mse:.6f}")
    
    # Create optimizer
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    mse_values = [initial_mse]
    
    print(f"Starting decoder-only training for {shape_type} shape, SNR: {noise_level} dB...")
    
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
        
        # Calculate current MSE
        with torch.no_grad():
            current_recon, _ = model(sample_tensor)
            current_mse = ((current_recon - sample_tensor) ** 2).mean().item()
            mse_values.append(current_mse)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, MSE: {current_mse:.6f}")
    
    # Calculate final MSE
    with torch.no_grad():
        final_recon, _ = model(sample_tensor)
        final_mse = ((final_recon - sample_tensor) ** 2).mean().item()
        print(f"Final MSE after decoder-only training: {final_mse:.6f}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Decoder-only Training Loss ({shape_type} shape, SNR: {noise_level} dB)")
    plt.savefig(f"{decoder_dir}/training_loss.png")
    plt.close()
    
    # Plot MSE values
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs+1), mse_values)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"MSE of Reconstruction ({shape_type} shape, SNR: {noise_level} dB)")
    plt.savefig(f"{decoder_dir}/mse_values.png")
    plt.close()
    
    # Save numerical data
    np.save(f"{decoder_dir}/losses.npy", np.array(losses))
    np.save(f"{decoder_dir}/mse_values.npy", np.array(mse_values))
    
    # Save log file
    with open(f"{decoder_dir}/training_params.txt", "w") as f:
        f.write(f"Shape type: {shape_type}\n")
        f.write(f"Shape path: {shape_path}\n")
        f.write(f"Noise level (SNR): {noise_level} dB\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Initial MSE: {initial_mse:.6f}\n")
        f.write(f"Final MSE: {final_mse:.6f}\n")
        f.write(f"MSE improvement: {initial_mse - final_mse:.6f} ({(1 - final_mse/initial_mse) * 100:.2f}%)\n")
    
    # Save decoder state
    torch.save(model.decoder.state_dict(), f"{decoder_dir}/decoder_state.pt")
    
    print(f"Decoder-only training completed for {shape_type} shape, SNR: {noise_level} dB")
    print(f"Results saved to: {decoder_dir}/")
    
    return final_mse

def plot_noise_comparison(noise_levels, initial_mse_values, final_mse_values, output_dir):
    """
    Plot the comparison of MSE values for initial and final shapes across different noise levels
    """
    plt.figure(figsize=(12, 8))
    
    # Plot lines
    plt.plot(noise_levels, initial_mse_values, 'o-', color='red', label='Initial Shape')
    plt.plot(noise_levels, final_mse_values, 'o-', color='blue', label='Final Shape')
    
    # Add grid and labels
    plt.grid(True)
    plt.xlabel("Noise Level (SNR in dB)")
    plt.ylabel("MSE after 100 epochs of decoder-only training")
    plt.title("Comparison of Initial vs Final Shape Performance Across Noise Levels")
    
    # Add legend
    plt.legend()
    
    # Invert x-axis (higher SNR means less noise)
    plt.gca().invert_xaxis()
    
    # Save figure
    comparison_path = os.path.join(output_dir, "noise_level_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {os.path.abspath(comparison_path)}")
    
    # Also save the data
    results_data = {
        'noise_levels': noise_levels,
        'initial_mse_values': initial_mse_values,
        'final_mse_values': final_mse_values
    }
    np.save(os.path.join(output_dir, "noise_comparison_data.npy"), results_data)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run hyperspectral autoencoder noise experiment on SWIR data.")
    parser.add_argument("--cache", type=str, default="cache/aviris_tiles_swir_all.pt", help="Path to cache file for processed data")
    parser.add_argument("--use-cache", action="store_true", help="Use cached data if available")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (e.g., 0, 1, 2, 3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs for full training")
    parser.add_argument("--decoder-epochs", type=int, default=100, help="Number of epochs for decoder-only training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for optimizer")
    
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")
    
    # Check if the model path exists
    # model_path = "outputs_three_stage_20250216_180408/stageA/shape2spec_stageA.pt"
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
    
    # Create base output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"noise_experiment_swir_all_results_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create cache directory if it doesn't exist
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    
    # Define noise levels to test (in dB)
    noise_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    # noise_levels = [0, 20]  # Uncomment for quick testing
    
    # Dictionary to store paths to initial and final shapes for each noise level
    shape_paths = {}
    
    # Run full training for each noise level
    for noise_level in noise_levels:
        print(f"\n{'='*50}")
        print(f"Running full training with noise level: {noise_level} dB")
        print(f"{'='*50}\n")
        
        initial_shape_path, final_shape_path, output_dir = train_with_noise_level(
            model_path=model_path,
            output_dir=base_output_dir,
            noise_level=noise_level,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            cache_file=args.cache,
            use_cache=args.use_cache
        )
        
        # Store paths for later use
        shape_paths[noise_level] = {
            'initial': initial_shape_path,
            'final': final_shape_path,
            'output_dir': output_dir
        }
    
    # Create directory for decoder-only results
    decoder_output_dir = os.path.join(base_output_dir, "decoder_only_results")
    os.makedirs(decoder_output_dir, exist_ok=True)
    
    # Run decoder-only training for each shape and noise level
    initial_mse_values = []
    final_mse_values = []
    
    for noise_level in noise_levels:
        print(f"\n{'='*50}")
        print(f"Running decoder-only training for noise level: {noise_level} dB")
        print(f"{'='*50}\n")
        
        # Get paths to initial and final shapes
        paths = shape_paths[noise_level]
        
        # Train decoder with initial shape
        print(f"\nTraining decoder with initial shape, SNR: {noise_level} dB")
        initial_mse = train_decoder_only(
            model_path=model_path,
            shape_path=paths['initial'],
            output_dir=decoder_output_dir,
            noise_level=noise_level,
            num_epochs=args.decoder_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            cache_file=args.cache,
            use_cache=args.use_cache
        )
        initial_mse_values.append(initial_mse)
        
        # Train decoder with final shape
        print(f"\nTraining decoder with final shape, SNR: {noise_level} dB")
        final_mse = train_decoder_only(
            model_path=model_path,
            shape_path=paths['final'],
            output_dir=decoder_output_dir,
            noise_level=noise_level,
            num_epochs=args.decoder_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            cache_file=args.cache,
            use_cache=args.use_cache
        )
        final_mse_values.append(final_mse)
    
    # Plot comparison of decoder-only training results
    plot_noise_comparison(
        noise_levels=noise_levels,
        initial_mse_values=initial_mse_values,
        final_mse_values=final_mse_values,
        output_dir=base_output_dir
    )
    
    print("\nAll experiments completed!")
    print(f"Results saved to: {os.path.abspath(base_output_dir)}")

if __name__ == "__main__":
    main()
