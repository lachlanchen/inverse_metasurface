import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Function to load AVIRIS data from multiple folders with proper path handling
def load_aviris_data(aviris_base_path="AVIRIS_SWIR", tile_size=100, num_bands=100, cache_file=None, use_cache=False):
    """
    Load AVIRIS hyperspectral data from all subfolders and crop it into tiles

    Parameters:
    aviris_base_path: Base path to the AVIRIS_SWIR directory
    tile_size: Size of the tiles (square)
    num_bands: Number of spectral bands to use (the model expects 100)
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
    if not os.path.exists(aviris_base_path) or not os.path.isdir(aviris_base_path):
        raise FileNotFoundError(f"AVIRIS_SWIR directory not found at: {os.path.abspath(aviris_base_path)}")
    
    print(f"Looking for AVIRIS_SWIR data in: {os.path.abspath(aviris_base_path)}")
    
    # Get list of subfolders
    subfolders = [f for f in os.listdir(aviris_base_path) 
                 if os.path.isdir(os.path.join(aviris_base_path, f))]
    
    if not subfolders:
        raise FileNotFoundError(f"No subfolders found in AVIRIS_SWIR directory: {os.path.abspath(aviris_base_path)}")
    
    print(f"Found {len(subfolders)} subfolders: {', '.join(subfolders)}")
    
    # List to store all loaded data tensors
    all_data_list = []
    
    # Process each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(aviris_base_path, subfolder)
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
            # If the data has channels as the first dimension, transpose it
            if data.shape[2] != num_bands and data.shape[0] == num_bands:
                data = data.permute(1, 2, 0)
                print(f"Rearranged data to shape: {data.shape}")
        
        # If the data has more than num_bands bands, select a subset or combine bands
        if data.shape[2] > num_bands:
            print(f"Reducing bands from {data.shape[2]} to {num_bands}")
            # Bin the bands (averages groups of bands)
            bin_size = data.shape[2] // num_bands
            binned_data = torch.zeros((data.shape[0], data.shape[1], num_bands), dtype=data.dtype)
            
            for i in range(num_bands):
                start_idx = i * bin_size
                end_idx = min((i + 1) * bin_size, data.shape[2])
                binned_data[:, :, i] = torch.mean(data[:, :, start_idx:end_idx], dim=2)
            
            data = binned_data
            print(f"Data shape after band reduction: {data.shape}")
        
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
        num_bands_actual = all_data_list[0].shape[-1]
        all_same_bands = all(tensor.shape[-1] == num_bands_actual for tensor in all_data_list)
        
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
    plt.savefig(save_path)
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

# Hyperspectral Autoencoder that uses shape2spectrum as filters
class HyperspectralAutoencoder(nn.Module):
    def __init__(self, shape2spec_model_path, target_snr=1):
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
        Signal power calculation uses detach() to prevent noise calculation from participating
        in gradient propagation.
        """
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

        # ----------------- Add noise if needed ------------------
        # encoded_channels_first = self.add_noise(encoded_channels_first)
        
        # Convert encoded data back to channels-last format [B,H,W,C]
        encoded = encoded_channels_first.permute(0, 2, 3, 1)
        
        # Decode: use the CNN decoder to expand from 11 to 100 bands
        decoded_channels_first = self.decoder(encoded_channels_first)
        
        # Convert back to original format [B,H,W,C]
        decoded = decoded_channels_first.permute(0, 2, 3, 1)
        
        return decoded, encoded

# Training and visualization
def train_and_visualize_autoencoder(model_path, output_dir, batch_size=10, num_epochs=500, 
                              learning_rate=0.001, freeze_decoder=False, alternating_freeze=True,
                              freeze_interval=10, cache_file=None, use_cache=False):
    """
    Train and visualize the hyperspectral autoencoder
    
    Parameters:
    model_path: Path to the pretrained shape2spec model
    output_dir: Directory to save outputs
    batch_size: Batch size for training
    num_epochs: Number of training epochs
    learning_rate: Learning rate for optimizer
    freeze_decoder: If True, freeze the decoder so only shape parameters are optimized
    alternating_freeze: If True, alternate between freezing decoder and optimizing both
    freeze_interval: Number of epochs to spend in each freeze/unfreeze state
    cache_file: Path to cache file for storing processed data
    use_cache: If True, try to load from cache file first
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}/")
    
    # Create subfolder for intermediate visualizations
    viz_dir = os.path.join(output_dir, "intermediate_viz")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable debug mode to print shapes
    debug = True
    
    # Load AVIRIS_SWIR data from both folders with caching support
    data = load_aviris_data(aviris_base_path="AVIRIS_SWIR", tile_size=100, num_bands=100, cache_file=cache_file, use_cache=use_cache)
    data = data.to(device)
    
    if debug:
        print("Data shape:", data.shape)
    
    # Create dataset and dataloader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size//2, shuffle=True)  # Use smaller batches for training
    
    # Initialize model
    model = HyperspectralAutoencoder(model_path).to(device)
    
    # Function to toggle decoder freezing
    def set_decoder_frozen(model, freeze=True):
        if freeze:
            print("Freezing decoder - only shape parameters will be optimized")
            # Set requires_grad=False for all decoder parameters
            for param in model.decoder.parameters():
                param.requires_grad = False
        else:
            print("Unfreezing decoder - optimizing both shape and decoder")
            # Set requires_grad=True for all decoder parameters
            for param in model.decoder.parameters():
                param.requires_grad = True
                
        # Print trainable parameters count
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"Number of trainable parameters: {len(trainable_params)}")
        for i, p in enumerate(trainable_params[:5]):  # Show first 5 only
            print(f"Trainable param {i}: shape {p.shape}")
        if len(trainable_params) > 5:
            print(f"...and {len(trainable_params)-5} more")
    
    # Initial freeze state based on parameters
    if alternating_freeze:
        # Start with decoder frozen if using alternating strategy
        set_decoder_frozen(model, freeze=True)
    else:
        # Otherwise use the freeze_decoder parameter
        set_decoder_frozen(model, freeze=freeze_decoder)
    
    # Print model structure if in debug mode
    if debug:
        print("Model initialized successfully")
        print("Model structure:")
        print(model)
    
    # Get initial shape and filters for visualization
    initial_shape = model.get_current_shape().detach().cpu().numpy()
    initial_filters = model.get_current_filters().detach().cpu().numpy()
    
    if debug:
        print("Initial shape size:", initial_shape.shape)
        print("Initial filters size:", initial_filters.shape)
    
    # Save initial shape with C4 symmetry
    plot_shape_with_c4(initial_shape, "Initial Shape", f"{output_dir}/initial_shape.png")
    
    # Save initial filters
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(initial_filters[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title("Initial Spectral Filters")
    plt.legend()
    plt.savefig(f"{output_dir}/initial_filters.png")
    plt.close()
    
    # Before training, get reconstruction of sample at index 5 (not 50)
    with torch.no_grad():
        # Use index 5 instead of 50 since our batch size is only 10
        sample_idx = min(5, len(data) - 1)  # Make sure index is within range
        sample = data[sample_idx:sample_idx+1]  # Add batch dimension
        if debug:
            print("Sample shape:", sample.shape)
        initial_recon, encoded = model(sample)
        if debug:
            print("Initial recon shape:", initial_recon.shape)
            print("Encoded shape:", encoded.shape)
        initial_recon = initial_recon.detach().cpu().numpy()[0]  # Remove batch dimension
        encoded = encoded.detach().cpu().numpy()[0]  # Remove batch dimension
        sample = sample.detach().cpu().numpy()[0]  # Remove batch dimension
    
    # Save initial reconstruction of sample
    plt.figure(figsize=(16, 8))
    # Plot a subset of bands for clarity
    bands_to_plot = [0, 25, 50, 75, 99]  # First, middle, and last bands
    
    for i, band in enumerate(bands_to_plot):
        plt.subplot(2, len(bands_to_plot), i+1)
        plt.imshow(sample[:, :, band], cmap='viridis')
        plt.title(f'Original Band {band}')
        plt.colorbar()
        
        plt.subplot(2, len(bands_to_plot), i+1+len(bands_to_plot))
        plt.imshow(initial_recon[:, :, band], cmap='viridis')
        plt.title(f'Initial Recon Band {band}')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/initial_reconstruction_sample_{sample_idx}.png")
    plt.close()
    
    # Training loop
    losses = []
    
    print("Starting training...")
    
    # Determine optimizer learning rate based on freeze state
    current_lr = 0.005 if alternating_freeze or freeze_decoder else learning_rate
    
    # Create optimizer - will only optimize parameters with requires_grad=True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        # Toggle freezing if using alternating strategy
        if alternating_freeze and epoch > 0 and epoch % freeze_interval == 0:
            # Check current state of decoder parameters
            is_frozen = not model.decoder.parameters().__next__().requires_grad
            # Toggle state
            set_decoder_frozen(model, freeze=not is_frozen)
            
            # Update optimizer with new learning rate based on freeze state
            is_frozen = not is_frozen  # Get new state after toggling
            current_lr = 0.005 if is_frozen else learning_rate
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr)
            print(f"Switched optimization mode. New learning rate: {current_lr}")
        
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
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Save intermediate shapes and visualizations
        if (epoch+1) % 10 == 0 or epoch == 0:
            # Get current shape
            current_shape = model.get_current_shape().detach().cpu().numpy()
            
            # Save shape visualization to intermediate directory
            plot_shape_with_c4(
                current_shape, 
                f"Shape at Epoch {epoch+1}", 
                f"{viz_dir}/shape_epoch_{epoch+1}.png"
            )
            
            # Every 50 epochs, save the current filters too
            if (epoch+1) % 50 == 0:
                current_filters = model.get_current_filters().detach().cpu().numpy()
                plt.figure(figsize=(12, 8))
                for i in range(11):
                    plt.plot(current_filters[i], label=f'Filter {i}' if i % 3 == 0 else None)
                plt.grid(True)
                plt.xlabel("Wavelength Index")
                plt.ylabel("Filter Value")
                plt.title(f"Spectral Filters at Epoch {epoch+1}")
                plt.legend()
                plt.savefig(f"{viz_dir}/filters_epoch_{epoch+1}.png")
                plt.close()
    
    # Get final shape and filters
    final_shape = model.get_current_shape().detach().cpu().numpy()
    final_filters = model.get_current_filters().detach().cpu().numpy()
    
    # Save final shape
    plt.figure(figsize=(6, 6))
    plt.scatter(final_shape[:, 1], final_shape[:, 2], color='blue', s=100 * final_shape[:, 0])
    plt.xlim(-0.7, 0.7)
    plt.ylim(-0.7, 0.7)
    plt.grid(True)
    plt.title("Final Shape (Q1 Points)")
    plt.savefig(f"{output_dir}/final_shape.png")
    plt.close()
    
    # Save final filters
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(final_filters[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title("Final Spectral Filters")
    plt.legend()
    plt.savefig(f"{output_dir}/final_filters.png")
    plt.close()
    
    # Compare initial and final filters
    plt.figure(figsize=(15, 10))
    for i in range(11):
        plt.subplot(4, 3, i+1)
        plt.plot(initial_filters[i], 'r--', label='Initial')
        plt.plot(final_filters[i], 'b-', label='Final')
        plt.grid(True)
        plt.title(f'Filter {i}')
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/filter_comparison.png")
    plt.close()
    
    # After training, get reconstruction of the same sample
    with torch.no_grad():
        sample = data[sample_idx:sample_idx+1]  # Add batch dimension
        final_recon, encoded = model(sample)
        final_recon = final_recon.detach().cpu().numpy()[0]  # Remove batch dimension
        encoded = encoded.detach().cpu().numpy()[0]  # Remove batch dimension
        sample = sample.detach().cpu().numpy()[0]  # Remove batch dimension
    
    # Save final reconstruction of sample
    plt.figure(figsize=(16, 8))
    # Plot a subset of bands for clarity
    for i, band in enumerate(bands_to_plot):
        plt.subplot(2, len(bands_to_plot), i+1)
        plt.imshow(sample[:, :, band], cmap='viridis')
        plt.title(f'Original Band {band}')
        plt.colorbar()
        
        plt.subplot(2, len(bands_to_plot), i+1+len(bands_to_plot))
        plt.imshow(final_recon[:, :, band], cmap='viridis')
        plt.title(f'Final Recon Band {band}')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_reconstruction_sample_{sample_idx}.png")
    plt.close()
    
    # Visualize the encoded representation (11 bands)
    plt.figure(figsize=(15, 6))
    for i in range(min(11, encoded.shape[-1])):
        plt.subplot(2, 6, i+1)
        plt.imshow(encoded[:, :, i], cmap='viridis')
        plt.title(f'Encoded Band {i}')
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/encoded_representation_sample_{sample_idx}.png")
    plt.close()
    
    # Save loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.savefig(f"{output_dir}/training_loss.png")
    plt.close()
    
    # Also save numerical data
    np.save(f"{output_dir}/initial_shape.npy", initial_shape)
    np.save(f"{output_dir}/final_shape.npy", final_shape)
    np.save(f"{output_dir}/initial_filters.npy", initial_filters)
    np.save(f"{output_dir}/final_filters.npy", final_filters)
    np.save(f"{output_dir}/losses.npy", np.array(losses))
    np.save(f"{output_dir}/sample_{sample_idx}.npy", sample)
    np.save(f"{output_dir}/initial_recon_{sample_idx}.npy", initial_recon)
    np.save(f"{output_dir}/final_recon_{sample_idx}.npy", final_recon)
    np.save(f"{output_dir}/encoded_{sample_idx}.npy", encoded)
    
    # Create log file to save training parameters
    with open(f"{output_dir}/training_params.txt", "w") as f:
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Decoder frozen: {freeze_decoder}\n")
        f.write(f"Alternating freeze: {alternating_freeze}\n")
        f.write(f"Freeze interval: {freeze_interval}\n")
        f.write(f"Cache file: {cache_file}\n")
        f.write(f"Used cache: {use_cache}\n")
        f.write("\n")
        
        # Save initial shape parameters
        f.write("Initial shape parameters:\n")
        f.write(f"raw_params:\n{model.raw_params.detach().cpu().numpy()}\n")
        f.write(f"raw_shift: {model.raw_shift.item()}\n\n")
        
        # Save final shape parameters after training
        f.write("Final shape parameters after training:\n")
        f.write(f"raw_params:\n{model.raw_params.detach().cpu().numpy()}\n")
        f.write(f"raw_shift: {model.raw_shift.item()}\n")
    
    print(f"Training completed. All results saved to {output_dir}/")
    return model, losses, output_dir

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a hyperspectral autoencoder with shape optimization.")
    parser.add_argument("--cache", type=str, default="cache/aviris_tiles_all.pt", help="Path to cache file for processed data")
    parser.add_argument("--use-cache", action="store_true", help="Use cached data if available")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (e.g., 0, 1, 2, 3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--freeze-decoder", action="store_true", help="Freeze decoder weights")
    parser.add_argument("--alternating-freeze", action="store_true", help="Alternate between freezing decoder and optimizing both")
    parser.add_argument("--freeze-interval", type=int, default=10, help="Number of epochs to spend in each freeze/unfreeze state")
    
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")
    
    # Check if the model path exists
    model_path = "outputs_three_stage_20250216_180408/stageA/shape2spec_stageA.pt"
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
    
    output_dir = "hyperspectral_autoencoder_all"
    
    # Create cache directory if it doesn't exist
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    
    # Print PyTorch version for debugging
    print(f"PyTorch version: {torch.__version__}")
    
    # Use the model with real AVIRIS data from all folders
    model, losses, output_dir = train_and_visualize_autoencoder(
        model_path=model_path,
        output_dir=output_dir,
        batch_size=args.batch_size,  
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        freeze_decoder=args.freeze_decoder,
        alternating_freeze=args.alternating_freeze,
        freeze_interval=args.freeze_interval,
        cache_file=args.cache,
        use_cache=args.use_cache
    )
