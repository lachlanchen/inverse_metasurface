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

# Function to split data into train and test sets
def split_data(data, test_size=0.2, seed=42):
    """
    Split data into training and test sets
    
    Parameters:
    data: PyTorch tensor of shape [n_samples, height, width, channels]
    test_size: Fraction of data to use for testing (default: 0.2)
    seed: Random seed for reproducibility
    
    Returns:
    tuple: (train_data, test_data) as PyTorch tensors
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Calculate the split point
    n_samples = data.size(0)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    # Create a random permutation of indices
    indices = torch.randperm(n_samples)
    
    # Split the data
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_data = data[train_indices]
    test_data = data[test_indices]
    
    print(f"Data split into {n_train} training samples and {n_test} test samples")
    
    return train_data, test_data

# Function to load AVIRIS data with proper path handling
def load_aviris_data(aviris_path, tile_size=100, num_bands=100, cache_file=None, use_cache=False):
    """
    Load AVIRIS hyperspectral data and crop it into tiles
    """
    import spectral
    import numpy as np
    import torch
    
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
    
    # Check if file exists and print absolute path for debugging
    abs_path = os.path.abspath(aviris_path)
    print(f"Looking for AVIRIS data at: {abs_path}")
    
    if not os.path.exists(aviris_path):
        # Try alternate paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        alternative_paths = [
            os.path.join(base_dir, aviris_path),
            # Try some variations of the path
            os.path.join(base_dir, "AVIRIS", "AV320231008t173943_L2A_OE_main_98b13fff", "AV320231008t173943_L2A_OE_main_98b13fff_RFL_ORT.hdr"),
            os.path.join("AVIRIS", "AV320231008t173943_L2A_OE_main_98b13fff", "AV320231008t173943_L2A_OE_main_98b13fff_RFL_ORT.hdr"),
            # Try without any directory structure
            "AV320231008t173943_L2A_OE_main_98b13fff_RFL_ORT.hdr"
        ]
        
        for alt_path in alternative_paths:
            print(f"Trying alternative path: {os.path.abspath(alt_path)}")
            if os.path.exists(alt_path):
                aviris_path = alt_path
                print(f"Found file at: {os.path.abspath(aviris_path)}")
                break
        
        if not os.path.exists(aviris_path):
            # List contents of directories to help diagnose the issue
            print("\nListing contents of current directory:")
            print(os.listdir("."))
            
            if os.path.exists("AVIRIS"):
                print("\nListing contents of AVIRIS directory:")
                print(os.listdir("AVIRIS"))
                
                aviris_subdir = "AVIRIS/AV320231008t173943_L2A_OE_main_98b13fff"
                if os.path.exists(aviris_subdir):
                    print(f"\nListing contents of {aviris_subdir}:")
                    print(os.listdir(aviris_subdir))
            
            raise FileNotFoundError(f"Could not find AVIRIS data file. Tried paths: {[abs_path] + alternative_paths}")

    # Open the ENVI image using the header file
    try:
        img = spectral.open_image(aviris_path)
        print(f"Successfully opened AVIRIS data file: {aviris_path}")
    except Exception as e:
        print(f"Error opening AVIRIS data with spectral library: {str(e)}")
        raise

    # Load the image data into a NumPy array
    data = img.load()
    print(f"Original data shape: {data.shape}")

    # Handle any no-data values in the AVIRIS data
    # Replace -9999 (common no-data value) with 0
    if np.min(data) < -1000:
        data = np.where(data < -1000, 0, data)
        print(f"Replaced no-data values. New min value: {np.min(data)}")

    # Normalize the data to [0, 1] range
    data_min = np.min(data)
    data_max = np.max(data)
    data = (data - data_min) / (data_max - data_min)
    print(f"Data normalized to range [0, 1]. Min: {np.min(data)}, Max: {np.max(data)}")

    # If the data has more than 100 bands, select a subset or combine bands
    if data.shape[2] > num_bands:
        print(f"Reducing bands from {data.shape[2]} to {num_bands}")
        # Option 2: Bin the bands (averages groups of bands)
        bin_size = data.shape[2] // num_bands
        binned_data = np.zeros((data.shape[0], data.shape[1], num_bands))
        
        for i in range(num_bands):
            start_idx = i * bin_size
            end_idx = min((i + 1) * bin_size, data.shape[2])
            binned_data[:, :, i] = np.mean(data[:, :, start_idx:end_idx], axis=2)
        
        data = binned_data

    print(f"Processed data shape: {data.shape}")

    # Calculate how many tiles we can extract
    h_tiles = data.shape[0] // tile_size
    w_tiles = data.shape[1] // tile_size
    total_tiles = h_tiles * w_tiles

    print(f"Creating {h_tiles}×{w_tiles} = {total_tiles} tiles of size {tile_size}×{tile_size}")

    # Create the tiles
    tiles = []
    for i in range(h_tiles):
        for j in range(w_tiles):
            h_start = i * tile_size
            h_end = (i + 1) * tile_size
            w_start = j * tile_size
            w_end = (j + 1) * tile_size
            
            tile = data[h_start:h_end, w_start:w_end, :]
            tiles.append(tile)

    # Convert to tensor
    tiles_tensor = torch.tensor(np.array(tiles), dtype=torch.float32)
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
    def __init__(self, shape2spec_model_path, target_snr=None):
        super().__init__()

        # 目标信噪比（单位 dB）作为模型参数
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
        向 tensor 添加高斯白噪声，噪声水平由目标信噪比 self.target_snr 决定。
        计算信号功率时使用 detach() 来防止噪声计算参与梯度传播。
        """
        if self.target_snr is None:
            return tensor
            
        # 计算信号功率（均值平方），并使用 detach() 分离计算图
        signal_power = tensor.detach().pow(2).mean()
        # 将信号功率转换为分贝
        signal_power_db = 10 * torch.log10(signal_power)
        # 计算噪声功率（分贝）
        noise_power_db = signal_power_db - self.target_snr
        # 将噪声功率转换回线性尺度
        noise_power = 10 ** (noise_power_db / 10)
        # 生成与输入 tensor 形状相同的高斯白噪声
        noise = torch.randn_like(tensor) * torch.sqrt(noise_power)
        # 返回添加噪声后的 tensor
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

        # ----------------- 添加噪声 ------------------
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
def train_with_noise_level(model_path, output_dir, noise_level, train_data, test_data, batch_size=10, num_epochs=500, learning_rate=0.001):
    """
    Train and visualize the hyperspectral autoencoder with a specific noise level
    
    Parameters:
    model_path: Path to the pretrained shape2spec model
    output_dir: Directory to save outputs
    noise_level: SNR level in dB to apply during training
    train_data: Training dataset tensor
    test_data: Test dataset tensor for evaluation
    batch_size: Batch size for training
    num_epochs: Number of training epochs
    learning_rate: Learning rate for optimizer
    
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
    
    # Move data to device
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    
    print("Training data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)
    
    # Create dataset and dataloader for training
    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size//2, shuffle=True)
    
    # Create dataset and dataloader for testing
    test_dataset = TensorDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size//2, shuffle=False)
    
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
    
    # Evaluate initial MSE on test set
    with torch.no_grad():
        test_losses = []
        for batch in test_dataloader:
            x_test = batch[0]
            recon_test, _ = model(x_test)
            test_loss = ((recon_test - x_test) ** 2).mean().item()
            test_losses.append(test_loss)
        initial_test_mse = sum(test_losses) / len(test_losses)
        print(f"Initial MSE on test set: {initial_test_mse:.6f}")
    
    # Training loop
    train_losses = []
    test_mse_values = [initial_test_mse]
    condition_numbers = [initial_condition_number]
    
    print(f"Starting training with SNR: {noise_level} dB...")
    
    # Create optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Evaluation on test set
        model.eval()
        with torch.no_grad():
            test_losses = []
            for batch in test_dataloader:
                x_test = batch[0]
                recon_test, _ = model(x_test)
                test_loss = ((recon_test - x_test) ** 2).mean().item()
                test_losses.append(test_loss)
            current_test_mse = sum(test_losses) / len(test_losses)
            test_mse_values.append(current_test_mse)
            
            # Calculate condition number for current epoch
            current_filters = model.get_current_filters().detach().cpu()
            current_condition_number = calculate_condition_number(current_filters)
            condition_numbers.append(current_condition_number)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Test MSE: {current_test_mse:.6f}, Condition Number: {current_condition_number:.4f}")
        
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
    
    # Calculate final MSE on test set
    model.eval()
    with torch.no_grad():
        test_losses = []
        for batch in test_dataloader:
            x_test = batch[0]
            recon_test, _ = model(x_test)
            test_loss = ((recon_test - x_test) ** 2).mean().item()
            test_losses.append(test_loss)
        final_test_mse = sum(test_losses) / len(test_losses)
        print(f"Final MSE on test set: {final_test_mse:.6f}")
    
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
    
    # Plot training loss and test MSE
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_mse_values[1:], label='Test MSE')  # Skip first value to align with training epochs
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Loss and Test MSE (SNR: {noise_level} dB)")
    plt.legend()
    loss_plot_path = f"{output_dir}/training_test_loss.png"
    plt.savefig(loss_plot_path)
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
    np.save(f"{output_dir}/train_losses.npy", np.array(train_losses))
    np.save(f"{output_dir}/test_mse_values.npy", np.array(test_mse_values))
    
    # Create log file to save training parameters
    with open(f"{output_dir}/training_params.txt", "w") as f:
        f.write(f"Noise level (SNR): {noise_level} dB\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Training data size: {len(train_data)}\n")
        f.write(f"Test data size: {len(test_data)}\n")
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
        
        # Save MSE information on test set
        f.write(f"Initial MSE on test set: {initial_test_mse:.6f}\n")
        f.write(f"Final MSE on test set: {final_test_mse:.6f}\n")
        f.write(f"Test MSE improvement: {initial_test_mse - final_test_mse:.6f} ({(1 - final_test_mse/initial_test_mse) * 100:.2f}%)\n")
    
    print(f"Training with SNR {noise_level} dB completed. All results saved to {output_dir}/")
    
    # Return paths to the initial and final shapes for later training
    initial_shape_npy_path = f"{output_dir}/initial_shape.npy"
    final_shape_npy_path = f"{output_dir}/final_shape.npy"
    return initial_shape_npy_path, final_shape_npy_path, output_dir

# Function to train the decoder only with a fixed shape
def train_decoder_only(model_path, shape_path, output_dir, noise_level, train_data, test_data, 
                      num_epochs=100, batch_size=10, learning_rate=0.001):
    """
    Train only the decoder with a fixed shape
    
    Parameters:
    model_path: Path to the pretrained shape2spec model
    shape_path: Path to the saved shape numpy file
    output_dir: Directory to save outputs
    noise_level: SNR level in dB
    train_data: Training data tensor
    test_data: Test data tensor for evaluation
    num_epochs: Number of training epochs
    batch_size: Batch size for training
    learning_rate: Learning rate for optimizer
    
    Returns:
    float: Final test MSE after training
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
    
    # Move data to device
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    
    print("Training data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)
    
    # Create dataset and dataloader for training
    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size//2, shuffle=True)
    
    # Create dataset and dataloader for testing
    test_dataset = TensorDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size//2, shuffle=False)
    
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
    
    # Evaluate initial MSE on test set
    model.eval()
    with torch.no_grad():
        test_losses = []
        for batch in test_dataloader:
            x_test = batch[0]
            recon_test, _ = model(x_test)
            test_loss = ((recon_test - x_test) ** 2).mean().item()
            test_losses.append(test_loss)
        initial_test_mse = sum(test_losses) / len(test_losses)
        print(f"Initial MSE on test set with fixed {shape_type} shape: {initial_test_mse:.6f}")
    
    # Create optimizer
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    test_mse_values = [initial_test_mse]
    
    print(f"Starting decoder-only training for {shape_type} shape, SNR: {noise_level} dB...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Evaluation on test set
        model.eval()
        with torch.no_grad():
            test_losses = []
            for batch in test_dataloader:
                x_test = batch[0]
                recon_test, _ = model(x_test)
                test_loss = ((recon_test - x_test) ** 2).mean().item()
                test_losses.append(test_loss)
            current_test_mse = sum(test_losses) / len(test_losses)
            test_mse_values.append(current_test_mse)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Test MSE: {current_test_mse:.6f}")
    
    # Calculate final MSE on test set
    model.eval()
    with torch.no_grad():
        test_losses = []
        for batch in test_dataloader:
            x_test = batch[0]
            recon_test, _ = model(x_test)
            test_loss = ((recon_test - x_test) ** 2).mean().item()
            test_losses.append(test_loss)
        final_test_mse = sum(test_losses) / len(test_losses)
        print(f"Final MSE on test set after decoder-only training: {final_test_mse:.6f}")
    
    # Plot training loss and test MSE
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_mse_values[1:], label='Test MSE')  # Skip first value to align with training epochs
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Decoder-only Training Loss and Test MSE ({shape_type} shape, SNR: {noise_level} dB)")
    plt.legend()
    plt.savefig(f"{decoder_dir}/train_test_loss.png")
    plt.close()
    
    # Save numerical data
    np.save(f"{decoder_dir}/train_losses.npy", np.array(train_losses))
    np.save(f"{decoder_dir}/test_mse_values.npy", np.array(test_mse_values))
    
    # Save log file
    with open(f"{decoder_dir}/training_params.txt", "w") as f:
        f.write(f"Shape type: {shape_type}\n")
        f.write(f"Shape path: {shape_path}\n")
        f.write(f"Noise level (SNR): {noise_level} dB\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Training data size: {len(train_data)}\n")
        f.write(f"Test data size: {len(test_data)}\n")
        f.write(f"Initial MSE on test set: {initial_test_mse:.6f}\n")
        f.write(f"Final MSE on test set: {final_test_mse:.6f}\n")
        f.write(f"MSE improvement on test set: {initial_test_mse - final_test_mse:.6f} ({(1 - final_test_mse/initial_test_mse) * 100:.2f}%)\n")
    
    # Save decoder state
    torch.save(model.decoder.state_dict(), f"{decoder_dir}/decoder_state.pt")
    
    print(f"Decoder-only training completed for {shape_type} shape, SNR: {noise_level} dB")
    print(f"Results saved to: {decoder_dir}/")
    
    return final_test_mse

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
    plt.ylabel("MSE on Test Set after 100 epochs of decoder-only training")
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
    parser = argparse.ArgumentParser(description="Run hyperspectral autoencoder noise experiment.")
    parser.add_argument("--cache", type=str, default="cache/aviris_tiles.pt", help="Path to cache file for processed data")
    parser.add_argument("--use-cache", action="store_true", help="Use cached data if available")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (e.g., 0, 1, 2, 3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs for full training")
    parser.add_argument("--decoder-epochs", type=int, default=100, help="Number of epochs for decoder-only training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use for testing")
    
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
    
    # Create base output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"noise_experiment_results_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create cache directory if it doesn't exist
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    
    # Try to find the correct path to the AVIRIS data
    aviris_path = "AVIRIS/AV320231008t173943_L2A_OE_main_98b13fff/AV320231008t173943_L2A_OE_main_98b13fff_RFL_ORT.hdr"
    
    # Load full dataset
    print("Loading AVIRIS data...")
    full_data = load_aviris_data(aviris_path, tile_size=100, num_bands=100, cache_file=args.cache, use_cache=args.use_cache)
    
    # Split into train and test sets
    print(f"Splitting data into train and test sets (test_size={args.test_size})...")
    train_data, test_data = split_data(full_data, test_size=args.test_size)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Define noise levels to test (in dB)
    noise_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    
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
            train_data=train_data,
            test_data=test_data,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate
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
            train_data=train_data,
            test_data=test_data,
            num_epochs=args.decoder_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        initial_mse_values.append(initial_mse)
        
        # Train decoder with final shape
        print(f"\nTraining decoder with final shape, SNR: {noise_level} dB")
        final_mse = train_decoder_only(
            model_path=model_path,
            shape_path=paths['final'],
            output_dir=decoder_output_dir,
            noise_level=noise_level,
            train_data=train_data,
            test_data=test_data,
            num_epochs=args.decoder_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
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
