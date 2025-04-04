import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Function to load AVIRIS data with proper path handling
def load_aviris_data(aviris_path, tile_size=128, num_bands=100):
    """
    Load AVIRIS hyperspectral data and crop it into tiles

    Parameters:
    aviris_path: Path to the AVIRIS header file
    tile_size: Size of the tiles (square)
    num_bands: Number of spectral bands to use (the model expects 100)

    Returns:
    torch.Tensor of shape [num_tiles, tile_size, tile_size, num_bands]
    """
    import spectral
    import numpy as np
    import torch
    
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
    Forward: threshold x at `thresh` (returns 1.0 where x>=thresh, 0 otherwise).
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
    def __init__(self, shape2spec_model_path):
        super().__init__()
        
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
                              freeze_interval=10):
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
    
    # Try to find the correct path to the AVIRIS data
    # CORRECT PATH: Use the path from your directory structure
    aviris_path = "AVIRIS/AV320231008t173943_L2A_OE_main_98b13fff/AV320231008t173943_L2A_OE_main_98b13fff_RFL_ORT.hdr"
    
    # Load real AVIRIS data
    data = load_aviris_data(aviris_path, tile_size=128, num_bands=100)
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
    
    output_dir = "hyperspectral_autoencoder"
    
    # Print PyTorch version for debugging
    print(f"PyTorch version: {torch.__version__}")
    
    # Use the model with real AVIRIS data
    model, losses, output_dir = train_and_visualize_autoencoder(
        model_path=model_path,
        output_dir=output_dir,
        batch_size=64,  # Reduced from 128 to avoid memory issues with larger tiles
        num_epochs=5000,  # Set to 500 epochs by default
        learning_rate=0.001,
        freeze_decoder=False,  # This is ignored when alternating_freeze is True
        alternating_freeze=False,  # Use alternating optimization strategy 
        freeze_interval=10  # Toggle freezing every 10 epochs
    )