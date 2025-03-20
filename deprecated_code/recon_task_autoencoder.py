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
        self.raw_params = nn.Parameter(torch.randn(4, 2) * 0.1)  # Initialize close to zero
        self.raw_shift = nn.Parameter(torch.randn(1) * 0.1)      # Initialize close to zero
        
        # Decoder: Simple convolutional network to reconstruct 100 bands from 11
        self.decoder = nn.Sequential(
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
        with torch.no_grad():
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
        
        # Create spectral dimension reduction using differentiable shape filtering
        # Manual implementation of spectral filtering without Conv2d
        # This is simpler and avoids the potential dimension issues
        
        # Create output tensor for encoded data
        encoded_channels_first = torch.zeros(batch_size, 11, height, width, device=x.device)
        
        # Perform the spectral filtering (equivalent to 1x1 convolution)
        # For each output band, compute weighted sum over input bands
        for i in range(11):  # 11 output bands
            # Get the filter for this output band - shape [100]
            band_filter = filters[i] / 100.0  # Normalize
            
            # For each of the 100 input spectral bands, multiply by filter weight
            # and sum them up to get one output band
            for j in range(100):  # 100 input bands
                encoded_channels_first[:, i] += x_channels_first[:, j] * band_filter[j]
        
        # Convert encoded data back to channels-last format [B,H,W,C]
        encoded = encoded_channels_first.permute(0, 2, 3, 1)
        
        # Decode: use the CNN decoder to expand from 11 to 100 bands
        decoded_channels_first = self.decoder(encoded_channels_first)
        
        # Convert back to original format [B,H,W,C]
        decoded = decoded_channels_first.permute(0, 2, 3, 1)
        
        return decoded, encoded

# Function to generate synthetic hyperspectral data
def generate_synthetic_hyperspectral_data(batch_size=64, height=32, width=32, bands=100):
    """Generate synthetic hyperspectral data with spectral patterns"""
    # Create synthetic data with spectral patterns
    data = torch.zeros(batch_size, height, width, bands)
    
    # Create random spatial patterns
    for i in range(batch_size):
        # Generate random spatial patterns
        num_patterns = np.random.randint(3, 8)
        for _ in range(num_patterns):
            # Random position and size
            cx = np.random.randint(0, width)
            cy = np.random.randint(0, height)
            radius = np.random.randint(2, 10)
            
            # Random spectral signature (simple Gaussian)
            center = np.random.randint(20, 80)
            width_spectrum = np.random.randint(5, 20)
            amplitude = np.random.uniform(0.5, 1.0)
            
            # Create spectral signature
            spectrum = amplitude * np.exp(-((np.arange(bands) - center) ** 2) / (2 * width_spectrum ** 2))
            
            # Apply to spatial region (simple circle)
            for x in range(max(0, cx-radius), min(width, cx+radius)):
                for y in range(max(0, cy-radius), min(height, cy+radius)):
                    if ((x-cx)**2 + (y-cy)**2) <= radius**2:
                        data[i, y, x, :] += torch.tensor(spectrum, dtype=torch.float32)
    
    # Normalize to [0, 1]
    data = data / data.max()
    
    return data

# Training and visualization
def train_and_visualize_autoencoder(model_path, output_dir, batch_size=10, num_epochs=100, learning_rate=0.001):
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}/")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable debug mode to print shapes
    debug = True
    
    # Generate synthetic data (in real scenario, load your data here)
    data = generate_synthetic_hyperspectral_data(batch_size=batch_size, height=32, width=32, bands=100)
    data = data.to(device)
    
    if debug:
        print("Data shape:", data.shape)
    
    # Create dataset and dataloader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = HyperspectralAutoencoder(model_path).to(device)
    
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
    
    # Save initial shape
    plt.figure(figsize=(6, 6))
    plt.scatter(initial_shape[:, 1], initial_shape[:, 2], color='blue', s=100 * initial_shape[:, 0])
    plt.xlim(-0.7, 0.7)
    plt.ylim(-0.7, 0.7)
    plt.grid(True)
    plt.title("Initial Shape (Q1 Points)")
    plt.savefig(f"{output_dir}/initial_shape.png")
    plt.close()
    
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
        sample_idx = 5
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
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    
    print("Starting training...")
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
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Save intermediate results
        if (epoch+1) % 10 == 0 or epoch == 0:
            # Save current shape
            current_shape = model.get_current_shape().detach().cpu().numpy()
            plt.figure(figsize=(6, 6))
            plt.scatter(current_shape[:, 1], current_shape[:, 2], color='blue', s=100 * current_shape[:, 0])
            plt.xlim(-0.7, 0.7)
            plt.ylim(-0.7, 0.7)
            plt.grid(True)
            plt.title(f"Shape at Epoch {epoch+1}")
            plt.savefig(f"{output_dir}/shape_epoch_{epoch+1}.png")
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
    
    # Save final reconstruction of sample 50
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
    plt.savefig(f"{output_dir}/final_reconstruction_sample_50.png")
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
    
    print(f"Training completed. All results saved to {output_dir}/")
    return model, losses, output_dir

if __name__ == "__main__":
    model_path = "outputs_three_stage_20250216_180408/stageA/shape2spec_stageA.pt"
    output_dir = "hyperspectral_autoencoder"
    
    # Print PyTorch version for debugging
    print(f"PyTorch version: {torch.__version__}")
    
    model, losses, output_dir = train_and_visualize_autoencoder(
        model_path=model_path,
        output_dir=output_dir,
        batch_size=10,     # Reduced batch size
        num_epochs=2000,     # Reduced epochs for faster testing
        learning_rate=0.001
    )
