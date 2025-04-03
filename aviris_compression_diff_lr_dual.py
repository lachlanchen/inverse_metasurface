#!/usr/bin/env python3
"""
AVIRIS Compression with Simple Encoder and Selectable Decoder
-----------------------------------------------------------
- Creates tiles from AVIRIS_SIMPLE_SELECT data
- Implements a compression model with linear filter encoder
- Supports two decoder options: AWAN or simple 3-layer CNN
- Adds noise with random SNR from 10-40dB using reparameterization trick
- Visualizes reconstructions and filter evolution
- Tracks train/test MSE metrics
- Uses separate learning rates for encoder and decoder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from pathlib import Path
import random
from datetime import datetime
from AWAN import AWAN

class AvirisDataset(Dataset):
    """Dataset for AVIRIS tiles"""
    def __init__(self, tiles):
        self.tiles = tiles
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        return self.tiles[idx]

class LinearEncoder(nn.Module):
    """Simple linear encoder that multiplies input with filter matrix A"""
    def __init__(self, in_dim=100, out_dim=11):
        super(LinearEncoder, self).__init__()
        self.filter_A = nn.Parameter(torch.randn(out_dim, in_dim))
        # Initialize with values between 0 and 1 as requested
        nn.init.uniform_(self.filter_A, 0., 1.)
    
    def forward(self, x):
        # Input shape: (batch, channels, height, width)
        batch, C, H, W = x.shape
        
        # Reshape to (batch*height*width, channels)
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Apply filter A: Z = X·A^T
        z = torch.matmul(x_flat, self.filter_A.t())
        
        # Reshape back to (batch, out_dim, height, width)
        z = z.reshape(batch, H, W, -1).permute(0, 3, 1, 2)
        
        return z

class SimpleCNNDecoder(nn.Module):
    """Simple 3-layer CNN decoder"""
    def __init__(self, in_channels=11, out_channels=100):
        super(SimpleCNNDecoder, self).__init__()
        
        # Define intermediate channel sizes
        mid_channels = 64
        
        # First layer: in_channels -> mid_channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Second layer: mid_channels -> mid_channels
        self.layer2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Third layer: mid_channels -> out_channels
        self.layer3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Sigmoid to ensure output in [0,1] range
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class CompressionModel(nn.Module):
    """Compression model with linear encoder and selectable decoder"""
    def __init__(self, in_channels=100, latent_dim=11, decoder_type='awan'):
        super(CompressionModel, self).__init__()
        self.encoder = LinearEncoder(in_dim=in_channels, out_dim=latent_dim)
        
        # Select decoder based on type
        if decoder_type.lower() == 'awan':
            self.decoder = AWAN(inplanes=latent_dim, planes=in_channels, channels=8, n_DRBs=2)
        elif decoder_type.lower() == 'cnn':
            self.decoder = SimpleCNNDecoder(in_channels=latent_dim, out_channels=in_channels)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}. Choose 'awan' or 'cnn'.")
        
    def add_noise(self, z, min_snr_db=10, max_snr_db=40):
        """Add random noise with SNR between min_snr_db and max_snr_db"""
        batch_size = z.shape[0]
        # Random SNR for each image in batch
        snr_db = torch.rand(batch_size, 1, 1, 1, device=z.device) * (max_snr_db - min_snr_db) + min_snr_db
        snr = 10 ** (snr_db / 10)
        
        # Calculate signal power
        signal_power = torch.mean(z ** 2, dim=(1, 2, 3), keepdim=True)
        
        # Calculate noise power based on SNR
        noise_power = signal_power / snr
        
        # Generate Gaussian noise (reparameterization trick)
        noise = torch.randn_like(z) * torch.sqrt(noise_power)
        
        # Add noise to signal
        z_noisy = z + noise
        
        return z_noisy
    
    def forward(self, x, add_noise=True, min_snr_db=10, max_snr_db=40):
        # Encode
        z = self.encoder(x)
        
        # Add noise if specified (during training)
        if add_noise:
            z = self.add_noise(z, min_snr_db, max_snr_db)
        
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, z

def create_tiles(data, tile_size=256, overlap=0):
    """Create tiles from a large image"""
    # Check data shape and convert if necessary
    if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
        # Data is in (C, H, W) format, convert to (H, W, C)
        data = data.permute(1, 2, 0)
    
    H, W, C = data.shape
    tiles = []
    
    stride = tile_size - overlap
    for i in range(0, H - tile_size + 1, stride):
        for j in range(0, W - tile_size + 1, stride):
            tile = data[i:i+tile_size, j:j+tile_size, :]
            # Convert to (C, H, W) format for PyTorch
            tile = tile.permute(2, 0, 1)
            tiles.append(tile)
    
    return tiles

def process_and_cache_data(args):
    """Process AVIRIS data and cache tiles"""
    # Define cache directory and file
    cache_dir = args.use_cache
    tile_size = args.tile_size
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache filename includes tile size
    cache_file = os.path.join(cache_dir, f"tiles_{tile_size}.pt")
    
    # Use existing cache if available
    if os.path.exists(cache_file) and not args.force_cache:
        print(f"Using existing cache: {cache_file}")
        return cache_file
    
    # Get input directories
    base_dir = "AVIRIS_SIMPLE_SELECT"
    if args.folder == "all":
        subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    else:
        subfolders = [args.folder]
    
    print(f"Processing {len(subfolders)} folders: {', '.join(subfolders)}")
    
    # Process each subfolder
    all_tiles = []
    for subfolder in subfolders:
        torch_dir = os.path.join(base_dir, subfolder, "torch")
        if not os.path.exists(torch_dir):
            print(f"Skipping {subfolder}: torch directory not found")
            continue
        
        # Load data
        data_file = os.path.join(torch_dir, "aviris_selected.pt")
        if not os.path.exists(data_file):
            print(f"Skipping {subfolder}: data file not found")
            continue
        
        print(f"Loading data from {data_file}")
        data = torch.load(data_file)
        print(f"Data shape: {data.shape}")
        
        # Create tiles
        print(f"Creating {tile_size}x{tile_size} tiles...")
        tiles = create_tiles(data, tile_size=tile_size)
        print(f"Created {len(tiles)} tiles from {subfolder}")
        
        all_tiles.extend(tiles)
    
    # Convert to tensor and save
    all_tiles_tensor = torch.stack(all_tiles)
    print(f"Total tiles: {len(all_tiles)}, Shape: {all_tiles_tensor.shape}")
    
    # Save to cache
    torch.save(all_tiles_tensor, cache_file)
    print(f"Saved tiles to: {cache_file}")
    
    return cache_file

def visualize_filter(filter_A, save_path):
    """Visualize the filter matrix as 11 individual subplots"""
    latent_dim, in_channels = filter_A.shape
    
    # Create a figure with subplots
    fig, axes = plt.subplots(latent_dim, 1, figsize=(12, 2*latent_dim), sharex=True)
    
    # Plot each row of the filter matrix in a separate subplot
    for i in range(latent_dim):
        axes[i].plot(filter_A[i], 'b-')
        axes[i].set_title(f"Filter {i+1}")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel("Value")
    
    # Set common labels
    axes[-1].set_xlabel("Input Channel (0-99)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Also create a combined plot for easy comparison
    plt.figure(figsize=(12, 8))
    
    # Plot each row of the filter matrix as a line
    for i in range(latent_dim):
        plt.plot(filter_A[i], label=f"Filter {i+1}")
    
    plt.title("Filter Matrix Visualization (11×100)")
    plt.xlabel("Input Channel (0-99)")
    plt.ylabel("Filter Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the combined plot
    combined_path = save_path.replace('.png', '_combined.png')
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close()

def visualize_reconstruction(model, data_loader, device, save_path, num_samples=4):
    """Visualize original and reconstructed images"""
    model.eval()
    
    # Get samples from data loader
    x = next(iter(data_loader))[:num_samples].to(device)
    
    # Get reconstructions
    with torch.no_grad():
        x_recon, z = model(x, add_noise=False)
    
    # Move to CPU for visualization
    x = x.cpu()
    x_recon = x_recon.cpu()
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    
    for i in range(num_samples):
        # Select a random channel to visualize
        channel = random.randint(0, x.shape[1]-1)
        
        # Original
        im0 = axes[i, 0].imshow(x[i, channel], cmap='viridis')
        axes[i, 0].set_title(f"Original (Ch {channel})")
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Reconstructed
        im1 = axes[i, 1].imshow(x_recon[i, channel], cmap='viridis')
        axes[i, 1].set_title(f"Reconstructed (Ch {channel})")
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Difference
        diff = torch.abs(x[i, channel] - x_recon[i, channel])
        im2 = axes[i, 2].imshow(diff, cmap='hot')
        mse = torch.mean(diff**2).item()
        axes[i, 2].set_title(f"Difference (MSE: {mse:.6f})")
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_loss_curves(train_losses, test_losses, save_path):
    """Plot training and test loss curves"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def train_model(model, train_loader, test_loader, args):
    """Train the model and save visualizations"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Define separate optimizers for encoder and decoder
    encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=args.encoder_lr)
    decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=args.decoder_lr)
    
    # Define schedulers for both optimizers
    encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        encoder_optimizer, 'min', patience=5, factor=0.5)
    decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        decoder_optimizer, 'min', patience=5, factor=0.5)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Initialize lists to store losses
    train_losses = []
    test_losses = []
    
    # Create directories for visualizations
    filter_dir = os.path.join(args.output_dir, "filter_evolution")
    recon_dir = os.path.join(args.output_dir, "reconstructions")
    os.makedirs(filter_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    
    # Save initial filter visualization
    visualize_filter(model.encoder.filter_A.detach().cpu().numpy(), 
                     os.path.join(filter_dir, "filter_initial.png"))
    
    # Train for the specified number of epochs
    best_test_loss = float('inf')
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch_idx, x in enumerate(pbar):
                x = x.to(device)
                
                # Forward pass
                x_recon, z = model(x, add_noise=True, min_snr_db=args.min_snr, max_snr_db=args.max_snr)
                
                # Calculate loss
                loss = criterion(x_recon, x)
                
                # Backward pass and optimization with separate optimizers
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                
                # Update progress bar
                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": epoch_loss / (batch_idx + 1)})
        
        # Calculate average epoch loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x in test_loader:
                x = x.to(device)
                x_recon, z = model(x, add_noise=False)
                loss = criterion(x_recon, x)
                test_loss += loss.item()
        
        # Calculate average test loss
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # Update schedulers
        encoder_scheduler.step(avg_test_loss)
        decoder_scheduler.step(avg_test_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"Saved new best model with test loss: {best_test_loss:.6f}")
        
        # Visualize filter and reconstruction periodically
        if (epoch + 1) % args.viz_interval == 0 or epoch == 0 or epoch == args.epochs - 1:
            visualize_filter(model.encoder.filter_A.detach().cpu().numpy(), 
                             os.path.join(filter_dir, f"filter_epoch_{epoch+1}.png"))
            
            visualize_reconstruction(model, test_loader, device, 
                                     os.path.join(recon_dir, f"recon_epoch_{epoch+1}.png"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
    print(f"Training complete! Final model saved to: {args.output_dir}")
    
    return model, train_losses, test_losses

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AVIRIS Compression with Linear Encoder and Selectable Decoder')
    
    # Data processing arguments
    parser.add_argument('--tile_size', type=int, default=256, help='Tile size (default: 256)')
    parser.add_argument('--use_cache', type=str, default='cache_simple', help='Cache directory (default: cache_simple)')
    parser.add_argument('-f', '--folder', type=str, default='all', 
                        help='Subfolder of AVIRIS_SIMPLE_SELECT to process (or "all")')
    parser.add_argument('--force_cache', action='store_true', help='Force cache recreation even if it exists')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='awan', choices=['awan', 'cnn'],
                        help='Decoder model to use: awan or cnn (default: awan)')
    parser.add_argument('--latent_dim', type=int, default=11, help='Latent dimension (default: 11)')
    parser.add_argument('--min_snr', type=float, default=10, help='Minimum SNR in dB (default: 10)')
    parser.add_argument('--max_snr', type=float, default=40, help='Maximum SNR in dB (default: 40)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--encoder_lr', type=float, default=1e-3, help='Encoder learning rate (default: 1e-3)')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='Decoder learning rate (default: 1e-4)')
    parser.add_argument('--test_split', type=float, default=0.2, help='Test split ratio (default: 0.2)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results-simple-select', 
                       help='Output directory (default: results-simple-select)')
    parser.add_argument('--viz_interval', type=int, default=5, help='Visualization interval in epochs (default: 5)')
    
    args = parser.parse_args()
    
    # Add datetime and model type to output directory if not explicitly provided
    if args.output_dir == 'results-simple-select':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results-{args.model}-{timestamp}"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments to a file for reference
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Process and cache data
    cache_file = process_and_cache_data(args)
    
    # Load cached data
    print(f"Loading cached tiles from: {cache_file}")
    tiles = torch.load(cache_file)
    print(f"Loaded {tiles.shape[0]} tiles with shape {tiles.shape[1:]} (C×H×W)")
    
    # Create dataset
    dataset = AvirisDataset(tiles)
    
    # Split into train and test sets
    test_size = int(len(dataset) * args.test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"Dataset split: {train_size} training samples, {test_size} test samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    in_channels = tiles.shape[1]  # Number of spectral bands
    model = CompressionModel(in_channels=in_channels, latent_dim=args.latent_dim, decoder_type=args.model)
    print(f"Model initialized with {in_channels} input channels, {args.latent_dim} latent dimensions, and {args.model} decoder")
    
    # Train model
    model, train_losses, test_losses = train_model(model, train_loader, test_loader, args)
    
    # Plot loss curves
    plot_loss_curves(train_losses, test_losses, os.path.join(args.output_dir, "loss_curves.png"))
    
    print("\nTraining complete!")
    print(f"Results saved to: {args.output_dir}")
    print("- Best and final models saved")
    print("- Filter evolution visualizations")
    print("- Reconstruction visualizations")
    print("- Loss curves")

if __name__ == "__main__":
    main()