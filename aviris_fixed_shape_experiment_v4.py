#!/usr/bin/env python3
"""
AVIRIS Fixed Shape Experiment (v4)
----------------------------
This script extends the original AVIRIS compression model to:
1. Record important shapes from Stage 1 (initial, lowest condition number, lowest test MSE)
2. Run Stage 2 with fixed shapes from Stage 1, optimizing only the decoder
3. Compare performance of different fixed shapes
4. Includes filter evolution visualization from the original compression pipeline

Usage:
    # Run both stages
    python aviris_fixed_shape_experiment_v4.py --use_fsf --model awan --tile_size 100 --epochs 100 
    --stage2_epochs 100 --batch_size 64 --encoder_lr 1e-3 --decoder_lr 5e-4 --min_snr 10 --max_snr 40 
    --shape2filter_path "outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt" 
    --filter2shape_path "outputs_three_stage_20250322_145925/stageC/spec2shape_stageC.pt" 
    --filter_scale_factor 10.0
    
    # Skip stage 1 and only run stage 2
    python aviris_fixed_shape_experiment_v4.py --use_fsf --model awan --tile_size 100 
    --stage2_epochs 100 --batch_size 64 --decoder_lr 5e-4 --min_snr 10 --max_snr 40 
    --shape2filter_path "outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt" 
    --skip_stage1 --load_shapes_dir results_fixed_shape_awan_20250402_052223/recorded_shapes/ 
    --filter_scale_factor 10.0
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
import random
from datetime import datetime
import numpy.linalg as LA

# Import filter visualization functions from aviris_compression_diff_lr_dual_fsf_corr
try:
    from aviris_compression_diff_lr_dual_fsf_corr import (
        plot_shape_with_c4, 
        visualize_filter,
        visualize_filter_with_shape,
        calculate_condition_number
    )
    print("Successfully imported filter visualization functions")
except ImportError:
    # Define the functions here if import fails
    def plot_shape_with_c4(shape, title, save_path=None, show=False, ax=None):
        """Plot shape with C4 symmetry replication in a minimal academic style"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        
        ax.set_xlim(-0.7, 0.7)  # Fixed limits as requested
        ax.set_ylim(-0.7, 0.7)
        
        # Extract active points
        presence = shape[:, 0] > 0.5
        active_points = shape[presence, 1:3]
        
        # Plot original Q1 points
        ax.scatter(shape[presence, 1], shape[presence, 2], color='red', s=50)
        
        # Apply C4 symmetry and plot the polygon
        if len(active_points) > 0:
            c4_points = replicate_c4(active_points)
            sorted_points = sort_points_by_angle(c4_points)
            
            # If we have enough points for a polygon
            if len(sorted_points) >= 3:
                # Close the polygon
                polygon = np.vstack([sorted_points, sorted_points[0]])
                ax.plot(polygon[:, 0], polygon[:, 1], 'k-', linewidth=1.5)
                ax.fill(polygon[:, 0], polygon[:, 1], 'lightblue', alpha=0.5)
            else:
                # Just plot the points
                ax.scatter(c4_points[:, 0], c4_points[:, 1], color='blue', alpha=0.4, s=30)
        
        ax.set_title(title, fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True)
        
        if save_path and ax is None:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show and ax is None:
            plt.show()
        elif ax is None:
            plt.close()
        
        return ax
    
    def calculate_condition_number(filters):
        """
        Calculate condition number of the spectral filters matrix.
        
        Parameters:
        filters: Tensor or ndarray of shape [11, 100] representing the spectral filters
        
        Returns:
        float: Condition number
        """
        # Convert to numpy for condition number calculation
        if isinstance(filters, torch.Tensor):
            filters_np = filters.detach().cpu().numpy()
        else:
            filters_np = filters
        
        # Use singular value decomposition to calculate condition number
        u, s, vh = LA.svd(filters_np)
        
        # Condition number is the ratio of largest to smallest singular value
        # Add small epsilon to prevent division by zero
        condition_number = s[0] / (s[-1] + 1e-10)
        
        return condition_number
    
    def visualize_filter_with_shape(filter_A, shape_pred, filter_output, save_path):
        """
        Visualize the filter matrix, its corresponding shape, and reconstructed filter
        
        Parameters:
        filter_A: Original filter parameter (numpy array or tensor)
        shape_pred: Predicted shape from filter2shape (numpy array or tensor)
        filter_output: Reconstructed filter from shape2filter (numpy array or tensor)
        save_path: Path to save the visualization
        """
        # Convert to numpy if needed
        if isinstance(filter_A, torch.Tensor):
            filter_A_np = filter_A.detach().cpu().numpy()
        else:
            filter_A_np = filter_A
            
        if isinstance(shape_pred, torch.Tensor):
            shape_pred_np = shape_pred.detach().cpu().numpy()
        else:
            shape_pred_np = shape_pred
            
        if isinstance(filter_output, torch.Tensor):
            filter_output_np = filter_output.detach().cpu().numpy()
        else:
            filter_output_np = filter_output
        
        # Calculate condition numbers
        filter_cond = calculate_condition_number(filter_A_np)
        recon_cond = calculate_condition_number(filter_output_np)
        
        # Create a 2x2 grid
        fig = plt.figure(figsize=(18, 10))
        
        # Plot original filter (top left)
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        for i in range(filter_A_np.shape[0]):
            ax1.plot(filter_A_np[i], label=f"Filter {i+1}" if i % 3 == 0 else None)
        ax1.set_title(f"Original Filter (Condition Number: {filter_cond:.4f})")
        ax1.set_xlabel("Wavelength Index")
        ax1.set_ylabel("Filter Value")
        ax1.grid(True, alpha=0.3)
        if filter_A_np.shape[0] <= 11:  # Only show legend for small number of filters
            ax1.legend()
        
        # Plot the shape (top right)
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        plot_shape_with_c4(shape_pred_np, "Predicted Shape", show=False, ax=ax2)
        
        # Plot reconstructed filter (bottom left)
        ax3 = plt.subplot2grid((2, 2), (1, 0))
        for i in range(filter_output_np.shape[0]):
            ax3.plot(filter_output_np[i], label=f"Filter {i+1}" if i % 3 == 0 else None)
        ax3.set_title(f"Reconstructed Filter (Condition Number: {recon_cond:.4f})")
        ax3.set_xlabel("Wavelength Index")
        ax3.set_ylabel("Filter Value")
        ax3.grid(True, alpha=0.3)
        if filter_output_np.shape[0] <= 11:  # Only show legend for small number of filters
            ax3.legend()
        
        # Plot the difference (bottom right)
        ax4 = plt.subplot2grid((2, 2), (1, 1))
        diff = np.abs(filter_A_np - filter_output_np)
        for i in range(diff.shape[0]):
            ax4.plot(diff[i], label=f"Filter {i+1}" if i % 3 == 0 else None)
        mse = np.mean(diff**2)
        ax4.set_title(f"Difference (MSE: {mse:.6f})")
        ax4.set_xlabel("Wavelength Index")
        ax4.set_ylabel("Absolute Difference")
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
    def visualize_filter(filter_A, save_path, include_shape=False, shape_pred=None, filter_output=None):
        """Visualize the filter matrix as individual subplots"""
        # Ensure filter_A is numpy if it's a tensor
        if isinstance(filter_A, torch.Tensor):
            filter_A_np = filter_A.detach().cpu().numpy()
        else:
            filter_A_np = filter_A
            
        latent_dim, in_channels = filter_A_np.shape
        
        # Create a figure with subplots
        fig, axes = plt.subplots(latent_dim, 1, figsize=(12, 2*latent_dim), sharex=True)
        
        # Handle the case where latent_dim=1
        axes = [axes] if latent_dim == 1 else axes
        
        # Plot each row of the filter matrix in a separate subplot
        for i in range(latent_dim):
            axes[i].plot(filter_A_np[i], 'b-')
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
            plt.plot(filter_A_np[i], label=f"Filter {i+1}")
        
        plt.title("Filter Matrix Visualization")
        plt.xlabel("Input Channel (0-99)")
        plt.ylabel("Filter Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the combined plot
        combined_path = save_path.replace('.png', '_combined.png')
        plt.savefig(combined_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        # If shape_pred and filter_output are provided, also visualize the filter2shape2filter results
        if include_shape and shape_pred is not None and filter_output is not None:
            fsf_path = save_path.replace('.png', '_with_shape.png')
            visualize_filter_with_shape(filter_A_np, shape_pred, filter_output, fsf_path)
    
    print("Defined local filter visualization functions")

# Import AWAN if you have it
try:
    from AWAN import AWAN
except ImportError:
    print("Warning: Could not import AWAN, only CNN decoder will be available.")

# Import filter2shape2filter models and utilities
try:
    from filter2shape2filter_pipeline import (
        Shape2FilterModel, Filter2ShapeVarLen, create_pipeline, load_models,
        replicate_c4, sort_points_by_angle
    )
except ImportError:
    print("Warning: Could not import filter2shape2filter_pipeline.")

# Set random seed for reproducibility
def set_seed(seed=42):
    """Set random seed for reproducibility across Python, NumPy, PyTorch and CUDA"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")

# Set seed at the beginning of your script
set_seed(42)

def save_data_to_csv(data, headers, save_path):
    """
    Save data to CSV file
    
    Parameters:
    data: List of lists or numpy array with data to save
    headers: List of headers for each column
    save_path: Path to save the CSV file
    """
    import csv
    import os
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # Convert data to list format if it's a numpy array
    if isinstance(data, np.ndarray):
        # If it's a 1D array, convert to 2D column
        if len(data.shape) == 1:
            data = np.column_stack([np.arange(len(data)), data])
        data_list = data.tolist()
    else:
        data_list = data
        
    # Write data to CSV
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data_list)
    
    print(f"Saved data to CSV: {save_path}")

class AvirisDataset(Dataset):
    """Dataset for AVIRIS tiles"""
    def __init__(self, tiles):
        self.tiles = tiles
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        return self.tiles[idx]

class LinearEncoder(nn.Module):
    """Linear encoder that multiplies input with filter matrix A,
       integrating the filter2shape2filter pipeline"""
    def __init__(self, in_dim=100, out_dim=11, use_fsf=True, 
                 shape2filter_path=None, filter2shape_path=None, 
                 filter_scale_factor=50.0):
        super(LinearEncoder, self).__init__()
        self.filter_H = nn.Parameter(torch.randn(out_dim, in_dim))
        # Initialize with values between 0 and 1
        nn.init.uniform_(self.filter_H, 0., 1.)
        
        self.use_fsf = use_fsf
        self.filter_scale_factor = filter_scale_factor
        self.pipeline = None
        self.current_shape = None
        self.filter_output = None
        
        # Initialize the filter2shape2filter pipeline if requested
        if use_fsf and shape2filter_path and filter2shape_path:
            try:
                device = torch.device("cpu")  # Will be moved to the right device later
                self.shape2filter, self.filter2shape = load_models(
                    shape2filter_path, filter2shape_path, device)
                self.pipeline = create_pipeline(self.shape2filter, self.filter2shape, no_grad_frozen=False)
                print("FSF pipeline initialized in LinearEncoder")
            except Exception as e:
                print(f"Error initializing FSF pipeline: {e}")
                self.use_fsf = False
                self.pipeline = None
        else:
            self.use_fsf = False
            self.pipeline = None

    @property
    def filter_A(self):
        # Get normalized filter through pipeline if available
        if self.use_fsf and self.pipeline is not None:
            _, filter_norm = self.pipeline(self.filter_H.unsqueeze(0))
            return filter_norm[0]
        return self.filter_H
    
    def to(self, device):
        # Override to method to move pipeline models to the same device
        super().to(device)
        if self.pipeline is not None:
            self.shape2filter.to(device)
            self.filter2shape.to(device)
            # Recreate pipeline with models on the correct device
            self.pipeline = create_pipeline(self.shape2filter, self.filter2shape, no_grad_frozen=False)
        return self
    
    def forward(self, x):
        # Input shape: (batch, channels, height, width)
        batch, C, H, W = x.shape
        
        # Reshape to (batch*height*width, channels)
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        
        if self.use_fsf and self.pipeline is not None:
            # Run filter through pipeline to get shape and reconstructed filter
            shape_pred, filter_output = self.pipeline(self.filter_A.unsqueeze(0))
            
            # Store for visualization
            self.current_shape = shape_pred[0].detach().cpu()
            self.filter_output = filter_output[0].detach().cpu()
            
            # Use the filter output from the pipeline, scaled by the factor
            z = torch.matmul(x_flat, filter_output[0].t() / self.filter_scale_factor)
        else:
            # Use the learnable filter directly
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
    def __init__(self, in_channels=100, latent_dim=11, decoder_type='awan', 
                 use_fsf=True, shape2filter_path=None, filter2shape_path=None,
                 filter_scale_factor=50.0):
        super(CompressionModel, self).__init__()
        
        self.encoder = LinearEncoder(
            in_dim=in_channels, 
            out_dim=latent_dim,
            use_fsf=use_fsf,
            shape2filter_path=shape2filter_path,
            filter2shape_path=filter2shape_path,
            filter_scale_factor=filter_scale_factor
        )
        
        # Select decoder based on type
        if decoder_type.lower() == 'awan':
            self.decoder = AWAN(inplanes=latent_dim, planes=in_channels, channels=128, n_DRBs=2)
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

class FixedShapeEncoder(nn.Module):
    """Simple fixed shape encoder that directly uses precomputed filters"""
    def __init__(self, shape, in_dim=100, filter_scale_factor=50.0, device=None):
        super(FixedShapeEncoder, self).__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.filter_scale_factor = filter_scale_factor
        self.shape = shape.to(device)
        
        # Load shape2filter model
        try:
            self.shape2filter = Shape2FilterModel().to(device)
            self.shape2filter.eval()  # Set to evaluation mode
            
            # Precompute filter from shape, store as buffer (not parameter)
            with torch.no_grad():
                self.register_buffer(
                    'fixed_filter', 
                    self.shape2filter(self.shape.unsqueeze(0))[0]
                )
            print(f"Fixed shape encoder initialized with filter of shape {self.fixed_filter.shape}")
        except Exception as e:
            print(f"Error initializing shape2filter model: {e}")
            raise
    
    def forward(self, x):
        # Input shape: (batch, channels, height, width)
        batch, C, H, W = x.shape
        
        # Reshape to (batch*height*width, channels)
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Use the fixed filter
        z = torch.matmul(x_flat, self.fixed_filter.t() / self.filter_scale_factor)
        
        # Reshape back to (batch, out_dim, height, width)
        z = z.reshape(batch, H, W, -1).permute(0, 3, 1, 2)
        
        return z

class FixedShapeModel(nn.Module):
    """Model with fixed shape encoder and trainable decoder"""
    def __init__(self, shape, in_channels=100, decoder_type='awan', filter_scale_factor=50.0, device=None):
        super(FixedShapeModel, self).__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create encoder with fixed filter
        self.encoder = FixedShapeEncoder(
            shape=shape,
            in_dim=in_channels,
            filter_scale_factor=filter_scale_factor,
            device=device
        )
        
        # Number of latent dimensions equals number of rows in filter
        latent_dim = self.encoder.fixed_filter.shape[0]
        
        # Create decoder based on type
        if decoder_type.lower() == 'awan':
            self.decoder = AWAN(inplanes=latent_dim, planes=in_channels, channels=128, n_DRBs=2)
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
        
        # Generate Gaussian noise
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

def visualize_shape(shape, save_path):
    """Visualize shape with C4 symmetry replication and save to file"""
    # Convert shape to numpy if it's a tensor
    if isinstance(shape, torch.Tensor):
        shape_np = shape.detach().cpu().numpy()
    else:
        shape_np = shape
    
    # Create figure
    plt.figure(figsize=(5, 5))
    plt.xlim(-0.7, 0.7)  # Fixed limits
    plt.ylim(-0.7, 0.7)
    
    # Extract active points (where presence > 0.5)
    presence = shape_np[:, 0] > 0.5
    active_points = shape_np[presence, 1:3]
    
    # Plot original points
    plt.scatter(shape_np[presence, 1], shape_np[presence, 2], color='red', s=50)
    
    # Apply C4 symmetry and plot
    if len(active_points) > 0:
        # Replicate with C4 symmetry
        c4_points = []
        for i in range(len(active_points)):
            x, y = active_points[i]
            c4_points.append([x, y])       # Q1: original
            c4_points.append([-y, x])      # Q2: rotate 90°
            c4_points.append([-x, -y])     # Q3: rotate 180°
            c4_points.append([y, -x])      # Q4: rotate 270°
        
        c4_points = np.array(c4_points)
        
        # Sort points by angle for polygon drawing
        if len(c4_points) >= 3:
            center = np.mean(c4_points, axis=0)
            angles = np.arctan2(c4_points[:, 1] - center[1], c4_points[:, 0] - center[0])
            idx = np.argsort(angles)
            sorted_points = c4_points[idx]
            
            # Close the polygon
            polygon = np.vstack([sorted_points, sorted_points[0]])
            plt.plot(polygon[:, 0], polygon[:, 1], 'k-', linewidth=1.5)
            plt.fill(polygon[:, 0], polygon[:, 1], 'lightblue', alpha=0.5)
        else:
            # Just plot the points
            plt.scatter(c4_points[:, 0], c4_points[:, 1], color='blue', alpha=0.4, s=30)
    
    # Format plot
    plt.title('Shape Visualization with C4 Replication')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved shape visualization to: {save_path}")
    plt.close()

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

def visualize_reconstruction(model, data_loader, device, save_path, num_samples=4):
    """Visualize original and reconstructed images with consistent colorbar scaling"""
    model.eval()
    
    # Get samples from data loader
    x = next(iter(data_loader))[:num_samples].to(device)
    
    # Get reconstructions
    with torch.no_grad():
        x_recon, z = model(x, add_noise=False)
    
    # Move to CPU for visualization
    x = x.cpu()
    x_recon = x_recon.cpu()
    
    # Use fixed channels for consistency instead of random channels
    channels = []
    for i in range(num_samples):
        # Use evenly spaced channels
        channel_idx = (i * (x.shape[1] // num_samples)) % x.shape[1]
        channels.append(channel_idx)
    
    # Find global min and max for consistent colorbar scaling
    global_min = float('inf')
    global_max = float('-inf')
    
    for i in range(num_samples):
        channel = channels[i]
        global_min = min(global_min, x[i, channel].min().item(), x_recon[i, channel].min().item())
        global_max = max(global_max, x[i, channel].max().item(), x_recon[i, channel].max().item())
    
    # Also find global min/max for difference images
    diff_min = float('inf')
    diff_max = float('-inf')
    
    for i in range(num_samples):
        channel = channels[i]
        diff = torch.abs(x[i, channel] - x_recon[i, channel])
        diff_min = min(diff_min, diff.min().item())
        diff_max = max(diff_max, diff.max().item())
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    
    for i in range(num_samples):
        channel = channels[i]
        
        # Original
        im0 = axes[i, 0].imshow(x[i, channel], cmap='viridis', vmin=global_min, vmax=global_max)
        axes[i, 0].set_title(f"Original (Ch {channel})")
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Reconstructed
        im1 = axes[i, 1].imshow(x_recon[i, channel], cmap='viridis', vmin=global_min, vmax=global_max)
        axes[i, 1].set_title(f"Reconstructed (Ch {channel})")
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Difference - use consistent scaling across all difference images
        diff = torch.abs(x[i, channel] - x_recon[i, channel])
        im2 = axes[i, 2].imshow(diff, cmap='hot', vmin=diff_min, vmax=diff_max)
        mse = torch.mean(diff**2).item()
        axes[i, 2].set_title(f"Difference (MSE: {mse:.6f})")
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved reconstruction visualization to: {save_path}")
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
    print(f"Saved loss curves to: {save_path}")
    plt.close()

def train_model_stage1(model, train_loader, test_loader, args):
    """Train model in stage 1 and record key shapes"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Define separate optimizers for encoder and decoder
    encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=args.encoder_lr)
    decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=args.decoder_lr)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Initialize lists to store losses and condition numbers
    train_losses = []
    test_losses = []
    condition_numbers = []
    
    # Create directories for visualizations and recorded shapes
    filter_dir = os.path.join(args.output_dir, "filter_evolution")
    recon_dir = os.path.join(args.output_dir, "reconstructions")
    shapes_dir = os.path.join(args.output_dir, "recorded_shapes")
    csv_dir = os.path.join(args.output_dir, "csv_data")
    
    os.makedirs(filter_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(shapes_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    print(f"Created output directories:\n- {filter_dir}\n- {recon_dir}\n- {shapes_dir}\n- {csv_dir}")
    
    # Dictionary to store recorded shapes
    recorded_shapes = {}
    recorded_metrics = {
        'initial': {'condition_number': float('inf'), 'test_mse': float('inf')},
        'lowest_condition_number': {'condition_number': float('inf'), 'test_mse': float('inf')},
        'lowest_test_mse': {'condition_number': float('inf'), 'test_mse': float('inf')},
        'final': {'condition_number': float('inf'), 'test_mse': float('inf')}
    }
    
    # Run a dummy forward pass to initialize shape
    if args.use_fsf and model.encoder.pipeline is not None:
        dummy_input = next(iter(train_loader))[:1].to(device)
        with torch.no_grad():
            model.encoder(dummy_input)
        
        # Record initial shape
        initial_shape = model.encoder.current_shape.clone()
        initial_filter_output = model.encoder.filter_output.clone()
        recorded_shapes['initial'] = initial_shape
        
        # Calculate condition number
        condition_number = calculate_condition_number(initial_filter_output)
        condition_numbers.append(condition_number)
        recorded_metrics['initial']['condition_number'] = condition_number
        
        # Save initial shape
        np_save_path = os.path.join(shapes_dir, "initial_shape.npy")
        np.save(np_save_path, initial_shape.detach().cpu().numpy())
        print(f"Saved initial shape to: {np_save_path}")
        
        # Save visualization of initial shape
        viz_save_path = os.path.join(shapes_dir, "initial_shape.png")
        visualize_shape(initial_shape, viz_save_path)
        
        # Save visualization of initial filter
        filter_viz_path = os.path.join(filter_dir, "filter_initial.png")
        visualize_filter(
            model.encoder.filter_A.detach().cpu(),
            filter_viz_path,
            include_shape=True,
            shape_pred=model.encoder.current_shape,
            filter_output=model.encoder.filter_output
        )
        print(f"Saved initial filter visualization to: {filter_viz_path}")
        
        print(f"Recorded initial shape with condition number: {condition_number:.4f}")
    
    # Train for the specified number of epochs
    best_test_loss = float('inf')
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Stage 1 Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch_idx, x in enumerate(pbar):
                x = x.to(device)
                
                # Forward pass
                x_recon, z = model(x, add_noise=True, min_snr_db=args.min_snr, max_snr_db=args.max_snr)
                
                # Calculate loss
                loss = criterion(x_recon, x)
                
                # Backward pass and optimization
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
        
        # Get updated shape and filter output
        if args.use_fsf and model.encoder.pipeline is not None:
            with torch.no_grad():
                dummy_input = next(iter(train_loader))[:1].to(device)
                model.encoder(dummy_input)
            
            current_shape = model.encoder.current_shape.clone()
            current_filter_output = model.encoder.filter_output.clone()
            
            # Calculate condition number
            current_condition_number = calculate_condition_number(current_filter_output)
            condition_numbers.append(current_condition_number)
            
            # Visualize filter periodically
            if (epoch + 1) % args.viz_interval == 0 or epoch == args.epochs - 1:
                filter_viz_path = os.path.join(filter_dir, f"filter_epoch_{epoch+1}.png")
                visualize_filter(
                    model.encoder.filter_A.detach().cpu(),
                    filter_viz_path,
                    include_shape=True,
                    shape_pred=model.encoder.current_shape,
                    filter_output=model.encoder.filter_output
                )
                print(f"Saved filter visualization to: {filter_viz_path}")
            
            # Check for lowest condition number
            if current_condition_number < recorded_metrics['lowest_condition_number']['condition_number']:
                recorded_shapes['lowest_condition_number'] = current_shape
                recorded_metrics['lowest_condition_number']['condition_number'] = current_condition_number
                recorded_metrics['lowest_condition_number']['test_mse'] = avg_test_loss
                
                # Save shape
                np_save_path = os.path.join(shapes_dir, "lowest_condition_number_shape.npy")
                np.save(np_save_path, current_shape.detach().cpu().numpy())
                print(f"Saved lowest condition number shape to: {np_save_path}")
                
                # Save visualization
                viz_save_path = os.path.join(shapes_dir, "lowest_condition_number_shape.png")
                visualize_shape(current_shape, viz_save_path)
                
                # Save filter visualization
                filter_viz_path = os.path.join(filter_dir, "lowest_condition_number_filter.png")
                visualize_filter(
                    model.encoder.filter_A.detach().cpu(),
                    filter_viz_path,
                    include_shape=True,
                    shape_pred=current_shape,
                    filter_output=current_filter_output
                )
                
                print(f"New lowest condition number: {current_condition_number:.4f}")
            
            # Check for lowest test MSE
            if avg_test_loss < recorded_metrics['lowest_test_mse']['test_mse']:
                recorded_shapes['lowest_test_mse'] = current_shape
                recorded_metrics['lowest_test_mse']['condition_number'] = current_condition_number
                recorded_metrics['lowest_test_mse']['test_mse'] = avg_test_loss
                
                # Save shape
                np_save_path = os.path.join(shapes_dir, "lowest_test_mse_shape.npy")
                np.save(np_save_path, current_shape.detach().cpu().numpy())
                print(f"Saved lowest test MSE shape to: {np_save_path}")
                
                # Save visualization
                viz_save_path = os.path.join(shapes_dir, "lowest_test_mse_shape.png")
                visualize_shape(current_shape, viz_save_path)
                
                # Save filter visualization
                filter_viz_path = os.path.join(filter_dir, "lowest_test_mse_filter.png")
                visualize_filter(
                    model.encoder.filter_A.detach().cpu(),
                    filter_viz_path,
                    include_shape=True,
                    shape_pred=current_shape,
                    filter_output=current_filter_output
                )
                
                print(f"New lowest test MSE: {avg_test_loss:.6f}")
        
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            model_save_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model with test loss: {best_test_loss:.6f} to: {model_save_path}")
        
        # Visualize reconstruction periodically
        if (epoch + 1) % args.viz_interval == 0:
            recon_save_path = os.path.join(recon_dir, f"recon_epoch_{epoch+1}.png")
            visualize_reconstruction(model, test_loader, device, recon_save_path)
    
    # Record final shape
    if args.use_fsf and model.encoder.pipeline is not None:
        final_shape = model.encoder.current_shape.clone()
        final_filter_output = model.encoder.filter_output.clone()
        recorded_shapes['final'] = final_shape
        
        # Calculate condition number
        final_condition_number = calculate_condition_number(final_filter_output)
        recorded_metrics['final']['condition_number'] = final_condition_number
        recorded_metrics['final']['test_mse'] = avg_test_loss
        
        # Save final shape
        np_save_path = os.path.join(shapes_dir, "final_shape.npy")
        np.save(np_save_path, final_shape.detach().cpu().numpy())
        print(f"Saved final shape to: {np_save_path}")
        
        # Save visualization
        viz_save_path = os.path.join(shapes_dir, "final_shape.png")
        visualize_shape(final_shape, viz_save_path)
        
        # Save final filter visualization
        filter_viz_path = os.path.join(filter_dir, "final_filter.png")
        visualize_filter(
            model.encoder.filter_A.detach().cpu(),
            filter_viz_path,
            include_shape=True,
            shape_pred=final_shape,
            filter_output=final_filter_output
        )
        
        print(f"Recorded final shape with condition number: {final_condition_number:.4f}")
    
    # Save metrics for all recorded shapes
    metrics_save_path = os.path.join(shapes_dir, "shape_metrics.txt")
    with open(metrics_save_path, 'w') as f:
        for shape_name, metrics in recorded_metrics.items():
            if shape_name in recorded_shapes:
                f.write(f"{shape_name} shape:\n")
                f.write(f"  Condition Number: {metrics['condition_number']:.4f}\n")
                f.write(f"  Test MSE: {metrics['test_mse']:.6f}\n\n")
    print(f"Saved shape metrics to: {metrics_save_path}")
    
    # Save condition numbers
    condition_path = os.path.join(args.output_dir, "condition_numbers.npy")
    np.save(condition_path, np.array(condition_numbers))
    print(f"Saved condition numbers to: {condition_path}")
    
    # Plot condition numbers
    if condition_numbers:
        # Linear scale plot
        plt.figure(figsize=(10, 6))
        plt.plot(condition_numbers, 'b-')
        plt.title('Filter Condition Number Evolution')
        plt.xlabel('Epochs')
        plt.ylabel('Condition Number')
        plt.grid(True, alpha=0.3)
        cond_plot_path = os.path.join(args.output_dir, "condition_number_evolution.png")
        plt.savefig(cond_plot_path, dpi=300)
        print(f"Saved condition number plot to: {cond_plot_path}")
        plt.close()

        # Log scale plot
        plt.figure(figsize=(10, 6))
        plt.semilogy(condition_numbers, 'r-')
        plt.title('Filter Condition Number Evolution (Log Scale)')
        plt.xlabel('Epochs')
        plt.ylabel('Condition Number (log scale)')
        plt.grid(True, alpha=0.3)
        cond_log_plot_path = os.path.join(args.output_dir, "condition_number_evolution_log.png")
        plt.savefig(cond_log_plot_path, dpi=300)
        print(f"Saved log-scale condition number plot to: {cond_log_plot_path}")
        plt.close()
    
    # Save train and test losses as CSV
    stage1_csv_path = os.path.join(csv_dir, "stage1_losses.csv")
    loss_data = np.column_stack((
        np.arange(1, len(train_losses) + 1),  # Epoch numbers
        np.array(train_losses),               # Train losses
        np.array(test_losses)                 # Test losses
    ))
    save_data_to_csv(loss_data, ["Epoch", "Train_Loss", "Test_Loss"], stage1_csv_path)
    
    # Save condition numbers as CSV
    condition_csv_path = os.path.join(csv_dir, "condition_numbers.csv")
    condition_data = np.column_stack((
        np.arange(len(condition_numbers)),    # Iteration/epoch numbers
        np.array(condition_numbers)           # Condition numbers
    ))
    save_data_to_csv(condition_data, ["Iteration", "Condition_Number"], condition_csv_path)
    
    # Save shape metrics as CSV
    metrics_csv_path = os.path.join(csv_dir, "shape_metrics.csv")
    metrics_data = []
    headers = ["Shape_Type", "Condition_Number", "Test_MSE"]
    
    for shape_name, metrics in recorded_metrics.items():
        if shape_name in recorded_shapes:
            metrics_data.append([
                shape_name,
                metrics['condition_number'],
                metrics['test_mse']
            ])
    
    save_data_to_csv(metrics_data, headers, metrics_csv_path)
    
    return recorded_shapes, recorded_metrics, train_losses, test_losses

def train_with_fixed_shape(shape_name, shape, train_loader, test_loader, args):
    """Train model with fixed shape, optimizing only the decoder"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory for this shape
    shape_dir = os.path.join(args.output_dir, f"stage2_{shape_name}")
    recon_dir = os.path.join(shape_dir, "reconstructions")
    csv_dir = os.path.join(shape_dir, "csv_data")
    os.makedirs(shape_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    print(f"Created output directory for {shape_name} shape: {shape_dir}")
    
    # Get input dimensions from data
    in_channels = next(iter(train_loader)).shape[1]
    
    # Create model with fixed shape
    model = FixedShapeModel(
        shape=shape,
        in_channels=in_channels,
        decoder_type=args.model,
        filter_scale_factor=args.filter_scale_factor,
        device=device
    )
    
    # Move model to device
    model = model.to(device)
    
    # Only optimize decoder parameters
    optimizer = optim.Adam(model.decoder.parameters(), lr=args.decoder_lr)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Initialize lists to store losses
    train_losses = []
    test_losses = []
    
    # Train for specified number of epochs
    best_test_loss = float('inf')
    print(f"\nTraining Stage 2 model with {shape_name} shape...")
    
    # First let's visualize the initial reconstruction before training
    recon_path = os.path.join(recon_dir, "recon_epoch_0.png")
    visualize_reconstruction(model, test_loader, device, recon_path)
    print(f"Saved initial reconstruction to: {recon_path}")
    
    # Evaluate initial loss
    model.eval()
    with torch.no_grad():
        initial_train_loss = 0
        for x in train_loader:
            x = x.to(device)
            x_recon, _ = model(x, add_noise=False)
            loss = criterion(x_recon, x)
            initial_train_loss += loss.item()
        initial_train_loss /= len(train_loader)
        
        initial_test_loss = 0
        for x in test_loader:
            x = x.to(device)
            x_recon, _ = model(x, add_noise=False)
            loss = criterion(x_recon, x)
            initial_test_loss += loss.item()
        initial_test_loss /= len(test_loader)
    
    train_losses.append(initial_train_loss)
    test_losses.append(initial_test_loss)
    print(f"Initial train loss: {initial_train_loss:.6f}, test loss: {initial_test_loss:.6f}")
    
    for epoch in range(args.stage2_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Stage 2 [{shape_name}] Epoch {epoch+1}/{args.stage2_epochs}") as pbar:
            for batch_idx, x in enumerate(pbar):
                x = x.to(device)
                
                # Forward pass
                x_recon, z = model(x, add_noise=True, min_snr_db=args.min_snr, max_snr_db=args.max_snr)
                
                # Calculate loss
                loss = criterion(x_recon, x)
                
                # Backward pass and optimize decoder only
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
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
        
        print(f"[{shape_name}] Epoch {epoch+1}/{args.stage2_epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            model_path = os.path.join(shape_dir, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model with test loss: {best_test_loss:.6f} to: {model_path}")
            
            # Save best reconstruction visualization
            best_recon_path = os.path.join(shape_dir, "best_reconstruction.png")
            visualize_reconstruction(model, test_loader, device, best_recon_path)
            print(f"Saved best reconstruction visualization to: {best_recon_path}")
        
        # Visualize reconstruction periodically
        if (epoch + 1) % args.viz_interval == 0 or epoch == args.stage2_epochs - 1:
            recon_path = os.path.join(recon_dir, f"recon_epoch_{epoch+1}.png")
            visualize_reconstruction(model, test_loader, device, recon_path)
            print(f"Saved reconstruction for epoch {epoch+1} to: {recon_path}")
    
    # Save loss values and plots
    loss_path = os.path.join(shape_dir, "loss_values.npz")
    np.savez(loss_path, train_losses=np.array(train_losses), test_losses=np.array(test_losses))
    print(f"Saved loss values to: {loss_path}")
    
    # Save loss values as CSV
    loss_csv_path = os.path.join(csv_dir, "losses.csv")
    loss_data = np.column_stack((
        np.arange(len(train_losses)),  # Epoch numbers (starting at 0 for initial evaluation)
        np.array(train_losses),        # Train losses 
        np.array(test_losses)          # Test losses
    ))
    save_data_to_csv(loss_data, ["Epoch", "Train_Loss", "Test_Loss"], loss_csv_path)
    print(f"Saved loss values to CSV: {loss_csv_path}")
    
    # Plot loss curves
    plot_path = os.path.join(shape_dir, "loss_curves.png")
    plot_loss_curves(train_losses, test_losses, plot_path)
    print(f"Saved loss curves to: {plot_path}")
    
    # Save final model
    final_model_path = os.path.join(shape_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to: {final_model_path}")
    
    # Save summary of training results
    summary_path = os.path.join(shape_dir, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Training summary for {shape_name} shape\n")
        f.write("="*40 + "\n\n")
        f.write(f"Total epochs: {args.stage2_epochs}\n")
        f.write(f"Initial train loss: {train_losses[0]:.6f}\n")
        f.write(f"Initial test loss: {test_losses[0]:.6f}\n")
        f.write(f"Final train loss: {train_losses[-1]:.6f}\n")
        f.write(f"Final test loss: {test_losses[-1]:.6f}\n")
        f.write(f"Best test loss: {best_test_loss:.6f}\n")
        f.write(f"Train loss improvement: {(1 - train_losses[-1]/train_losses[0])*100:.2f}%\n")
        f.write(f"Test loss improvement: {(1 - test_losses[-1]/test_losses[0])*100:.2f}%\n")
    print(f"Saved training summary to: {summary_path}")
    
    # Save training summary as CSV
    summary_csv_path = os.path.join(csv_dir, "training_summary.csv")
    summary_data = [
        ["Shape_Type", shape_name],
        ["Total_Epochs", args.stage2_epochs],
        ["Initial_Train_Loss", train_losses[0]],
        ["Initial_Test_Loss", test_losses[0]],
        ["Final_Train_Loss", train_losses[-1]],
        ["Final_Test_Loss", test_losses[-1]],
        ["Best_Test_Loss", best_test_loss],
        ["Train_Improvement_Pct", (1 - train_losses[-1]/train_losses[0])*100],
        ["Test_Improvement_Pct", (1 - test_losses[-1]/test_losses[0])*100]
    ]
    save_data_to_csv(summary_data, ["Metric", "Value"], summary_csv_path)
    print(f"Saved training summary to CSV: {summary_csv_path}")
    
    print(f"Stage 2 training for {shape_name} shape complete!")
    
    return train_losses, test_losses

def plot_stage2_comparison(stage2_results, args):
    """Create comparison plots for all stage 2 results in a single figure"""
    plots_dir = os.path.join(args.output_dir, "comparison_plots")
    csv_dir = os.path.join(args.output_dir, "csv_data")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    print(f"Created comparison plots directory: {plots_dir}")
    print(f"Created CSV data directory: {csv_dir}")
    
    # Create a 2x2 grid of subplots in a single figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot configurations
    plot_configs = [
        {'data': 'train_losses', 'scale': 'linear', 'title': 'Training Loss (Linear Scale)', 'pos': (0, 0)},
        {'data': 'train_losses', 'scale': 'log', 'title': 'Training Loss (Log Scale)', 'pos': (0, 1)},
        {'data': 'test_losses', 'scale': 'linear', 'title': 'Test Loss (Linear Scale)', 'pos': (1, 0)},
        {'data': 'test_losses', 'scale': 'log', 'title': 'Test Loss (Log Scale)', 'pos': (1, 1)}
    ]
    
    # Colors for different shapes
    colors = {
        'initial': 'blue',
        'lowest_condition_number': 'green',
        'lowest_test_mse': 'red',
        'final': 'purple',
        'random': 'gray'
    }
    
    # Find global min/max for consistent y-axis scaling
    global_min = {
        'train_losses': float('inf'),
        'test_losses': float('inf')
    }
    
    global_max = {
        'train_losses': float('-inf'),
        'test_losses': float('-inf')
    }
    
    # Create CSV data structure for combined comparison
    combined_train_data = []
    combined_test_data = []
    headers_train = ["Epoch"]
    headers_test = ["Epoch"]
    
    # Find longest loss array for epoch counting
    max_epochs = 0
    for data_type in ['train_losses', 'test_losses']:
        for shape_name, losses in stage2_results[data_type].items():
            if len(losses) > max_epochs:
                max_epochs = len(losses)
    
    # Initialize epoch column
    epoch_column = list(range(max_epochs))
    combined_train_data.append(epoch_column)
    combined_test_data.append(epoch_column)
    
    for data_type in ['train_losses', 'test_losses']:
        for shape_name, losses in stage2_results[data_type].items():
            if losses:  # Check if losses list is not empty
                global_min[data_type] = min(global_min[data_type], min(losses))
                global_max[data_type] = max(global_max[data_type], max(losses))
                
                # Add data to combined CSV data structure
                if data_type == 'train_losses':
                    headers_train.append(f"{shape_name}")
                    # Pad with NaN for matching length
                    padded_losses = losses + [float('nan')] * (max_epochs - len(losses))
                    combined_train_data.append(padded_losses)
                else:  # test_losses
                    headers_test.append(f"{shape_name}")
                    # Pad with NaN for matching length
                    padded_losses = losses + [float('nan')] * (max_epochs - len(losses))
                    combined_test_data.append(padded_losses)
    
    # Save combined CSV data
    train_csv_path = os.path.join(csv_dir, "stage2_train_losses_comparison.csv")
    test_csv_path = os.path.join(csv_dir, "stage2_test_losses_comparison.csv")
    
    # Transpose data for CSV (epochs in rows, shapes in columns)
    combined_train_data_transposed = list(map(list, zip(*combined_train_data)))
    combined_test_data_transposed = list(map(list, zip(*combined_test_data)))
    
    save_data_to_csv(combined_train_data_transposed, headers_train, train_csv_path)
    save_data_to_csv(combined_test_data_transposed, headers_test, test_csv_path)
    
    # Create each subplot
    for config in plot_configs:
        row, col = config['pos']
        ax = axes[row, col]
        data_type = config['data']
        
        # Add each shape's loss curve
        for shape_name, losses in stage2_results[data_type].items():
            color = colors.get(shape_name, 'orange')
            x = range(1, len(losses) + 1)
            ax.plot(x, losses, label=f"{shape_name}", color=color, linewidth=2)
        
        ax.set_title(config['title'], fontsize=14)
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('MSE Loss', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set y-limits for consistent scaling (linear scale only)
        if config['scale'] == 'linear':
            # Add a small buffer to make the plot look better
            buffer = (global_max[data_type] - global_min[data_type]) * 0.05
            ax.set_ylim(global_min[data_type] - buffer, global_max[data_type] + buffer)
        
        # Set log scale if needed
        if config['scale'] == 'log':
            ax.set_yscale('log')
    
    # Add a single legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5, fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the legend at the top
    
    # Save the combined plot
    combined_path = os.path.join(plots_dir, "combined_comparison.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined comparison plot to: {combined_path}")
    plt.close()
    
    # Also create individual plots for reference
    for config in plot_configs:
        plt.figure(figsize=(10, 6))
        data_type = config['data']
        
        # Create individual comparison CSV files for each plot type
        if config['scale'] == 'linear':
            individual_csv_path = os.path.join(csv_dir, f"{data_type}_{config['scale']}_comparison.csv")
            individual_data = []
            individual_headers = ["Epoch"]
            
            # Find max length for this specific comparison
            max_len = 0
            for shape_name, losses in stage2_results[data_type].items():
                if len(losses) > max_len:
                    max_len = len(losses)
            
            epoch_column = list(range(max_len))
            individual_data.append(epoch_column)
            
            for shape_name, losses in stage2_results[data_type].items():
                individual_headers.append(shape_name)
                padded_losses = losses + [float('nan')] * (max_len - len(losses))
                individual_data.append(padded_losses)
            
            # Transpose for CSV
            individual_data_transposed = list(map(list, zip(*individual_data)))
            save_data_to_csv(individual_data_transposed, individual_headers, individual_csv_path)
        
        for shape_name, losses in stage2_results[data_type].items():
            color = colors.get(shape_name, 'orange')
            x = range(1, len(losses) + 1)
            plt.plot(x, losses, label=f"{shape_name}", color=color, linewidth=2)
        
        plt.title(config['title'], fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Set y-limits for consistent scaling (linear scale only)
        if config['scale'] == 'linear':
            # Add a small buffer to make the plot look better
            buffer = (global_max[data_type] - global_min[data_type]) * 0.05
            plt.ylim(global_min[data_type] - buffer, global_max[data_type] + buffer)
        
        if config['scale'] == 'log':
            plt.yscale('log')
        
        filename = f"{config['data']}_{config['scale']}.png"
        individual_path = os.path.join(plots_dir, filename)
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        print(f"Saved individual plot to: {individual_path}")
        plt.close()
    
    # Save condition numbers for each shape type as CSV
    if stage2_results['condition_numbers']:
        cond_csv_path = os.path.join(csv_dir, "stage2_condition_numbers.csv")
        cond_data = []
        cond_headers = ["Shape_Type", "Condition_Number"]
        
        for shape_name, cond_number in stage2_results['condition_numbers'].items():
            cond_data.append([shape_name, cond_number])
        
        save_data_to_csv(cond_data, cond_headers, cond_csv_path)
        print(f"Saved condition numbers to CSV: {cond_csv_path}")
        
if __name__ == "__main__":
    main()

def run_stage2(recorded_shapes, recorded_metrics, train_loader, test_loader, args):
    """Run stage 2 with all recorded shapes"""
    stage2_results = {
        'train_losses': {},
        'test_losses': {},
        'condition_numbers': {}
    }
    
    # Add random baseline shape if desired
    if args.add_random_baseline and recorded_shapes:
        template_shape = next(iter(recorded_shapes.values()))
        random_shape = torch.rand_like(template_shape)
        # Ensure the first column (presence) follows the same pattern as other shapes
        # Set first point to always be present, then decreasing probability for subsequent points
        random_shape[0, 0] = 1.0  # First point always present
        for i in range(1, random_shape.shape[0]):
            random_shape[i, 0] = float(torch.rand(1) > 0.3)  # Approx. 70% chance of presence
        
        recorded_shapes['random'] = random_shape
        
        # Calculate condition number for random shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Create a temporary shape2filter model to get condition number
            shape2filter = Shape2FilterModel().to(device)
            with torch.no_grad():
                random_filter = shape2filter(random_shape.unsqueeze(0).to(device))[0]
                random_condition_number = calculate_condition_number(random_filter.cpu())
        except Exception as e:
            print(f"Error calculating condition number for random shape: {e}")
            random_condition_number = 1000.0  # Default high value
        
        recorded_metrics['random'] = {
            'condition_number': random_condition_number,
            'test_mse': 0.1  # Default value, will be updated during training
        }
        
        # Save random shape
        shapes_dir = os.path.join(args.output_dir, "recorded_shapes")
        np_save_path = os.path.join(shapes_dir, "random_shape.npy")
        np.save(np_save_path, random_shape.detach().cpu().numpy())
        print(f"Saved random shape to: {np_save_path}")
        
        # Save visualization
        viz_save_path = os.path.join(shapes_dir, "random_shape.png")
        visualize_shape(random_shape, viz_save_path)
        print(f"Saved random shape visualization to: {viz_save_path}")
        
        print(f"Added random baseline shape with condition number: {random_condition_number:.4f}")
    
    # Create directory for combined CSV data
    csv_dir = os.path.join(args.output_dir, "csv_data")
    os.makedirs(csv_dir, exist_ok=True)
    
    # Save shape metrics to CSV before training
    shape_metrics_csv = os.path.join(csv_dir, "shape_metrics_before_training.csv")
    shape_metrics_data = []
    for shape_name, metrics in recorded_metrics.items():
        if shape_name in recorded_shapes:
            shape_metrics_data.append([
                shape_name,
                metrics['condition_number'],
                metrics['test_mse']
            ])
    
    # Save shape metrics CSV
    save_data_to_csv(shape_metrics_data, 
                  ["Shape_Type", "Condition_Number", "Initial_Test_MSE"], 
                  shape_metrics_csv)
    print(f"Saved shape metrics CSV to: {shape_metrics_csv}")
    
    # Train with each recorded shape
    for shape_name, shape in recorded_shapes.items():
        print(f"\n=== Stage 2: Training with fixed {shape_name} shape ===")
        if shape_name in recorded_metrics:
            print(f"Shape condition number: {recorded_metrics[shape_name]['condition_number']:.4f}")
            stage2_results['condition_numbers'][shape_name] = recorded_metrics[shape_name]['condition_number']
        
        try:
            train_losses, test_losses = train_with_fixed_shape(
                shape_name, shape, train_loader, test_loader, args)
            
            stage2_results['train_losses'][shape_name] = train_losses
            stage2_results['test_losses'][shape_name] = test_losses
            print(f"Successfully completed training with {shape_name} shape")
        except Exception as e:
            print(f"Error training with {shape_name} shape: {e}")
    
    # Save final comparison metrics after all training
    final_metrics_csv = os.path.join(csv_dir, "shape_metrics_after_training.csv")
    final_metrics_data = []
    headers = ["Shape_Type", "Condition_Number", "Initial_Test_MSE", "Final_Test_MSE", "Best_Test_MSE", "Improvement_Percent"]
    
    for shape_name in recorded_shapes.keys():
        if shape_name in stage2_results['test_losses']:
            test_losses = stage2_results['test_losses'][shape_name]
            if test_losses:
                initial_test_mse = test_losses[0]
                final_test_mse = test_losses[-1]
                best_test_mse = min(test_losses)
                improvement_pct = (1 - final_test_mse/initial_test_mse) * 100
                
                condition_number = stage2_results['condition_numbers'].get(shape_name, float('nan'))
                
                final_metrics_data.append([
                    shape_name,
                    condition_number,
                    initial_test_mse,
                    final_test_mse,
                    best_test_mse,
                    improvement_pct
                ])
    
    # Save final metrics
    save_data_to_csv(final_metrics_data, headers, final_metrics_csv)
    print(f"Saved final comparison metrics to: {final_metrics_csv}")
    
    return stage2_results

def load_shapes_from_directory(load_shapes_dir):
    """Load recorded shapes from directory"""
    recorded_shapes = {}
    recorded_metrics = {}
    
    # Load shape files
    shape_files = {
        'initial': 'initial_shape.npy',
        'lowest_condition_number': 'lowest_condition_number_shape.npy',
        'lowest_test_mse': 'lowest_test_mse_shape.npy',
        'final': 'final_shape.npy'
    }
    
    for shape_name, file_name in shape_files.items():
        file_path = os.path.join(load_shapes_dir, file_name)
        if os.path.exists(file_path):
            shape_data = np.load(file_path)
            recorded_shapes[shape_name] = torch.tensor(shape_data, dtype=torch.float32)
            print(f"Loaded {shape_name} shape from {file_path}")
    
    # Load metrics if available
    metrics_file = os.path.join(load_shapes_dir, "shape_metrics.txt")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            
        current_shape = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.endswith("shape:"):
                current_shape = line.split()[0]
                if current_shape not in recorded_metrics:
                    recorded_metrics[current_shape] = {}
            elif current_shape and "Condition Number:" in line:
                recorded_metrics[current_shape]['condition_number'] = float(line.split(":")[-1].strip())
            elif current_shape and "Test MSE:" in line:
                recorded_metrics[current_shape]['test_mse'] = float(line.split(":")[-1].strip())
        
        print(f"Loaded metrics for {len(recorded_metrics)} shapes")
    else:
        print(f"Warning: Metrics file {metrics_file} not found. Using default values.")
        for shape_name in recorded_shapes:
            recorded_metrics[shape_name] = {
                'condition_number': 100.0,  # Default value
                'test_mse': 0.001  # Default value
            }
    
    return recorded_shapes, recorded_metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AVIRIS Fixed Shape Experiment')
    
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
    
    # Filter2Shape2Filter arguments
    parser.add_argument('--use_fsf', action='store_true', help='Use filter2shape2filter pipeline in encoder')
    parser.add_argument('--shape2filter_path', type=str, 
                        default="outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt",
                        help='Path to the shape2filter model weights')
    parser.add_argument('--filter2shape_path', type=str, 
                        default="outputs_three_stage_20250322_145925/stageC/spec2shape_stageC.pt",
                        help='Path to the filter2shape model weights')
    parser.add_argument('--filter_scale_factor', type=float, default=50.0,
                        help='Scale factor to divide FSF pipeline output by (default: 50.0)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for stage 1 (default: 50)')
    parser.add_argument('--stage2_epochs', type=int, default=100, 
                        help='Number of epochs for stage 2 with fixed shapes (default: 100)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--encoder_lr', type=float, default=1e-3, help='Encoder learning rate (default: 1e-3)')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='Decoder learning rate (default: 1e-4)')
    parser.add_argument('--test_split', type=float, default=0.2, help='Test split ratio (default: 0.2)')
    
    # Experiment control
    parser.add_argument('--skip_stage1', action='store_true', help='Skip stage 1 and load shapes from a directory')
    parser.add_argument('--skip_stage2', action='store_true', help='Skip stage 2 and only run stage 1')
    parser.add_argument('--load_shapes_dir', type=str, default=None, 
                        help='Directory to load shapes from when skipping stage 1')
    parser.add_argument('--add_random_baseline', action='store_true', 
                        help='Add random shape as baseline in stage 2')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory (default: results_fixed_shape_[model]_[timestamp])')
    parser.add_argument('--viz_interval', type=int, default=5, help='Visualization interval in epochs (default: 5)')
    
    args = parser.parse_args()
    
    # Add timestamp to output directory if not provided
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results_fixed_shape_{args.model}_{timestamp}"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Created output directory: {args.output_dir}")
    
    # Save arguments to a file for reference
    args_path = os.path.join(args.output_dir, 'args.txt')
    with open(args_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    print(f"Saved arguments to: {args_path}")
    
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize variables for shapes and metrics
    recorded_shapes = {}
    recorded_metrics = {}
    
    # Stage 1: Train model and record shapes
    if not args.skip_stage1:
        # Check if FSF pipeline models exist
        if args.use_fsf:
            if os.path.exists(args.shape2filter_path) and os.path.exists(args.filter2shape_path):
                print("Filter2Shape2Filter integration enabled.")
                print(f"Using filter scale factor: {args.filter_scale_factor}")
            else:
                print("Warning: FSF model paths not found. Disabling FSF integration.")
                args.use_fsf = False
        
        print("\n=== Stage 1: Learning filter shape and decoder ===")
        
        # Create model
        in_channels = tiles.shape[1]
        model = CompressionModel(
            in_channels=in_channels, 
            latent_dim=args.latent_dim, 
            decoder_type=args.model,
            use_fsf=args.use_fsf,
            shape2filter_path=args.shape2filter_path,
            filter2shape_path=args.filter2shape_path,
            filter_scale_factor=args.filter_scale_factor
        )
        
        print(f"Model initialized with:")
        print(f"- {in_channels} input channels")
        print(f"- {args.latent_dim} latent dimensions")
        print(f"- {args.model} decoder")
        print(f"- FSF pipeline: {'Enabled' if args.use_fsf else 'Disabled'}")
        
        # Train model and record shapes
        recorded_shapes, recorded_metrics, train_losses, test_losses = train_model_stage1(
            model, train_loader, test_loader, args)
        
        # Plot loss curves for stage 1
        loss_plot_path = os.path.join(args.output_dir, "stage1_loss_curves.png")
        plot_loss_curves(train_losses, test_losses, loss_plot_path)
        
        print("\nStage 1 complete!")
        print(f"Recorded shapes: {list(recorded_shapes.keys())}")
        
    else:
        # Skip stage 1 and load shapes from directory
        if args.load_shapes_dir is None:
            raise ValueError("Must provide --load_shapes_dir when using --skip_stage1")
        
        print(f"\n=== Skipping Stage 1, loading shapes from {args.load_shapes_dir} ===")
        recorded_shapes, recorded_metrics = load_shapes_from_directory(args.load_shapes_dir)
    
    # Stage 2: Train with fixed shapes
    if not args.skip_stage2 and recorded_shapes:
        print("\n=== Stage 2: Training with fixed shapes ===")
        
        # Run stage 2 with all recorded shapes
        stage2_results = run_stage2(recorded_shapes, recorded_metrics, train_loader, test_loader, args)
        
        # Create comparison plots
        plot_stage2_comparison(stage2_results, args)
        
        print("\nStage 2 complete!")
        print("Comparison plots have been created.")
    
    print(f"\nExperiment completed successfully!")
    print(f"All results saved to: {args.output_dir}")