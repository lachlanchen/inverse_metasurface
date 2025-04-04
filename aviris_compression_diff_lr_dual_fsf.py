#!/usr/bin/env python3
"""
AVIRIS Compression with Simple Encoder and Selectable Decoder
-----------------------------------------------------------
- Creates tiles from AVIRIS_SIMPLE_SELECT data
- Implements a compression model with linear filter encoder
- Supports two decoder options: AWAN or simple 3-layer CNN
- Adds noise with random SNR from 10-40dB using reparameterization trick
- Visualizes reconstructions and filter evolution
- Includes filter2shape2filter visualization to relate filters to physical structures
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

# Filter2Shape2Filter imports
import numpy.linalg as LA

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

class Filter2Shape2FilterFrozen(nn.Module):
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
        if self.no_grad_frozen:
            with torch.no_grad():
                shape_pred = self.filter2shape(spec_input)
                spec_chain = self.shape2filter_frozen(shape_pred)
        else:
            shape_pred = self.filter2shape(spec_input)
            spec_chain = self.shape2filter_frozen(shape_pred)
        return shape_pred, spec_chain

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

def load_models(shape2filter_path, filter2shape_path, device=None):
    """
    Load the pretrained shape2filter and filter2shape models
    
    Parameters:
    shape2filter_path: Path to the pretrained shape2filter model
    filter2shape_path: Path to the pretrained filter2shape model
    device: Device to load the models to (if None, will use CUDA if available)
    
    Returns:
    tuple: (shape2filter, filter2shape) models
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the shape2filter model
    shape2filter = Shape2FilterModel()
    shape2filter.load_state_dict(torch.load(shape2filter_path, map_location=device))
    shape2filter = shape2filter.to(device)
    
    # Load the filter2shape model
    filter2shape = Filter2ShapeVarLen()
    filter2shape.load_state_dict(torch.load(filter2shape_path, map_location=device))
    filter2shape = filter2shape.to(device)
    
    return shape2filter, filter2shape

def create_pipeline(shape2filter, filter2shape, no_grad_frozen=True):
    """
    Create the filter2shape2filter pipeline
    
    Parameters:
    shape2filter: The loaded shape2filter model
    filter2shape: The loaded filter2shape model
    no_grad_frozen: Whether to use no_grad for the frozen shape2filter model
    
    Returns:
    Filter2Shape2FilterFrozen: The pipeline
    """
    return Filter2Shape2FilterFrozen(filter2shape, shape2filter, no_grad_frozen=no_grad_frozen)

def run_pipeline(input_filter, shape2filter_path=None, filter2shape_path=None, device=None, 
               return_shape=True, visualize=False, output_dir=None):
    """
    Simple function to run a filter through the filter2shape2filter pipeline
    
    Parameters:
    input_filter: Tensor of shape [11, 100] - the input filter parameters
    shape2filter_path: Path to the shape2filter model weights (optional)
    filter2shape_path: Path to the filter2shape model weights (optional)
    device: Device to run on (optional)
    return_shape: Whether to also return the intermediate shape (default: False)
    visualize: Whether to visualize the results (default: False)
    output_dir: Directory to save visualizations if visualize=True (optional)
    
    Returns:
    If return_shape=True: (shape, reconstructed_filter)
    If return_shape=False: reconstructed_filter
    Both without batch dimension (shape: [4, 3], filter: [11, 100])
    """
    # Default paths if not provided
    if shape2filter_path is None:
        shape2filter_path = "outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt"
    if filter2shape_path is None:
        filter2shape_path = "outputs_three_stage_20250322_145925/stageC/spec2shape_stageC.pt"
    
    # Set device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure input_filter is a tensor
    if not isinstance(input_filter, torch.Tensor):
        input_filter = torch.tensor(input_filter, dtype=torch.float32)
    
    # Move to device
    input_filter = input_filter.to(device)
    
    # Add batch dimension if needed
    if input_filter.dim() == 2:
        input_filter = input_filter.unsqueeze(0)  # [11, 100] -> [1, 11, 100]
    
    # Load models
    shape2filter, filter2shape = load_models(shape2filter_path, filter2shape_path, device)
    
    # Create pipeline
    pipeline = create_pipeline(shape2filter, filter2shape)
    
    # Run pipeline
    with torch.no_grad():
        shape_pred, recon_filter = pipeline(input_filter)
    
    # Remove batch dimension
    shape_pred = shape_pred[0]  # [1, 4, 3] -> [4, 3]
    recon_filter = recon_filter[0]  # [1, 11, 100] -> [11, 100]
    
    # Visualize if requested
    if visualize:
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"filter2shape2filter_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate condition numbers
        input_cn = calculate_condition_number(input_filter[0])
        recon_cn = calculate_condition_number(recon_filter)
        
        # Plot input filter
        plot_filter(
            input_filter[0].detach().cpu().numpy(),
            f"Input Filter (CN: {input_cn:.4f})",
            save_path=f"{output_dir}/input_filter.png",
            show=False
        )
        
        # Plot predicted shape
        plot_shape_with_c4(
            shape_pred.detach().cpu().numpy(),
            "Predicted Shape",
            save_path=f"{output_dir}/predicted_shape.png",
            show=False
        )
        
        # Plot reconstructed filter
        plot_filter(
            recon_filter.detach().cpu().numpy(),
            f"Reconstructed Filter (CN: {recon_cn:.4f})",
            save_path=f"{output_dir}/reconstructed_filter.png",
            show=False
        )
        
        print(f"Visualizations saved to {output_dir}/")
    
    # Return outputs based on return_shape flag
    if return_shape:
        return shape_pred.detach().cpu(), recon_filter.detach().cpu()
    else:
        return recon_filter.detach().cpu()

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

# The rest of AVIRIS compression code

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

def visualize_filter_with_shape(filter_A, save_path, shape2filter_path=None, filter2shape_path=None):
    """
    Visualize the filter matrix and its corresponding shape and reconstructed filter
    using the filter2shape2filter pipeline
    """
    # Convert to tensor if needed
    if isinstance(filter_A, np.ndarray):
        filter_A = torch.tensor(filter_A, dtype=torch.float32)
    
    # Get the shape and reconstructed filter
    shape_pred, recon_filter = run_pipeline(
        filter_A, 
        shape2filter_path=shape2filter_path, 
        filter2shape_path=filter2shape_path,
        return_shape=True, 
        visualize=False
    )
    
    # Convert to numpy
    filter_A_np = filter_A.detach().cpu().numpy() if isinstance(filter_A, torch.Tensor) else filter_A
    shape_pred_np = shape_pred.detach().cpu().numpy() if isinstance(shape_pred, torch.Tensor) else shape_pred
    recon_filter_np = recon_filter.detach().cpu().numpy() if isinstance(recon_filter, torch.Tensor) else recon_filter
    
    # Calculate condition numbers
    filter_cond = calculate_condition_number(filter_A_np)
    recon_cond = calculate_condition_number(recon_filter_np)
    
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
    for i in range(recon_filter_np.shape[0]):
        ax3.plot(recon_filter_np[i], label=f"Filter {i+1}" if i % 3 == 0 else None)
    ax3.set_title(f"Reconstructed Filter (Condition Number: {recon_cond:.4f})")
    ax3.set_xlabel("Wavelength Index")
    ax3.set_ylabel("Filter Value")
    ax3.grid(True, alpha=0.3)
    if recon_filter_np.shape[0] <= 11:  # Only show legend for small number of filters
        ax3.legend()
    
    # Plot the difference (bottom right)
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    diff = np.abs(filter_A_np - recon_filter_np)
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
    
    # Also save individual filter plots
    filter_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    
    # Original filter
    plt.figure(figsize=(12, 8))
    for i in range(filter_A_np.shape[0]):
        plt.plot(filter_A_np[i], label=f"Filter {i+1}")
    plt.title(f"Original Filter (Condition Number: {filter_cond:.4f})")
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{filter_dir}/{base_name}_original.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Shape
    plt.figure(figsize=(8, 8))
    plot_shape_with_c4(shape_pred_np, "Predicted Shape", show=False)
    plt.tight_layout()
    plt.savefig(f"{filter_dir}/{base_name}_shape.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Reconstructed filter
    plt.figure(figsize=(12, 8))
    for i in range(recon_filter_np.shape[0]):
        plt.plot(recon_filter_np[i], label=f"Filter {i+1}")
    plt.title(f"Reconstructed Filter (Condition Number: {recon_cond:.4f})")
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{filter_dir}/{base_name}_reconstructed.png", dpi=300, bbox_inches="tight")
    plt.close()

def visualize_filter(filter_A, save_path, shape2filter_path=None, filter2shape_path=None):
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
    
    # If shape2filter_path and filter2shape_path are provided, also visualize the filter2shape2filter pipeline
    if shape2filter_path is not None and filter2shape_path is not None:
        # Use a separate file to avoid overwriting
        fsf_path = save_path.replace('.png', '_with_shape.png')
        try:
            # Convert the filter to a tensor if it's a numpy array
            if isinstance(filter_A, np.ndarray):
                filter_A_tensor = torch.tensor(filter_A, dtype=torch.float32)
                visualize_filter_with_shape(filter_A_tensor, fsf_path, shape2filter_path, filter2shape_path)
            else:
                visualize_filter_with_shape(filter_A, fsf_path, shape2filter_path, filter2shape_path)
        except Exception as e:
            print(f"Error in filter2shape2filter visualization: {e}")
            print("Continuing without filter2shape2filter visualization.")

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
    filter_A_np = model.encoder.filter_A.detach().cpu().numpy()
    visualize_filter(
        filter_A_np, 
        os.path.join(filter_dir, "filter_initial.png"),
        shape2filter_path=args.shape2filter_path,
        filter2shape_path=args.filter2shape_path
    )
    
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
            filter_A_np = model.encoder.filter_A.detach().cpu().numpy()
            visualize_filter(
                filter_A_np, 
                os.path.join(filter_dir, f"filter_epoch_{epoch+1}.png"),
                shape2filter_path=args.shape2filter_path,
                filter2shape_path=args.filter2shape_path
            )
            
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
    
    # Filter2Shape2Filter arguments
    parser.add_argument('--shape2filter_path', type=str, 
                        default="outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt",
                        help='Path to the shape2filter model weights')
    parser.add_argument('--filter2shape_path', type=str, 
                        default="outputs_three_stage_20250322_145925/stageC/spec2shape_stageC.pt",
                        help='Path to the filter2shape model weights')
    parser.add_argument('--disable_fsf', action='store_true', 
                        help='Disable filter2shape2filter visualization')
    
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
    
    # Check if filter2shape2filter models exist
    if args.disable_fsf:
        args.shape2filter_path = None
        args.filter2shape_path = None
    else:
        if not os.path.exists(args.shape2filter_path):
            print(f"Warning: shape2filter model not found at {args.shape2filter_path}")
            print("Filter2Shape2Filter visualization will be disabled.")
            args.shape2filter_path = None
            args.filter2shape_path = None
        elif not os.path.exists(args.filter2shape_path):
            print(f"Warning: filter2shape model not found at {args.filter2shape_path}")
            print("Filter2Shape2Filter visualization will be disabled.")
            args.shape2filter_path = None
            args.filter2shape_path = None
        else:
            print("Filter2Shape2Filter visualization enabled.")
    
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