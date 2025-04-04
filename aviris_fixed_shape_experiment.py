#!/usr/bin/env python3
"""
AVIRIS Fixed Shape Experiment
----------------------------
This script extends the original AVIRIS compression model to:
1. Record important shapes from Stage 1 (initial, lowest condition number, lowest test MSE)
2. Run Stage 2 with fixed shapes from Stage 1, optimizing only the decoder
3. Compare performance of different fixed shapes

Usage:
    # Run both stages
    python aviris_fixed_shape_experiment.py --use_fsf --model awan --tile_size 100 --epochs 100 
    --stage2_epochs 100 --batch_size 64 --encoder_lr 1e-3 --decoder_lr 5e-4 --min_snr 10 --max_snr 40 
    --shape2filter_path "outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt" 
    --filter2shape_path "outputs_three_stage_20250322_145925/stageC/spec2shape_stageC.pt" 
    --filter_scale_factor 10.0
    
    # Skip stage 1 and only run stage 2
    python aviris_fixed_shape_experiment.py --use_fsf --model awan --tile_size 100 
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

# Calculate condition number of filters
def calculate_condition_number(filters):
    """Calculate condition number of the spectral filters matrix"""
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
    
    os.makedirs(filter_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(shapes_dir, exist_ok=True)
    
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
        np.save(os.path.join(shapes_dir, "initial_shape.npy"), initial_shape.detach().cpu().numpy())
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
            
            # Check for lowest condition number
            if current_condition_number < recorded_metrics['lowest_condition_number']['condition_number']:
                recorded_shapes['lowest_condition_number'] = current_shape
                recorded_metrics['lowest_condition_number']['condition_number'] = current_condition_number
                recorded_metrics['lowest_condition_number']['test_mse'] = avg_test_loss
                
                # Save shape
                np.save(os.path.join(shapes_dir, "lowest_condition_number_shape.npy"), 
                      current_shape.detach().cpu().numpy())
                print(f"New lowest condition number: {current_condition_number:.4f}")
            
            # Check for lowest test MSE
            if avg_test_loss < recorded_metrics['lowest_test_mse']['test_mse']:
                recorded_shapes['lowest_test_mse'] = current_shape
                recorded_metrics['lowest_test_mse']['condition_number'] = current_condition_number
                recorded_metrics['lowest_test_mse']['test_mse'] = avg_test_loss
                
                # Save shape
                np.save(os.path.join(shapes_dir, "lowest_test_mse_shape.npy"), 
                      current_shape.detach().cpu().numpy())
                print(f"New lowest test MSE: {avg_test_loss:.6f}")
        
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"Saved new best model with test loss: {best_test_loss:.6f}")
        
        # Visualize reconstruction periodically
        if (epoch + 1) % args.viz_interval == 0:
            visualize_reconstruction(model, test_loader, device, 
                                    os.path.join(recon_dir, f"recon_epoch_{epoch+1}.png"))
    
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
        np.save(os.path.join(shapes_dir, "final_shape.npy"), final_shape.detach().cpu().numpy())
        print(f"Recorded final shape with condition number: {final_condition_number:.4f}")
    
    # Save metrics for all recorded shapes
    with open(os.path.join(shapes_dir, "shape_metrics.txt"), 'w') as f:
        for shape_name, metrics in recorded_metrics.items():
            if shape_name in recorded_shapes:
                f.write(f"{shape_name} shape:\n")
                f.write(f"  Condition Number: {metrics['condition_number']:.4f}\n")
                f.write(f"  Test MSE: {metrics['test_mse']:.6f}\n\n")
    
    # Save condition numbers
    np.save(os.path.join(args.output_dir, "condition_numbers.npy"), np.array(condition_numbers))
    
    # Plot condition numbers
    if condition_numbers:
        plt.figure(figsize=(10, 6))
        plt.plot(condition_numbers, 'b-')
        plt.title('Filter Condition Number Evolution')
        plt.xlabel('Epochs')
        plt.ylabel('Condition Number')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(args.output_dir, "condition_number_evolution.png"), dpi=300)
        plt.close()

        # Log scale plot
        plt.figure(figsize=(10, 6))
        plt.semilogy(condition_numbers, 'r-')
        plt.title('Filter Condition Number Evolution (Log Scale)')
        plt.xlabel('Epochs')
        plt.ylabel('Condition Number (log scale)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(args.output_dir, "condition_number_evolution_log.png"), dpi=300)
        plt.close()
    
    return recorded_shapes, recorded_metrics, train_losses, test_losses

def train_with_fixed_shape(shape_name, shape, train_loader, test_loader, args):
    """Train model with fixed shape, optimizing only the decoder"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory for this shape
    shape_dir = os.path.join(args.output_dir, f"stage2_{shape_name}")
    os.makedirs(shape_dir, exist_ok=True)
    os.makedirs(os.path.join(shape_dir, "reconstructions"), exist_ok=True)
    
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
            torch.save(model.state_dict(), os.path.join(shape_dir, "best_model.pt"))
            print(f"Saved new best model with test loss: {best_test_loss:.6f}")
        
        # Visualize reconstruction periodically
        if (epoch + 1) % args.viz_interval == 0 or epoch == args.stage2_epochs - 1:
            visualize_reconstruction(model, test_loader, device, 
                                   os.path.join(shape_dir, "reconstructions", f"recon_epoch_{epoch+1}.png"))
    
    # Save loss values and plots
    np.savez(os.path.join(shape_dir, "loss_values.npz"), 
            train_losses=np.array(train_losses), 
            test_losses=np.array(test_losses))
    
    # Plot loss curves
    plot_loss_curves(train_losses, test_losses, os.path.join(shape_dir, "loss_curves.png"))
    
    print(f"Stage 2 training for {shape_name} shape complete!")
    
    return train_losses, test_losses

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
        random_shape[:, 0] = (random_shape[:, 0] > 0.5).float()
        recorded_shapes['random'] = random_shape
        recorded_metrics['random'] = {
            'condition_number': 1000.0,  # Default high value
            'test_mse': 0.1  # Default high value
        }
        print("Added random baseline shape")
    
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
        except Exception as e:
            print(f"Error training with {shape_name} shape: {e}")
    
    return stage2_results

def plot_stage2_comparison(stage2_results, args):
    """Create comparison plots for all stage 2 results"""
    plots_dir = os.path.join(args.output_dir, "comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create 4 plots: train/test × linear/log scale
    plot_configs = [
        {'data': 'train_losses', 'scale': 'linear', 'title': 'Training Loss (Linear Scale)'},
        {'data': 'train_losses', 'scale': 'log', 'title': 'Training Loss (Log Scale)'},
        {'data': 'test_losses', 'scale': 'linear', 'title': 'Test Loss (Linear Scale)'},
        {'data': 'test_losses', 'scale': 'log', 'title': 'Test Loss (Log Scale)'}
    ]
    
    # Colors for different shapes
    colors = {
        'initial': 'blue',
        'lowest_condition_number': 'green',
        'lowest_test_mse': 'red',
        'final': 'purple',
        'random': 'gray'
    }
    
    # Create each plot
    for config in plot_configs:
        plt.figure(figsize=(12, 8))
        
        # Add each shape's loss curve
        for shape_name, losses in stage2_results[config['data']].items():
            color = colors.get(shape_name, 'orange')
            x = range(1, len(losses) + 1)
            plt.plot(x, losses, label=f"{shape_name}", color=color, linewidth=2)
        
        plt.title(config['title'], fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('MSE Loss', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Set log scale if needed
        if config['scale'] == 'log':
            plt.yscale('log')
        
        # Save the plot
        filename = f"{config['data']}_{config['scale']}.png"
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Comparison plots saved to {plots_dir}")

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
        plot_loss_curves(train_losses, test_losses, os.path.join(args.output_dir, "stage1_loss_curves.png"))
        
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

if __name__ == "__main__":
    main()