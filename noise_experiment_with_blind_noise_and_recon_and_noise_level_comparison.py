#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import numpy.linalg as LA
import shutil
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import math
import random
import json
from matplotlib.colors import Normalize
from matplotlib import cm

# Import common functions and classes from the original script
from noise_experiment_with_blind_noise import (
    load_aviris_forest_data, is_valid_tile,
    Shape2FilterModel, Filter2ShapeVarLen, Filter2Shape2FilterFrozen,
    calculate_condition_number, replicate_c4, sort_points_by_angle, plot_shape_with_c4,
    generate_initial_filter, DecoderCNN5Layer
)

# from noise_experiment_with_blind_noise import HyperspectralAutoencoderRandomNoise

from AWAN import AWAN
latent_dim = 11
in_channels = 100

###############################################################################
# METRICS FUNCTIONS
###############################################################################
def calculate_psnr(original, reconstructed, data_range=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between original and reconstructed images.
    
    Parameters:
    original: Original image tensor
    reconstructed: Reconstructed image tensor
    data_range: The data range of the input image (default: 1.0 for normalized images)
    
    Returns:
    float: PSNR value in dB
    """
    # Ensure tensors are on CPU and converted to numpy arrays
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    
    # Calculate MSE
    mse = np.mean((original - reconstructed) ** 2)
    
    # Avoid division by zero
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    
    return psnr

def calculate_sam(original, reconstructed, epsilon=1e-8):
    """
    Calculate Spectral Angle Mapper (SAM) between original and reconstructed images.
    
    Parameters:
    original: Original image tensor of shape [H, W, C] or [B, H, W, C]
    reconstructed: Reconstructed image tensor of shape [H, W, C] or [B, H, W, C]
    epsilon: Small value to avoid division by zero
    
    Returns:
    float: Mean SAM value in radians
    """
    # Ensure tensors are on CPU and converted to numpy arrays
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    
    # Handle batch dimension if present
    if len(original.shape) == 4:  # [B, H, W, C]
        # Reshape to [B*H*W, C]
        orig_reshaped = original.reshape(-1, original.shape[-1])
        recon_reshaped = reconstructed.reshape(-1, reconstructed.shape[-1])
    else:  # [H, W, C]
        # Reshape to [H*W, C]
        orig_reshaped = original.reshape(-1, original.shape[-1])
        recon_reshaped = reconstructed.reshape(-1, reconstructed.shape[-1])
    
    # Calculate dot product for each pixel
    dot_product = np.sum(orig_reshaped * recon_reshaped, axis=1)
    
    # Calculate magnitudes
    orig_mag = np.sqrt(np.sum(orig_reshaped ** 2, axis=1))
    recon_mag = np.sqrt(np.sum(recon_reshaped ** 2, axis=1))
    
    # Avoid division by zero
    valid_pixels = (orig_mag > epsilon) & (recon_mag > epsilon)
    
    if not np.any(valid_pixels):
        return 0.0  # All pixels are zeros
    
    # Calculate cosine similarity
    cos_sim = np.zeros_like(dot_product)
    cos_sim[valid_pixels] = dot_product[valid_pixels] / (orig_mag[valid_pixels] * recon_mag[valid_pixels])
    
    # Clip to [-1, 1] to handle numerical errors
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    # Calculate angle in radians
    angles = np.arccos(cos_sim)
    
    # Return mean angle
    return np.mean(angles)

###############################################################################
# VISUALIZATION FUNCTIONS
###############################################################################
def visualize_reconstruction(model, sample_tensor, device, save_path, band_idx=50):
    """
    Create visualization of original, reconstructed, and difference images with unified colorbar
    
    Parameters:
    model: Model to use for reconstruction
    sample_tensor: Input tensor to reconstruct (single sample with batch dimension)
    device: Device to use for computation
    save_path: Path to save the visualization
    band_idx: Index of spectral band to visualize (default: 50, middle of 100 bands)
    """
    with torch.no_grad():
        # Ensure sample is on the correct device
        sample = sample_tensor.to(device)
        
        # Get reconstruction
        recon, _, snr = model(sample)
        
        # Move tensors to CPU for visualization
        sample_np = sample.cpu().numpy()[0]  # Remove batch dimension
        recon_np = recon.cpu().numpy()[0]    # Remove batch dimension
        
        # Calculate difference
        diff_np = sample_np - recon_np
        
        # Extract specific spectral band
        sample_band = sample_np[:, :, band_idx]
        recon_band = recon_np[:, :, band_idx]
        diff_band = diff_np[:, :, band_idx]
        
        # Calculate global min and max for unified colormap
        vmin = min(sample_band.min(), recon_band.min())
        vmax = max(sample_band.max(), recon_band.max())
        
        # Calculate symmetric limits for difference
        diff_abs_max = max(abs(diff_band.min()), abs(diff_band.max()))
        
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Plot original
        plt.subplot(1, 3, 1)
        im1 = plt.imshow(sample_band, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title(f'Original (Band {band_idx})')
        plt.colorbar(im1, fraction=0.046, pad=0.04)
        
        # Plot reconstruction
        plt.subplot(1, 3, 2)
        im2 = plt.imshow(recon_band, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title(f'Reconstructed (Band {band_idx}, SNR: {snr:.2f} dB)')
        plt.colorbar(im2, fraction=0.046, pad=0.04)
        
        # Plot difference
        plt.subplot(1, 3, 3)
        im3 = plt.imshow(diff_band, cmap='coolwarm', vmin=-diff_abs_max, vmax=diff_abs_max)
        plt.title(f'Difference (Band {band_idx})')
        plt.colorbar(im3, fraction=0.046, pad=0.04)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate metrics
        mse = ((sample_np - recon_np) ** 2).mean()
        psnr = calculate_psnr(sample_np, recon_np)
        sam = calculate_sam(sample_np, recon_np)
        
        return mse, psnr, sam

###############################################################################
# MODIFIED AUTOENCODER MODEL WITH RANDOM NOISE
###############################################################################
class HyperspectralAutoencoderRandomNoise(nn.Module):
    def __init__(self, shape2filter_path, filter2shape_path, min_snr=10, max_snr=40, 
                 initial_filter_params=None, filter_scale_factor=10.0):
        super().__init__()

        # Target SNR range (in dB) as model parameters
        self.min_snr = min_snr
        self.max_snr = max_snr
        
        # Filter scale factor
        self.filter_scale_factor = filter_scale_factor
        
        # Device for model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the pretrained models
        self.shape2filter = Shape2FilterModel()
        self.shape2filter.load_state_dict(torch.load(shape2filter_path, map_location=self.device))
        self.shape2filter = self.shape2filter.to(self.device)
        
        self.filter2shape = Filter2ShapeVarLen()
        self.filter2shape.load_state_dict(torch.load(filter2shape_path, map_location=self.device))
        self.filter2shape = self.filter2shape.to(self.device)
        
        # Freeze both models
        for param in self.shape2filter.parameters():
            param.requires_grad = False
            
        for param in self.filter2shape.parameters():
            param.requires_grad = False
        
        # Create the filter2shape2filter pipeline
        self.pipeline = Filter2Shape2FilterFrozen(self.filter2shape, self.shape2filter, no_grad_frozen=False)
        
        # Initialize learnable filter parameters (11 x 100)
        # Use provided initial parameters if available, otherwise generate new ones
        if initial_filter_params is not None:
            self.filter_params = nn.Parameter(initial_filter_params.clone().to(self.device))
        else:
            # Default to random initialization
            self.filter_params = nn.Parameter(torch.rand(11, 100, device=self.device))
        
        # Decoder: AWAN
        self.decoder = AWAN(inplanes=latent_dim, planes=in_channels, channels=128, n_DRBs=2)
    
    def get_current_filter(self):
        """Return the current learnable filter parameters"""
        return self.filter_params
    
    def get_current_shape(self):
        """Get the current shape from filter2shape model"""
        filter = self.get_current_filter().unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            shape = self.filter2shape(filter)[0]  # Remove batch dimension
        return shape
    
    def get_reconstructed_filter(self):
        """Get the reconstructed filter from the full pipeline"""
        filter = self.get_current_filter().unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            _, recon_filter = self.pipeline(filter)
        return recon_filter[0]  # Remove batch dimension

    def get_reconstructed_filter_with_grad(self):
        """Get the reconstructed filter from the full pipeline"""
        filter = self.get_current_filter().unsqueeze(0)  # Add batch dimension
        _, recon_filter = self.pipeline(filter)
        return recon_filter[0]  # Remove batch dimension

    def add_random_noise(self, tensor):
        """
        Add Gaussian white noise to tensor, with noise level randomly chosen
        between self.min_snr and self.max_snr for each batch.
        """
        # Choose a random noise level within the specified range
        target_snr = random.uniform(self.min_snr, self.max_snr)
            
        # Calculate signal power (mean square), using detach() to separate calculation graph
        signal_power = tensor.detach().pow(2).mean()
        # Convert signal power to decibels
        signal_power_db = 10 * torch.log10(signal_power)
        # Calculate noise power (decibels)
        noise_power_db = signal_power_db - target_snr
        # Convert noise power back to linear scale
        noise_power = 10 ** (noise_power_db / 10)
        # Generate Gaussian white noise with the same shape as input tensor
        noise = torch.randn_like(tensor) * torch.sqrt(noise_power)
        # Return tensor with added noise
        return tensor + noise, target_snr
    
    def forward(self, x):
        """
        Forward pass of the autoencoder with random noise level
        Input: x - Hyperspectral data of shape [batch_size, height, width, 100]
                or shape [batch_size, 100, height, width]
        """
        # Ensure input is in BHWC format
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D tensor, got shape {x.shape}")
            
        if x.shape[1] == 100 and x.shape[3] != 100:  # Input is in BCHW format
            x = x.permute(0, 2, 3, 1)  # Convert to BHWC
            
        # Get dimensions
        batch_size, height, width, spectral_bands = x.shape
        
        # Get reconstructed filter (important fix)
        filter = self.get_reconstructed_filter_with_grad()  # Shape: [11, 100]
        
        # Convert input from BHWC to BCHW format for PyTorch convolution
        x_channels_first = x.permute(0, 3, 1, 2)
        
        # Normalize filter for filtering using the scale factor
        filter_normalized = filter / self.filter_scale_factor
        
        # Use efficient tensor operations for spectral filtering
        # Einstein summation: 'bchw,oc->bohw'
        # This performs the weighted sum across spectral dimension for each output band
        encoded_channels_first = torch.einsum('bchw,oc->bohw', x_channels_first, filter_normalized)

        # ----------------- Add random noise ------------------
        encoded_channels_first, applied_snr = self.add_random_noise(encoded_channels_first)
        
        # Convert encoded data back to channels-last format [B,H,W,C]
        encoded = encoded_channels_first.permute(0, 2, 3, 1)
        
        # Decode: use the CNN decoder to expand from 11 to 100 bands
        decoded_channels_first = self.decoder(encoded_channels_first)
        
        # Convert back to original format [B,H,W,C]
        decoded = decoded_channels_first.permute(0, 2, 3, 1)
        
        return decoded, encoded, applied_snr

###############################################################################
# FIXED SHAPE MODEL WITH FIXED NOISE LEVEL
###############################################################################
class FixedShapeModel(nn.Module):
    """Model with fixed shape encoder and trainable decoder with fixed noise level"""
    def __init__(self, shape, shape2filter_path, noise_level=30, filter_scale_factor=10.0, device=None):
        super(FixedShapeModel, self).__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.noise_level = noise_level
        self.filter_scale_factor = filter_scale_factor
        
        # Load shape2filter model
        self.shape2filter = Shape2FilterModel().to(device)
        self.shape2filter.load_state_dict(torch.load(shape2filter_path, map_location=device))
        self.shape2filter.eval()  # Set to evaluation mode
        
        for param in self.shape2filter.parameters():
            param.requires_grad = False
        
        # Convert shape to tensor if it's a numpy array
        if isinstance(shape, np.ndarray):
            self.shape = torch.tensor(shape, dtype=torch.float32).to(device)
        else:
            self.shape = shape.to(device)
        
        # Precompute filter from shape
        with torch.no_grad():
            self.register_buffer(
                'fixed_filter', 
                self.shape2filter(self.shape.unsqueeze(0))[0]
            )
        
        print(f"Fixed shape encoder initialized with filter of shape {self.fixed_filter.shape}")
        
        # Decoder
        self.decoder = AWAN(inplanes=latent_dim, planes=in_channels, channels=128, n_DRBs=2)
    
    def add_noise(self, z):
        """Add noise with fixed SNR level"""
        # Calculate signal power
        signal_power = z.detach().pow(2).mean()
        # Convert signal power to decibels
        signal_power_db = 10 * torch.log10(signal_power)
        # Calculate noise power (decibels)
        noise_power_db = signal_power_db - self.noise_level
        # Convert noise power back to linear scale
        noise_power = 10 ** (noise_power_db / 10)
        # Generate Gaussian white noise
        noise = torch.randn_like(z) * torch.sqrt(noise_power)
        # Return tensor with added noise
        return z + noise
    
    def forward(self, x, add_noise=True):
        """
        Forward pass of the fixed shape model
        Input: x - Hyperspectral data of shape [B, H, W, C] or [B, C, H, W]
        """
        # Ensure input is in BHWC format
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D tensor, got shape {x.shape}")
            
        if x.shape[1] == 100 and x.shape[3] != 100:  # Input is in BCHW format
            x = x.permute(0, 2, 3, 1)  # Convert to BHWC
            
        # Get dimensions
        batch_size, height, width, spectral_bands = x.shape
        
        # Convert to channels-first for processing
        x_channels_first = x.permute(0, 3, 1, 2)
        
        # Apply the fixed filter
        filter_normalized = self.fixed_filter / self.filter_scale_factor
        encoded_channels_first = torch.einsum('bchw,oc->bohw', x_channels_first, filter_normalized)
        
        # Add noise if specified
        if add_noise:
            encoded_channels_first = self.add_noise(encoded_channels_first)
        
        # Decode
        decoded_channels_first = self.decoder(encoded_channels_first)
        
        # Convert back to BHWC format
        encoded = encoded_channels_first.permute(0, 2, 3, 1)
        decoded = decoded_channels_first.permute(0, 2, 3, 1)
        
        return decoded, encoded

###############################################################################
# TRAINING FUNCTION WITH RANDOM NOISE
###############################################################################
def train_with_random_noise(shape2filter_path, filter2shape_path, output_dir, min_snr=10, max_snr=40, 
                           initial_filter_params=None, batch_size=10, num_epochs=500, 
                           encoder_lr=0.001, decoder_lr=0.001, filter_scale_factor=10.0,
                           cache_file=None, use_cache=False, folder_patterns="all",
                           train_data=None, test_data=None):
    """
    Train and visualize the hyperspectral autoencoder with random noise levels between min_snr and max_snr
    using filter2shape2filter architecture with separate learning rates for encoder and decoder
    
    Parameters:
    shape2filter_path: Path to the pretrained shape2filter model
    filter2shape_path: Path to the pretrained filter2shape model
    output_dir: Directory to save outputs
    min_snr: Minimum SNR level in dB for random noise
    max_snr: Maximum SNR level in dB for random noise
    initial_filter_params: Initial filter parameters (11x100)
    batch_size: Batch size for training
    num_epochs: Number of training epochs
    encoder_lr: Learning rate for encoder (filter parameters)
    decoder_lr: Learning rate for decoder
    filter_scale_factor: Scaling factor for filter normalization
    cache_file: Path to cache file for storing processed data
    use_cache: If True, try to load from cache file first
    folder_patterns: Comma-separated list of folder name patterns to include, or 'all' for all folders
    train_data: Optional training data, if None will be loaded
    test_data: Optional testing data, if None will be loaded
    
    Returns:
    tuple: (initial_shape_path, least_mse_shape_path, lowest_cn_shape_path, final_shape_path, output_dir)
    """
    # Create output directory with noise range
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    noise_dir = f"blind_noise_{min_snr}dB_to_{max_snr}dB"
    output_dir = os.path.join(output_dir, noise_dir + f"_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}/")
    
    # Create subfolders
    viz_dir = os.path.join(output_dir, "intermediate_viz")
    recon_dir = os.path.join(output_dir, "reconstructions")
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load or use provided data
    if train_data is None:
        data = load_aviris_forest_data(base_path="AVIRIS_FOREST_SIMPLE_SELECT", tile_size=128, 
                                      cache_file=cache_file, use_cache=use_cache, folder_patterns=folder_patterns)
        data = data.to(device)
    else:
        data = train_data.to(device)
    
    print("Training data shape:", data.shape)
    
    # Create dataset and dataloader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # If no initial filter parameters were provided, generate them
    if initial_filter_params is None:
        initial_filter_params = generate_initial_filter(device)
    
    # Initialize model with random noise level range and initial filter parameters
    model = HyperspectralAutoencoderRandomNoise(
        shape2filter_path, filter2shape_path, 
        min_snr=min_snr, max_snr=max_snr, 
        initial_filter_params=initial_filter_params,
        filter_scale_factor=filter_scale_factor
    ).to(device)
    
    print(f"Model initialized with random noise range: {min_snr} to {max_snr} dB")
    print(f"Filter scale factor: {filter_scale_factor}")
    print(f"Encoder learning rate: {encoder_lr}")
    print(f"Decoder learning rate: {decoder_lr}")
    
    # Get initial filter, shape, and filters for visualization
    initial_filter = model.get_current_filter().detach().cpu().numpy()
    initial_shape = model.get_current_shape().detach().cpu().numpy()
    
    # Calculate initial condition number
    initial_condition_number = calculate_condition_number(model.get_reconstructed_filter().detach().cpu())
    print(f"Initial condition number: {initial_condition_number:.4f}")
    
    # Save initial shape with C4 symmetry
    initial_shape_path = f"{output_dir}/initial_shape.png"
    plot_shape_with_c4(initial_shape, f"Initial Shape", initial_shape_path)
    print(f"Initial shape saved to: {os.path.abspath(initial_shape_path)}")
    
    # Save initial filter
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(initial_filter[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Initial Spectral Parameters (SNR: {min_snr}-{max_snr} dB)")
    plt.legend()
    initial_filter_path = f"{output_dir}/initial_filter.png"
    plt.savefig(initial_filter_path)
    plt.close()
    
    # Also save the reconstructed filter from the pipeline
    initial_recon_filter = model.get_reconstructed_filter().detach().cpu().numpy()
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(initial_recon_filter[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Initial Reconstructed Filter (SNR: {min_snr}-{max_snr} dB)")
    plt.legend()
    initial_recon_filter_path = f"{output_dir}/initial_recon_filter.png"
    plt.savefig(initial_recon_filter_path)
    plt.close()
    
    # Before training, get reconstruction of sample from train set
    sample_idx = min(5, len(data) - 1)  # Make sure index is within range
    sample_tensor = data[sample_idx:sample_idx+1]  # Keep as tensor with batch dimension
    
    # If test data is provided, also get test sample
    test_sample_tensor = None
    if test_data is not None:
        test_sample_idx = min(5, len(test_data) - 1)
        test_sample_tensor = test_data[test_sample_idx:test_sample_idx+1].to(device)
    
    # Generate initial reconstruction for visualization
    initial_recon_path = os.path.join(recon_dir, "initial_reconstruction.png")
    initial_mse, initial_psnr, initial_sam = visualize_reconstruction(
        model, sample_tensor, device, initial_recon_path)
    
    print(f"Initial metrics - MSE: {initial_mse:.6f}, PSNR: {initial_psnr:.2f} dB, SAM: {initial_sam:.6f} rad")
    
    # Initialize tracking variables
    losses = []
    condition_numbers = [initial_condition_number]
    train_mse_values = [initial_mse]
    train_psnr_values = [initial_psnr]
    train_sam_values = [initial_sam]
    applied_snr_values = []
    
    # Variables for tracking best metrics
    lowest_condition_number = float('inf')
    lowest_cn_shape = None
    lowest_cn_filter = None
    lowest_cn_epoch = -1
    
    lowest_train_mse = initial_mse
    lowest_mse_shape = initial_shape.copy()
    lowest_mse_filter = initial_filter.copy()
    lowest_mse_epoch = -1
    
    # Create separate optimizers for encoder and decoder
    encoder_optimizer = optim.Adam([model.filter_params], lr=encoder_lr)
    decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=decoder_lr)
    
    criterion = nn.MSELoss()
    
    print(f"Starting training with random noise SNR range: {min_snr} to {max_snr} dB...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        epoch_snr_values = []
        
        model.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = batch[0]
            
            # Forward pass with random noise
            recon, _, batch_snr = model(x)
            epoch_snr_values.append(batch_snr)
            
            # Calculate loss
            loss = criterion(recon, x)
            
            # Backward pass and optimize
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss and SNR for epoch
        avg_loss = epoch_loss / num_batches
        avg_snr = sum(epoch_snr_values) / len(epoch_snr_values) if epoch_snr_values else 0
        applied_snr_values.append(avg_snr)
        
        losses.append(avg_loss)
        
        # Calculate condition number for current epoch
        current_filter = model.get_reconstructed_filter().detach().cpu()
        current_condition_number = calculate_condition_number(current_filter)
        condition_numbers.append(current_condition_number)
        
        # Get current shape for visualization
        current_shape = model.get_current_shape().detach().cpu().numpy()
        current_filter_raw = model.get_current_filter().detach().cpu()
        
        # Check if this is the lowest condition number so far
        if current_condition_number < lowest_condition_number:
            lowest_condition_number = current_condition_number
            lowest_cn_shape = current_shape.copy()
            lowest_cn_filter = current_filter_raw.clone()
            lowest_cn_epoch = epoch
            print(f"New lowest condition number: {lowest_condition_number:.4f} at epoch {epoch+1}")
        
        # Evaluate model on sample
        model.eval()
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:  # Every 10 epochs
            recon_path = os.path.join(recon_dir, f"reconstruction_epoch_{epoch+1}.png")
            current_mse, current_psnr, current_sam = visualize_reconstruction(
                model, sample_tensor, device, recon_path)
            
            # Save metrics
            train_mse_values.append(current_mse)
            train_psnr_values.append(current_psnr)
            train_sam_values.append(current_sam)
            
            print(f"Epoch {epoch+1} metrics - MSE: {current_mse:.6f}, PSNR: {current_psnr:.2f} dB, SAM: {current_sam:.6f} rad")
            
            # Check if this is the lowest MSE so far
            if current_mse < lowest_train_mse:
                lowest_train_mse = current_mse
                lowest_mse_shape = current_shape.copy()
                lowest_mse_filter = current_filter_raw.clone()
                lowest_mse_epoch = epoch
                print(f"New lowest MSE: {lowest_train_mse:.6f} at epoch {epoch+1}")
        
        # Save intermediate shapes and visualizations
        if (epoch+1) % (num_epochs // 2) == 0 or epoch == 0:
            # Save shape visualization to intermediate directory
            plot_shape_with_c4(
                current_shape, 
                f"Shape at Epoch {epoch+1}", 
                f"{viz_dir}/shape_epoch_{epoch+1}.png"
            )
            
            # Save filter visualization
            plt.figure(figsize=(10, 6))
            for i in range(11):
                plt.plot(current_filter.numpy()[i], label=f'Filter {i}' if i % 3 == 0 else None)
            plt.grid(True)
            plt.xlabel("Wavelength Index")
            plt.ylabel("Filter Value")
            plt.title(f"Filter at Epoch {epoch+1}")
            plt.legend()
            plt.savefig(f"{viz_dir}/filter_epoch_{epoch+1}.png")
            plt.close()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Avg SNR: {avg_snr:.2f} dB, Condition Number: {current_condition_number:.4f}")
    
    # Get final filter and shape
    final_filter = model.get_current_filter().detach().cpu()
    final_shape = model.get_current_shape().detach().cpu().numpy()
    
    # Calculate final condition number
    final_condition_number = calculate_condition_number(model.get_reconstructed_filter().detach().cpu())
    print(f"Final condition number: {final_condition_number:.4f}")
    
    # Get final reconstruction and metrics
    final_recon_path = os.path.join(recon_dir, "final_reconstruction.png")
    final_mse, final_psnr, final_sam = visualize_reconstruction(
        model, sample_tensor, device, final_recon_path)
    
    print(f"Final metrics - MSE: {final_mse:.6f}, PSNR: {final_psnr:.2f} dB, SAM: {final_sam:.6f} rad")
    
    # Save final shape
    final_shape_path = f"{output_dir}/final_shape.png"
    plot_shape_with_c4(final_shape, f"Final Shape", final_shape_path)
    print(f"Final shape saved to: {os.path.abspath(final_shape_path)}")
    
    # Save final filter
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(final_filter.numpy()[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Final Spectral Parameters (SNR: {min_snr}-{max_snr} dB)")
    plt.legend()
    final_filter_path = f"{output_dir}/final_filter.png"
    plt.savefig(final_filter_path)
    plt.close()
    
    # Save reconstructed filter
    final_recon_filter = model.get_reconstructed_filter().detach().cpu().numpy()
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(final_recon_filter[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Final Reconstructed Filter (SNR: {min_snr}-{max_snr} dB)")
    plt.legend()
    final_recon_filter_path = f"{output_dir}/final_recon_filter.png"
    plt.savefig(final_recon_filter_path)
    plt.close()
    
    # Create directories for lowest CN and MSE results
    lowest_cn_dir = os.path.join(output_dir, "lowest_cn")
    lowest_mse_dir = os.path.join(output_dir, "lowest_mse")
    os.makedirs(lowest_cn_dir, exist_ok=True)
    os.makedirs(lowest_mse_dir, exist_ok=True)
    
    # Save the lowest condition number shape and filter
    lowest_cn_shape_path = f"{lowest_cn_dir}/shape.png"
    plot_shape_with_c4(lowest_cn_shape, f"Shape with Lowest CN: {lowest_condition_number:.4f}", lowest_cn_shape_path)
    
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(lowest_cn_filter.numpy()[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Spectral Parameters with Lowest Condition Number: {lowest_condition_number:.4f}")
    plt.legend()
    lowest_cn_filter_path = f"{lowest_cn_dir}/filter.png"
    plt.savefig(lowest_cn_filter_path)
    plt.close()
    
    # Save the lowest MSE shape and filter
    lowest_mse_shape_path = f"{lowest_mse_dir}/shape.png"
    plot_shape_with_c4(lowest_mse_shape, f"Shape with Lowest MSE: {lowest_train_mse:.6f}", lowest_mse_shape_path)
    
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(lowest_mse_filter.numpy()[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Spectral Parameters with Lowest MSE: {lowest_train_mse:.6f}")
    plt.legend()
    lowest_mse_filter_path = f"{lowest_mse_dir}/filter.png"
    plt.savefig(lowest_mse_filter_path)
    plt.close()
    
    # Save as NumPy files
    np.save(f"{output_dir}/initial_shape.npy", initial_shape)
    np.save(f"{output_dir}/final_shape.npy", final_shape)
    np.save(f"{lowest_cn_dir}/shape.npy", lowest_cn_shape)
    np.save(f"{lowest_mse_dir}/shape.npy", lowest_mse_shape)
    
    np.save(f"{output_dir}/initial_filter.npy", initial_filter)
    np.save(f"{output_dir}/final_filter.npy", final_filter.numpy())
    np.save(f"{lowest_cn_dir}/filter.npy", lowest_cn_filter.numpy())
    np.save(f"{lowest_mse_dir}/filter.npy", lowest_mse_filter.numpy())
    
    # Plot metrics during training
    # 1. Condition Number
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs+1), condition_numbers, 'r-')
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Condition Number")
    plt.title(f"Filter Matrix Condition Number During Training (SNR: {min_snr}-{max_snr} dB)")
    plt.axhline(y=lowest_condition_number, color='g', linestyle='--', 
                label=f'Lowest CN: {lowest_condition_number:.4f} (Epoch {lowest_cn_epoch+1})')
    plt.legend()
    plt.savefig(f"{output_dir}/condition_number.png")
    plt.close()
    
    # 2. Training Loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Loss (SNR: {min_snr}-{max_snr} dB)")
    plt.savefig(f"{output_dir}/training_loss.png")
    plt.close()
    
    # 3. Applied SNR
    plt.figure(figsize=(10, 5))
    plt.plot(applied_snr_values)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Average SNR (dB)")
    plt.title(f"Average Applied SNR During Training (Range: {min_snr}-{max_snr} dB)")
    plt.ylim(min_snr-1, max_snr+1)
    plt.savefig(f"{output_dir}/applied_snr.png")
    plt.close()
    
    # 4. MSE values
    plt.figure(figsize=(10, 5))
    epochs_with_metrics = list(range(0, num_epochs+1, 10))
    if epochs_with_metrics[-1] != num_epochs:
        epochs_with_metrics.append(num_epochs)
    
    plt.plot(epochs_with_metrics, train_mse_values, 'b-o', label='MSE')
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"MSE During Training (SNR: {min_snr}-{max_snr} dB)")
    plt.legend()
    plt.savefig(f"{output_dir}/mse_values.png")
    plt.close()
    
    # 5. PSNR values
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_with_metrics, train_psnr_values, 'g-o', label='PSNR')
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title(f"PSNR During Training (SNR: {min_snr}-{max_snr} dB)")
    plt.legend()
    plt.savefig(f"{output_dir}/psnr_values.png")
    plt.close()
    
    # 6. SAM values
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_with_metrics, train_sam_values, 'm-o', label='SAM')
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("SAM (rad)")
    plt.title(f"SAM During Training (SNR: {min_snr}-{max_snr} dB)")
    plt.legend()
    plt.savefig(f"{output_dir}/sam_values.png")
    plt.close()
    
    # 7. Combined metrics (MSE, PSNR, SAM)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # MSE subplot
    ax1.plot(epochs_with_metrics, train_mse_values, 'b-o', label='MSE')
    ax1.set_ylabel("MSE")
    ax1.set_title(f"Image Quality Metrics During Training (SNR: {min_snr}-{max_snr} dB)")
    ax1.grid(True)
    ax1.legend()
    
    # PSNR subplot
    ax2.plot(epochs_with_metrics, train_psnr_values, 'g-o', label='PSNR')
    ax2.set_ylabel("PSNR (dB)")
    ax2.grid(True)
    ax2.legend()
    
    # SAM subplot
    ax3.plot(epochs_with_metrics, train_sam_values, 'm-o', label='SAM')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("SAM (rad)")
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_metrics.png")
    plt.close()
    
    # Save model
    model_save_path = f"{output_dir}/model_state.pt"
    torch.save({
        'filter_params': model.filter_params.detach().cpu(),
        'decoder_state_dict': model.decoder.state_dict(),
        'filter_scale_factor': model.filter_scale_factor
    }, model_save_path)
    
    # Save all numerical data
    np.save(f"{output_dir}/condition_numbers.npy", np.array(condition_numbers))
    np.save(f"{output_dir}/losses.npy", np.array(losses))
    np.save(f"{output_dir}/train_mse_values.npy", np.array(train_mse_values))
    np.save(f"{output_dir}/train_psnr_values.npy", np.array(train_psnr_values))
    np.save(f"{output_dir}/train_sam_values.npy", np.array(train_sam_values))
    np.save(f"{output_dir}/applied_snr_values.npy", np.array(applied_snr_values))
    
    # Create log file to save training parameters
    with open(f"{output_dir}/training_params.txt", "w") as f:
        f.write(f"Noise level range (SNR): {min_snr} to {max_snr} dB\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Encoder learning rate: {encoder_lr}\n")
        f.write(f"Decoder learning rate: {decoder_lr}\n")
        f.write(f"Filter scale factor: {filter_scale_factor}\n")
        f.write(f"Cache file: {cache_file}\n")
        f.write(f"Used cache: {use_cache}\n")
        f.write(f"Folder patterns: {folder_patterns}\n")
        f.write("\n")
        
        # Save condition number information
        f.write(f"Initial condition number: {initial_condition_number:.4f}\n")
        f.write(f"Final condition number: {final_condition_number:.4f}\n")
        f.write(f"Lowest condition number: {lowest_condition_number:.4f} at epoch {lowest_cn_epoch+1}\n")
        f.write(f"Condition number change: {final_condition_number - initial_condition_number:.4f}\n\n")
        
        # Save metrics information
        f.write(f"Initial metrics - MSE: {initial_mse:.6f}, PSNR: {initial_psnr:.2f} dB, SAM: {initial_sam:.6f} rad\n")
        f.write(f"Final metrics - MSE: {final_mse:.6f}, PSNR: {final_psnr:.2f} dB, SAM: {final_sam:.6f} rad\n")
        f.write(f"Lowest MSE: {lowest_train_mse:.6f} at epoch {lowest_mse_epoch+1}\n")
        f.write(f"MSE improvement: {initial_mse - final_mse:.6f} ({(1 - final_mse/initial_mse) * 100:.2f}%)\n")
        f.write(f"PSNR improvement: {final_psnr - initial_psnr:.2f} dB\n")
        f.write(f"SAM improvement: {initial_sam - final_sam:.6f} rad ({(1 - final_sam/initial_sam) * 100:.2f}%)\n")
    
    print(f"Training with random noise SNR range {min_snr} to {max_snr} dB completed.")
    print(f"All results saved to {output_dir}/")
    
    # Return paths to all shapes for later training
    initial_shape_npy_path = f"{output_dir}/initial_shape.npy"
    final_shape_npy_path = f"{output_dir}/final_shape.npy"
    lowest_cn_shape_npy_path = f"{lowest_cn_dir}/shape.npy"
    lowest_mse_shape_npy_path = f"{lowest_mse_dir}/shape.npy"
    
    return initial_shape_npy_path, lowest_mse_shape_npy_path, lowest_cn_shape_npy_path, final_shape_npy_path, output_dir

###############################################################################
# FIXED SHAPE DECODER TRAINING WITH METRICS
###############################################################################
def train_with_fixed_shape(shape_name, shape, shape2filter_path, train_loader, test_loader, 
                          noise_level, num_epochs, batch_size, decoder_lr, filter_scale_factor, output_dir):
    """
    Train model with fixed shape encoder and optimize only the decoder
    
    Parameters:
    shape_name: Name of the shape for logs and outputs
    shape: Tensor or array representing the shape
    shape2filter_path: Path to shape2filter model
    train_loader: Training data loader
    test_loader: Testing data loader
    noise_level: SNR level in dB to use
    num_epochs: Number of epochs for training
    batch_size: Batch size for training
    decoder_lr: Learning rate for decoder optimizer
    filter_scale_factor: Scaling factor for filter normalization
    output_dir: Directory to save outputs
    
    Returns:
    tuple: Dictionary with metrics over epochs
    """
    # Create output directory
    shape_dir = os.path.join(output_dir, f"noise_{noise_level}dB_{shape_name}")
    recon_dir = os.path.join(shape_dir, "reconstructions")
    os.makedirs(shape_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with fixed shape
    model = FixedShapeModel(
        shape=shape,
        shape2filter_path=shape2filter_path,
        noise_level=noise_level,
        filter_scale_factor=filter_scale_factor,
        device=device
    )
    model = model.to(device)
    
    # Create optimizer for decoder only
    optimizer = optim.Adam(model.decoder.parameters(), lr=decoder_lr)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Track metrics
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'train_psnr': [],
        'test_psnr': [],
        'train_sam': [],
        'test_sam': []
    }
    
    # Get sample batch for visualization
    train_sample = next(iter(train_loader))
    if isinstance(train_sample, list) or isinstance(train_sample, tuple):
        train_sample = train_sample[0]
    train_sample = train_sample[:1].to(device)
    
    test_sample = next(iter(test_loader))
    if isinstance(test_sample, list) or isinstance(test_sample, tuple):
        test_sample = test_sample[0]
    test_sample = test_sample[:1].to(device)
    
    # Initial visualization and metrics
    model.eval()
    with torch.no_grad():
        # Train sample
        train_recon, _ = model(train_sample, add_noise=False)
        train_mse = ((train_recon - train_sample) ** 2).mean().item()
        train_psnr = calculate_psnr(train_sample.cpu(), train_recon.cpu())
        train_sam = calculate_sam(train_sample.cpu(), train_recon.cpu())
        
        # Test sample
        test_recon, _ = model(test_sample, add_noise=False)
        test_mse = ((test_recon - test_sample) ** 2).mean().item()
        test_psnr = calculate_psnr(test_sample.cpu(), test_recon.cpu())
        test_sam = calculate_sam(test_sample.cpu(), test_recon.cpu())
    
    # Save initial metrics
    metrics['train_loss'].append(train_mse)
    metrics['test_loss'].append(test_mse)
    metrics['train_psnr'].append(train_psnr)
    metrics['test_psnr'].append(test_psnr)
    metrics['train_sam'].append(train_sam)
    metrics['test_sam'].append(test_sam)
    
    # Initial visualization
    initial_recon_path = os.path.join(recon_dir, "reconstruction_epoch_0.png")
    with torch.no_grad():
        model.eval()
        train_out, _ = model(train_sample, add_noise=False)
        
        # Plot original, reconstruction, and difference
        plt.figure(figsize=(15, 5))
        
        # Select middle band for visualization
        band_idx = train_sample.shape[-1] // 2
        
        # Get data as numpy
        orig = train_sample[0].permute(2, 0, 1)[band_idx].cpu().numpy()
        recon = train_out[0].permute(2, 0, 1)[band_idx].cpu().numpy()
        diff = orig - recon
        
        # Find global min/max for colorbar
        vmin = min(orig.min(), recon.min())
        vmax = max(orig.max(), recon.max())
        
        # Calculate symmetric limits for difference
        diff_abs_max = max(abs(diff.min()), abs(diff.max()))
        
        # Plot original
        plt.subplot(1, 3, 1)
        im1 = plt.imshow(orig, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title('Original')
        plt.colorbar(im1, fraction=0.046, pad=0.04)
        
        # Plot reconstruction
        plt.subplot(1, 3, 2)
        im2 = plt.imshow(recon, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title('Reconstructed')
        plt.colorbar(im2, fraction=0.046, pad=0.04)
        
        # Plot difference
        plt.subplot(1, 3, 3)
        im3 = plt.imshow(diff, cmap='coolwarm', vmin=-diff_abs_max, vmax=diff_abs_max)
        plt.title('Difference')
        plt.colorbar(im3, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(initial_recon_path)
        plt.close()
    
    # Print initial metrics
    print(f"Initial {shape_name} metrics at {noise_level} dB:")
    print(f"  Train - MSE: {train_mse:.6f}, PSNR: {train_psnr:.2f} dB, SAM: {train_sam:.6f} rad")
    print(f"  Test  - MSE: {test_mse:.6f}, PSNR: {test_psnr:.2f} dB, SAM: {test_sam:.6f} rad")
    
    # Best metrics tracking
    best_test_loss = test_mse
    best_epoch = 0
    best_model_state = model.decoder.state_dict()
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            if isinstance(batch, list) or isinstance(batch, tuple):
                x = batch[0]
            else:
                x = batch
            
            x = x.to(device)
            
            # Forward pass
            recon, _ = model(x, add_noise=True)
            
            # Loss
            loss = criterion(recon, x)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            num_batches += 1
        
        # Calculate average training loss
        avg_train_loss = train_epoch_loss / num_batches
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            # Evaluate on train sample
            train_recon, _ = model(train_sample, add_noise=False)
            train_mse = ((train_recon - train_sample) ** 2).mean().item()
            train_psnr = calculate_psnr(train_sample.cpu(), train_recon.cpu())
            train_sam = calculate_sam(train_sample.cpu(), train_recon.cpu())
            
            # Evaluate on test sample
            test_recon, _ = model(test_sample, add_noise=False)
            test_mse = ((test_recon - test_sample) ** 2).mean().item()
            test_psnr = calculate_psnr(test_sample.cpu(), test_recon.cpu())
            test_sam = calculate_sam(test_sample.cpu(), test_recon.cpu())
            
            # Test set evaluation
            test_loss = 0.0
            for batch in test_loader:
                if isinstance(batch, list) or isinstance(batch, tuple):
                    x = batch[0]
                else:
                    x = batch
                
                x = x.to(device)
                recon, _ = model(x, add_noise=False)
                loss = criterion(recon, x)
                test_loss += loss.item()
            
            avg_test_loss = test_loss / len(test_loader)
        
        # Save metrics
        metrics['train_loss'].append(avg_train_loss)
        metrics['test_loss'].append(avg_test_loss)
        metrics['train_psnr'].append(train_psnr)
        metrics['test_psnr'].append(test_psnr)
        metrics['train_sam'].append(train_sam)
        metrics['test_sam'].append(test_sam)
        
        # Check if this is the best model so far
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_epoch = epoch
            best_model_state = model.decoder.state_dict()
            
            # Save best reconstruction
            best_recon_path = os.path.join(shape_dir, "best_reconstruction.png")
            with torch.no_grad():
                model.eval()
                train_out, _ = model(train_sample, add_noise=False)
                
                # Plot original, reconstruction, and difference
                plt.figure(figsize=(15, 5))
                
                # Select middle band for visualization
                band_idx = train_sample.shape[-1] // 2
                
                # Get data as numpy
                orig = train_sample[0].permute(2, 0, 1)[band_idx].cpu().numpy()
                recon = train_out[0].permute(2, 0, 1)[band_idx].cpu().numpy()
                diff = orig - recon
                
                # Find global min/max for colorbar
                vmin = min(orig.min(), recon.min())
                vmax = max(orig.max(), recon.max())
                
                # Calculate symmetric limits for difference
                diff_abs_max = max(abs(diff.min()), abs(diff.max()))
                
                # Plot original
                plt.subplot(1, 3, 1)
                im1 = plt.imshow(orig, cmap='viridis', vmin=vmin, vmax=vmax)
                plt.title('Original')
                plt.colorbar(im1, fraction=0.046, pad=0.04)
                
                # Plot reconstruction
                plt.subplot(1, 3, 2)
                im2 = plt.imshow(recon, cmap='viridis', vmin=vmin, vmax=vmax)
                plt.title(f'Best Reconstructed (Epoch {epoch+1})')
                plt.colorbar(im2, fraction=0.046, pad=0.04)
                
                # Plot difference
                plt.subplot(1, 3, 3)
                im3 = plt.imshow(diff, cmap='coolwarm', vmin=-diff_abs_max, vmax=diff_abs_max)
                plt.title('Difference')
                plt.colorbar(im3, fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                plt.savefig(best_recon_path)
                plt.close()
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}, {shape_name} shape at {noise_level} dB:")
            print(f"  Train - MSE: {avg_train_loss:.6f}, PSNR: {train_psnr:.2f} dB, SAM: {train_sam:.6f} rad")
            print(f"  Test  - MSE: {avg_test_loss:.6f}, PSNR: {test_psnr:.2f} dB, SAM: {test_sam:.6f} rad")
            
            # Save reconstruction
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                recon_path = os.path.join(recon_dir, f"reconstruction_epoch_{epoch+1}.png")
                with torch.no_grad():
                    model.eval()
                    train_out, _ = model(train_sample, add_noise=False)
                    
                    # Plot original, reconstruction, and difference
                    plt.figure(figsize=(15, 5))
                    
                    # Select middle band for visualization
                    band_idx = train_sample.shape[-1] // 2
                    
                    # Get data as numpy
                    orig = train_sample[0].permute(2, 0, 1)[band_idx].cpu().numpy()
                    recon = train_out[0].permute(2, 0, 1)[band_idx].cpu().numpy()
                    diff = orig - recon
                    
                    # Find global min/max for colorbar
                    vmin = min(orig.min(), recon.min())
                    vmax = max(orig.max(), recon.max())
                    
                    # Calculate symmetric limits for difference
                    diff_abs_max = max(abs(diff.min()), abs(diff.max()))
                    
                    # Plot original
                    plt.subplot(1, 3, 1)
                    im1 = plt.imshow(orig, cmap='viridis', vmin=vmin, vmax=vmax)
                    plt.title('Original')
                    plt.colorbar(im1, fraction=0.046, pad=0.04)
                    
                    # Plot reconstruction
                    plt.subplot(1, 3, 2)
                    im2 = plt.imshow(recon, cmap='viridis', vmin=vmin, vmax=vmax)
                    plt.title(f'Reconstructed (Epoch {epoch+1})')
                    plt.colorbar(im2, fraction=0.046, pad=0.04)
                    
                    # Plot difference
                    plt.subplot(1, 3, 3)
                    im3 = plt.imshow(diff, cmap='coolwarm', vmin=-diff_abs_max, vmax=diff_abs_max)
                    plt.title('Difference')
                    plt.colorbar(im3, fraction=0.046, pad=0.04)
                    
                    plt.tight_layout()
                    plt.savefig(recon_path)
                    plt.close()
    
    # Save model
    model_save_path = os.path.join(shape_dir, "decoder_model.pt")
    torch.save(model.decoder.state_dict(), model_save_path)
    
    # Save best model
    best_model_save_path = os.path.join(shape_dir, "best_decoder_model.pt")
    torch.save(best_model_state, best_model_save_path)
    
    # Plot training curves
    # Plot MSE
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs+1), metrics['train_loss'], 'b-', label='Train Loss')
    plt.plot(range(num_epochs+1), metrics['test_loss'], 'r-', label='Test Loss')
    plt.axvline(x=best_epoch, color='g', linestyle='--', 
                label=f'Best Epoch ({best_epoch}): {best_test_loss:.6f}')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title(f'{shape_name} Training/Test MSE at {noise_level} dB')
    plt.legend()
    plt.savefig(os.path.join(shape_dir, 'mse_curves.png'))
    plt.close()
    
    # Plot PSNR
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs+1), metrics['train_psnr'], 'b-', label='Train PSNR')
    plt.plot(range(num_epochs+1), metrics['test_psnr'], 'r-', label='Test PSNR')
    plt.axvline(x=best_epoch, color='g', linestyle='--', 
                label=f'Best Epoch ({best_epoch})')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.title(f'{shape_name} Training/Test PSNR at {noise_level} dB')
    plt.legend()
    plt.savefig(os.path.join(shape_dir, 'psnr_curves.png'))
    plt.close()
    
    # Plot SAM
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs+1), metrics['train_sam'], 'b-', label='Train SAM')
    plt.plot(range(num_epochs+1), metrics['test_sam'], 'r-', label='Test SAM')
    plt.axvline(x=best_epoch, color='g', linestyle='--', 
                label=f'Best Epoch ({best_epoch})')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('SAM (radians)')
    plt.title(f'{shape_name} Training/Test SAM at {noise_level} dB')
    plt.legend()
    plt.savefig(os.path.join(shape_dir, 'sam_curves.png'))
    plt.close()
    
    # Save metrics as numpy arrays
    np.save(os.path.join(shape_dir, 'train_loss.npy'), np.array(metrics['train_loss']))
    np.save(os.path.join(shape_dir, 'test_loss.npy'), np.array(metrics['test_loss']))
    np.save(os.path.join(shape_dir, 'train_psnr.npy'), np.array(metrics['train_psnr']))
    np.save(os.path.join(shape_dir, 'test_psnr.npy'), np.array(metrics['test_psnr']))
    np.save(os.path.join(shape_dir, 'train_sam.npy'), np.array(metrics['train_sam']))
    np.save(os.path.join(shape_dir, 'test_sam.npy'), np.array(metrics['test_sam']))
    
    return metrics

###############################################################################
# TRAINING WITH FIXED SHAPES AT DIFFERENT NOISE LEVELS
###############################################################################
def train_multiple_fixed_shapes(shapes_dict, shape2filter_path, output_dir, 
                               noise_levels, num_epochs, batch_size, decoder_lr, 
                               filter_scale_factor, train_loader, test_loader):
    """
    Train multiple fixed shapes at different noise levels
    
    Parameters:
    shapes_dict: Dictionary of shapes with paths
    shape2filter_path: Path to shape2filter model
    output_dir: Directory to save results
    noise_levels: List of noise levels (SNR in dB)
    num_epochs: Number of epochs for each training
    batch_size: Batch size for training
    decoder_lr: Learning rate for decoder
    filter_scale_factor: Scale factor for filter normalization
    train_loader: DataLoader for training data
    test_loader: DataLoader for test data
    
    Returns:
    dict: Dictionary with results for each shape and noise level
    """
    # Create directory for results
    results_dir = os.path.join(output_dir, "fixed_shape_comparison")
    os.makedirs(results_dir, exist_ok=True)
    
    # Results dictionary to store metrics for each shape and noise level
    results = {
        'train_loss': {shape_name: [] for shape_name in shapes_dict},
        'test_loss': {shape_name: [] for shape_name in shapes_dict},
        'train_psnr': {shape_name: [] for shape_name in shapes_dict},
        'test_psnr': {shape_name: [] for shape_name in shapes_dict},
        'train_sam': {shape_name: [] for shape_name in shapes_dict},
        'test_sam': {shape_name: [] for shape_name in shapes_dict},
        'all_metrics': {}
    }
    
    # Load shapes
    shapes = {}
    for shape_name, shape_path in shapes_dict.items():
        if os.path.exists(shape_path):
            shapes[shape_name] = np.load(shape_path)
            print(f"Loaded {shape_name} shape from: {shape_path}")
        else:
            print(f"Warning: {shape_name} shape not found at {shape_path}")
    
    # Train each shape with each noise level
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for noise_level in noise_levels:
        print(f"\n{'='*50}")
        print(f"Training with noise level: {noise_level} dB")
        print(f"{'='*50}")
        
        noise_results_dir = os.path.join(results_dir, f"noise_{noise_level}dB")
        os.makedirs(noise_results_dir, exist_ok=True)
        
        # Track metrics for shapes at this noise level for comparison
        noise_metrics = {
            'train_loss': {},
            'test_loss': {},
            'train_psnr': {},
            'test_psnr': {},
            'train_sam': {},
            'test_sam': {}
        }
        
        # Train each shape
        for shape_name, shape in shapes.items():
            print(f"\nTraining {shape_name} shape with noise level {noise_level} dB")
            
            # Train the model
            metrics = train_with_fixed_shape(
                shape_name=shape_name,
                shape=shape,
                shape2filter_path=shape2filter_path,
                train_loader=train_loader,
                test_loader=test_loader,
                noise_level=noise_level,
                num_epochs=num_epochs,
                batch_size=batch_size,
                decoder_lr=decoder_lr,
                filter_scale_factor=filter_scale_factor,
                output_dir=results_dir
            )
            
            # Save final metrics
            results['train_loss'][shape_name].append(metrics['train_loss'][-1])
            results['test_loss'][shape_name].append(metrics['test_loss'][-1])
            results['train_psnr'][shape_name].append(metrics['train_psnr'][-1])
            results['test_psnr'][shape_name].append(metrics['test_psnr'][-1])
            results['train_sam'][shape_name].append(metrics['train_sam'][-1])
            results['test_sam'][shape_name].append(metrics['test_sam'][-1])
            
            # Save metrics for comparison at this noise level
            noise_metrics['train_loss'][shape_name] = metrics['train_loss']
            noise_metrics['test_loss'][shape_name] = metrics['test_loss']
            noise_metrics['train_psnr'][shape_name] = metrics['train_psnr']
            noise_metrics['test_psnr'][shape_name] = metrics['test_psnr']
            noise_metrics['train_sam'][shape_name] = metrics['train_sam']
            noise_metrics['test_sam'][shape_name] = metrics['test_sam']
            
            # Also store all metrics for this shape and noise level
            results['all_metrics'][(shape_name, noise_level)] = metrics
        
        # Create comparison plots for this noise level
        # 1. MSE comparison
        plt.figure(figsize=(12, 8))
        for shape_name in shapes:
            plt.plot(range(num_epochs+1), noise_metrics['train_loss'][shape_name], 
                    label=f"{shape_name.capitalize()} Train")
            plt.plot(range(num_epochs+1), noise_metrics['test_loss'][shape_name], 
                    linestyle='--', label=f"{shape_name.capitalize()} Test")
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.title(f'MSE Comparison at {noise_level} dB')
        plt.legend()
        plt.savefig(os.path.join(noise_results_dir, 'mse_comparison.png'))
        plt.close()
        
        # 2. PSNR comparison
        plt.figure(figsize=(12, 8))
        for shape_name in shapes:
            plt.plot(range(num_epochs+1), noise_metrics['train_psnr'][shape_name], 
                    label=f"{shape_name.capitalize()} Train")
            plt.plot(range(num_epochs+1), noise_metrics['test_psnr'][shape_name], 
                    linestyle='--', label=f"{shape_name.capitalize()} Test")
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('PSNR (dB)')
        plt.title(f'PSNR Comparison at {noise_level} dB')
        plt.legend()
        plt.savefig(os.path.join(noise_results_dir, 'psnr_comparison.png'))
        plt.close()
        
        # 3. SAM comparison
        plt.figure(figsize=(12, 8))
        for shape_name in shapes:
            plt.plot(range(num_epochs+1), noise_metrics['train_sam'][shape_name], 
                    label=f"{shape_name.capitalize()} Train")
            plt.plot(range(num_epochs+1), noise_metrics['test_sam'][shape_name], 
                    linestyle='--', label=f"{shape_name.capitalize()} Test")
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('SAM (radians)')
        plt.title(f'SAM Comparison at {noise_level} dB')
        plt.legend()
        plt.savefig(os.path.join(noise_results_dir, 'sam_comparison.png'))
        plt.close()
    
    # Create overall comparison plots across noise levels
    # 1. Final MSE comparison
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    for shape_name in shapes:
        plt.plot(noise_levels, results['train_loss'][shape_name], 'o-', 
                label=f"{shape_name.capitalize()}")
    plt.grid(True)
    plt.xlabel('Noise Level (SNR dB)')
    plt.ylabel('Final Train MSE')
    plt.title('Final Training MSE Across Noise Levels')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for shape_name in shapes:
        plt.plot(noise_levels, results['test_loss'][shape_name], 'o-', 
                label=f"{shape_name.capitalize()}")
    plt.grid(True)
    plt.xlabel('Noise Level (SNR dB)')
    plt.ylabel('Final Test MSE')
    plt.title('Final Testing MSE Across Noise Levels')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'final_mse_across_noise.png'))
    plt.close()
    
    # 2. Final PSNR comparison
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    for shape_name in shapes:
        plt.plot(noise_levels, results['train_psnr'][shape_name], 'o-', 
                label=f"{shape_name.capitalize()}")
    plt.grid(True)
    plt.xlabel('Noise Level (SNR dB)')
    plt.ylabel('Final Train PSNR (dB)')
    plt.title('Final Training PSNR Across Noise Levels')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for shape_name in shapes:
        plt.plot(noise_levels, results['test_psnr'][shape_name], 'o-', 
                label=f"{shape_name.capitalize()}")
    plt.grid(True)
    plt.xlabel('Noise Level (SNR dB)')
    plt.ylabel('Final Test PSNR (dB)')
    plt.title('Final Testing PSNR Across Noise Levels')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'final_psnr_across_noise.png'))
    plt.close()
    
    # 3. Final SAM comparison
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    for shape_name in shapes:
        plt.plot(noise_levels, results['train_sam'][shape_name], 'o-', 
                label=f"{shape_name.capitalize()}")
    plt.grid(True)
    plt.xlabel('Noise Level (SNR dB)')
    plt.ylabel('Final Train SAM (radians)')
    plt.title('Final Training SAM Across Noise Levels')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for shape_name in shapes:
        plt.plot(noise_levels, results['test_sam'][shape_name], 'o-', 
                label=f"{shape_name.capitalize()}")
    plt.grid(True)
    plt.xlabel('Noise Level (SNR dB)')
    plt.ylabel('Final Test SAM (radians)')
    plt.title('Final Testing SAM Across Noise Levels')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'final_sam_across_noise.png'))
    plt.close()
    
    # Save numeric results
    for shape_name in shapes:
        np.save(os.path.join(results_dir, f'{shape_name}_train_loss.npy'), np.array(results['train_loss'][shape_name]))
        np.save(os.path.join(results_dir, f'{shape_name}_test_loss.npy'), np.array(results['test_loss'][shape_name]))
        np.save(os.path.join(results_dir, f'{shape_name}_train_psnr.npy'), np.array(results['train_psnr'][shape_name]))
        np.save(os.path.join(results_dir, f'{shape_name}_test_psnr.npy'), np.array(results['test_psnr'][shape_name]))
        np.save(os.path.join(results_dir, f'{shape_name}_train_sam.npy'), np.array(results['train_sam'][shape_name]))
        np.save(os.path.join(results_dir, f'{shape_name}_test_sam.npy'), np.array(results['test_sam'][shape_name]))
    
    # Save noise levels
    np.save(os.path.join(results_dir, 'noise_levels.npy'), np.array(noise_levels))
    
    # Create summary csv
    with open(os.path.join(results_dir, 'summary.csv'), 'w') as f:
        # Write header
        f.write('Shape,Noise_Level,Train_MSE,Test_MSE,Train_PSNR,Test_PSNR,Train_SAM,Test_SAM\n')
        
        # Write data
        for i, noise_level in enumerate(noise_levels):
            for shape_name in shapes:
                f.write(f"{shape_name},{noise_level},{results['train_loss'][shape_name][i]:.6f},"
                        f"{results['test_loss'][shape_name][i]:.6f},{results['train_psnr'][shape_name][i]:.2f},"
                        f"{results['test_psnr'][shape_name][i]:.2f},{results['train_sam'][shape_name][i]:.6f},"
                        f"{results['test_sam'][shape_name][i]:.6f}\n")
    
    return results

###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hyperspectral autoencoder with blind noise, reconstruction visualization, and noise level comparison")
    
    # Data processing arguments
    parser.add_argument('--cache', type=str, default="cache_filtered/aviris_tiles_forest.pt", 
                       help="Path to cache file for processed data")
    parser.add_argument('--use-cache', action='store_true', help="Use cached data if available")
    parser.add_argument('-f', '--folders', type=str, default="all", 
                       help="Comma-separated list of folder name patterns to include, or 'all' for all folders")
    
    # Model arguments
    parser.add_argument('--gpu', type=int, default=None, help="GPU ID to use (e.g., 0, 1, 2, 3)")
    parser.add_argument('--filter-scale', type=float, default=10.0, 
                       help="Scaling factor for filter normalization (default: 10.0)")
    
    # Noise arguments
    parser.add_argument('--min-snr', type=float, default=10, 
                       help="Minimum SNR level in dB for random noise (default: 10)")
    parser.add_argument('--max-snr', type=float, default=40, 
                       help="Maximum SNR level in dB for random noise (default: 40)")
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument('--epochs', type=int, default=50, 
                       help="Number of epochs for full training (default: 50)")
    parser.add_argument('--decoder-epochs', type=int, default=20, 
                       help="Number of epochs for decoder-only training (default: 20)")
    parser.add_argument('--encoder-lr', type=float, default=0.001, 
                       help="Learning rate for encoder (default: 0.001)")
    parser.add_argument('--decoder-lr', type=float, default=0.001, 
                       help="Learning rate for decoder (default: 0.001)")
    
    # Experiment control
    parser.add_argument('--skip-stage1', action='store_true', 
                       help="Skip stage 1 and load shapes from directory")
    parser.add_argument('--load-shapes-dir', type=str, default=None, 
                       help="Directory to load shapes from when skipping stage 1")
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default=None, 
                       help="Output directory (default: blind_noise_experiment_[timestamp])")
    
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
    
    # Check if the model paths exist
    shape2filter_path = "outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt"
    filter2shape_path = "outputs_three_stage_20250322_145925/stageC/spec2shape_stageC.pt"
    
    for model_path in [shape2filter_path, filter2shape_path]:
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            print("Searching for model file in current directory...")
            
            # Try to find the model file in the current directory structure
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.pt'):
                        if "shape2spec" in file and model_path == shape2filter_path:
                            shape2filter_path = os.path.join(root, file)
                            print(f"Using shape2filter model file: {shape2filter_path}")
                        elif "spec2shape" in file and model_path == filter2shape_path:
                            filter2shape_path = os.path.join(root, file)
                            print(f"Using filter2shape model file: {filter2shape_path}")
    
    # Create base output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = f"blind_noise_experiment_{folder_suffix}_{timestamp}"
    else:
        base_output_dir = args.output_dir
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create cache directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    
    # Save command-line arguments to output directory
    with open(os.path.join(base_output_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Set device for generating initial parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate a unified initial filter to use across all experiments
    initial_filter_params = generate_initial_filter(device)
    
    # Save the initial filter for reference
    np.save(f"{base_output_dir}/unified_initial_filter.npy", initial_filter_params.detach().cpu().numpy())
    
    # Load and split data
    print("Loading and splitting data into train/test sets...")
    data = load_aviris_forest_data(base_path="AVIRIS_FOREST_SIMPLE_SELECT", tile_size=128, 
                                  cache_file=cache_path, use_cache=args.use_cache, folder_patterns=folder_patterns)
    
    # Verify data is in BHWC format
    if data.shape[1] == 100:  # If in BCHW format
        data = data.permute(0, 2, 3, 1)  # Convert to BHWC
        print(f"Converted data to BHWC format: {data.shape}")
    
    # Split data into training and testing sets (80% train, 20% test)
    num_samples = data.shape[0]
    indices = torch.randperm(num_samples)
    train_size = int(0.8 * num_samples)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_data = data[train_indices]
    test_data = data[test_indices]
    
    print(f"Data split into {train_data.shape[0]} training and {test_data.shape[0]} testing samples")
    
    # Create data loaders for stage 2
    train_loader = DataLoader(
        TensorDataset(train_data), 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        TensorDataset(test_data), 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # Get shapes either from stage 1 or from provided directory
    shapes_dict = {}
    
    if args.skip_stage1:
        if args.load_shapes_dir is None:
            raise ValueError("Must provide --load-shapes-dir when using --skip-stage1")
        
        print(f"\n=== Skipping Stage 1, loading shapes from {args.load_shapes_dir} ===")
        
        # Try to load shapes from directory
        for shape_name in ['initial', 'lowest_mse', 'lowest_cn', 'final']:
            shape_path = os.path.join(args.load_shapes_dir, f"{shape_name}_shape.npy")
            if not os.path.exists(shape_path):
                # Try alternative path
                shape_path = os.path.join(args.load_shapes_dir, shape_name, "shape.npy")
            
            if os.path.exists(shape_path):
                shapes_dict[shape_name] = shape_path
                print(f"Found {shape_name} shape at: {shape_path}")
            else:
                print(f"Warning: Could not find {shape_name} shape")
        
        if not shapes_dict:
            raise ValueError(f"No shapes found in {args.load_shapes_dir}")
    
    else:
        # STAGE 1: Run training with random noise to generate shapes
        print(f"\n{'='*50}")
        print(f"STAGE 1: Training with random noise in range {args.min_snr} to {args.max_snr} dB")
        print(f"{'='*50}\n")
        
        initial_shape_path, lowest_mse_shape_path, lowest_cn_shape_path, final_shape_path, stage1_output_dir = train_with_random_noise(
            shape2filter_path=shape2filter_path,
            filter2shape_path=filter2shape_path,
            output_dir=base_output_dir,
            min_snr=args.min_snr,
            max_snr=args.max_snr,
            initial_filter_params=initial_filter_params,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            encoder_lr=args.encoder_lr,
            decoder_lr=args.decoder_lr,
            filter_scale_factor=args.filter_scale,
            cache_file=cache_path,
            use_cache=args.use_cache,
            folder_patterns=folder_patterns,
            train_data=train_data,
            test_data=test_data
        )
        
        # Create dictionary of shape paths for Stage 2
        shapes_dict = {
            'initial': initial_shape_path,
            'lowest_mse': lowest_mse_shape_path,
            'lowest_cn': lowest_cn_shape_path,
            'final': final_shape_path
        }
    
    # STAGE 2: Train with fixed shapes at different noise levels
    print(f"\n{'='*50}")
    print(f"STAGE 2: Training with fixed shapes at different noise levels")
    print(f"{'='*50}\n")
    
    # Define noise levels for fixed shape training
    fixed_noise_levels = [10, 20, 30, 40]
    
    # Run training for each shape at each noise level
    results = train_multiple_fixed_shapes(
        shapes_dict=shapes_dict,
        shape2filter_path=shape2filter_path,
        output_dir=base_output_dir,
        noise_levels=fixed_noise_levels,
        num_epochs=args.decoder_epochs,
        batch_size=args.batch_size,
        decoder_lr=args.decoder_lr,
        filter_scale_factor=args.filter_scale,
        train_loader=train_loader,
        test_loader=test_loader
    )
    
    print(f"\nAll experiments completed! Results saved to: {os.path.abspath(base_output_dir)}")

if __name__ == "__main__":
    main()