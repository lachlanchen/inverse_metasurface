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
        self.decoder = AWAN(inplanes=latent_dim, planes=in_channels, channels=128, n_DRBs=4)
    
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
        self.decoder = AWAN(inplanes=latent_dim, planes=in_channels, channels=128, n_DRBs=4)
    
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