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

from noise_experiment_mixed_noise_base import (
    calculate_mse_in_batch,
    calculate_psnr_in_batch,
    calculate_sam_in_batch,
    calculate_metrics_in_batch,
    calculate_psnr, calculate_sam, visualize_reconstruction, visualize_reconstruction_spectrum
)

from AWAN import AWAN
latent_dim = 11
in_channels = 100

# def calculate_metrics_in_batch(original_data, model, batch_size=16, device=None, data_range=1.0, epsilon=1e-8, 
#                               desc="Calculating metrics", is_fixed_shape=False):
#     """
#     Calculate MSE, PSNR, and SAM metrics in batches with a single model forward pass.
    
#     Parameters:
#     original_data: Tensor containing original data
#     model: Model to use for reconstruction
#     batch_size: Size of batches to process
#     device: Device to use for computation (if None, will be detected from model)
#     data_range: The data range of the input image (default: 1.0 for normalized images)
#     epsilon: Small value to avoid division by zero for SAM calculation
#     desc: Description for the progress bar
    
#     Returns:
#     tuple: (mse, psnr, sam) values averaged across all samples
#     """
#     import torch
#     import numpy as np
#     from tqdm import tqdm
    
#     # Determine device if not provided
#     if device is None:
#         device = next(model.parameters()).device
    
#     # Ensure original data is a PyTorch tensor
#     if not isinstance(original_data, torch.Tensor):
#         original_data = torch.tensor(original_data)
    
#     # Check if data is already on the correct device
#     data_on_device = original_data.device == device
    
#     # Set model to evaluation mode
#     model.eval()
    
#     # Process in a single batch if data is small or already on device
#     if data_on_device or original_data.shape[0] <= batch_size:
#         with torch.no_grad():
#             # Single forward pass
#             if is_fixed_shape:
#                 reconstructed, _ = model(original_data)
#                 snr = 0
#             else:
#                 reconstructed, _, snr = model(original_data)
            
#             # Convert SNR to scalar if it's a tensor
#             if torch.is_tensor(snr):
#                 snr = snr.item()
            
#             # Move to CPU for calculations
#             original_cpu = original_data.cpu().numpy()
#             reconstructed_cpu = reconstructed.cpu().numpy()
            
#             # Calculate MSE
#             mse = np.mean((original_cpu - reconstructed_cpu) ** 2)
            
#             # Calculate PSNR
#             if mse == 0:
#                 psnr = float('inf')
#             else:
#                 psnr = 20 * np.log10(data_range / np.sqrt(mse))
            
#             # Calculate SAM
#             # Handle batch dimension if present
#             if len(original_cpu.shape) == 4:  # [B, H, W, C]
#                 # Reshape to [B*H*W, C]
#                 orig_reshaped = original_cpu.reshape(-1, original_cpu.shape[-1])
#                 recon_reshaped = reconstructed_cpu.reshape(-1, reconstructed_cpu.shape[-1])
#             else:  # [H, W, C]
#                 # Reshape to [H*W, C]
#                 orig_reshaped = original_cpu.reshape(-1, original_cpu.shape[-1])
#                 recon_reshaped = reconstructed_cpu.reshape(-1, reconstructed_cpu.shape[-1])
            
#             # Calculate dot product for each pixel
#             dot_product = np.sum(orig_reshaped * recon_reshaped, axis=1)
            
#             # Calculate magnitudes
#             orig_mag = np.sqrt(np.sum(orig_reshaped ** 2, axis=1))
#             recon_mag = np.sqrt(np.sum(recon_reshaped ** 2, axis=1))
            
#             # Avoid division by zero
#             valid_pixels = (orig_mag > epsilon) & (recon_mag > epsilon)
            
#             if not np.any(valid_pixels):
#                 sam = 0.0  # All pixels are zeros
#             else:
#                 # Calculate cosine similarity
#                 cos_sim = np.zeros_like(dot_product)
#                 cos_sim[valid_pixels] = dot_product[valid_pixels] / (orig_mag[valid_pixels] * recon_mag[valid_pixels])
                
#                 # Clip to [-1, 1] to handle numerical errors
#                 cos_sim = np.clip(cos_sim, -1.0, 1.0)
                
#                 # Calculate angle in radians
#                 angles = np.arccos(cos_sim)
                
#                 # Calculate mean angle
#                 sam = np.mean(angles)
            
#             return mse, psnr, sam
    
#     # Otherwise, process in batches
#     # Variables for MSE calculation
#     total_squared_diff = 0.0
#     total_samples = 0
    
#     # Variables for SAM calculation
#     all_angles = []
    
#     with torch.no_grad():
#         num_samples = original_data.shape[0]
#         for i in tqdm(range(0, num_samples, batch_size), desc=desc):
#             # Get batch
#             end_idx = min(i + batch_size, num_samples)
#             x = original_data[i:end_idx].to(device)
            
#             # Single forward pass for this batch
#             # reconstructed, _, _ = model(x)
#              # Single forward pass
#             if is_fixed_shape:
#                 reconstructed, _ = model(x)
#                 snr = 0
#             else:
#                 reconstructed, _, snr = model(x)
            
#             # MSE calculation for the batch
#             batch_squared_diff_sum = ((reconstructed - x) ** 2).sum().item()
#             batch_elements = x.numel()
#             total_squared_diff += batch_squared_diff_sum
#             total_samples += batch_elements
            
#             # Process each sample in the batch for SAM
#             for j in range(x.size(0)):
#                 # Get original and reconstructed for this sample
#                 orig = x[j].cpu().numpy()
#                 recon = reconstructed[j].cpu().numpy()
                
#                 # Reshape to [H*W, C]
#                 orig_reshaped = orig.reshape(-1, orig.shape[-1])
#                 recon_reshaped = recon.reshape(-1, recon.shape[-1])
                
#                 # Calculate dot product
#                 dot_product = np.sum(orig_reshaped * recon_reshaped, axis=1)
                
#                 # Calculate magnitudes
#                 orig_mag = np.sqrt(np.sum(orig_reshaped ** 2, axis=1))
#                 recon_mag = np.sqrt(np.sum(recon_reshaped ** 2, axis=1))
                
#                 # Avoid division by zero
#                 valid_pixels = (orig_mag > epsilon) & (recon_mag > epsilon)
                
#                 if not np.any(valid_pixels):
#                     sample_angles = np.zeros_like(dot_product)
#                 else:
#                     # Calculate cosine similarity
#                     cos_sim = np.zeros_like(dot_product)
#                     cos_sim[valid_pixels] = dot_product[valid_pixels] / (orig_mag[valid_pixels] * recon_mag[valid_pixels])
                    
#                     # Clip to [-1, 1] to handle numerical errors
#                     cos_sim = np.clip(cos_sim, -1.0, 1.0)
                    
#                     # Calculate angle in radians
#                     sample_angles = np.arccos(cos_sim)
                
#                 all_angles.append(sample_angles)
    
#     # Calculate final MSE
#     mse = total_squared_diff / total_samples
    
#     # Calculate PSNR
#     if mse == 0:
#         psnr = float('inf')
#     else:
#         psnr = 20 * np.log10(data_range / np.sqrt(mse))
    
#     # Calculate final SAM
#     all_angles = np.concatenate(all_angles)
#     sam = np.mean(all_angles)
    
#     return mse, psnr, sam


# def calculate_mse_in_batch(original_data, model, batch_size=16, device=None):
#     """
#     Calculate Mean Squared Error by processing data in batches.
    
#     Parameters:
#     original_data: Tensor containing original data
#     model: Model to use for reconstruction
#     batch_size: Size of batches to process
#     device: Device to use for computation (if None, will be detected from model)
    
#     Returns:
#     float: Average MSE across all samples
#     """
#     import torch
    
#     # Determine device if not provided
#     if device is None:
#         device = next(model.parameters()).device
    
#     # Ensure original data is a PyTorch tensor
#     if not isinstance(original_data, torch.Tensor):
#         original_data = torch.tensor(original_data)
    
#     # Check if data is already on the correct device
#     data_on_device = original_data.device == device
    
#     # Set model to evaluation mode
#     model.eval()
    
#     # Process in a single batch if data is small or already on device
#     if data_on_device or original_data.shape[0] <= batch_size:
#         with torch.no_grad():
#             reconstructed, _, _ = model(original_data)
#             mse = ((reconstructed - original_data) ** 2).mean().item()
#         return mse
    
#     # Otherwise, process in batches
#     total_mse = 0.0
#     total_samples = 0
    
#     with torch.no_grad():
#         num_samples = original_data.shape[0]
#         for i in tqdm(range(0, num_samples, batch_size)):
#             # Get batch
#             end_idx = min(i + batch_size, num_samples)
#             x = original_data[i:end_idx].to(device)
            
#             # Forward pass through model
#             reconstructed, _, _ = model(x)
            
#             # Calculate MSE for this batch
#             batch_mse = ((reconstructed - x) ** 2).mean().item()
            
#             # Accumulate weighted MSE (by batch size)
#             current_batch_size = x.size(0)
#             total_mse += batch_mse * current_batch_size
#             total_samples += current_batch_size
    
#     # Calculate average MSE across all samples
#     average_mse = total_mse / total_samples
    
#     return average_mse

# def calculate_psnr_in_batch(original_data, model, batch_size=16, device=None, data_range=1.0):
#     """
#     Calculate Peak Signal-to-Noise Ratio by processing data in batches.
    
#     Parameters:
#     original_data: Tensor containing original data
#     model: Model to use for reconstruction
#     batch_size: Size of batches to process
#     device: Device to use for computation (if None, will be detected from model)
#     data_range: The data range of the input image (default: 1.0 for normalized images)
    
#     Returns:
#     float: Average PSNR across all samples, matching the calculate_psnr function
#     """
#     import torch
#     import numpy as np
    
#     # Determine device if not provided
#     if device is None:
#         device = next(model.parameters()).device
    
#     # Ensure original data is a PyTorch tensor
#     if not isinstance(original_data, torch.Tensor):
#         original_data = torch.tensor(original_data)
    
#     # Check if data is already on the correct device
#     data_on_device = original_data.device == device
    
#     # Set model to evaluation mode
#     model.eval()
    
#     # Process in a single batch if data is small or already on device
#     if data_on_device or original_data.shape[0] <= batch_size:
#         with torch.no_grad():
#             reconstructed, _, _ = model(original_data)
            
#             # Move to CPU for consistency with the original calculate_psnr function
#             original_cpu = original_data.cpu().numpy()
#             reconstructed_cpu = reconstructed.cpu().numpy()
            
#             # Calculate MSE
#             mse = np.mean((original_cpu - reconstructed_cpu) ** 2)
            
#             # Avoid division by zero
#             if mse == 0:
#                 return float('inf')
            
#             # Calculate PSNR
#             psnr = 20 * np.log10(data_range / np.sqrt(mse))
            
#         return psnr
    
#     # Otherwise, process in batches
#     all_original = []
#     all_reconstructed = []
    
#     with torch.no_grad():
#         num_samples = original_data.shape[0]
#         for i in tqdm(range(0, num_samples, batch_size)):
#             # Get batch
#             end_idx = min(i + batch_size, num_samples)
#             x = original_data[i:end_idx].to(device)
            
#             # Forward pass through model
#             reconstructed, _, _ = model(x)
            
#             # Store original and reconstructed on CPU
#             all_original.append(x.cpu())
#             all_reconstructed.append(reconstructed.cpu())
    
#     # Concatenate all batches
#     all_original = torch.cat(all_original, dim=0).numpy()
#     all_reconstructed = torch.cat(all_reconstructed, dim=0).numpy()
    
#     # Calculate MSE
#     mse = np.mean((all_original - all_reconstructed) ** 2)
    
#     # Avoid division by zero
#     if mse == 0:
#         return float('inf')
    
#     # Calculate PSNR
#     psnr = 20 * np.log10(data_range / np.sqrt(mse))
    
#     return psnr

# def calculate_sam_in_batch(original_data, model, batch_size=16, device=None, epsilon=1e-8):
#     """
#     Calculate Spectral Angle Mapper by processing data in batches.
    
#     Parameters:
#     original_data: Tensor containing original data
#     model: Model to use for reconstruction
#     batch_size: Size of batches to process
#     device: Device to use for computation (if None, will be detected from model)
#     epsilon: Small value to avoid division by zero
    
#     Returns:
#     float: Average SAM across all samples, matching the calculate_sam function
#     """
#     import torch
#     import numpy as np
    
#     # Determine device if not provided
#     if device is None:
#         device = next(model.parameters()).device
    
#     # Ensure original data is a PyTorch tensor
#     if not isinstance(original_data, torch.Tensor):
#         original_data = torch.tensor(original_data)
    
#     # Check if data is already on the correct device
#     data_on_device = original_data.device == device
    
#     # Set model to evaluation mode
#     model.eval()
    
#     # Process in a single batch if data is small or already on device
#     if data_on_device or original_data.shape[0] <= batch_size:
#         with torch.no_grad():
#             reconstructed, _, _ = model(original_data)
            
#             # Move to CPU for calculation, matching the original calculate_sam function
#             orig_cpu = original_data.cpu().numpy()
#             recon_cpu = reconstructed.cpu().numpy()
            
#             # Calculate SAM using numpy
#             # Handle batch dimension if present
#             if len(orig_cpu.shape) == 4:  # [B, H, W, C]
#                 # Reshape to [B*H*W, C]
#                 orig_reshaped = orig_cpu.reshape(-1, orig_cpu.shape[-1])
#                 recon_reshaped = recon_cpu.reshape(-1, recon_cpu.shape[-1])
#             else:  # [H, W, C]
#                 # Reshape to [H*W, C]
#                 orig_reshaped = orig_cpu.reshape(-1, orig_cpu.shape[-1])
#                 recon_reshaped = recon_cpu.reshape(-1, recon_cpu.shape[-1])
            
#             # Calculate dot product for each pixel
#             dot_product = np.sum(orig_reshaped * recon_reshaped, axis=1)
            
#             # Calculate magnitudes
#             orig_mag = np.sqrt(np.sum(orig_reshaped ** 2, axis=1))
#             recon_mag = np.sqrt(np.sum(recon_reshaped ** 2, axis=1))
            
#             # Avoid division by zero
#             valid_pixels = (orig_mag > epsilon) & (recon_mag > epsilon)
            
#             if not np.any(valid_pixels):
#                 return 0.0  # All pixels are zeros
            
#             # Calculate cosine similarity
#             cos_sim = np.zeros_like(dot_product)
#             cos_sim[valid_pixels] = dot_product[valid_pixels] / (orig_mag[valid_pixels] * recon_mag[valid_pixels])
            
#             # Clip to [-1, 1] to handle numerical errors
#             cos_sim = np.clip(cos_sim, -1.0, 1.0)
            
#             # Calculate angle in radians
#             angles = np.arccos(cos_sim)
            
#             # Return mean angle
#             return np.mean(angles)
    
#     # Otherwise, process in batches
#     all_angles = []
    
#     with torch.no_grad():
#         num_samples = original_data.shape[0]
#         for i in tqdm(range(0, num_samples, batch_size)):
#             # Get batch
#             end_idx = min(i + batch_size, num_samples)
#             x = original_data[i:end_idx].to(device)
            
#             # Forward pass through model
#             reconstructed, _, _ = model(x)
            
#             # Process each sample in the batch
#             for j in range(x.size(0)):
#                 # Get original and reconstructed for this sample
#                 orig = x[j].cpu().numpy()
#                 recon = reconstructed[j].cpu().numpy()
                
#                 # Reshape to [H*W, C]
#                 orig_reshaped = orig.reshape(-1, orig.shape[-1])
#                 recon_reshaped = recon.reshape(-1, recon.shape[-1])
                
#                 # Calculate dot product
#                 dot_product = np.sum(orig_reshaped * recon_reshaped, axis=1)
                
#                 # Calculate magnitudes
#                 orig_mag = np.sqrt(np.sum(orig_reshaped ** 2, axis=1))
#                 recon_mag = np.sqrt(np.sum(recon_reshaped ** 2, axis=1))
                
#                 # Avoid division by zero
#                 valid_pixels = (orig_mag > epsilon) & (recon_mag > epsilon)
                
#                 if not np.any(valid_pixels):
#                     sample_angles = np.zeros_like(dot_product)
#                 else:
#                     # Calculate cosine similarity
#                     cos_sim = np.zeros_like(dot_product)
#                     cos_sim[valid_pixels] = dot_product[valid_pixels] / (orig_mag[valid_pixels] * recon_mag[valid_pixels])
                    
#                     # Clip to [-1, 1] to handle numerical errors
#                     cos_sim = np.clip(cos_sim, -1.0, 1.0)
                    
#                     # Calculate angle in radians
#                     sample_angles = np.arccos(cos_sim)
                
#                 all_angles.append(sample_angles)
    
#     # Concatenate all angles and calculate mean
#     all_angles = np.concatenate(all_angles)
#     return np.mean(all_angles)


# ###############################################################################
# # METRICS FUNCTIONS
# ###############################################################################
# def calculate_psnr(original, reconstructed, data_range=1.0):
#     """
#     Calculate Peak Signal-to-Noise Ratio (PSNR) between original and reconstructed images.
    
#     Parameters:
#     original: Original image tensor
#     reconstructed: Reconstructed image tensor
#     data_range: The data range of the input image (default: 1.0 for normalized images)
    
#     Returns:
#     float: PSNR value in dB
#     """
#     # Ensure tensors are on CPU and converted to numpy arrays
#     if isinstance(original, torch.Tensor):
#         original = original.detach().cpu().numpy()
#     if isinstance(reconstructed, torch.Tensor):
#         reconstructed = reconstructed.detach().cpu().numpy()
    
#     # Calculate MSE
#     mse = np.mean((original - reconstructed) ** 2)
    
#     # Avoid division by zero
#     if mse == 0:
#         return float('inf')
    
#     # Calculate PSNR
#     psnr = 20 * np.log10(data_range / np.sqrt(mse))
    
#     return psnr

# def calculate_sam(original, reconstructed, epsilon=1e-8):
#     """
#     Calculate Spectral Angle Mapper (SAM) between original and reconstructed images.
    
#     Parameters:
#     original: Original image tensor of shape [H, W, C] or [B, H, W, C]
#     reconstructed: Reconstructed image tensor of shape [H, W, C] or [B, H, W, C]
#     epsilon: Small value to avoid division by zero
    
#     Returns:
#     float: Mean SAM value in radians
#     """
#     # Ensure tensors are on CPU and converted to numpy arrays
#     if isinstance(original, torch.Tensor):
#         original = original.detach().cpu().numpy()
#     if isinstance(reconstructed, torch.Tensor):
#         reconstructed = reconstructed.detach().cpu().numpy()
    
#     # Handle batch dimension if present
#     if len(original.shape) == 4:  # [B, H, W, C]
#         # Reshape to [B*H*W, C]
#         orig_reshaped = original.reshape(-1, original.shape[-1])
#         recon_reshaped = reconstructed.reshape(-1, reconstructed.shape[-1])
#     else:  # [H, W, C]
#         # Reshape to [H*W, C]
#         orig_reshaped = original.reshape(-1, original.shape[-1])
#         recon_reshaped = reconstructed.reshape(-1, reconstructed.shape[-1])
    
#     # Calculate dot product for each pixel
#     dot_product = np.sum(orig_reshaped * recon_reshaped, axis=1)
    
#     # Calculate magnitudes
#     orig_mag = np.sqrt(np.sum(orig_reshaped ** 2, axis=1))
#     recon_mag = np.sqrt(np.sum(recon_reshaped ** 2, axis=1))
    
#     # Avoid division by zero
#     valid_pixels = (orig_mag > epsilon) & (recon_mag > epsilon)
    
#     if not np.any(valid_pixels):
#         return 0.0  # All pixels are zeros
    
#     # Calculate cosine similarity
#     cos_sim = np.zeros_like(dot_product)
#     cos_sim[valid_pixels] = dot_product[valid_pixels] / (orig_mag[valid_pixels] * recon_mag[valid_pixels])
    
#     # Clip to [-1, 1] to handle numerical errors
#     cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
#     # Calculate angle in radians
#     angles = np.arccos(cos_sim)
    
#     # Return mean angle
#     return np.mean(angles)

###############################################################################
# VISUALIZATION FUNCTIONS
###############################################################################


# def visualize_reconstruction(model, sample_tensor, device, save_path, band_idx=50):
#     """
#     Create visualization of original, reconstructed, and difference images with unified colorbar
#     for multiple samples and multiple spectral bands.
    
#     Parameters:
#     model: Model to use for reconstruction
#     sample_tensor: Input tensor to reconstruct (can handle multiple samples)
#     device: Device to use for computation
#     save_path: Path to save the visualization
#     band_idx: Index of spectral band to visualize (default: 50, middle of 100 bands)
#                 (Will be used as one of the bands in multi-band visualization)
    
#     Returns:
#     tuple: (mse, psnr, sam) values averaged across all samples
#     """
#     # Define bands to visualize (including the specified band_idx)
#     band_indices = [5, 25, 50, 75, 95]
#     if band_idx not in band_indices:
#         band_indices.append(band_idx)
#         band_indices.sort()
    
#     with torch.no_grad():
#         # Ensure sample is on the correct device
#         sample = sample_tensor.to(device)
        
#         # Get reconstruction
#         recon, _, snr = model(sample)
        
#         # Convert SNR to a scalar if it's a tensor
#         if torch.is_tensor(snr):
#             snr = snr.item()
        
#         # Move tensors to CPU for visualization
#         sample_np = sample.cpu().numpy()  # Shape: [batch_size, H, W, C]
#         recon_np = recon.cpu().numpy()    # Shape: [batch_size, H, W, C]
        
#         # Calculate difference
#         diff_np = sample_np - recon_np
        
#         # Initialize metrics storage
#         all_mse = []
#         all_psnr = []
#         all_sam = []
        
#         # Process each sample
#         for sample_idx in range(sample_np.shape[0]):
#             # Calculate metrics for this sample
#             mse = ((sample_np[sample_idx] - recon_np[sample_idx]) ** 2).mean()
#             psnr = calculate_psnr(sample_np[sample_idx], recon_np[sample_idx])
#             sam = calculate_sam(sample_np[sample_idx], recon_np[sample_idx])
            
#             # Store metrics
#             all_mse.append(mse)
#             all_psnr.append(psnr)
#             all_sam.append(sam)
            
#             # Create a figure for this sample with multiple bands
#             plt.figure(figsize=(15, 5 * len(band_indices)))
            
#             # Process each band
#             for i, b_idx in enumerate(band_indices):
#                 # Extract specific spectral band
#                 sample_band = sample_np[sample_idx, :, :, b_idx]
#                 recon_band = recon_np[sample_idx, :, :, b_idx]
#                 diff_band = diff_np[sample_idx, :, :, b_idx]
                
#                 # Calculate global min and max for unified colormap
#                 vmin = min(sample_band.min(), recon_band.min())
#                 vmax = max(sample_band.max(), recon_band.max())
                
#                 # Calculate symmetric limits for difference
#                 diff_abs_max = max(abs(diff_band.min()), abs(diff_band.max()))
                
#                 # Plot original
#                 plt.subplot(len(band_indices), 3, 3*i + 1)
#                 im1 = plt.imshow(sample_band, cmap='viridis', vmin=vmin, vmax=vmax)
#                 plt.title(f'Original (Band {b_idx})')
#                 plt.colorbar(im1, fraction=0.046, pad=0.04)
                
#                 # Plot reconstruction
#                 plt.subplot(len(band_indices), 3, 3*i + 2)
#                 im2 = plt.imshow(recon_band, cmap='viridis', vmin=vmin, vmax=vmax)
#                 plt.title(f'Reconstructed (Band {b_idx})')
#                 plt.colorbar(im2, fraction=0.046, pad=0.04)
                
#                 # Plot difference
#                 plt.subplot(len(band_indices), 3, 3*i + 3)
#                 im3 = plt.imshow(diff_band, cmap='coolwarm', vmin=-diff_abs_max, vmax=diff_abs_max)
#                 plt.title(f'Difference (Band {b_idx})')
#                 plt.colorbar(im3, fraction=0.046, pad=0.04)
            
#             # Add SNR information as a suptitle
#             plt.suptitle(f'Sample {sample_idx+1} - SNR: {snr:.2f} dB, PSNR: {psnr:.2f} dB, SAM: {sam:.4f} rad', fontsize=16)
            
#             # Adjust layout and save
#             plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
            
#             # Create directory structure if needed
#             os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            
#             # Save figure and data
#             sample_save_path = f"{save_path.rsplit('.', 1)[0]}_sample_{sample_idx+1}.png"
#             plt.savefig(sample_save_path, dpi=300, bbox_inches='tight')
#             plt.close()
            
#             # Save plot data for future reference
#             plot_data = {
#                 'bands': {},
#                 'metrics': {
#                     'mse': float(mse),
#                     'psnr': float(psnr),
#                     'sam': float(sam),
#                     'snr': float(snr)
#                 }
#             }
            
#             for b_idx in band_indices:
#                 plot_data['bands'][str(b_idx)] = {
#                     'original': sample_np[sample_idx, :, :, b_idx].tolist(),
#                     'reconstructed': recon_np[sample_idx, :, :, b_idx].tolist(),
#                     'difference': diff_np[sample_idx, :, :, b_idx].tolist()
#                 }
            
#             data_save_path = f"{save_path.rsplit('.', 1)[0]}_sample_{sample_idx+1}_data.json"
#             with open(data_save_path, 'w') as f:
#                 json.dump(plot_data, f)
        
#         # For backward compatibility, return averaged metrics
#         avg_mse = sum(all_mse) / len(all_mse)
#         avg_psnr = sum(all_psnr) / len(all_psnr)
#         avg_sam = sum(all_sam) / len(all_sam)
        
#         return avg_mse, avg_psnr, avg_sam

# def visualize_reconstruction(model, sample_tensor, device, save_path, band_idx=50):
#     """
#     Create visualization of original, reconstructed, and difference images with unified colorbar
#     for the first, middle, and last samples, and multiple spectral bands.
#     Uses batch computation for metrics.
    
#     Parameters:
#     model: Model to use for reconstruction
#     sample_tensor: Input tensor to reconstruct (can handle multiple samples)
#     device: Device to use for computation
#     save_path: Path to save the visualization
#     band_idx: Index of spectral band to visualize (default: 50, middle of 100 bands)
#                 (Will be used as one of the bands in multi-band visualization)
    
#     Returns:
#     tuple: (mse, psnr, sam) values averaged across all samples
#     """
#     # Define bands to visualize (including the specified band_idx)
#     band_indices = [5, 25, 50, 75, 95]
#     if band_idx not in band_indices:
#         band_indices.append(band_idx)
#         band_indices.sort()
    
#     # # Calculate metrics using batch functions
#     # mse = calculate_mse_in_batch(sample_tensor, model, device=device)
#     # psnr = calculate_psnr_in_batch(sample_tensor, model, device=device)
#     # sam = calculate_sam_in_batch(sample_tensor, model, device=device)
#     mse, psnr, sam = calculate_metrics_in_batch(sample_tensor, model, device=device)
    
#     with torch.no_grad():
#         # Ensure sample is on the correct device
#         batch_size = sample_tensor.shape[0]
        
#         # Get reconstruction for visualization of select samples
#         # Select first, middle and last samples for visualization
#         if batch_size <= 3:
#             # If we have 3 or fewer samples, just use all of them
#             indices_to_visualize = list(range(batch_size))
#         else:
#             # Otherwise, use first, middle, and last
#             indices_to_visualize = [0, batch_size // 2, batch_size - 1]
        
#         # Extract samples to visualize
#         samples_to_visualize = sample_tensor[indices_to_visualize].to(device)
        
#         # Get reconstruction
#         recon, _, snr = model(samples_to_visualize)
        
#         # Convert SNR to a scalar if it's a tensor
#         if torch.is_tensor(snr):
#             snr = snr.item()
        
#         # Move tensors to CPU for visualization
#         sample_np = samples_to_visualize.cpu().numpy()  # Shape: [vis_batch_size, H, W, C]
#         recon_np = recon.cpu().numpy()                 # Shape: [vis_batch_size, H, W, C]
        
#         # Calculate difference
#         diff_np = sample_np - recon_np
        
#         # Initialize metrics storage for the samples we visualize
#         vis_sample_metrics = []
        
#         # Process each sample for visualization
#         for i, sample_idx in enumerate(indices_to_visualize):
#             # Calculate per-sample metrics for display
#             sample_mse = ((sample_np[i] - recon_np[i]) ** 2).mean()
#             sample_psnr = calculate_psnr(sample_np[i], recon_np[i])
#             sample_sam = calculate_sam(sample_np[i], recon_np[i])
            
#             vis_sample_metrics.append({
#                 'mse': float(sample_mse),
#                 'psnr': float(sample_psnr),
#                 'sam': float(sample_sam),
#                 'snr': float(snr)
#             })
            
#             # Create a figure for this sample with multiple bands
#             plt.figure(figsize=(15, 5 * len(band_indices)))
            
#             # Process each band
#             for j, b_idx in enumerate(band_indices):
#                 # Extract specific spectral band
#                 sample_band = sample_np[i, :, :, b_idx]
#                 recon_band = recon_np[i, :, :, b_idx]
#                 diff_band = diff_np[i, :, :, b_idx]
                
#                 # Calculate global min and max for unified colormap
#                 vmin = min(sample_band.min(), recon_band.min())
#                 vmax = max(sample_band.max(), recon_band.max())
                
#                 # Calculate symmetric limits for difference
#                 diff_abs_max = max(abs(diff_band.min()), abs(diff_band.max()))
                
#                 # Plot original
#                 plt.subplot(len(band_indices), 3, 3*j + 1)
#                 im1 = plt.imshow(sample_band, cmap='viridis', vmin=vmin, vmax=vmax)
#                 plt.title(f'Original (Band {b_idx})')
#                 plt.colorbar(im1, fraction=0.046, pad=0.04)
                
#                 # Plot reconstruction
#                 plt.subplot(len(band_indices), 3, 3*j + 2)
#                 im2 = plt.imshow(recon_band, cmap='viridis', vmin=vmin, vmax=vmax)
#                 plt.title(f'Reconstructed (Band {b_idx})')
#                 plt.colorbar(im2, fraction=0.046, pad=0.04)
                
#                 # Plot difference
#                 plt.subplot(len(band_indices), 3, 3*j + 3)
#                 im3 = plt.imshow(diff_band, cmap='coolwarm', vmin=-diff_abs_max, vmax=diff_abs_max)
#                 plt.title(f'Difference (Band {b_idx})')
#                 plt.colorbar(im3, fraction=0.046, pad=0.04)
            
#             # Add SNR information as a suptitle
#             plt.suptitle(f'Sample {sample_idx+1} of {batch_size} - SNR: {snr:.2f} dB, PSNR: {sample_psnr:.2f} dB, SAM: {sample_sam:.4f} rad', fontsize=16)
            
#             # Adjust layout and save
#             plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
            
#             # Create directory structure if needed
#             os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            
#             # Save figure and data
#             sample_save_path = f"{save_path.rsplit('.', 1)[0]}_sample_{sample_idx+1}.png"
#             plt.savefig(sample_save_path, dpi=300, bbox_inches='tight')
#             plt.close()
            
#             # Save plot data for future reference
#             plot_data = {
#                 'bands': {},
#                 'metrics': vis_sample_metrics[i]
#             }
            
#             for b_idx in band_indices:
#                 plot_data['bands'][str(b_idx)] = {
#                     'original': sample_np[i, :, :, b_idx].tolist(),
#                     'reconstructed': recon_np[i, :, :, b_idx].tolist(),
#                     'difference': diff_np[i, :, :, b_idx].tolist()
#                 }
            
#             data_save_path = f"{save_path.rsplit('.', 1)[0]}_sample_{sample_idx+1}_data.json"
#             with open(data_save_path, 'w') as f:
#                 json.dump(plot_data, f)
        
#         # Save overall metrics
#         overall_metrics = {
#             'mse': float(mse),
#             'psnr': float(psnr),
#             'sam': float(sam),
#             'snr': float(snr),
#             'num_samples': batch_size,
#             'visualized_indices': indices_to_visualize.copy()
#         }
        
#         metrics_save_path = f"{save_path.rsplit('.', 1)[0]}_overall_metrics.json"
#         with open(metrics_save_path, 'w') as f:
#             json.dump(overall_metrics, f)
            
#         # Return overall metrics for backward compatibility
#         return mse, psnr, sam

# def visualize_reconstruction_spectrum(model, sample_tensor, device, save_path, band_idx=None):
#     """
#     Create visualization of original vs reconstructed spectra for selected pixels.
#     Maintains the same API as visualize_reconstruction for easy integration.
    
#     Parameters:
#     model: Model to use for reconstruction
#     sample_tensor: Input tensor to reconstruct (can handle multiple samples)
#     device: Device to use for computation
#     save_path: Path to save the visualization
#     band_idx: Not used here, included for API compatibility
    
#     Returns:
#     tuple: (mse, psnr, sam) values averaged across all samples
#     """
#     # Import common functions from the original script
#     # from calculate_psnr import calculate_psnr
#     # from calculate_sam import calculate_sam
    
#     # Make sure the spectrum directory exists
#     spectrum_dir = os.path.join(os.path.dirname(save_path), "reconstruction_spectrum")
#     os.makedirs(spectrum_dir, exist_ok=True)
    
#     with torch.no_grad():
#         # Ensure sample is on the correct device
#         sample = sample_tensor.to(device)
        
#         # Get reconstruction
#         recon, _, snr = model(sample)
        
#         # Convert SNR to a scalar if it's a tensor
#         if torch.is_tensor(snr):
#             snr = snr.item()
        
#         # Move tensors to CPU for visualization
#         sample_np = sample.cpu().numpy()  # Shape: [batch_size, H, W, C]
#         recon_np = recon.cpu().numpy()    # Shape: [batch_size, H, W, C]
        
#         # Initialize metrics storage
#         all_mse = []
#         all_psnr = []
#         all_sam = []
        
#         # Process each sample
#         for sample_idx in range(sample_np.shape[0]):
#             # Calculate metrics for this sample
#             mse = ((sample_np[sample_idx] - recon_np[sample_idx]) ** 2).mean()
#             psnr = calculate_psnr(sample_np[sample_idx], recon_np[sample_idx])
#             sam = calculate_sam(sample_np[sample_idx], recon_np[sample_idx])
            
#             # Store metrics
#             all_mse.append(mse)
#             all_psnr.append(psnr)
#             all_sam.append(sam)
            
#             # Get image dimensions for this sample
#             h, w = sample_np[sample_idx].shape[0:2]
            
#             # # Set pixel coordinates (5 pixels in different regions)
#             # pixel_coordinates = [
#             #     (h//4, w//4),        # Top-left quadrant
#             #     (h//4, 3*w//4),      # Top-right quadrant
#             #     (3*h//4, w//4),      # Bottom-left quadrant
#             #     (3*h//4, 3*w//4),    # Bottom-right quadrant
#             #     (h//2, w//2)         # Center
#             # ]

#             # Set pixel coordinates (9 pixels in different regions)
#             pixel_coordinates = [
#                 (h//4, w//4),        # Top-left quadrant
#                 (h//4, w//2),        # Top-middle
#                 (h//4, 3*w//4),      # Top-right quadrant
#                 (h//2, w//4),        # Left-middle
#                 (h//2, w//2),        # Center
#                 (h//2, 3*w//4),      # Right-middle
#                 (3*h//4, w//4),      # Bottom-left quadrant
#                 (3*h//4, w//2),      # Bottom-middle
#                 (3*h//4, 3*w//4)     # Bottom-right quadrant
#             ]
            
#             # Number of spectral bands
#             num_bands = sample_np[sample_idx].shape[-1]
#             band_indices = np.arange(num_bands)
            
#             # Create a combined plot with all pixels
#             plt.figure(figsize=(12, 8))
#             plt.title(f'Sample {sample_idx+1} - All Pixels - SNR: {snr:.2f} dB, PSNR: {psnr:.2f} dB, SAM: {sam:.4f} rad')
            
#             # Colors for different pixels
#             colors = ['blue', 'green', 'red', 'purple', 'orange']
            
#             # Plot each pixel's spectrum
#             for i, (y, x) in enumerate(pixel_coordinates):
#                 color = colors[i % len(colors)]
                
#                 # Extract spectra
#                 original_spectrum = sample_np[sample_idx, y, x, :]
#                 recon_spectrum = recon_np[sample_idx, y, x, :]
                
#                 # Plot original (solid) and reconstructed (dashed)
#                 plt.plot(band_indices, original_spectrum, color=color, linestyle='-', 
#                          label=f'Original Pixel ({y},{x})')
#                 plt.plot(band_indices, recon_spectrum, color=color, linestyle='--', 
#                          label=f'Recon Pixel ({y},{x})')
            
#             plt.xlabel('Spectral Band')
#             plt.ylabel('Value')
#             plt.legend()
#             plt.grid(True, alpha=0.3)
            
#             # Save combined plot
#             base_filename = os.path.basename(save_path).rsplit('.', 1)[0]
#             combined_path = os.path.join(spectrum_dir, f"{base_filename}_sample_{sample_idx+1}_combined.png")
#             plt.savefig(combined_path, dpi=300, bbox_inches='tight')
#             plt.close()
            
#             # Create individual plots for each pixel (in a single figure with subplots)
#             num_pixels = len(pixel_coordinates)
#             fig, axs = plt.subplots(num_pixels, 1, figsize=(10, 3*num_pixels), sharex=True)
            
#             if num_pixels == 1:
#                 axs = [axs]  # Make iterable if only one subplot
                
#             for i, (y, x) in enumerate(pixel_coordinates):
#                 color = colors[i % len(colors)]
                
#                 # Extract spectra
#                 original_spectrum = sample_np[sample_idx, y, x, :]
#                 recon_spectrum = recon_np[sample_idx, y, x, :]
                
#                 # Calculate error
#                 error = original_spectrum - recon_spectrum
                
#                 # Plot on corresponding subplot
#                 axs[i].plot(band_indices, original_spectrum, color=color, linestyle='-', label='Original')
#                 axs[i].plot(band_indices, recon_spectrum, color=color, linestyle='--', label='Reconstructed')
#                 axs[i].set_title(f'Pixel ({y},{x})')
#                 axs[i].set_ylabel('Value')
#                 axs[i].grid(True, alpha=0.3)
#                 axs[i].legend()
                
#                 # Add error as small subplot or secondary axis
#                 ax2 = axs[i].twinx()
#                 ax2.plot(band_indices, error, color='gray', alpha=0.7, linestyle=':')
#                 ax2.set_ylabel('Error', color='gray')
#                 ax2.tick_params(axis='y', labelcolor='gray')
            
#             # Set common x-label
#             fig.text(0.5, 0.04, 'Spectral Band', ha='center')
            
#             # Set overall title
#             fig.suptitle(f'Sample {sample_idx+1} - Individual Pixel Spectra - SNR: {snr:.2f} dB', fontsize=16)
            
#             plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for suptitle
            
#             # Save individual plots
#             individual_path = os.path.join(spectrum_dir, f"{base_filename}_sample_{sample_idx+1}_individual.png")
#             plt.savefig(individual_path, dpi=300, bbox_inches='tight')
#             plt.close()
            
#             # Save plot data
#             plot_data = {
#                 'metrics': {
#                     'mse': float(mse),
#                     'psnr': float(psnr),
#                     'sam': float(sam),
#                     'snr': float(snr)
#                 },
#                 'pixels': {}
#             }
            
#             for i, (y, x) in enumerate(pixel_coordinates):
#                 plot_data['pixels'][f"pixel_{y}_{x}"] = {
#                     'original': sample_np[sample_idx, y, x, :].tolist(),
#                     'reconstructed': recon_np[sample_idx, y, x, :].tolist(),
#                     'error': (sample_np[sample_idx, y, x, :] - recon_np[sample_idx, y, x, :]).tolist()
#                 }
            
#             data_path = os.path.join(spectrum_dir, f"{base_filename}_sample_{sample_idx+1}_spectrum_data.json")
#             with open(data_path, 'w') as f:
#                 json.dump(plot_data, f)
        
#         # For backward compatibility, return averaged metrics
#         avg_mse = sum(all_mse) / len(all_mse)
#         avg_psnr = sum(all_psnr) / len(all_psnr)
#         avg_sam = sum(all_sam) / len(all_sam)
        
#         return avg_mse, avg_psnr, avg_sam

# def visualize_reconstruction_spectrum(model, sample_tensor, device, save_path, band_idx=None):
#     """
#     Create visualization of original vs reconstructed spectra for selected pixels.
#     Only visualizes first, middle, and last sample (or all samples if ≤ 3).
#     Uses batch computation for metrics.
    
#     Parameters:
#     model: Model to use for reconstruction
#     sample_tensor: Input tensor to reconstruct (can handle multiple samples)
#     device: Device to use for computation
#     save_path: Path to save the visualization
#     band_idx: Not used here, included for API compatibility
    
#     Returns:
#     tuple: (mse, psnr, sam) values averaged across all samples
#     """
#     # Make sure the spectrum directory exists
#     spectrum_dir = os.path.join(os.path.dirname(save_path), "reconstruction_spectrum")
#     os.makedirs(spectrum_dir, exist_ok=True)
    
#     # # Calculate metrics for ALL samples using batch functions
#     # mse = calculate_mse_in_batch(sample_tensor, model, device=device)
#     # psnr = calculate_psnr_in_batch(sample_tensor, model, device=device)
#     # sam = calculate_sam_in_batch(sample_tensor, model, device=device)
#     # Calculate all metrics with a single function call (one model pass)
#     # mse, psnr, sam = calculate_metrics_in_batch(sample_tensor, model, device=device)
#     mse, psnr, sam = 0, 0, 0
    
#     with torch.no_grad():
#         # Get batch size
#         batch_size = sample_tensor.shape[0]
        
#         # Determine which samples to visualize (first, middle, last)
#         if batch_size <= 3:
#             # If we have 3 or fewer samples, visualize all of them
#             indices_to_visualize = list(range(batch_size))
#         else:
#             # Otherwise, use first, middle, and last
#             indices_to_visualize = [0, batch_size // 2, batch_size - 1]
        
#         # Extract samples to visualize
#         samples_to_visualize = sample_tensor[indices_to_visualize].to(device)
        
#         # Get reconstruction only for samples to visualize
#         recon, _, snr = model(samples_to_visualize)
        
#         # Convert SNR to a scalar if it's a tensor
#         if torch.is_tensor(snr):
#             snr = snr.item()
        
#         # Move tensors to CPU for visualization
#         sample_np = samples_to_visualize.cpu().numpy()  # Shape: [num_vis_samples, H, W, C]
#         recon_np = recon.cpu().numpy()                  # Shape: [num_vis_samples, H, W, C]
        
#         # Initialize metrics storage for visualized samples
#         vis_sample_metrics = []
        
#         # Process each sample for visualization
#         for i, sample_idx in enumerate(indices_to_visualize):
#             # Calculate per-sample metrics for display purposes
#             sample_mse = ((sample_np[i] - recon_np[i]) ** 2).mean()
#             sample_psnr = calculate_psnr(sample_np[i], recon_np[i])
#             sample_sam = calculate_sam(sample_np[i], recon_np[i])
            
#             # Store this sample's metrics
#             vis_sample_metrics.append({
#                 'mse': float(sample_mse),
#                 'psnr': float(sample_psnr),
#                 'sam': float(sample_sam),
#                 'snr': float(snr)
#             })
            
#             # Get image dimensions for this sample
#             h, w = sample_np[i].shape[0:2]
            
#             # Set pixel coordinates (9 pixels in different regions)
#             pixel_coordinates = [
#                 (h//4, w//4),        # Top-left quadrant
#                 (h//4, w//2),        # Top-middle
#                 (h//4, 3*w//4),      # Top-right quadrant
#                 (h//2, w//4),        # Left-middle
#                 (h//2, w//2),        # Center
#                 (h//2, 3*w//4),      # Right-middle
#                 (3*h//4, w//4),      # Bottom-left quadrant
#                 (3*h//4, w//2),      # Bottom-middle
#                 (3*h//4, 3*w//4)     # Bottom-right quadrant
#             ]
            
#             # Number of spectral bands
#             num_bands = sample_np[i].shape[-1]
#             band_indices = np.arange(num_bands)
            
#             # Create a combined plot with all pixels
#             plt.figure(figsize=(12, 8))
#             plt.title(f'Sample {sample_idx+1} of {batch_size} - All Pixels - SNR: {snr:.2f} dB, PSNR: {sample_psnr:.2f} dB, SAM: {sample_sam:.4f} rad')
            
#             # Colors for different pixels
#             colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
            
#             # Plot each pixel's spectrum
#             for j, (y, x) in enumerate(pixel_coordinates):
#                 color = colors[j % len(colors)]
                
#                 # Extract spectra
#                 original_spectrum = sample_np[i, y, x, :]
#                 recon_spectrum = recon_np[i, y, x, :]
                
#                 # Plot original (solid) and reconstructed (dashed)
#                 plt.plot(band_indices, original_spectrum, color=color, linestyle='-', 
#                          label=f'Original Pixel ({y},{x})')
#                 plt.plot(band_indices, recon_spectrum, color=color, linestyle='--', 
#                          label=f'Recon Pixel ({y},{x})')
            
#             plt.xlabel('Spectral Band')
#             plt.ylabel('Value')
#             plt.legend()
#             plt.grid(True, alpha=0.3)
            
#             # Save combined plot
#             base_filename = os.path.basename(save_path).rsplit('.', 1)[0]
#             combined_path = os.path.join(spectrum_dir, f"{base_filename}_sample_{sample_idx+1}_combined.png")
#             plt.savefig(combined_path, dpi=300, bbox_inches='tight')
#             plt.close()
            
#             # Create individual plots for each pixel (in a single figure with subplots)
#             num_pixels = len(pixel_coordinates)
#             fig, axs = plt.subplots(num_pixels, 1, figsize=(10, 3*num_pixels), sharex=True)
            
#             if num_pixels == 1:
#                 axs = [axs]  # Make iterable if only one subplot
                
#             for j, (y, x) in enumerate(pixel_coordinates):
#                 color = colors[j % len(colors)]
                
#                 # Extract spectra
#                 original_spectrum = sample_np[i, y, x, :]
#                 recon_spectrum = recon_np[i, y, x, :]
                
#                 # Calculate error
#                 error = original_spectrum - recon_spectrum
                
#                 # Plot on corresponding subplot
#                 axs[j].plot(band_indices, original_spectrum, color=color, linestyle='-', label='Original')
#                 axs[j].plot(band_indices, recon_spectrum, color=color, linestyle='--', label='Reconstructed')
#                 axs[j].set_title(f'Pixel ({y},{x})')
#                 axs[j].set_ylabel('Value')
#                 axs[j].grid(True, alpha=0.3)
#                 axs[j].legend()
                
#                 # Add error as secondary axis
#                 ax2 = axs[j].twinx()
#                 ax2.plot(band_indices, error, color='gray', alpha=0.7, linestyle=':')
#                 ax2.set_ylabel('Error', color='gray')
#                 ax2.tick_params(axis='y', labelcolor='gray')
            
#             # Set common x-label
#             fig.text(0.5, 0.04, 'Spectral Band', ha='center')
            
#             # Set overall title
#             fig.suptitle(f'Sample {sample_idx+1} of {batch_size} - Individual Pixel Spectra - SNR: {snr:.2f} dB', fontsize=16)
            
#             plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for suptitle
            
#             # Save individual plots
#             individual_path = os.path.join(spectrum_dir, f"{base_filename}_sample_{sample_idx+1}_individual.png")
#             plt.savefig(individual_path, dpi=300, bbox_inches='tight')
#             plt.close()
            
#             # Save plot data
#             plot_data = {
#                 'metrics': vis_sample_metrics[i],
#                 'pixels': {}
#             }
            
#             for j, (y, x) in enumerate(pixel_coordinates):
#                 plot_data['pixels'][f"pixel_{y}_{x}"] = {
#                     'original': sample_np[i, y, x, :].tolist(),
#                     'reconstructed': recon_np[i, y, x, :].tolist(),
#                     'error': (sample_np[i, y, x, :] - recon_np[i, y, x, :]).tolist()
#                 }
            
#             data_path = os.path.join(spectrum_dir, f"{base_filename}_sample_{sample_idx+1}_spectrum_data.json")
#             with open(data_path, 'w') as f:
#                 json.dump(plot_data, f)
        
#         # Save overall metrics for all samples
#         overall_metrics = {
#             'mse': float(mse),
#             'psnr': float(psnr),
#             'sam': float(sam),
#             'snr': float(snr),
#             'num_samples': batch_size,
#             'visualized_indices': indices_to_visualize
#         }
        
#         metrics_save_path = os.path.join(spectrum_dir, f"{base_filename}_overall_spectrum_metrics.json")
#         with open(metrics_save_path, 'w') as f:
#             json.dump(overall_metrics, f)
        
#         # For backward compatibility, return metrics for all samples
#         return mse, psnr, sam

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
    
        # Create a filter mask that only keeps filters 1, 3, 5, 7, 9, 11 (indices 0, 2, 4, 6, 8, 10)
        self.filter_mask = torch.zeros(11, 100, device=self.device)
        self.filter_mask[0] = 1.0
        self.filter_mask[2] = 1.0
        self.filter_mask[4] = 1.0
        self.filter_mask[6] = 1.0
        self.filter_mask[8] = 1.0
        self.filter_mask[10] = 1.0

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
            _, recon_filter = self.pipeline(recon_filter)
        return recon_filter[0]  # Remove batch dimension

    def get_reconstructed_filter_with_grad(self):
        """Get the reconstructed filter from the full pipeline"""
        filter = self.get_current_filter().unsqueeze(0)  # Add batch dimension
        _, recon_filter = self.pipeline(filter)
        _, recon_filter = self.pipeline(recon_filter)
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
        
        # Apply the filter mask to selectively use only certain filters
        filter = filter * self.filter_mask
        

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
    def __init__(self, shape, shape2filter_path, noise_level=30, min_snr=10, max_snr=40, filter_scale_factor=10.0, device=None):
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
    
        # Create a filter mask that only keeps filters 1, 3, 5, 7, 9, 11 (indices 0, 2, 4, 6, 8, 10)
        self.filter_mask = torch.zeros(11, 100, device=self.device)
        self.filter_mask[0] = 1.0
        self.filter_mask[2] = 1.0
        self.filter_mask[4] = 1.0
        self.filter_mask[6] = 1.0
        self.filter_mask[8] = 1.0
        self.filter_mask[10] = 1.0
        
        print(f"Initialized HyperspectralAutoencoderRandomNoise with filter mask keeping filters 1,3,5,7,9,11")


    def add_fixed_noise(self, z):
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
        # filter_normalized = self.fixed_filter / self.filter_scale_factor
        # Apply the fixed filter with mask
        masked_filter = self.fixed_filter * self.filter_mask
        filter_normalized = masked_filter / self.filter_scale_factor
        encoded_channels_first = torch.einsum('bchw,oc->bohw', x_channels_first, filter_normalized)
        
        # Add noise if specified
        if add_noise:
            encoded_channels_first = self.add_random_noise(encoded_channels_first)
        else:
            encoded_channels_first = self.add_fixed_noise(encoded_channels_first)

        # Decode
        decoded_channels_first = self.decoder(encoded_channels_first)
        
        # Convert back to BHWC format
        encoded = encoded_channels_first.permute(0, 2, 3, 1)
        decoded = decoded_channels_first.permute(0, 2, 3, 1)
        
        return decoded, encoded