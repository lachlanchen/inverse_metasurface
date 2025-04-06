#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
from tqdm import tqdm
import json
import pandas as pd
import glob

# Import common functions and classes
from noise_experiment_mixed_noise_base import (
    calculate_psnr,
    calculate_sam,
    calculate_metrics_in_batch,
    FixedShapeModel,
    AWAN
)

from noise_experiment_with_blind_noise import (
    Shape2FilterModel,
    load_aviris_forest_data
)


def calculate_condition_number(filter_matrix):
    """
    Calculate the condition number of the filter matrix
    
    Parameters:
    filter_matrix: Filter matrix to calculate condition number for
    
    Returns:
    float: Condition number
    """
    import numpy.linalg as LA
    
    if torch.is_tensor(filter_matrix):
        filter_matrix = filter_matrix.detach().cpu().numpy()
    
    # Compute singular values
    u, s, vh = LA.svd(filter_matrix)
    
    # Condition number is the ratio of largest to smallest singular value
    condition_number = s[0] / s[-1]
    
    return condition_number


def load_model(shape_name, base_dir, shape2filter_path, noise_level, filter_scale_factor, device, use_s4=False):
    """
    Load the model for a specific shape with best weights
    
    Parameters:
    shape_name: Name of the shape (initial, lowest_cn, lowest_mse, final)
    base_dir: Base directory containing the experiment results
    shape2filter_path: Path to shape2filter model
    noise_level: SNR level in dB
    filter_scale_factor: Scaling factor for filter normalization
    device: Device to run the model on
    use_s4: Whether to use S4 model
    
    Returns:
    model: Loaded model
    """
    # Find the shape path
    shape_path_pattern = os.path.join(base_dir, f"blind_noise_10.0dB_to_40.0dB_*/{shape_name}_shape.npy")
    
    # Use glob to find the matching file
    shape_files = glob.glob(shape_path_pattern)
    
    if not shape_files:
        # Try alternative path structure
        shape_path_pattern = os.path.join(base_dir, f"blind_noise_10.0dB_to_40.0dB_*/{shape_name}/shape.npy")
        shape_files = glob.glob(shape_path_pattern)
    
    if not shape_files:
        raise FileNotFoundError(f"Could not find shape file for {shape_name}")
    
    # Use the first matching file
    shape_file = shape_files[0]
    print(f"Loading shape from: {shape_file}")
    shape = np.load(shape_file)
    
    # Initialize the model
    if use_s4:
        from shape2filter_with_s4_comparison import Shape2FilterWithS4
        model = FixedShapeModel(
            shape=shape,
            shape2filter_path=shape2filter_path,
            noise_level=noise_level,
            filter_scale_factor=filter_scale_factor,
            device=device,
            use_s4=use_s4
        )
    else:
        # Load shape2filter model
        shape2filter = Shape2FilterModel().to(device)
        shape2filter.load_state_dict(torch.load(shape2filter_path, map_location=device))
        shape2filter.eval()
        
        for param in shape2filter.parameters():
            param.requires_grad = False
        
        # Create model directly without using FixedShapeModel
        tensor_shape = torch.tensor(shape, dtype=torch.float32).to(device)
        
        # Precompute filter from shape
        with torch.no_grad():
            fixed_filter = shape2filter(tensor_shape.unsqueeze(0))[0]
        
        # Create decoder and move to device
        decoder = AWAN(inplanes=11, planes=100, channels=128, n_DRBs=2).to(device)
        
        # Create a simple model class
        class SimpleFixedShapeModel(nn.Module):
            def __init__(self, fixed_filter, decoder, noise_level, filter_scale_factor):
                super().__init__()
                self.register_buffer('fixed_filter', fixed_filter)
                self.decoder = decoder
                self.noise_level = noise_level
                self.filter_scale_factor = filter_scale_factor
                
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
            
            def forward(self, x, add_noise=False):
                """Forward pass"""
                # Ensure input is in BHWC format
                if len(x.shape) != 4:
                    raise ValueError(f"Expected 4D tensor, got shape {x.shape}")
                    
                if x.shape[1] == 100 and x.shape[3] != 100:  # Input is in BCHW format
                    x = x.permute(0, 2, 3, 1)  # Convert to BHWC
                    
                # Convert to channels-first for processing
                x_channels_first = x.permute(0, 3, 1, 2)
                
                # Apply the fixed filter
                filter_normalized = self.fixed_filter / self.filter_scale_factor
                encoded_channels_first = torch.einsum('bchw,oc->bohw', x_channels_first, filter_normalized)
                
                # Add noise if needed
                if add_noise:
                    encoded_channels_first = self.add_fixed_noise(encoded_channels_first)
                
                # Decode
                decoded_channels_first = self.decoder(encoded_channels_first)
                
                # Convert back to BHWC format
                encoded = encoded_channels_first.permute(0, 2, 3, 1)
                decoded = decoded_channels_first.permute(0, 2, 3, 1)
                
                return decoded, encoded, self.noise_level
        
        # Create model
        model = SimpleFixedShapeModel(
            fixed_filter=fixed_filter,
            decoder=decoder,
            noise_level=noise_level,
            filter_scale_factor=filter_scale_factor
        ).to(device)
    
    # Find and load the best decoder weights
    best_weights_path = os.path.join(base_dir, f"fixed_shape_comparison/shape_{shape_name}/best_decoder_model.pt")
    
    if not os.path.exists(best_weights_path):
        raise FileNotFoundError(f"Could not find best weights for {shape_name} at {best_weights_path}")
    
    print(f"Loading best decoder weights from: {best_weights_path}")
    
    # Load state dict to device directly
    state_dict = torch.load(best_weights_path, map_location=device)
    model.decoder.load_state_dict(state_dict)
    
    # Make sure model is on the correct device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    print(f"Model loaded successfully and moved to {device}")
    
    return model


def select_test_samples(test_data, num_samples=10):
    """
    Select a fixed set of test samples for evaluation
    
    Parameters:
    test_data: Full test dataset
    num_samples: Number of samples to select
    
    Returns:
    selected_samples: Selected samples
    indices: Indices of selected samples in the test data
    """
    test_size = len(test_data)
    
    if test_size <= num_samples:
        # If we have fewer samples than requested, use all of them
        indices = list(range(test_size))
    else:
        # Otherwise, select evenly spaced samples including first and last
        indices = [0]  # Always include the first sample
        
        # Add evenly spaced indices
        if num_samples > 2:
            step = (test_size - 1) / (num_samples - 1)
            for i in range(1, num_samples - 1):
                indices.append(min(test_size - 1, int(i * step)))
        
        # Always include the last sample
        if num_samples > 1:
            indices.append(test_size - 1)
    
    # Select the samples
    selected_samples = test_data[indices]
    
    print(f"Selected {len(indices)} test samples at indices: {indices}")
    
    return selected_samples, indices


def select_pixel_locations(image_shape, num_pixels=25, strategy="grid"):
    """
    Select pixel locations for spectral visualization
    
    Parameters:
    image_shape: Shape of the image (H, W, C)
    num_pixels: Number of pixels to select
    strategy: Strategy for selecting pixels ("grid", "random", "corners")
    
    Returns:
    list: List of (y, x) pixel coordinates
    """
    h, w = image_shape[0:2]
    
    if strategy == "grid":
        # Create an approximately square grid of pixels
        grid_size = int(np.ceil(np.sqrt(num_pixels)))
        
        # Calculate step sizes
        h_step = max(1, h // grid_size)
        w_step = max(1, w // grid_size)
        
        # Generate grid points
        locations = []
        for i in range(grid_size):
            for j in range(grid_size):
                y = min(h-1, i * h_step + h_step // 2)
                x = min(w-1, j * w_step + w_step // 2)
                locations.append((y, x))
                
                if len(locations) >= num_pixels:
                    return locations[:num_pixels]
        
        return locations
    
    elif strategy == "random":
        # Generate random pixel locations
        import random
        locations = []
        for _ in range(num_pixels):
            y = random.randint(0, h-1)
            x = random.randint(0, w-1)
            locations.append((y, x))
        return locations
    
    elif strategy == "corners":
        # Select pixels at corners and edges
        locations = [
            (0, 0),                # Top-left
            (0, w//2),             # Top-middle
            (0, w-1),              # Top-right
            (h//2, 0),             # Middle-left
            (h//2, w//2),          # Center
            (h//2, w-1),           # Middle-right
            (h-1, 0),              # Bottom-left
            (h-1, w//2),           # Bottom-middle
            (h-1, w-1),            # Bottom-right
        ]
        
        # Add more points if needed
        if num_pixels > 9:
            # Add more points along the edges and inside
            num_extra = num_pixels - 9
            
            # Add points along top and bottom edges
            for i in range(num_extra // 4):
                x = (i + 1) * w // (num_extra // 4 + 1)
                locations.append((0, x))  # Top edge
                locations.append((h-1, x))  # Bottom edge
            
            # Add points along left and right edges
            for i in range(num_extra // 4):
                y = (i + 1) * h // (num_extra // 4 + 1)
                locations.append((y, 0))  # Left edge
                locations.append((y, w-1))  # Right edge
            
            # Add interior points
            for i in range(num_extra - num_extra//2):
                y = (i + 1) * h // (num_extra - num_extra//2 + 1)
                x = (i + 1) * w // (num_extra - num_extra//2 + 1)
                locations.append((y, x))
        
        return locations[:num_pixels]
    
    else:
        raise ValueError(f"Unknown pixel selection strategy: {strategy}")


def visualize_sample_spectrum(original, reconstruction, save_path_base, sample_idx, 
                              num_pixels=25, pixel_strategy="grid", metrics=None):
    """
    Create visualization of spectral response for specific pixels
    
    Parameters:
    original: Original image (H, W, C)
    reconstruction: Reconstructed image (H, W, C)
    save_path_base: Base path to save the visualization
    sample_idx: Index of the sample
    num_pixels: Number of pixels to visualize
    pixel_strategy: Strategy for selecting pixels
    metrics: Dictionary of metrics to display
    """
    # Create spectrum directory
    spectrum_dir = os.path.dirname(save_path_base)
    os.makedirs(spectrum_dir, exist_ok=True)
    
    # Get base filename without extension
    base_filename = os.path.basename(save_path_base).rsplit('.', 1)[0]
    
    # Calculate error
    error = original - reconstruction
    
    # Get number of spectral bands
    num_bands = original.shape[2]
    band_indices = np.arange(num_bands)
    
    # Select pixel locations
    pixel_locations = select_pixel_locations(original.shape, num_pixels, pixel_strategy)
    
    # Define colors for different pixels
    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, len(pixel_locations)))
    
    # ---------- Create combined plot with all pixels ----------
    plt.figure(figsize=(15, 10))
    plt.title(f"Sample {sample_idx} - All Pixels Spectral Response")
    
    # Plot each pixel's spectrum
    for i, (y, x) in enumerate(pixel_locations):
        color = colors[i]
        
        # Extract spectra
        original_spectrum = original[y, x, :]
        recon_spectrum = reconstruction[y, x, :]
        
        # Plot original (solid) and reconstructed (dashed)
        plt.plot(band_indices, original_spectrum, color=color, linestyle='-', 
                 alpha=0.7, label=f'Original ({y},{x})')
        plt.plot(band_indices, recon_spectrum, color=color, linestyle='--', 
                 alpha=0.7, label=f'Recon ({y},{x})')
    
    plt.xlabel('Spectral Band')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # Create custom legend with one entry per pixel location
    from matplotlib.lines import Line2D
    legend_elements = []
    for i, (y, x) in enumerate(pixel_locations):
        legend_elements.append(Line2D([0], [0], color=colors[i], label=f'Pixel ({y},{x})'))
    
    # Create two columns of legend entries to save space
    plt.legend(handles=legend_elements, loc='upper right', 
               bbox_to_anchor=(1.15, 1), ncol=2)
    
    # Save combined plot
    combined_path = os.path.join(spectrum_dir, f"{base_filename}_combined.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ---------- Create individual plots for each pixel ----------
    # Calculate number of rows and columns for subplots
    n_cols = min(5, num_pixels)
    n_rows = (num_pixels + n_cols - 1) // n_cols
    
    plt.figure(figsize=(n_cols * 4, n_rows * 3))
    
    # Add overall title
    if metrics:
        plt.suptitle(f"Sample {sample_idx} - Individual Pixel Spectra - SNR: {metrics['noise_level']}dB, PSNR: {metrics['psnr']:.2f}dB", 
                    fontsize=16, y=0.98)
    else:
        plt.suptitle(f"Sample {sample_idx} - Individual Pixel Spectra", fontsize=16, y=0.98)
    
    # Plot each pixel in its own subplot
    for i, (y, x) in enumerate(pixel_locations):
        ax = plt.subplot(n_rows, n_cols, i+1)
        
        # Extract spectra
        original_spectrum = original[y, x, :]
        recon_spectrum = reconstruction[y, x, :]
        diff_spectrum = error[y, x, :]
        
        # Plot original and reconstruction
        ax.plot(band_indices, original_spectrum, 'b-', alpha=0.7, label='Original')
        ax.plot(band_indices, recon_spectrum, 'r--', alpha=0.7, label='Reconstructed')
        
        # Add a twin axis for difference
        ax2 = ax.twinx()
        ax2.plot(band_indices, diff_spectrum, 'g:', alpha=0.5, label='Difference')
        ax2.set_ylabel('Error', color='g', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='g', labelsize=8)
        
        # Set titles and labels
        ax.set_title(f"Pixel ({y},{x})", fontsize=10)
        if i % n_cols == 0:  # Only add y-label for leftmost plots
            ax.set_ylabel('Value')
        if i >= (n_rows-1) * n_cols:  # Only add x-label for bottom plots
            ax.set_xlabel('Band')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    # Save individual plots
    individual_path = os.path.join(spectrum_dir, f"{base_filename}_individual.png")
    plt.savefig(individual_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ---------- Save spectral data as JSON ----------
    spectrum_data = {
        'pixels': {},
    }
    
    if metrics:
        spectrum_data['metrics'] = metrics
    
    for i, (y, x) in enumerate(pixel_locations):
        spectrum_data['pixels'][f"pixel_{y}_{x}"] = {
            'original': original[y, x, :].tolist(),
            'reconstructed': reconstruction[y, x, :].tolist(),
            'difference': error[y, x, :].tolist(),
            'coordinates': [int(y), int(x)]
        }
    
    # Save data
    data_path = os.path.join(spectrum_dir, f"{base_filename}_spectrum_data.json")
    with open(data_path, 'w') as f:
        json.dump(spectrum_data, f, indent=2)
    
    return combined_path, individual_path, data_path


def evaluate_model(model, test_samples, full_test_data, noise_level, output_dir, shape_name, device,
                  visualize_spectrum=False, num_spectrum_pixels=25, spectrum_pixel_strategy="grid"):
    """
    Evaluate a model on test samples
    
    Parameters:
    model: Model to evaluate
    test_samples: Test samples to evaluate on
    full_test_data: Full test dataset for overall metrics
    noise_level: Current noise level
    output_dir: Directory to save results
    shape_name: Name of the shape
    device: Device to run evaluation on
    visualize_spectrum: Whether to visualize spectral response
    num_spectrum_pixels: Number of pixels to visualize for spectral response
    spectrum_pixel_strategy: Strategy for selecting pixels
    
    Returns:
    results: Dictionary with evaluation results
    """
    # Create output directory
    shape_noise_dir = os.path.join(output_dir, f"shape_{shape_name}/noise_{noise_level}dB")
    os.makedirs(shape_noise_dir, exist_ok=True)
    
    # Create subdirectories
    recon_dir = os.path.join(shape_noise_dir, "reconstructions")
    numpy_dir = os.path.join(shape_noise_dir, "numpy_arrays")
    metrics_dir = os.path.join(shape_noise_dir, "metrics")
    spectrum_dir = os.path.join(shape_noise_dir, "spectrum")
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(numpy_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    if visualize_spectrum:
        os.makedirs(spectrum_dir, exist_ok=True)
    
    # Set model noise level
    model.noise_level = noise_level
    model.eval()
    
    # Get sample indices
    sample_indices = list(range(len(test_samples)))
    
    # Initialize results dictionary
    results = {
        'sample_metrics': [],
        'overall_metrics': {},
        'spectrum_paths': {}
    }
    
    # Evaluate each sample
    for i, sample_idx in enumerate(sample_indices):
        sample = test_samples[i:i+1].to(device)
        
        with torch.no_grad():
            # Forward pass with fixed noise
            recon, encoded, _ = model(sample, add_noise=False)
            
            # Move to CPU for metrics and visualization
            sample_np = sample.cpu().numpy()
            recon_np = recon.cpu().numpy()
            
            # Calculate metrics for this sample
            mse = ((sample_np - recon_np) ** 2).mean()
            psnr = calculate_psnr(sample_np, recon_np)
            sam = calculate_sam(sample_np, recon_np)
            
            # Store metrics
            sample_metrics = {
                'sample_idx': sample_idx,
                'mse': float(mse),
                'psnr': float(psnr),
                'sam': float(sam),
                'noise_level': noise_level
            }
            results['sample_metrics'].append(sample_metrics)
            
            # Save numpy arrays (full data)
            np.save(os.path.join(numpy_dir, f"original_sample_{sample_idx}.npy"), sample_np[0])
            np.save(os.path.join(numpy_dir, f"recon_sample_{sample_idx}.npy"), recon_np[0])
            
            # Save reconstruction visualization
            visualize_sample(sample_np[0], recon_np[0], os.path.join(recon_dir, f"sample_{sample_idx}.png"), 
                            sample_idx, metrics=sample_metrics)
            
            # Visualize spectral response if requested
            if visualize_spectrum:
                combined_path, individual_path, data_path = visualize_sample_spectrum(
                    original=sample_np[0],
                    reconstruction=recon_np[0],
                    save_path_base=os.path.join(spectrum_dir, f"sample_{sample_idx}"),
                    sample_idx=sample_idx,
                    num_pixels=num_spectrum_pixels,
                    pixel_strategy=spectrum_pixel_strategy,
                    metrics=sample_metrics
                )
                
                # Store paths
                results['spectrum_paths'][sample_idx] = {
                    'combined': combined_path,
                    'individual': individual_path,
                    'data': data_path
                }
    
    # Calculate overall metrics on full test dataset
    with torch.no_grad():
        # Calculate batch metrics on smaller chunks to avoid OOM
        batch_size = 16  # Smaller batch size for evaluation
        
        # Initialize accumulators
        total_mse = 0.0
        total_psnr = 0.0
        total_sam = 0.0
        num_batches = 0
        
        # Process test data in batches
        for i in tqdm(range(0, len(full_test_data), batch_size), desc=f"Calculating overall metrics for {shape_name} at {noise_level}dB"):
            # Get batch
            end_idx = min(i + batch_size, len(full_test_data))
            batch = full_test_data[i:end_idx].to(device)
            
            # Forward pass
            recon, _, _ = model(batch, add_noise=False)
            
            # Calculate metrics for this batch
            batch_mse = ((recon - batch) ** 2).mean().item()
            batch_psnr = calculate_psnr(batch.cpu().numpy(), recon.cpu().numpy())
            batch_sam = calculate_sam(batch.cpu().numpy(), recon.cpu().numpy())
            
            # Accumulate metrics
            total_mse += batch_mse
            total_psnr += batch_psnr
            total_sam += batch_sam
            num_batches += 1
        
        # Calculate averages
        overall_mse = total_mse / num_batches
        overall_psnr = total_psnr / num_batches
        overall_sam = total_sam / num_batches
        
        # Calculate condition number (if the filter attribute exists)
        if hasattr(model, 'fixed_filter'):
            condition_number = calculate_condition_number(model.fixed_filter.detach().cpu())
        else:
            # If we can't get the filter directly, set to NaN
            condition_number = float('nan')
        
        # Store overall metrics
        results['overall_metrics'] = {
            'mse': float(overall_mse),
            'psnr': float(overall_psnr),
            'sam': float(overall_sam),
            'condition_number': float(condition_number),
            'noise_level': noise_level
        }
    
    # Save per-sample metrics to CSV
    sample_metrics_df = pd.DataFrame(results['sample_metrics'])
    sample_metrics_df.to_csv(os.path.join(metrics_dir, "sample_metrics.csv"), index=False)
    
    # Save overall metrics to JSON
    with open(os.path.join(metrics_dir, "overall_metrics.json"), "w") as f:
        json.dump(results['overall_metrics'], f, indent=4)
    
    print(f"Evaluation of {shape_name} at {noise_level}dB - MSE: {overall_mse:.6f}, PSNR: {overall_psnr:.2f}dB, SAM: {overall_sam:.6f} rad, CN: {condition_number:.2f}")
    
    return results


def visualize_sample(original, reconstruction, save_path, sample_idx, metrics=None, band_indices=None):
    """
    Create visualization of original, reconstructed, and difference images
    
    Parameters:
    original: Original image (H, W, C)
    reconstruction: Reconstructed image (H, W, C)
    save_path: Path to save the visualization
    sample_idx: Index of the sample
    metrics: Dictionary of metrics to display
    band_indices: Specific band indices to visualize (if None, middle band is used)
    """
    # Calculate difference
    difference = original - reconstruction
    
    # If band_indices not specified, use middle band
    if band_indices is None:
        if original.shape[2] > 50:  # Check if we have enough bands
            band_indices = [5, 25, 50, 75, 95]  # Sample across the spectrum
        else:
            band_indices = [original.shape[2] // 2]  # Middle band
    
    # Create figure and plot
    fig, axs = plt.subplots(len(band_indices), 3, figsize=(15, 5 * len(band_indices)))
    
    # If only one band, make axes indexable
    if len(band_indices) == 1:
        axs = np.array([axs])
    
    # Plot each band
    for i, band_idx in enumerate(band_indices):
        # Extract specific band
        orig_band = original[:, :, band_idx]
        recon_band = reconstruction[:, :, band_idx]
        diff_band = difference[:, :, band_idx]
        
        # Calculate global min and max for unified colormap
        vmin = min(orig_band.min(), recon_band.min())
        vmax = max(orig_band.max(), recon_band.max())
        
        # Calculate symmetric limits for difference
        diff_abs_max = max(abs(diff_band.min()), abs(diff_band.max()))
        
        # Plot original
        im1 = axs[i, 0].imshow(orig_band, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i, 0].set_title(f'Original (Band {band_idx})')
        plt.colorbar(im1, ax=axs[i, 0], fraction=0.046, pad=0.04)
        
        # Plot reconstruction
        im2 = axs[i, 1].imshow(recon_band, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i, 1].set_title(f'Reconstructed (Band {band_idx})')
        plt.colorbar(im2, ax=axs[i, 1], fraction=0.046, pad=0.04)
        
        # Plot difference
        im3 = axs[i, 2].imshow(diff_band, cmap='coolwarm', vmin=-diff_abs_max, vmax=diff_abs_max)
        axs[i, 2].set_title(f'Difference (Band {band_idx})')
        plt.colorbar(im3, ax=axs[i, 2], fraction=0.046, pad=0.04)
    
    # Add metrics as suptitle if provided
    if metrics:
        plt.suptitle(f"Sample {sample_idx} - MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f}dB, SAM: {metrics['sam']:.6f} rad, SNR: {metrics['noise_level']}dB", 
                   fontsize=16)
    else:
        plt.suptitle(f"Sample {sample_idx}")
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_comparison_plots(output_dir, shapes, noise_levels):
    """
    Create comparison plots across all shapes and noise levels
    
    Parameters:
    output_dir: Output directory
    shapes: List of shape names
    noise_levels: List of noise levels
    """
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Initialize data structures for metrics
    metrics = {
        'shape': [],
        'noise_level': [],
        'mse': [],
        'psnr': [],
        'sam': [],
        'condition_number': []
    }
    
    # Collect metrics from each shape and noise level
    for shape in shapes:
        for noise_level in noise_levels:
            metrics_file = os.path.join(output_dir, f"shape_{shape}/noise_{noise_level}dB/metrics/overall_metrics.json")
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    shape_metrics = json.load(f)
                
                # Add to metrics
                metrics['shape'].append(shape)
                metrics['noise_level'].append(noise_level)
                metrics['mse'].append(shape_metrics['mse'])
                metrics['psnr'].append(shape_metrics['psnr'])
                metrics['sam'].append(shape_metrics['sam'])
                metrics['condition_number'].append(shape_metrics.get('condition_number', float('nan')))
    
    # Check if we have any data
    if len(metrics['shape']) == 0:
        print("No data available for comparison plots. Skipping plot generation.")
        return
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(comparison_dir, "all_metrics.csv"), index=False)
    
    # Create plots comparing metrics across noise levels for each shape
    unique_shapes = metrics_df['shape'].unique()
    
    if len(unique_shapes) > 0:
        # 1. MSE vs Noise Level
        plt.figure(figsize=(12, 8))
        for shape in unique_shapes:
            shape_data = metrics_df[metrics_df['shape'] == shape]
            if not shape_data.empty:
                plt.plot(shape_data['noise_level'], shape_data['mse'], 'o-', label=shape)
        
        plt.title('MSE vs Noise Level')
        plt.xlabel('Noise Level (dB)')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(comparison_dir, "mse_vs_noise.png"), dpi=300)
        plt.close()
        
        # 2. PSNR vs Noise Level
        plt.figure(figsize=(12, 8))
        for shape in unique_shapes:
            shape_data = metrics_df[metrics_df['shape'] == shape]
            if not shape_data.empty:
                plt.plot(shape_data['noise_level'], shape_data['psnr'], 'o-', label=shape)
        
        plt.title('PSNR vs Noise Level')
        plt.xlabel('Noise Level (dB)')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(comparison_dir, "psnr_vs_noise.png"), dpi=300)
        plt.close()
        
        # 3. SAM vs Noise Level
        plt.figure(figsize=(12, 8))
        for shape in unique_shapes:
            shape_data = metrics_df[metrics_df['shape'] == shape]
            if not shape_data.empty:
                plt.plot(shape_data['noise_level'], shape_data['sam'], 'o-', label=shape)
        
        plt.title('SAM vs Noise Level')
        plt.xlabel('Noise Level (dB)')
        plt.ylabel('SAM (rad)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(comparison_dir, "sam_vs_noise.png"), dpi=300)
        plt.close()
        
        # 4. Condition Number Comparison
        condition_df = metrics_df.drop_duplicates('shape')
        if len(condition_df) > 0:
            plt.figure(figsize=(10, 6))
            plt.bar(condition_df['shape'], condition_df['condition_number'])
            
            plt.title('Filter Condition Number by Shape')
            plt.xlabel('Shape')
            plt.ylabel('Condition Number')
            plt.grid(True, axis='y')
            plt.savefig(os.path.join(comparison_dir, "condition_number.png"), dpi=300)
            plt.close()
    
    # Create heatmap visualizations if we have enough data
    unique_noise_levels = metrics_df['noise_level'].unique()
    
    if len(unique_shapes) > 0 and len(unique_noise_levels) > 0:
        try:
            # 5. MSE Heatmap
            plt.figure(figsize=(10, 8))
            mseheat = metrics_df.pivot(index='shape', columns='noise_level', values='mse')
            plt.imshow(mseheat, cmap='viridis')
            plt.colorbar(label='MSE')
            plt.title('MSE by Shape and Noise Level')
            plt.xlabel('Noise Level (dB)')
            plt.ylabel('Shape')
            plt.xticks(range(len(unique_noise_levels)), [str(n) for n in sorted(unique_noise_levels)])
            plt.yticks(range(len(unique_shapes)), unique_shapes)
            
            # Add text annotations
            for i in range(len(unique_shapes)):
                for j in range(len(unique_noise_levels)):
                    try:
                        value = mseheat.iloc[i, j]
                        if not pd.isna(value):
                            plt.text(j, i, f"{value:.5f}", ha="center", va="center", color="white")
                    except (IndexError, KeyError):
                        continue
            
            plt.savefig(os.path.join(comparison_dir, "mse_heatmap.png"), dpi=300)
            plt.close()
            
            # 6. PSNR Heatmap
            plt.figure(figsize=(10, 8))
            psnrheat = metrics_df.pivot(index='shape', columns='noise_level', values='psnr')
            plt.imshow(psnrheat, cmap='viridis')
            plt.colorbar(label='PSNR (dB)')
            plt.title('PSNR by Shape and Noise Level')
            plt.xlabel('Noise Level (dB)')
            plt.ylabel('Shape')
            plt.xticks(range(len(unique_noise_levels)), [str(n) for n in sorted(unique_noise_levels)])
            plt.yticks(range(len(unique_shapes)), unique_shapes)
            
            # Add text annotations
            for i in range(len(unique_shapes)):
                for j in range(len(unique_noise_levels)):
                    try:
                        value = psnrheat.iloc[i, j]
                        if not pd.isna(value):
                            plt.text(j, i, f"{value:.2f}", ha="center", va="center", color="white")
                    except (IndexError, KeyError):
                        continue
            
            plt.savefig(os.path.join(comparison_dir, "psnr_heatmap.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error creating heatmaps: {e}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate trained models on fixed test samples at different noise levels")
    
    # Data arguments
    parser.add_argument('--cache', type=str, default="cache_filtered/aviris_tiles_forest.pt", 
                       help="Path to cache file for processed data")
    parser.add_argument('--use-cache', action='store_true', help="Use cached data if available")
    parser.add_argument('-f', '--folders', type=str, default="all", 
                       help="Comma-separated list of folder name patterns to include, or 'all' for all folders")
    
    # Model arguments
    parser.add_argument('--experiment-dir', type=str, required=True,
                       help="Directory containing the experiment results")
    parser.add_argument('--shape2filter-path', type=str, default="outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt",
                      help="Path to shape2filter model")
    parser.add_argument('--filter-scale', type=float, default=10.0, 
                       help="Scaling factor for filter normalization (default: 10.0)")
    parser.add_argument('--use-s4', action='store_true', 
                       help="Use S4 model instead of regular shape2filter")
    
    # Evaluation arguments
    parser.add_argument('--noise-levels', type=str, default="10,20,30,40",
                       help="Comma-separated list of noise levels to evaluate")
    parser.add_argument('--num-samples', type=int, default=10,
                       help="Number of fixed test samples to evaluate")
    parser.add_argument('--batch-size', type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument('--output-dir', type=str, default=None,
                       help="Output directory (default: evaluation_results_[timestamp])")
    parser.add_argument('--gpu', type=int, default=None, 
                       help="GPU ID to use (e.g., 0, 1, 2, 3)")
    parser.add_argument('--shapes', type=str, default="initial,lowest_cn,lowest_mse,final",
                       help="Comma-separated list of shapes to evaluate")
    
    # Spectrum visualization arguments
    parser.add_argument('--visualize-spectrum', action='store_true', 
                       help="Generate spectral visualizations for each sample")
    parser.add_argument('--num-spectrum-pixels', type=int, default=25,
                       help="Number of pixels to visualize for spectral response")
    parser.add_argument('--spectrum-pixel-strategy', type=str, default="grid",
                       help="Strategy for selecting pixels (grid, random, corners)")
    
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        output_dir = f"evaluation_results_{timestamp}"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}/")
    
    # Save command-line arguments
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
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
    
    # Load and split data
    print("Loading data...")
    data = load_aviris_forest_data(base_path="AVIRIS_FOREST_SIMPLE_SELECT", tile_size=128, 
                                  cache_file=cache_path, use_cache=args.use_cache, folder_patterns=folder_patterns)
    
    # Verify data is in BHWC format
    if data.shape[1] == 100:  # If in BCHW
        data = data.permute(0, 2, 3, 1)  # Convert to BHWC
        print(f"Converted data to BHWC format: {data.shape}")
    
    # Split data into training and testing sets (70% train, 30% test)
    num_samples = data.shape[0]
    indices = torch.randperm(num_samples)
    train_size = int(0.7 * num_samples)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_data = data[train_indices]
    test_data = data[test_indices]
    
    print(f"Data split into {train_data.shape[0]} training and {test_data.shape[0]} testing samples")
    
    # Select fixed test samples
    test_samples, sample_indices = select_test_samples(test_data, num_samples=args.num_samples)
    
    # Parse noise levels
    noise_levels = [float(n) for n in args.noise_levels.split(',')]
    
    # Parse shapes to evaluate
    shapes = args.shapes.split(',')
    
    # Print spectrum visualization settings
    if args.visualize_spectrum:
        print(f"Spectrum visualization enabled with {args.num_spectrum_pixels} pixels using '{args.spectrum_pixel_strategy}' strategy")
    
    # Create empty results dictionary
    results = {shape: {level: None for level in noise_levels} for shape in shapes}
    
    # Evaluate each shape model at each noise level
    for shape in shapes:
        print(f"\n{'='*50}")
        print(f"Evaluating {shape} shape model")
        print(f"{'='*50}")
        
        for noise_level in noise_levels:
            print(f"\nEvaluating at noise level: {noise_level} dB")
            
            try:
                # Load the model
                model = load_model(
                    shape_name=shape, 
                    base_dir=args.experiment_dir, 
                    shape2filter_path=args.shape2filter_path,
                    noise_level=noise_level, 
                    filter_scale_factor=args.filter_scale, 
                    device=device,
                    use_s4=args.use_s4
                )
                
                # Evaluate
                shape_results = evaluate_model(
                    model=model, 
                    test_samples=test_samples,
                    full_test_data=test_data,
                    noise_level=noise_level, 
                    output_dir=output_dir, 
                    shape_name=shape,
                    device=device,
                    visualize_spectrum=args.visualize_spectrum,
                    num_spectrum_pixels=args.num_spectrum_pixels,
                    spectrum_pixel_strategy=args.spectrum_pixel_strategy
                )
                
                # Store results
                results[shape][noise_level] = shape_results
                
            except Exception as e:
                print(f"Error evaluating {shape} at {noise_level}dB: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    create_comparison_plots(output_dir, shapes, noise_levels)
    
    print(f"\nEvaluation complete. Results saved to {os.path.abspath(output_dir)}/")
    
    return results


if __name__ == "__main__":
    main()