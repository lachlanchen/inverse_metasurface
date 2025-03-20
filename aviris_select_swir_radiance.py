import numpy as np
import matplotlib.pyplot as plt
import spectral
import os
import pandas as pd
from spectral.io import envi
from tqdm import tqdm
import shutil
# Add imports for PyTorch
import torch

# Define paths for the radiance data
base_folder = 'AVIRIS/f170429t01p00r11rdn_e'
img_file = 'f170429t01p00r11rdn_e_sc01_ort_img'
hdr_file = f'{base_folder}/{img_file}.hdr'
csv_file = 'partial_crys_data/partial_crys_C0.0.csv'

# Create output directory
output_folder = 'AVIRIS_SWIR_RADIANCE'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create additional directories for NumPy and PyTorch exports
numpy_folder = f'{output_folder}/numpy'
torch_folder = f'{output_folder}/torch'
images_folder = f'{output_folder}/images'

for folder in [numpy_folder, torch_folder, images_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

output_img_file = f'{output_folder}/{img_file}'
output_hdr_file = f'{output_folder}/{img_file}.hdr'

# Step 1: Open the AVIRIS image and check its properties
print(f"Opening AVIRIS image from: {hdr_file}")
img = spectral.open_image(hdr_file)
print(f"Image dimensions: {img.shape}")
print(f"Number of bands: {img.nbands}")

# Define nodata value (from the terminal output)
nodata_value = -9999.0

# Read the wavelengths from the CSV file
print(f"\nReading wavelengths from CSV: {csv_file}")
csv_data = pd.read_csv(csv_file)
csv_wavelengths = csv_data['Wavelength_um'].to_numpy() * 1000  # Convert μm to nm
print(f"Found {len(csv_wavelengths)} wavelengths in CSV file")
print(f"CSV wavelength range: {np.min(csv_wavelengths):.2f} to {np.max(csv_wavelengths):.2f} nm")

if hasattr(img, 'bands') and hasattr(img.bands, 'centers'):
    aviris_wavelengths = np.array(img.bands.centers)
    print(f"AVIRIS wavelength range: {np.min(aviris_wavelengths):.2f} to {np.max(aviris_wavelengths):.2f} nm")
    
    # For each wavelength in the CSV, find the closest band in AVIRIS
    wavelength_mapping = []
    
    for i, wl_nm in enumerate(csv_wavelengths):
        # Find the closest matching band in AVIRIS data
        band_idx = np.abs(aviris_wavelengths - wl_nm).argmin()
        
        wavelength_mapping.append({
            'CSV_Index': i,
            'CSV_Wavelength_nm': wl_nm,
            'CSV_Wavelength_um': wl_nm/1000,
            'AVIRIS_Band': band_idx,
            'AVIRIS_Wavelength_nm': aviris_wavelengths[band_idx],
            'AVIRIS_Wavelength_um': aviris_wavelengths[band_idx]/1000,
            'Difference_nm': abs(wl_nm - aviris_wavelengths[band_idx])
        })
    
    # Sort by difference and select top 100 best matching bands
    wavelength_mapping = sorted(wavelength_mapping, key=lambda x: x['Difference_nm'])
    wavelength_mapping = wavelength_mapping[:100]  # Keep only the 100 best matches
    
    # Get the selected bands
    selected_bands = [item['AVIRIS_Band'] for item in wavelength_mapping]
    
    # Create a DataFrame for better display
    mapping_df = pd.DataFrame(wavelength_mapping)
    print(f"\nSelected {len(selected_bands)} closest matching AVIRIS bands")
    print("First 10 wavelength mappings:")
    print(mapping_df.head(10).to_string(index=False))
    
    # Extract the selected bands and save them to a new file
    print(f"\nExtracting selected bands...")
    
    # Prepare a new data array for selected bands
    rows, cols, _ = img.shape
    new_data = np.zeros((rows, cols, len(selected_bands)), dtype=np.float32)
    rescaled_data = np.zeros((rows, cols, len(selected_bands)), dtype=np.float32)
    
    # First pass: Collect global statistics from all selected bands
    print("First pass: Collecting global statistics...")
    all_valid_values = []
    sample_size = 100000  # Sample size to keep memory usage reasonable

    for band_idx in tqdm(selected_bands, desc="Analyzing bands"):
        # Read the band data
        band_data = img.read_band(band_idx)
        
        # Create mask for nodata values
        valid_mask = band_data != nodata_value
        
        # Get valid data
        valid_data = band_data[valid_mask]
        
        # Randomly sample values to add to our global statistics
        if len(valid_data) > 0:
            # Take a random sample to avoid memory issues
            if len(valid_data) > sample_size:
                rng = np.random.default_rng()
                indices = rng.choice(len(valid_data), size=sample_size, replace=False)
                sample = valid_data[indices]
            else:
                sample = valid_data
            
            all_valid_values.extend(sample)
    
    # Calculate global statistics
    if all_valid_values:
        all_valid_values = np.array(all_valid_values)
        global_min = np.min(all_valid_values)
        global_max = np.max(all_valid_values)
        global_mean = np.mean(all_valid_values)
        global_std = np.std(all_valid_values)
        
        # Calculate percentiles for robust scaling
        p0_5 = np.percentile(all_valid_values, 0.5)
        p99_5 = np.percentile(all_valid_values, 99.5)
        
        # Use 3 sigma rule for outlier detection
        outlier_low = global_mean - 3 * global_std
        outlier_high = global_mean + 3 * global_std
        
        # Choose the more conservative approach (either percentile or sigma)
        robust_min = max(p0_5, outlier_low)
        robust_max = min(p99_5, outlier_high)
    else:
        print("Warning: No valid data found!")
        robust_min = -350
        robust_max = 356
    
    print(f"Global statistics after sampling {len(all_valid_values)} values:")
    print(f"  Full range: [{global_min:.4f}, {global_max:.4f}]")
    print(f"  Mean: {global_mean:.4f}, Std: {global_std:.4f}")
    print(f"  0.5-99.5 percentile: [{p0_5:.4f}, {p99_5:.4f}]")
    print(f"  3-sigma range: [{outlier_low:.4f}, {outlier_high:.4f}]")
    print(f"  Chosen robust range for scaling: [{robust_min:.4f}, {robust_max:.4f}]")
    
    # Create a list to store band statistics
    band_stats = []
    
    # Second pass: Extract and normalize the bands
    print("\nSecond pass: Extracting and normalizing bands...")
    for i, band_idx in enumerate(tqdm(selected_bands, desc="Processing bands")):
        # Read the band data
        band_data = img.read_band(band_idx)
        
        # Create mask for nodata values
        valid_mask = band_data != nodata_value
        
        # Store original statistics
        valid_data = band_data[valid_mask]
        if len(valid_data) > 0:
            orig_min = np.min(valid_data)
            orig_max = np.max(valid_data)
            orig_mean = np.mean(valid_data)
            orig_std = np.std(valid_data)
        else:
            orig_min = orig_max = orig_mean = orig_std = 0
        
        # Create a copy of the data for processing
        processed_data = band_data.copy().astype(np.float32)
        
        # Clip values to the robust range and convert to float [0, 1]
        normalized_data = np.zeros_like(processed_data, dtype=np.float32)
        
        # Only normalize valid data
        mask = valid_mask & (processed_data >= robust_min) & (processed_data <= robust_max)
        
        # Apply normalization to the valid range
        if np.any(mask) and (robust_max > robust_min):
            normalized_data[mask] = (processed_data[mask] - robust_min) / (robust_max - robust_min)
        
        # Set everything else to 0
        normalized_data[~mask] = 0
        
        # Get statistics after normalization
        valid_normalized = normalized_data[normalized_data > 0]
        if len(valid_normalized) > 0:
            norm_min = np.min(valid_normalized)
            norm_max = np.max(valid_normalized)
            norm_mean = np.mean(valid_normalized)
            norm_std = np.std(valid_normalized)
        else:
            norm_min = norm_max = norm_mean = norm_std = 0
        
        # Store the stats
        band_stats.append({
            'Position': i,
            'AVIRIS_Band': band_idx,
            'Wavelength_nm': aviris_wavelengths[band_idx],
            'Orig_Min_Value': orig_min,
            'Orig_Max_Value': orig_max,
            'Orig_Mean_Value': orig_mean,
            'Orig_Std_Dev': orig_std,
            'Norm_Min': norm_min,
            'Norm_Max': norm_max,
            'Norm_Mean': norm_mean,
            'Norm_Std': norm_std,
            'Valid_Pixels': np.sum(valid_mask),
            'Normalized_Pixels': np.sum(mask),
            'Zero_Pixels': np.sum(normalized_data == 0)
        })
        
        # Store the original band data for ENVI output
        new_data[:,:,i] = processed_data
        
        # Store the normalized data for visualization and NumPy/PyTorch
        rescaled_data[:,:,i] = normalized_data
    
    # Convert band stats to DataFrame and display summary
    stats_df = pd.DataFrame(band_stats)
    print("\nBand Statistics Summary:")
    print(f"Original min value across all bands: {stats_df['Orig_Min_Value'].min()}")
    print(f"Original max value across all bands: {stats_df['Orig_Max_Value'].max()}")
    print(f"Original mean of means: {stats_df['Orig_Mean_Value'].mean()}")
    print(f"Normalized min value across all bands: {stats_df['Norm_Min'].min()}")
    print(f"Normalized max value across all bands: {stats_df['Norm_Max'].max()}")
    print(f"Normalized mean of means: {stats_df['Norm_Mean'].mean()}")
    
    # Create a new header dictionary based on the original
    original_header = envi.read_envi_header(hdr_file)
    
    # Update the header for the new file
    new_header = original_header.copy()
    new_header['bands'] = len(selected_bands)
    
    # Update wavelength information
    if 'wavelength' in new_header:
        new_header['wavelength'] = [str(aviris_wavelengths[idx]) for idx in selected_bands]
    
    # Write the original data (not normalized) to an ENVI file
    print(f"Writing selected bands to: {output_img_file}")
    envi.save_image(output_hdr_file, new_data, metadata=new_header, force=True)
    
    # Check if we need to rename the file to match the original format
    if os.path.exists(f"{output_img_file}.img") and not os.path.exists(f"{base_folder}/{img_file}.img"):
        print(f"Renaming output file to match original format")
        shutil.move(f"{output_img_file}.img", output_img_file)
    
    # Save normalized data in NumPy (.npy) format
    print(f"Saving normalized data in NumPy format: {numpy_folder}/aviris_swir_radiance.npy")
    np.save(f"{numpy_folder}/aviris_swir_radiance.npy", rescaled_data)
    
    # Save wavelength information with the NumPy data
    selected_wavelengths = np.array([aviris_wavelengths[idx] for idx in selected_bands])
    np.save(f"{numpy_folder}/wavelengths.npy", selected_wavelengths)
    
    # Save normalized data in PyTorch (.pt) format
    print(f"Saving normalized data in PyTorch format: {torch_folder}/aviris_swir_radiance.pt")
    torch_data = torch.from_numpy(rescaled_data)
    torch.save(torch_data, f"{torch_folder}/aviris_swir_radiance.pt")
    
    # Save wavelength information with the PyTorch data
    torch_wavelengths = torch.from_numpy(selected_wavelengths)
    torch.save(torch_wavelengths, f"{torch_folder}/wavelengths.pt")
    
    # For visualization - use a fixed range [0, 1] for the normalized data
    vmin = 0
    vmax = 1
    
    print(f"Using visualization range: vmin={vmin:.4f}, vmax={vmax:.4f}")
    
    # Create a figure with consistent scaling for selected frames
    frames_to_plot = [0, 24, 49, 99]  # 0-indexed, so these are frames 1, 25, 50, 100
    frames_to_plot = [idx for idx in frames_to_plot if idx < len(selected_bands)]  # Make sure we don't go out of bounds
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Save each individual band as an image
    print(f"Saving individual band images to: {images_folder}")
    for frame_idx in tqdm(range(len(selected_bands)), desc="Saving band images"):
        # Get the normalized band data
        band_data = rescaled_data[:,:,frame_idx]
        
        # Create a masked array for zeros
        masked_data = np.ma.masked_where(band_data == 0, band_data)
        
        # Create a figure for this band
        plt.figure(figsize=(8, 8))
        plt.imshow(masked_data, cmap='viridis', vmin=vmin, vmax=vmax)
        
        # Get the wavelength for this band
        original_band_idx = selected_bands[frame_idx]
        original_wavelength = aviris_wavelengths[original_band_idx]
        
        plt.title(f'Band {frame_idx+1}: {original_wavelength:.2f} nm ({original_wavelength/1000:.2f} μm)')
        plt.colorbar(label='Normalized Radiance')
        plt.tight_layout()
        
        # Save the figure
        wavelength_str = f"{original_wavelength:.2f}".replace('.', 'p')
        plt.savefig(f'{images_folder}/band_{frame_idx+1:03d}_{wavelength_str}nm.png', dpi=150)
        plt.close()
    
    # Create overview plot for selected frames
    for i, frame_idx in enumerate(frames_to_plot):
        if i < len(axes):
            # Get statistics for this frame from our precomputed stats
            frame_stats = stats_df.iloc[frame_idx]
            print(f"\nFrame {frame_idx+1} (Band {frame_stats['AVIRIS_Band']}, {frame_stats['Wavelength_nm']:.2f} nm):")
            print(f"  Original Min: {frame_stats['Orig_Min_Value']:.6f}, Original Max: {frame_stats['Orig_Max_Value']:.6f}")
            print(f"  Normalized Min: {frame_stats['Norm_Min']:.6f}, Normalized Max: {frame_stats['Norm_Max']:.6f}")
            print(f"  Normalized Mean: {frame_stats['Norm_Mean']:.6f}, Normalized Std: {frame_stats['Norm_Std']:.6f}")
            print(f"  Valid Pixels: {frame_stats['Valid_Pixels']}, Normalized Pixels: {frame_stats['Normalized_Pixels']}")
            
            # Get the band data
            band_data = rescaled_data[:,:,frame_idx]
            
            # Create a masked array for zeros
            masked_data = np.ma.masked_where(band_data == 0, band_data)
            
            # Plot on the corresponding subplot with consistent scale
            im = axes[i].imshow(masked_data, cmap='viridis', vmin=vmin, vmax=vmax)
            
            # Get the wavelength for this band
            original_band_idx = selected_bands[frame_idx]
            original_wavelength = aviris_wavelengths[original_band_idx]
            
            axes[i].set_title(f'Frame {frame_idx+1}: {original_wavelength:.2f} nm ({original_wavelength/1000:.2f} μm)')
            plt.colorbar(im, ax=axes[i], label='Normalized Radiance')
    
    # Hide any unused subplots
    for i in range(len(frames_to_plot), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/selected_frames.png', dpi=300)
    plt.close()
    
    # Save the wavelength mapping to a CSV file for reference
    mapping_df.to_csv(f'{output_folder}/wavelength_mapping.csv', index=False)
    print(f"Saved wavelength mapping to: {output_folder}/wavelength_mapping.csv")
    
    # Save the band statistics to a CSV file for reference
    stats_df.to_csv(f'{output_folder}/band_statistics.csv', index=False)
    print(f"Saved band statistics to: {output_folder}/band_statistics.csv")
    
    # Generate metadata files for the NumPy and PyTorch exports
    # Convert NumPy types to Python native types to avoid JSON serialization issues
    metadata = {
        'shape': tuple(int(x) for x in rescaled_data.shape),
        'wavelengths_nm': [float(x) for x in selected_wavelengths.tolist()],
        'wavelengths_um': [float(x) for x in (selected_wavelengths / 1000).tolist()],
        'nodata_value': float(0.0),  # We've replaced nodata with 0
        'original_range': [float(robust_min), float(robust_max)],
        'normalized_range': [0.0, 1.0],
        'original_bands': [int(x) for x in selected_bands]
    }
    
    # Save metadata as JSON
    import json
    with open(f"{numpy_folder}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(f"{torch_folder}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nProcessing complete!")
    print(f"Data saved to:")
    print(f"  - ENVI format: {output_img_file} (original values)")
    print(f"  - NumPy format: {numpy_folder}/aviris_swir_radiance.npy (normalized [0,1])")
    print(f"  - PyTorch format: {torch_folder}/aviris_swir_radiance.pt (normalized [0,1])")
    print(f"  - Individual band images: {images_folder}/")
    print(f"  - Normalized data range: [{robust_min:.4f}, {robust_max:.4f}] -> [0, 1]")

else:
    print("No wavelength information found in the image header.")
    exit(1)
