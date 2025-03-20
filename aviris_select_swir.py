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

# Define paths
base_folder = 'AVIRIS/AV320231008t173943_L2A_OE_main_98b13fff'
img_file = 'AV320231008t173943_L2A_OE_main_98b13fff_RFL_ORT'
hdr_file = f'{base_folder}/{img_file}.hdr'
csv_file = 'partial_crys_data/partial_crys_C0.0.csv'

# Create output directory
output_folder = 'AVIRIS_SWIR'
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

# Step 1: Check the min/max of the original data first
print(f"Opening AVIRIS image from: {hdr_file}")
img = spectral.open_image(hdr_file)
print(f"Image dimensions: {img.shape}")
print(f"Number of bands: {img.nbands}")

# Sample a few bands to get an idea of the original data range
sample_bands = [0, 50, 100, 150, 200, 250]
print("\nOriginal data sample statistics:")
data_ranges = []

for band_idx in sample_bands:
    band_data = img.read_band(band_idx)
    nodata_value = -9999.0
    masked_data = np.ma.masked_where(band_data == nodata_value, band_data)
    
    min_val = np.ma.min(masked_data)
    max_val = np.ma.max(masked_data)
    mean_val = np.ma.mean(masked_data)
    
    data_ranges.append((min_val, max_val))
    
    print(f"Band {band_idx:3d}: Min={min_val:.6f}, Max={max_val:.6f}, Mean={mean_val:.6f}")

# Calculate overall min/max for consistent scaling
overall_min = min(min_val for min_val, _ in data_ranges)
overall_max = max(max_val for _, max_val in data_ranges)
print(f"\nOverall data range: Min={overall_min:.6f}, Max={overall_max:.6f}")

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
    selected_bands = []
    wavelength_mapping = []
    
    for i, wl_nm in enumerate(csv_wavelengths):
        # Find the closest matching band in AVIRIS data
        band_idx = np.abs(aviris_wavelengths - wl_nm).argmin()
        
        # Avoid duplicates
        if band_idx not in selected_bands:
            selected_bands.append(band_idx)
            wavelength_mapping.append({
                'CSV_Index': i,
                'CSV_Wavelength_nm': wl_nm,
                'CSV_Wavelength_um': wl_nm/1000,
                'AVIRIS_Band': band_idx,
                'AVIRIS_Wavelength_nm': aviris_wavelengths[band_idx],
                'AVIRIS_Wavelength_um': aviris_wavelengths[band_idx]/1000,
                'Difference_nm': abs(wl_nm - aviris_wavelengths[band_idx])
            })
    
    # Create a DataFrame for better display
    mapping_df = pd.DataFrame(wavelength_mapping)
    print(f"\nSelected {len(selected_bands)} unique AVIRIS bands")
    print("First 10 wavelength mappings:")
    print(mapping_df.head(10).to_string(index=False))
    
    # Extract the selected bands and save them to a new file
    print(f"\nExtracting selected bands...")
    
    # Prepare a new data array for selected bands
    rows, cols, _ = img.shape
    new_data = np.zeros((rows, cols, len(selected_bands)), dtype=np.float32)
    
    # Create a list to store band statistics
    band_stats = []
    
    # Extract the selected bands with tqdm progress bar
    for i, band_idx in enumerate(tqdm(selected_bands, desc="Extracting bands")):
        band_data = img.read_band(band_idx)
        
        # Get statistics for this band (mask nodata values)
        nodata_value = -9999.0
        masked_data = np.ma.masked_where(band_data == nodata_value, band_data)
        
        min_val = np.ma.min(masked_data)
        max_val = np.ma.max(masked_data)
        mean_val = np.ma.mean(masked_data)
        std_val = np.ma.std(masked_data)
        
        # Store the stats
        band_stats.append({
            'Position': i,
            'AVIRIS_Band': band_idx,
            'Wavelength_nm': aviris_wavelengths[band_idx],
            'Min_Value': min_val,
            'Max_Value': max_val, 
            'Mean_Value': mean_val,
            'Std_Dev': std_val,
            'Valid_Pixels': np.ma.count(masked_data),
            'Invalid_Pixels': np.ma.count_masked(masked_data)
        })
        
        # Store the band data in our new array
        new_data[:,:,i] = band_data
    
    # Convert band stats to DataFrame and display summary
    stats_df = pd.DataFrame(band_stats)
    print("\nBand Statistics Summary:")
    print(f"Min value across all bands: {stats_df['Min_Value'].min()}")
    print(f"Max value across all bands: {stats_df['Max_Value'].max()}")
    print(f"Mean of mean values: {stats_df['Mean_Value'].mean()}")
    
    # Identify potential problematic bands (with extreme values or constant values)
    problematic_bands = stats_df[
        (stats_df['Min_Value'] == stats_df['Max_Value']) |  # Constant value bands
        ((np.abs(stats_df['Min_Value']) < 1e-5) & (np.abs(stats_df['Max_Value']) < 1e-5))  # Near-zero bands
    ]
    
    if not problematic_bands.empty:
        print("\nPotential problem bands (constant or near-zero values):")
        print(problematic_bands[['Position', 'AVIRIS_Band', 'Wavelength_nm', 'Min_Value', 'Max_Value']].to_string(index=False))
    
    # Create a new header dictionary based on the original
    original_header = envi.read_envi_header(hdr_file)
    
    # Update the header for the new file
    new_header = original_header.copy()
    new_header['bands'] = len(selected_bands)
    
    # Update wavelength information
    if 'wavelength' in new_header:
        new_header['wavelength'] = [str(aviris_wavelengths[idx]) for idx in selected_bands]
    
    # Write the new data to an ENVI file
    # First, make sure we're using the same format as the original
    print(f"Writing selected bands to: {output_img_file}")
    envi.save_image(output_hdr_file, new_data, metadata=new_header, force=True)
    
    # Check if we need to rename the file to match the original format
    # ENVI save_image automatically adds .img extension, but original might not have it
    if os.path.exists(f"{output_img_file}.img") and not os.path.exists(f"{base_folder}/{img_file}.img"):
        print(f"Renaming output file to match original format")
        shutil.move(f"{output_img_file}.img", output_img_file)
    
    # Save data in NumPy (.npy) format
    print(f"Saving data in NumPy format: {numpy_folder}/aviris_swir.npy")
    np.save(f"{numpy_folder}/aviris_swir.npy", new_data)
    
    # Save wavelength information with the NumPy data
    selected_wavelengths = np.array([aviris_wavelengths[idx] for idx in selected_bands])
    np.save(f"{numpy_folder}/wavelengths.npy", selected_wavelengths)
    
    # Save data in PyTorch (.pt) format
    print(f"Saving data in PyTorch format: {torch_folder}/aviris_swir.pt")
    torch_data = torch.from_numpy(new_data)
    torch.save(torch_data, f"{torch_folder}/aviris_swir.pt")
    
    # Save wavelength information with the PyTorch data
    torch_wavelengths = torch.from_numpy(selected_wavelengths)
    torch.save(torch_wavelengths, f"{torch_folder}/wavelengths.pt")
    
    # Plot specific frames (1, 25, 50, 100) from the new dataset
    print(f"\nCreating visualizations of selected frames...")
    
    # Load the new data file to verify it worked
    new_img = spectral.open_image(output_hdr_file)
    
    # Plot specific frames
    frames_to_plot = [0, 24, 49, 99]  # 0-indexed, so these are frames 1, 25, 50, 100
    frames_to_plot = [idx for idx in frames_to_plot if idx < len(selected_bands)]  # Make sure we don't go out of bounds
    
    # Create a figure with consistent scaling
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Calculate consistent scaling for visualization across all valid data
    all_valid_data = []
    for frame_idx in frames_to_plot:
        band_data = new_img.read_band(frame_idx)
        valid_data = band_data[band_data != -9999.0]
        all_valid_data.extend(valid_data)
    
    if all_valid_data:
        p2, p98 = np.percentile(all_valid_data, (2, 98))
        vmin, vmax = p2, p98
    else:
        vmin, vmax = 0, 1  # Fallback if no valid data
    
    print(f"Using consistent scale for visualization: vmin={vmin:.4f}, vmax={vmax:.4f}")
    
    # Save each individual band as an image
    print(f"Saving individual band images to: {images_folder}")
    for frame_idx in tqdm(range(len(selected_bands)), desc="Saving band images"):
        # Get the band data
        band_data = new_img.read_band(frame_idx)
        
        # Use masked array for nodata values
        nodata_value = -9999.0
        masked_data = np.ma.masked_where(band_data == nodata_value, band_data)
        
        # Normalize the data for visualization
        norm_data = np.clip((masked_data - vmin) / (vmax - vmin), 0, 1)
        
        # Create a figure for this band
        plt.figure(figsize=(8, 8))
        plt.imshow(norm_data, cmap='viridis')
        
        # Get the wavelength for this band
        original_band_idx = selected_bands[frame_idx]
        original_wavelength = aviris_wavelengths[original_band_idx]
        
        plt.title(f'Band {frame_idx+1}: {original_wavelength:.2f} nm ({original_wavelength/1000:.2f} μm)')
        plt.colorbar(label='Normalized Reflectance')
        plt.tight_layout()
        
        # Save the figure
        wavelength_str = f"{original_wavelength:.2f}".replace('.', 'p')
        plt.savefig(f'{images_folder}/band_{frame_idx+1:03d}_{wavelength_str}nm.png', dpi=150)
        plt.close()
    
    for i, frame_idx in enumerate(frames_to_plot):
        if i < len(axes):
            # Get statistics for this frame from our precomputed stats
            frame_stats = stats_df.iloc[frame_idx]
            print(f"\nFrame {frame_idx+1} (Band {frame_stats['AVIRIS_Band']}, {frame_stats['Wavelength_nm']:.2f} nm):")
            print(f"  Min: {frame_stats['Min_Value']:.6f}, Max: {frame_stats['Max_Value']:.6f}")
            print(f"  Mean: {frame_stats['Mean_Value']:.6f}, Std: {frame_stats['Std_Dev']:.6f}")
            
            # Read the band directly
            band_data = new_img.read_band(frame_idx)
            
            # Use masked array for nodata values
            nodata_value = -9999.0
            masked_data = np.ma.masked_where(band_data == nodata_value, band_data)
            
            # Plot on the corresponding subplot with consistent scale
            im = axes[i].imshow(masked_data, cmap='viridis', vmin=vmin, vmax=vmax)
            
            # Get the original wavelength for this band
            original_band_idx = selected_bands[frame_idx]
            original_wavelength = aviris_wavelengths[original_band_idx]
            
            axes[i].set_title(f'Frame {frame_idx+1}: {original_wavelength:.2f} nm ({original_wavelength/1000:.2f} μm)')
            plt.colorbar(im, ax=axes[i], label='Reflectance')
    
    # Hide any unused subplots
    for i in range(len(frames_to_plot), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/selected_frames.png', dpi=300)
    plt.show()
    
    # Save the wavelength mapping to a CSV file for reference
    mapping_df.to_csv(f'{output_folder}/wavelength_mapping.csv', index=False)
    print(f"Saved wavelength mapping to: {output_folder}/wavelength_mapping.csv")
    
    # Save the band statistics to a CSV file for reference
    stats_df.to_csv(f'{output_folder}/band_statistics.csv', index=False)
    print(f"Saved band statistics to: {output_folder}/band_statistics.csv")
    
    # Generate metadata files for the NumPy and PyTorch exports
    # Convert NumPy types to Python native types to avoid JSON serialization issues
    metadata = {
        'shape': tuple(int(x) for x in new_data.shape),
        'wavelengths_nm': [float(x) for x in selected_wavelengths.tolist()],
        'wavelengths_um': [float(x) for x in (selected_wavelengths / 1000).tolist()],
        'nodata_value': float(-9999.0),
        'min_value': float(stats_df['Min_Value'].min()),
        'max_value': float(stats_df['Max_Value'].max()),
        'mean_value': float(stats_df['Mean_Value'].mean()),
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
    print(f"  - ENVI format: {output_img_file}")
    print(f"  - NumPy format: {numpy_folder}/aviris_swir.npy")
    print(f"  - PyTorch format: {torch_folder}/aviris_swir.pt")
    print(f"  - Individual band images: {images_folder}/")

else:
    print("No wavelength information found in the image header.")
    exit(1)
