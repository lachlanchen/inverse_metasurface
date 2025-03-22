import numpy as np
import matplotlib.pyplot as plt
import spectral
import os
import pandas as pd
from spectral.io import envi
from tqdm import tqdm
import shutil
import torch
import json
from pathlib import Path
from scipy import stats

def process_aviris_folder(base_folder, subfolder, csv_file, output_base, std_threshold=3.0):
    """
    Process a single AVIRIS data folder with improved outlier handling
    
    Parameters:
    -----------
    base_folder : str
        Base AVIRIS directory
    subfolder : str
        Name of the subfolder to process
    csv_file : str
        Path to CSV file with wavelength data
    output_base : str
        Base output directory
    std_threshold : float
        Number of standard deviations to use for outlier detection
    """
    print(f"\n{'='*80}")
    print(f"Processing {subfolder}")
    print(f"{'='*80}")
    
    # Define paths
    folder_path = os.path.join(base_folder, subfolder)
    
    # Find the image file (assuming it's the one without .hdr extension)
    img_files = [f for f in os.listdir(folder_path) 
               if not f.endswith('.hdr') and not f.endswith('.py') and not f.endswith('.png')]
    
    if not img_files:
        print(f"No image files found in {folder_path}")
        return False
    
    img_file = img_files[0]
    hdr_file = f'{folder_path}/{img_file}.hdr'
    
    # Check if header file exists
    if not os.path.exists(hdr_file):
        print(f"Header file not found: {hdr_file}")
        return False
    
    # Create output directory structure
    output_folder = os.path.join(output_base, subfolder)
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
    try:
        img = spectral.open_image(hdr_file)
        print(f"Image dimensions: {img.shape}")
        print(f"Number of bands: {img.nbands}")
    except Exception as e:
        print(f"Error opening image: {e}")
        return False

    # Sample a few bands to get an idea of the original data range
    sample_bands = [0, min(50, img.nbands-1), min(100, img.nbands-1), 
                   min(150, img.nbands-1), min(200, img.nbands-1), min(250, img.nbands-1)]
    sample_bands = [b for b in sample_bands if b < img.nbands]
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
    try:
        csv_data = pd.read_csv(csv_file)
        csv_wavelengths = csv_data['Wavelength_um'].to_numpy() * 1000  # Convert μm to nm
        print(f"Found {len(csv_wavelengths)} wavelengths in CSV file")
        print(f"CSV wavelength range: {np.min(csv_wavelengths):.2f} to {np.max(csv_wavelengths):.2f} nm")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False

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
            
            # Check if this is a constant-value band
            is_constant = (min_val == max_val) or std_val < 1e-6
            
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
                'Invalid_Pixels': np.ma.count_masked(masked_data),
                'Is_Constant': is_constant
            })
            
            # Store the band data in our new array
            new_data[:,:,i] = band_data
        
        # Convert band stats to DataFrame and display summary
        stats_df = pd.DataFrame(band_stats)
        print("\nBand Statistics Summary:")
        print(f"Min value across all bands: {stats_df['Min_Value'].min()}")
        print(f"Max value across all bands: {stats_df['Max_Value'].max()}")
        print(f"Mean of mean values: {stats_df['Mean_Value'].mean()}")
        
        # Identify constant value bands
        constant_bands = stats_df[stats_df['Is_Constant']]
        if not constant_bands.empty:
            print(f"\nDetected {len(constant_bands)} constant-value bands:")
            print(constant_bands[['Position', 'AVIRIS_Band', 'Wavelength_nm', 'Min_Value']].head().to_string(index=False))
            print("These bands will be handled specially during processing.")
        
        # Collect all valid data points to compute global statistics
        print("\nComputing global statistics for outlier detection...")
        all_valid_data = []
        sample_size = min(1000, rows * cols)  # Limit sample size for memory efficiency
        
        for i in tqdm(range(len(selected_bands)), desc="Sampling data"):
            if stats_df.iloc[i]['Is_Constant']:
                continue  # Skip constant bands
                
            band_data = new_data[:,:,i].flatten()
            # Filter out nodata values
            valid_mask = band_data != -9999.0
            
            if np.sum(valid_mask) > 0:
                # Random sampling to keep memory usage reasonable
                valid_data = band_data[valid_mask]
                if len(valid_data) > sample_size:
                    indices = np.random.choice(len(valid_data), sample_size, replace=False)
                    all_valid_data.extend(valid_data[indices].tolist())
                else:
                    all_valid_data.extend(valid_data.tolist())
        
        # Compute global statistics
        global_mean = np.mean(all_valid_data)
        global_std = np.std(all_valid_data)
        
        # Define outlier thresholds based on normal distribution
        lower_threshold = global_mean - std_threshold * global_std
        upper_threshold = global_mean + std_threshold * global_std
        
        print(f"\nGlobal statistics for outlier detection:")
        print(f"Mean: {global_mean:.6f}")
        print(f"Standard Deviation: {global_std:.6f}")
        print(f"Lower threshold ({std_threshold} std): {lower_threshold:.6f}")
        print(f"Upper threshold ({std_threshold} std): {upper_threshold:.6f}")
        
        # Clean up extreme values using statistical approach
        cleaned_data = new_data.copy()
        
        # For each band, handle outliers
        print("\nCleaning extreme values using statistical approach...")
        for i in tqdm(range(len(selected_bands)), desc="Cleaning extreme values"):
            band_stats_row = stats_df.iloc[i]
            
            # Skip processing if band is constant
            if band_stats_row['Is_Constant']:
                # For constant bands, set to a neutral value (like the global mean)
                # but preserve nodata values
                nodata_mask = cleaned_data[:,:,i] == -9999.0
                valid_mask = ~nodata_mask
                
                # Set valid pixels to global mean or another appropriate value
                cleaned_data[valid_mask, i] = global_mean
                continue
            
            # Create mask for nodata values
            nodata_mask = cleaned_data[:,:,i] == -9999.0
            valid_mask = ~nodata_mask
            
            # Get band data for valid pixels
            valid_data = cleaned_data[valid_mask, i]
            
            # For band-specific outlier detection, can also use band-specific statistics
            band_mean = band_stats_row['Mean_Value']
            band_std = band_stats_row['Std_Dev']
            
            # Handle case where std is very small or zero
            if band_std < 1e-6:
                # Use global stats instead
                band_lower = lower_threshold
                band_upper = upper_threshold
            else:
                # Use a mix of global and band-specific thresholds for better results
                band_lower = max(lower_threshold, band_mean - std_threshold * band_std)
                band_upper = min(upper_threshold, band_mean + std_threshold * band_std)
            
            # Clip values outside threshold range
            np.clip(valid_data, band_lower, band_upper, out=valid_data)
            
            # Update the cleaned data
            cleaned_data[valid_mask, i] = valid_data
        
        # Calculate new min/max after cleaning
        new_min = np.inf
        new_max = -np.inf
        
        for i in range(len(selected_bands)):
            if stats_df.iloc[i]['Is_Constant']:
                continue  # Skip constant bands for min/max calculation
                
            band_data = cleaned_data[:,:,i]
            nodata_mask = band_data == -9999.0
            valid_mask = ~nodata_mask
            
            if np.any(valid_mask):
                band_min = np.min(band_data[valid_mask])
                band_max = np.max(band_data[valid_mask])
                
                new_min = min(new_min, band_min)
                new_max = max(new_max, band_max)
        
        print(f"\nData range after cleaning: Min={new_min:.6f}, Max={new_max:.6f}")
        
        # Rescale data to 0-1 range for valid pixels
        rescaled_data = cleaned_data.copy()
        
        # For each band, rescale valid data to 0-1
        print("\nRescaling data to 0-1 range...")
        for i in tqdm(range(len(selected_bands)), desc="Rescaling data"):
            band_data = rescaled_data[:,:,i]
            nodata_mask = band_data == -9999.0
            valid_mask = ~nodata_mask
            
            if stats_df.iloc[i]['Is_Constant']:
                # For constant bands, set to a neutral value (0.5)
                rescaled_data[valid_mask, i] = 0.5
                continue
            
            # Calculate min/max for this band (after cleaning)
            band_min = np.min(band_data[valid_mask]) if np.any(valid_mask) else new_min
            band_max = np.max(band_data[valid_mask]) if np.any(valid_mask) else new_max
            
            # Avoid division by zero
            if abs(band_max - band_min) > 1e-6:
                # Rescale valid data to 0-1
                rescaled_data[valid_mask, i] = (band_data[valid_mask] - band_min) / (band_max - band_min)
            else:
                # If min == max, set all valid values to 0.5
                rescaled_data[valid_mask, i] = 0.5
            
            # Ensure nodata values remain as nodata
            rescaled_data[nodata_mask, i] = -9999.0
        
        # Create a new header dictionary based on the original
        original_header = envi.read_envi_header(hdr_file)
        
        # Update the header for the new file
        new_header = original_header.copy()
        new_header['bands'] = len(selected_bands)
        
        # Update wavelength information
        if 'wavelength' in new_header:
            new_header['wavelength'] = [str(aviris_wavelengths[idx]) for idx in selected_bands]
        
        # Write the new data to an ENVI file
        print(f"Writing selected bands to: {output_img_file}")
        envi.save_image(output_hdr_file, rescaled_data, metadata=new_header, force=True)
        
        # Check if we need to rename the file to match the original format
        if os.path.exists(f"{output_img_file}.img") and not os.path.exists(f"{folder_path}/{img_file}.img"):
            print(f"Renaming output file to match original format")
            shutil.move(f"{output_img_file}.img", output_img_file)
        
        # Save data in NumPy (.npy) format
        print(f"Saving data in NumPy format: {numpy_folder}/aviris_swir.npy")
        np.save(f"{numpy_folder}/aviris_swir.npy", rescaled_data)
        
        # Save wavelength information with the NumPy data
        selected_wavelengths = np.array([aviris_wavelengths[idx] for idx in selected_bands])
        np.save(f"{numpy_folder}/wavelengths.npy", selected_wavelengths)
        
        # Save data in PyTorch (.pt) format
        print(f"Saving data in PyTorch format: {torch_folder}/aviris_swir.pt")
        torch_data = torch.from_numpy(rescaled_data)
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
        
        # For rescaled data, use 0-1 range
        vmin, vmax = 0, 1
        
        print(f"Using consistent scale for visualization: vmin={vmin:.4f}, vmax={vmax:.4f}")
        
        # Save each individual band as an image
        print(f"Saving individual band images to: {images_folder}")
        for frame_idx in tqdm(range(len(selected_bands)), desc="Saving band images"):
            # Get the band data
            band_data = new_img.read_band(frame_idx)
            
            # Use masked array for nodata values
            nodata_value = -9999.0
            masked_data = np.ma.masked_where(band_data == nodata_value, band_data)
            
            # Create a figure for this band
            plt.figure(figsize=(8, 8))
            plt.imshow(masked_data, cmap='viridis', vmin=vmin, vmax=vmax)
            
            # Get the wavelength for this band
            original_band_idx = selected_bands[frame_idx]
            original_wavelength = aviris_wavelengths[original_band_idx]
            
            # Note if this is a constant band
            is_constant = stats_df.iloc[frame_idx]['Is_Constant']
            constant_note = " (CONSTANT BAND)" if is_constant else ""
            
            plt.title(f'Band {frame_idx+1}: {original_wavelength:.2f} nm ({original_wavelength/1000:.2f} μm){constant_note}')
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
                print(f"  Constant band: {frame_stats['Is_Constant']}")
                
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
                
                # Note if this is a constant band
                is_constant = frame_stats['Is_Constant']
                constant_note = " (CONSTANT BAND)" if is_constant else ""
                
                axes[i].set_title(f'Frame {frame_idx+1}: {original_wavelength:.2f} nm ({original_wavelength/1000:.2f} μm){constant_note}')
                plt.colorbar(im, ax=axes[i], label='Reflectance')
        
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
            'nodata_value': float(-9999.0),
            'min_value': float(0.0),  # After rescaling
            'max_value': float(1.0),  # After rescaling
            'original_min': float(stats_df['Min_Value'].min()),
            'original_max': float(stats_df['Max_Value'].max()),
            'mean_value': float(stats_df['Mean_Value'].mean()),
            'original_bands': [int(x) for x in selected_bands],
            'std_threshold_used': float(std_threshold),
            'global_mean': float(global_mean),
            'global_std': float(global_std),
            'constant_bands': [int(i) for i in stats_df[stats_df['Is_Constant']].index.tolist()]
        }
        
        # Save metadata as JSON
        with open(f"{numpy_folder}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        with open(f"{torch_folder}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\nProcessing complete for this folder!")
        print(f"Data saved to:")
        print(f"  - ENVI format: {output_img_file}")
        print(f"  - NumPy format: {numpy_folder}/aviris_swir.npy")
        print(f"  - PyTorch format: {torch_folder}/aviris_swir.pt")
        print(f"  - Individual band images: {images_folder}/")
        
        return True

    else:
        print("No wavelength information found in the image header.")
        return False


def main():
    # Define paths
    base_folder = 'AVIRIS'
    output_base = 'AVIRIS_SWIR'
    csv_file = 'partial_crys_data/partial_crys_C0.0.csv'
    
    # Statistical threshold for outlier detection (number of standard deviations)
    std_threshold = 3.0
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_base):
        os.makedirs(output_base)
    
    # Get list of subfolders in AVIRIS directory
    subfolders = [f for f in os.listdir(base_folder) 
                  if os.path.isdir(os.path.join(base_folder, f))]
    
    if not subfolders:
        print(f"No subfolders found in {base_folder}")
        return
    
    print(f"Found {len(subfolders)} subfolders to process: {', '.join(subfolders)}")
    
    # Process each subfolder
    results = {}
    for subfolder in subfolders:
        success = process_aviris_folder(base_folder, subfolder, csv_file, output_base, std_threshold)
        results[subfolder] = "Success" if success else "Failed"
    
    # Print summary
    print("\n" + "="*80)
    print("Processing Summary")
    print("="*80)
    for subfolder, result in results.items():
        print(f"{subfolder}: {result}")


if __name__ == "__main__":
    main()
