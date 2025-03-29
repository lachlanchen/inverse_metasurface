#!/usr/bin/env python3
"""
AVIRIS SWIR Band Selector with Interpolation
--------------------------------------------
Selects SWIR bands from AVIRIS data, removes bad bands through interpolation,
handles masked regions, and normalizes data to 0-1 range.
Specifically selects wavelengths in the 1000nm to 2.5μm range.

Features:
- Skips already processed files
- Uses chunked processing to avoid memory issues
- Creates detailed visualizations of results
"""

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
import scipy.interpolate
import seaborn as sns
import argparse
import time

def remove_and_interpolate_bad_bands_chunked(wavelengths, band_data, bad_ranges=None, chunk_size=500):
    """
    Remove bad bands and interpolate their values using a chunked approach to save memory.
    
    Parameters:
    -----------
    wavelengths : numpy.ndarray
        Array of wavelength values in nm
    band_data : numpy.ndarray
        Array of band data to interpolate
    bad_ranges : list of tuples, optional
        List of (min_wavelength, max_wavelength) tuples defining bad band ranges
        Default is [(1263, 1562), (1761, 1958)] nm
    chunk_size : int
        Number of rows to process at once
        
    Returns:
    --------
    interpolated_data : numpy.ndarray
        Band data with bad bands replaced by interpolated values
    bad_bands_mask : numpy.ndarray
        Boolean mask where True indicates a bad band
    """
    if bad_ranges is None:
        # Default bad wavelength ranges in nm 
        bad_ranges = [(1263, 1562), (1761, 1958)]
    
    # Initialize mask (all False = all good bands)
    bad_bands_mask = np.zeros_like(wavelengths, dtype=bool)
    
    # Mark bad bands in each range
    for min_wl, max_wl in bad_ranges:
        bad_bands_mask |= (wavelengths >= min_wl) & (wavelengths <= max_wl)
    
    # Print info about removed bands
    bad_band_indices = np.where(bad_bands_mask)[0]
    
    if len(bad_band_indices) > 0:
        print(f"Identified {len(bad_band_indices)} bad bands:")
        for i, band_range in enumerate(np.split(bad_band_indices, np.where(np.diff(bad_band_indices) != 1)[0] + 1)):
            start_idx, end_idx = band_range[0], band_range[-1]
            start_wl, end_wl = wavelengths[start_idx], wavelengths[end_idx]
            print(f"  Range {i+1}: Bands {start_idx}-{end_idx} ({start_wl:.2f}-{end_wl:.2f} nm)")
    else:
        print("No bad bands identified within the specified wavelength ranges")
    
    # If no bad bands, return original data
    if not np.any(bad_bands_mask):
        return band_data, bad_bands_mask
    
    # Create a copy of the data for interpolation
    interpolated_data = band_data.copy()
    
    # Get dimensions
    rows, cols, num_bands = band_data.shape
    
    # Process data in chunks to save memory
    print("Interpolating bad bands for each pixel (chunked processing)...")
    num_chunks = (rows + chunk_size - 1) // chunk_size
    
    for chunk_idx in tqdm(range(num_chunks), desc="Processing row chunks"):
        # Calculate chunk bounds
        start_row = chunk_idx * chunk_size
        end_row = min(start_row + chunk_size, rows)
        
        # Process each row in this chunk
        for r in range(start_row, end_row):
            for c in range(cols):
                # Get the spectrum for this pixel
                spectrum = band_data[r, c, :]
                
                # Check if this is a valid pixel (not all nodata)
                nodata_mask = spectrum == -9999.0
                if np.all(nodata_mask):
                    # Skip all-nodata pixels
                    continue
                
                # Extract good bands for this pixel
                good_mask = ~(bad_bands_mask | nodata_mask)
                
                # If we don't have enough good bands for interpolation, skip
                if np.sum(good_mask) < 2:
                    continue
                
                # Get wavelengths and values for good bands
                good_wavelengths = wavelengths[good_mask]
                good_values = spectrum[good_mask]
                
                # Create interpolation function
                try:
                    interp_func = scipy.interpolate.interp1d(
                        good_wavelengths, good_values,
                        kind='linear', bounds_error=False, fill_value='extrapolate'
                    )
                    
                    # Interpolate only the bad bands that aren't nodata
                    interp_mask = bad_bands_mask & ~nodata_mask
                    if np.any(interp_mask):
                        interpolated_data[r, c, interp_mask] = interp_func(wavelengths[interp_mask])
                except Exception as e:
                    # If interpolation fails, just leave the original values
                    pass
    
    return interpolated_data, bad_bands_mask

def is_processing_complete(output_folder):
    """
    Check if a folder has been fully processed by looking for key output files.
    
    Parameters:
    -----------
    output_folder : str
        Path to the output folder
        
    Returns:
    --------
    bool
        True if processing is complete, False otherwise
    """
    # Check for key output files
    numpy_file = os.path.join(output_folder, "numpy", "aviris_swir.npy")
    torch_file = os.path.join(output_folder, "torch", "aviris_swir.pt")
    stats_file = os.path.join(output_folder, "stats", "data_distribution.png")
    
    # If all files exist, consider processing complete
    if os.path.exists(numpy_file) and os.path.exists(torch_file) and os.path.exists(stats_file):
        # Check file sizes to ensure they're not empty
        if (os.path.getsize(numpy_file) > 1000 and 
            os.path.getsize(torch_file) > 1000 and 
            os.path.getsize(stats_file) > 1000):
            return True
    
    return False

def save_checkpoint(subfolder, data, checkpoint_name, output_base):
    """
    Save a checkpoint during processing
    
    Parameters:
    -----------
    subfolder : str
        Name of the subfolder being processed
    data : numpy.ndarray
        Data to save as checkpoint
    checkpoint_name : str
        Name of the checkpoint
    output_base : str
        Base output directory
    """
    checkpoint_dir = os.path.join(output_base, subfolder, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_file = os.path.join(checkpoint_dir, f"{checkpoint_name}.npy")
    np.save(checkpoint_file, data)
    print(f"Saved checkpoint: {checkpoint_file}")

def load_checkpoint(subfolder, checkpoint_name, output_base):
    """
    Load a checkpoint from a previous run
    
    Parameters:
    -----------
    subfolder : str
        Name of the subfolder
    checkpoint_name : str
        Name of the checkpoint
    output_base : str
        Base output directory
        
    Returns:
    --------
    numpy.ndarray or None
        Loaded data if checkpoint exists, None otherwise
    """
    checkpoint_file = os.path.join(output_base, subfolder, "checkpoints", f"{checkpoint_name}.npy")
    
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint: {checkpoint_file}")
        try:
            return np.load(checkpoint_file)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    return None

def process_aviris_folder(base_folder, subfolder, csv_file, output_base, std_threshold=3.0, chunk_size=500, 
                         force_reprocess=False):
    """
    Process a single AVIRIS data folder with improved outlier handling
    and bad band interpolation
    
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
    chunk_size : int
        Number of rows to process at once during interpolation
    force_reprocess : bool
        If True, reprocess even if output files exist
    """
    print(f"\n{'='*80}")
    print(f"Processing {subfolder}")
    print(f"{'='*80}")
    
    # Define paths
    folder_path = os.path.join(base_folder, subfolder)
    
    # Create output directory structure
    output_folder = os.path.join(output_base, subfolder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check if this folder has already been processed
    if is_processing_complete(output_folder) and not force_reprocess:
        print(f"Skipping {subfolder} - already processed")
        return True
    
    # Create additional directories for NumPy and PyTorch exports
    numpy_folder = f'{output_folder}/numpy'
    torch_folder = f'{output_folder}/torch'
    images_folder = f'{output_folder}/images'
    stats_folder = f'{output_folder}/stats'  # For statistics
    checkpoint_folder = f'{output_folder}/checkpoints'  # For intermediate results

    for folder in [numpy_folder, torch_folder, images_folder, stats_folder, checkpoint_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
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
        
        # Filter wavelengths to include only SWIR range (1000nm to 2.5μm)
        swir_min = 1000.0  # nm
        swir_max = 2500.0  # nm (2.5 μm)
        
        # Filter CSV wavelengths to SWIR range
        csv_swir_mask = (csv_wavelengths >= swir_min) & (csv_wavelengths <= swir_max)
        csv_swir_wavelengths = csv_wavelengths[csv_swir_mask]
        
        print(f"Filtered CSV wavelengths to SWIR range ({swir_min}-{swir_max} nm)")
        print(f"Found {len(csv_swir_wavelengths)} wavelengths in SWIR range")
        
        # For each wavelength in the filtered CSV, find the closest band in AVIRIS
        selected_bands = []
        wavelength_mapping = []
        
        for i, wl_nm in enumerate(csv_swir_wavelengths):
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
        print(f"\nSelected {len(selected_bands)} unique AVIRIS bands in SWIR range")
        print("First 10 wavelength mappings:")
        print(mapping_df.head(10).to_string(index=False))
        
        # Save the wavelength mapping early in case we crash later
        os.makedirs(output_folder, exist_ok=True)
        mapping_df.to_csv(f'{output_folder}/wavelength_mapping.csv', index=False)
        
        # Get selected wavelengths from the mapping
        selected_wavelengths = np.array([aviris_wavelengths[idx] for idx in selected_bands])
        
        # Check for band extraction checkpoint
        extracted_data = load_checkpoint(subfolder, "extracted_bands", output_base)
        
        if extracted_data is None:
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
            
            # Save checkpoint after extraction
            save_checkpoint(subfolder, new_data, "extracted_bands", output_base)
            
            # Save band statistics
            stats_df = pd.DataFrame(band_stats)
            stats_df.to_csv(f'{output_folder}/band_statistics.csv', index=False)
        else:
            # Load existing data and statistics
            new_data = extracted_data
            print("Using previously extracted bands from checkpoint")
            
            # Check if statistics file exists
            stats_file = f'{output_folder}/band_statistics.csv'
            if os.path.exists(stats_file):
                stats_df = pd.read_csv(stats_file)
                print("Loaded existing band statistics")
            else:
                # Recreate statistics if file doesn't exist
                print("Regenerating band statistics...")
                band_stats = []
                rows, cols, num_bands = new_data.shape
                
                for i in tqdm(range(num_bands), desc="Computing statistics"):
                    band_data = new_data[:,:,i]
                    
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
                        'AVIRIS_Band': selected_bands[i],
                        'Wavelength_nm': aviris_wavelengths[selected_bands[i]],
                        'Min_Value': min_val,
                        'Max_Value': max_val, 
                        'Mean_Value': mean_val,
                        'Std_Dev': std_val,
                        'Valid_Pixels': np.ma.count(masked_data),
                        'Invalid_Pixels': np.ma.count_masked(masked_data),
                        'Is_Constant': is_constant
                    })
                
                stats_df = pd.DataFrame(band_stats)
                stats_df.to_csv(f'{output_folder}/band_statistics.csv', index=False)
        
        # Check for interpolation checkpoint
        interpolated_data = load_checkpoint(subfolder, "interpolated_bands", output_base)
        
        if interpolated_data is None:
            # Identify and interpolate bad bands
            print(f"\nIdentifying and interpolating bad bands...")
            interpolated_data, bad_bands_mask = remove_and_interpolate_bad_bands_chunked(
                selected_wavelengths, new_data, chunk_size=chunk_size)
            
            # Save checkpoint after interpolation
            save_checkpoint(subfolder, interpolated_data, "interpolated_bands", output_base)
            save_checkpoint(subfolder, bad_bands_mask, "bad_bands_mask", output_base)
        else:
            # Load existing bad bands mask
            bad_bands_mask = load_checkpoint(subfolder, "bad_bands_mask", output_base)
            if bad_bands_mask is None:
                # If mask isn't found, recreate it
                print("Recreating bad bands mask...")
                bad_bands_mask = np.zeros_like(selected_wavelengths, dtype=bool)
                
                # Default bad wavelength ranges in nm 
                bad_ranges = [(1263, 1562), (1761, 1958)]
                
                # Mark bad bands in each range
                for min_wl, max_wl in bad_ranges:
                    bad_bands_mask |= (selected_wavelengths >= min_wl) & (selected_wavelengths <= max_wl)
            
            print(f"Using previously interpolated bands from checkpoint")
        
        # Replace new_data with interpolated_data for further processing
        new_data = interpolated_data
        
        # Convert band stats to DataFrame and display summary
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
        
        # Check for global stats checkpoint
        global_stats = load_checkpoint(subfolder, "global_stats", output_base)
        
        if global_stats is None:
            # Collect all valid data points to compute global statistics
            print("\nComputing global statistics for outlier detection...")
            all_valid_data = []
            rows, cols, _ = new_data.shape
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
            
            # Save global stats
            global_stats = {
                'mean': global_mean,
                'std': global_std,
                'lower_threshold': global_mean - std_threshold * global_std,
                'upper_threshold': global_mean + std_threshold * global_std
            }
            save_checkpoint(subfolder, np.array([global_mean, global_std, 
                                              global_stats['lower_threshold'], 
                                              global_stats['upper_threshold']]), 
                            "global_stats", output_base)
        else:
            # Load existing global stats
            global_mean = global_stats[0]
            global_std = global_stats[1]
            lower_threshold = global_stats[2]
            upper_threshold = global_stats[3]
            global_stats = {
                'mean': global_mean,
                'std': global_std,
                'lower_threshold': lower_threshold,
                'upper_threshold': upper_threshold
            }
            print("Using previously computed global statistics from checkpoint")
        
        # Define outlier thresholds based on normal distribution
        lower_threshold = global_stats['lower_threshold']
        upper_threshold = global_stats['upper_threshold']
        
        print(f"\nGlobal statistics for outlier detection:")
        print(f"Mean: {global_mean:.6f}")
        print(f"Standard Deviation: {global_std:.6f}")
        print(f"Lower threshold ({std_threshold} std): {lower_threshold:.6f}")
        print(f"Upper threshold ({std_threshold} std): {upper_threshold:.6f}")
        
        # Check for cleaned data checkpoint
        cleaned_data = load_checkpoint(subfolder, "cleaned_data", output_base)
        
        if cleaned_data is None:
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
                valid_data = cleaned_data[:,:,i][valid_mask]
                
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
                cleaned_data[:,:,i][valid_mask] = valid_data
            
            # Save checkpoint after cleaning
            save_checkpoint(subfolder, cleaned_data, "cleaned_data", output_base)
        else:
            print("Using previously cleaned data from checkpoint")
        
        # Check for filled data checkpoint
        filled_data = load_checkpoint(subfolder, "filled_data", output_base)
        
        if filled_data is None:
            # Set masked regions (nodata values) to the mean value of each band
            filled_data = cleaned_data.copy()
            
            print("\nSetting masked regions to mean values...")
            for i in tqdm(range(len(selected_bands)), desc="Processing masked regions"):
                # Get nodata mask for this band
                nodata_mask = filled_data[:,:,i] == -9999.0
                
                # If there are any nodata values
                if np.any(nodata_mask):
                    # Get the mean value for this band
                    valid_mask = ~nodata_mask
                    
                    if np.any(valid_mask):
                        # If we have valid data, use its mean
                        band_mean = np.mean(filled_data[:,:,i][valid_mask])
                    else:
                        # If entire band is nodata, use global mean
                        band_mean = global_mean
                    
                    # Replace nodata values with mean
                    filled_data[:,:,i][nodata_mask] = band_mean
            
            # Save checkpoint after filling nodata
            save_checkpoint(subfolder, filled_data, "filled_data", output_base)
        else:
            print("Using previously filled data from checkpoint")
        
        # Calculate new min/max after cleaning and filling
        new_min = np.inf
        new_max = -np.inf
        
        for i in range(len(selected_bands)):
            if stats_df.iloc[i]['Is_Constant']:
                continue  # Skip constant bands for min/max calculation
                
            band_data = filled_data[:,:,i]
            band_min = np.min(band_data)
            band_max = np.max(band_data)
            
            new_min = min(new_min, band_min)
            new_max = max(new_max, band_max)
        
        print(f"\nData range after cleaning: Min={new_min:.6f}, Max={new_max:.6f}")
        
        # Check for rescaled data checkpoint
        rescaled_data = load_checkpoint(subfolder, "rescaled_data", output_base)
        
        if rescaled_data is None:
            # Rescale data to 0-1 range (with no nodata values since we replaced them)
            rescaled_data = filled_data.copy()
            
            # Store original data for statistics
            original_cleaned_data = filled_data.copy()
            
            # For each band, rescale data to 0-1
            print("\nRescaling data to 0-1 range...")
            for i in tqdm(range(len(selected_bands)), desc="Rescaling data"):
                band_data = rescaled_data[:,:,i]
                
                if stats_df.iloc[i]['Is_Constant']:
                    # For constant bands, set to a neutral value (0.5)
                    rescaled_data[:,:,i] = 0.5
                    continue
                
                # Calculate min/max for this band (after cleaning)
                band_min = np.min(band_data)
                band_max = np.max(band_data)
                
                # Avoid division by zero
                if abs(band_max - band_min) > 1e-6:
                    # Rescale data to 0-1
                    rescaled_data[:,:,i] = (band_data - band_min) / (band_max - band_min)
                else:
                    # If min == max, set all values to 0.5
                    rescaled_data[:,:,i] = 0.5
            
            # Save the original data for comparisons
            save_checkpoint(subfolder, original_cleaned_data, "original_cleaned_data", output_base)
            
            # Save checkpoint after rescaling
            save_checkpoint(subfolder, rescaled_data, "rescaled_data", output_base)
        else:
            print("Using previously rescaled data from checkpoint")
            # Load original data for comparisons
            original_cleaned_data = load_checkpoint(subfolder, "original_cleaned_data", output_base)
            if original_cleaned_data is None:
                # If original data not found, use filled data
                original_cleaned_data = filled_data
        
        # Create visualization of data distribution before and after normalization
        print("\nCreating histograms of data distribution...")
        
        # Sample data for visualization (to avoid memory issues)
        sample_size = min(50000, rows * cols)
        row_indices = np.random.choice(rows, size=min(500, rows), replace=False)
        col_indices = np.random.choice(cols, size=min(100, cols), replace=False)
        
        # Create meshgrid for sampling
        r_mesh, c_mesh = np.meshgrid(row_indices, col_indices, indexing='ij')
        r_samples = r_mesh.flatten()
        c_samples = c_mesh.flatten()
        
        # Sample original cleaned data
        original_samples = []
        for i in range(len(selected_bands)):
            original_samples.extend(original_cleaned_data[r_samples, c_samples, i].flatten())
        
        # Sample rescaled data
        rescaled_samples = []
        for i in range(len(selected_bands)):
            rescaled_samples.extend(rescaled_data[r_samples, c_samples, i].flatten())
        
        # Create histograms
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(original_samples, kde=True, bins=50)
        plt.title('Data Distribution Before Normalization')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        sns.histplot(rescaled_samples, kde=True, bins=50)
        plt.title('Data Distribution After Normalization')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{stats_folder}/data_distribution.png', dpi=300)
        plt.close()
        
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
        
        # Create more detailed visualizations
        print("\nCreating additional visualizations...")
        
        # 1. Plot histogram of all band min/max/mean values
        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(selected_wavelengths, stats_df['Min_Value'], 'b-', label='Min')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Minimum Value')
        plt.title('Minimum Values Across Wavelengths')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 2)
        plt.plot(selected_wavelengths, stats_df['Max_Value'], 'r-', label='Max')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Maximum Value')
        plt.title('Maximum Values Across Wavelengths')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 3)
        plt.plot(selected_wavelengths, stats_df['Mean_Value'], 'g-', label='Mean')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Mean Value')
        plt.title('Mean Values Across Wavelengths')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{stats_folder}/band_statistics.png', dpi=300)
        plt.close()
        
        # 2. Visualize distribution of values across all bands with bad bands highlighted
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        # Plot band means
        plt.plot(range(len(selected_bands)), stats_df['Mean_Value'], 'b-', alpha=0.7)
        
        # Highlight bad bands
        for i in range(len(selected_bands)):
            if bad_bands_mask[i]:
                plt.axvspan(i-0.5, i+0.5, color='r', alpha=0.2)
        
        plt.xlabel('Band Index')
        plt.ylabel('Mean Value')
        plt.title('Mean Values by Band Index with Bad Bands Highlighted')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # Plot band standard deviations
        plt.plot(range(len(selected_bands)), stats_df['Std_Dev'], 'g-', alpha=0.7)
        
        # Highlight bad bands
        for i in range(len(selected_bands)):
            if bad_bands_mask[i]:
                plt.axvspan(i-0.5, i+0.5, color='r', alpha=0.2)
        
        plt.xlabel('Band Index')
        plt.ylabel('Standard Deviation')
        plt.title('Standard Deviations by Band Index with Bad Bands Highlighted')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{stats_folder}/bad_bands_visualization.png', dpi=300)
        plt.close()
        
        # 3. Create a heatmap of correlation between bands
        print("Computing band correlation matrix...")
        
        # Sample bands to compute correlation (to keep memory usage reasonable)
        sample_data = np.zeros((min(5000, rows*cols), len(selected_bands)))
        
        # Randomly sample pixels
        rand_rows = np.random.choice(rows, size=min(500, rows), replace=False)
        rand_cols = np.random.choice(cols, size=min(10, cols), replace=False)
        
        # Extract data for sampled pixels
        sample_idx = 0
        for r in rand_rows:
            for c in rand_cols:
                if sample_idx < sample_data.shape[0]:
                    sample_data[sample_idx, :] = rescaled_data[r, c, :]
                    sample_idx += 1
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(sample_data.T)
        
        # Visualize correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='viridis', vmin=-1, vmax=1)
        plt.title('Band Correlation Matrix')
        plt.savefig(f'{stats_folder}/band_correlation.png', dpi=300)
        plt.close()
        
        # Load the new data file to verify it worked
        new_img = spectral.open_image(output_hdr_file)
        
        # Plot specific frames (1, 25, 50, 100) from the new dataset
        print(f"\nCreating visualizations of selected frames...")
        
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
            
            # Create a figure for this band
            plt.figure(figsize=(8, 8))
            plt.imshow(band_data, cmap='viridis', vmin=vmin, vmax=vmax)
            
            # Get the wavelength for this band
            original_band_idx = selected_bands[frame_idx]
            original_wavelength = aviris_wavelengths[original_band_idx]
            
            # Note if this is a constant band
            is_constant = stats_df.iloc[frame_idx]['Is_Constant']
            constant_note = " (CONSTANT BAND)" if is_constant else ""
            bad_band_note = " (INTERPOLATED)" if (frame_idx < len(bad_bands_mask) and bad_bands_mask[frame_idx]) else ""
            
            plt.title(f'Band {frame_idx+1}: {original_wavelength:.2f} nm ({original_wavelength/1000:.2f} μm)'
                      f'{constant_note}{bad_band_note}')
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
                
                # Plot on the corresponding subplot with consistent scale
                im = axes[i].imshow(band_data, cmap='viridis', vmin=vmin, vmax=vmax)
                
                # Get the original wavelength for this band
                original_band_idx = selected_bands[frame_idx]
                original_wavelength = aviris_wavelengths[original_band_idx]
                
                # Note if this is a constant band
                is_constant = frame_stats['Is_Constant']
                constant_note = " (CONSTANT BAND)" if is_constant else ""
                bad_band_note = " (INTERPOLATED)" if (frame_idx < len(bad_bands_mask) and bad_bands_mask[frame_idx]) else ""
                
                axes[i].set_title(f'Frame {frame_idx+1}: {original_wavelength:.2f} nm ({original_wavelength/1000:.2f} μm)'
                                  f'{constant_note}{bad_band_note}')
                plt.colorbar(im, ax=axes[i], label='Reflectance')
        
        # Hide any unused subplots
        for i in range(len(frames_to_plot), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_folder}/selected_frames.png', dpi=300)
        plt.close()
        
        # 4. Create image showing original vs. normalized data
        print("Creating comparison of original vs. normalized data...")
        
        # Select a few bands for comparison
        comparison_bands = [0, len(selected_bands)//3, 2*len(selected_bands)//3, len(selected_bands)-1]
        comparison_bands = [b for b in comparison_bands if b < len(selected_bands)]
        
        fig, axes = plt.subplots(len(comparison_bands), 2, figsize=(12, 4*len(comparison_bands)))
        
        for i, band_idx in enumerate(comparison_bands):
            # Get original data stats
            band_stats_row = stats_df.iloc[band_idx]
            orig_min = band_stats_row['Min_Value']
            orig_max = band_stats_row['Max_Value']
            
            # Get original band data
            orig_band = original_cleaned_data[:,:,band_idx]
            norm_band = rescaled_data[:,:,band_idx]
            
            # Original data
            im1 = axes[i, 0].imshow(orig_band, cmap='viridis')
            plt.colorbar(im1, ax=axes[i, 0])
            axes[i, 0].set_title(f'Original Band {band_idx+1}: {selected_wavelengths[band_idx]:.2f} nm\n'
                                 f'Range: {orig_min:.4f} to {orig_max:.4f}')
            
            # Normalized data
            im2 = axes[i, 1].imshow(norm_band, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(im2, ax=axes[i, 1])
            axes[i, 1].set_title(f'Normalized Band {band_idx+1}: {selected_wavelengths[band_idx]:.2f} nm\n'
                                 f'Range: 0 to 1')
        
        plt.tight_layout()
        plt.savefig(f'{stats_folder}/original_vs_normalized.png', dpi=300)
        plt.close()
        
        # Save the wavelength mapping to a CSV file for reference
        mapping_df.to_csv(f'{output_folder}/wavelength_mapping.csv', index=False)
        print(f"Saved wavelength mapping to: {output_folder}/wavelength_mapping.csv")
        
        # Save the band statistics to a CSV file for reference
        stats_df.to_csv(f'{output_folder}/band_statistics.csv', index=False)
        print(f"Saved band statistics to: {output_folder}/band_statistics.csv")
        
        # Update metadata to include bad band interpolation information
        metadata = {
            'shape': tuple(int(x) for x in rescaled_data.shape),
            'wavelengths_nm': [float(x) for x in selected_wavelengths.tolist()],
            'wavelengths_um': [float(x) for x in (selected_wavelengths / 1000).tolist()],
            'min_value': float(0.0),  # After rescaling
            'max_value': float(1.0),  # After rescaling
            'original_min': float(stats_df['Min_Value'].min()),
            'original_max': float(stats_df['Max_Value'].max()),
            'mean_value': float(stats_df['Mean_Value'].mean()),
            'original_bands': [int(x) for x in selected_bands],
            'std_threshold_used': float(std_threshold),
            'global_mean': float(global_mean),
            'global_std': float(global_std),
            'constant_bands': [int(i) for i in stats_df[stats_df['Is_Constant']].index.tolist()],
            'interpolated_bands': [int(i) for i, is_bad in enumerate(bad_bands_mask) if is_bad],
            'bad_wavelength_ranges_nm': [[float(min_wl), float(max_wl)] for min_wl, max_wl in [(1263, 1562), (1761, 1958)]],
            'masked_regions_filled_with': 'band_mean_value',
            'processing_steps': [
                'Extract SWIR bands (1000-2500nm)',
                'Identify and interpolate bad bands',
                'Replace nodata values with band mean values',
                'Clean outliers using statistical approach',
                'Rescale each band to 0-1 range'
            ]
        }
        
        # Save metadata as JSON
        with open(f"{numpy_folder}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        with open(f"{torch_folder}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Remove checkpoints to save disk space (optional)
        if os.path.exists(checkpoint_folder):
            print("\nRemoving checkpoints to save disk space...")
            shutil.rmtree(checkpoint_folder)
        
        print("\nProcessing complete for this folder!")
        print(f"Data saved to:")
        print(f"  - ENVI format: {output_img_file}")
        print(f"  - NumPy format: {numpy_folder}/aviris_swir.npy")
        print(f"  - PyTorch format: {torch_folder}/aviris_swir.pt")
        print(f"  - Individual band images: {images_folder}/")
        print(f"  - Statistical visualizations: {stats_folder}/")
        
        return True

    else:
        print("No wavelength information found in the image header.")
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process AVIRIS data with bad band interpolation')
    parser.add_argument('--base', default='AVIRIS', help='Base AVIRIS directory')
    parser.add_argument('--output', default='AVIRIS_SWIR_INTP', help='Output directory')
    parser.add_argument('--csv', default='partial_crys_data/partial_crys_C0.0.csv', help='CSV file with wavelength data')
    parser.add_argument('--std', type=float, default=3.0, help='Number of standard deviations for outlier detection')
    parser.add_argument('--chunk', type=int, default=500, help='Chunk size for interpolation')
    parser.add_argument('--force', action='store_true', help='Force reprocessing of already processed folders')
    parser.add_argument('--subfolder', help='Process only a specific subfolder')
    args = parser.parse_args()
    
    # Define paths
    base_folder = args.base
    output_base = args.output
    csv_file = args.csv
    
    # Statistical threshold for outlier detection (number of standard deviations)
    std_threshold = args.std
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_base):
        os.makedirs(output_base)
    
    # Get list of subfolders in AVIRIS directory
    if args.subfolder:
        subfolders = [args.subfolder]
        if not os.path.isdir(os.path.join(base_folder, args.subfolder)):
            print(f"Error: Subfolder {args.subfolder} not found in {base_folder}")
            return
    else:
        subfolders = [f for f in os.listdir(base_folder) 
                      if os.path.isdir(os.path.join(base_folder, f))]
    
    if not subfolders:
        print(f"No subfolders found in {base_folder}")
        return
    
    print(f"Found {len(subfolders)} subfolders to process: {', '.join(subfolders)}")
    
    # Process each subfolder
    results = {}
    for subfolder in subfolders:
        start_time = time.time()
        try:
            success = process_aviris_folder(base_folder, subfolder, csv_file, output_base, 
                                           std_threshold, args.chunk, args.force)
            end_time = time.time()
            duration = end_time - start_time
            results[subfolder] = f"{'Success' if success else 'Failed'} (Time: {duration:.1f}s)"
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[subfolder] = f"Error: {str(e)}"
    
    # Print summary
    print("\n" + "="*80)
    print("Processing Summary")
    print("="*80)
    for subfolder, result in results.items():
        print(f"{subfolder}: {result}")


if __name__ == "__main__":
    main()
