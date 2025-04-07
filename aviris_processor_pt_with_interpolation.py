
# aviris_processor_pt_with_interpolation.py

#!/usr/bin/env python3
"""
AVIRIS Data Processing Pipeline with Bad Band Interpolation
----------------------------------------------------------
Steps:
1. Remove bad bands and perform interpolation
2. Identify invalid regions (mask)
3. Clamp values to 0-1 range
4. Select wavelengths based on CSV file
5. Rescale data cube to 0-1 range for each subfolder
6. Create tiles avoiding invalid regions
7. Visualize sample bands and pixel spectra
"""

import torch
import numpy as np
import os
import pandas as pd
import spectral
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import interpolate
import random
import shutil
from spectral.io import envi

# Utility functions for HDR parsing and file finding
def parse_hdr_file(hdr_path):
    """
    Parse AVIRIS HDR file to get metadata and possibly a mask

    Parameters:
    hdr_path: Path to HDR file

    Returns:
    dict: Metadata from HDR file, including mask if available
    """
    metadata = {}

    if not os.path.exists(hdr_path):
        return metadata

    try:
        with open(hdr_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                metadata[key.strip()] = value.strip()
            elif ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()

        # Check for data_ignore_value or no_data_value
        ignore_keys = ['data ignore value', 'ignore_value', 'no_data_value', 'nodata']
        for key in ignore_keys:
            if key.lower() in [k.lower() for k in metadata.keys()]:
                actual_key = next(k for k in metadata.keys() if k.lower() == key.lower())
                try:
                    # Try to convert to float, might be in quotes or have spaces
                    value = metadata[actual_key].strip('"\'')
                    metadata['ignore_value'] = float(value)
                    break
                except ValueError:
                    pass

    except Exception as e:
        print(f"Warning: Error parsing HDR file {hdr_path}: {e}")

    return metadata

def find_hdr_file(data_file):
    """
    Find the corresponding HDR file for a data file

    Parameters:
    data_file: Path to data file

    Returns:
    str: Path to HDR file if found, None otherwise
    """
    # Common HDR file patterns
    possible_extensions = ['.hdr', '.HDR']

    data_path = Path(data_file)

    # Try direct replacement of extension
    for ext in possible_extensions:
        hdr_path = data_path.with_suffix(ext)
        if hdr_path.exists():
            return str(hdr_path)

    # Look for any HDR file in the same directory with similar name
    data_name = data_path.stem
    for file in data_path.parent.glob('*'):
        if file.suffix.lower() in ['.hdr', '.HDR'] and data_name.lower() in file.stem.lower():
            return str(file)

    # Look for any HDR file in parent directory
    for file in data_path.parent.glob('*.hdr'):
        return str(file)

    return None

def remove_and_interpolate_bad_bands_chunked(wavelengths, band_data, bad_ranges=None, chunk_size=500, nodata_value=-9999.0):
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
    nodata_value : float
        Value to use for invalid/no-data pixels

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
                nodata_mask = spectrum <= nodata_value
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
                    interp_func = interpolate.interp1d(
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

# Functions for tile validation
def is_valid_tile(tile, mask=None, min_valid_percentage=0.05, max_valid_percentage=0.95,
                 max_constant_spectral_percentage=0.2, spectral_variance_threshold=1e-6):
    """
    Check if a tile contains valid data (not all 0s or all 1s, and not constant across wavelengths)

    Parameters:
    tile: Tensor of shape (C, H, W)
    mask: Optional mask tensor of shape (H, W) where True indicates valid pixels
    min_valid_percentage: Minimum percentage of non-zero values required
    max_valid_percentage: Maximum percentage of values that can be 1.0
    spectral_variance_threshold: Threshold for considering spectral values as constant

    Returns:
    bool: True if tile is valid, False otherwise
    """
    # If mask is provided, check if enough valid pixels
    if mask is not None:
        # If less than 80% of pixels are valid, reject the tile
        if mask.float().mean().item() < 0.8:
            return False

    # Check if tile is all zeros or very close to it
    zero_percentage = (tile == 0).float().mean().item()
    if zero_percentage > (1 - min_valid_percentage):
        return False

    # Check if tile is all ones or very close to it
    one_percentage = (tile == 1).float().mean().item()
    if one_percentage > max_valid_percentage:
        return False

    # Check if there's enough variance in the data
    if torch.var(tile) < 1e-4:
        return False

    # Check for constant values across spectral dimension (C)
    C, H, W = tile.shape
    # Reshape to (C, H*W) for easier spectral variance calculation
    reshaped = tile.reshape(C, -1)
    # Calculate variance along spectral dimension for each pixel
    spectral_variances = torch.var(reshaped, dim=0)
    # Count pixels with essentially zero spectral variance
    constant_spectral_pixels = (spectral_variances < spectral_variance_threshold).float().mean().item()

    # If too many pixels have constant spectral values, reject the tile
    if constant_spectral_pixels > max_constant_spectral_percentage:
        return False

    return True

def create_filtered_tiles_with_mask(data, mask, tile_size=128, overlap=0, min_valid_percentage=0.05,
                                  max_valid_percentage=0.95, max_constant_spectral_percentage=0.2,
                                  spectral_variance_threshold=1e-6):
    """
    Create tiles from data, using mask to filter out invalid regions

    Parameters:
    data: Tensor of shape (C, H, W)
    mask: Tensor of shape (H, W) where True indicates valid pixels
    tile_size: Size of output tiles
    overlap: Overlap between adjacent tiles
    min_valid_percentage: Minimum percentage of non-zero values required
    max_valid_percentage: Maximum percentage of values that can be 1.0
    max_constant_spectral_percentage: Maximum percentage of pixels allowed to have constant spectral values
    spectral_variance_threshold: Threshold for considering spectral values as constant

    Returns:
    list: List of valid tiles
    """
    C, H, W = data.shape
    tiles = []

    stride = tile_size - overlap
    total_tiles = ((H - tile_size) // stride + 1) * ((W - tile_size) // stride + 1)

    valid_tiles = 0
    skipped_tiles = 0
    skipped_mask_tiles = 0
    skipped_validation_tiles = 0

    with tqdm(total=total_tiles, desc="Creating tiles") as pbar:
        for i in range(0, H - tile_size + 1, stride):
            for j in range(0, W - tile_size + 1, stride):
                # Extract tile
                tile = data[:, i:i+tile_size, j:j+tile_size]

                # Extract mask for this tile
                tile_mask = mask[i:i+tile_size, j:j+tile_size]

                # Check mask coverage first
                mask_valid = tile_mask.float().mean().item() >= 0.8

                if not mask_valid:
                    skipped_mask_tiles += 1
                    skipped_tiles += 1
                    pbar.update(1)
                    continue

                # Now check other validation criteria
                if is_valid_tile(
                    tile,
                    min_valid_percentage=min_valid_percentage,
                    max_valid_percentage=max_valid_percentage,
                    max_constant_spectral_percentage=max_constant_spectral_percentage,
                    spectral_variance_threshold=spectral_variance_threshold
                ):
                    tiles.append(tile)
                    valid_tiles += 1
                else:
                    skipped_validation_tiles += 1
                    skipped_tiles += 1

                pbar.update(1)

    print(f"Created {valid_tiles} valid tiles, skipped {skipped_tiles} invalid tiles")
    print(f"  - {skipped_mask_tiles} tiles skipped due to invalid mask")
    print(f"  - {skipped_validation_tiles} tiles skipped due to other validation criteria")
    return tiles

def visualize_sample_bands(tiles, wavelengths, output_folder, num_samples=3, seed=23):
    """
    Visualize sample bands from the tiles

    Parameters:
    tiles: Tensor of shape (N, C, H, W)
    wavelengths: List of wavelengths in nm
    output_folder: Folder to save visualizations
    num_samples: Number of sample tiles to visualize
    seed: Random seed for reproducibility
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Set random seed for reproducibility
    random.seed(seed)

    # Convert wavelengths to um for plotting
    wavelengths_um = [w/1000 for w in wavelengths]

    # Number of tiles and bands
    num_tiles = tiles.shape[0]
    num_bands = tiles.shape[1]

    # Select sample tiles - first, middle, last
    if num_tiles >= 3:
        sample_indices = [0, num_tiles//2, num_tiles-1]
    else:
        sample_indices = list(range(min(num_tiles, num_samples)))

    # Select sample bands - first, middle, last
    sample_bands = [0, num_bands//2, num_bands-1]

    # Create visualizations for each sample tile
    for tile_idx in sample_indices:
        tile = tiles[tile_idx]

        # Create figure for sample bands
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, band_idx in enumerate(sample_bands):
            # Get band data
            band_data = tile[band_idx].cpu().numpy()
            wavelength = wavelengths_um[band_idx]

            # Plot band
            im = axes[i].imshow(band_data, cmap='viridis', vmin=0, vmax=1)
            axes[i].set_title(f'Band {band_idx+1} ({wavelength:.3f} μm)')
            plt.colorbar(im, ax=axes[i])

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'sample_tile_{tile_idx+1}_bands.png'), dpi=300)
        plt.close()

        # Create figure for sample pixel spectra
        fig, ax = plt.subplots(figsize=(10, 6))

        # Select sample pixels - use corners and center
        tile_h, tile_w = tile.shape[1], tile.shape[2]
        sample_pixels = [
            (0, 0),                    # top-left
            (0, tile_w-1),             # top-right
            (tile_h//2, tile_w//2),    # center
            (tile_h-1, 0),             # bottom-left
            (tile_h-1, tile_w-1)       # bottom-right
        ]

        for r, c in sample_pixels:
            # Get pixel spectrum
            spectrum = tile[:, r, c].cpu().numpy()

            # Plot spectrum
            ax.plot(wavelengths_um, spectrum, marker='o', markersize=3, label=f'Pixel ({r},{c})')

        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Reflectance')
        ax.set_title(f'Sample Pixel Spectra from Tile {tile_idx+1}')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'sample_tile_{tile_idx+1}_spectra.png'), dpi=300)
        plt.close()

def process_aviris_data(aviris_folder, subfolder, csv_file, output_folder, tile_size=128, overlap=0,
                        nodata_value=-9999.0, do_interpolation=False, bad_ranges=None, chunk_size=500):
    """
    Process AVIRIS data:
    1. Remove bad bands and interpolate (if enabled)
    2. Identify invalid regions
    3. Clamp to 0-1 range
    4. Select wavelengths based on CSV file
    5. Rescale data to 0-1 range
    6. Create tiles from valid regions
    7. Visualize sample bands and pixel spectra

    Parameters:
    aviris_folder: Base AVIRIS directory
    subfolder: Name of subfolder to process
    csv_file: Path to CSV file with wavelength data
    output_folder: Folder to save output files
    tile_size: Size of tiles to create
    overlap: Overlap between adjacent tiles
    nodata_value: Value to use for invalid/no-data pixels
    do_interpolation: Whether to interpolate bad bands
    bad_ranges: List of tuples defining bad band ranges in nm
    chunk_size: Chunk size for interpolation

    Returns:
    list: List of valid tiles as tensors, or None if processing failed
    dict: Metadata including wavelength information
    """
    print(f"\n{'='*60}")
    print(f"Processing {subfolder}")
    print(f"{'='*60}")

    # Define paths
    folder_path = os.path.join(aviris_folder, subfolder)

    # Find the image file (assuming it's the one without .hdr extension)
    img_files = [f for f in os.listdir(folder_path)
                if not f.endswith('.hdr') and not f.endswith('.py') and not f.endswith('.png')]

    if not img_files:
        print(f"No image files found in {folder_path}")
        return None, None

    img_file = img_files[0]
    img_path = os.path.join(folder_path, img_file)
    hdr_file = os.path.join(folder_path, f'{img_file}.hdr')

    # Check if header file exists
    if not os.path.exists(hdr_file):
        print(f"Header file not found: {hdr_file}")
        # Try to find alternative HDR file
        hdr_file = find_hdr_file(img_path)
        if not hdr_file:
            print("No HDR file found")
            return None, None
        print(f"Found alternative HDR file: {hdr_file}")

    # Open AVIRIS image
    print(f"Opening AVIRIS image from: {hdr_file}")
    try:
        img = spectral.open_image(hdr_file)
        print(f"Image dimensions: {img.shape}")
        print(f"Number of bands: {img.nbands}")
    except Exception as e:
        print(f"Error opening image: {e}")
        return None, None

    # Check for wavelength information
    if not (hasattr(img, 'bands') and hasattr(img.bands, 'centers')):
        print("No wavelength information found in the image header.")
        return None, None

    # Get wavelength information
    aviris_wavelengths = np.array(img.bands.centers)
    print(f"AVIRIS wavelength range: {np.min(aviris_wavelengths):.2f} to {np.max(aviris_wavelengths):.2f} nm")

    # Read wavelengths from CSV
    print(f"Reading wavelengths from CSV: {csv_file}")
    try:
        csv_data = pd.read_csv(csv_file)
        if 'Wavelength_um' not in csv_data.columns:
            print(f"Error: CSV file doesn't contain 'Wavelength_um' column")
            print(f"Available columns: {', '.join(csv_data.columns)}")
            return None, None

        csv_wavelengths = csv_data['Wavelength_um'].to_numpy() * 1000  # Convert μm to nm
        print(f"Found {len(csv_wavelengths)} wavelengths in CSV file")
        print(f"CSV wavelength range: {np.min(csv_wavelengths):.2f} to {np.max(csv_wavelengths):.2f} nm")

        # Filter to only include wavelengths in 1.0-2.5 μm range (1000-2500 nm)
        swir_min = 1000.0  # 1.0 μm in nm
        swir_max = 2500.0  # 2.5 μm in nm
        csv_swir_mask = (csv_wavelengths >= swir_min) & (csv_wavelengths <= swir_max)
        csv_swir_wavelengths = csv_wavelengths[csv_swir_mask]
        print(f"Filtered CSV wavelengths to SWIR range ({swir_min}-{swir_max} nm)")
        print(f"Found {len(csv_swir_wavelengths)} wavelengths in SWIR range")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, None

    # Parse metadata from HDR file
    metadata = parse_hdr_file(hdr_file)
    print(f"Parsed {len(metadata)} metadata entries from HDR file")

    # Determine no-data value from metadata or use default
    if 'ignore_value' in metadata:
        nodata_value = float(metadata['ignore_value'])
        print(f"Using no-data value from metadata: {nodata_value}")
    else:
        print(f"Using default no-data value: {nodata_value}")

    # Read all data from the image
    print(f"Reading all bands from AVIRIS image...")
    rows, cols, nbands = img.shape
    data = np.zeros((rows, cols, nbands), dtype=np.float32)
    for i in tqdm(range(nbands), desc="Reading bands"):
        data[:,:,i] = img.read_band(i)

    # Step 1: Remove bad bands and interpolate (if enabled)
    if do_interpolation:
        print("\nStep 1: Performing bad band interpolation...")
        if bad_ranges is None:
            # Default bad wavelength ranges in nm
            bad_ranges = [(1263, 1562), (1761, 1958)]

        # Interpolate bad bands
        data, bad_bands_mask = remove_and_interpolate_bad_bands_chunked(
            aviris_wavelengths, data, bad_ranges=bad_ranges,
            chunk_size=chunk_size, nodata_value=nodata_value
        )

        # Create visualization of interpolation results
        interp_vis_dir = os.path.join(output_folder, 'interpolation_vis')
        os.makedirs(interp_vis_dir, exist_ok=True)

        # Visualize band statistics with interpolated bands highlighted
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        # Calculate band means for valid pixels
        band_means = []
        for i in range(nbands):
            band_data = data[:,:,i]
            valid_mask = band_data > nodata_value
            if np.any(valid_mask):
                band_means.append(np.mean(band_data[valid_mask]))
            else:
                band_means.append(0)

        # Plot band means
        plt.plot(range(nbands), band_means, 'b-', alpha=0.7)

        # Highlight interpolated bands
        for i in range(nbands):
            if i < len(bad_bands_mask) and bad_bands_mask[i]:
                plt.axvspan(i-0.5, i+0.5, color='r', alpha=0.2)

        plt.xlabel('Band Index')
        plt.ylabel('Mean Value')
        plt.title('Mean Values by Band Index with Interpolated Bands Highlighted')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        # Plot wavelengths vs mean values
        plt.plot(aviris_wavelengths, band_means, 'g-', alpha=0.7)

        # Highlight bad wavelength ranges
        for min_wl, max_wl in bad_ranges:
            plt.axvspan(min_wl, max_wl, color='r', alpha=0.2)

        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Mean Value')
        plt.title('Mean Values by Wavelength with Bad Bands Highlighted')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{interp_vis_dir}/interpolation_visualization_{subfolder}.png', dpi=300)
        plt.close()

    # Step 2: Identify invalid regions
    print("\nStep 2: Identifying invalid regions...")
    valid_mask = np.ones((rows, cols), dtype=bool)
    for i in range(nbands):
        valid_mask = np.logical_and(valid_mask, data[:,:,i] > nodata_value)

    # Report how many valid pixels we have
    valid_percentage = np.mean(valid_mask) * 100
    print(f"Valid pixels: {valid_percentage:.2f}% of total")

    # Save visualization of valid mask
    mask_vis_dir = os.path.join(output_folder, 'mask_visualization')
    os.makedirs(mask_vis_dir, exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.imshow(valid_mask, cmap='gray')
    plt.colorbar(label='Valid (1) / Invalid (0)')
    plt.title(f'Valid Pixel Mask - {subfolder}\n{valid_percentage:.2f}% Valid')
    plt.tight_layout()
    plt.savefig(f'{mask_vis_dir}/valid_mask_{subfolder}.png', dpi=300)
    plt.close()

    # Step 3: Clamp all values to 0-1 range
    print("\nStep 3: Hard clamping all values to 0-1 range...")
    data = np.clip(data, 0, 1)
    print("Data hard-clamped to [0,1] range")

    # Step 4: Select wavelengths based on CSV file
    print("\nStep 4: Selecting wavelengths based on CSV file...")

    # For each wavelength in the filtered CSV, find the closest matching band in AVIRIS
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
    print(f"Selected {len(selected_bands)} unique AVIRIS bands in SWIR range")
    print("First 5 wavelength mappings:")
    print(mapping_df.head(5).to_string(index=False))

    # Save the wavelength mapping
    os.makedirs(output_folder, exist_ok=True)
    mapping_df.to_csv(f'{output_folder}/wavelength_mapping_{subfolder}.csv', index=False)

    # Extract the selected bands
    print("Extracting selected bands...")
    selected_data = np.zeros((rows, cols, len(selected_bands)), dtype=np.float32)
    for i, band_idx in enumerate(tqdm(selected_bands, desc="Extracting bands")):
        selected_data[:,:,i] = data[:,:,band_idx]

    # Get selected wavelengths from the mapping
    selected_wavelengths = np.array([aviris_wavelengths[idx] for idx in selected_bands])

    # Step 5: Rescale the selected 100 band cube for each subfolder
    print("\nStep 5: Rescaling the selected data cube to 0-1 range...")

    # Find global min and max across all bands in selected data (considering only valid pixels)
    # global_min = np.inf
    # global_max = -np.inf
    global_min = 1
    global_max = 0

    for i in range(len(selected_bands)):
        band_data = selected_data[:,:,i]
        valid_band_data = band_data[valid_mask]

        if len(valid_band_data) > 0:
            band_min = np.min(valid_band_data)
            band_max = np.max(valid_band_data)

            global_min = min(global_min, band_min)
            global_max = max(global_max, band_max)

    print(f"Global data range: Min={global_min:.6f}, Max={global_max:.6f}")

    # Rescale selected data using global min/max
    if global_max > global_min:
        # Apply global normalization to the entire selected data cube
        selected_data = (selected_data - global_min) / (global_max - global_min)
        # Clip to ensure 0-1 range after normalization
        selected_data = np.clip(selected_data, 0, 1)
    else:
        print("Warning: Global min equals global max, setting all values to 0.5")
        selected_data.fill(0.5)

    # Convert to PyTorch tensor with channels first (C, H, W)
    processed_tensor = torch.from_numpy(selected_data).permute(2, 0, 1)

    # Convert mask to PyTorch tensor
    mask_tensor = torch.from_numpy(valid_mask)

    # Step 6: Create tiles from valid regions
    print(f"\nStep 6: Creating {tile_size}x{tile_size} tiles from valid regions...")
    tiles = create_filtered_tiles_with_mask(
        processed_tensor,
        mask_tensor,
        tile_size=tile_size,
        overlap=overlap
    )

    # Step 7: Save sample visualizations of tiles
    print("\nStep 7: Creating visualizations of sample bands and spectra...")
    if tiles:
        tiles_tensor = torch.stack(tiles)
        vis_dir = os.path.join(output_folder, 'tile_visualization', subfolder)
        visualize_sample_bands(tiles_tensor, selected_wavelengths.tolist(), vis_dir)

    # Create metadata
    metadata_dict = {
        'subfolder': subfolder,
        'original_shape': (rows, cols, len(selected_bands)),
        'num_tiles': len(tiles),
        'tile_size': tile_size,
        'overlap': overlap,
        'wavelengths_nm': [float(x) for x in selected_wavelengths.tolist()],
        'wavelengths_um': [float(x/1000) for x in selected_wavelengths.tolist()],
        'original_bands': [int(x) for x in selected_bands],
        'nodata_value': float(nodata_value),
        'valid_pixel_percentage': float(valid_percentage),
        'interpolation_performed': do_interpolation,
        'processing_steps': [
            '1. Remove bad bands and interpolate' if do_interpolation else '1. No bad band interpolation performed',
            '2. Identify invalid regions',
            '3. Clamp all values to 0-1 range',
            '4. Select wavelengths based on CSV file',
            '5. Rescale data to 0-1 range',
            '6. Create tiles from valid regions',
            '7. Visualize sample bands and spectra'
        ]
    }

    if do_interpolation:
        metadata_dict.update({
            'bad_wavelength_ranges_nm': [[float(min_wl), float(max_wl)] for min_wl, max_wl in bad_ranges],
            'interpolated_bands': [int(i) for i, is_bad in enumerate(bad_bands_mask) if is_bad],
        })

    return tiles, metadata_dict

def process_and_cache_data(args):
    """
    Process and cache data from multiple AVIRIS folders

    Parameters:
    args: Command-line arguments

    Returns:
    str: Path to cached file
    """
    # Create cache directory if it doesn't exist
    cache_dir = args.output_dir
    os.makedirs(cache_dir, exist_ok=True)

    # Add timestamp to cache filenames for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get CSV file basename for cache filename
    csv_path = Path(args.csv_file)
    wavelength_info = csv_path.stem

    # Define output folder structure
    if args.interpolation:
        cache_folder = os.path.join(
            cache_dir,
            f"aviris_processed_intp_{wavelength_info}_{args.tile_size}_{timestamp}"
        )
    else:
        cache_folder = os.path.join(
            cache_dir,
            f"aviris_processed_{wavelength_info}_{args.tile_size}_{timestamp}"
        )

    # Create output folder
    os.makedirs(cache_folder, exist_ok=True)

    # Define cache filename
    if args.interpolation:
        cache_file = os.path.join(cache_folder, f"aviris_tiles_intp.pt")
        metadata_file = os.path.join(cache_folder, f"aviris_metadata_intp.json")
    else:
        cache_file = os.path.join(cache_folder, f"aviris_tiles.pt")
        metadata_file = os.path.join(cache_folder, f"aviris_metadata.json")

    # Determine which folders to process
    base_dir = Path(args.input_dir)
    if not base_dir.exists():
        print(f"Error: Base directory '{args.input_dir}' does not exist")
        return None

    if args.subfolder == "all":
        subfolders = [f.name for f in base_dir.iterdir() if f.is_dir()]
    else:
        # Handle comma-separated list of subfolders
        subfolder_list = [s.strip() for s in args.subfolder.split(',')]
        subfolders = []
        for subfolder in subfolder_list:
            subfolder_path = base_dir / subfolder
            if not subfolder_path.exists() or not subfolder_path.is_dir():
                print(f"Warning: Subfolder '{subfolder}' not found in {args.input_dir}")
            else:
                subfolders.append(subfolder)

    if not subfolders:
        print(f"No subfolders found to process in {args.input_dir}")
        return None

    print(f"Processing {len(subfolders)} folders: {', '.join(subfolders)}")

    # Verify CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' does not exist")
        return None

    # Define bad bands ranges if interpolation is enabled
    bad_ranges = None
    if args.interpolation:
        bad_ranges = [(1263, 1562), (1761, 1958)]  # Default bad wavelength ranges in nm
        print(f"Interpolation enabled with bad wavelength ranges: {bad_ranges}")

    # Process each subfolder
    all_tiles = []
    all_metadata = []

    for subfolder in subfolders:
        print(f"\nProcessing {subfolder}...")
        # Create a temporary output folder
        temp_output = os.path.join(cache_folder, "temp", subfolder)
        os.makedirs(temp_output, exist_ok=True)

        # Process the data
        tiles, metadata = process_aviris_data(
            args.input_dir,
            subfolder,
            args.csv_file,
            temp_output,
            tile_size=args.tile_size,
            overlap=args.overlap,
            nodata_value=args.nodata_value,
            do_interpolation=args.interpolation,
            bad_ranges=bad_ranges,
            chunk_size=args.chunk_size
        )

        # Add to our collection if processing was successful
        if tiles and metadata:
            all_tiles.extend(tiles)
            all_metadata.append(metadata)

    # Save all tiles to a single PT file
    if all_tiles:
        all_tiles_tensor = torch.stack(all_tiles)
        torch.save(all_tiles_tensor, cache_file)
        print(f"\nSaved {len(all_tiles)} tiles to: {cache_file}")
        print(f"Tile tensor shape: {all_tiles_tensor.shape}")

        # Save combined metadata
        combined_metadata = {
            'total_tiles': len(all_tiles),
            'tile_size': args.tile_size,
            'overlap': args.overlap,
            'nodata_value': args.nodata_value,
            'interpolation_performed': args.interpolation,
            'sources': [meta['subfolder'] for meta in all_metadata],
            'wavelengths_nm': all_metadata[0]['wavelengths_nm'] if all_metadata else [],
            'wavelengths_um': all_metadata[0]['wavelengths_um'] if all_metadata else [],
            'csv_file': str(args.csv_file),
            'base_dir': str(args.input_dir),
            'processing_date': timestamp,
            'command_line_args': vars(args)
        }

        if args.interpolation and all_metadata:
            combined_metadata.update({
                'bad_wavelength_ranges_nm': all_metadata[0].get('bad_wavelength_ranges_nm', []),
            })

        with open(metadata_file, 'w') as f:
            json.dump(combined_metadata, f, indent=2)
        print(f"Saved combined metadata to: {metadata_file}")

        # Generate global visualization of tiles and spectra
        print("\nGenerating global visualizations...")
        global_vis_dir = os.path.join(cache_folder, 'global_visualization')
        visualize_sample_bands(
            all_tiles_tensor,
            all_metadata[0]['wavelengths_nm'] if all_metadata else [],
            global_vis_dir
        )

        return cache_file
    else:
        print("Warning: No valid tiles created!")
        return None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='AVIRIS Data Processing Pipeline with Bad Band Interpolation')
    parser.add_argument('--input_dir', required=True, help='Input AVIRIS directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--csv_file', required=True, help='CSV file with wavelength data (must have Wavelength_um column)')
    parser.add_argument('--tile_size', type=int, default=128, help='Size of output tiles')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap between adjacent tiles')
    parser.add_argument('--subfolder', default='all', help='Specific subfolder(s) to process, "all", or comma-separated list')
    parser.add_argument('--nodata_value', type=float, default=-9999.0, help='Value to use for invalid/no-data pixels')
    parser.add_argument('--interpolation', '--intp', action='store_true', help='Enable bad band interpolation')
    parser.add_argument('--chunk_size', type=int, default=500, help='Chunk size for interpolation processing')
    parser.add_argument('--clean_temp', action='store_true', help='Clean up temporary files after processing')
    args = parser.parse_args()

    # Process and cache the data
    cache_file = process_and_cache_data(args)
    if cache_file:
        print(f"\nProcessing complete. Final output: {cache_file}")
    else:
        print("\nProcessing failed. No output file created.")

    # Clean up temporary files if requested
    if args.clean_temp:
        output_dir = args.output_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path(args.csv_file)
        wavelength_info = csv_path.stem

        if args.interpolation:
            cache_folder = os.path.join(output_dir, f"aviris_processed_intp_{wavelength_info}_{args.tile_size}_{timestamp}")
        else:
            cache_folder = os.path.join(output_dir, f"aviris_processed_{wavelength_info}_{args.tile_size}_{timestamp}")

        temp_dir = os.path.join(cache_folder, "temp")
        if os.path.exists(temp_dir):
            print("Cleaning up temporary files...")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)

    main()
