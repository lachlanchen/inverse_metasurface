#!/usr/bin/env python3
"""
AVIRIS Data Processing Pipeline
-------------------------------
Combined script for:
1. Loading AVIRIS data and identifying invalid masks (<= -9999)
2. Clamping values to 0-1 range
3. Selecting specific wavelengths from CSV file
4. Creating tiles only from valid regions
5. Caching results to a single PT file
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
import re
from datetime import datetime

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
    max_constant_spectral_percentage: Maximum percentage of pixels allowed to have constant spectral values
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

def create_filtered_tiles_with_mask(data, mask, tile_size=256, overlap=0, min_valid_percentage=0.05, 
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

# Combined function for processing AVIRIS data
def process_aviris_data(aviris_folder, subfolder, csv_file, output_folder, tile_size=256, overlap=0, 
                        nodata_value=-9999.0):
    """
    Process AVIRIS data:
    1. Identify invalid mask (values <= nodata_value)
    2. Select wavelengths based on CSV file
    3. Clamp valid values to 0-1 range
    4. Create tiles from valid regions
    
    Parameters:
    aviris_folder: Base AVIRIS directory
    subfolder: Name of subfolder to process
    csv_file: Path to CSV file with wavelength data
    output_folder: Folder to save output files
    tile_size: Size of tiles to create
    overlap: Overlap between adjacent tiles
    nodata_value: Value to use for invalid/no-data pixels
    
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
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, None
    
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
    
    # Parse metadata from HDR file
    metadata = parse_hdr_file(hdr_file)
    print(f"Parsed {len(metadata)} metadata entries from HDR file")
    
    # Determine no-data value from metadata or use default
    if 'ignore_value' in metadata:
        nodata_value = float(metadata['ignore_value'])
        print(f"Using no-data value from metadata: {nodata_value}")
    else:
        print(f"Using default no-data value: {nodata_value}")
    
    # For each wavelength in CSV, find the closest matching band in AVIRIS
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
    print("First 5 wavelength mappings:")
    print(mapping_df.head(5).to_string(index=False))
    
    # Save the wavelength mapping
    os.makedirs(output_folder, exist_ok=True)
    mapping_df.to_csv(f'{output_folder}/wavelength_mapping_{subfolder}.csv', index=False)
    
    # Extract the selected bands
    print(f"\nExtracting selected bands...")
    
    # Prepare a new data array for selected bands
    rows, cols, _ = img.shape
    new_data = np.zeros((rows, cols, len(selected_bands)), dtype=np.float32)
    
    # Create a mask for invalid data (True for valid, False for invalid)
    # Initialize as all valid
    valid_mask = np.ones((rows, cols), dtype=bool)
    
    # Extract the selected bands
    for i, band_idx in enumerate(tqdm(selected_bands, desc="Extracting bands")):
        band_data = img.read_band(band_idx)
        
        # Update the valid mask - mark any pixel with value <= nodata_value as invalid
        valid_mask = np.logical_and(valid_mask, band_data > nodata_value)
        
        # Store the band data
        new_data[:,:,i] = band_data
    
    # Report how many valid pixels we have
    valid_percentage = np.mean(valid_mask) * 100
    print(f"Valid pixels: {valid_percentage:.2f}% of total")
    
    # Create a copy to store processed data
    processed_data = new_data.copy()
    
    # Replace invalid values with mean values of valid regions by band
    print("\nReplacing invalid values with band means...")
    for i in range(len(selected_bands)):
        # Get invalid pixels for this band based on the combined mask
        band_invalid_mask = ~valid_mask
        
        # If there are any invalid values
        if np.any(band_invalid_mask):
            # Get valid values for this band
            valid_values = processed_data[:,:,i][valid_mask]
            
            if len(valid_values) > 0:
                # Use mean of valid values
                band_mean = np.mean(valid_values)
            else:
                # If no valid values, use 0
                band_mean = 0.0
            
            # Replace invalid values with mean
            processed_data[:,:,i][band_invalid_mask] = band_mean
    
    # Clamp values to 0-1 range
    print("\nClamping data to 0-1 range...")
    for i in tqdm(range(len(selected_bands)), desc="Clamping bands"):
        band_data = processed_data[:,:,i]
        
        # Only consider valid pixels for min/max calculation
        valid_band_data = band_data[valid_mask]
        
        if len(valid_band_data) > 0:
            min_val = np.min(valid_band_data)
            max_val = np.max(valid_band_data)
            
            # Avoid division by zero
            if max_val > min_val:
                # Normalize to 0-1 range (only normalize valid pixels)
                processed_data[:,:,i] = np.where(
                    valid_mask,
                    (band_data - min_val) / (max_val - min_val),
                    band_data  # Leave invalid pixels as they are
                )
            else:
                # If min == max, set all valid values to 0.5
                processed_data[:,:,i] = np.where(valid_mask, 0.5, band_data)
        else:
            # If no valid data in this band, set all to 0
            processed_data[:,:,i] = 0.0
    
    # Convert to PyTorch tensor with channels first (C, H, W)
    processed_tensor = torch.from_numpy(processed_data).permute(2, 0, 1)
    
    # Convert mask to PyTorch tensor
    mask_tensor = torch.from_numpy(valid_mask)
    
    # Create tiles from valid regions
    print(f"\nCreating {tile_size}x{tile_size} tiles from valid regions...")
    tiles = create_filtered_tiles_with_mask(
        processed_tensor, 
        mask_tensor,
        tile_size=tile_size, 
        overlap=overlap
    )
    
    # Create metadata
    metadata_dict = {
        'subfolder': subfolder,
        'original_shape': (rows, cols, len(selected_bands)),
        'num_tiles': len(tiles),
        'tile_size': tile_size,
        'overlap': overlap,
        'wavelengths_nm': [float(x) for x in aviris_wavelengths[selected_bands].tolist()],
        'wavelengths_um': [float(x/1000) for x in aviris_wavelengths[selected_bands].tolist()],
        'original_bands': [int(x) for x in selected_bands],
        'nodata_value': float(nodata_value),
        'valid_pixel_percentage': float(valid_percentage),
        'processing_steps': [
            'Identify invalid values (values <= nodata_value)',
            'Select bands closest to CSV wavelengths',
            'Replace invalid values with band means',
            'Clamp each band to 0-1 range',
            'Create tiles from valid regions'
        ]
    }
    
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
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    # Add timestamp to cache filenames for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get CSV file basename for cache filename
    csv_path = Path(args.csv_file)
    wavelength_info = csv_path.stem
    
    # Define cache filename
    cache_file = os.path.join(
        cache_dir, 
        f"aviris_processed_{wavelength_info}_{args.tile_size}_{timestamp}.pt"
    )
    metadata_file = os.path.join(
        cache_dir, 
        f"aviris_metadata_{wavelength_info}_{args.tile_size}_{timestamp}.json"
    )
    
    # Check if we should use existing cache
    if not args.force_cache:
        # Look for existing files with similar pattern
        pattern = f"aviris_processed_{wavelength_info}_{args.tile_size}_*.pt"
        existing_files = list(Path(cache_dir).glob(pattern))
        
        if existing_files:
            # Sort by modification time (newest first)
            existing_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            newest_file = existing_files[0]
            print(f"Using existing cache: {newest_file}")
            return str(newest_file)
    
    # Determine which folders to process
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory '{args.base_dir}' does not exist")
        return None
        
    if args.subfolder == "all":
        subfolders = [f.name for f in base_dir.iterdir() if f.is_dir()]
    else:
        subfolder_path = base_dir / args.subfolder
        if not subfolder_path.exists() or not subfolder_path.is_dir():
            print(f"Error: Subfolder '{args.subfolder}' not found in {args.base_dir}")
            return None
        subfolders = [args.subfolder]
    
    if not subfolders:
        print(f"No subfolders found in {args.base_dir}")
        return None
        
    print(f"Processing {len(subfolders)} folders: {', '.join(subfolders)}")
    
    # Verify CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' does not exist")
        return None
    
    # Process each subfolder
    all_tiles = []
    all_metadata = []
    
    for subfolder in subfolders:
        print(f"Processing {subfolder}...")
        # Create a temporary output folder
        temp_output = os.path.join(cache_dir, "temp", subfolder)
        os.makedirs(temp_output, exist_ok=True)
        
        # Process the data
        tiles, metadata = process_aviris_data(
            args.base_dir, 
            subfolder, 
            args.csv_file, 
            temp_output,
            tile_size=args.tile_size,
            overlap=args.overlap,
            nodata_value=args.nodata_value
        )
        
        # Add to our collection if processing was successful
        if tiles and metadata:
            all_tiles.extend(tiles)
            all_metadata.append(metadata)
    
    # Save all tiles to a single PT file
    if all_tiles:
        all_tiles_tensor = torch.stack(all_tiles)
        torch.save(all_tiles_tensor, cache_file)
        print(f"Saved {len(all_tiles)} tiles to: {cache_file}")
        print(f"Tile tensor shape: {all_tiles_tensor.shape}")
        
        # Save combined metadata
        combined_metadata = {
            'total_tiles': len(all_tiles),
            'tile_size': args.tile_size,
            'overlap': args.overlap,
            'nodata_value': args.nodata_value,
            'sources': [meta['subfolder'] for meta in all_metadata],
            'wavelengths_nm': all_metadata[0]['wavelengths_nm'] if all_metadata else [],
            'wavelengths_um': all_metadata[0]['wavelengths_um'] if all_metadata else [],
            'csv_file': str(args.csv_file),
            'base_dir': str(args.base_dir),
            'processing_date': timestamp,
            'command_line_args': vars(args)
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(combined_metadata, f, indent=2)
        print(f"Saved combined metadata to: {metadata_file}")
        
        return cache_file
    else:
        print("Warning: No valid tiles created!")
        # Create an empty file to indicate processing was attempted
        empty_file = os.path.join(
            cache_dir, 
            f"aviris_processed_EMPTY_{wavelength_info}_{args.tile_size}_{timestamp}.pt"
        )
        torch.save(torch.zeros((0, args.tile_size, args.tile_size)), empty_file)
        print(f"Created empty file: {empty_file}")
        return empty_file

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='AVIRIS Data Processing Pipeline')
    parser.add_argument('--base_dir', default='AVIRIS', help='Base AVIRIS directory')
    parser.add_argument('--csv_file', default='partial_crys_data/partial_crys_C0.0.csv', 
                        help='CSV file with wavelength data (must have Wavelength_um column)')
    parser.add_argument('--cache_dir', default='cache', help='Directory to store cached results')
    parser.add_argument('--tile_size', type=int, default=256, help='Size of output tiles')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap between adjacent tiles')
    parser.add_argument('--subfolder', default='all', help='Specific subfolder to process, or "all"')
    parser.add_argument('--force_cache', action='store_true', help='Force regeneration of cache')
    parser.add_argument('--nodata_value', type=float, default=-9999.0, help='Value to use for invalid/no-data pixels')
    parser.add_argument('--clean_temp', action='store_true', help='Clean up temporary files after processing')
    args = parser.parse_args()
    
    # Process and cache the data
    cache_file = process_and_cache_data(args)
    if cache_file:
        print(f"Processing complete. Final output: {cache_file}")
    else:
        print("Processing failed. No output file created.")
    
    # Clean up temporary files if requested
    if args.clean_temp and os.path.exists(os.path.join(args.cache_dir, "temp")):
        import shutil
        shutil.rmtree(os.path.join(args.cache_dir, "temp"))
        print("Cleaned up temporary files")

if __name__ == "__main__":
    main()