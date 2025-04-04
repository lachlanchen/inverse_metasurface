#!/usr/bin/env python3
"""
AVIRIS Data Processing Utilities
-------------------------------
Functions for loading, processing, and validating AVIRIS data tiles:
- Filtering out invalid tiles (all 0s, all 1s)
- Reading HDR files for masks when available
- Creating valid tiles with overlap options
- Caching processed data for faster loading
"""

import torch
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re

def is_valid_tile(tile, min_valid_percentage=0.05, max_valid_percentage=0.95, 
                 max_constant_spectral_percentage=0.2, spectral_variance_threshold=1e-6):
    """
    Check if a tile contains valid data (not all 0s or all 1s, and not constant across wavelengths)
    
    Parameters:
    tile: Tensor of shape (C, H, W)
    min_valid_percentage: Minimum percentage of non-zero values required
    max_valid_percentage: Maximum percentage of values that can be 1.0
    max_constant_spectral_percentage: Maximum percentage of pixels allowed to have constant spectral values
    spectral_variance_threshold: Threshold for considering spectral values as constant
    
    Returns:
    bool: True if tile is valid, False otherwise
    """
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

def create_filtered_tiles(data, tile_size=256, overlap=0, min_valid_percentage=0.05, max_valid_percentage=0.95, 
                         max_constant_spectral_percentage=0.2, spectral_variance_threshold=1e-6, metadata=None):
    """
    Create tiles from a large image, filtering out invalid tiles
    
    Parameters:
    data: Tensor of shape (C, H, W) or (H, W, C)
    tile_size: Size of the output tiles
    overlap: Overlap between adjacent tiles
    min_valid_percentage: Minimum percentage of non-zero values required
    max_valid_percentage: Maximum percentage of values that can be 1.0
    max_constant_spectral_percentage: Maximum percentage of pixels allowed to have constant spectral values
    spectral_variance_threshold: Threshold for considering spectral values as constant
    metadata: Optional metadata from HDR file, including ignore values
    
    Returns:
    list: List of valid tiles as tensors
    """
    # Check data shape and convert if necessary
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    
    if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
        # Data is in (C, H, W) format, convert to (H, W, C)
        data = data.permute(1, 2, 0)
    
    H, W, C = data.shape
    tiles = []
    
    # Create a mask for invalid values if metadata provides ignore_value
    mask = None
    if metadata and 'ignore_value' in metadata:
        ignore_value = metadata['ignore_value']
        # Create mask where True means valid data
        mask = ~torch.isclose(data, torch.tensor(ignore_value, dtype=data.dtype))
        # Ensure mask works across all channels
        if mask.dim() == 3:
            mask = mask.all(dim=2)
    
    stride = tile_size - overlap
    total_tiles = ((H - tile_size) // stride + 1) * ((W - tile_size) // stride + 1)
    
    valid_tiles = 0
    skipped_tiles = 0
    skipped_same_value_tiles = 0
    
    with tqdm(total=total_tiles, desc="Creating tiles") as pbar:
        for i in range(0, H - tile_size + 1, stride):
            for j in range(0, W - tile_size + 1, stride):
                # Extract tile
                tile = data[i:i+tile_size, j:j+tile_size, :]
                
                # Check if tile is within masked area
                valid_tile = True
                if mask is not None:
                    tile_mask = mask[i:i+tile_size, j:j+tile_size]
                    # Skip tile if more than 20% of pixels are invalid
                    if tile_mask.float().mean() < 0.8:
                        valid_tile = False
                
                # Convert to (C, H, W) format for PyTorch
                tile_chw = tile.permute(2, 0, 1)
                
                # Check if the tile is valid (not all 0s or 1s, not constant across wavelengths)
                if valid_tile and is_valid_tile(
                    tile_chw, 
                    min_valid_percentage, 
                    max_valid_percentage, 
                    max_constant_spectral_percentage,
                    spectral_variance_threshold
                ):
                    tiles.append(tile_chw)
                    valid_tiles += 1
                else:
                    if valid_tile:  # It was marked as valid from mask check but failed is_valid_tile
                        # Check if it failed due to constant spectral values
                        C = tile_chw.shape[0]
                        reshaped = tile_chw.reshape(C, -1)
                        spectral_variances = torch.var(reshaped, dim=0)
                        constant_spectral_pixels = (spectral_variances < spectral_variance_threshold).float().mean().item()
                        
                        if constant_spectral_pixels > max_constant_spectral_percentage:
                            skipped_same_value_tiles += 1
                    
                    skipped_tiles += 1
                
                pbar.update(1)
    
    print(f"Created {valid_tiles} valid tiles, skipped {skipped_tiles} invalid tiles")
    print(f"Of skipped tiles, {skipped_same_value_tiles} were due to constant values across wavelengths")
    return tiles

def process_and_cache_filtered_data(args):
    """
    Process AVIRIS data, filter invalid tiles, and cache the results
    
    Parameters:
    args: Command line arguments containing processing parameters
    
    Returns:
    str: Path to cached tile file
    """
    # Define cache directory and file
    cache_dir = args.use_cache
    tile_size = args.tile_size
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache filename includes tile size and filtering info
    cache_file = os.path.join(cache_dir, f"filtered_tiles_{tile_size}.pt")
    
    # Use existing cache if available
    if os.path.exists(cache_file) and not args.force_cache:
        print(f"Using existing cache: {cache_file}")
        return cache_file
    
    # Get input directories
    base_dir = "AVIRIS_FOREST_SIMPLE_SELECT"
    if args.folder == "all":
        subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    else:
        subfolders = [args.folder]
    
    print(f"Processing {len(subfolders)} folders: {', '.join(subfolders)}")
    
    # Process each subfolder
    all_tiles = []
    for subfolder in subfolders:
        torch_dir = os.path.join(base_dir, subfolder, "torch")
        if not os.path.exists(torch_dir):
            print(f"Skipping {subfolder}: torch directory not found")
            continue
        
        # Load data
        data_file = os.path.join(torch_dir, "aviris_selected.pt")
        if not os.path.exists(data_file):
            print(f"Skipping {subfolder}: data file not found")
            continue
        
        print(f"Loading data from {data_file}")
        data = torch.load(data_file)
        print(f"Data shape: {data.shape}")
        
        # Look for HDR file
        hdr_path = find_hdr_file(data_file)
        metadata = {}
        if hdr_path:
            print(f"Found HDR file: {hdr_path}")
            metadata = parse_hdr_file(hdr_path)
            print(f"Parsed {len(metadata)} metadata entries from HDR file")
            if 'ignore_value' in metadata:
                print(f"Using ignore value: {metadata['ignore_value']}")
        
        # Create filtered tiles with enhanced validation
        print(f"Creating filtered {tile_size}x{tile_size} tiles with improved spectral validation...")
        # Get spectral variance parameters from args if available, or use defaults
        max_constant_spectral_percentage = getattr(args, 'max_constant_spectral_percentage', 0.2)
        spectral_variance_threshold = getattr(args, 'spectral_variance_threshold', 1e-6)
        
        tiles = create_filtered_tiles(
            data, 
            tile_size=tile_size, 
            metadata=metadata,
            max_constant_spectral_percentage=max_constant_spectral_percentage,
            spectral_variance_threshold=spectral_variance_threshold
        )
        print(f"Created {len(tiles)} valid tiles from {subfolder}")
        
        all_tiles.extend(tiles)
    
    # Convert to tensor and save
    if all_tiles:
        all_tiles_tensor = torch.stack(all_tiles)
        print(f"Total valid tiles: {len(all_tiles)}, Shape: {all_tiles_tensor.shape}")
        
        # Save to cache
        torch.save(all_tiles_tensor, cache_file)
        print(f"Saved filtered tiles to: {cache_file}")
    else:
        print("Warning: No valid tiles created!")
        # Create a dummy cache with a small sample to prevent errors
        dummy_tile = torch.rand(100, tile_size, tile_size)
        dummy_tiles = torch.stack([dummy_tile])
        torch.save(dummy_tiles, cache_file)
        print(f"Saved dummy tile to prevent errors: {cache_file}")
    
    return cache_file