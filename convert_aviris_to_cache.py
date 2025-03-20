import torch
import numpy as np
import spectral
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def process_aviris_to_tiles(aviris_path, tile_size=100, output_file=None, 
                            percentile_clip=(0.5, 99.5), outlier_threshold=3.0,
                            visualize=True, viz_dir="aviris_visualization",
                            verbose=True):
    """
    Process AVIRIS orthorectified image data into tiles with uniform normalization.
    
    Parameters:
    aviris_path: Path to AVIRIS header file
    tile_size: Size of tiles to extract (square)
    output_file: Path to save the processed data (.pt file)
    percentile_clip: Percentiles to determine global data range
    outlier_threshold: Z-score threshold to identify extreme outliers
    visualize: Whether to generate visualizations
    viz_dir: Directory to save visualizations
    verbose: Whether to print detailed progress information
    
    Returns:
    torch.Tensor of shape [num_tiles, tile_size, tile_size, num_bands]
    """
    if verbose:
        print(f"Loading AVIRIS data from: {aviris_path}")
    
    # Check if file exists
    if not os.path.exists(aviris_path):
        raise FileNotFoundError(f"AVIRIS file not found: {aviris_path}")
    
    # Open the AVIRIS image
    img = spectral.open_image(aviris_path)
    
    if verbose:
        print(f"Image dimensions: {img.shape}")
        print(f"Number of bands: {img.nbands}")
        
        if hasattr(img, 'bands') and hasattr(img.bands, 'centers'):
            wavelengths = np.array(img.bands.centers)
            print(f"Wavelength range: {min(wavelengths):.2f} to {max(wavelengths):.2f} nm")
    
    # Get dimensions of the image
    rows, cols, bands = img.shape
    
    # First identify nodata values in the first band
    if verbose:
        print("Reading first band to identify nodata values...")
    
    first_band = img.read_band(0)
    
    # Check for common nodata values and create a valid pixel mask
    nodata_candidates = [-9999.0, -9999, 0]
    valid_mask = np.ones_like(first_band, dtype=bool)
    
    for val in nodata_candidates:
        val_mask = first_band == val
        val_count = np.sum(val_mask)
        if val_count > (0.01 * first_band.size):  # If more than 1% of pixels
            if verbose:
                print(f"Found likely nodata value: {val} ({val_count} pixels, {val_count/first_band.size*100:.2f}%)")
            valid_mask = valid_mask & ~val_mask
    
    # Also check for NaN values
    nan_mask = np.isnan(first_band)
    if np.sum(nan_mask) > 0:
        if verbose:
            print(f"Found {np.sum(nan_mask)} NaN values")
        valid_mask = valid_mask & ~nan_mask
    
    # Report on valid data percentage
    valid_pixels = np.sum(valid_mask)
    total_pixels = valid_mask.size
    valid_percent = (valid_pixels / total_pixels) * 100
    
    if verbose:
        print(f"Valid pixels: {valid_pixels} ({valid_percent:.2f}% of total)")
    
    # Create visualization directory if needed
    if visualize:
        os.makedirs(viz_dir, exist_ok=True)
    
    # Now analyze bands to establish global statistics for normalization
    if verbose:
        print("\nSampling bands to determine global statistics...")
    
    # Sampling every nth band to estimate global statistics
    sample_step = max(1, bands // 20)  # Sample about 20 bands
    sample_indices = list(range(0, bands, sample_step))
    if bands-1 not in sample_indices:
        sample_indices.append(bands-1)  # Make sure to include last band
    
    # Arrays to collect statistics across bands
    all_means = []
    all_stds = []
    all_mins = []
    all_maxs = []
    all_p_lows = []
    all_p_highs = []
    
    # Analyze sample bands
    for band_idx in tqdm(sample_indices, desc="Analyzing bands", disable=not verbose):
        band_data = img.read_band(band_idx)
        valid_band_data = band_data[valid_mask]
        
        if len(valid_band_data) > 0:
            # Calculate statistics on valid data
            min_val = np.min(valid_band_data)
            max_val = np.max(valid_band_data)
            mean_val = np.mean(valid_band_data)
            std_val = np.std(valid_band_data)
            p_low, p_high = np.percentile(valid_band_data, percentile_clip)
            
            # Store values
            all_means.append(mean_val)
            all_stds.append(std_val)
            all_mins.append(min_val)
            all_maxs.append(max_val)
            all_p_lows.append(p_low)
            all_p_highs.append(p_high)
            
            # Display info (but limit to first few bands)
            if verbose and len(all_means) <= 5:
                wavelength = wavelengths[band_idx] if hasattr(img, 'bands') else band_idx
                print(f"Band {band_idx} ({wavelength:.2f} nm) statistics:")
                print(f"  Range: {min_val:.4f} to {max_val:.4f}")
                print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
                print(f"  {percentile_clip[0]}-{percentile_clip[1]} percentile: {p_low:.4f} to {p_high:.4f}")
    
    # Calculate global statistics based on sampled bands
    global_mean = np.mean(all_means)
    global_std = np.mean(all_stds)  # Using mean of stddevs as a typical measure
    global_min = np.min(all_mins)
    global_max = np.max(all_maxs)
    global_p_low = np.min(all_p_lows)
    global_p_high = np.max(all_p_highs)
    
    # Determine global threshold for outlier detection
    # Values more than outlier_threshold standard deviations from the mean
    # will be considered outliers
    outlier_low = global_mean - outlier_threshold * global_std
    outlier_high = global_mean + outlier_threshold * global_std
    
    # Use percentile values as data range bounds
    data_min = global_p_low
    data_max = global_p_high
    
    if verbose:
        print(f"\nGlobal statistics:")
        print(f"  Mean: {global_mean:.4f}, StdDev: {global_std:.4f}")
        print(f"  Min: {global_min:.4f}, Max: {global_max:.4f}")
        print(f"  {percentile_clip[0]}-{percentile_clip[1]} percentile range: {global_p_low:.4f} to {global_p_high:.4f}")
        print(f"  Outlier threshold: z-score > {outlier_threshold}")
        print(f"  Values outside [{outlier_low:.4f}, {outlier_high:.4f}] will be treated as outliers")
        print(f"  Normalization range: [{data_min:.4f}, {data_max:.4f}]")
    
    # If we have bands with extreme values
    if global_min < outlier_low or global_max > outlier_high:
        if verbose:
            print("\nDetected potential outlier values:")
            print(f"  Global min {global_min:.4f} < outlier threshold {outlier_low:.4f}" if global_min < outlier_low else "")
            print(f"  Global max {global_max:.4f} > outlier threshold {outlier_high:.4f}" if global_max > outlier_high else "")
    
    # Calculate how many tiles we can extract
    h_tiles = rows // tile_size
    w_tiles = cols // tile_size
    total_tiles = h_tiles * w_tiles
    
    if verbose:
        print(f"\nWill create {h_tiles}×{w_tiles} = {total_tiles} tiles of size {tile_size}×{tile_size}")
    
    # Pre-allocate array for all extracted tiles
    tiles_array = np.zeros((total_tiles, tile_size, tile_size, bands), dtype=np.float32)
    
    # Process data for each band
    if verbose:
        print(f"\nProcessing data and creating tiles...")
    
    # Process each band with uniform normalization
    band_min_max = []  # Store actual min/max after processing
    
    for band_idx in tqdm(range(bands), desc="Processing bands", disable=not verbose):
        # Read the band
        band_data = img.read_band(band_idx)
        
        # Apply mask to replace invalid data with zero
        band_data_clean = np.where(valid_mask, band_data, 0)
        
        # Replace outliers with zeros
        outlier_mask = (band_data_clean < outlier_low) | (band_data_clean > outlier_high)
        if np.any(outlier_mask):
            band_data_clean[outlier_mask] = 0
            if verbose and band_idx % 20 == 0:  # Show for some bands
                outlier_count = np.sum(outlier_mask)
                print(f"  Band {band_idx}: Replaced {outlier_count} outliers ({outlier_count/(rows*cols)*100:.2f}%)")
        
        # Normalize to [0,1] using the global data range
        normalized_band = np.clip(band_data_clean, data_min, data_max)
        normalized_band = ((normalized_band - data_min) / (data_max - data_min)).astype(np.float32)
        
        # Store the actual min/max of normalized data
        band_min_max.append((np.min(normalized_band), np.max(normalized_band)))
        
        # Extract tiles for this band
        tile_idx = 0
        for i in range(h_tiles):
            for j in range(w_tiles):
                h_start = i * tile_size
                h_end = (i + 1) * tile_size
                w_start = j * tile_size
                w_end = (j + 1) * tile_size
                
                # Store this band's data for this tile
                tiles_array[tile_idx, :, :, band_idx] = normalized_band[h_start:h_end, w_start:w_end]
                tile_idx += 1
        
        # Create visualizations for a few bands
        if visualize and (band_idx < 5 or band_idx % 50 == 0 or band_idx > bands-5):
            wavelength = wavelengths[band_idx] if hasattr(img, 'bands') else band_idx
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original data
            im0 = axes[0].imshow(band_data, cmap='viridis')
            axes[0].set_title(f"Band {band_idx} ({wavelength:.2f} nm) - Raw Data")
            plt.colorbar(im0, ax=axes[0])
            
            # Masked data
            masked_data = np.copy(band_data)
            masked_data[~valid_mask | outlier_mask] = np.nan
            im1 = axes[1].imshow(masked_data, cmap='viridis')
            axes[1].set_title(f"Band {band_idx} - Valid Data Only")
            plt.colorbar(im1, ax=axes[1])
            
            # Normalized data
            im2 = axes[2].imshow(normalized_band, cmap='viridis', vmin=0, vmax=1)
            axes[2].set_title(f"Band {band_idx} - Normalized")
            plt.colorbar(im2, ax=axes[2])
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"band_{band_idx}_visualization.png"), dpi=150)
            plt.close()
            
            # Histogram of values
            plt.figure(figsize=(10, 6))
            valid_data = band_data[valid_mask]
            plt.hist(valid_data, bins=100, alpha=0.6, color='blue')
            plt.axvline(data_min, color='r', linestyle='--', label=f'Min: {data_min:.4f}')
            plt.axvline(data_max, color='g', linestyle='--', label=f'Max: {data_max:.4f}')
            plt.axvline(outlier_low, color='k', linestyle=':', label=f'Outlier Low: {outlier_low:.4f}')
            plt.axvline(outlier_high, color='k', linestyle='-.', label=f'Outlier High: {outlier_high:.4f}')
            plt.title(f"Band {band_idx} ({wavelength:.2f} nm) - Histogram")
            plt.legend()
            plt.savefig(os.path.join(viz_dir, f"band_{band_idx}_histogram.png"), dpi=150)
            plt.close()
    
    # Convert to tensor
    tiles_tensor = torch.tensor(tiles_array, dtype=torch.float32)
    
    if verbose:
        print(f"Final tiles tensor shape: {tiles_tensor.shape}")
    
    # Save to output file if specified
    if output_file:
        if verbose:
            print(f"Saving processed data to: {output_file}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save tensor
        torch.save(tiles_tensor, output_file)
        
        # Also save normalization metadata for reference
        metadata = {
            'aviris_file': aviris_path,
            'tile_size': tile_size,
            'total_tiles': total_tiles,
            'tile_shape': f"{tile_size}x{tile_size}x{bands}",
            'percentile_clip': percentile_clip,
            'outlier_threshold': outlier_threshold,
            'global_mean': global_mean,
            'global_std': global_std,
            'global_min': global_min,
            'global_max': global_max,
            'data_min': data_min,
            'data_max': data_max,
            'outlier_low': outlier_low,
            'outlier_high': outlier_high,
            'has_wavelengths': hasattr(img, 'bands') and hasattr(img.bands, 'centers'),
        }
        
        if hasattr(img, 'bands') and hasattr(img.bands, 'centers'):
            metadata['wavelengths_nm'] = img.bands.centers
        
        # Save metadata as NumPy file
        np.save(output_file.replace('.pt', '_metadata.npy'), metadata)
        
        if verbose:
            print(f"Data saved successfully to {output_file}")
            print(f"Metadata saved to {output_file.replace('.pt', '_metadata.npy')}")
    
    return tiles_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert AVIRIS data to tiled PyTorch tensor with uniform normalization")
    parser.add_argument("--input", type=str, 
                        default="AVIRIS/f170429t01p00r11rdn_e/f170429t01p00r11rdn_e_sc01_ort_img.hdr",
                        help="Input AVIRIS header file")
    parser.add_argument("--output", type=str, 
                        default="cache/aviris_tiles_swir_radiance.pt",
                        help="Output PyTorch file path")
    parser.add_argument("--tile-size", type=int, default=100, 
                        help="Size of tiles to extract (square)")
    parser.add_argument("--percentile-low", type=float, default=0.5,
                        help="Lower percentile for data range (default: 0.5)")
    parser.add_argument("--percentile-high", type=float, default=99.5,
                        help="Upper percentile for data range (default: 99.5)")
    parser.add_argument("--outlier-threshold", type=float, default=3.0,
                        help="Z-score threshold for outlier detection (default: 3.0)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualization generation")
    parser.add_argument("--viz-dir", type=str, default="aviris_visualization",
                        help="Directory to save visualizations")
    parser.add_argument("--quiet", action="store_true", 
                        help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Process the data
    try:
        process_aviris_to_tiles(
            aviris_path=args.input,
            tile_size=args.tile_size,
            output_file=args.output,
            percentile_clip=(args.percentile_low, args.percentile_high),
            outlier_threshold=args.outlier_threshold,
            visualize=not args.no_viz,
            viz_dir=args.viz_dir,
            verbose=not args.quiet
        )
        print(f"Successfully converted AVIRIS data to {args.output}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
