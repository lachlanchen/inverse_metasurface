import numpy as np
import matplotlib.pyplot as plt
import spectral
import os
from pathlib import Path
import sys
from scipy import stats

# Target wavelengths in μm
TARGET_WAVELENGTHS_UM = [1.0, 1.5, 2.0, 2.5]
# Convert to nm for easier comparison
TARGET_WAVELENGTHS_NM = [w * 1000 for w in TARGET_WAVELENGTHS_UM]

def find_header_file(folder_path):
    """Find the appropriate header file for orthorectified image data"""
    # List of header files to prioritize in order of preference
    header_priorities = [
        "*sc01_ort_img.hdr",   # Orthorectified image data
        "*ort_img.hdr",        # Also orthorectified image data
        "*RFL_ORT.hdr",        # Reflectance orthorectified (capitalized)
        "*rfl_ort.hdr",        # Reflectance orthorectified
        "*.img.hdr",           # Any image file
        "*.hdr"                # Any header as last resort
    ]
    
    for pattern in header_priorities:
        matching_files = list(Path(folder_path).glob(pattern))
        if matching_files:
            print(f"Found {len(matching_files)} files matching pattern {pattern}")
            # Just take the first one if multiple matches
            return str(matching_files[0])
    
    print(f"No suitable header files found in {folder_path}")
    return None

def robust_normalize(data, valid_mask=None, clip_sigma=3.0, percentile_range=(1, 99)):
    """
    Apply robust normalization to avoid extreme values
    
    Parameters:
    data: Input data array
    valid_mask: Boolean mask of valid pixels (if None, assume all are valid)
    clip_sigma: Number of sigmas for outlier removal
    percentile_range: Percentile range for normalization
    
    Returns:
    normalized_data: Data scaled to [0,1] with extreme values handled
    """
    if valid_mask is None:
        valid_mask = np.ones_like(data, dtype=bool)
    
    # Get valid data
    valid_data = data[valid_mask]
    
    if len(valid_data) == 0:
        print("No valid data for normalization")
        return np.zeros_like(data)
    
    # Step 1: Get initial statistics
    mean_val = np.mean(valid_data)
    std_val = np.std(valid_data)
    p_low, p_high = np.percentile(valid_data, percentile_range)
    
    print(f"  Initial stats - Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    print(f"  Initial {percentile_range[0]}-{percentile_range[1]} percentile: {p_low:.4f} to {p_high:.4f}")
    
    # Step 2: Apply sigma clipping to find extreme outliers
    sigma_mask = (valid_data > mean_val - clip_sigma * std_val) & (valid_data < mean_val + clip_sigma * std_val)
    if np.sum(sigma_mask) < len(valid_data):
        outlier_count = len(valid_data) - np.sum(sigma_mask)
        print(f"  Removed {outlier_count} outliers ({outlier_count/len(valid_data)*100:.2f}%) using {clip_sigma}-sigma clipping")
        
        # Recalculate statistics without outliers
        valid_data_filtered = valid_data[sigma_mask]
        mean_val = np.mean(valid_data_filtered)
        std_val = np.std(valid_data_filtered)
        p_low, p_high = np.percentile(valid_data_filtered, percentile_range)
        
        print(f"  After outlier removal - Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        print(f"  After outlier removal - {percentile_range[0]}-{percentile_range[1]} percentile: {p_low:.4f} to {p_high:.4f}")
    
    # Step 3: Create a copy with nodata values replaced by mean
    processed_data = np.copy(data)
    processed_data[~valid_mask] = mean_val
    
    # Step 4: Normalize using percentile range after outlier removal
    if p_high > p_low:
        normalized_data = np.clip((processed_data - p_low) / (p_high - p_low), 0, 1)
    else:
        normalized_data = np.zeros_like(processed_data)
    
    return normalized_data

def process_aviris_folder(folder_path):
    """Process AVIRIS data in the given folder"""
    folder_name = os.path.basename(folder_path)
    print(f"\n==== Processing {folder_name} ====")
    
    # Find a header file
    hdr_file = find_header_file(folder_path)
    if not hdr_file:
        print(f"Skipping {folder_name} - no header file found")
        return
    
    print(f"Using header file: {os.path.basename(hdr_file)}")
    
    try:
        # Open the AVIRIS image
        img = spectral.open_image(hdr_file)
        print(f"Image dimensions: {img.shape}")
        print(f"Number of bands: {img.nbands}")
        
        # Check if wavelength information is available
        if not hasattr(img, 'bands') or not hasattr(img.bands, 'centers'):
            print(f"No wavelength information found in {hdr_file}")
            return
        
        wavelengths = np.array(img.bands.centers)
        print(f"Wavelength range: {min(wavelengths):.2f} to {max(wavelengths):.2f} nm")
        
        # Create a figure to hold all four wavelengths
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        # Find bands closest to target wavelengths
        for i, (target_um, target_nm) in enumerate(zip(TARGET_WAVELENGTHS_UM, TARGET_WAVELENGTHS_NM)):
            # Find closest band
            band_idx = np.abs(wavelengths - target_nm).argmin()
            actual_nm = wavelengths[band_idx]
            
            print(f"Target: {target_um} μm → Using band {band_idx} ({actual_nm:.2f} nm)")
            
            # Read band data - with error handling
            try:
                band_data = img.read_band(band_idx)
            except Exception as e:
                print(f"  Error reading band {band_idx}: {str(e)}")
                axes[i].text(0.5, 0.5, f"Error reading band {band_idx}",
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[i].transAxes)
                continue
            
            # Identify valid data - handle multiple potential nodata values
            # Common nodata values and large anomalies
            nodata_candidates = [-9999.0, -9999, -9.99e+36, 0]
            valid_mask = np.ones_like(band_data, dtype=bool)
            
            # Check for NaN values
            nan_mask = np.isnan(band_data)
            if np.any(nan_mask):
                print(f"  Found {np.sum(nan_mask)} NaN values")
                valid_mask = ~nan_mask
            
            # Check for common nodata values (only if they appear frequently)
            for val in nodata_candidates:
                val_mask = band_data == val
                val_count = np.sum(val_mask)
                if val_count > 0:
                    val_percentage = (val_count / band_data.size) * 100
                    if val_percentage > 1.0:  # If more than 1% of pixels
                        print(f"  Found likely nodata value: {val} ({val_count} pixels, {val_percentage:.2f}%)")
                        valid_mask = valid_mask & ~val_mask
            
            # Get basic stats for valid data
            valid_data = band_data[valid_mask]
            valid_count = np.sum(valid_mask)
            invalid_count = np.sum(~valid_mask)
            print(f"  Valid pixels: {valid_count} ({valid_count/band_data.size*100:.2f}%)")
            print(f"  Invalid pixels: {invalid_count} ({invalid_count/band_data.size*100:.2f}%)")
            
            if len(valid_data) > 0:
                # Calculate simple stats before normalization
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)
                
                print(f"  Raw data range: {min_val:.4f} to {max_val:.4f}")
                print(f"  Raw mean: {mean_val:.4f}, std: {std_val:.4f}")
                
                # Normalize data with robust handling of extreme values
                normalized_data = robust_normalize(
                    band_data, 
                    valid_mask=valid_mask,
                    clip_sigma=3.0,
                    percentile_range=(1, 99)
                )
                
                # Plot on the corresponding subplot
                ax = axes[i]
                im = ax.imshow(normalized_data, cmap='viridis')
                ax.set_title(f'{target_um} μm (Band {band_idx}: {actual_nm:.2f} nm)')
                plt.colorbar(im, ax=ax, label='Normalized Reflectance')
            else:
                print(f"  Warning: No valid data found in band {band_idx}")
                axes[i].text(0.5, 0.5, f"No valid data in band {band_idx}",
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[i].transAxes)
        
        # Set the overall title
        fig.suptitle(f"AVIRIS Data: {folder_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        # Save the figure in the data folder
        output_file = os.path.join(folder_path, f"{folder_name}_wavelength_comparison.png")
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Saved visualization to: {output_file}")
        plt.close(fig)
        
    except Exception as e:
        print(f"Error processing {folder_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return

def main():
    # Get the base AVIRIS directory
    aviris_base = Path("AVIRIS")
    
    if not aviris_base.exists() or not aviris_base.is_dir():
        print("AVIRIS directory not found in current location")
        return
    
    # Find all subdirectories
    aviris_folders = [f for f in aviris_base.iterdir() if f.is_dir()]
    
    if not aviris_folders:
        print("No subdirectories found in AVIRIS directory")
        return
    
    print(f"Found {len(aviris_folders)} AVIRIS data folders")
    
    # Process each folder
    for folder in aviris_folders:
        process_aviris_folder(str(folder))
    
    print("\nVisualization of all AVIRIS folders complete!")

if __name__ == "__main__":
    main()
