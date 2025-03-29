#!/usr/bin/env python3
"""
AVIRIS Spectrum Analyzer
------------------------
Analyzes hyperspectral data from AVIRIS by sampling at strategic points and 
generating spectrum plots and wavelength histograms.
Includes bad band removal and interpolation.
"""

import numpy as np
import matplotlib.pyplot as plt
import spectral
import os
from spectral.io import envi
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import scipy.interpolate

# Define paths
BASE_DIR = os.path.expanduser("~/ProjectsLFS/iccp_rcwa/S4/iccp_test")
DATA_DIR = os.path.join(BASE_DIR, "AVIRIS/f230919t01p00r11rfl")
OUTPUT_DIR = os.path.join(BASE_DIR, "AVIRIS_analysis/f230919t01p00r11rfl_stats")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "spectra"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "histograms"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "spatial"), exist_ok=True)

def load_aviris_data():
    """Load the AVIRIS hyperspectral data"""
    img_file = os.path.join(DATA_DIR, "f230919t01p00r11_rfl")
    hdr_file = os.path.join(DATA_DIR, "f230919t01p00r11_rfl.hdr")
    
    print(f"Loading AVIRIS data from {hdr_file}")
    
    # Check if the header file exists
    if not os.path.exists(hdr_file):
        print(f"Error: Header file {hdr_file} not found!")
        raise FileNotFoundError(f"Header file {hdr_file} not found")
    
    # Check if the data file exists
    if not os.path.exists(img_file):
        print(f"Error: Data file {img_file} not found!")
        raise FileNotFoundError(f"Data file {img_file} not found")
    
    # Load the image
    try:
        img = spectral.open_image(hdr_file)
        print(f"Image dimensions: {img.shape}")
        print(f"Number of bands: {img.nbands}")
        # print(f"Max: {max(img)}")
        # print(f"Min: {min(img)}")
        print(f"Max: {img.asarray().max()}")
        print(f"Min: {img.asarray().min()}")
        

        raise
        
        # Extract wavelength information if available
        wavelengths = None
        if hasattr(img, 'bands') and hasattr(img.bands, 'centers'):
            wavelengths = np.array(img.bands.centers)
            print(f"Wavelength range: {np.min(wavelengths):.2f} to {np.max(wavelengths):.2f} nm")
            
            # Check if we have the right number of wavelengths
            if len(wavelengths) != img.nbands:
                print(f"Warning: Number of wavelengths ({len(wavelengths)}) doesn't match number of bands ({img.nbands})")
        else:
            print("No wavelength information found in the header file")
            # Create default wavelength array
            wavelengths = np.arange(img.nbands)
            
        # Display additional information from the header if available
        if hasattr(img, 'metadata'):
            print("\nHeader metadata:")
            for key, value in img.metadata.items():
                if key not in ['description', 'wavelength', 'band names']:
                    print(f"  {key}: {value}")
        
        return img, wavelengths
        
    except Exception as e:
        print(f"Error loading AVIRIS data: {e}")
        raise e

def remove_and_interpolate_bad_bands(wavelengths, bad_ranges=None):
    """
    Create a mask for bad bands based on wavelength ranges.
    
    Parameters:
    -----------
    wavelengths : numpy.ndarray
        Array of wavelength values in nm
    bad_ranges : list of tuples, optional
        List of (min_wavelength, max_wavelength) tuples defining bad band ranges
        Default is [(1263, 1562), (1761, 1958)] nm
        
    Returns:
    --------
    bad_bands_mask : numpy.ndarray
        Boolean mask where True indicates a bad band
    """
    if bad_ranges is None:
        # Default bad wavelength ranges in nm based on the provided instructions
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
    
    return bad_bands_mask

def interpolate_bad_bands(spectrum, bad_bands_mask):
    """
    Interpolate values over bad bands in a spectrum.
    
    Parameters:
    -----------
    spectrum : numpy.ndarray
        1D array of spectral values
    bad_bands_mask : numpy.ndarray
        Boolean mask where True indicates a bad band
        
    Returns:
    --------
    interpolated_spectrum : numpy.ndarray
        Spectrum with bad bands replaced by interpolated values
    """
    # Create a copy of the spectrum
    interpolated_spectrum = spectrum.copy()
    
    # Check for NaN or invalid values in the spectrum
    valid_mask = (~np.isnan(spectrum)) & (spectrum > -1000)
    
    # If all values are invalid, return the original spectrum
    if not np.any(valid_mask):
        return spectrum
    
    # Get indices of all bands and good bands
    all_indices = np.arange(len(spectrum))
    
    # Combine the bad bands mask with the valid mask
    # We only interpolate over bad bands and preserve no-data values in other bands
    combined_mask = bad_bands_mask & valid_mask
    
    # If there are no bad bands that have valid data, return the original spectrum
    if not np.any(combined_mask):
        return spectrum
    
    # Get the good bands (not in bad_bands_mask and valid data)
    good_mask = (~bad_bands_mask) & valid_mask
    good_indices = all_indices[good_mask]
    good_values = spectrum[good_mask]
    
    # Interpolate only if we have good values
    if len(good_values) > 1:  # Need at least 2 points for interpolation
        # Use linear interpolation to fill bad bands
        interp_func = scipy.interpolate.interp1d(
            good_indices, good_values, 
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        
        # Apply interpolation to bad bands
        interpolated_spectrum[combined_mask] = interp_func(all_indices[combined_mask])
    
    return interpolated_spectrum

class InterpolatedImage:
    """
    Wrapper class for hyperspectral image that provides on-the-fly interpolation
    of bad spectral bands with optional normalization.
    """
    def __init__(self, original_img, bad_bands_mask, normalize=False):
        self.img = original_img
        self.bad_bands_mask = bad_bands_mask
        self.shape = original_img.shape
        self.nbands = original_img.nbands
        self.normalize = normalize
        
        # Copy attributes from original image
        for attr in dir(original_img):
            if not attr.startswith('__') and attr not in ['read_pixel']:
                try:
                    setattr(self, attr, getattr(original_img, attr))
                except:
                    pass

    def read_pixel(self, row, col):
        """Read a pixel, interpolate bad bands, and optionally normalize"""
        try:
            spectrum = self.img.read_pixel(row, col)
            interpolated = interpolate_bad_bands(spectrum, self.bad_bands_mask)
            
            if self.normalize:
                # Normalize only valid values (not NaN or no-data)
                valid_mask = (~np.isnan(interpolated)) & (interpolated > -1000)
                if np.any(valid_mask):
                    valid_min = np.min(interpolated[valid_mask])
                    valid_max = np.max(interpolated[valid_mask])
                    
                    # Only normalize if we have a valid range
                    if valid_max > valid_min:
                        # Create a copy to avoid modifying original values
                        normalized = interpolated.copy()
                        # Apply normalization only to valid values
                        normalized[valid_mask] = (interpolated[valid_mask] - valid_min) / (valid_max - valid_min)
                        return normalized
            
            return interpolated
            
        except Exception as e:
            # If interpolation fails, return the original spectrum
            print(f"Warning: Interpolation failed for pixel ({row},{col}): {e}")
            return self.img.read_pixel(row, col)

def get_sample_points(img_shape, max_attempts=100, sample_band=50):
    """Generate sample points at corners, center, and half-corners"""
    rows, cols, _ = img_shape
    
    # Define margin to avoid edge artifacts (5% of dimensions)
    margin_r = int(rows * 0.05)
    margin_c = int(cols * 0.05)
    
    # Potential point locations
    points_to_try = {
        "top_left": (margin_r, margin_c),
        "top_right": (margin_r, cols - margin_c - 1),
        "bottom_left": (rows - margin_r - 1, margin_c),
        "bottom_right": (rows - margin_r - 1, cols - margin_c - 1),
        "center": (rows // 2, cols // 2),
        "mid_top": (margin_r, cols // 2),
        "mid_right": (rows // 2, cols - margin_c - 1),
        "mid_bottom": (rows - margin_r - 1, cols // 2),
        "mid_left": (rows // 2, margin_c)
    }
    
    # Generate valid points based on actual data values
    points = {}
    invalid_count = 0
    
    print("Finding valid sampling points...")
    for name, (r, c) in points_to_try.items():
        points[name] = (r, c)
    
    # Generate random points that contain valid data
    random_points = {}
    random_attempts = 0
    random_count = 0
    
    while random_count < 10 and random_attempts < max_attempts:
        r = np.random.randint(margin_r, rows - margin_r)
        c = np.random.randint(margin_c, cols - margin_c)
        random_attempts += 1
        
        # We'll verify validity later during spectrum extraction
        random_points[f"random_{random_count+1}"] = (r, c)
        random_count += 1
    
    if random_count < 10:
        print(f"Warning: Could only find {random_count} valid random points after {max_attempts} attempts")
    
    return {**points, **random_points}

def plot_spectra(img, points, wavelengths=None):
    """Plot spectra for all sample points"""
    # Create a colormap for the plots
    point_colors = plt.cm.tab20(np.linspace(0, 1, len(points)))
    
    plt.figure(figsize=(14, 8))
    
    spectra_data = {}
    valid_points = 0
    
    for i, (point_name, (r, c)) in enumerate(points.items()):
        # Extract spectrum at this point
        try:
            # Use read_pixel method instead of direct indexing
            # This handles the data format properly
            spectrum = img.read_pixel(r, c)
            
            # Ensure we have a 1D array
            if isinstance(spectrum, np.ndarray) and spectrum.ndim > 1:
                spectrum = spectrum.flatten()
            
            # Handle potential NaN or invalid values
            if np.any(np.isnan(spectrum)) or np.any(spectrum < -1000):
                print(f"Warning: Point {point_name} at ({r},{c}) contains invalid values.")
                continue
                
            # Store spectrum for later analysis
            spectra_data[point_name] = spectrum
            valid_points += 1
            
            # Create x-axis for plotting
            if wavelengths is not None:
                x = wavelengths
                plt.xlabel('Wavelength (nm)')
            else:
                x = np.arange(len(spectrum))
                plt.xlabel('Band Index')
            
            # Plot spectrum with point-specific color
            plt.plot(x, spectrum, label=f"{point_name} ({r}, {c})", color=point_colors[i])
            
        except Exception as e:
            print(f"Error processing point {point_name} at ({r},{c}): {e}")
    
    plt.title('Spectra at Various Sample Points')
    plt.ylabel('Reflectance')
    plt.grid(True, alpha=0.3)
    
    # Only add legend if we have valid points
    if valid_points > 0:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        print("Warning: No valid points found for spectral plotting.")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "spectra", "all_spectra.png"), dpi=300)
    plt.close()
    
    # Save individual spectra too
    for i, (point_name, spectrum) in enumerate(spectra_data.items()):
        plt.figure(figsize=(10, 6))
        
        if wavelengths is not None:
            x = wavelengths
            plt.xlabel('Wavelength (nm)')
        else:
            x = np.arange(len(spectrum))
            plt.xlabel('Band Index')
            
        plt.plot(x, spectrum, color=point_colors[i % len(point_colors)])
        plt.title(f'Spectrum at {point_name}')
        plt.ylabel('Reflectance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "spectra", f"spectrum_{point_name}.png"), dpi=300)
        plt.close()
    
    # Create a CSV with all spectra for further analysis
    if len(spectra_data) > 0:
        if wavelengths is not None:
            df = pd.DataFrame({'Wavelength (nm)': wavelengths})
            for point_name, spectrum in spectra_data.items():
                df[point_name] = spectrum
        else:
            df = pd.DataFrame()
            for point_name, spectrum in spectra_data.items():
                df[point_name] = spectrum
                
        df.to_csv(os.path.join(OUTPUT_DIR, "spectra", "all_spectra.csv"), index=False)
    else:
        print("No valid spectra data to save to CSV.")
    
    return spectra_data

def create_wavelength_histograms(img, wavelengths=None, sample_size=10000):
    """Create histograms for each wavelength by sampling the image"""
    rows, cols, bands = img.shape
    
    # Determine sampling rate
    total_pixels = rows * cols
    
    # This is a large image, so we'll use random sampling
    print(f"Randomly sampling {sample_size} pixels for histograms")
    
    # Initialize a matrix to hold our samples
    samples = np.zeros((sample_size, bands))
    valid_samples = 0
    max_attempts = sample_size * 5  # Limit attempts to avoid infinite loops
    attempts = 0
    
    print("Collecting samples for histogram analysis...")
    # Sample random bands to speed up the process
    sample_bands = [0, bands//4, bands//2, 3*bands//4, bands-1]
    
    while valid_samples < sample_size and attempts < max_attempts:
        # Generate random coordinates
        r = np.random.randint(0, rows)
        c = np.random.randint(0, cols)
        attempts += 1
        
        try:
            # Read the pixel spectrum
            pixel_spectrum = img.read_pixel(r, c)
            
            # Check if this is a valid pixel (check a few bands)
            is_valid = True
            for band in sample_bands:
                if band < bands:
                    if np.isnan(pixel_spectrum[band]) or pixel_spectrum[band] < -1000:
                        is_valid = False
                        break
            
            if is_valid:
                samples[valid_samples, :] = pixel_spectrum
                valid_samples += 1
                
                # Progress indicator
                if valid_samples % 100 == 0:
                    print(f"  Collected {valid_samples}/{sample_size} valid samples...")
        
        except Exception as e:
            # Just skip this pixel
            pass
    
    # Trim the samples array to the actual number of valid samples
    samples = samples[:valid_samples, :]
    
    print(f"Collected {valid_samples} valid samples for histogram analysis")
    if valid_samples < sample_size:
        print(f"Warning: Could only collect {valid_samples}/{sample_size} valid samples after {attempts} attempts")
    
    # Wavelength bands to visualize histograms for
    # Choose bands distributed across the spectrum
    num_hist_bands = min(20, bands)  # Max 20 histograms for readability
    band_indices = np.linspace(0, bands-1, num_hist_bands, dtype=int)
    
    # Create a summary figure with multiple histograms
    fig, axes = plt.subplots(5, 4, figsize=(20, 16), constrained_layout=True)
    axes = axes.flatten()
    
    # Track overall data statistics
    all_data = []
    band_stats = []
    
    # Plot histograms for selected bands
    for i, band_idx in enumerate(band_indices):
        if i < len(axes):
            # Extract band data
            band_data = samples[:, band_idx]
            
            # Filter out no-data values (often set to very negative values)
            valid_mask = (band_data > -1000) & (~np.isnan(band_data))
            valid_data = band_data[valid_mask]
            
            if len(valid_data) > 0:
                all_data.extend(valid_data)
                
                # Calculate statistics
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                mean_val = np.mean(valid_data)
                median_val = np.median(valid_data)
                std_val = np.std(valid_data)
                
                # Store band statistics
                band_stats.append({
                    'Band': band_idx,
                    'Wavelength_nm': wavelengths[band_idx] if wavelengths is not None else None,
                    'Min': min_val,
                    'Max': max_val,
                    'Mean': mean_val,
                    'Median': median_val,
                    'Std': std_val,
                    'Valid_Samples': len(valid_data),
                    'Invalid_Samples': len(band_data) - len(valid_data)
                })
                
                # Create histogram
                sns.histplot(valid_data, kde=True, ax=axes[i])
                
                # Add wavelength information
                if wavelengths is not None:
                    axes[i].set_title(f'Band {band_idx} ({wavelengths[band_idx]:.2f} nm)')
                else:
                    axes[i].set_title(f'Band {band_idx}')
                
                # Add statistics
                stat_text = f"Min: {min_val:.4f}\nMax: {max_val:.4f}\nMean: {mean_val:.4f}\nStd: {std_val:.4f}"
                axes[i].text(0.98, 0.98, stat_text, 
                            transform=axes[i].transAxes, 
                            horizontalalignment='right',
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Hide any unused axes
    for i in range(len(band_indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Histograms of Reflectance Values for Selected Bands', fontsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR, "histograms", "band_histograms.png"), dpi=300)
    plt.close()
    
    # Create a CSV with band statistics
    stats_df = pd.DataFrame(band_stats)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "histograms", "band_statistics.csv"), index=False)
    
    # Plot overall data distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(all_data, kde=True, bins=100)
    plt.title('Overall Reflectance Distribution (All Sampled Bands)')
    plt.xlabel('Reflectance')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "histograms", "overall_distribution.png"), dpi=300)
    plt.close()
    
    # Plot boxplots for band statistics
    plt.figure(figsize=(14, 8))
    stats_df_long = pd.melt(stats_df, 
                           id_vars=['Band', 'Wavelength_nm'], 
                           value_vars=['Min', 'Max', 'Mean', 'Median', 'Std'],
                           var_name='Statistic', value_name='Value')
    
    # If wavelengths are available, use them for x-axis
    if wavelengths is not None:
        pivot_df = stats_df_long.pivot(index='Wavelength_nm', columns='Statistic', values='Value')
        pivot_df = pivot_df.sort_index()
        pivot_df.plot()
        plt.xlabel('Wavelength (nm)')
    else:
        pivot_df = stats_df_long.pivot(index='Band', columns='Statistic', values='Value')
        pivot_df = pivot_df.sort_index()
        pivot_df.plot()
        plt.xlabel('Band Index')
    
    plt.ylabel('Value')
    plt.title('Band Statistics Across Spectrum')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "histograms", "band_statistics_plot.png"), dpi=300)
    plt.close()
    
    return band_stats

def visualize_spatial_samples(img, points, bands_to_plot=None):
    """Visualize the spatial distribution of sample points and selected bands"""
    rows, cols, bands = img.shape
    
    print("Visualizing spatial distribution of sample points...")
    
    # If no specific bands are provided, choose a few distributed across the spectrum
    if bands_to_plot is None:
        bands_to_plot = [0, bands//4, bands//2, 3*bands//4, bands-1]
    
    # Limit the number of bands to visualize
    bands_to_plot = bands_to_plot[:5]  # Max 5 bands for clarity
    
    # Create a figure for spatial visualization
    fig, axes = plt.subplots(1, len(bands_to_plot), figsize=(5*len(bands_to_plot), 5))
    if len(bands_to_plot) == 1:
        axes = [axes]
    
    # Plot each band as a 2D image
    for i, band_idx in enumerate(bands_to_plot):
        print(f"  Visualizing band {band_idx}...")
        
        # Load the band data
        try:
            # For large images, we need to subsample to avoid memory issues
            # Let's downsample the image for visualization
            downsample_factor = max(1, rows // 1000, cols // 1000)
            
            if downsample_factor > 1:
                print(f"  Downsampling image by factor of {downsample_factor} for visualization...")
                
                # Read the band data in chunks
                band_data = np.zeros((rows//downsample_factor, cols//downsample_factor))
                
                for r in range(0, rows, downsample_factor):
                    if r + downsample_factor > rows:
                        continue
                    for c in range(0, cols, downsample_factor):
                        if c + downsample_factor > cols:
                            continue
                        try:
                            # Read the center pixel of each chunk
                            pixel_value = img.read_pixel(r + downsample_factor//2, c + downsample_factor//2)[band_idx]
                            band_data[r//downsample_factor, c//downsample_factor] = pixel_value
                        except:
                            # If error, use a nodata value
                            band_data[r//downsample_factor, c//downsample_factor] = -9999
            else:
                # Read the entire band
                band_data = img.read_band(band_idx)
            
            # Handle NaN and no-data values
            valid_mask = (band_data > -1000) & (~np.isnan(band_data))
            
            if np.sum(valid_mask) > 0:
                # Calculate valid min/max for consistent scaling
                valid_min = np.percentile(band_data[valid_mask], 5)  # 5th percentile to avoid outliers
                valid_max = np.percentile(band_data[valid_mask], 95)  # 95th percentile to avoid outliers
                
                # Create masked array for better visualization
                masked_data = np.ma.masked_where(~valid_mask, band_data)
                
                # Display the band image
                im = axes[i].imshow(masked_data, cmap='viridis', vmin=valid_min, vmax=valid_max)
                
                # Add colorbar
                plt.colorbar(im, ax=axes[i], label='Reflectance')
                
                # Adjust point coordinates for downsampling
                if downsample_factor > 1:
                    # Mark all sample points - adjust coordinates for downsampling
                    for point_name, (r, c) in points.items():
                        marker = 'o' if not point_name.startswith('random') else 'x'
                        color = 'red' if not point_name.startswith('random') else 'white'
                        
                        # Adjust coordinates
                        r_adj = r // downsample_factor
                        c_adj = c // downsample_factor
                        
                        if 0 <= r_adj < band_data.shape[0] and 0 <= c_adj < band_data.shape[1]:
                            axes[i].plot(c_adj, r_adj, marker=marker, markersize=10, color=color)
                            
                            # Add labels for non-random points
                            if not point_name.startswith('random'):
                                axes[i].annotate(point_name, (c_adj, r_adj), xytext=(5, 5), textcoords='offset points',
                                               color='white', fontsize=8, backgroundcolor='black')
                else:
                    # Mark all sample points with original coordinates
                    for point_name, (r, c) in points.items():
                        marker = 'o' if not point_name.startswith('random') else 'x'
                        color = 'red' if not point_name.startswith('random') else 'white'
                        
                        if 0 <= r < band_data.shape[0] and 0 <= c < band_data.shape[1]:
                            axes[i].plot(c, r, marker=marker, markersize=10, color=color)
                            
                            # Add labels for non-random points
                            if not point_name.startswith('random'):
                                axes[i].annotate(point_name, (c, r), xytext=(5, 5), textcoords='offset points',
                                               color='white', fontsize=8, backgroundcolor='black')
            else:
                axes[i].text(0.5, 0.5, "No valid data in this band", 
                            ha='center', va='center', transform=axes[i].transAxes)
            
            # Set title
            if hasattr(img, 'bands') and hasattr(img.bands, 'centers'):
                axes[i].set_title(f'Band {band_idx} ({img.bands.centers[band_idx]:.2f} nm)')
            else:
                axes[i].set_title(f'Band {band_idx}')
                
        except Exception as e:
            print(f"  Error visualizing band {band_idx}: {e}")
            axes[i].text(0.5, 0.5, f"Error loading band {band_idx}", 
                        ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "spatial", "sample_points_visualization.png"), dpi=300)
    plt.close()

def analyze_outliers(img, band_stats, percentile_threshold=99.9):
    """Analyze potential outliers in the data"""
    rows, cols, bands = img.shape
    
    # Find bands with the highest standard deviation (potentially problematic)
    stats_df = pd.DataFrame(band_stats)
    high_std_bands = stats_df.nlargest(5, 'Std')
    
    print("\nPotential problematic bands (highest std dev):")
    print(high_std_bands[['Band', 'Wavelength_nm', 'Min', 'Max', 'Mean', 'Std']].to_string(index=False))
    
    # Visualize these bands
    bands_to_check = high_std_bands['Band'].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, band_idx in enumerate(bands_to_check):
        if i < len(axes) - 1:  # Reserve last plot for histogram
            # Load the band data
            band_data = img.read_band(band_idx)
            
            # Handle NaN and no-data values
            valid_mask = (band_data > -1000) & (~np.isnan(band_data))
            masked_data = np.ma.masked_where(~valid_mask, band_data)
            
            # Calculate threshold for outliers
            valid_data = band_data[valid_mask]
            threshold = np.percentile(valid_data, percentile_threshold)
            
            # Mark outliers in red
            outlier_mask = (band_data > threshold) & valid_mask
            
            # Plot the band
            im = axes[i].imshow(masked_data, cmap='viridis')
            plt.colorbar(im, ax=axes[i])
            
            # Overlay outliers
            if np.any(outlier_mask):
                outlier_y, outlier_x = np.where(outlier_mask)
                axes[i].scatter(outlier_x, outlier_y, color='red', s=1, alpha=0.5)
                
            # Title with band info
            if hasattr(img, 'bands') and hasattr(img.bands, 'centers'):
                axes[i].set_title(f'Band {band_idx} ({img.bands.centers[band_idx]:.2f} nm)\n'
                                f'Outliers > {percentile_threshold}th percentile')
            else:
                axes[i].set_title(f'Band {band_idx}\nOutliers > {percentile_threshold}th percentile')
    
    # Create histogram of all bands' standard deviations
    sns.histplot(stats_df['Std'], kde=True, ax=axes[-1])
    axes[-1].set_title('Distribution of Band Standard Deviations')
    axes[-1].set_xlabel('Standard Deviation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "spatial", "outlier_analysis.png"), dpi=300)
    plt.close()
    
    # Return potential problematic bands
    return high_std_bands

def main():
    print("Starting AVIRIS data analysis")
    
    try:
        # Load the data
        img, wavelengths = load_aviris_data()
        
        # Process bad bands if wavelength information is available
        if wavelengths is not None:
            print("Identifying and interpolating bad wavelength bands...")
            bad_bands_mask = remove_and_interpolate_bad_bands(wavelengths)
            
            # Wrap the original image with our interpolation handler
            # Enable normalization to rescale values to 0-1 range
            img_interpolated = InterpolatedImage(img, bad_bands_mask, normalize=True)
            
            # Use the interpolated image for all subsequent processing
            img = img_interpolated
            print("Created interpolated image wrapper with bad bands removed and normalization to 0-1 range")
        else:
            print("No wavelength information available, skipping bad band removal")
            # Initialize empty mask for the summary report
            bad_bands_mask = np.zeros(img.nbands, dtype=bool)
        
        # Get sample points
        points = get_sample_points(img.shape)
        print(f"Generated {len(points)} sample points")
        
        # Plot spectra for sample points
        print("Plotting spectra for sample points...")
        spectra_data = plot_spectra(img, points, wavelengths)
        
        # Check if we got any valid spectral data
        if not spectra_data:
            print("Warning: No valid spectral data was extracted. Adjusting approach...")
            
            # Try a focused approach with a smaller number of points
            print("Trying with manually selected points...")
            
            # Create a simple grid of points to try
            rows, cols, _ = img.shape
            grid_points = {}
            
            # Try a 3x3 grid of points
            for i in range(3):
                for j in range(3):
                    r = int(rows * (0.2 + 0.3 * i))
                    c = int(cols * (0.2 + 0.3 * j))
                    grid_points[f"grid_{i+1}_{j+1}"] = (r, c)
            
            print(f"Testing {len(grid_points)} grid points...")
            spectra_data = plot_spectra(img, grid_points, wavelengths)
            
            if spectra_data:
                print("Found valid data with grid approach!")
                points = grid_points
            else:
                print("Still no valid data. Will continue with limited analysis.")
        
        # Create histograms for wavelength bands
        print("Creating wavelength histograms...")
        band_stats = create_wavelength_histograms(img, wavelengths)
        
        # Visualize spatial distribution of samples
        print("Visualizing spatial distribution of samples...")
        visualize_spatial_samples(img, points)
        
        # Analyze potential outliers
        if band_stats:
            print("Analyzing potential outliers...")
            problem_bands = analyze_outliers(img, band_stats)
        else:
            print("Skipping outlier analysis due to insufficient data.")
            problem_bands = pd.DataFrame()
        
        print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}")
        
        # Write summary report
        with open(os.path.join(OUTPUT_DIR, "analysis_summary.txt"), 'w') as f:
            f.write("AVIRIS Data Analysis Summary\n")
            f.write("==========================\n\n")
            f.write(f"Data source: {DATA_DIR}\n")
            f.write(f"Image dimensions: {img.shape}\n")
            f.write(f"Number of bands: {img.nbands}\n")
            
            if wavelengths is not None:
                f.write(f"Wavelength range: {np.min(wavelengths):.2f} to {np.max(wavelengths):.2f} nm\n\n")
                f.write("Processing applied:\n")
                f.write("  - Bad bands identified and interpolated\n")
                f.write("  - Values normalized to 0-1 range after interpolation\n\n")
                f.write("Bad bands interpolated:\n")
                bad_band_indices = np.where(bad_bands_mask)[0]
                if len(bad_band_indices) > 0:
                    for i, band_range in enumerate(np.split(bad_band_indices, np.where(np.diff(bad_band_indices) != 1)[0] + 1)):
                        start_idx, end_idx = band_range[0], band_range[-1]
                        start_wl, end_wl = wavelengths[start_idx], wavelengths[end_idx]
                        f.write(f"  - Range {i+1}: Bands {start_idx}-{end_idx} ({start_wl:.2f}-{end_wl:.2f} nm)\n")
                else:
                    f.write("  No bad bands identified within the specified wavelength ranges\n")
            
            f.write("\nSample points analyzed:\n")
            for point_name, (r, c) in points.items():
                if not point_name.startswith('random'):
                    f.write(f"  - {point_name}: ({r}, {c})\n")
            
            if not problem_bands.empty:
                f.write("\nPotential problematic bands (highest standard deviation):\n")
                f.write(problem_bands[['Band', 'Wavelength_nm', 'Min', 'Max', 'Mean', 'Std']].to_string(index=False))
            else:
                f.write("\nNo problematic bands identified or analysis skipped.\n")
            
            f.write("\n\nAnalysis outputs:\n")
            f.write("  - Spectra plots: ./spectra/\n")
            f.write("  - Wavelength histograms: ./histograms/\n")
            f.write("  - Spatial visualizations: ./spatial/\n")
            
    except Exception as e:
        print(f"Error in main analysis: {e}")
        import traceback
        traceback.print_exc()
        
        # Create minimal output directory
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Write error report
        with open(os.path.join(OUTPUT_DIR, "error_report.txt"), 'w') as f:
            f.write("AVIRIS Data Analysis Error Report\n")
            f.write("===============================\n\n")
            f.write(f"Error occurred: {e}\n\n")
            f.write("Traceback:\n")
            traceback.print_exc(file=f)
            
        print(f"Error report saved to {os.path.join(OUTPUT_DIR, 'error_report.txt')}")

if __name__ == "__main__":
    main()
