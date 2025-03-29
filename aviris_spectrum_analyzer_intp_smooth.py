#!/usr/bin/env python3
"""
AVIRIS_SWIR_INTP Visualizer
---------------------------
Visualizes already processed AVIRIS SWIR data to verify correctness.
No preprocessing is applied - just visualization of the existing data.
"""

import numpy as np
import matplotlib.pyplot as plt
import spectral
import os
import pandas as pd
from tqdm import tqdm
import seaborn as sns

# Define paths - adjust these to your requirements
BASE_DIR = os.path.expanduser("~/ProjectsLFS/iccp_rcwa/S4/iccp_test")
DATA_DIR = os.path.join(BASE_DIR, "AVIRIS_SWIR_INTP_SMOOTH")
OUTPUT_DIR = os.path.join(BASE_DIR, "AVIRIS_SMOOTH_verification")

def create_output_dirs(subfolder):
    """Create output directories for a given subfolder"""
    folder_path = os.path.join(OUTPUT_DIR, subfolder)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join(folder_path, "spectra"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "spatial"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "statistics"), exist_ok=True)
    return folder_path

def get_subfolders():
    """Get list of AVIRIS subfolders"""
    return [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

def load_processed_data(subfolder):
    """Load the processed AVIRIS data for a subfolder"""
    # Find the header file
    hdr_files = [f for f in os.listdir(os.path.join(DATA_DIR, subfolder)) 
                 if f.endswith('.hdr')]
    
    if not hdr_files:
        raise FileNotFoundError(f"No header file found in {os.path.join(DATA_DIR, subfolder)}")
    
    hdr_file = os.path.join(DATA_DIR, subfolder, hdr_files[0])
    
    # Load the image
    img = spectral.open_image(hdr_file)
    
    # Load wavelength data if available
    wavelengths = None
    try:
        wavelengths_file = os.path.join(DATA_DIR, subfolder, "numpy", "wavelengths.npy")
        if os.path.exists(wavelengths_file):
            wavelengths = np.load(wavelengths_file)
    except Exception as e:
        print(f"Warning: Couldn't load wavelengths file: {e}")
    
    # Load band statistics if available
    stats_df = None
    try:
        stats_file = os.path.join(DATA_DIR, subfolder, "band_statistics.csv")
        if os.path.exists(stats_file):
            stats_df = pd.read_csv(stats_file)
            print(f"Loaded band statistics with columns: {', '.join(stats_df.columns)}")
    except Exception as e:
        print(f"Warning: Couldn't load band statistics: {e}")
    
    return img, wavelengths, stats_df

def get_sample_points(img_shape):
    """Generate sample points at strategic locations"""
    rows, cols, _ = img_shape
    
    # Define margin to avoid edge artifacts (5% of dimensions)
    margin_r = int(rows * 0.05)
    margin_c = int(cols * 0.05)
    
    # Create a grid of points
    points = {}
    
    # 3x3 grid across the image
    for i in range(3):
        for j in range(3):
            r = margin_r + int((rows - 2*margin_r) * i / 2)
            c = margin_c + int((cols - 2*margin_c) * j / 2)
            points[f"grid_{i+1}_{j+1}"] = (r, c)
    
    # Add 5 random points
    for i in range(5):
        r = np.random.randint(margin_r, rows - margin_r)
        c = np.random.randint(margin_c, cols - margin_c)
        points[f"random_{i+1}"] = (r, c)
    
    return points

def plot_spectra(img, points, wavelengths, output_folder, subfolder):
    """Plot spectra for all sample points"""
    print(f"Plotting spectra for {subfolder}...")
    
    # Create a colormap for the plots
    point_colors = plt.cm.tab20(np.linspace(0, 1, len(points)))
    
    plt.figure(figsize=(14, 8))
    
    spectra_data = {}
    
    for i, (point_name, (r, c)) in enumerate(points.items()):
        # Extract spectrum at this point
        try:
            # Use read_pixel method
            spectrum = img.read_pixel(r, c)
            
            # Ensure we have a 1D array
            if isinstance(spectrum, np.ndarray) and spectrum.ndim > 1:
                spectrum = spectrum.flatten()
            
            # Store spectrum for later analysis
            spectra_data[point_name] = spectrum
            
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
    
    plt.title(f'Spectra at Various Sample Points - {subfolder}')
    plt.ylabel('Normalized Reflectance (0-1)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "spectra", "all_spectra.png"), dpi=300)
    plt.close()
    
    # Save individual spectra
    for i, (point_name, spectrum) in enumerate(spectra_data.items()):
        plt.figure(figsize=(10, 6))
        
        if wavelengths is not None:
            x = wavelengths
            plt.xlabel('Wavelength (nm)')
        else:
            x = np.arange(len(spectrum))
            plt.xlabel('Band Index')
            
        plt.plot(x, spectrum, color=point_colors[i % len(point_colors)])
        plt.title(f'Spectrum at {point_name} - {subfolder}')
        plt.ylabel('Normalized Reflectance (0-1)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)  # Force 0-1 range for visualization
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "spectra", f"spectrum_{point_name}.png"), dpi=300)
        plt.close()
    
    # Create a CSV with all spectra
    if spectra_data:
        if wavelengths is not None:
            df = pd.DataFrame({'Wavelength (nm)': wavelengths})
            for point_name, spectrum in spectra_data.items():
                df[point_name] = spectrum
        else:
            df = pd.DataFrame()
            for point_name, spectrum in spectra_data.items():
                df[point_name] = spectrum
                
        df.to_csv(os.path.join(output_folder, "spectra", "all_spectra.csv"), index=False)
    
    return spectra_data

def visualize_spatial_data(img, points, output_folder, subfolder, wavelengths=None):
    """Visualize spatial data at selected wavelengths/bands"""
    print(f"Creating spatial visualizations for {subfolder}...")
    
    rows, cols, bands = img.shape
    
    # Select bands to visualize (start, middle, end of range)
    bands_to_plot = [0, bands//4, bands//2, 3*bands//4, bands-1]
    
    # Create a multi-panel figure
    fig, axes = plt.subplots(1, len(bands_to_plot), figsize=(5*len(bands_to_plot), 5))
    if len(bands_to_plot) == 1:
        axes = [axes]
    
    for i, band_idx in enumerate(bands_to_plot):
        # Load band data
        band_data = img.read_band(band_idx)
        
        # Display the band image with 0-1 scale
        im = axes[i].imshow(band_data, cmap='viridis', vmin=0, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], label='Normalized Reflectance')
        
        # Plot sample points
        for point_name, (r, c) in points.items():
            marker = 'o' if not point_name.startswith('random') else 'x'
            color = 'red' if not point_name.startswith('random') else 'white'
            
            if 0 <= r < band_data.shape[0] and 0 <= c < band_data.shape[1]:
                axes[i].plot(c, r, marker=marker, markersize=10, color=color)
                
                # Add labels for non-random points only
                if not point_name.startswith('random'):
                    axes[i].annotate(point_name, (c, r), xytext=(5, 5), textcoords='offset points',
                                   color='white', fontsize=8, backgroundcolor='black')
        
        # Set title with wavelength if available
        if wavelengths is not None and band_idx < len(wavelengths):
            axes[i].set_title(f'Band {band_idx+1} ({wavelengths[band_idx]:.2f} nm)')
        else:
            axes[i].set_title(f'Band {band_idx+1}')
    
    plt.suptitle(f'Spatial Data Visualization - {subfolder}', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "spatial", "band_visualization.png"), dpi=300)
    plt.close()
    
    # Create individual images for each band with 0-1 scale
    print("Creating individual band images...")
    os.makedirs(os.path.join(output_folder, "spatial", "bands"), exist_ok=True)
    
    for band_idx in tqdm(range(bands)):
        band_data = img.read_band(band_idx)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(band_data, cmap='viridis', vmin=0, vmax=1)
        
        if wavelengths is not None and band_idx < len(wavelengths):
            plt.title(f'Band {band_idx+1} ({wavelengths[band_idx]:.2f} nm)')
        else:
            plt.title(f'Band {band_idx+1}')
            
        plt.colorbar(label='Normalized Reflectance (0-1)')
        plt.tight_layout()
        
        if wavelengths is not None and band_idx < len(wavelengths):
            wavelength_str = f"{wavelengths[band_idx]:.2f}".replace('.', 'p')
            filename = f"band_{band_idx+1:03d}_{wavelength_str}nm.png"
        else:
            filename = f"band_{band_idx+1:03d}.png"
            
        plt.savefig(os.path.join(output_folder, "spatial", "bands", filename), dpi=150)
        plt.close()

def analyze_statistics(img, wavelengths, stats_df, output_folder, subfolder):
    """Create statistical visualizations of the data"""
    print(f"Analyzing statistical properties for {subfolder}...")
    
    # Check if we need to generate statistics
    generate_new_stats = False
    if stats_df is None:
        generate_new_stats = True
    else:
        # Check if the loaded stats_df has the expected columns
        # The column names might be "Min_Value", "Max_Value" instead of "Min", "Max", etc.
        required_columns = ['Min_Value', 'Max_Value', 'Mean_Value', 'Std_Dev', 'Wavelength_nm']
        missing_columns = [col for col in required_columns if col not in stats_df.columns]
        if missing_columns:
            # If missing key columns, regenerate stats
            print(f"Statistics file missing columns: {missing_columns}. Will generate new statistics.")
            generate_new_stats = True
    
    # If we need to generate statistics, do it now
    if generate_new_stats:
        print("Generating statistics...")
        rows, cols, bands = img.shape
        
        # Sample a subset of pixels
        sample_size = min(10000, rows * cols)
        sample_rows = np.random.choice(rows, size=min(500, rows), replace=False)
        sample_cols = np.random.choice(cols, size=min(sample_size // 500, cols), replace=False)
        
        # Collect statistics for each band
        band_stats = []
        for band_idx in range(bands):
            # Get data for this band
            band_data = np.array([img.read_pixel(r, c)[band_idx] for r in sample_rows for c in sample_cols])
            
            # Calculate statistics
            min_val = np.min(band_data)
            max_val = np.max(band_data)
            mean_val = np.mean(band_data)
            median_val = np.median(band_data)
            std_val = np.std(band_data)
            
            band_stats.append({
                'Band': band_idx,
                'Wavelength_nm': wavelengths[band_idx] if wavelengths is not None and band_idx < len(wavelengths) else None,
                'Min_Value': min_val,
                'Max_Value': max_val,
                'Mean_Value': mean_val,
                'Median_Value': median_val,
                'Std_Dev': std_val
            })
        
        stats_df = pd.DataFrame(band_stats)
        
        # Save the generated statistics for reference
        stats_df.to_csv(os.path.join(output_folder, "statistics", "band_statistics.csv"), index=False)
    
    # Define column mappings - map the column names we expect to use to the actual columns in the dataframe
    min_col = 'Min_Value' if 'Min_Value' in stats_df.columns else 'Min' if 'Min' in stats_df.columns else None
    max_col = 'Max_Value' if 'Max_Value' in stats_df.columns else 'Max' if 'Max' in stats_df.columns else None
    mean_col = 'Mean_Value' if 'Mean_Value' in stats_df.columns else 'Mean' if 'Mean' in stats_df.columns else None
    median_col = 'Median_Value' if 'Median_Value' in stats_df.columns else 'Median' if 'Median' in stats_df.columns else None
    std_col = 'Std_Dev' if 'Std_Dev' in stats_df.columns else 'Std' if 'Std' in stats_df.columns else None
    
    # Create histograms of value distribution
    rows, cols, bands = img.shape
    
    # Sample data for histogram
    sample_size = min(50000, rows * cols)
    sample_rows = np.random.choice(rows, size=min(500, rows), replace=False)
    sample_cols = np.random.choice(cols, size=min(sample_size // 500, cols), replace=False)
    
    all_values = []
    for r in sample_rows:
        for c in sample_cols:
            try:
                pixel_data = img.read_pixel(r, c)
                all_values.extend(pixel_data)
            except:
                pass
    
    # Create distribution histogram
    plt.figure(figsize=(12, 6))
    sns.histplot(all_values, bins=100, kde=True)
    plt.title(f'Distribution of Normalized Reflectance Values - {subfolder}')
    plt.xlabel('Normalized Reflectance (0-1)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "statistics", "value_distribution.png"), dpi=300)
    plt.close()
    
    # Plot statistics across wavelengths if we have the necessary columns
    if 'Wavelength_nm' in stats_df.columns and not stats_df['Wavelength_nm'].isna().all():
        plt.figure(figsize=(15, 10))
        
        # Only create plots if we have the data
        if min_col and max_col:
            plt.subplot(3, 1, 1)
            plt.plot(stats_df['Wavelength_nm'], stats_df[min_col], 'b-', label='Min')
            plt.plot(stats_df['Wavelength_nm'], stats_df[max_col], 'r-', label='Max')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Value')
            plt.title('Min/Max Values Across Wavelengths')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        if mean_col and median_col:
            plt.subplot(3, 1, 2)
            plt.plot(stats_df['Wavelength_nm'], stats_df[mean_col], 'g-', label='Mean')
            if median_col in stats_df.columns:  # Make sure median exists
                plt.plot(stats_df['Wavelength_nm'], stats_df[median_col], 'm-', label='Median')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Value')
            plt.title('Mean/Median Values Across Wavelengths')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        if std_col:
            plt.subplot(3, 1, 3)
            plt.plot(stats_df['Wavelength_nm'], stats_df[std_col], 'k-')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Standard Deviation')
            plt.title('Standard Deviation Across Wavelengths')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "statistics", "wavelength_statistics.png"), dpi=300)
        plt.close()
    
    # Create a heatmap showing correlation between bands
    print("Creating band correlation heatmap...")
    sample_size = min(5000, rows*cols)
    sample_rows = np.random.choice(rows, size=min(100, rows), replace=False)
    sample_cols = np.random.choice(cols, size=min(sample_size // 100, cols), replace=False)
    
    # Sample data for correlation analysis
    band_samples = np.zeros((len(sample_rows) * len(sample_cols), bands))
    
    idx = 0
    for r in sample_rows:
        for c in sample_cols:
            try:
                if idx < band_samples.shape[0]:
                    band_samples[idx, :] = img.read_pixel(r, c)
                    idx += 1
            except:
                pass
    
    # Compute correlation matrix (using a subset of bands if there are many)
    max_bands_corr = min(30, bands)  # Limit for visualization
    if bands > max_bands_corr:
        band_indices = np.linspace(0, bands-1, max_bands_corr, dtype=int)
        band_samples_subset = band_samples[:, band_indices]
    else:
        band_indices = np.arange(bands)
        band_samples_subset = band_samples
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(band_samples_subset.T)
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    
    # Create labels with wavelengths if available
    if wavelengths is not None:
        # Make sure band_indices are all valid for indexing into wavelengths
        valid_indices = [i for i in band_indices if i < len(wavelengths)]
        if valid_indices:
            labels = [f"{i+1}\n({wavelengths[i]:.0f}nm)" for i in valid_indices]
        else:
            labels = [f"{i+1}" for i in range(len(corr_matrix))]
    else:
        labels = [f"{i+1}" for i in range(len(corr_matrix))]
    
    # Fix the xticklabels and yticklabels parameters
    # Only use labels if there aren't too many
    if len(labels) <= 20:
        ax = sns.heatmap(corr_matrix, cmap='viridis', vmin=-1, vmax=1, 
                        xticklabels=labels, yticklabels=labels)
    else:
        # If too many labels, don't show them
        ax = sns.heatmap(corr_matrix, cmap='viridis', vmin=-1, vmax=1)
    
    plt.title(f'Band Correlation Matrix - {subfolder}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "statistics", "band_correlation.png"), dpi=300)
    plt.close()
    
    return stats_df

def create_verification_report(subfolder, output_folder, stats_df, wavelengths):
    """Create a verification report summarizing the data"""
    report_file = os.path.join(output_folder, "verification_report.txt")
    
    with open(report_file, 'w') as f:
        f.write(f"AVIRIS_SWIR_INTP Verification Report - {subfolder}\n")
        f.write("="*50 + "\n\n")
        
        # Basic data information
        img_files = [file for file in os.listdir(os.path.join(DATA_DIR, subfolder)) 
                   if file.endswith('.hdr')]
        
        if img_files:
            img_file = img_files[0].replace('.hdr', '')
            f.write(f"Data source: {os.path.join(DATA_DIR, subfolder)}\n")
            f.write(f"Image file: {img_file}\n\n")
        else:
            f.write(f"Data source: {os.path.join(DATA_DIR, subfolder)}\n")
            f.write("Image file: Not found\n\n")
        
        # Data range verification
        if stats_df is not None:
            # Determine column names
            min_col = 'Min_Value' if 'Min_Value' in stats_df.columns else 'Min' if 'Min' in stats_df.columns else None
            max_col = 'Max_Value' if 'Max_Value' in stats_df.columns else 'Max' if 'Max' in stats_df.columns else None
            mean_col = 'Mean_Value' if 'Mean_Value' in stats_df.columns else 'Mean' if 'Mean' in stats_df.columns else None
            
            if min_col and max_col and mean_col:
                min_val = stats_df[min_col].min()
                max_val = stats_df[max_col].max()
                mean_val = stats_df[mean_col].mean()
                
                f.write("Data range verification:\n")
                f.write(f"  - Minimum value: {min_val:.6f}\n")
                f.write(f"  - Maximum value: {max_val:.6f}\n")
                f.write(f"  - Mean value: {mean_val:.6f}\n\n")
                
                # Check if data is properly normalized (0-1 range)
                if min_val < 0 or max_val > 1:
                    f.write("WARNING: Data range is not strictly within 0-1 bounds!\n")
                    f.write(f"  - Values < 0: {(stats_df[min_col] < 0).sum()} bands\n")
                    f.write(f"  - Values > 1: {(stats_df[max_col] > 1).sum()} bands\n\n")
                else:
                    f.write("Data is properly normalized within 0-1 range.\n\n")
        
        # Wavelength information
        if wavelengths is not None:
            f.write("Wavelength information:\n")
            f.write(f"  - Number of wavelengths: {len(wavelengths)}\n")
            f.write(f"  - Range: {np.min(wavelengths):.2f} to {np.max(wavelengths):.2f} nm\n\n")
        
        # List of visualization outputs
        f.write("Visualization outputs:\n")
        f.write(f"  - Spectra plots: {os.path.join(output_folder, 'spectra')}\n")
        f.write(f"  - Spatial visualizations: {os.path.join(output_folder, 'spatial')}\n")
        f.write(f"  - Statistical analysis: {os.path.join(output_folder, 'statistics')}\n\n")
        
        f.write("Verification complete.\n")

def process_subfolder(subfolder):
    """Process a single subfolder"""
    print(f"\n{'='*80}")
    print(f"Processing {subfolder}")
    print(f"{'='*80}")
    
    try:
        # Create output directory
        output_folder = create_output_dirs(subfolder)
        
        # Load the data
        img, wavelengths, stats_df = load_processed_data(subfolder)
        print(f"Loaded image with shape: {img.shape}")
        
        if wavelengths is not None:
            print(f"Loaded wavelength data: {len(wavelengths)} bands, range {np.min(wavelengths):.2f} to {np.max(wavelengths):.2f} nm")
        
        # Generate sample points
        points = get_sample_points(img.shape)
        
        # Create visualizations
        plot_spectra(img, points, wavelengths, output_folder, subfolder)
        visualize_spatial_data(img, points, output_folder, subfolder, wavelengths)
        stats_df = analyze_statistics(img, wavelengths, stats_df, output_folder, subfolder)
        
        # Create verification report
        create_verification_report(subfolder, output_folder, stats_df, wavelengths)
        
        print(f"Verification complete for {subfolder}. Results saved to {output_folder}")
        return True
        
    except Exception as e:
        print(f"Error processing {subfolder}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Starting AVIRIS_SWIR_INTP data verification")
    
    # Create main output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of subfolders
    subfolders = get_subfolders()
    
    if not subfolders:
        print(f"No subfolders found in {DATA_DIR}")
        return
    
    print(f"Found {len(subfolders)} subfolders to process: {', '.join(subfolders)}")
    
    # Process each subfolder
    results = {}
    for subfolder in subfolders:
        result = process_subfolder(subfolder)
        results[subfolder] = "Success" if result else "Failed"
    
    # Write summary report
    with open(os.path.join(OUTPUT_DIR, "verification_summary.txt"), 'w') as f:
        f.write("AVIRIS_SWIR_INTP Verification Summary\n")
        f.write("="*40 + "\n\n")
        
        for subfolder, status in results.items():
            f.write(f"{subfolder}: {status}\n")
    
    print("\nVerification complete!")
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
