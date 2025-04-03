#!/usr/bin/env python3
"""
AVIRIS Raw Data Analyzer
------------------------
This script reads original AVIRIS hyperspectral data without any processing,
creates histograms for each wavelength, randomly selects pixels to plot spectra,
and saves all visualizations to analyze the true distribution of values.
"""

import numpy as np
import matplotlib.pyplot as plt
import spectral
import os
import pandas as pd
from tqdm import tqdm
import seaborn as sns

# Define paths using actual directories from the system
BASE_DIR = os.path.expanduser("~/ProjectsLFS/iccp_rcwa/S4/iccp_test")
DATA_DIR = os.path.join(BASE_DIR, "AVIRIS")
OUTPUT_DIR = os.path.join(BASE_DIR, "AVIRIS_RAW_ANALYSIS")

# Number of random pixels to select for spectrum plots
NUM_RANDOM_PIXELS = 10

def create_output_dirs(dataset_name):
    """Create output directories for visualizations"""
    folder_path = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join(folder_path, "histograms"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "wavelength_frames"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "spectra"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "statistics"), exist_ok=True)
    return folder_path

def get_aviris_datasets():
    """Get list of AVIRIS datasets from the data directory"""
    datasets = []
    
    # Check all directories in the AVIRIS folder
    for dir_name in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_name)
        
        # Skip if not a directory
        if not os.path.isdir(dir_path):
            continue
            
        # Find header files in the directory
        hdr_files = []
        for file in os.listdir(dir_path):
            if file.endswith('.hdr'):
                hdr_files.append(file)
                
        # If header files found, add to datasets
        if hdr_files:
            datasets.append({
                'name': dir_name,
                'path': dir_path,
                'hdr_files': hdr_files
            })
    
    return datasets

def load_aviris_data(dataset_path, hdr_file):
    """Load AVIRIS hyperspectral data using the header file"""
    hdr_path = os.path.join(dataset_path, hdr_file)
    
    # Open the hyperspectral image
    img = spectral.open_image(hdr_path)
    
    # Get wavelength information from header if available
    wavelengths = None
    try:
        if hasattr(img, 'metadata') and 'wavelength' in img.metadata:
            wavelengths = np.array(img.metadata['wavelength'], dtype=float)
    except Exception as e:
        print(f"Warning: Couldn't extract wavelength data: {e}")
    
    return img, wavelengths

def select_random_pixels(img_shape):
    """Select random pixels for spectrum analysis"""
    rows, cols, _ = img_shape
    
    # Calculate margins (5% of dimensions) to avoid edge artifacts
    margin_r = int(rows * 0.05)
    margin_c = int(cols * 0.05)
    
    # Ensure valid margins
    margin_r = max(1, min(margin_r, rows // 10))
    margin_c = max(1, min(margin_c, cols // 10))
    
    # Generate random positions
    random_positions = []
    for i in range(NUM_RANDOM_PIXELS):
        r = np.random.randint(margin_r, rows - margin_r)
        c = np.random.randint(margin_c, cols - margin_c)
        random_positions.append((r, c, f"pixel_{i+1}"))
    
    return random_positions

def plot_wavelength_histograms(img, wavelengths, output_folder, dataset_name):
    """Create histograms for each wavelength band"""
    print(f"Generating histograms for each wavelength band for {dataset_name}...")
    
    rows, cols, bands = img.shape
    
    # Create a summary figure with selected histograms
    num_summary_plots = min(bands, 9)
    summary_indices = np.linspace(0, bands-1, num_summary_plots, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # For storing overall statistics
    band_stats = []
    
    # Process each band
    for band_idx in tqdm(range(bands)):
        # Read the band data
        band_data = img.read_band(band_idx)
        
        # Flatten for histogram
        band_data_flat = band_data.flatten()
        
        # Remove any NaN or Inf values
        valid_data = band_data_flat[np.isfinite(band_data_flat)]
        
        if len(valid_data) == 0:
            print(f"Warning: Band {band_idx} contains only non-finite values, skipping...")
            continue
        
        # Calculate statistics
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        mean_val = np.mean(valid_data)
        median_val = np.median(valid_data)
        std_val = np.std(valid_data)
        
        # Store statistics
        band_stats.append({
            'Band': band_idx,
            'Wavelength_nm': wavelengths[band_idx] if wavelengths is not None and band_idx < len(wavelengths) else None,
            'Min': min_val,
            'Max': max_val,
            'Mean': mean_val,
            'Median': median_val,
            'Std_Dev': std_val
        })
        
        # Create individual histogram for each band
        plt.figure(figsize=(10, 6))
        
        # Create histogram with KDE
        sns.histplot(valid_data, bins=100, kde=True)
        
        if wavelengths is not None and band_idx < len(wavelengths):
            plt.title(f'Value Distribution - Band {band_idx+1} ({wavelengths[band_idx]:.2f} nm)')
            wavelength_str = f"{wavelengths[band_idx]:.2f}".replace('.', 'p')
            filename = f"histogram_band_{band_idx+1:03d}_{wavelength_str}nm.png"
        else:
            plt.title(f'Value Distribution - Band {band_idx+1}')
            filename = f"histogram_band_{band_idx+1:03d}.png"
            
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistical information to the plot
        plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.4f}')
        plt.axvline(median_val, color='g', linestyle='-.', label=f'Median: {median_val:.4f}')
        plt.text(0.02, 0.95, f'Min: {min_val:.4f}\nMax: {max_val:.4f}\nStd: {std_val:.4f}',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "histograms", filename), dpi=300)
        plt.close()
        
        # Add to summary plot if in the selected indices
        if band_idx in summary_indices:
            summary_idx = np.where(summary_indices == band_idx)[0][0]
            if summary_idx < len(axes):
                sns.histplot(valid_data, bins=50, kde=True, ax=axes[summary_idx])
                
                if wavelengths is not None and band_idx < len(wavelengths):
                    axes[summary_idx].set_title(f'Band {band_idx+1} ({wavelengths[band_idx]:.2f} nm)')
                else:
                    axes[summary_idx].set_title(f'Band {band_idx+1}')
                    
                axes[summary_idx].axvline(mean_val, color='r', linestyle='--')
                axes[summary_idx].axvline(median_val, color='g', linestyle='-.')
                axes[summary_idx].grid(True, alpha=0.3)
    
    # Complete the summary figure
    plt.suptitle(f'Value Distribution Summary - {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(output_folder, "histograms", "summary_histograms.png"), dpi=300)
    plt.close()
    
    # Save band statistics
    stats_df = pd.DataFrame(band_stats)
    stats_df.to_csv(os.path.join(output_folder, "statistics", "band_statistics.csv"), index=False)
    
    return stats_df

def save_wavelength_frames(img, wavelengths, output_folder, dataset_name):
    """Save visualization frames for each wavelength band"""
    print(f"Saving wavelength frames for {dataset_name}...")
    
    rows, cols, bands = img.shape
    
    # Create a summary figure with selected frames
    num_summary_frames = min(bands, 9)
    summary_indices = np.linspace(0, bands-1, num_summary_frames, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # Process each band
    for band_idx in tqdm(range(bands)):
        # Read the band data
        band_data = img.read_band(band_idx)
        
        # Create individual frame visualization
        plt.figure(figsize=(10, 8))
        
        # Determine scale based on data
        vmin = np.nanmin(band_data)
        vmax = np.nanmax(band_data)
        
        # Display frame
        im = plt.imshow(band_data, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label='Pixel Value')
        
        if wavelengths is not None and band_idx < len(wavelengths):
            plt.title(f'Wavelength Frame - Band {band_idx+1} ({wavelengths[band_idx]:.2f} nm)')
            wavelength_str = f"{wavelengths[band_idx]:.2f}".replace('.', 'p')
            filename = f"frame_band_{band_idx+1:03d}_{wavelength_str}nm.png"
        else:
            plt.title(f'Wavelength Frame - Band {band_idx+1}')
            filename = f"frame_band_{band_idx+1:03d}.png"
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "wavelength_frames", filename), dpi=300)
        plt.close()
        
        # Add to summary plot if in the selected indices
        if band_idx in summary_indices:
            summary_idx = np.where(summary_indices == band_idx)[0][0]
            if summary_idx < len(axes):
                im = axes[summary_idx].imshow(band_data, cmap='viridis', vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=axes[summary_idx], fraction=0.046, pad=0.04)
                
                if wavelengths is not None and band_idx < len(wavelengths):
                    axes[summary_idx].set_title(f'Band {band_idx+1} ({wavelengths[band_idx]:.2f} nm)')
                else:
                    axes[summary_idx].set_title(f'Band {band_idx+1}')
    
    # Complete the summary figure
    plt.suptitle(f'Wavelength Frame Summary - {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(output_folder, "wavelength_frames", "summary_frames.png"), dpi=300)
    plt.close()

def plot_pixel_spectra(img, wavelengths, pixels, output_folder, dataset_name):
    """Plot spectra for randomly selected pixels"""
    print(f"Plotting spectra for {len(pixels)} random pixels in {dataset_name}...")
    
    rows, cols, bands = img.shape
    
    # Create a figure for all spectra
    plt.figure(figsize=(14, 8))
    
    # Create a colormap for the plots
    pixel_colors = plt.cm.tab10(np.linspace(0, 1, len(pixels)))
    
    # Store pixel spectra for CSV export
    pixel_spectra = {}
    
    for i, (row, col, pixel_name) in enumerate(pixels):
        try:
            # Read pixel spectrum
            spectrum = img.read_pixel(row, col)
            
            # Ensure we have a 1D array
            if isinstance(spectrum, np.ndarray) and spectrum.ndim > 1:
                spectrum = spectrum.flatten()
                
            # Store for CSV export
            pixel_spectra[pixel_name] = spectrum
            
            # Create x-axis for plotting
            if wavelengths is not None:
                x = wavelengths
                plt.xlabel('Wavelength (nm)')
            else:
                x = np.arange(len(spectrum))
                plt.xlabel('Band Index')
                
            # Plot spectrum
            plt.plot(x, spectrum, label=f"{pixel_name} ({row}, {col})", color=pixel_colors[i % len(pixel_colors)])
            
            # Create individual spectrum plot
            plt.figure(figsize=(10, 6))
            plt.plot(x, spectrum, color=pixel_colors[i % len(pixel_colors)])
            
            # Add title and labels
            if wavelengths is not None:
                plt.xlabel('Wavelength (nm)')
            else:
                plt.xlabel('Band Index')
                
            plt.ylabel('Pixel Value')
            plt.title(f'Spectrum at Position ({row}, {col}) - {dataset_name}')
            plt.grid(True, alpha=0.3)
            
            # Save individual spectrum plot
            plt.savefig(os.path.join(output_folder, "spectra", f"spectrum_{pixel_name}.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error plotting spectrum for pixel {pixel_name} at ({row}, {col}): {e}")
    
    # Complete and save the combined spectra plot
    plt.figure(1)  # Return to the first figure
    plt.ylabel('Pixel Value')
    plt.title(f'Spectra for Random Pixels - {dataset_name}')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "spectra", "all_spectra.png"), dpi=300)
    plt.close()
    
    # Save pixel spectra data to CSV
    if pixel_spectra:
        if wavelengths is not None:
            df = pd.DataFrame({'Wavelength_nm': wavelengths})
            for pixel_name, spectrum in pixel_spectra.items():
                if len(spectrum) == len(wavelengths):
                    df[pixel_name] = spectrum
        else:
            df = pd.DataFrame()
            for pixel_name, spectrum in pixel_spectra.items():
                df[pixel_name] = spectrum
                
        df.to_csv(os.path.join(output_folder, "spectra", "all_spectra.csv"), index=False)

def create_overall_statistics(stats_df, wavelengths, output_folder, dataset_name):
    """Create overall statistical visualizations and reports"""
    print(f"Generating overall statistics for {dataset_name}...")
    
    # Plot statistics across wavelengths if available
    if 'Wavelength_nm' in stats_df.columns and not stats_df['Wavelength_nm'].isna().all():
        plt.figure(figsize=(15, 12))
        
        # Min/Max plot
        plt.subplot(4, 1, 1)
        plt.plot(stats_df['Wavelength_nm'], stats_df['Min'], 'b-', label='Min')
        plt.plot(stats_df['Wavelength_nm'], stats_df['Max'], 'r-', label='Max')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Value')
        plt.title('Min/Max Values Across Wavelengths')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Mean/Median plot
        plt.subplot(4, 1, 2)
        plt.plot(stats_df['Wavelength_nm'], stats_df['Mean'], 'g-', label='Mean')
        plt.plot(stats_df['Wavelength_nm'], stats_df['Median'], 'm-', label='Median')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Value')
        plt.title('Mean/Median Values Across Wavelengths')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Standard deviation plot
        plt.subplot(4, 1, 3)
        plt.plot(stats_df['Wavelength_nm'], stats_df['Std_Dev'], 'k-')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Standard Deviation')
        plt.title('Standard Deviation Across Wavelengths')
        plt.grid(True, alpha=0.3)
        
        # Data range plot
        plt.subplot(4, 1, 4)
        plt.fill_between(stats_df['Wavelength_nm'], stats_df['Min'], stats_df['Max'], 
                         alpha=0.3, color='blue', label='Data Range')
        plt.plot(stats_df['Wavelength_nm'], stats_df['Mean'], 'g-', label='Mean')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Value')
        plt.title('Data Range Across Wavelengths')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "statistics", "wavelength_statistics.png"), dpi=300)
        plt.close()
    
    # Create overall value distribution plots
    plt.figure(figsize=(12, 8))
    
    # Plot histogram of means
    plt.subplot(2, 2, 1)
    sns.histplot(stats_df['Mean'], bins=30, kde=True)
    plt.xlabel('Mean Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Mean Values')
    plt.grid(True, alpha=0.3)
    
    # Plot histogram of standard deviations
    plt.subplot(2, 2, 2)
    sns.histplot(stats_df['Std_Dev'], bins=30, kde=True)
    plt.xlabel('Standard Deviation')
    plt.ylabel('Frequency')
    plt.title('Distribution of Standard Deviations')
    plt.grid(True, alpha=0.3)
    
    # Plot histogram of min values
    plt.subplot(2, 2, 3)
    sns.histplot(stats_df['Min'], bins=30, kde=True)
    plt.xlabel('Minimum Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Minimum Values')
    plt.grid(True, alpha=0.3)
    
    # Plot histogram of max values
    plt.subplot(2, 2, 4)
    sns.histplot(stats_df['Max'], bins=30, kde=True)
    plt.xlabel('Maximum Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Maximum Values')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Overall Statistical Distributions - {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(output_folder, "statistics", "overall_statistics.png"), dpi=300)
    plt.close()
    
    # Create summary statistics report
    summary_file = os.path.join(output_folder, "statistics", "summary_statistics.txt")
    with open(summary_file, 'w') as f:
        f.write(f"AVIRIS Raw Data Analysis - {dataset_name}\n")
        f.write("="*50 + "\n\n")
        
        # Overall summary
        f.write("Overall Data Distribution:\n")
        f.write(f"  - Global minimum value: {stats_df['Min'].min():.6f}\n")
        f.write(f"  - Global maximum value: {stats_df['Max'].max():.6f}\n")
        f.write(f"  - Overall mean value: {stats_df['Mean'].mean():.6f}\n")
        f.write(f"  - Average standard deviation: {stats_df['Std_Dev'].mean():.6f}\n\n")
        
        # Value range
        value_range = stats_df['Max'].max() - stats_df['Min'].min()
        f.write(f"Data range: {value_range:.6f}\n\n")
        
        # Normalization check
        if stats_df['Min'].min() >= 0 and stats_df['Max'].max() <= 1:
            f.write("Data appears to be normalized to 0-1 range.\n\n")
        elif stats_df['Min'].min() >= 0 and stats_df['Max'].max() <= 255:
            f.write("Data appears to be in 8-bit (0-255) range.\n\n")
        elif stats_df['Min'].min() >= 0 and stats_df['Max'].max() <= 65535:
            f.write("Data appears to be in 16-bit (0-65535) range.\n\n")
        else:
            f.write(f"Data range: {stats_df['Min'].min():.6f} to {stats_df['Max'].max():.6f}\n\n")
        
        # Wavelength information
        if 'Wavelength_nm' in stats_df.columns and not stats_df['Wavelength_nm'].isna().all():
            f.write("Wavelength Information:\n")
            f.write(f"  - Wavelength range: {stats_df['Wavelength_nm'].min():.2f} to {stats_df['Wavelength_nm'].max():.2f} nm\n")
            f.write(f"  - Number of bands: {len(stats_df)}\n\n")
        
        f.write("Analysis complete.\n")

def process_dataset(dataset, output_base_dir):
    """Process a single AVIRIS dataset"""
    dataset_name = dataset['name']
    dataset_path = dataset['path']
    hdr_files = dataset['hdr_files']
    
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*80}")
    
    for hdr_file in hdr_files:
        # Extract the base name (without .hdr)
        base_name = hdr_file.replace('.hdr', '')
        
        print(f"\nProcessing file: {base_name}")
        
        try:
            # Create output directory
            output_folder = create_output_dirs(f"{dataset_name}/{base_name}")
            
            # Load the hyperspectral data
            print(f"Loading data from {os.path.join(dataset_path, hdr_file)}...")
            img, wavelengths = load_aviris_data(dataset_path, hdr_file)
            
            # Get data shape
            rows, cols, bands = img.shape
            print(f"Data loaded: {rows} x {cols} x {bands}")
            
            if wavelengths is not None:
                print(f"Wavelength data: {len(wavelengths)} bands, range {np.min(wavelengths):.2f} to {np.max(wavelengths):.2f} nm")
            else:
                print("No wavelength data available")
            
            # Select random pixels
            random_pixels = select_random_pixels(img.shape)
            
            # Generate visualizations
            stats_df = plot_wavelength_histograms(img, wavelengths, output_folder, f"{dataset_name}/{base_name}")
            save_wavelength_frames(img, wavelengths, output_folder, f"{dataset_name}/{base_name}")
            plot_pixel_spectra(img, wavelengths, random_pixels, output_folder, f"{dataset_name}/{base_name}")
            create_overall_statistics(stats_df, wavelengths, output_folder, f"{dataset_name}/{base_name}")
            
            print(f"Processing complete for {base_name}. Results saved to {output_folder}")
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to process all AVIRIS datasets"""
    print("Starting AVIRIS Raw Data Analysis...")
    
    # Create main output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of datasets
    datasets = get_aviris_datasets()
    
    if not datasets:
        print(f"No AVIRIS datasets found in {DATA_DIR}")
        return
    
    print(f"Found {len(datasets)} datasets to process:")
    for i, dataset in enumerate(datasets):
        print(f"  {i+1}. {dataset['name']} - {len(dataset['hdr_files'])} files")
    
    # Process each dataset
    for dataset in datasets:
        process_dataset(dataset, OUTPUT_DIR)
    
    print("\nAnalysis complete!")
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
