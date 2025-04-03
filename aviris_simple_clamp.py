#!/usr/bin/env python3
"""
AVIRIS Simple Clamp and Visualize
--------------------------------
This script simply:
1. Loads AVIRIS hyperspectral data
2. Clamps all values to 0-1 range
3. Saves and visualizes the processed data
"""

import numpy as np
import matplotlib.pyplot as plt
import spectral
import os
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import shutil

# Define paths using actual directories from the system
BASE_DIR = os.path.expanduser("~/ProjectsLFS/iccp_rcwa/S4/iccp_test")
DATA_DIR = os.path.join(BASE_DIR, "AVIRIS")
OUTPUT_DIR = os.path.join(BASE_DIR, "AVIRIS_SIMPLE_CLAMPED")

# Number of random pixels to select for spectrum plots
NUM_RANDOM_PIXELS = 10

def create_output_dirs(dataset_name):
    """Create output directories for visualizations and processed data"""
    folder_path = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join(folder_path, "processed_data"), exist_ok=True)
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

def clamp_data(img):
    """
    Load all image data and clamp to 0-1 range
    
    Parameters:
    - img: Hyperspectral image
    
    Returns:
    - clamped_img: 3D numpy array with values clamped to 0-1 range
    """
    rows, cols, bands = img.shape
    print(f"Loading and clamping image data ({rows}x{cols}x{bands})...")
    
    # Create output array
    clamped_img = np.zeros((rows, cols, bands), dtype=np.float32)
    
    # Process each band
    for band_idx in tqdm(range(bands), desc="Processing bands"):
        # Read band data
        band_data = img.read_band(band_idx)
        
        # Clamp to 0-1 range
        clamped_band = np.clip(band_data, 0.0, 1.0).astype(np.float32)
        
        # Store in the 3D array
        clamped_img[:, :, band_idx] = clamped_band
    
    return clamped_img

def save_processed_image(processed_img, wavelengths, output_folder, dataset_name, hdr_template=None):
    """
    Save the processed image as ENVI file with header
    
    Parameters:
    - processed_img: 3D numpy array containing the processed image
    - wavelengths: Array of wavelength values (optional)
    - output_folder: Output directory path
    - dataset_name: Name of the dataset
    - hdr_template: Original header file to use as template (optional)
    
    Returns:
    - output_file: Path to saved file
    """
    print(f"Saving processed image for {dataset_name}...")
    
    # Create output directory for processed data
    processed_dir = os.path.join(output_folder, "processed_data")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Output file paths
    output_base = os.path.join(processed_dir, f"{dataset_name}_clamped")
    output_file = f"{output_base}.img"
    output_hdr = f"{output_base}.hdr"
    
    # Save as ENVI file
    processed_img_float32 = processed_img.astype(np.float32)
    
    # Write binary data
    with open(output_file, 'wb') as f:
        processed_img_float32.tofile(f)
    
    # Create header file
    rows, cols, bands = processed_img.shape
    
    # If we have a template header, modify it
    if hdr_template:
        # Copy the original header
        shutil.copy(hdr_template, output_hdr)
        
        # Modify key parameters
        with open(output_hdr, 'r') as f:
            header_lines = f.readlines()
        
        # Update parameters
        new_header_lines = []
        for line in header_lines:
            if line.strip().startswith('data type'):
                line = 'data type = 4\n'  # 4 = float32
            elif line.strip().startswith('byte order'):
                line = 'byte order = 0\n'  # 0 = little endian
            elif line.strip().startswith('interleave'):
                line = 'interleave = bsq\n'  # band sequential
            
            # Add description if found
            if line.strip().startswith('description') and '{' in line and '}' in line:
                line = 'description = {Clamped AVIRIS data (0-1 range)}\n'
                
            new_header_lines.append(line)
        
        # Write modified header
        with open(output_hdr, 'w') as f:
            f.writelines(new_header_lines)
    else:
        # Create a basic header from scratch
        with open(output_hdr, 'w') as f:
            f.write("ENVI\n")
            f.write("description = {Clamped AVIRIS data (0-1 range)}\n")
            f.write(f"samples = {cols}\n")
            f.write(f"lines = {rows}\n")
            f.write(f"bands = {bands}\n")
            f.write("header offset = 0\n")
            f.write("file type = ENVI Standard\n")
            f.write("data type = 4\n")  # 4 = float32
            f.write("interleave = bsq\n")
            f.write("byte order = 0\n")  # 0 = little endian
            
            # Add wavelength information if available
            if wavelengths is not None:
                f.write("wavelength units = nm\n")
                f.write(f"wavelength = {{\n")
                f.write(",".join([f"{w:.6f}" for w in wavelengths]))
                f.write("\n}")
    
    # Also save wavelengths as numpy array if available
    if wavelengths is not None:
        numpy_dir = os.path.join(processed_dir, "numpy")
        os.makedirs(numpy_dir, exist_ok=True)
        np.save(os.path.join(numpy_dir, "wavelengths.npy"), wavelengths)
    
    print(f"Processed image saved to {output_file}")
    
    return output_file

def select_random_pixels(img_shape):
    """Select random pixels for spectrum analysis"""
    rows, cols, _ = img_shape
    
    # Calculate margins (5% of dimensions) to avoid edge artifacts
    margin_r = max(1, int(rows * 0.05))
    margin_c = max(1, int(cols * 0.05))
    
    # Generate random positions
    random_positions = []
    for i in range(NUM_RANDOM_PIXELS):
        r = np.random.randint(margin_r, rows - margin_r)
        c = np.random.randint(margin_c, cols - margin_c)
        random_positions.append((r, c, f"pixel_{i+1}"))
    
    return random_positions

def plot_pixel_spectra(img_data, wavelengths, pixels, output_folder, dataset_name):
    """Plot spectra for randomly selected pixels"""
    print(f"Plotting spectra for {len(pixels)} random pixels in {dataset_name}...")
    
    rows, cols, bands = img_data.shape
    
    # Create a figure for all spectra
    plt.figure(figsize=(14, 8))
    
    # Create a colormap for the plots
    pixel_colors = plt.cm.tab10(np.linspace(0, 1, len(pixels)))
    
    # Store pixel spectra for CSV export
    pixel_spectra = {}
    
    for i, (row, col, pixel_name) in enumerate(pixels):
        try:
            # Extract spectrum for this pixel
            spectrum = img_data[row, col, :]
            
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

def plot_wavelength_histograms(img_data, wavelengths, output_folder, dataset_name):
    """Create histograms for each wavelength band"""
    print(f"Generating histograms for selected wavelength bands for {dataset_name}...")
    
    rows, cols, bands = img_data.shape
    
    # Create a summary figure with selected histograms (9 bands evenly distributed)
    num_summary_plots = min(bands, 9)
    summary_indices = np.linspace(0, bands-1, num_summary_plots, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # For storing overall statistics
    band_stats = []
    
    # Calculate statistics for all bands first
    for band_idx in tqdm(range(bands), desc="Analyzing bands"):
        # Extract this band
        band_data = img_data[:, :, band_idx].flatten()
        
        # Calculate statistics
        min_val = np.min(band_data)
        max_val = np.max(band_data)
        mean_val = np.mean(band_data)
        median_val = np.median(band_data)
        std_val = np.std(band_data)
        
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
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(band_stats)
    
    # Save band statistics
    stats_df.to_csv(os.path.join(output_folder, "statistics", "band_statistics.csv"), index=False)
    
    # Plot histograms for summary bands
    for i, band_idx in enumerate(summary_indices):
        if i >= len(axes):
            break
            
        # Extract this band
        band_data = img_data[:, :, band_idx].flatten()
        
        # Get statistics from DataFrame
        stats = stats_df[stats_df['Band'] == band_idx].iloc[0]
        
        # Create histogram
        sns.histplot(band_data, bins=50, kde=True, ax=axes[i])
        
        # Add title
        if wavelengths is not None and band_idx < len(wavelengths):
            title = f'Band {band_idx+1} ({wavelengths[band_idx]:.2f} nm)'
            axes[i].set_title(title)
        else:
            title = f'Band {band_idx+1}'
            axes[i].set_title(title)
        
        # Add lines for mean and median
        axes[i].axvline(stats['Mean'], color='r', linestyle='--')
        axes[i].axvline(stats['Median'], color='g', linestyle='-.')
        axes[i].grid(True, alpha=0.3)
    
    # Complete the summary figure
    plt.suptitle(f'Value Distribution Summary - {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(output_folder, "histograms", "summary_histograms.png"), dpi=300)
    plt.close()
    
    # Plot histograms for key bands 
    key_bands = list(summary_indices)
    key_bands.sort()
    
    # Plot histograms for key bands
    for band_idx in key_bands:
        # Extract this band
        band_data = img_data[:, :, band_idx].flatten()
        
        # Get statistics from DataFrame
        stats = stats_df[stats_df['Band'] == band_idx].iloc[0]
        
        # Create individual histogram
        plt.figure(figsize=(10, 6))
        
        # Create histogram with KDE
        sns.histplot(band_data, bins=100, kde=True)
        
        # Add title
        if wavelengths is not None and band_idx < len(wavelengths):
            title = f'Value Distribution - Band {band_idx+1} ({wavelengths[band_idx]:.2f} nm)'
            plt.title(title)
            wavelength_str = f"{wavelengths[band_idx]:.2f}".replace('.', 'p')
            filename = f"histogram_band_{band_idx+1:03d}_{wavelength_str}nm.png"
        else:
            title = f'Value Distribution - Band {band_idx+1}'
            plt.title(title)
            filename = f"histogram_band_{band_idx+1:03d}.png"
        
        # Add labels and grid
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistical information to the plot
        plt.axvline(stats['Mean'], color='r', linestyle='--', label=f'Mean: {stats["Mean"]:.4f}')
        plt.axvline(stats['Median'], color='g', linestyle='-.', label=f'Median: {stats["Median"]:.4f}')
        plt.text(0.02, 0.95, f'Min: {stats["Min"]:.4f}\nMax: {stats["Max"]:.4f}\nStd: {stats["Std_Dev"]:.4f}',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        plt.legend()
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "histograms", filename), dpi=300)
        plt.close()
    
    return stats_df

def save_wavelength_frames(img_data, wavelengths, output_folder, dataset_name):
    """Save visualization frames for each wavelength band"""
    print(f"Saving wavelength frames for {dataset_name}...")
    
    rows, cols, bands = img_data.shape
    
    # Create a summary figure with selected frames
    num_summary_frames = min(bands, 9)
    summary_indices = np.linspace(0, bands-1, num_summary_frames, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # Process summary frames
    for i, band_idx in enumerate(summary_indices):
        if i >= len(axes):
            break
            
        # Extract band data
        band_data = img_data[:, :, band_idx]
        
        # Display frame in summary plot
        im = axes[i].imshow(band_data, cmap='viridis', vmin=0, vmax=1)
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Add title
        if wavelengths is not None and band_idx < len(wavelengths):
            title = f'Band {band_idx+1} ({wavelengths[band_idx]:.2f} nm)'
            axes[i].set_title(title)
        else:
            title = f'Band {band_idx+1}'
            axes[i].set_title(title)
    
    # Complete the summary figure
    plt.suptitle(f'Wavelength Frame Summary - {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(output_folder, "wavelength_frames", "summary_frames.png"), dpi=300)
    plt.close()
    
    # Save individual frames for key bands
    print("Saving individual frames for key bands...")
    key_bands = list(summary_indices)
    key_bands.sort()
    
    # Save frames for key bands
    for band_idx in key_bands:
        # Extract band data
        band_data = img_data[:, :, band_idx]
        
        # Create individual frame visualization
        plt.figure(figsize=(10, 8))
        
        # Display frame
        im = plt.imshow(band_data, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im, label='Pixel Value')
        
        # Add title
        if wavelengths is not None and band_idx < len(wavelengths):
            title = f'Wavelength Frame - Band {band_idx+1} ({wavelengths[band_idx]:.2f} nm)'
            plt.title(title)
            wavelength_str = f"{wavelengths[band_idx]:.2f}".replace('.', 'p')
            filename = f"frame_band_{band_idx+1:03d}_{wavelength_str}nm.png"
        else:
            title = f'Wavelength Frame - Band {band_idx+1}'
            plt.title(title)
            filename = f"frame_band_{band_idx+1:03d}.png"
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "wavelength_frames", filename), dpi=300)
        plt.close()

def create_overall_statistics(stats_df, wavelengths, output_folder, dataset_name):
    """Create overall statistical visualizations and reports"""
    print(f"Generating overall statistics for {dataset_name}...")
    
    # Plot statistics across wavelengths if available
    if 'Wavelength_nm' in stats_df.columns and not stats_df['Wavelength_nm'].isna().all():
        plt.figure(figsize=(15, 12))
        
        # Create x-axis values
        x = stats_df['Wavelength_nm']
        
        # Min/Max plot
        plt.subplot(4, 1, 1)
        plt.plot(x, stats_df['Min'], 'b-', label='Min')
        plt.plot(x, stats_df['Max'], 'r-', label='Max')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Value')
        plt.title('Min/Max Values Across Wavelengths')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Mean/Median plot
        plt.subplot(4, 1, 2)
        plt.plot(x, stats_df['Mean'], 'g-', label='Mean')
        plt.plot(x, stats_df['Median'], 'm-', label='Median')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Value')
        plt.title('Mean/Median Values Across Wavelengths')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Standard deviation plot
        plt.subplot(4, 1, 3)
        plt.plot(x, stats_df['Std_Dev'], 'k-')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Standard Deviation')
        plt.title('Standard Deviation Across Wavelengths')
        plt.grid(True, alpha=0.3)
        
        # Data range plot
        plt.subplot(4, 1, 4)
        plt.fill_between(x, stats_df['Min'], stats_df['Max'], 
                         alpha=0.3, color='blue', label='Data Range')
        plt.plot(x, stats_df['Mean'], 'g-', label='Mean')
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
        f.write(f"AVIRIS Simple Clamped Data Analysis - {dataset_name}\n")
        f.write("="*50 + "\n\n")
        
        # Processing steps
        f.write("Processing Steps Applied:\n")
        f.write("  1. Clamped values to 0-1 range\n\n")
        
        # Overall summary
        f.write("Overall Data Distribution:\n")
        f.write(f"  - Global minimum value: {stats_df['Min'].min():.6f}\n")
        f.write(f"  - Global maximum value: {stats_df['Max'].max():.6f}\n")
        f.write(f"  - Overall mean value: {stats_df['Mean'].mean():.6f}\n")
        f.write(f"  - Average standard deviation: {stats_df['Std_Dev'].mean():.6f}\n\n")
        
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
            
            # Clamp data to 0-1 range
            clamped_img = clamp_data(img)
            
            # Save processed image
            save_processed_image(clamped_img, wavelengths, output_folder, f"{dataset_name}_{base_name}", 
                               hdr_template=os.path.join(dataset_path, hdr_file))
            
            # Select random pixels for visualization
            random_pixels = select_random_pixels(clamped_img.shape)
            
            # Generate visualizations
            plot_pixel_spectra(clamped_img, wavelengths, random_pixels, output_folder, f"{dataset_name}/{base_name}")
            stats_df = plot_wavelength_histograms(clamped_img, wavelengths, output_folder, f"{dataset_name}/{base_name}")
            save_wavelength_frames(clamped_img, wavelengths, output_folder, f"{dataset_name}/{base_name}")
            create_overall_statistics(stats_df, wavelengths, output_folder, f"{dataset_name}/{base_name}")
            
            print(f"Processing complete for {base_name}. Results saved to {output_folder}")
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to process all AVIRIS datasets"""
    print("Starting AVIRIS Simple Clamp and Visualize...")
    
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
    
    print("\nProcessing complete!")
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
