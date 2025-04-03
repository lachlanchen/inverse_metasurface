#!/usr/bin/env python3
"""
AVIRIS Simple Wavelength Selector and Normalizer
------------------------------------------------
A simplified script to select wavelengths from AVIRIS data based on a CSV file
and clamp values to 0-1 range.

Features:
- Selects bands closest to wavelengths in CSV file
- Clamps all values to 0-1 range
- Saves results as PyTorch tensors
- Creates basic wavelength mapping information
"""

import numpy as np
import matplotlib.pyplot as plt
import spectral
import os
import pandas as pd
import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def select_and_clamp_data(aviris_folder, subfolder, csv_file, output_base):
    """
    Process AVIRIS data by selecting wavelengths that match the CSV file
    and clamping values to 0-1 range.
    
    Parameters:
    -----------
    aviris_folder : str
        Base AVIRIS directory
    subfolder : str
        Name of the subfolder to process
    csv_file : str
        Path to CSV file with wavelength data
    output_base : str
        Base output directory
        
    Returns:
    --------
    bool
        True if processing was successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Processing {subfolder}")
    print(f"{'='*60}")
    
    # Define paths
    folder_path = os.path.join(aviris_folder, subfolder)
    
    # Create output directory structure
    output_folder = os.path.join(output_base, subfolder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Create subdirectories
    torch_folder = os.path.join(output_folder, 'torch')
    images_folder = os.path.join(output_folder, 'images')
    os.makedirs(torch_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)
    
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
    
    # Read the wavelengths from the CSV file
    print(f"Reading wavelengths from CSV: {csv_file}")
    try:
        csv_data = pd.read_csv(csv_file)
        csv_wavelengths = csv_data['Wavelength_um'].to_numpy() * 1000  # Convert μm to nm
        print(f"Found {len(csv_wavelengths)} wavelengths in CSV file")
        print(f"CSV wavelength range: {np.min(csv_wavelengths):.2f} to {np.max(csv_wavelengths):.2f} nm")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False
    
    # Open the AVIRIS image
    print(f"Opening AVIRIS image from: {hdr_file}")
    try:
        img = spectral.open_image(hdr_file)
        print(f"Image dimensions: {img.shape}")
        print(f"Number of bands: {img.nbands}")
    except Exception as e:
        print(f"Error opening image: {e}")
        return False
    
    # Check if wavelength information is available
    if hasattr(img, 'bands') and hasattr(img.bands, 'centers'):
        aviris_wavelengths = np.array(img.bands.centers)
        print(f"AVIRIS wavelength range: {np.min(aviris_wavelengths):.2f} to {np.max(aviris_wavelengths):.2f} nm")
        
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
        print("First 10 wavelength mappings:")
        print(mapping_df.head(10).to_string(index=False))
        
        # Save the wavelength mapping
        mapping_df.to_csv(f'{output_folder}/wavelength_mapping.csv', index=False)
        
        # Extract the selected bands
        print(f"\nExtracting selected bands...")
        
        # Prepare a new data array for selected bands
        rows, cols, _ = img.shape
        new_data = np.zeros((rows, cols, len(selected_bands)), dtype=np.float32)
        
        # Create a list to store band statistics
        band_stats = []
        
        # Extract the selected bands
        for i, band_idx in enumerate(tqdm(selected_bands, desc="Extracting bands")):
            band_data = img.read_band(band_idx)
            
            # Get statistics for this band (mask nodata values)
            nodata_value = -9999.0
            masked_data = np.ma.masked_where(band_data == nodata_value, band_data)
            
            min_val = np.ma.min(masked_data)
            max_val = np.ma.max(masked_data)
            mean_val = np.ma.mean(masked_data)
            
            # Store the stats
            band_stats.append({
                'Position': i,
                'AVIRIS_Band': band_idx,
                'Wavelength_nm': aviris_wavelengths[band_idx],
                'Min_Value': float(min_val),
                'Max_Value': float(max_val), 
                'Mean_Value': float(mean_val)
            })
            
            # Store the band data in our new array
            new_data[:,:,i] = band_data
        
        # Replace nodata values with band mean values
        print("\nReplacing nodata values with band means...")
        for i in range(len(selected_bands)):
            # Get nodata mask for this band
            nodata_mask = new_data[:,:,i] == -9999.0
            
            # If there are any nodata values
            if np.any(nodata_mask):
                # Get the mean value for this band (excluding nodata)
                valid_mask = ~nodata_mask
                
                if np.any(valid_mask):
                    # If we have valid data, use its mean
                    band_mean = np.mean(new_data[:,:,i][valid_mask])
                else:
                    # If entire band is nodata, use 0
                    band_mean = 0.0
                
                # Replace nodata values with mean
                new_data[:,:,i][nodata_mask] = band_mean
        
        # Clamp values to 0-1 range
        print("\nClamping data to 0-1 range...")
        for i in tqdm(range(len(selected_bands)), desc="Clamping bands"):
            band_data = new_data[:,:,i]
            min_val = np.min(band_data)
            max_val = np.max(band_data)
            
            # Avoid division by zero
            if max_val > min_val:
                # Normalize to 0-1 range
                new_data[:,:,i] = (band_data - min_val) / (max_val - min_val)
            else:
                # If min == max, set all values to 0.5
                new_data[:,:,i] = 0.5
        
        # Save data as PyTorch tensor
        print(f"Saving data as PyTorch tensor: {torch_folder}/aviris_selected.pt")
        torch_data = torch.from_numpy(new_data)
        torch.save(torch_data, f"{torch_folder}/aviris_selected.pt")
        
        # Save wavelength information with the PyTorch data
        selected_wavelengths = np.array([aviris_wavelengths[idx] for idx in selected_bands])
        torch_wavelengths = torch.from_numpy(selected_wavelengths)
        torch.save(torch_wavelengths, f"{torch_folder}/wavelengths.pt")
        
        # Save band statistics
        stats_df = pd.DataFrame(band_stats)
        stats_df.to_csv(f'{output_folder}/band_statistics.csv', index=False)
        
        # Create a visualization of selected bands
        print("\nCreating visualization of selected bands...")
        visualize_bands(new_data, selected_wavelengths, images_folder)
        
        # Save metadata
        metadata = {
            'shape': tuple(int(x) for x in new_data.shape),
            'wavelengths_nm': [float(x) for x in selected_wavelengths.tolist()],
            'wavelengths_um': [float(x) for x in (selected_wavelengths / 1000).tolist()],
            'min_value': float(0.0),  # After clamping
            'max_value': float(1.0),  # After clamping
            'original_bands': [int(x) for x in selected_bands],
            'processing_steps': [
                'Select bands closest to CSV wavelengths',
                'Replace nodata values with band means',
                'Clamp each band to 0-1 range'
            ]
        }
        
        # Save metadata as JSON
        with open(f"{torch_folder}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\nProcessing complete!")
        return True
    else:
        print("No wavelength information found in the image header.")
        return False

def visualize_bands(data, wavelengths, output_folder):
    """
    Create visualizations of selected bands
    
    Parameters:
    -----------
    data : numpy.ndarray
        The selected and clamped band data
    wavelengths : numpy.ndarray
        Array of wavelength values in nm
    output_folder : str
        Folder to save visualizations
    """
    # Create a visualization of a few bands
    frames_to_plot = [0, len(wavelengths)//3, 2*len(wavelengths)//3, len(wavelengths)-1]
    frames_to_plot = [idx for idx in frames_to_plot if idx < len(wavelengths)]
    
    # Create a figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, frame_idx in enumerate(frames_to_plot):
        if i < len(axes):
            # Get the band data
            band_data = data[:,:,frame_idx]
            
            # Plot on the corresponding subplot
            im = axes[i].imshow(band_data, cmap='viridis', vmin=0, vmax=1)
            axes[i].set_title(f'Band {frame_idx+1}: {wavelengths[frame_idx]:.2f} nm ({wavelengths[frame_idx]/1000:.2f} μm)')
            plt.colorbar(im, ax=axes[i])
    
    # Hide any unused subplots
    for i in range(len(frames_to_plot), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/selected_bands.png', dpi=300)
    plt.close()
    
    # Save individual band images
    for i in tqdm(range(len(wavelengths)), desc="Saving band images"):
        band_data = data[:,:,i]
        
        plt.figure(figsize=(8, 8))
        plt.imshow(band_data, cmap='viridis', vmin=0, vmax=1)
        plt.title(f'Band {i+1}: {wavelengths[i]:.2f} nm ({wavelengths[i]/1000:.2f} μm)')
        plt.colorbar(label='Normalized Reflectance')
        plt.tight_layout()
        
        # Save the figure
        wavelength_str = f"{wavelengths[i]:.2f}".replace('.', 'p')
        plt.savefig(f'{output_folder}/band_{i+1:03d}_{wavelength_str}nm.png', dpi=150)
        plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process AVIRIS data with simple wavelength selection and clamping')
    parser.add_argument('--base', default='AVIRIS', help='Base AVIRIS directory')
    parser.add_argument('--output', default='AVIRIS_SIMPLE_SELECT', help='Output directory')
    parser.add_argument('--csv', default='partial_crys_data/partial_crys_C0.0.csv', help='CSV file with wavelength data')
    parser.add_argument('--subfolder', help='Process only a specific subfolder')
    args = parser.parse_args()
    
    # Define paths
    aviris_folder = args.base
    output_base = args.output
    csv_file = args.csv
    
    # Create output directory if it doesn't exist
    os.makedirs(output_base, exist_ok=True)
    
    # Get list of subfolders in AVIRIS directory
    if args.subfolder:
        subfolders = [args.subfolder]
        if not os.path.isdir(os.path.join(aviris_folder, args.subfolder)):
            print(f"Error: Subfolder {args.subfolder} not found in {aviris_folder}")
            return
    else:
        subfolders = [f for f in os.listdir(aviris_folder) 
                      if os.path.isdir(os.path.join(aviris_folder, f))]
    
    if not subfolders:
        print(f"No subfolders found in {aviris_folder}")
        return
    
    print(f"Found {len(subfolders)} subfolders to process: {', '.join(subfolders)}")
    
    # Process each subfolder
    results = {}
    for subfolder in subfolders:
        try:
            success = select_and_clamp_data(aviris_folder, subfolder, csv_file, output_base)
            results[subfolder] = 'Success' if success else 'Failed'
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[subfolder] = f"Error: {str(e)}"
    
    # Print summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    for subfolder, result in results.items():
        print(f"{subfolder}: {result}")

if __name__ == "__main__":
    main()
