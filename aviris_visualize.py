#!/usr/bin/env python3
"""
AVIRIS Data Visualization Tool
------------------------------
Visualizes processed AVIRIS data tiles:
- Shows random tiles at different wavelengths
- Displays spectral profiles from random pixels
- Creates RGB composite images from selected bands
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
from pathlib import Path
import random
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

def load_data(pt_file, metadata_file):
    """
    Load processed data and metadata
    
    Parameters:
    pt_file: Path to PT file containing tiles
    metadata_file: Path to JSON metadata file
    
    Returns:
    tuple: (tiles_data, metadata_dict)
    """
    print(f"Loading tiles from: {pt_file}")
    tiles_data = torch.load(pt_file)
    print(f"Loaded tensor with shape: {tiles_data.shape}")
    
    print(f"Loading metadata from: {metadata_file}")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Dataset contains {tiles_data.shape[0]} tiles, each {metadata['tile_size']}x{metadata['tile_size']} pixels")
    print(f"Each tile has {tiles_data.shape[1]} spectral bands")
    
    return tiles_data, metadata

def visualize_random_tiles(tiles_data, metadata, output_dir, num_tiles=5):
    """
    Visualize random tiles at different wavelengths
    
    Parameters:
    tiles_data: Tensor of shape (N, C, H, W) containing tiles
    metadata: Dictionary with metadata including wavelengths
    output_dir: Directory to save visualizations
    num_tiles: Number of random tiles to visualize
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get wavelength information
    wavelengths_nm = metadata.get('wavelengths_nm', [])
    if not wavelengths_nm and 'wavelengths_um' in metadata:
        # Convert from um to nm if needed
        wavelengths_nm = [w * 1000 for w in metadata['wavelengths_um']]
    
    # Number of tiles
    num_total_tiles = tiles_data.shape[0]
    num_bands = tiles_data.shape[1]
    
    # Randomly select tiles to visualize
    if num_total_tiles < num_tiles:
        num_tiles = num_total_tiles
        print(f"Warning: Only {num_total_tiles} tiles available. Visualizing all.")
    
    tile_indices = random.sample(range(num_total_tiles), num_tiles)
    
    # Select 3 bands to visualize (beginning, middle, end)
    band_indices = [0, num_bands // 2, num_bands - 1]
    
    # Visualize each selected tile
    for i, tile_idx in enumerate(tile_indices):
        # Get the tile data
        tile = tiles_data[tile_idx]
        
        # Create a figure with 3 subplots (one for each selected band)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for j, band_idx in enumerate(band_indices):
            # Get the band data
            band_data = tile[band_idx].numpy()
            
            # Get wavelength information
            wavelength_str = f"{wavelengths_nm[band_idx]:.2f} nm" if wavelengths_nm else f"Band {band_idx+1}"
            
            # Plot the band
            im = axes[j].imshow(band_data, cmap='viridis', vmin=0, vmax=1)
            axes[j].set_title(f"Tile {tile_idx+1}, {wavelength_str}")
            plt.colorbar(im, ax=axes[j])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"tile_{tile_idx+1}_bands.png"), dpi=200)
        plt.close()
        
        # Now create an RGB composite using three bands
        # Try to select bands in the red, green, and blue parts of the spectrum
        if num_bands >= 3:
            # Define approximate wavelength ranges for RGB (if available)
            if wavelengths_nm:
                # Rough approximations of RGB wavelength ranges
                red_idx = find_nearest_band(wavelengths_nm, 650)    # Red: ~650nm
                green_idx = find_nearest_band(wavelengths_nm, 550)  # Green: ~550nm
                blue_idx = find_nearest_band(wavelengths_nm, 450)   # Blue: ~450nm
            else:
                # If no wavelength info, use evenly spaced bands
                red_idx = int(num_bands * 0.8)
                green_idx = int(num_bands * 0.5)
                blue_idx = int(num_bands * 0.2)
            
            # Create RGB composite
            rgb = np.zeros((tile.shape[1], tile.shape[2], 3))
            rgb[:,:,0] = tile[red_idx].numpy()
            rgb[:,:,1] = tile[green_idx].numpy()
            rgb[:,:,2] = tile[blue_idx].numpy()
            
            # Clip values to [0, 1] range
            rgb = np.clip(rgb, 0, 1)
            
            # Plot RGB composite
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb)
            if wavelengths_nm:
                plt.title(f"Tile {tile_idx+1} RGB Composite\nR: {wavelengths_nm[red_idx]:.2f}nm, G: {wavelengths_nm[green_idx]:.2f}nm, B: {wavelengths_nm[blue_idx]:.2f}nm")
            else:
                plt.title(f"Tile {tile_idx+1} RGB Composite\nR: Band {red_idx+1}, G: Band {green_idx+1}, B: Band {blue_idx+1}")
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f"tile_{tile_idx+1}_rgb.png"), dpi=200)
            plt.close()
    
    print(f"Saved tile visualizations to: {output_dir}")

def visualize_spectral_profiles(tiles_data, metadata, output_dir, num_tiles=3, num_pixels=5):
    """
    Visualize spectral profiles from random pixels in random tiles
    
    Parameters:
    tiles_data: Tensor of shape (N, C, H, W) containing tiles
    metadata: Dictionary with metadata including wavelengths
    output_dir: Directory to save visualizations
    num_tiles: Number of random tiles to visualize
    num_pixels: Number of random pixels to profile per tile
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get wavelength information
    wavelengths = None
    if 'wavelengths_nm' in metadata and metadata['wavelengths_nm']:
        wavelengths = np.array(metadata['wavelengths_nm'])
    elif 'wavelengths_um' in metadata and metadata['wavelengths_um']:
        wavelengths = np.array(metadata['wavelengths_um']) * 1000  # Convert to nm
    
    # Number of tiles and bands
    num_total_tiles = tiles_data.shape[0]
    num_bands = tiles_data.shape[1]
    tile_size = tiles_data.shape[2]  # Assuming square tiles
    
    # Randomly select tiles
    if num_total_tiles < num_tiles:
        num_tiles = num_total_tiles
    
    tile_indices = random.sample(range(num_total_tiles), num_tiles)
    
    # Visualize spectral profiles for each selected tile
    for i, tile_idx in enumerate(tile_indices):
        # Get the tile
        tile = tiles_data[tile_idx]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Show the tile (using a middle band)
        middle_band = num_bands // 2
        ax1.imshow(tile[middle_band], cmap='viridis')
        
        # Randomly select pixels
        pixel_coords = []
        for _ in range(num_pixels):
            y = random.randint(0, tile_size - 1)
            x = random.randint(0, tile_size - 1)
            pixel_coords.append((y, x))
        
        # Plot points on the image
        colors = plt.cm.tab10(np.linspace(0, 1, num_pixels))
        for j, (y, x) in enumerate(pixel_coords):
            ax1.plot(x, y, 'o', markersize=8, color=colors[j])
            ax1.text(x+5, y+5, f"P{j+1}", color=colors[j], fontweight='bold')
        
        ax1.set_title(f"Tile {tile_idx+1}")
        
        # Plot spectral profiles
        for j, (y, x) in enumerate(pixel_coords):
            # Extract spectral profile
            profile = tile[:, y, x].numpy()
            
            # X-axis: wavelengths or band indices
            if wavelengths is not None:
                ax2.plot(wavelengths, profile, '-', color=colors[j], linewidth=2, label=f"P{j+1} ({y},{x})")
                ax2.set_xlabel("Wavelength (nm)")
            else:
                ax2.plot(range(num_bands), profile, '-', color=colors[j], linewidth=2, label=f"P{j+1} ({y},{x})")
                ax2.set_xlabel("Band Index")
        
        ax2.set_ylabel("Reflectance")
        ax2.set_title("Spectral Profiles")
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"tile_{tile_idx+1}_spectral_profiles.png"), dpi=200)
        plt.close()
    
    print(f"Saved spectral profile visualizations to: {output_dir}")

def visualize_wavelength_coverage(metadata, output_dir):
    """
    Visualize coverage of selected wavelengths
    
    Parameters:
    metadata: Dictionary with metadata including wavelengths
    output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get wavelength information
    wavelengths_nm = None
    if 'wavelengths_nm' in metadata and metadata['wavelengths_nm']:
        wavelengths_nm = np.array(metadata['wavelengths_nm'])
    elif 'wavelengths_um' in metadata and metadata['wavelengths_um']:
        wavelengths_nm = np.array(metadata['wavelengths_um']) * 1000  # Convert to nm
    
    if wavelengths_nm is None or len(wavelengths_nm) == 0:
        print("No wavelength information available in metadata")
        return
    
    # Plot wavelength distribution
    plt.figure(figsize=(12, 6))
    
    # Create histogram of wavelengths
    plt.hist(wavelengths_nm, bins=20, alpha=0.7, color='blue')
    
    # Mark important wavelength regions
    regions = [
        (400, 450, 'blue', 'Blue'),
        (450, 500, 'royalblue', 'Blue-Green'),
        (500, 550, 'green', 'Green'),
        (550, 600, 'yellowgreen', 'Yellow-Green'),
        (600, 650, 'orange', 'Orange'),
        (650, 700, 'red', 'Red'),
        (700, 1000, 'darkred', 'NIR'),
        (1000, 2500, 'darkmagenta', 'SWIR')
    ]
    
    ymax = plt.gca().get_ylim()[1]
    
    for start, end, color, label in regions:
        # Only draw if we have wavelengths in this range
        if np.any((wavelengths_nm >= start) & (wavelengths_nm <= end)):
            plt.axvspan(start, end, alpha=0.2, color=color, label=label)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Number of Bands')
    plt.title('Wavelength Distribution')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wavelength_distribution.png"), dpi=200)
    plt.close()
    
    # Create a table of wavelengths
    plt.figure(figsize=(10, len(wavelengths_nm) * 0.25 + 2))
    plt.axis('off')
    
    # Limit to displaying at most 100 wavelengths for readability
    display_wavelengths = wavelengths_nm
    if len(wavelengths_nm) > 100:
        # Sample every Nth wavelength
        step = len(wavelengths_nm) // 100 + 1
        display_wavelengths = wavelengths_nm[::step]
    
    table_data = [["Band Index", "Wavelength (nm)", "Wavelength (µm)"]]
    for i, wl in enumerate(display_wavelengths):
        # Find original index if we're sampling
        if len(wavelengths_nm) > 100:
            original_idx = i * step
        else:
            original_idx = i
        table_data.append([str(original_idx + 1), f"{wl:.2f}", f"{wl/1000:.4f}"])
    
    table = plt.table(cellText=table_data, colWidths=[0.2, 0.4, 0.4], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style the header
    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    plt.title('Band Wavelength Table', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wavelength_table.png"), dpi=200)
    plt.close()
    
    print(f"Saved wavelength distribution visualization to: {output_dir}")

def visualize_false_color_composites(tiles_data, metadata, output_dir, num_tiles=3):
    """
    Create false color composites for scientific visualization
    
    Parameters:
    tiles_data: Tensor of shape (N, C, H, W) containing tiles
    metadata: Dictionary with metadata including wavelengths
    output_dir: Directory to save visualizations
    num_tiles: Number of random tiles to visualize
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get wavelength information
    wavelengths_nm = None
    if 'wavelengths_nm' in metadata and metadata['wavelengths_nm']:
        wavelengths_nm = np.array(metadata['wavelengths_nm'])
    elif 'wavelengths_um' in metadata and metadata['wavelengths_um']:
        wavelengths_nm = np.array(metadata['wavelengths_um']) * 1000  # Convert to nm
    
    # If no wavelength info, we can't create meaningful false color composites
    if wavelengths_nm is None or len(wavelengths_nm) == 0:
        print("No wavelength information available for false color composites")
        return
    
    # Number of tiles
    num_total_tiles = tiles_data.shape[0]
    
    # Randomly select tiles
    if num_total_tiles < num_tiles:
        num_tiles = num_total_tiles
    
    tile_indices = random.sample(range(num_total_tiles), num_tiles)
    
    # Define different false color schemes
    color_schemes = [
        {
            'name': 'NIR_RGB',
            'description': 'NIR-Red-Green (Vegetation appears red)',
            'bands': [find_nearest_band(wavelengths_nm, 850), 
                     find_nearest_band(wavelengths_nm, 650), 
                     find_nearest_band(wavelengths_nm, 550)]
        },
        {
            'name': 'SWIR_NIR_Red',
            'description': 'SWIR-NIR-Red (Highlights moisture content)',
            'bands': [find_nearest_band(wavelengths_nm, 1650),
                     find_nearest_band(wavelengths_nm, 850),
                     find_nearest_band(wavelengths_nm, 650)]
        },
        {
            'name': 'NIR_SWIR_Blue',
            'description': 'NIR-SWIR-Blue (Urban features stand out)',
            'bands': [find_nearest_band(wavelengths_nm, 850),
                     find_nearest_band(wavelengths_nm, 1650),
                     find_nearest_band(wavelengths_nm, 450)]
        }
    ]
    
    # Create composite for each tile and color scheme
    for i, tile_idx in enumerate(tile_indices):
        # Get the tile
        tile = tiles_data[tile_idx]
        
        # Create composites for each color scheme
        for scheme in color_schemes:
            # Skip if we don't have the required bands
            valid_bands = True
            for band_idx in scheme['bands']:
                if band_idx >= tile.shape[0]:
                    valid_bands = False
                    break
            
            if not valid_bands:
                continue
            
            # Create RGB composite
            rgb = np.zeros((tile.shape[1], tile.shape[2], 3))
            for j, band_idx in enumerate(scheme['bands']):
                rgb[:,:,j] = tile[band_idx].numpy()
            
            # Clip values to [0, 1] range
            rgb = np.clip(rgb, 0, 1)
            
            # Apply contrast enhancement - linear stretch
            for j in range(3):
                p2 = np.percentile(rgb[:,:,j], 2)
                p98 = np.percentile(rgb[:,:,j], 98)
                if p98 > p2:
                    rgb[:,:,j] = np.clip((rgb[:,:,j] - p2) / (p98 - p2), 0, 1)
            
            # Plot the composite
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb)
            plt.title(f"Tile {tile_idx+1}: {scheme['name']} Composite\n{scheme['description']}\n" + 
                     f"R: {wavelengths_nm[scheme['bands'][0]]:.0f}nm, G: {wavelengths_nm[scheme['bands'][1]]:.0f}nm, B: {wavelengths_nm[scheme['bands'][2]]:.0f}nm")
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f"tile_{tile_idx+1}_{scheme['name']}.png"), dpi=200)
            plt.close()
    
    print(f"Saved false color composite visualizations to: {output_dir}")

def find_nearest_band(wavelengths, target_wavelength):
    """
    Find index of the closest wavelength to target
    
    Parameters:
    wavelengths: Array of wavelengths
    target_wavelength: Target wavelength
    
    Returns:
    int: Index of closest wavelength
    """
    return np.abs(np.array(wavelengths) - target_wavelength).argmin()

def visualize_stacked_bands(tiles_data, metadata, output_dir, num_tiles=3):
    """
    Create visualizations of multiple bands stacked in a grid
    
    Parameters:
    tiles_data: Tensor of shape (N, C, H, W) containing tiles
    metadata: Dictionary with metadata including wavelengths
    output_dir: Directory to save visualizations
    num_tiles: Number of random tiles to visualize
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get wavelength information
    wavelengths_nm = None
    if 'wavelengths_nm' in metadata and metadata['wavelengths_nm']:
        wavelengths_nm = np.array(metadata['wavelengths_nm'])
    elif 'wavelengths_um' in metadata and metadata['wavelengths_um']:
        wavelengths_nm = np.array(metadata['wavelengths_um']) * 1000  # Convert to nm
    
    # Number of tiles and bands
    num_total_tiles = tiles_data.shape[0]
    num_bands = tiles_data.shape[1]
    
    # Randomly select tiles
    if num_total_tiles < num_tiles:
        num_tiles = num_total_tiles
    
    tile_indices = random.sample(range(num_total_tiles), num_tiles)
    
    # Select bands to visualize (at most 16 bands for readability)
    if num_bands <= 16:
        band_indices = list(range(num_bands))
    else:
        # Sample bands across the spectrum
        band_indices = np.linspace(0, num_bands-1, 16, dtype=int).tolist()
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(len(band_indices))))
    
    # Visualize each selected tile
    for i, tile_idx in enumerate(tile_indices):
        # Get the tile
        tile = tiles_data[tile_idx]
        
        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()
        
        # Plot each band
        for j, band_idx in enumerate(band_indices):
            if j < len(axes):
                # Get the band data
                band_data = tile[band_idx].numpy()
                
                # Get wavelength label if available
                if wavelengths_nm is not None:
                    wavelength_label = f"{wavelengths_nm[band_idx]:.0f} nm"
                else:
                    wavelength_label = f"Band {band_idx+1}"
                
                # Plot the band
                im = axes[j].imshow(band_data, cmap='viridis', vmin=0, vmax=1)
                axes[j].set_title(wavelength_label, fontsize=9)
                axes[j].set_xticks([])
                axes[j].set_yticks([])
        
        # Hide unused subplots
        for j in range(len(band_indices), len(axes)):
            axes[j].axis('off')
        
        plt.suptitle(f"Tile {tile_idx+1}: Multiple Bands Visualization", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(os.path.join(output_dir, f"tile_{tile_idx+1}_multi_band.png"), dpi=250)
        plt.close()
    
    print(f"Saved multi-band visualizations to: {output_dir}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Visualize processed AVIRIS data')
    parser.add_argument('--pt_file', required=True, help='Path to PT file with processed tiles')
    parser.add_argument('--metadata', required=True, help='Path to metadata JSON file')
    parser.add_argument('--output', default='visualizations', help='Output directory for visualizations')
    parser.add_argument('--num_tiles', type=int, default=5, help='Number of random tiles to visualize')
    parser.add_argument('--num_pixels', type=int, default=5, help='Number of random pixels for spectral profiles')
    args = parser.parse_args()
    
    # Check if files exist
    pt_path = Path(args.pt_file)
    metadata_path = Path(args.metadata)
    
    if not pt_path.exists():
        print(f"Error: PT file not found: {args.pt_file}")
        return
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {args.metadata}")
        return
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f"{args.output}_{timestamp}"
    os.makedirs(output_base, exist_ok=True)
    
    # Load data
    tiles_data, metadata = load_data(args.pt_file, args.metadata)
    
    # Create various visualizations
    visualize_random_tiles(tiles_data, metadata, os.path.join(output_base, "tiles"), args.num_tiles)
    visualize_spectral_profiles(tiles_data, metadata, os.path.join(output_base, "spectral"), args.num_tiles, args.num_pixels)
    visualize_wavelength_coverage(metadata, os.path.join(output_base, "wavelengths"))
    visualize_false_color_composites(tiles_data, metadata, os.path.join(output_base, "false_color"), args.num_tiles)
    visualize_stacked_bands(tiles_data, metadata, os.path.join(output_base, "multi_band"), args.num_tiles)
    
    print(f"All visualizations saved to: {output_base}")
    print("Visualization types:")
    print("  - tiles: Individual bands from random tiles")
    print("  - spectral: Spectral profiles from random pixels")
    print("  - wavelengths: Distribution of wavelengths covered")
    print("  - false_color: False color composites for scientific visualization")
    print("  - multi_band: Grid display of multiple bands for the same tile")

if __name__ == "__main__":
    main()
