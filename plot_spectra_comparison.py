#!/usr/bin/env python3
# plot_spectra_comparison.py
# Script to plot S4 and neural network spectra from NPZ files

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import glob
from datetime import datetime
import pandas as pd
from tqdm import tqdm

def setup_publication_style(font_scale=1.2):
    """Set up matplotlib for publication quality figures with custom font scaling."""
    base_font_size = 10 * font_scale
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
        'font.size': base_font_size,
        'mathtext.fontset': 'dejavusans',
        'axes.labelsize': base_font_size,
        'axes.titlesize': base_font_size * 1.1,
        'xtick.labelsize': base_font_size * 0.9,
        'ytick.labelsize': base_font_size * 0.9,
        'legend.fontsize': base_font_size * 0.9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.8,
        'lines.linewidth': 1.5,
        'lines.markersize': 3,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'axes.grid': False,
    })

def create_output_directory(name_prefix="spectrum_plots"):
    """Create output directory with date-time suffix."""
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{name_prefix}_{dt_str}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_viridis_colors(num_colors=11):
    """Get colors from viridis colormap for consistent styling."""
    viridis_colors = plt.cm.viridis(np.linspace(0, 1, num_colors))
    return viridis_colors

def get_inferno_colors(num_colors=11):
    """Get colors from inferno colormap for consistent styling."""
    inferno_colors = plt.cm.inferno(np.linspace(0, 1, num_colors))
    return inferno_colors

def replicate_c4(points):
    """
    Given an array of Q1 vertices (Nx2), replicates them to fill all four quadrants using C4 symmetry.
    """
    replicated = []
    for (x, y) in points:
        replicated.append([x, y])
        replicated.append([-y, x])
        replicated.append([-x, -y])
        replicated.append([y, -x])
    return np.array(replicated, dtype=np.float32)

def sort_points_by_angle(points):
    """
    Sorts a set of 2D points by their polar angle around the centroid.
    """
    if len(points) < 3:
        return points
    cx, cy = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    idx = np.argsort(angles)
    return points[idx]

def get_shape_polygon_from_tensor(shape_tensor):
    """
    Extract a polygon from a shape tensor
    Args:
        shape_tensor: shape 4x3 tensor
    Returns:
        full_polygon, q1_vertices
    """
    # Process shape to get polygon

    print("shape_tensor: ", shape_tensor)
    valid_vertices = shape_tensor[:, 0] > 0.5
    
    if np.sum(valid_vertices) > 0:
        q1_vertices = shape_tensor[valid_vertices, 1:3]
        full_polygon = replicate_c4(q1_vertices)
        full_polygon = sort_points_by_angle(full_polygon)
        return full_polygon, q1_vertices
    else:
        return None, None

def plot_shape(shape_tensor, ax):
    """Plot shape with transparent background and no grid."""
    full_polygon, q1_vertices = get_shape_polygon_from_tensor(shape_tensor)
    
    if full_polygon is not None:
        # Make figure background transparent
        ax.patch.set_alpha(0.0)
        
        # Plot the shape outline with a clean look
        closed_polygon = np.concatenate([full_polygon, full_polygon[0:1]], axis=0)
        ax.plot(closed_polygon[:, 0], closed_polygon[:, 1], '-', color='#3366CC', linewidth=1.8)
        
        # Add original vertices with elegant markers
        ax.scatter(q1_vertices[:, 0], q1_vertices[:, 1], color='#CC3366', 
                  s=40, marker='o', zorder=10, edgecolors='white', linewidths=1)
        
        # Clean axis appearance
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#444444')
        ax.spines['left'].set_color('#444444')
        
        # Set clean tick parameters
        ax.tick_params(direction='out', length=4, width=0.8, colors='#444444')
        
        # Set limits with padding
        ax_range = max(np.max(np.abs(full_polygon[:, 0])), np.max(np.abs(full_polygon[:, 1])))
        padding = 0.2
        ax_limit = ax_range + padding
        ax.set_xlim(-ax_limit, ax_limit)
        ax.set_ylim(-ax_limit, ax_limit)
        
        # Add minimal labels
        ax.set_xlabel("x", fontsize=10, color='#444444')
        ax.set_ylabel("y", fontsize=10, color='#444444')

def plot_shape_minimal(shape_tensor, ax):
    """Plot only the shape with transparent background and no decorations.
    Maintains relative size within a fixed -0.5 to 0.5 coordinate range."""
    full_polygon, q1_vertices = get_shape_polygon_from_tensor(shape_tensor)
    
    if full_polygon is not None:
        # Make figure background transparent
        ax.patch.set_alpha(0.0)
        
        # Plot the shape outline
        closed_polygon = np.concatenate([full_polygon, full_polygon[0:1]], axis=0)
        ax.plot(closed_polygon[:, 0], closed_polygon[:, 1], '-', color='#3366CC', linewidth=1.8)
        
        # Add original vertices in first quadrant with elegant markers (same as in plot_shape)
        ax.scatter(q1_vertices[:, 0], q1_vertices[:, 1], color='#CC3366', 
                  s=40, marker='o', zorder=10, edgecolors='white', linewidths=1)
        
        # Remove all axes elements
        ax.set_axis_off()
        
        # Set fixed limits to -0.5 to 0.5 to standardize the display area
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        
        # Ensure equal aspect ratio
        ax.set_aspect('equal')

def plot_spectra(nn_spectra, s4_spectra, wavelength_indices, c_values, ax):
    """Plot neural network and S4 spectra on the same axes."""
    viridis_colors = get_viridis_colors(len(c_values))
    
    # Make background transparent
    ax.patch.set_alpha(0.0)
    
    # Clean axis appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    
    # Set tick parameters
    ax.tick_params(direction='out', length=4, width=0.8, colors='#444444')
    
    # Plot S4 spectra (solid lines)
    for i, c in enumerate(c_values):
        ax.plot(wavelength_indices, s4_spectra[i], color=viridis_colors[i], 
                linewidth=1.5, label=f"S4, c={c:.1f}")
        
    # Plot NN spectra (dashed lines)
    for i, c in enumerate(c_values):
        ax.plot(wavelength_indices, nn_spectra[i], '--', color=viridis_colors[i], 
                linewidth=1.5, dashes=(5, 2), alpha=0.9, label=f"NN, c={c:.1f}")
    
    # Clean up axes
    ax.set_xlim(wavelength_indices[0], wavelength_indices[-1])
    ax.set_ylim(0, 1)
    
    # Add labels
    ax.set_xlabel("Wavelength Index", fontsize=10, color='#444444')
    ax.set_ylabel("Transmittance", fontsize=10, color='#444444')
    
    return ax

def plot_mse(nn_spectra, s4_spectra, wavelength_indices, c_values, ax):
    """Plot MSE between neural network and S4 spectra."""
    viridis_colors = get_viridis_colors(len(c_values))
    
    # Make background transparent
    ax.patch.set_alpha(0.0)
    
    # Clean axis appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    
    # Set tick parameters
    ax.tick_params(direction='out', length=4, width=0.8, colors='#444444')
    
    # Calculate MSE for each c value
    mse_values = np.mean((s4_spectra - nn_spectra)**2, axis=1)
    
    # Plot MSE bars
    ax.bar(range(len(c_values)), mse_values, color='#777777', alpha=0.7)
    
    # Set x-axis labels
    ax.set_xticks(range(len(c_values)))
    ax.set_xticklabels([f'{c:.1f}' for c in c_values])
    
    # Add labels
    ax.set_xlabel("c Value", fontsize=10, color='#444444')
    ax.set_ylabel("MSE", fontsize=10, color='#444444')
    
    return ax, mse_values

def calculate_metrics(nn_spectra, s4_spectra, c_values):
    """Calculate MSE, MAE, and max error for each c value."""
    metrics = {
        'c_value': c_values,
        'mse': np.mean((s4_spectra - nn_spectra)**2, axis=1),
        'mae': np.mean(np.abs(s4_spectra - nn_spectra), axis=1),
        'max_error': np.max(np.abs(s4_spectra - nn_spectra), axis=1)
    }
    
    return metrics

def find_npz_files(base_dir):
    """Find all NPZ files in the given directory structure."""
    pattern = os.path.join(base_dir, '**', '*_data.npz')
    return glob.glob(pattern, recursive=True)

def process_npz_file(npz_file, output_dir, font_scale=1.2):
    """Process a single NPZ file and create visualizations."""
    try:
        data = np.load(npz_file, allow_pickle=True)
        
        # Extract data
        shape = data['shape']
        s4_spectra = data['s4_spectra']
        nn_spectra = data['nn_spectra']
        c_values = data['c_values']
        wavelength_indices = data['wavelength_indices']
        
        # Get sample ID from filename
        filename = os.path.basename(npz_file)
        sample_id = filename.split('_data.npz')[0]
        
        # Calculate metrics
        metrics = calculate_metrics(nn_spectra, s4_spectra, c_values)
        
        # Create output paths
        shape_output = os.path.join(output_dir, f"{sample_id}_shape.png")
        shape_minimal_output = os.path.join(output_dir, f"{sample_id}_shape_minimal.png")  # New output path
        s4_output = os.path.join(output_dir, f"{sample_id}_s4.png")
        nn_output = os.path.join(output_dir, f"{sample_id}_nn.png")
        combined_output = os.path.join(output_dir, f"{sample_id}_combined.png")
        mse_output = os.path.join(output_dir, f"{sample_id}_mse.png")
        mse_spectrum_output = os.path.join(output_dir, f"{sample_id}_mse_spectrum.png")
        all_output = os.path.join(output_dir, f"{sample_id}_all.png")
        
        # New output path for the combined S4, NN, and MSE plot
        combined_with_mse_output = os.path.join(output_dir, f"{sample_id}_combined_with_mse.png")
        
        # Process all the existing plots first
        # 1. Plot shape
        fig_shape, ax_shape = plt.subplots(figsize=(3, 3))
        plot_shape(shape, ax_shape)
        plt.tight_layout()
        fig_shape.savefig(shape_output, dpi=300, transparent=True)
        plt.close(fig_shape)
        
        # 1b. NEW: Plot minimal shape
        fig_shape_minimal, ax_shape_minimal = plt.subplots(figsize=(3, 3))
        plot_shape_minimal(shape, ax_shape_minimal)
        # No tight_layout needed as there are no axis elements
        fig_shape_minimal.savefig(shape_minimal_output, dpi=300, transparent=True)
        plt.close(fig_shape_minimal)
        
        # 2. Plot S4 spectra only
        fig_s4, ax_s4 = plt.subplots(figsize=(5, 4))
        ax_s4.patch.set_alpha(0.0)
        ax_s4.spines['top'].set_visible(False)
        ax_s4.spines['right'].set_visible(False)
        viridis_colors = get_viridis_colors(len(c_values))
        
        for i, c in enumerate(c_values):
            ax_s4.plot(wavelength_indices, s4_spectra[i], color=viridis_colors[i], 
                      linewidth=1.5, label=f"c={c:.1f}")
            
        ax_s4.set_xlim(wavelength_indices[0], wavelength_indices[-1])
        ax_s4.set_ylim(0, 1)
        ax_s4.set_xlabel("Wavelength Index", fontsize=10, color='#444444')
        ax_s4.set_ylabel("Transmittance (S4)", fontsize=10, color='#444444')
        plt.tight_layout()
        fig_s4.savefig(s4_output, dpi=300, transparent=True)
        plt.close(fig_s4)
        
        # 3. Plot NN spectra only
        fig_nn, ax_nn = plt.subplots(figsize=(5, 4))
        ax_nn.patch.set_alpha(0.0)
        ax_nn.spines['top'].set_visible(False)
        ax_nn.spines['right'].set_visible(False)
        
        for i, c in enumerate(c_values):
            ax_nn.plot(wavelength_indices, nn_spectra[i], '--', color=viridis_colors[i], 
                      linewidth=1.5, dashes=(5, 2), label=f"c={c:.1f}")
            
        ax_nn.set_xlim(wavelength_indices[0], wavelength_indices[-1])
        ax_nn.set_ylim(0, 1)
        ax_nn.set_xlabel("Wavelength Index", fontsize=10, color='#444444')
        ax_nn.set_ylabel("Transmittance (NN)", fontsize=10, color='#444444')
        plt.tight_layout()
        fig_nn.savefig(nn_output, dpi=300, transparent=True)
        plt.close(fig_nn)
        
        # 4. Plot combined S4 and NN spectra
        fig_combined, ax_combined = plt.subplots(figsize=(5, 4))
        plot_spectra(nn_spectra, s4_spectra, wavelength_indices, c_values, ax_combined)
        plt.tight_layout()
        fig_combined.savefig(combined_output, dpi=300, transparent=True)
        plt.close(fig_combined)
        
        # 5. Plot MSE bars
        fig_mse, ax_mse = plt.subplots(figsize=(5, 4))
        _, mse_values = plot_mse(nn_spectra, s4_spectra, wavelength_indices, c_values, ax_mse)
        plt.tight_layout()
        fig_mse.savefig(mse_output, dpi=300, transparent=True)
        plt.close(fig_mse)
        
        # 6. Plot MSE spectrum
        fig_mse_spectrum, ax_mse_spectrum = plt.subplots(figsize=(5, 4))
        ax_mse_spectrum.patch.set_alpha(0.0)
        ax_mse_spectrum.spines['top'].set_visible(False)
        ax_mse_spectrum.spines['right'].set_visible(False)
        
        # Calculate MSE across wavelength for each c value
        for i, c in enumerate(c_values):
            mse_spectrum = (s4_spectra[i] - nn_spectra[i])**2
            ax_mse_spectrum.plot(wavelength_indices, mse_spectrum, color=viridis_colors[i], 
                    linewidth=1.5, label=f"c={c:.1f}")
            
        ax_mse_spectrum.set_xlim(wavelength_indices[0], wavelength_indices[-1])
        ax_mse_spectrum.set_ylim(0, 1)  # Set y-limit from 0 to 1
        ax_mse_spectrum.set_xlabel("Wavelength Index", fontsize=10, color='#444444')
        ax_mse_spectrum.set_ylabel("MSE", fontsize=10, color='#444444')
        plt.tight_layout()
        fig_mse_spectrum.savefig(mse_spectrum_output, dpi=300, transparent=True)
        plt.close(fig_mse_spectrum)
        
        # 7. Plot all in one figure
        fig_all = plt.figure(figsize=(10, 8))
        gs = GridSpec(2, 2, figure=fig_all)
        
        # Shape (top left)
        ax_shape_all = fig_all.add_subplot(gs[0, 0])
        plot_shape(shape, ax_shape_all)
        
        # Combined spectra (top right)
        ax_spectra_all = fig_all.add_subplot(gs[0, 1])
        plot_spectra(nn_spectra, s4_spectra, wavelength_indices, c_values, ax_spectra_all)
        
        # MSE (bottom left)
        ax_mse_all = fig_all.add_subplot(gs[1, 0])
        plot_mse(nn_spectra, s4_spectra, wavelength_indices, c_values, ax_mse_all)
        
        # Error visualization (bottom right)
        ax_error = fig_all.add_subplot(gs[1, 1])
        ax_error.patch.set_alpha(0.0)
        ax_error.spines['top'].set_visible(False)
        ax_error.spines['right'].set_visible(False)
        
        # Plot average error across wavelengths for each c value
        wavelength_error = np.mean(np.abs(s4_spectra - nn_spectra), axis=0)
        ax_error.plot(wavelength_indices, wavelength_error, color='#777777', linewidth=1.5)
        ax_error.fill_between(wavelength_indices, 0, wavelength_error, color='#AAAAAA', alpha=0.3)
        
        ax_error.set_xlim(wavelength_indices[0], wavelength_indices[-1])
        ax_error.set_xlabel("Wavelength Index", fontsize=10, color='#444444')
        ax_error.set_ylabel("Average Absolute Error", fontsize=10, color='#444444')
        
        plt.tight_layout()
        fig_all.savefig(all_output, dpi=300, transparent=True)
        plt.close(fig_all)
        
        # 8. NEW PLOT: Combined S4, NN, and MSE spectrum with inferno colormap
        fig_combined_mse, (ax_spectra, ax_mse) = plt.subplots(2, 1, figsize=(7, 6), gridspec_kw={'height_ratios': [2, 1]})
        
        # Top subplot: S4 and NN spectra
        ax_spectra.patch.set_alpha(0.0)
        ax_spectra.spines['top'].set_visible(False)
        ax_spectra.spines['right'].set_visible(False)
        
        # Plot S4 spectra (solid lines)
        for i, c in enumerate(c_values):
            ax_spectra.plot(wavelength_indices, s4_spectra[i], color=viridis_colors[i], 
                    linewidth=1.5, label=f"S4, c={c:.1f}")
            
        # Plot NN spectra (dashed lines)
        for i, c in enumerate(c_values):
            ax_spectra.plot(wavelength_indices, nn_spectra[i], '--', color=viridis_colors[i], 
                    linewidth=1.5, dashes=(5, 2), alpha=0.9, label=f"NN, c={c:.1f}")
        
        ax_spectra.set_xlim(wavelength_indices[0], wavelength_indices[-1])
        ax_spectra.set_ylim(0, 1)
        ax_spectra.set_ylabel("Transmittance", fontsize=10, color='#444444')
        
        # Bottom subplot: MSE in inferno colormap
        ax_mse.patch.set_alpha(0.0)
        ax_mse.spines['top'].set_visible(False)
        ax_mse.spines['right'].set_visible(False)
        
        # Get inferno colors
        inferno_colors = get_inferno_colors(len(c_values))
        
        # Calculate MSE across wavelength for each c value
        for i, c in enumerate(c_values):
            mse_spectrum = (s4_spectra[i] - nn_spectra[i])**2
            ax_mse.plot(wavelength_indices, mse_spectrum, color=inferno_colors[i], 
                   linewidth=1.5, label=f"c={c:.1f}")
            
        ax_mse.set_xlim(wavelength_indices[0], wavelength_indices[-1])
        ax_mse.set_ylim(0, 1)
        ax_mse.set_xlabel("Wavelength Index", fontsize=10, color='#444444')
        ax_mse.set_ylabel("MSE", fontsize=10, color='#444444')
        
        plt.tight_layout()
        fig_combined_mse.savefig(combined_with_mse_output, dpi=300, transparent=True)
        plt.close(fig_combined_mse)
        
        # Return metrics for this sample
        result = {
            'sample_id': sample_id,
            'npz_file': npz_file,
            'avg_mse': np.mean(metrics['mse']),
            'avg_mae': np.mean(metrics['mae']),
            'max_error': np.max(metrics['max_error']),
            'c_values': c_values,
            'mse_values': metrics['mse'],
            'mae_values': metrics['mae'],
            'max_error_values': metrics['max_error']
        }
        
        return result
    
    except Exception as e:
        print(f"Error processing {npz_file}: {str(e)}")
        return None

def main():
    # Set random seed for reproducibility
    np.random.seed(23)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot S4 and neural network spectra from NPZ files")
    parser.add_argument('--base_dir', type=str, default='simulator_verification',
                       help='Base directory containing NPZ files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for visualizations (default: auto-generated)')
    parser.add_argument('--font_scale', type=float, default=1.2,
                       help='Scale factor for font sizes (default: 1.2)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    args = parser.parse_args()
    
    # Setup publication style
    setup_publication_style(args.font_scale)
    
    # Create output directory
    output_dir = args.output_dir if args.output_dir else create_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")
    
    # Find all NPZ files
    npz_files = find_npz_files(args.base_dir)
    
    if not npz_files:
        print(f"Error: No NPZ files found in {args.base_dir}")
        return
    
    print(f"Found {len(npz_files)} NPZ files")
    
    # Process files
    results = []
    
    for npz_file in tqdm(npz_files, desc="Processing NPZ files"):
        result = process_npz_file(npz_file, output_dir, args.font_scale)
        if result:
            results.append(result)
    
    # Save metrics to CSV
    if results:
        # Create summary dataframe
        summary_df = pd.DataFrame([
            {
                'sample_id': result['sample_id'],
                'file_path': result['npz_file'],
                'avg_mse': result['avg_mse'],
                'avg_mae': result['avg_mae'],
                'max_error': result['max_error']
            }
            for result in results
        ])
        
        # Create detailed dataframe with per-c-value metrics
        detailed_rows = []
        for result in results:
            for i, c in enumerate(result['c_values']):
                detailed_rows.append({
                    'sample_id': result['sample_id'],
                    'c_value': c,
                    'mse': result['mse_values'][i],
                    'mae': result['mae_values'][i],
                    'max_error': result['max_error_values'][i]
                })
        
        detailed_df = pd.DataFrame(detailed_rows)
        
        # Save to CSV
        summary_csv = os.path.join(output_dir, "metrics_summary.csv")
        detailed_csv = os.path.join(output_dir, "metrics_detailed.csv")
        
        summary_df.to_csv(summary_csv, index=False)
        detailed_df.to_csv(detailed_csv, index=False)
        
        print(f"Saved metrics to {summary_csv} and {detailed_csv}")
        
        # Save data used for plots
        data_dir = os.path.join(output_dir, "plot_data")
        os.makedirs(data_dir, exist_ok=True)
        
        for result in results:
            data_file = os.path.join(data_dir, f"{result['sample_id']}_metrics.csv")
            metric_df = pd.DataFrame({
                'c_value': result['c_values'],
                'mse': result['mse_values'],
                'mae': result['mae_values'],
                'max_error': result['max_error_values']
            })
            metric_df.to_csv(data_file, index=False)
    
    print(f"Processed {len(results)} files successfully.")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    main()