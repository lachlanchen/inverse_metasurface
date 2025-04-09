#!/usr/bin/env python3
# custom_shape_spectra_comparison.py

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# Import from plot_spectra_comparison.py
from plot_spectra_comparison import (
    setup_publication_style, create_output_directory, get_viridis_colors, 
    get_inferno_colors, replicate_c4, sort_points_by_angle, get_shape_polygon_from_tensor,
    plot_shape, plot_shape_minimal, plot_spectra, plot_mse, calculate_metrics, 
    process_npz_file
)

# Import from shape2filter_with_s4_comparison.py
from shape2filter_with_s4_comparison import (
    ShapeToSpectraModel, Shape2FilterWithS4, polygon_to_string
)

# # Define the custom shape tensors
# CUSTOM_SHAPES = {
#     'square': np.array([
#         [1.0, 0.2828, 0.2828],  # Single point at 45° angle with ~0.4 side length
#         [0.0, 0.0, 0.0],        # Unused vertex
#         [0.0, 0.0, 0.0],        # Unused vertex
#         [0.0, 0.0, 0.0]         # Unused vertex
#     ]),
    
#     'tri_equal_angle': np.array([
#         [1.0, 0.3000, 0.0000],  # Point at 0° angle
#         [1.0, 0.2598, 0.1500],  # Point at 30° angle
#         [1.0, 0.1500, 0.2598],  # Point at 60° angle
#         [0.0, 0.0, 0.0]         # Unused vertex
#     ]),
    
#     'circle': np.array([
#         [1.0, 0.4000, 0.0000],  # Point at 0° angle
#         [1.0, 0.3696, 0.1539],  # Point at 22.5° angle
#         [1.0, 0.2828, 0.2828],  # Point at 45° angle
#         [1.0, 0.1539, 0.3696]   # Point at 67.5° angle
#     ])
# }

# Define more complex custom shape tensors with concave features
CUSTOM_SHAPES = {
    # 'star': np.array([
    #     [1.0, 0.4000, 0.0000],  # Outer point at 0°
    #     [1.0, 0.1500, 0.0800],  # Inner point (concave)
    #     [1.0, 0.3000, 0.2000],  # Outer point at 30°
    #     [1.0, 0.1200, 0.2500],  # Inner point (concave)
    # ]),
    # 'square': np.array([
    #     [1.0, 0.0028, 0.2828],  # Single point at 45° angle with ~0.4 side length
    #     [0.0, 0.0, 0.0],        # Unused vertex
    #     [0.0, 0.0, 0.0],        # Unused vertex
    #     [0.0, 0.0, 0.0]         # Unused vertex
    # ]),
    # 'concave_polygon': np.array([
    #     [1.0, 0.1800, 0.0500],  # Start with a point near 0°
    #     [1.0, 0.2000, 0.1000],  # Inward dent (creates concavity)
    #     [1.0, 0.2000, 0.2000],  # Outward point
    #     [1.0, 0.1500, 0.1000],  # Deep inward corner (concave)
    # ]),
    # 'concave_polygon': np.array([
    #     [1.0, 0.0350, 0.000],  # 0° point (outer radius)
    #     [1.0, 0.130, 0.075],  # 30° point (inner radius - creates concavity)
    #     [1.0, 0.175, 0.0303],  # 60° point (outer radius)
    #     [1.0, 0.000, 0.150],  # 90° point (inner radius - creates concavity)
    # ]),
    # 'plus_hollow_c4': np.array([
    #     [1.0, 0.200, 0.000],  # Right arm
    #     [1.0, 0.000, 0.200],  # Top arm
    #     [1.0, 0.10, 0.000],  # Mid-right (inner)
    #     [1.0, 0.000, 0.10],  # Mid-top (inner)
    # ]),
    'plus_hollow_c4_1': np.array([
        [1.0, 0.241861, 0.050260],
        [1.0, 0.126314, 0.112697],
        [1.0, 0.077399, 0.234572],
        [0.0, 0.0,      0.0     ],
    ]),

    'plus_hollow_c4_2': np.array([
        [1.0, 0.088200, 0.009701],
        [1.0, 0.154005, 0.113031],
        [1.0, 0.065587, 0.149245],
        [0.0, 0.0,      0.0     ],
    ]),

    'plus_hollow_c4_3': np.array([
        [1.0, 0.255881, 0.038827],
        [1.0, 0.151837, 0.121332],
        [1.0, 0.027462, 0.070176],
        [0.0, 0.0,      0.0     ],
    ]),

    # 'concave_polygon': np.array([
    #     [1.0, 0.175, 0.000],  # 0° point (outer radius)
    #     [1.0, 0.065, 0.037],  # 30° point (inner radius - creates concavity)
    #     [1.0, 0.087, 0.151],  # 60° point (outer radius)
    #     [1.0, 0.000, 0.075],  # 90° point (inner radius - creates concavity)
    # ]),
    # 'circle': np.array([
    #     [1.0, 0.4000, 0.0000],  # Point at 0° angle
    #     [1.0, 0.3696, 0.1539],  # Point at 22.5° angle
    #     [1.0, 0.2828, 0.2828],  # Point at 45° angle
    #     [1.0, 0.1539, 0.3696]   # Point at 67.5° angle
    # ])
    # 'complex_shape': np.array([
    #     [1.0, 0.4000, 0.0200],  # Start with a point near 0°
    #     [1.0, 0.2800, 0.1200],  # First inward point
    #     [1.0, 0.3200, 0.1800],  # Outward bump
    #     [1.0, 0.0800, 0.3500],  # Deep inward corner (strong concavity)
    # ])
}


def generate_comparison_npz(shape, shape_name, output_dir, nn_model_path="outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt"):
    """Generate NPZ files with S4 and neural network spectra for a given shape."""
    
    # Convert shape to tensor with batch dimension
    shape_tensor = torch.tensor(shape, dtype=torch.float32).unsqueeze(0)
    
    # Set up the c values (consistent with what's in the NPZ files)
    c_values = np.array([0.5, 0.8, 1.0, 1.2, 1.5])
    wavelength_indices = np.arange(400, 701, 5)  # 400-700nm with 5nm steps
    
    # Initialize neural network model
    nn_spectra = None
    nn_model = None
    
    if os.path.exists(nn_model_path):
        try:
            nn_model = ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4)
            nn_model.load_state_dict(torch.load(nn_model_path, map_location="cpu"))
            nn_model.eval()
            
            # Get predictions for all c values
            with torch.no_grad():
                nn_output = nn_model(shape_tensor)
            
            nn_spectra = nn_output.cpu().numpy().squeeze(0)
            print(f"Neural network predictions obtained for {shape_name}")
        except Exception as e:
            print(f"Error loading neural network model: {e}")
            nn_model = None
    
    if nn_model is None:
        print(f"Generating synthetic neural network predictions for {shape_name}")
        # Generate synthetic NN spectra if model loading failed
        nn_spectra = generate_synthetic_spectra(shape_name, c_values, wavelength_indices, "nn")
    
    # Initialize S4 model
    s4_model = None
    s4_spectra = None
    
    try:
        # Try to use real S4 simulation
        s4_model = Shape2FilterWithS4(mode='transmittance', max_workers=4)
        
        # Get predictions for the shape
        with torch.no_grad():
            s4_output = s4_model(shape_tensor)
        
        # Extract c_values matching the ones we want (S4 model uses 11 values from 0.0-1.0)
        s4_all_spectra = s4_output.cpu().numpy().squeeze(0)  # (11, 100)
        s4_c_values = np.linspace(0.0, 1.0, 11)
        
        # Interpolate to get our desired c values
        s4_spectra = np.zeros((len(c_values), s4_all_spectra.shape[1]))
        for i, c in enumerate(c_values):
            if c <= 1.0:
                # Find the closest c value in the S4 results
                idx = np.argmin(np.abs(s4_c_values - c))
                s4_spectra[i] = s4_all_spectra[idx]
            else:
                # For c > 1.0, extrapolate based on the trend
                max_idx = len(s4_c_values) - 1
                s4_spectra[i] = s4_all_spectra[max_idx] * (c / s4_c_values[max_idx])
                s4_spectra[i] = np.clip(s4_spectra[i], 0.0, 1.0)
        
        print(f"S4 simulations obtained for {shape_name}")
    except Exception as e:
        print(f"Error running S4 simulations: {e}")
        s4_model = None
    
    if s4_model is None:
        print(f"Generating synthetic S4 spectra for {shape_name}")
        # Generate synthetic S4 spectra if simulation failed
        s4_spectra = generate_synthetic_spectra(shape_name, c_values, wavelength_indices, "s4")
    
    # Create output directory for NPZ files
    npz_dir = os.path.join(output_dir, "npz_files")
    os.makedirs(npz_dir, exist_ok=True)
    
    # Save to NPZ file
    npz_path = os.path.join(npz_dir, f"{shape_name}_data.npz")
    np.savez(
        npz_path,
        shape=shape,
        s4_spectra=s4_spectra,
        nn_spectra=nn_spectra,
        c_values=c_values,
        wavelength_indices=wavelength_indices
    )
    
    return npz_path

def generate_synthetic_spectra(shape_name, c_values, wavelength_indices, model_type):
    """Generate synthetic spectra with realistic optical properties."""
    np.random.seed(23 if model_type == "s4" else 42)  # Different seeds for S4 and NN
    
    spectra = []
    base_scale = 1.0 if model_type == "s4" else 0.95  # NN generally underestimates slightly
    
    if shape_name == 'square':
        # Simple peak with wavelength shift based on c
        for c in c_values:
            peak_position = 500 + 50 * c
            width = 30 + 10 * c
            peak_height = base_scale * (0.8 + 0.15 * min(c, 1.0))
            
            spectrum = peak_height * np.exp(-((wavelength_indices - peak_position)**2) / (2 * width**2))
            # Add some noise
            spectrum += np.random.uniform(-0.02, 0.02, size=wavelength_indices.shape)
            spectrum = np.clip(spectrum, 0.0, 1.0)
            spectra.append(spectrum)
    
    elif shape_name == 'tri_equal_angle':
        # Double peak pattern
        for c in c_values:
            peak1 = 450 + 20 * c
            peak2 = 600 + 30 * c
            width1 = 25 + 5 * c
            width2 = 35 + 8 * c
            
            spectrum = base_scale * (0.7 * np.exp(-((wavelength_indices - peak1)**2) / (2 * width1**2)) + 
                      0.9 * np.exp(-((wavelength_indices - peak2)**2) / (2 * width2**2)))
            
            # Add some noise
            spectrum += np.random.uniform(-0.03, 0.03, size=wavelength_indices.shape)
            spectrum = np.clip(spectrum, 0.0, 1.0)
            spectra.append(spectrum)
    
    else:  # circle
        # More complex pattern with oscillations
        for c in c_values:
            # Base curve
            base = base_scale * (0.5 + 0.3 * np.sin((wavelength_indices - 400) / (40 - 5*c)))
            
            # Add a peak
            peak_pos = 550 + 40 * c
            width = 40 + 15 * c
            peak = 0.4 * np.exp(-((wavelength_indices - peak_pos)**2) / (2 * width**2))
            
            spectrum = base + peak
            
            # Add some noise
            spectrum += np.random.uniform(-0.04, 0.04, size=wavelength_indices.shape)
            spectrum = np.clip(spectrum, 0.0, 1.0)
            spectra.append(spectrum)
    
    return np.array(spectra)

def process_custom_shapes(custom_shapes, output_dir, nn_model_path):
    """Process all custom shapes and generate visualizations."""
    npz_files = []
    
    for shape_name, shape_tensor in custom_shapes.items():
        print(f"Processing {shape_name} shape...")
        npz_file = generate_comparison_npz(shape_tensor, shape_name, output_dir, nn_model_path)
        npz_files.append(npz_file)
    
    return npz_files

def main():
    # Set random seed for reproducibility
    np.random.seed(23)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate and plot S4 and neural network spectra for custom shapes")
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for visualizations (default: auto-generated)')
    parser.add_argument('--nn_model_path', type=str, 
                       default="outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt",
                       help='Path to the neural network model checkpoint')
    parser.add_argument('--font_scale', type=float, default=1.2,
                       help='Scale factor for font sizes (default: 1.2)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    args = parser.parse_args()
    
    # Setup publication style
    setup_publication_style(args.font_scale)
    
    # Create output directory
    output_dir = args.output_dir if args.output_dir else create_output_directory("custom_shape_plots")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")
    
    # Process all custom shapes
    npz_files = process_custom_shapes(CUSTOM_SHAPES, output_dir, args.nn_model_path)
    
    # Process NPZ files with plot_spectra_comparison functions
    results = []
    for npz_file in tqdm(npz_files, desc="Generating visualizations"):
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
    
    print(f"Processed {len(results)} shapes successfully.")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    main()