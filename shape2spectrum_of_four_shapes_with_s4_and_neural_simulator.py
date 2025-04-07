#!/usr/bin/env python
# shape2spectrum_of_four_shapes_with_s4_and_neural_simulator.py
# Script to visualize four shapes and their spectra using both S4 and neural network

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import concurrent.futures
from scipy.interpolate import interp1d

# Import necessary functions and classes from shape2filter_with_s4_comparison.py
from shape2filter_with_s4_comparison import (
    replicate_c4, sort_points_by_angle, polygon_to_string, run_s4_for_c,
    ShapeToSpectraModel, read_results_csv
)

def load_shape_file(file_path):
    """Load a shape file and return its tensor representation."""
    if not os.path.exists(file_path):
        print(f"Error: Shape file not found: {file_path}")
        return None
    shape = np.load(file_path)
    print(f"Loaded shape with shape: {shape.shape}")
    return shape

def get_nn_spectra(shape, nn_model, device="cpu"):
    """Get spectrum for a shape using neural network model."""
    # Convert shape to tensor and add batch dimension
    shape_tensor = torch.tensor(shape, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get neural network prediction
    with torch.no_grad():
        nn_spectra = nn_model(shape_tensor)  # (1, 11, 100)
    
    nn_spectra_np = nn_spectra.squeeze(0).cpu().numpy()  # (11, 100)
    return nn_spectra_np

def get_s4_spectra(shape):
    """Get spectrum for a shape using S4 simulation."""
    valid_vertices = shape[:, 0] > 0.5
    if np.sum(valid_vertices) > 0:
        q1_vertices = shape[valid_vertices, 1:3]
        full_polygon = replicate_c4(q1_vertices)
        full_polygon = sort_points_by_angle(full_polygon)
        polygon_str = polygon_to_string(full_polygon)
        
        # Run S4 simulations for all c values
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = f"shape_{dt_str}"
        s4_spectra_list = []
        c_values = np.linspace(0.0, 1.0, 11)
        
        # Use ThreadPoolExecutor to run concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda c: (c, run_s4_for_c(polygon_str, c)), c_values))
        
        # Create result tensor
        s4_spectra = np.zeros((11, 100), dtype=np.float32)
        
        for i, (c, s4_csv) in enumerate(results):
            if s4_csv is not None:
                wv_s4, R_s4, T_s4 = read_results_csv(s4_csv)
                # Make sure the spectra are the right length
                if len(T_s4) == 100:
                    s4_spectra[i] = T_s4
                else:
                    # Interpolate to 100 points if necessary
                    x_original = np.arange(len(T_s4))
                    x_new = np.linspace(0, len(T_s4)-1, 100)
                    f = interp1d(x_original, T_s4, kind='linear', bounds_error=False, fill_value=(T_s4[0], T_s4[-1]))
                    s4_spectra[i] = f(x_new)
            else:
                print(f"[ERROR] S4 failed for c = {c:.1f} for UID {uid}")
        
        return s4_spectra
    else:
        # If no valid vertices, return zeros
        return np.zeros((11, 100), dtype=np.float32)

def create_shape_plot(shape, title, output_path):
    """Create a plot of the shape with minimal styling."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Make background transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    valid_vertices = shape[:, 0] > 0.5
    if np.sum(valid_vertices) > 0:
        q1_vertices = shape[valid_vertices, 1:3]
        full_polygon = replicate_c4(q1_vertices)
        full_polygon = sort_points_by_angle(full_polygon)
        
        # Create closed polygon
        closed_polygon = np.concatenate([full_polygon, full_polygon[0:1]], axis=0)
        ax.plot(closed_polygon[:, 0], closed_polygon[:, 1], 'b-', linewidth=1.5)
        
        # Add Q1 vertices with markers
        ax.scatter(q1_vertices[:, 0], q1_vertices[:, 1], color='red', s=50, marker='o')
    
    # Keep ticks and tick labels
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add dashed boundary
    for spine in ax.spines.values():
        spine.set_linestyle('--')
        spine.set_linewidth(2)
    
    # Set title
    ax.set_title(title, fontsize=14, pad=10)
    
    # Save the figure with transparent background
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)

def create_spectrum_plot(spectrum, title, output_path):
    """Create a plot of the spectrum with minimal styling."""
    # Use wider aspect ratio for spectrum plots
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Make background transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # Create a colormap using viridis
    cmap = plt.cm.viridis
    c_values = np.linspace(0.0, 1.0, spectrum.shape[0])
    
    for i, c in enumerate(c_values):
        ax.plot(np.arange(1, 101), spectrum[i], 
                color=cmap(i/10), linewidth=2.5)
    
    # Remove ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add dashed boundary
    for spine in ax.spines.values():
        spine.set_linestyle('--')
        spine.set_linewidth(2)
    
    # Set y-limits
    ax.set_ylim(0, 1)
    
    # No title for spectrum plots
    
    # Save the figure and the data with transparent background
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    
    # Save the data alongside the figure
    data_path = output_path.replace('.png', '_data.npz')
    np.savez(data_path, spectrum=spectrum, c_values=c_values, wavelength=np.arange(1, 101))
    
    plt.close(fig)

def main():
    # Create a timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = f"shape_spectrum_plots_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define paths to shape files
    base_path = "blind_noise_experiment_all_20250405_235131/blind_noise_10.0dB_to_40.0dB_20250405_235300"
    shape_files = {
        "initial": os.path.join(base_path, "initial_shape.npy"),
        "final": os.path.join(base_path, "final_shape.npy"),
        "lowest_cn": os.path.join(base_path, "lowest_cn", "shape.npy"),
        "lowest_mse": os.path.join(base_path, "lowest_mse", "shape.npy")
    }
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(23)
    np.random.seed(23)
    
    # Load neural network model
    model_path = "outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt"
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        # Try to find the model
        model_paths = os.popen("find ~/ProjectsLFS -name 'shape2spec_stageA.pt'").read().strip().split('\n')
        if model_paths and model_paths[0]:
            model_path = model_paths[0]
            print(f"Found model at: {model_path}")
    
    nn_model = ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4).to(device)
    try:
        nn_model.load_state_dict(torch.load(model_path, map_location=device))
        nn_model.eval()
        print("Neural network model loaded successfully.")
    except Exception as e:
        print(f"Error loading neural network model: {e}")
        print("Will continue with an untrained model.")
    
    # Process each shape
    for name, path in shape_files.items():
        print(f"Processing {name} shape from {path}...")
        
        # Load shape
        shape = load_shape_file(path)
        if shape is None:
            continue
        
        # Get neural network prediction
        print(f"Getting neural network prediction for {name} shape...")
        nn_spectra = get_nn_spectra(shape, nn_model, device)
        
        # Get S4 simulation results
        print(f"Running S4 simulations for {name} shape...")
        s4_spectra = get_s4_spectra(shape)
        
        # Create individual plots
        # 1. Shape plot
        shape_path = os.path.join(output_dir, f"{name}_shape.png")
        create_shape_plot(shape, f"{name.capitalize()} Shape", shape_path)
        
        # 2. Neural network spectrum plot
        nn_path = os.path.join(output_dir, f"{name}_nn_spectrum.png")
        create_spectrum_plot(nn_spectra, f"{name.capitalize()} Neural Network Spectrum", nn_path)
        
        # 3. S4 simulation spectrum plot
        s4_path = os.path.join(output_dir, f"{name}_s4_spectrum.png")
        create_spectrum_plot(s4_spectra, f"{name.capitalize()} S4 Simulation Spectrum", s4_path)
        
        # Also save the shape data
        shape_data_path = os.path.join(output_dir, f"{name}_shape_data.npz")
        np.savez(shape_data_path, shape=shape)
        
        print(f"Saved {name} shape plots and data")
    
    print(f"All plots and data saved in: {output_dir}")

if __name__ == "__main__":
    main()