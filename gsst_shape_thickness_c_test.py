#!/usr/bin/env python3
"""
gsst_shape_thickness_c_test.py

Tests multiple GSST configurations:
1. Multiple shapes (with C4 symmetry)
2. Multiple thicknesses
3. All 11 crystallinity values (0.0 to 1.0)

Results are organized by shape and thickness, with plots showing all c-values together.

Requirements:
- S4 simulator executable at ../build/S4
- metasurface_gsst_nir.lua in the current directory
- gsst_partial_crys_data folder with the c-value data files
"""

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import time
import json
import concurrent.futures
from pathlib import Path

# Configuration
crystallinity_values = np.round(np.linspace(0.0, 1.0, 11), 1)  # 0.0, 0.1, ..., 1.0
thickness_values = [100, 300, 500, 700, 900]  # nm
num_shapes = 3  # Number of shapes to test
s4_path = "../build/S4"
lua_script = "metasurface_gsst_nir.lua"

# Create base directories
os.makedirs("shape_results", exist_ok=True)

# Define shapes
def define_shapes():
    """Define the shapes we want to test."""
    shapes = {}

    # Shape 1: Simple circle (4 points in first quadrant)
    shape1_points = []
    angles = [0, 30, 60, 90]  # degrees
    r = 0.2  # radius
    for angle in angles:
        rad = np.radians(angle)
        x = r * np.cos(rad)
        y = r * np.sin(rad)
        shape1_points.append((x, y))
    shapes[1] = {
        'name': 'circle',
        'points': shape1_points,
        'points_str': ";".join(f"{p[0]:.6f},{p[1]:.6f}" for p in shape1_points)
    }

    # Shape 2: Ellipse (4 points in first quadrant)
    shape2_points = []
    angles = [0, 30, 60, 90]  # degrees
    rx, ry = 0.3, 0.15  # radii
    for angle in angles:
        rad = np.radians(angle)
        x = rx * np.cos(rad)
        y = ry * np.sin(rad)
        shape2_points.append((x, y))
    shapes[2] = {
        'name': 'ellipse',
        'points': shape2_points,
        'points_str': ";".join(f"{p[0]:.6f},{p[1]:.6f}" for p in shape2_points)
    }

    # Shape 3: Random polygon (4 points in first quadrant)
    np.random.seed(42)  # For reproducibility
    shape3_points = []
    angles = np.sort(np.random.uniform(0, np.pi/2, 4))
    radii = np.random.uniform(0.1, 0.3, 4)
    for i in range(4):
        r = radii[i]
        theta = angles[i]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        shape3_points.append((x, y))
    shapes[3] = {
        'name': 'random',
        'points': shape3_points,
        'points_str': ";".join(f"{p[0]:.6f},{p[1]:.6f}" for p in shape3_points)
    }

    return shapes

def visualize_shape(shape_id, points, output_dir):
    """Create a visualization of the C4 symmetric shape."""
    # Generate the full shape with C4 symmetry
    all_points = list(points)
    for x, y in points:
        all_points.append((-y, x))    # Quadrant 2: (-y, x)
    for x, y in points:
        all_points.append((-x, -y))   # Quadrant 3: (-x, -y)
    for x, y in points:
        all_points.append((y, -x))    # Quadrant 4: (y, -x)
    
    # Create the plot
    plt.figure(figsize=(6, 6))
    
    # Plot the points
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    
    # Close the polygon
    xs.append(xs[0])
    ys.append(ys[0])
    
    plt.plot(xs, ys, 'o-', color='blue', linewidth=2)
    
    # Fill the polygon
    poly = Polygon(all_points, closed=True, alpha=0.3, color='blue')
    plt.gca().add_patch(poly)
    
    # Set plot properties
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.xlim(-0.35, 0.35)
    plt.ylim(-0.35, 0.35)
    plt.title(f"Shape #{shape_id}")
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/shape_{shape_id}.png", dpi=300, bbox_inches='tight')
    plt.close()

def run_s4_simulation(shape_str, c_value, thickness, shape_id, shape_name):
    """Run a single S4 simulation for the given shape, c-value, and thickness."""
    # Construct the command
    cmd = f'{s4_path} -a "{shape_str} -c {c_value} -t {thickness} -v -s" {lua_script}'
    
    print(f"Running shape={shape_id} ({shape_name}), thickness={thickness}nm, c={c_value}...")
    start_time = time.time()
    
    # Run the command
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        print(f"Error running simulation for shape={shape_id}, thickness={thickness}nm, c={c_value}")
        print("STDERR:", result.stderr)
        return None
    
    # Find the output file
    output_file = None
    for line in result.stdout.splitlines():
        if "Saved to" in line:
            output_file = line.split("Saved to", 1)[1].strip()
            break
    
    elapsed = time.time() - start_time
    print(f"Completed shape={shape_id}, thickness={thickness}nm, c={c_value} in {elapsed:.1f} seconds")
    
    return output_file

def load_results(output_file):
    """Load results from a CSV file."""
    if not output_file or not os.path.exists(output_file):
        return None
    
    data = {'wavelength': [], 'R': [], 'T': []}
    with open(output_file, 'r') as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            
            parts = line.strip().split(',')
            if len(parts) >= 7:  # c_value,thickness_nm,wavelength_nm,n_eff,k_eff,R,T,R_plus_T
                data['wavelength'].append(float(parts[2]))
                data['R'].append(float(parts[5]))
                data['T'].append(float(parts[6]))
    
    # Sort by wavelength
    indices = np.argsort(data['wavelength'])
    data['wavelength'] = [data['wavelength'][i] for i in indices]
    data['R'] = [data['R'][i] for i in indices]
    data['T'] = [data['T'][i] for i in indices]
    
    return data

def create_plot_for_thickness(shape_id, shape_name, thickness, results_by_c):
    """Create a plot showing all c-values for a specific shape and thickness."""
    output_dir = f"shape_results/shape_{shape_id}_{shape_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # Use a colormap for c-values
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(crystallinity_values)))
    
    # Plot reflection
    plt.subplot(2, 1, 1)
    for i, c_value in enumerate(crystallinity_values):
        if c_value in results_by_c:
            plt.plot(results_by_c[c_value]['wavelength'], 
                     results_by_c[c_value]['R'], 
                     color=colors[i], 
                     linewidth=2,
                     label=f"c={c_value}")
    
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Reflection', fontsize=12)
    plt.title(f'GSST Reflection: Shape {shape_id} ({shape_name}), Thickness {thickness}nm', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Crystallinity")
    
    # Plot transmission
    plt.subplot(2, 1, 2)
    for i, c_value in enumerate(crystallinity_values):
        if c_value in results_by_c:
            plt.plot(results_by_c[c_value]['wavelength'], 
                     results_by_c[c_value]['T'], 
                     color=colors[i], 
                     linewidth=2,
                     label=f"c={c_value}")
    
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Transmission', fontsize=12)
    plt.title(f'GSST Transmission: Shape {shape_id} ({shape_name}), Thickness {thickness}nm', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Crystallinity")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/thickness_{thickness}nm_all_c.png", dpi=300)
    plt.close()

def create_heatmap(shape_id, shape_name, thickness, results_by_c):
    """Create heatmaps of reflection and transmission vs wavelength and crystallinity."""
    output_dir = f"shape_results/shape_{shape_id}_{shape_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data for the heatmap
    c_values = sorted(results_by_c.keys())
    if not c_values:
        return
    
    wavelengths = results_by_c[c_values[0]]['wavelength']
    
    # Create arrays for heatmap data
    heatmap_R = np.zeros((len(c_values), len(wavelengths)))
    heatmap_T = np.zeros((len(c_values), len(wavelengths)))
    
    for i, c in enumerate(c_values):
        for j, _ in enumerate(wavelengths):
            heatmap_R[i, j] = results_by_c[c]['R'][j]
            heatmap_T[i, j] = results_by_c[c]['T'][j]
    
    # Create the heatmap visualizations
    plt.figure(figsize=(15, 10))
    
    # Reflection heatmap
    plt.subplot(2, 1, 1)
    plt.imshow(heatmap_R, aspect='auto', origin='lower', 
               extent=[min(wavelengths), max(wavelengths), min(c_values), max(c_values)],
               cmap='viridis')
    plt.colorbar(label='Reflection')
    plt.ylabel('Crystallinity')
    plt.title(f'GSST Reflection Heatmap: Shape {shape_id} ({shape_name}), Thickness {thickness}nm')
    
    # Transmission heatmap
    plt.subplot(2, 1, 2)
    plt.imshow(heatmap_T, aspect='auto', origin='lower',
               extent=[min(wavelengths), max(wavelengths), min(c_values), max(c_values)],
               cmap='viridis')
    plt.colorbar(label='Transmission')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Crystallinity')
    plt.title(f'GSST Transmission Heatmap: Shape {shape_id} ({shape_name}), Thickness {thickness}nm')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_thickness_{thickness}nm.png", dpi=300)
    plt.close()

def main():
    """Main function to run all simulations and create visualizations."""
    print(f"Defining shapes...")
    shapes = define_shapes()
    
    # Visualize each shape
    for shape_id, shape_data in shapes.items():
        if shape_id > num_shapes:
            continue
        
        shape_name = shape_data['name']
        output_dir = f"shape_results/shape_{shape_id}_{shape_name}"
        visualize_shape(shape_id, shape_data['points'], output_dir)
    
    # Run simulations for each shape, thickness, and c-value
    for shape_id, shape_data in shapes.items():
        if shape_id > num_shapes:
            continue
        
        shape_name = shape_data['name']
        shape_str = shape_data['points_str']
        
        print(f"\nProcessing Shape {shape_id} ({shape_name})...")
        
        for thickness in thickness_values:
            print(f"\nThickness: {thickness}nm")
            
            # Store results for each c-value
            results_by_c = {}
            
            for c_value in crystallinity_values:
                output_file = run_s4_simulation(shape_str, c_value, thickness, shape_id, shape_name)
                
                if output_file:
                    data = load_results(output_file)
                    if data:
                        results_by_c[c_value] = data
                    
                    # Copy the file to our organized directory
                    output_dir = f"shape_results/shape_{shape_id}_{shape_name}"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    new_file = f"{output_dir}/c{c_value}_t{thickness}.csv"
                    try:
                        with open(output_file, 'r') as src, open(new_file, 'w') as dst:
                            dst.write(src.read())
                    except Exception as e:
                        print(f"Error copying {output_file} to {new_file}: {e}")
            
            # Create plots for this shape and thickness
            if results_by_c:
                create_plot_for_thickness(shape_id, shape_name, thickness, results_by_c)
                create_heatmap(shape_id, shape_name, thickness, results_by_c)
    
    print("\nAll simulations and visualizations complete!")

if __name__ == "__main__":
    main()
