#!/usr/bin/env python3
"""
gsst_comprehensive_test.py

Tests GSST metasurfaces with:
1. All 11 crystallinity values (0.0 to 1.0)
2. Multiple random freeform polygon shapes with C4 symmetry
3. Fixed thickness of 500nm

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
from matplotlib.colors import LinearSegmentedColormap
import time
import random
import concurrent.futures
import json

# Configuration
crystallinity_values = np.round(np.linspace(0.0, 1.0, 11), 1)  # 0.0, 0.1, ..., 1.0
thickness = 500  # nm (fixed for all simulations)
num_shapes = 5   # Number of random shapes to generate
s4_path = "../build/S4"
lua_script = "metasurface_gsst_nir.lua"

# Create output directories
os.makedirs("shapes", exist_ok=True)
os.makedirs("results-nir", exist_ok=True)
os.makedirs("visualization", exist_ok=True)

# Generate shapes with C4 symmetry
def generate_random_c4_shape(shape_id, max_radius=0.3, min_radius=0.1, max_points=4, seed=None):
    """Generate a random C4 symmetric shape with points in the first quadrant."""
    if seed is not None:
        np.random.seed(seed)
    
    # Decide how many points (2 to max_points)
    num_points = np.random.randint(2, max_points + 1)
    
    # Generate random angles in the first quadrant (0 to 90 degrees)
    angles = np.sort(np.random.uniform(0, np.pi/2, num_points))
    
    # Generate random radii
    radii = np.random.uniform(min_radius, max_radius, num_points)
    
    # Create points in the first quadrant
    points = []
    for i in range(num_points):
        r = radii[i]
        theta = angles[i]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append((x, y))
    
    # Sort by angle (counter-clockwise)
    points.sort(key=lambda p: np.arctan2(p[1], p[0]))
    
    # Generate all quadrants for visualization (but we only return first quadrant for S4)
    all_quadrants = list(points)
    for x, y in points:
        all_quadrants.append((-y, x))    # Quadrant 2: (-y, x)
    for x, y in points:
        all_quadrants.append((-x, -y))   # Quadrant 3: (-x, -y)
    for x, y in points:
        all_quadrants.append((y, -x))    # Quadrant 4: (y, -x)
    
    # Convert to string format for S4 (first quadrant only)
    shape_str = ";".join(f"{p[0]:.6f},{p[1]:.6f}" for p in points)
    
    # Visualize the shape
    visualize_shape(all_quadrants, shape_id)
    
    return shape_str, all_quadrants

def visualize_shape(points, shape_id):
    """Create a visualization of the C4 symmetric shape."""
    plt.figure(figsize=(6, 6))
    
    # Plot the points
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.plot(xs + [xs[0]], ys + [ys[0]], 'o-', color='blue', linewidth=1.5)
    
    # Fill the polygon
    poly = Polygon(points, closed=True, alpha=0.3, color='blue')
    plt.gca().add_patch(poly)
    
    # Set plot properties
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.xlim(-0.35, 0.35)
    plt.ylim(-0.35, 0.35)
    plt.title(f"Shape #{shape_id} (C4 Symmetric)")
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    
    # Save the figure
    plt.savefig(f"shapes/shape_{shape_id}.png", dpi=300, bbox_inches='tight')
    plt.close()

def run_s4_simulation(shape_str, c_value, shape_id):
    """Run a single S4 simulation for the given shape and c-value."""
    # Construct the command
    cmd = f'{s4_path} -a "{shape_str} -c {c_value} -t {thickness} -v -s" {lua_script}'
    
    print(f"Running shape={shape_id}, c={c_value}...")
    start_time = time.time()
    
    # Run the command
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        print(f"Error running simulation for shape={shape_id}, c={c_value}")
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
        return None
    
    # Find the output file
    output_file = None
    for line in result.stdout.splitlines():
        if "Saved to" in line:
            output_file = line.split("Saved to", 1)[1].strip()
            break
    
    elapsed = time.time() - start_time
    print(f"Completed shape={shape_id}, c={c_value} in {elapsed:.1f} seconds")
    
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
    
    # Make sure wavelengths are sorted
    sorted_indices = np.argsort(data['wavelength'])
    data['wavelength'] = [data['wavelength'][i] for i in sorted_indices]
    data['R'] = [data['R'][i] for i in sorted_indices]
    data['T'] = [data['T'][i] for i in sorted_indices]
    
    return data

def visualize_c_value_results(results, c_value):
    """Create visualization for a specific c-value across all shapes."""
    plt.figure(figsize=(15, 10))
    
    # Plot Reflection
    plt.subplot(2, 1, 1)
    for shape_id, data in results.items():
        plt.plot(data['wavelength'], data['R'], label=f"Shape {shape_id}")
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflection')
    plt.title(f'GSST Reflection Spectrum (c={c_value}, t={thickness}nm) for Different Shapes')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot Transmission
    plt.subplot(2, 1, 2)
    for shape_id, data in results.items():
        plt.plot(data['wavelength'], data['T'], label=f"Shape {shape_id}")
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission')
    plt.title(f'GSST Transmission Spectrum (c={c_value}, t={thickness}nm) for Different Shapes')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"visualization/c{c_value}_all_shapes.png", dpi=300)
    plt.close()

def visualize_shape_results(results, shape_id):
    """Create visualization for a specific shape across all c-values."""
    plt.figure(figsize=(15, 10))
    
    # Create a colormap for c-values
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(crystallinity_values)))
    
    # Plot Reflection
    plt.subplot(2, 1, 1)
    for i, c_value in enumerate(crystallinity_values):
        if c_value in results:
            plt.plot(results[c_value]['wavelength'], results[c_value]['R'], 
                     color=colors[i], label=f"c={c_value}")
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflection')
    plt.title(f'GSST Reflection Spectrum (Shape {shape_id}, t={thickness}nm) for Different Crystallinity Values')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot Transmission
    plt.subplot(2, 1, 2)
    for i, c_value in enumerate(crystallinity_values):
        if c_value in results:
            plt.plot(results[c_value]['wavelength'], results[c_value]['T'], 
                     color=colors[i], label=f"c={c_value}")
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission')
    plt.title(f'GSST Transmission Spectrum (Shape {shape_id}, t={thickness}nm) for Different Crystallinity Values')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"visualization/shape{shape_id}_all_c_values.png", dpi=300)
    plt.close()

def create_heatmap(results, shape_id):
    """Create heatmaps of R and T vs wavelength and crystallinity."""
    # Collect data for the heatmap
    c_values = sorted(results.keys())
    wavelengths = results[c_values[0]]['wavelength']
    
    # Create arrays for heatmap data
    heatmap_R = np.zeros((len(c_values), len(wavelengths)))
    heatmap_T = np.zeros((len(c_values), len(wavelengths)))
    
    for i, c in enumerate(c_values):
        for j, wl in enumerate(wavelengths):
            heatmap_R[i, j] = results[c]['R'][j]
            heatmap_T[i, j] = results[c]['T'][j]
    
    # Create the heatmap visualizations
    plt.figure(figsize=(15, 10))
    
    # Reflection heatmap
    plt.subplot(2, 1, 1)
    plt.imshow(heatmap_R, aspect='auto', origin='lower', 
               extent=[min(wavelengths), max(wavelengths), min(c_values), max(c_values)],
               cmap='viridis')
    plt.colorbar(label='Reflection')
    plt.ylabel('Crystallinity')
    plt.title(f'GSST Reflection Heatmap (Shape {shape_id}, t={thickness}nm)')
    
    # Transmission heatmap
    plt.subplot(2, 1, 2)
    plt.imshow(heatmap_T, aspect='auto', origin='lower',
               extent=[min(wavelengths), max(wavelengths), min(c_values), max(c_values)],
               cmap='viridis')
    plt.colorbar(label='Transmission')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Crystallinity')
    plt.title(f'GSST Transmission Heatmap (Shape {shape_id}, t={thickness}nm)')
    
    plt.tight_layout()
    plt.savefig(f"visualization/heatmap_shape{shape_id}.png", dpi=300)
    plt.close()

def main():
    """Main function to run all simulations and create visualizations."""
    print(f"Generating {num_shapes} random C4 symmetric shapes...")
    
    # Generate random shapes
    shapes = {}
    for i in range(1, num_shapes + 1):
        shape_str, points = generate_random_c4_shape(i, seed=i*42)
        shapes[i] = {'shape_str': shape_str, 'points': points}
    
    # Save shape data for reference
    with open('shapes/shapes_data.json', 'w') as f:
        # Convert points to list for JSON serialization
        shapes_json = {k: {'shape_str': v['shape_str'], 
                         'points': [[p[0], p[1]] for p in v['points']]} 
                      for k, v in shapes.items()}
        json.dump(shapes_json, f, indent=2)
    
    # Results will be organized by c-value and by shape
    results_by_c = {c: {} for c in crystallinity_values}
    results_by_shape = {i: {} for i in range(1, num_shapes + 1)}
    
    # Run simulations for all combinations of shapes and c-values
    print(f"Running simulations for {num_shapes} shapes × {len(crystallinity_values)} c-values...")
    
    # Consider using ProcessPoolExecutor for parallel execution, but keep it simple for now
    for shape_id, shape_data in shapes.items():
        for c_value in crystallinity_values:
            output_file = run_s4_simulation(shape_data['shape_str'], c_value, shape_id)
            if output_file:
                data = load_results(output_file)
                if data:
                    results_by_c[c_value][shape_id] = data
                    results_by_shape[shape_id][c_value] = data
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Visualize results for each c-value (comparing shapes)
    for c_value in crystallinity_values:
        if results_by_c[c_value]:
            visualize_c_value_results(results_by_c[c_value], c_value)
    
    # Visualize results for each shape (comparing c-values)
    for shape_id in range(1, num_shapes + 1):
        if results_by_shape[shape_id]:
            visualize_shape_results(results_by_shape[shape_id], shape_id)
            create_heatmap(results_by_shape[shape_id], shape_id)
    
    print("All simulations and visualizations complete!")

if __name__ == "__main__":
    main()
