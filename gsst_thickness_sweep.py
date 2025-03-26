#!/usr/bin/env python3
"""
gsst_thickness_sweep.py

Tests different GSST thicknesses from 100nm to 1000nm with a 0.2nm circle geometry.
Uses the S4 simulator with the metasurface_gsst_nir.lua script.

Requirements:
- S4 simulator executable at ../build/S4
- metasurface_gsst_nir.lua in the current directory
- gsst_partial_crys_data folder with the c-value data files
"""

import os
import subprocess
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time

# Configuration
c_value = 0.5  # crystallinity value (0.0 to 1.0)
thickness_values = np.linspace(100, 1000, 10, dtype=int)  # 100, 200, ..., 1000 nm
s4_path = "../build/S4"
lua_script = "metasurface_gsst_nir.lua"

# Create a C4 symmetric circle with radius 0.2
def generate_c4_circle(r=0.2, points_per_quadrant=2):
    """Generate a C4 symmetric circle with the specified radius.
    Only defines points in the first quadrant, which will be replicated by S4."""
    points = []
    
    # Add points in the first quadrant
    for i in range(points_per_quadrant):
        angle = (i * (np.pi/2)) / (points_per_quadrant - 1)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        points.append((x, y))
    
    # Return as a string format that S4 can understand
    return ";".join(f"{p[0]:.6f},{p[1]:.6f}" for p in points)

def run_s4_simulation(thickness):
    """Run a single S4 simulation for the given thickness."""
    circle_str = generate_c4_circle()
    
    # Construct the command
    cmd = f'{s4_path} -a "{circle_str} -c {c_value} -t {thickness} -v -s" {lua_script}'
    
    print(f"Running thickness = {thickness}nm...")
    start_time = time.time()
    
    # Run the command
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        print(f"Error running simulation for thickness={thickness}nm")
        print("STDERR:", result.stderr)
        return thickness, None
    
    # Find the output file
    output_file = None
    for line in result.stdout.splitlines():
        if "Saved to" in line:
            output_file = line.split("Saved to", 1)[1].strip()
            break
    
    elapsed = time.time() - start_time
    print(f"Completed thickness = {thickness}nm in {elapsed:.1f} seconds")
    
    return thickness, output_file

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
    
    return data

def main():
    """Main function to run simulations and create plots."""
    # Create results directory
    os.makedirs("thickness_results", exist_ok=True)
    
    # Run simulations in parallel
    results = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_s4_simulation, t): t for t in thickness_values}
        for future in concurrent.futures.as_completed(futures):
            thickness, output_file = future.result()
            if output_file:
                data = load_results(output_file)
                if data:
                    results[thickness] = data
    
    if not results:
        print("No results to plot!")
        return
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Reflection vs wavelength for each thickness
    plt.subplot(2, 1, 1)
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(thickness_values)))
    
    for i, thickness in enumerate(sorted(results.keys())):
        data = results[thickness]
        plt.plot(data['wavelength'], data['R'], color=colors[i], label=f"{thickness}nm")
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflection')
    plt.title(f'GSST Reflection Spectrum (c={c_value}) for Different Thicknesses')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Transmission vs wavelength for each thickness
    plt.subplot(2, 1, 2)
    for i, thickness in enumerate(sorted(results.keys())):
        data = results[thickness]
        plt.plot(data['wavelength'], data['T'], color=colors[i], label=f"{thickness}nm")
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission')
    plt.title(f'GSST Transmission Spectrum (c={c_value}) for Different Thicknesses')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"thickness_results/gsst_thickness_sweep_c{c_value}.png", dpi=300)
    plt.close()
    
    # Create heatmap plots
    sorted_thicknesses = sorted(results.keys())
    wavelengths = results[sorted_thicknesses[0]]['wavelength']
    
    # Extract data for heatmaps
    heatmap_data_R = np.zeros((len(sorted_thicknesses), len(wavelengths)))
    heatmap_data_T = np.zeros((len(sorted_thicknesses), len(wavelengths)))
    
    for i, thickness in enumerate(sorted_thicknesses):
        data = results[thickness]
        heatmap_data_R[i, :] = data['R']
        heatmap_data_T[i, :] = data['T']
    
    # Plot the heatmaps
    plt.figure(figsize=(15, 10))
    
    # Reflection heatmap
    plt.subplot(2, 1, 1)
    im = plt.imshow(heatmap_data_R, aspect='auto', origin='lower', 
                   extent=[min(wavelengths), max(wavelengths), min(sorted_thicknesses), max(sorted_thicknesses)],
                   cmap='viridis')
    plt.colorbar(im, label='Reflection')
    plt.ylabel('Thickness (nm)')
    plt.title(f'GSST Reflection Heatmap (c={c_value})')
    
    # Transmission heatmap
    plt.subplot(2, 1, 2)
    im = plt.imshow(heatmap_data_T, aspect='auto', origin='lower',
                   extent=[min(wavelengths), max(wavelengths), min(sorted_thicknesses), max(sorted_thicknesses)],
                   cmap='viridis')
    plt.colorbar(im, label='Transmission')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Thickness (nm)')
    plt.title(f'GSST Transmission Heatmap (c={c_value})')
    
    plt.tight_layout()
    plt.savefig(f"thickness_results/gsst_thickness_heatmap_c{c_value}.png", dpi=300)
    plt.close()
    
    print("All simulations and plotting complete!")

if __name__ == "__main__":
    main()
