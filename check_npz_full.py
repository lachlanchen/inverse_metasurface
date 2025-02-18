#!/usr/bin/env python3
import os
import sys
import csv
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch

############################################
# S4 and Utility Functions (using os.popen)
############################################

def run_s4_for_c(polygon_str, c_val):
    """
    Run the S4 binary with the given polygon string and c value
    using os.popen() instead of subprocess.
    Returns the path to the CSV file with results (or None if not found).
    """
    cmd = f'../build/S4 -a "{polygon_str} -c {c_val} -v -s" metasurface_fixed_shape_and_c_value.lua'
    print(f"[DEBUG] S4 command: {cmd}")
    # Use os.popen() to run the command and capture its output
    output = os.popen(cmd).read()
    if not output:
        print("[ERROR] S4 produced no output.")
        return None
    saved_path = None
    for line in output.splitlines():
        if "Saved to " in line:
            saved_path = line.split("Saved to",1)[1].strip()
            break
    return saved_path

def read_results_csv(csv_path):
    """
    Reads the CSV output from S4 and returns lists of wavelengths, reflectance, and transmission.
    """
    wv, Rv, Tv = [], [], []
    if not csv_path or not os.path.exists(csv_path):
        return wv, Rv, Tv
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lam = float(row["wavelength_um"])
                R_ = float(row["R"])
                T_ = float(row["T"])
                wv.append(lam)
                Rv.append(R_)
                Tv.append(T_)
            except Exception:
                pass
    data = sorted(zip(wv, Rv, Tv), key=lambda x: x[0])
    wv = [d[0] for d in data]
    Rv = [d[1] for d in data]
    Tv = [d[2] for d in data]
    return wv, Rv, Tv

def replicate_c4(points):
    """
    Given an array of Q1 vertices (Nx2), replicates them using C4 symmetry.
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
    angles = np.arctan2(points[:,1]-cy, points[:,0]-cx)
    idx = np.argsort(angles)
    return points[idx]

def polygon_to_string(polygon):
    """
    Converts an array of vertices (Nx2) to a semicolon-separated string.
    """
    return ";".join([f"{v[0]:.6f},{v[1]:.6f}" for v in polygon])

############################################
# NPZ Dataset Functions
############################################

def load_npz(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    return data["uids"], data["spectra"], data["shapes"]

############################################
# Main Function
############################################

def main():
    npz_file = "preprocessed_data.npz"  # Adjust path if needed.
    uids, spectra, shapes = load_npz(npz_file)
    
    # Print first sample information.
    print("First UID:")
    print(uids[0])
    print("\nFirst Spectrum (shape: {}):".format(spectra[0].shape))
    for i, spec_row in enumerate(spectra[0]):
        print(f"Row {i}:")
        print(spec_row)
    print("\nFirst Shape (shape: {}):".format(shapes[0].shape))
    print(shapes[0])
    
    # Select samples: first sample and three random others.
    total_samples = uids.shape[0]
    sample_indices = [0] + random.sample(range(1, total_samples), 3)
    
    # Create an output folder with current datetime to avoid clutter.
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("check_npz_plots", dt_str)
    os.makedirs(out_dir, exist_ok=True)
    
    # Define a uniform x-axis (1 to 100) for plotting.
    x_axis = np.arange(1, 101)
    
    # For each selected sample:
    for idx in sample_indices:
        uid = uids[idx]
        spec_sample = spectra[idx]   # shape: (11, 100) – 11 spectra
        shape_sample = shapes[idx]   # shape: (4, 3)
        print(f"\n[INFO] Processing sample UID: {uid}")
        
        # Reconstruct polygon from GT shape.
        valid = shape_sample[:,0] > 0.5
        if not np.any(valid):
            print(f"[WARN] No valid vertices for UID {uid}. Skipping sample.")
            continue
        q1 = shape_sample[valid, 1:3]
        full_polygon = replicate_c4(q1)
        full_polygon = sort_points_by_angle(full_polygon)
        polygon_str = polygon_to_string(full_polygon)
        print(f"[DEBUG] UID {uid} - Polygon string: {polygon_str}")
        
        # Prepare a figure with 11 subplots (one for each spectrum row)
        fig, axes = plt.subplots(11, 1, figsize=(10, 3*11), sharex=True)
        if axes.ndim == 1:
            axes = axes  # already 1D
        
        # For each c value (assumed c = i/10 for i=0,...,10)
        for i in range(11):
            c_val = i / 10.0
            ax = axes[i]
            # Plot the NPZ spectrum for row i.
            npz_spec = spec_sample[i]  # 1D array of length 100.
            ax.plot(x_axis, npz_spec, 'b-', label=f"NPZ Spectrum (c={c_val:.2f})")
            
            # Run S4 for the same c value.
            results_csv = run_s4_for_c(polygon_str, c_val)
            if results_csv is not None:
                wv, R, T = read_results_csv(results_csv)
                # Here we force a uniform x-axis (1 to 100).
                ax.plot(x_axis, R, 'g--', label=f"S4 Reflectance (c={c_val:.2f})")
            else:
                ax.text(0.5, 0.5, "S4 failed", transform=ax.transAxes, color='red')
            
            ax.set_ylabel("Reflectance")
            ax.set_title(f"c = {c_val:.2f}")
            ax.legend(fontsize=8)
        
        plt.xlabel("Uniform X-axis (1-100)")
        plt.suptitle(f"UID {uid}: Comparison of 11 Spectra (NPZ vs S4)", fontsize=16)
        outpng = os.path.join(out_dir, f"comparison_{uid}.png")
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(outpng, dpi=100)
        plt.close(fig)
        print(f"[INFO] Saved figure for UID {uid} to {outpng}")

if __name__ == "__main__":
    main()

