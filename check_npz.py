#!/usr/bin/env python3
import os
import sys
import csv
import subprocess
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch

############################################
# S4 and Utility Functions
############################################

def run_s4_for_c(polygon_str, c_val):
    """
    Run the S4 binary with the given polygon string and c value.
    Returns the path to the CSV file with results (or None if failed).
    """
    cmd = f'../build/S4 -a "{polygon_str} -c {c_val} -v -s" metasurface_fixed_shape_and_c_value.lua'
    print(f"[DEBUG] S4 command: {cmd}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[ERROR] S4 run failed!")
        print("=== STDOUT ===")
        print(proc.stdout)
        print("=== STDERR ===")
        print(proc.stderr)
        return None
    saved_path = None
    for line in proc.stdout.splitlines():
        if "Saved to " in line:
            saved_path = line.split("Saved to",1)[1].strip()
            break
    return saved_path

def read_results_csv(csv_path):
    """
    Reads the CSV output from S4 and returns three lists:
    wavelengths, reflectance, and transmission.
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
    Given an array of Q1 vertices (Nx2), replicate them using C4 symmetry.
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
    Sort a set of 2D points by their polar angle around the centroid.
    """
    if len(points) < 3:
        return points
    cx, cy = points.mean(axis=0)
    angles = np.arctan2(points[:,1]-cy, points[:,0]-cx)
    idx = np.argsort(angles)
    return points[idx]

def polygon_to_string(polygon):
    """
    Convert an array of vertices (Nx2) to a semicolon-separated string.
    """
    return ";".join([f"{v[0]:.6f},{v[1]:.6f}" for v in polygon])

############################################
# Main: Load NPZ, Print First Sample, Run S4 & Plot
############################################

def main():
    npz_file = "preprocessed_data.npz"  # Adjust if needed.
    data = np.load(npz_file, allow_pickle=True)
    
    uids = data["uids"]
    spectra = data["spectra"]   # shape: (320000, 11, 100)
    shapes  = data["shapes"]    # shape: (320000, 4, 3)
    
    # Print the first sample’s information.
    print("First UID:")
    print(uids[0])
    print("\nFirst Spectrum (shape: {}):".format(spectra[0].shape))
    for i, spec_row in enumerate(spectra[0]):
        print(f"Row {i}:")
        print(spec_row)
    print("\nFirst Shape (shape: {}):".format(shapes[0].shape))
    print(shapes[0])
    
    # Use the first sample.
    spec_sample = spectra[0]   # (11, 100) – 11 spectra corresponding to different c values.
    shape_sample = shapes[0]   # (4, 3)
    
    # Reconstruct the polygon from the ground-truth shape.
    # We assume that the first column is a presence flag (1 means valid).
    valid = shape_sample[:,0] > 0.5
    if not np.any(valid):
        print("No valid vertices found in the shape.")
        return
    q1 = shape_sample[valid, 1:3]  # valid Q1 vertices (x,y)
    # Replicate via C4 and sort.
    full_polygon = replicate_c4(q1)
    full_polygon = sort_points_by_angle(full_polygon)
    polygon_str = polygon_to_string(full_polygon)
    print("\nReconstructed polygon string (from GT shape):")
    print(polygon_str)
    
    # Define a uniform x-axis for plotting (1 to 100)
    x_axis = np.arange(1, 101)
    
    # Prepare a figure with 11 subplots (one for each spectrum row)
    fig, axes = plt.subplots(11, 1, figsize=(10, 3*11), sharex=True)
    # For each c value corresponding to the 11 spectra,
    # assume c = i/10 for i=0,...,10.
    for i in range(11):
        c_val = i / 10.0
        ax = axes[i]
        # Plot the NPZ spectrum for row i.
        npz_spec = spec_sample[i]  # shape (100,)
        ax.plot(x_axis, npz_spec, 'b-', label=f"NPZ Spectrum (c={c_val:.2f})")
        
        # Run S4 for the same c value using the same polygon string.
        results_csv = run_s4_for_c(polygon_str, c_val)
        if results_csv is not None:
            wv, R, T = read_results_csv(results_csv)
            # For comparison, we force a uniform x-axis (1 to 100) by simply using our x_axis.
            # (In practice, you might need to interpolate if lengths differ.)
            ax.plot(x_axis, R, 'g--', label=f"S4 Reflectance (c={c_val:.2f})")
        else:
            ax.text(0.5, 0.5, "S4 failed", transform=ax.transAxes, color='red')
        
        ax.set_ylabel("Reflectance")
        ax.set_title(f"c = {c_val:.2f}")
        ax.legend(fontsize=8)
    
    plt.xlabel("Uniform X-axis (1-100)")
    plt.suptitle("Comparison of 11 Spectra from NPZ vs. S4 Results", fontsize=16)
    outpng = f"npz_vs_s4_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(outpng, dpi=250)
    print(f"\n[INFO] Figure saved to {outpng}")
    plt.show()

if __name__ == "__main__":
    main()

