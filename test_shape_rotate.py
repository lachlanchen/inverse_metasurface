#!/usr/bin/env python3

import os
import glob
import csv
import time
import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def generate_random_polygon(n_verts=4):
    """
    Generate a random polygon with n_verts points in a range
    that ensures a 'reasonable' shape. You can also hardcode your shape if you wish.
    Returns a string like 'x1,y1;x2,y2;...'
    """
    # Example approach: random angles, random radius in [0.1, 0.2]
    angles = np.sort(np.random.uniform(0, 2*math.pi, n_verts))
    points = []
    for theta in angles:
        r = np.random.uniform(0.1, 0.2)
        x = r*math.cos(theta)
        y = r*math.sin(theta)
        points.append((x, y))
    poly_str = ";".join([f"{p[0]:.6f},{p[1]:.6f}" for p in points])
    return poly_str

def run_s4_for_polygon_and_c(poly_str, c_val, lua_script="metasurface_fixed_shape_and_c_value.lua", verbose=False):
    """
    Runs S4 with the given polygon string and c value, using the specified Lua script.
    Returns the path to the results CSV if found, or None otherwise.
    """
    cmd = f'../build/S4 -a "{poly_str} -c {c_val} -v -s" {lua_script}'
    if verbose:
        print("[INFO] Running S4:", cmd)
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[ERROR] S4 run failed for c={c_val:.2f}!")
        print("=== STDOUT ===")
        print(proc.stdout)
        print("=== STDERR ===")
        print(proc.stderr)
        return None

    # Try to parse the "Saved to <path>" line from stdout
    saved_path = None
    for line in proc.stdout.splitlines():
        if "Saved to" in line:
            saved_path = line.split("Saved to", 1)[1].strip()
            break

    # If that did not work, fallback: look for pattern results/fixed_shape_cX_*.csv
    if saved_path is None:
        pattern = f"results/fixed_shape_c{c_val:.1f}_*.csv"
        found = glob.glob(pattern)
        if found:
            # pick the newest
            saved_path = max(found, key=os.path.getctime)

    if saved_path and os.path.isfile(saved_path):
        if verbose:
            print(f"[INFO] Found CSV => {saved_path}")
        return saved_path
    else:
        if verbose:
            print(f"[WARN] Could not detect any output CSV for c={c_val}!")
        return None

def read_s4_csv(csv_path):
    """
    Reads the CSV file from S4 results. Expects columns with headers:
    'wavelength_um', 'R', 'T', etc.
    Returns (list_of_wavelengths, list_of_R, list_of_T).
    """
    lam_list, R_list, T_list = [], [], []
    if (csv_path is None) or (not os.path.isfile(csv_path)):
        return lam_list, R_list, T_list

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lam = float(row["wavelength_um"])
                R_  = float(row["R"])
                T_  = float(row["T"])
                lam_list.append(lam)
                R_list.append(R_)
                T_list.append(T_)
            except:
                continue

    # Sort by wavelength
    data = sorted(zip(lam_list, R_list, T_list), key=lambda x: x[0])
    lam_list = [d[0] for d in data]
    R_list   = [d[1] for d in data]
    T_list   = [d[2] for d in data]
    return lam_list, R_list, T_list

def main():
    # 1) Generate or define a single polygon:
    #    You can either generate a random one or hardcode your shape:
    # poly_str = "0.2,0.2; -0.2,0.2; -0.2,-0.2; 0.2,-0.2"
    poly_str = generate_random_polygon(n_verts=4)
    print("[INFO] Using polygon:", poly_str)

    # 2) c-values from 0.0 to 1.0
    c_values = [round(x, 1) for x in np.linspace(0, 1, 11)]
    print("[INFO] c-values:", c_values)

    # 3) For each c, run S4
    results = {}
    for c_val in c_values:
        csv_path = run_s4_for_polygon_and_c(poly_str, c_val, verbose=True)
        lam_list, R_list, T_list = read_s4_csv(csv_path)
        results[c_val] = (lam_list, R_list, T_list)

    # 4) Plot each c's spectrum (R, T, R+T) in a 1x3 subplot
    fig, axes = plt.subplots(1, 3, figsize=(15,5), sharex=True)

    # pick colormap
    color_map = plt.cm.get_cmap("viridis", len(c_values))

    for i, c_val in enumerate(c_values):
        lam, R, T = results[c_val]
        if not lam:
            continue  # skip if no data
        color = color_map(i)
        label = f"c={c_val:.1f}"
        # R
        axes[0].plot(lam, R, color=color, label=label)
        # T
        axes[1].plot(lam, T, color=color, label=label)
        # R+T
        RT = [abs(r)+t for r, t in zip(R, T)]
        axes[2].plot(lam, RT, color=color, label=label)

    axes[0].set_title("Reflection (R)")
    axes[1].set_title("Transmission (T)")
    axes[2].set_title("R + T")
    for ax in axes:
        ax.set_xlabel("Wavelength (um)")
        ax.legend(fontsize="small")
    axes[0].set_ylabel("Magnitude")

    out_png = "fuck_correct_single_shape_spectrum.png"
    plt.savefig(out_png, dpi=150)
    print(f"[INFO] Plot saved to {out_png}")
    plt.show()

if __name__ == "__main__":
    main()

