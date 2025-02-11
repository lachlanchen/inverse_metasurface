#!/usr/bin/env python3
"""
c4_three_column_plot.py

Produces an 11x3 grid of subplots (33 total).
Each row => one c value in [0..1..0.1].
Three columns in each row:
    - Left column: 4 lines for 4 shapes (nQ=1,2,3,4).
    - Middle column: 11 lines (rotations from 0.. pi/(2*nQ)), using nQ=4 shape.
    - Right column: 11 lines (scale from 0.5..1.5), again using nQ=4 shape.

We parse the reflection R from the S4 output CSV and plot (wavelength, R).
No axes, no spines, no legend.
"""

import os
import glob
import csv
import math
import time
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

###############################################################################
# 1) Generate C4-symmetric polygon
###############################################################################
def generate_c4_polygon(nQ=3, rmin=0.05, rmax=0.65, seed=None):
    """
    Returns (poly_str, points_list) for a C4-symmetric shape with nQ points
    in [0, pi/2], radius in [rmin, rmax], then replicated to all 4 quadrants,
    then shifted by (+0.5, +0.5).
    If 'seed' is given, we set np.random.seed(seed) to be reproducible.
    """
    if seed is not None:
        np.random.seed(seed)

    angles_1Q = np.sort(np.random.uniform(0.0, math.pi/2, nQ))
    radii_1Q  = np.random.uniform(rmin, rmax, nQ)

    pts_1Q = []
    for theta, r in zip(angles_1Q, radii_1Q):
        x = r*math.cos(theta)
        y = r*math.sin(theta)
        pts_1Q.append((x, y))

    def rotate_90_deg(x, y):
        return (-y, x)

    # Quadrant 1
    all_points = list(pts_1Q)
    # Quadrant 2
    pts_2Q = [rotate_90_deg(x, y) for (x,y) in pts_1Q]
    pts_2Q.sort(key=lambda p: math.atan2(p[1], p[0]))
    all_points.extend(pts_2Q)
    # Quadrant 3
    pts_3Q = [rotate_90_deg(px, py) for (px,py) in pts_2Q]
    pts_3Q.sort(key=lambda p: math.atan2(p[1], p[0]))
    all_points.extend(pts_3Q)
    # Quadrant 4
    pts_4Q = [rotate_90_deg(px, py) for (px,py) in pts_3Q]
    pts_4Q.sort(key=lambda p: math.atan2(p[1], p[0]))
    all_points.extend(pts_4Q)

    shifted = [(p[0]+0.5, p[1]+0.5) for p in all_points]
    poly_str = ";".join(f"{p[0]:.6f},{p[1]:.6f}" for p in shifted)
    return poly_str, shifted

###############################################################################
# 2) S4 Runner & CSV Parser
###############################################################################
def run_s4(poly_str, c_val, verbose=False, extra_args=""):
    """
    Calls ../build/S4 with metasurface_fixed_shape_and_c_value.lua
    and polygon, c, plus any extra args (like -rotate angle or -scale factor if needed).
    Returns path to CSV or None if fail.
    We expect the Lua script to print "Saved to <CSV>" to stdout.
    """
    lua_script = "metasurface_fixed_shape_and_c_value.lua"
    cmd = f'../build/S4 -a "{poly_str} -c {c_val} -v -s {extra_args}" {lua_script}'
    if verbose:
        print("[INFO] Running S4:", cmd)
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[ERROR] S4 run failed (c={c_val})!")
        print("--- STDOUT ---\n", proc.stdout)
        print("--- STDERR ---\n", proc.stderr)
        return None

    saved_path = None
    for line in proc.stdout.splitlines():
        if "Saved to" in line:
            saved_path = line.split("Saved to", 1)[1].strip()
            break

    # Fallback
    if saved_path is None:
        pattern = f"results/fixed_shape_c{c_val:.1f}_*.csv"
        found = glob.glob(pattern)
        if found:
            saved_path = max(found, key=os.path.getctime)

    if saved_path and os.path.isfile(saved_path):
        if verbose:
            print(f"[INFO] Found CSV => {saved_path}")
        return saved_path
    else:
        if verbose:
            print(f"[WARN] Could not find CSV for c={c_val}!")
        return None

def read_s4_csv(csv_path):
    """
    Read CSV from S4. Return (lam, R, T).
    """
    if not csv_path or not os.path.isfile(csv_path):
        return [], [], []
    lam_list, R_list, T_list = [], [], []
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
                pass
    # Sort
    data = sorted(zip(lam_list, R_list, T_list), key=lambda x: x[0])
    lam_list = [d[0] for d in data]
    R_list   = [d[1] for d in data]
    T_list   = [d[2] for d in data]
    return lam_list, R_list, T_list

###############################################################################
# 3) Main
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="More logs.")
    args = parser.parse_args()
    verbose = args.verbose

    # Minimal style: no spines, no ticks, no grid, no axis
    sns.set_style("white")
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.spines.top"]    = False
    plt.rcParams["axes.spines.right"]  = False
    plt.rcParams["axes.spines.left"]   = False
    plt.rcParams["axes.spines.bottom"] = False

    # We'll do 11 rows, each row => one c in [0..1..0.1]
    c_values = [round(x,1) for x in np.linspace(0,1,11)]
    nrows = len(c_values)
    ncols = 3

    # Create figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(15, 2.5*nrows),
                             sharex=False, sharey=False)

    # For each row => c
    #   Left col => 4 lines, shapes with nQ=1,2,3,4
    #   Middle col => 11 lines, rotate the nQ=4 shape in [0.. pi/(2*4)]
    #   Right col => 11 lines, scale the nQ=4 shape from 0.5..1.5
    # We'll do reflection R vs. wavelength in all subplots.

    # Pre-build the shapes for nQ=1..4 (just once).
    # We'll pick a fixed seed so they won't keep changing each call.
    shapes_nQ = {}
    for nQ in [1,2,3,4]:
        # poly_str, pts = generate_c4_polygon(nQ=nQ, rmin=0.05, rmax=0.2, seed=(1234+nQ))
        poly_str, pts = generate_c4_polygon(nQ=nQ, rmin=0.05, rmax=0.2, seed=(8888+nQ))
        shapes_nQ[nQ] = (poly_str, pts)

    # For the middle & right columns, we specifically use nQ=4 shape
    # Middle col => 11 angles
    nQ_4_poly_str, nQ_4_pts = shapes_nQ[4]
    angle_vals = np.linspace(0, math.pi/(2*4), 11)

    # Right col => 11 scale factors
    scale_vals = np.linspace(0.5, 1.5, 11)

    # We'll define a helper for rotating or scaling the "nQ=4" shape
    def rotate_polygon(points, theta):
        rpts = []
        for (x,y) in points:
            xr = x*math.cos(theta) - y*math.sin(theta)
            yr = x*math.sin(theta) + y*math.cos(theta)
            rpts.append((xr, yr))
        return ";".join(f"{xx:.6f},{yy:.6f}" for (xx,yy) in rpts)

    def scale_polygon(points, factor):
        spts = []
        for (x,y) in points:
            spts.append((factor*x, factor*y))
        return ";".join(f"{xx:.6f},{yy:.6f}" for (xx,yy) in spts)

    # We'll pick a colormap for the lines in middle & right columns
    # left col only has 4 lines, we can pick 4 distinct colors ourselves
    lines_4_colors = ["red", "green", "blue", "purple"]
    # rot_cmap = plt.cm.get_cmap("cividis", len(angle_vals))
    # scl_cmap = plt.cm.get_cmap("plasma", len(scale_vals))
    rot_cmap = plt.cm.get_cmap("viridis", len(angle_vals))
    scl_cmap = plt.cm.get_cmap("viridis", len(scale_vals))

    for row_i, c_val in enumerate(c_values):
        # Extract subplots
        ax_left  = axes[row_i][0]
        ax_mid   = axes[row_i][1]
        ax_right = axes[row_i][2]

        # Turn off all axes
        ax_left.set_axis_off()
        ax_mid.set_axis_off()
        ax_right.set_axis_off()

        # --- 1) Left col => 4 lines for nQ=1..4
        for idx, nQ_ in enumerate([1,2,3,4]):
            shape_str, _ = shapes_nQ[nQ_]
            csv_path = run_s4(shape_str, c_val, verbose=verbose)
            lam, R, T = read_s4_csv(csv_path)
            if lam:
                ax_left.plot(lam, R, color=lines_4_colors[idx], linewidth=1.5)

        # --- 2) Middle col => 11 lines for rotation of nQ=4 shape
        for iang, angle in enumerate(angle_vals):
            shape_str = rotate_polygon(nQ_4_pts, angle)
            csv_path = run_s4(shape_str, c_val, verbose=verbose)
            lam, R, T = read_s4_csv(csv_path)
            if lam:
                clr = rot_cmap(iang)
                ax_mid.plot(lam, R, color=clr, linewidth=1.2)

        # --- 3) Right col => 11 lines for scale factors on the nQ=4 shape
        for iscl, scl in enumerate(scale_vals):
            shape_str = scale_polygon(nQ_4_pts, scl)
            csv_path = run_s4(shape_str, c_val, verbose=verbose)
            lam, R, T = read_s4_csv(csv_path)
            if lam:
                clr = scl_cmap(iscl)
                ax_right.plot(lam, R, color=clr, linewidth=1.2)

    fig.tight_layout()
    out_png = "three_col_plot_minimal_axes.png"
    plt.savefig(out_png, dpi=150)
    print(f"[INFO] Final figure saved to {out_png}")
    plt.show()


if __name__ == "__main__":
    main()

