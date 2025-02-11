#!/usr/bin/env python3
"""
scale_shape_spectra_by_c.py

Generates a C4-symmetric polygon with 'nq' points, scales it from 0.5× to 1.5×,
and runs S4 for c values in [0..1..0.1]. Plots an 11×3 figure:
(rows=c, columns=R/T/R+T, each subplot has 11 lines for scale factors).
"""

import os
import glob
import csv
import time
import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

def generate_c4_symmetric_polygon(nQ=3, rmin=0.05, rmax=0.65):
    angles_1Q = np.sort(np.random.uniform(0.0, math.pi/2, nQ))
    radii_1Q  = np.random.uniform(rmin, rmax, nQ)
    pts_1Q = []
    for theta, r in zip(angles_1Q, radii_1Q):
        x = r*math.cos(theta)
        y = r*math.sin(theta)
        pts_1Q.append((x, y))

    def rotate_90_deg(x, y):
        return (-y, x)

    all_points = list(pts_1Q)
    # quadrant 2
    pts_2Q = [(rotate_90_deg(x,y)) for (x,y) in pts_1Q]
    pts_2Q.sort(key=lambda p: math.atan2(p[1], p[0]))
    all_points.extend(pts_2Q)
    # quadrant 3
    pts_3Q = [(rotate_90_deg(x2,y2)) for (x2,y2) in pts_2Q]
    pts_3Q.sort(key=lambda p: math.atan2(p[1], p[0]))
    all_points.extend(pts_3Q)
    # quadrant 4
    pts_4Q = [(rotate_90_deg(x3,y3)) for (x3,y3) in pts_3Q]
    pts_4Q.sort(key=lambda p: math.atan2(p[1], p[0]))
    all_points.extend(pts_4Q)

    shifted_points = [(x+0.5, y+0.5) for (x,y) in all_points]
    poly_str = ";".join(f"{p[0]:.6f},{p[1]:.6f}" for p in shifted_points)
    return poly_str, shifted_points

def run_s4_for_polygon_and_c(scaled_poly_str, c_val, lua_script="metasurface_fixed_shape_and_c_value.lua", verbose=False):
    cmd = f'../build/S4 -a "{scaled_poly_str} -c {c_val} -v -s" {lua_script}'
    if verbose:
        print("[INFO] Running S4:", cmd)
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[ERROR] S4 run failed for c={c_val:.2f}!")
        print("=== STDOUT ===")
        print(proc.stdout)
        print("=== STDERR ===")
        print(proc.stderr)
        return None

    saved_path = None
    for line in proc.stdout.splitlines():
        if "Saved to" in line:
            saved_path = line.split("Saved to", 1)[1].strip()
            break

    if saved_path is None:
        pattern = f"results/fixed_shape_c{c_val:.1f}_*.csv"
        found   = glob.glob(pattern)
        if found:
            saved_path = max(found, key=os.path.getctime)

    if saved_path and os.path.isfile(saved_path):
        if verbose:
            print(f"[INFO] Found CSV => {saved_path}")
        return saved_path
    else:
        if verbose:
            print(f"[WARN] Could not detect CSV for c={c_val}!")
        return None

def read_s4_csv(csv_path):
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
                pass

    data = sorted(zip(lam_list, R_list, T_list), key=lambda x: x[0])
    lam_list = [d[0] for d in data]
    R_list   = [d[1] for d in data]
    T_list   = [d[2] for d in data]
    return lam_list, R_list, T_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nq", type=int, default=3, help="Number of points in the first quadrant.")
    parser.add_argument("--verbose", action="store_true", help="Print debug info.")
    args = parser.parse_args()
    nQ = args.nq
    verbose = args.verbose

    # Use Seaborn to set a nice style:
    sns.set_theme(style="whitegrid")

    base_poly_str, base_points = generate_c4_symmetric_polygon(nQ, 0.05, 0.65)
    print(f"[INFO] Created a C4-symmetric polygon with {4*nQ} vertices.")
    print("Base polygon string:\n", base_poly_str)

    # Quick plot of the base shape
    fig_shape, ax_shape = plt.subplots(figsize=(4,4))
    xs = [p[0] for p in base_points]
    ys = [p[1] for p in base_points]
    xs_closed = xs + [xs[0]]
    ys_closed = ys + [ys[0]]
    ax_shape.plot(xs_closed, ys_closed, 'o-', color='darkblue', markersize=4)
    ax_shape.set_aspect('equal', adjustable='box')
    ax_shape.set_title("Base C4-Symmetric Polygon")
    plt.savefig("shape_c4_symmetric_base.png", dpi=150)
    print("[INFO] Base shape plot saved to shape_c4_symmetric_base.png")
    plt.show(block=False)

    # Scale factors & c values
    scale_values = [round(s, 1) for s in np.linspace(0.5, 1.5, 11)]
    c_values     = [round(c, 1) for c in np.linspace(0.0, 1.0, 11)]
    print("[INFO] scale factors:", scale_values)
    print("[INFO] c values:     ", c_values)

    results = {}

    def scale_polygon_string(points, factor):
        return ";".join(f"{factor*x:.6f},{factor*y:.6f}" for (x,y) in points)

    # Run S4 for each (scale, c)
    for s in scale_values:
        for c in c_values:
            scaled_poly_str = scale_polygon_string(base_points, s)
            csv_path = run_s4_for_polygon_and_c(scaled_poly_str, c, verbose=verbose)
            lam_list, R_list, T_list = read_s4_csv(csv_path)
            results[(s,c)] = (lam_list, R_list, T_list)
            time.sleep(0.05)

    # Plot 11x3 => rows=c, columns=(R,T,R+T), lines=scale
    fig, axes = plt.subplots(nrows=len(c_values), ncols=3,
                             figsize=(14, 3*len(c_values)),
                             sharex=True, sharey=False)

    scale_cmap = plt.cm.get_cmap("plasma", len(scale_values))

    for row_i, c in enumerate(c_values):
        ax_R  = axes[row_i, 0]
        ax_T  = axes[row_i, 1]
        ax_RT = axes[row_i, 2]
        for j, s in enumerate(scale_values):
            lam_list, R_list, T_list = results[(s,c)]
            if not lam_list:
                continue
            color = scale_cmap(j)
            label = f"scale={s}"
            ax_R.plot(lam_list, R_list, color=color, label=label)
            ax_T.plot(lam_list, T_list, color=color, label=label)
            rt = [abs(r)+t for (r,t) in zip(R_list, T_list)]
            ax_RT.plot(lam_list, rt, color=color, label=label)

        ax_R.set_title(f"c={c}: R")
        ax_T.set_title(f"c={c}: T")
        ax_RT.set_title(f"c={c}: R+T")

        if row_i == len(c_values) - 1:
            ax_R.set_xlabel("Wavelength (um)")
            ax_T.set_xlabel("Wavelength (um)")
            ax_RT.set_xlabel("Wavelength (um)")

        if row_i == 0:
            ax_R.set_ylabel("Reflectance R")
            ax_T.set_ylabel("Transmission T")
            ax_RT.set_ylabel("R + T")
            ax_R.legend(fontsize="x-small")
            ax_T.legend(fontsize="x-small")
            ax_RT.legend(fontsize="x-small")

    fig.suptitle("Scaling the C4-Symmetric Polygon (0.5x to 1.5x) vs. c Value", fontsize=14, y=1.02)
    fig.tight_layout()
    plt.savefig("scaled_shape_spectra_by_c.png", dpi=150)
    print("[INFO] Final plot saved to scaled_shape_spectra_by_c.png")
    plt.show()

if __name__ == "__main__":
    main()

