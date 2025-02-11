#!/usr/bin/env python3
"""
c4_shape_rotation_spectra_by_c.py

Generates a C4-symmetric shape with 'nq' points in the first quadrant,
replicates them to achieve 4-fold symmetry, shifts them by (0.5,0.5).
Then we define 10 rotation angles and 11 c values. For each combination
(angle, c), we run S4 and store Reflection and Transmission vs wavelength.

Finally, we plot an 11×3 grid:
  - 11 rows (one per c value)
  - 3 columns (R, T, R+T)
In each subplot, we overlay lines for all 10 angles.
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


###############################################################################
# 1) Construct a C4-symmetric shape
###############################################################################
def generate_c4_symmetric_polygon(nQ=3, rmin=0.05, rmax=0.65):
    """
    Generate a polygon that has 4-fold (C4) symmetry:
      - We define nQ points in the first quadrant with angles in [0, pi/2].
      - Each point has a random radius in [rmin, rmax].
      - Then replicate these points by +90°, +180°, +270°.
      - Finally shift everything by (+0.5, +0.5).

    Returns:
      (poly_str, points_list)
        poly_str: "x1,y1;x2,y2;..."
        points_list: list of (x, y)
    """
    angles_1Q = np.sort(np.random.uniform(0.0, math.pi/2, nQ))
    radii_1Q  = np.random.uniform(rmin, rmax, nQ)

    pts_1Q = []
    for theta, r in zip(angles_1Q, radii_1Q):
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        pts_1Q.append((x, y))

    def rotate_90_deg(x, y):
        return (-y, x)

    # quadrant 1
    all_points = list(pts_1Q)

    # quadrant 2
    pts_2Q = []
    for (x, y) in pts_1Q:
        x2, y2 = rotate_90_deg(x, y)
        pts_2Q.append((x2, y2))
    pts_2Q.sort(key=lambda p: math.atan2(p[1], p[0]))
    all_points.extend(pts_2Q)

    # quadrant 3
    pts_3Q = []
    for (x2, y2) in pts_2Q:
        x3, y3 = rotate_90_deg(x2, y2)
        pts_3Q.append((x3, y3))
    pts_3Q.sort(key=lambda p: math.atan2(p[1], p[0]))
    all_points.extend(pts_3Q)

    # quadrant 4
    pts_4Q = []
    for (x3, y3) in pts_3Q:
        x4, y4 = rotate_90_deg(x3, y3)
        pts_4Q.append((x4, y4))
    pts_4Q.sort(key=lambda p: math.atan2(p[1], p[0]))
    all_points.extend(pts_4Q)

    # shift by (0.5, 0.5)
    shifted_points = [(x+0.5, y+0.5) for (x, y) in all_points]

    poly_str = ";".join(f"{p[0]:.6f},{p[1]:.6f}" for p in shifted_points)
    return poly_str, shifted_points


###############################################################################
# 2) S4 Run / Parsing
###############################################################################
def run_s4_for_polygon_and_c(poly_str, c_val, lua_script="metasurface_fixed_shape_and_c_value.lua", verbose=False):
    """
    Runs S4 with the given polygon string and c value.
    Expects the Lua script to print "Saved to <CSV>".
    Returns the path to the results CSV if found, or None otherwise.
    """
    cmd = f'../build/S4 -a "{poly_str} -c {c_val} -v -s" {lua_script}'
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

    # Try to parse "Saved to <path>"
    saved_path = None
    for line in proc.stdout.splitlines():
        if "Saved to" in line:
            saved_path = line.split("Saved to", 1)[1].strip()
            break

    # Fallback
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
            print(f"[WARN] Could not detect output CSV for c={c_val}!")
        return None


def read_s4_csv(csv_path):
    """
    Reads the CSV file from S4 results. Expects columns:
      'wavelength_um', 'R', 'T'
    Returns (lam_list, R_list, T_list).
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
                pass

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
    parser.add_argument("--nq", type=int, default=3,
                        help="Number of points in the first quadrant.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print debug info.")
    args = parser.parse_args()
    nQ = args.nq
    verbose = args.verbose

    # 1) Generate the shape
    poly_str, points_list = generate_c4_symmetric_polygon(nQ, rmin=0.05, rmax=0.65)
    print(f"[INFO] Created a C4-symmetric polygon with {4*nQ} vertices.")
    print("Polygon string:\n", poly_str)

    # 2) Quick plot of the shape
    fig_shape, ax_shape = plt.subplots(figsize=(4,4))
    xs = [p[0] for p in points_list]
    ys = [p[1] for p in points_list]
    xs_closed = xs + [xs[0]]
    ys_closed = ys + [ys[0]]
    ax_shape.plot(xs_closed, ys_closed, 'b-o', markersize=3)
    ax_shape.set_aspect('equal', adjustable='box')
    ax_shape.set_title("C4-Symmetric Polygon")
    plt.savefig("shape_c4_symmetric.png", dpi=150)
    print("[INFO] Shape plot saved to shape_c4_symmetric.png")
    plt.show(block=False)

    # 3) Angles in [0, pi/(2*nQ)] => 10 steps
    num_angles = 10
    angle_max  = math.pi / (2*nQ)
    angle_list = np.linspace(0, angle_max, num_angles)

    # 4) c-values from 0.0 to 1.0 => 11 total
    c_values   = [round(x, 1) for x in np.linspace(0, 1, 11)]
    print(f"[INFO] angle range = [0, {angle_max:.4f}] (10 steps)")
    print("[INFO] c values:", c_values)

    # We'll store data as results[angle_index][c_val] = (lam, R, T)
    results = {}

    # Helper to rotate the polygon about (0,0)
    def rotate_polygon_string(points, theta):
        rpoints = []
        for (x, y) in points:
            xr = x*math.cos(theta) - y*math.sin(theta)
            yr = x*math.sin(theta) + y*math.cos(theta)
            rpoints.append((xr, yr))
        return ";".join(f"{p[0]:.6f},{p[1]:.6f}" for p in rpoints)

    # 5) For each angle, for each c
    for i, theta in enumerate(angle_list):
        results[i] = {}
        poly_str_rot = rotate_polygon_string(points_list, theta)
        for c_val in c_values:
            csv_path = run_s4_for_polygon_and_c(poly_str_rot, c_val, verbose=verbose)
            lam_list, R_list, T_list = read_s4_csv(csv_path)
            results[i][c_val] = (lam_list, R_list, T_list)
            # small sleep to avoid identical timestamps in file
            time.sleep(0.05)

    # 6) Rearrange the data so we can plot "row = c, columns = R/T/R+T," lines for angles
    #    We'll do an 11 x 3 figure => 11 rows for c, 3 columns for R/T/R+T
    #    Each subplot has 10 lines for angles.

    # Create a dict for easier row-based indexing:
    # results_by_c[c_val][angleIndex] = (lam, R, T)
    results_by_c = { c: {} for c in c_values }
    for i, theta in enumerate(angle_list):
        for c_val in c_values:
            results_by_c[c_val][i] = results[i][c_val]

    nrows = len(c_values)  # 11
    ncols = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3*nrows),
                             sharex=True, sharey=False)
    # color map for angles (10 lines)
    angle_cmap = plt.cm.get_cmap("tab10", len(angle_list))  # or "viridis"

    # for row_i, c_val
    for row_i, c_val in enumerate(c_values):
        ax_R  = axes[row_i, 0]
        ax_T  = axes[row_i, 1]
        ax_RT = axes[row_i, 2]

        # For each angle
        for i, theta in enumerate(angle_list):
            lam_list, R_list, T_list = results_by_c[c_val][i]
            if not lam_list:
                continue
            color = angle_cmap(i)
            angle_deg = theta * 180/math.pi
            label = f"angle={angle_deg:.1f}°"

            # R
            ax_R.plot(lam_list, R_list, color=color, label=label)
            # T
            ax_T.plot(lam_list, T_list, color=color, label=label)
            # R+T
            rt_list = [abs(r)+t for (r,t) in zip(R_list, T_list)]
            ax_RT.plot(lam_list, rt_list, color=color, label=label)

        # Titles
        ax_R.set_title(f"c={c_val}: R")
        ax_T.set_title(f"c={c_val}: T")
        ax_RT.set_title(f"c={c_val}: R+T")

        if row_i == nrows-1:
            ax_R.set_xlabel("Wavelength (um)")
            ax_T.set_xlabel("Wavelength (um)")
            ax_RT.set_xlabel("Wavelength (um)")

        if row_i == 0:
            # Put legend in first row
            ax_R.legend(fontsize="small")
            ax_T.legend(fontsize="small")
            ax_RT.legend(fontsize="small")

        if row_i == 0:
            ax_R.set_ylabel("Reflectance R")
            ax_T.set_ylabel("Transmission T")
            ax_RT.set_ylabel("R + T")

    fig.suptitle("C4-Symmetric Polygon: Each row = c, lines = angles", fontsize=16)
    fig.tight_layout()
    out_fig = "c4_shape_spectra_by_c.png"
    plt.savefig(out_fig, dpi=150)
    print(f"[INFO] Final plot saved to {out_fig}")

    plt.show()


if __name__ == "__main__":
    main()

