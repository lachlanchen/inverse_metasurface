#!/usr/bin/env python3
"""
c4_shape_rotation_spectra_by_c.py

Generates a C4-symmetric shape with 'nq' points in the first quadrant,
replicates them to get 4-fold symmetry, shifts them by (0.5,0.5).
Then defines 11 rotation angles and 11 c values. For each combination
(angle, c), runs S4 (unless previously cached) to get R,T vs wavelength.

Finally, plots an 11×3 grid (rows = c, columns = R, T, R+T). Each subplot
has 11 lines for the 11 angles. This version:
  - Has no title, no legend, no grid.
  - Uses a cividis colormap for a clean look.
  - Caches results in c4_spectra_cache.npz to skip re-runs later.
"""

import os
import glob
import csv
import time
import math
import subprocess
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # We'll use it just to set a simple style

###############################################################################
# 1) Construct a C4-symmetric shape
###############################################################################
def generate_c4_symmetric_polygon(nQ=3, rmin=0.05, rmax=0.65):
    """
    Generate a polygon that has 4-fold (C4) symmetry:
      - nQ random angles in [0, pi/2], each with radius in [rmin, rmax].
      - Replicate them by +90°, +180°, +270°.
      - Shift by (+0.5, +0.5).

    Returns: (poly_str, points_list)
    """
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
    pts_2Q = [rotate_90_deg(x, y) for (x, y) in pts_1Q]
    pts_2Q.sort(key=lambda p: math.atan2(p[1], p[0]))
    all_points.extend(pts_2Q)
    # quadrant 3
    pts_3Q = [rotate_90_deg(x2, y2) for (x2, y2) in pts_2Q]
    pts_3Q.sort(key=lambda p: math.atan2(p[1], p[0]))
    all_points.extend(pts_3Q)
    # quadrant 4
    pts_4Q = [rotate_90_deg(x3, y3) for (x3, y3) in pts_3Q]
    pts_4Q.sort(key=lambda p: math.atan2(p[1], p[0]))
    all_points.extend(pts_4Q)

    # shift
    shifted_points = [(x+0.5, y+0.5) for (x, y) in all_points]
    poly_str = ";".join(f"{p[0]:.6f},{p[1]:.6f}" for p in shifted_points)
    return poly_str, shifted_points


###############################################################################
# 2) S4 runner & CSV parser
###############################################################################
def run_s4_for_polygon_and_c(poly_str, c_val, lua_script, verbose=False):
    """
    Run S4 with given polygon + c. The lua script must print "Saved to <CSV>".
    Returns path to the CSV or None if fail.
    """
    cmd = f'../build/S4 -a "{poly_str} -c {c_val} -v -s" {lua_script}'
    if verbose:
        print("[INFO] Running S4:", cmd)
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[ERROR] S4 run failed for c={c_val:.1f}!")
        print("=== STDOUT ===\n", proc.stdout)
        print("=== STDERR ===\n", proc.stderr)
        return None

    saved_path = None
    for line in proc.stdout.splitlines():
        if "Saved to" in line:
            saved_path = line.split("Saved to", 1)[1].strip()
            break

    # fallback
    if saved_path is None:
        pat = f"results/fixed_shape_c{c_val:.1f}_*.csv"
        found = glob.glob(pat)
        if found:
            saved_path = max(found, key=os.path.getctime)

    if saved_path and os.path.isfile(saved_path):
        if verbose:
            print(f"[INFO] Found CSV => {saved_path}")
        return saved_path
    else:
        if verbose:
            print(f"[WARN] No CSV found for c={c_val:.1f}!")
        return None

def read_s4_csv(csv_path):
    """Reads S4 results CSV. Returns (lam[], R[], T[])."""
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
    # sort by wavelength
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
    parser.add_argument("--nq", type=int, default=3, help="Points in quadrant.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--lua-script", default="metasurface_fixed_shape_and_c_value.lua",
                        help="Name of the Lua script to run in S4.")
    args = parser.parse_args()

    nQ = args.nq
    verbose = args.verbose
    lua_script = args.lua_script

    # We want no grid, so let’s do a very simple style:
    # We'll also disable spines if we want total minimalism
    sns.set_style("white")  # or style=None
    plt.rcParams["axes.grid"] = False
    # turn off top/right spines
    plt.rcParams["axes.spines.top"]    = False
    plt.rcParams["axes.spines.right"]  = False
    # (Optionally turn off left/bottom spines as well if you want. 
    #  We'll keep them for clarity.)

    # For reproducible random polygon:
    np.random.seed(1234)

    # 1) Generate the shape
    poly_str, points_list = generate_c4_symmetric_polygon(nQ, 0.05, 0.65)
    print(f"[INFO] Created shape with {4*nQ} vertices.\nPolygon:\n{poly_str}")

    # 2) Plot the shape (no title, no grid, no legend):
    fig_shape, ax_shape = plt.subplots(figsize=(4,4))
    pts_x = [p[0] for p in points_list]
    pts_y = [p[1] for p in points_list]
    pts_x_closed = pts_x + [pts_x[0]]
    pts_y_closed = pts_y + [pts_y[0]]
    # choose a color that stands out
    ax_shape.plot(pts_x_closed, pts_y_closed, color="black", marker="o", markersize=3)
    ax_shape.set_aspect("equal", adjustable="box")
    plt.savefig("c4_shape_polygon.png", dpi=150)
    print("[INFO] Polygon figure saved to c4_shape_polygon.png")
    plt.show(block=False)

    # 3) 11 angles in [0, pi/(2*nQ)]
    num_angles = 11
    angle_max = math.pi / (2*nQ)
    angle_list = np.linspace(0, angle_max, num_angles)

    # 4) c values in [0..1], total 11
    c_values = [round(x,1) for x in np.linspace(0,1,11)]

    # We'll store results in results[i_angle][c_val] = (lam[], R[], T[])
    results = {}
    cache_file = "c4_spectra_cache.npz"

    # Check if we already have a cached .npz
    if os.path.exists(cache_file):
        print(f"[INFO] Found cache file {cache_file}, loading data => no re-run of S4.")
        data = np.load(cache_file, allow_pickle=True)
        results = data["results"].item()  # retrieve the dict
    else:
        # We'll create the data from scratch
        for i, theta in enumerate(angle_list):
            results[i] = {}
            # rotate polygon
            rotated_points = []
            for (x, y) in points_list:
                xr = x*math.cos(theta) - y*math.sin(theta)
                yr = x*math.sin(theta) + y*math.cos(theta)
                rotated_points.append((xr, yr))
            # build poly string
            rot_poly_str = ";".join(f"{p[0]:.6f},{p[1]:.6f}" for p in rotated_points)

            for c_val in c_values:
                csv_path = run_s4_for_polygon_and_c(rot_poly_str, c_val, lua_script, verbose=verbose)
                lam, R, T = read_s4_csv(csv_path)
                results[i][c_val] = (lam, R, T)
                time.sleep(0.05)

        # Save it
        np.savez(cache_file, results=results)
        print(f"[INFO] Cached results to {cache_file}")

    # 5) Now we plot => 11x3. No title, no legend, new colormap "cividis"
    # Re-map results so row=c, col = R/T/R+T, lines= angles
    results_by_c = {c_val: {} for c_val in c_values}
    for i in results:
        for c_val in results[i]:
            results_by_c[c_val][i] = results[i][c_val]

    fig, axes = plt.subplots(nrows=len(c_values), ncols=3,
                             figsize=(14, 2*len(c_values)),
                             sharex=True, sharey=False)

    colormap = plt.cm.get_cmap("cividis", num_angles)

    for row_i, c_val in enumerate(c_values):
        axR  = axes[row_i, 0]
        axT  = axes[row_i, 1]
        axRT = axes[row_i, 2]

        for i_angle in range(num_angles):
            lam, R, T = results_by_c[c_val][i_angle]
            if not lam:
                continue
            color = colormap(i_angle)
            # plot
            axR.plot(lam, R, color=color)
            axT.plot(lam, T, color=color)
            RT = [abs(r)+t for (r,t) in zip(R, T)]
            axRT.plot(lam, RT, color=color)

        # no titles
        if row_i == len(c_values)-1:
            axR.set_xlabel("Wavelength (um)")
            axT.set_xlabel("Wavelength (um)")
            axRT.set_xlabel("Wavelength (um)")
        # label y only on left-most
        if row_i == 0:
            axR.set_ylabel("R")
            axT.set_ylabel("T")
            axRT.set_ylabel("R+T")

    # no figure suptitle, no legend
    fig.tight_layout()
    outname = "c4_shape_spectra.png"
    plt.savefig(outname, dpi=150)
    print(f"[INFO] Final figure saved to {outname}")
    plt.show()


if __name__ == "__main__":
    main()

