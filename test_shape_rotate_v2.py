#!/usr/bin/env python3

import os
import glob
import csv
import time
import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# 1) Construct a C4-symmetric shape
###############################################################################
def generate_c4_symmetric_polygon(nQ=3, rmin=0.05, rmax=0.65):
    """
    Generate a polygon that has 4-fold (C4) symmetry:
      - We define nQ points in the first quadrant with angles in [0, pi/2].
      - Each point has a random radius in [rmin, rmax].
      - Then we replicate these points by +90°, +180°, +270° to cover full 2π.
      - Finally, we shift the entire set by (+0.5, +0.5).
    
    The vertices are output in ascending angle order (0 -> 2π).
    Returns:
      (poly_str, points_list)
        poly_str: "x1,y1;x2,y2;..."
        points_list: list of (x, y)  (for your own reference)
    """
    # Step A: pick nQ angles in [0, pi/2], sorted
    angles_1Q = np.sort(np.random.uniform(0.0, math.pi/2, nQ))

    # Step B: pick a random radius in [rmin, rmax] for each angle
    radii_1Q = np.random.uniform(rmin, rmax, nQ)

    # Step C: create points in first quadrant
    #         (We store (theta, r) so we can replicate easily.)
    pts_1Q = []
    for theta, r in zip(angles_1Q, radii_1Q):
        x = r*math.cos(theta)
        y = r*math.sin(theta)
        pts_1Q.append((x, y))

    # Step D: replicate to other quadrants (C4 symmetry).
    #         We'll do angles_1Q + 0, + pi/2, + pi, + 3pi/2
    #         The final ordering: we want angles from 0..2π in ascending order
    #         (0..pi/2 => pi/2..pi => pi..3pi/2 => 3pi/2..2pi).
    def rotate_90_deg(x, y):
        return (-y, x)  # rotation by +90° about origin
    # We apply that successively

    all_points = []
    # quadrant 1: as is
    all_points.extend(pts_1Q)
    # quadrant 2
    pts_2Q = []
    for (x, y) in pts_1Q:
        x2, y2 = rotate_90_deg(x, y)
        pts_2Q.append((x2, y2))
    # Sort them by angle to keep polygon ordering consistent
    pts_2Q.sort(key=lambda p: math.atan2(p[1], p[0]))
    all_points.extend(pts_2Q)
    # quadrant 3
    pts_3Q = []
    for (x, y) in pts_2Q:
        x3, y3 = rotate_90_deg(x, y)
        pts_3Q.append((x3, y3))
    pts_3Q.sort(key=lambda p: math.atan2(p[1], p[0]))
    all_points.extend(pts_3Q)
    # quadrant 4
    pts_4Q = []
    for (x, y) in pts_3Q:
        x4, y4 = rotate_90_deg(x, y)
        pts_4Q.append((x4, y4))
    pts_4Q.sort(key=lambda p: math.atan2(p[1], p[0]))
    all_points.extend(pts_4Q)

    # Now we shift everything by (0.5, 0.5)
    shifted_points = []
    for (x, y) in all_points:
        shifted_points.append((x + 0.5, y + 0.5))

    # Convert to "x1,y1;x2,y2;..." string
    poly_str = ";".join(f"{p[0]:.6f},{p[1]:.6f}" for p in shifted_points)
    return poly_str, shifted_points

###############################################################################
# 2) S4 Run / Parsing
###############################################################################
def run_s4_for_polygon_and_c(poly_str, c_val, lua_script="metasurface_fixed_shape_and_c_value.lua", verbose=False):
    """
    Runs S4 with the given polygon string and c value, using the specified Lua script.
    Returns the path to the results CSV if found, or None otherwise.
    """
    # Using '-v -s' so that the Lua script prints "Saved to <path>" line
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
      'wavelength_um', 'R', 'T', ...
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
                # If any parse error, skip
                pass

    # Sort by wavelength
    data = sorted(zip(lam_list, R_list, T_list), key=lambda x: x[0])
    lam_list = [d[0] for d in data]
    R_list   = [d[1] for d in data]
    T_list   = [d[2] for d in data]
    return lam_list, R_list, T_list


###############################################################################
# 3) Main: Generate shape, show it, rotate, run S4, plot results
###############################################################################
def main():
    # 1) Create the shape with C4 symmetry
    nQ = 2  # number of points in the first quadrant
    poly_str, points_list = generate_c4_symmetric_polygon(nQ, rmin=0.05, rmax=0.65)

    print("[INFO] Created a C4-symmetric polygon with", 4*nQ, "vertices.")
    print("Polygon string:\n", poly_str)

    # 2) Plot the shape so we can see it
    fig_shape, ax_shape = plt.subplots(figsize=(5,5))
    xs = [p[0] for p in points_list]
    ys = [p[1] for p in points_list]
    # close the polygon by repeating the first vertex
    xs_closed = xs + [xs[0]]
    ys_closed = ys + [ys[0]]
    ax_shape.plot(xs_closed, ys_closed, 'b-o', markersize=3)
    ax_shape.set_aspect('equal', adjustable='box')
    ax_shape.set_title("C4-Symmetric Polygon (shifted by +0.5,+0.5)")
    plt.savefig("shape_c4_symmetric.png", dpi=150)
    print("[INFO] Shape plot saved to shape_c4_symmetric.png")
    plt.show(block=False)  # show shape in one window, but continue code

    # 3) Angles: from 0 to pi/(2*nQ) in 10 steps
    num_angles = 10
    angle_max = math.pi / (2*nQ)
    angle_list = np.linspace(0, angle_max, num_angles)

    # 4) c-values: 0.0 .. 1.0 (11 total)
    c_values = [round(x, 1) for x in np.linspace(0, 1, 11)]
    print("[INFO] Will test angles in [0, {:.4f}] rad ({} steps)".format(angle_max, num_angles))
    print("[INFO] c values:", c_values)

    # We'll store results in a dict: results[angle_index][c_val] = (lam_list, R_list, T_list)
    results = {}
    
    # A helper to rotate the entire polygon string by 'theta'
    def rotate_polygon_string(points, theta):
        # rotate each point
        rpoints = []
        for (x, y) in points:
            x_r = x*math.cos(theta) - y*math.sin(theta)
            y_r = x*math.sin(theta) + y*math.cos(theta)
            rpoints.append((x_r, y_r))
        # convert to "x1,y1;x2,y2;..." string
        return ";".join(f"{p[0]:.6f},{p[1]:.6f}" for p in rpoints)

    # 5) Loop over angles, run S4 for each c
    for i, theta in enumerate(angle_list):
        results[i] = {}
        # rotate the shape by angle 'theta'
        poly_str_rot = rotate_polygon_string(points_list, theta)

        for c_val in c_values:
            csv_path = run_s4_for_polygon_and_c(poly_str_rot, c_val, verbose=False)
            lam_list, R_list, T_list = read_s4_csv(csv_path)
            results[i][c_val] = (lam_list, R_list, T_list)
            # short sleep so that filenames differ (if needed)
            time.sleep(0.05)

    # 6) Plot the results in a 10×3 grid: each row = angle, columns = (R, T, R+T)
    fig, axes = plt.subplots(nrows=num_angles, ncols=3, figsize=(14, 3*num_angles),
                             sharex=True, sharey=False)
    # color map for the 11 c values
    color_map = plt.cm.get_cmap("viridis", len(c_values))

    for i, theta in enumerate(angle_list):
        ax_R  = axes[i, 0]
        ax_T  = axes[i, 1]
        ax_RT = axes[i, 2]

        # for each c, plot
        for j, c_val in enumerate(c_values):
            lam_list, R_list, T_list = results[i][c_val]
            if not lam_list:
                continue
            color = color_map(j)
            label = f"c={c_val}"
            # R
            ax_R.plot(lam_list, R_list, color=color, label=label)
            # T
            ax_T.plot(lam_list, T_list, color=color, label=label)
            # R+T
            RT_list = [abs(r)+t for (r,t) in zip(R_list, T_list)]
            ax_RT.plot(lam_list, RT_list, color=color, label=label)

        # Titles
        angle_deg = theta*180.0/math.pi
        ax_R.set_title(f"Angle={angle_deg:.1f}°: R")
        ax_T.set_title(f"Angle={angle_deg:.1f}°: T")
        ax_RT.set_title(f"Angle={angle_deg:.1f}°: R+T")

        if i == num_angles-1:
            ax_R.set_xlabel("Wavelength (um)")
            ax_T.set_xlabel("Wavelength (um)")
            ax_RT.set_xlabel("Wavelength (um)")

        # Y-label only on first col
        if i == 0:
            ax_R.legend(fontsize="small")
            ax_T.legend(fontsize="small")
            ax_RT.legend(fontsize="small")

    axes[0,0].set_ylabel("Reflectance R")
    axes[0,1].set_ylabel("Transmission T")
    axes[0,2].set_ylabel("R + T")

    fig.suptitle("C4-Symmetric Polygon: Rotation vs. c-value Spectra", fontsize=16)
    fig.tight_layout()
    out_fig = "c4_shape_rotation_spectra.png"
    plt.savefig(out_fig, dpi=150)
    print(f"[INFO] Final 10x3 spectra plot saved to {out_fig}")
    plt.show()

if __name__ == "__main__":
    main()

