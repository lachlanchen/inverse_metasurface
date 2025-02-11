#!/usr/bin/env python3

import os
import math
import glob
import time
import shutil
import argparse
import numpy as np
import subprocess
import concurrent.futures

import matplotlib.pyplot as plt
import seaborn as sns

##############################################################################
# 1) Generate C4-symmetric shapes with fixed angles
##############################################################################
def generate_c4_polygon_fixed_angles(nQ=4):
    """
    Creates a C4-symmetric polygon by placing nQ points in [0, pi/2] at fixed angles:
      nQ=1 => angle [0°]
      nQ=2 => angles [0°, 45°]
      nQ=3 => angles [0°, 30°, 60°]
      nQ=4 => angles [0°, 22.5°, 45°, 67.5°]
    Each point has radius in [0.2, 0.2] (i.e. a fixed radius?), or 0.3, etc.
    Then replicate to +90°, +180°, +270°, shift by (+0.5, +0.5).
    Returns (poly_str, list_of_points).
    """

    # Angles (in degrees) for each nQ:
    if nQ == 1:
        deg_list = [0.0]
        r = 0.2
    elif nQ == 2:
        deg_list = [0.0, 45.0]
        r = 0.2
    elif nQ == 3:
        deg_list = [0.0, 30.0, 60.0]
        r = 0.2
    elif nQ == 4:
        deg_list = [0.0, 22.5, 45.0, 67.5]
        r = 0.2
    else:
        raise ValueError("nQ must be in {1,2,3,4} for this fixed-angles version.")

    # Create points in first quadrant
    # (all with the same radius 'r' for simplicity)
    pts_1Q = []
    for deg in deg_list:
        rad = math.radians(deg)
        x = r * math.cos(rad)
        y = r * math.sin(rad)
        pts_1Q.append((x, y))

    def rotate_90_deg(x, y):
        return (-y, x)

    # Quadrant 1
    all_points = list(pts_1Q)
    # Quadrant 2
    pts_2Q = [rotate_90_deg(px, py) for (px,py) in pts_1Q]
    # Quadrant 3
    pts_3Q = [rotate_90_deg(px, py) for (px,py) in pts_2Q]
    # Quadrant 4
    pts_4Q = [rotate_90_deg(px, py) for (px,py) in pts_3Q]

    # We can just extend them in order
    all_points.extend(pts_2Q)
    all_points.extend(pts_3Q)
    all_points.extend(pts_4Q)

    # shift by (+0.5, +0.5)
    shifted = [(p[0]+0.5, p[1]+0.5) for p in all_points]

    # build polygon string
    poly_str = ";".join(f"{p[0]:.6f},{p[1]:.6f}" for p in shifted)
    return poly_str, shifted

##############################################################################
# 2) S4 runner, storing output CSVs in a subfolder, reading reflection
##############################################################################
def run_s4_and_get_R(poly_str, c_val, outfolder="myresults", verbose=False):
    """
    Runs S4 for the given polygon & c, using metasurface_fixed_shape_and_c_value.lua,
    storing CSV in e.g. ./results/*. After the run, we move the CSV into `outfolder/`.
    Returns (lam_list, R_list).
    (We skip T in this function for brevity, or we could also parse it.)
    """
    lua_script = "metasurface_fixed_shape_and_c_value.lua"
    cmd = f'../build/S4 -a "{poly_str} -c {c_val} -v -s" {lua_script}'
    if verbose:
        print("[S4] c=%.1f =>" % c_val, cmd)
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        if verbose:
            print("[ERROR] S4 failed!")
            print("STDOUT:", proc.stdout)
            print("STDERR:", proc.stderr)
        return [], []

    # parse the "Saved to <CSV>"
    saved_path = None
    for line in proc.stdout.splitlines():
        if "Saved to" in line:
            saved_path = line.split("Saved to",1)[1].strip()
            break

    # fallback
    if saved_path is None:
        pattern = f"results/fixed_shape_c{c_val:.1f}_*.csv"
        found = glob.glob(pattern)
        if found:
            saved_path = max(found, key=os.path.getctime)

    if not saved_path or not os.path.isfile(saved_path):
        if verbose:
            print(f"[WARN] No CSV for c={c_val} found.")
        return [], []

    # Move it to outfolder
    os.makedirs(outfolder, exist_ok=True)
    new_path = os.path.join(outfolder, os.path.basename(saved_path))
    try:
        shutil.move(saved_path, new_path)
    except:
        pass  # if it can't move, ignore

    # Now read reflection from new_path
    lam_list, R_list = [], []
    with open(new_path, "r", newline="") as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lam  = float(row["wavelength_um"])
                Rval = float(row["R"])
                lam_list.append(lam)
                R_list.append(Rval)
            except:
                pass
    # sort by lam
    data = sorted(zip(lam_list, R_list), key=lambda x:x[0])
    lam_list = [d[0] for d in data]
    R_list   = [d[1] for d in data]
    return lam_list, R_list

##############################################################################
# 3) Building tasks for parallel execution
##############################################################################
def parallel_tasks():
    """
    Creates a list of (task_id, c_val, 'left'/'mid'/'right', shape or rotation or scale).
    We will generate them all, run them in parallel, store reflection arrays in a dict.
    Then we can do the final plotting step after.
    """
    # c in {0.0..1.0} => 11 total
    c_vals = [round(x,1) for x in np.linspace(0,1,11)]

    tasks = []
    # For the left column => nQ=1..4 shapes
    # For the middle => 11 angles with nQ=4
    # For the right => 11 scales with nQ=4

    # Pre-generate the 4 shapes (nQ=1..4)
    shapes_nQ = {}
    for nQ in [1,2,3,4]:
        poly_str, pts = generate_c4_polygon_fixed_angles(nQ)
        shapes_nQ[nQ] = (poly_str, pts)

    # We'll define the middle angles for nQ=4
    nQ4_poly_str, nQ4_pts = shapes_nQ[4]
    angle_list = np.linspace(0, math.pi/(2*4), 11)

    def rotate_polygon(pts, theta):
        rpts = []
        for (x,y) in pts:
            xr = x*math.cos(theta) - y*math.sin(theta)
            yr = x*math.sin(theta) + y*math.cos(theta)
            rpts.append((xr, yr))
        return ";".join(f"{xx:.6f},{yy:.6f}" for (xx,yy) in rpts)

    def scale_polygon(pts, factor):
        spts = []
        for (x,y) in pts:
            spts.append((factor*x, factor*y))
        return ";".join(f"{xx:.6f},{yy:.6f}" for (xx,yy) in spts)

    scale_vals = np.linspace(0.5, 1.5, 11)

    # We'll define a key in our results dict => (row_c, 'left'/'mid'/'right', index)
    # Left col has index in {1,2,3,4}, middle col => index in [0..10], right col => index in [0..10].
    # We'll store the polygon_str in tasks, and the c_val, and a label for which subplot.

    # LEFT column => 4 lines from nQ=1..4
    for c_val in c_vals:
        for nQ in [1,2,3,4]:
            poly_str, pts = shapes_nQ[nQ]
            # We'll create a "task key"
            # e.g. key= ('left', c_val, nQ)
            tasks.append({
                'key': ('left', c_val, nQ),
                'c_val': c_val,
                'polygon_str': poly_str
            })

    # MIDDLE => rotate the nQ=4 shape
    for c_val in c_vals:
        for iang, angle in enumerate(angle_list):
            poly_str = rotate_polygon(nQ4_pts, angle)
            tasks.append({
                'key': ('mid', c_val, iang),
                'c_val': c_val,
                'polygon_str': poly_str
            })

    # RIGHT => scale the nQ=4 shape
    for c_val in c_vals:
        for iscl, scl in enumerate(scale_vals):
            poly_str = scale_polygon(nQ4_pts, scl)
            tasks.append({
                'key': ('right', c_val, iscl),
                'c_val': c_val,
                'polygon_str': poly_str
            })

    return tasks

def run_one_task(task):
    """
    Worker function for parallel execution.
    Uses run_s4_and_get_R() to get reflection data.
    Returns (task['key'], lam_list, R_list).
    """
    key = task['key']
    c_val = task['c_val']
    poly_str = task['polygon_str']

    lam, R = run_s4_and_get_R(poly_str, c_val, outfolder="myresults", verbose=False)
    return (key, lam, R)

##############################################################################
# 4) Main script: build tasks, run in parallel, plot in 11x3 with no axes
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Number of parallel S4 tasks.")
    parser.add_argument("--verbose", action="store_true",
                        help="If set, show debug logs.")
    args = parser.parse_args()

    # 1) Minimal style
    sns.set_style("white")
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.spines.top"]    = False
    plt.rcParams["axes.spines.right"]  = False
    plt.rcParams["axes.spines.left"]   = False
    plt.rcParams["axes.spines.bottom"] = False

    # Build tasks
    tasks = parallel_tasks()
    print(f"[INFO] We have {len(tasks)} S4 tasks in total.")

    # 2) Run in parallel
    results_dict = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(run_one_task, t): t for t in tasks}
        for fut in concurrent.futures.as_completed(future_map):
            t = future_map[fut]
            try:
                key, lam, R = fut.result()
                # store in a dictionary => results_dict[key] = (lam, R)
                results_dict[key] = (lam, R)
            except Exception as e:
                print("[ERROR] Task failed:", t, e)
                results_dict[t['key']] = ([],[])

    print("[INFO] All parallel tasks completed. Building the final plot...")

    # 3) Now we create the figure with 11 rows, 3 columns
    # row => c in [0..1..0.1]
    c_vals = [round(x,1) for x in np.linspace(0,1,11)]
    fig, axes = plt.subplots(nrows=len(c_vals), ncols=3,
                             figsize=(15,2.5*len(c_vals)),
                             sharex=False, sharey=False)

    # We'll define color sets:
    #  left col => 4 fixed colors for nQ=1..4
    colors_left = ["red","green","blue","purple"]
    #  middle col => viridis 11
    colors_mid = plt.cm.viridis(np.linspace(0,1,11))
    #  right col => also viridis 11
    colors_right = plt.cm.viridis(np.linspace(0,1,11))

    # We'll also reconstruct the angle_list & scale_list
    angle_list = np.linspace(0, math.pi/(2*4), 11)
    scale_list = np.linspace(0.5,1.5,11)

    # 4) Fill each subplot
    for row_i, c_val in enumerate(c_vals):
        ax_left  = axes[row_i][0]
        ax_mid   = axes[row_i][1]
        ax_right = axes[row_i][2]
        # no axes
        ax_left.set_axis_off()
        ax_mid.set_axis_off()
        ax_right.set_axis_off()

        # LEFT col => 4 lines for nQ=1..4
        for idx,nQ_ in enumerate([1,2,3,4]):
            key = ('left', c_val, nQ_)
            lam,R = results_dict.get(key,([],[]))
            if lam:
                ax_left.plot(lam,R,color=colors_left[idx],linewidth=1.5)

        # MIDDLE => 11 lines for angle 0..10
        for iang in range(11):
            key = ('mid', c_val, iang)
            lam,R = results_dict.get(key,([],[]))
            if lam:
                ax_mid.plot(lam,R,color=colors_mid[iang],linewidth=1.2)

        # RIGHT => 11 lines for scale 0..10
        for iscl in range(11):
            key = ('right', c_val, iscl)
            lam,R = results_dict.get(key,([],[]))
            if lam:
                ax_right.plot(lam,R,color=colors_right[iscl],linewidth=1.2)

    fig.tight_layout()
    out_png = "three_col_plot_minimal_axes.png"
    plt.savefig(out_png, dpi=150)
    print(f"[INFO] Final plot saved to {out_png}")
    plt.show()

if __name__=="__main__":
    main()

