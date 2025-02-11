#!/usr/bin/env python3
"""
eight_shapes_rotation_scale.py

Creates 8 C4-symmetric shapes:
  1..4: "Regular" with radius=0.35 (nQ=1..4 points in first quadrant, angles evenly spaced).
  5..8: "Random" with radius in [0.1..0.3], nQ=1..4, using a fixed seed to be reproducible.

Then draws an 11×3 figure (rows= c in [0..1..0.1]):
  - Left column: each row has 8 lines (the 8 shapes).
  - Middle column: rotation in 11 steps for ONE chosen shape (arg --nq in [1..8]).
  - Right column: scale in 11 steps from 0.5..1.5 for that same chosen shape.

Reflection R vs. wavelength is plotted, no axes or legend. S4 calls are run in parallel, 
and results CSV files are moved to "myresults/" subfolder.
"""

import os
import math
import glob
import csv
import time
import shutil
import argparse
import numpy as np
import subprocess
import concurrent.futures

import matplotlib.pyplot as plt
import seaborn as sns

##############################################################################
# 1) Functions to generate shapes
##############################################################################

def generate_c4_regular(nQ=1):
    """
    A "regular" shape for nQ=1..4, each with radius=0.35, angles evenly spaced in [0, pi/2].
      nQ=1 => angle=0
      nQ=2 => angles=0, 45
      nQ=3 => angles=0, 30, 60
      nQ=4 => angles=0, 22.5, 45, 67.5
    Then replicate to 4 quadrants => 4*nQ points, shift by (0.5, 0.5).
    Returns (poly_str, list_of_points).
    """
    r = 0.35
    if nQ==1:
        deg_list = [0.0]
    elif nQ==2:
        deg_list = [0.0, 45.0]
    elif nQ==3:
        deg_list = [0.0, 30.0, 60.0]
    elif nQ==4:
        deg_list = [0.0, 22.5, 45.0, 67.5]
    else:
        raise ValueError("nQ in [1..4] for generate_c4_regular()")

    # Points in Q1
    pts_1Q = []
    for deg in deg_list:
        rad = math.radians(deg)
        x = r*math.cos(rad)
        y = r*math.sin(rad)
        pts_1Q.append((x,y))

    def rot90(x,y): 
        return (-y,x)

    # quadrants
    all_points = list(pts_1Q)
    q2 = [rot90(px,py) for (px,py) in pts_1Q]
    q3 = [rot90(px,py) for (px,py) in q2]
    q4 = [rot90(px,py) for (px,py) in q3]
    all_points.extend(q2)
    all_points.extend(q3)
    all_points.extend(q4)

    shifted = [(xx+0.5, yy+0.5) for (xx,yy) in all_points]
    poly_str = ";".join(f"{xx:.6f},{yy:.6f}" for (xx,yy) in shifted)
    return poly_str, shifted

def generate_c4_random(nQ=1, seed=123, rmin=0.1, rmax=0.3):
    """
    A "random" shape for nQ=1..4, angles in [0..pi/2] equally spaced,
    but each radius is random in [rmin..rmax], using a fixed seed 
    to get reproducible random radii. Then replicate 4 quadrants, shift.
    Returns (poly_str, list_of_points).
    """
    np.random.seed(seed)
    if nQ==1:
        deg_list = [0.0]
    elif nQ==2:
        deg_list = [0.0, 45.0]
    elif nQ==3:
        deg_list = [0.0, 30.0, 60.0]
    elif nQ==4:
        deg_list = [0.0, 22.5, 45.0, 67.5]
    else:
        raise ValueError("nQ in [1..4] for generate_c4_random()")

    # pick random radii in [rmin..rmax] for each angle
    nAngles = len(deg_list)
    radii = np.random.uniform(rmin, rmax, nAngles)

    pts_1Q = []
    for i,deg in enumerate(deg_list):
        rad = math.radians(deg)
        r_  = radii[i]
        x = r_*math.cos(rad)
        y = r_*math.sin(rad)
        pts_1Q.append((x,y))

    def rot90(x,y): 
        return (-y,x)

    all_points = list(pts_1Q)
    q2 = [rot90(px,py) for (px,py) in pts_1Q]
    q3 = [rot90(px,py) for (px,py) in q2]
    q4 = [rot90(px,py) for (px,py) in q3]
    all_points.extend(q2)
    all_points.extend(q3)
    all_points.extend(q4)

    shifted = [(xx+0.5, yy+0.5) for (xx,yy) in all_points]
    poly_str = ";".join(f"{xx:.6f},{yy:.6f}" for (xx,yy) in shifted)
    return poly_str, shifted

##############################################################################
# 2) S4 runner, storing CSV in myresults/, reading reflection
##############################################################################
def run_s4_and_get_R(poly_str, c_val, outfolder="myresults", verbose=False):
    """
    Runs S4 => reflection R vs wavelength, moves CSV into outfolder, returns (lam,R).
    """
    lua_script = "metasurface_fixed_shape_and_c_value.lua"
    cmd = f'../build/S4 -a "{poly_str} -c {c_val} -v -s" {lua_script}'
    if verbose:
        print(f"[S4] c={c_val:.1f} => {cmd}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode!=0:
        if verbose:
            print("[ERROR] S4 run failed!")
            print("STDOUT:", proc.stdout)
            print("STDERR:", proc.stderr)
        return [],[]

    # parse "Saved to <CSV>"
    saved_path = None
    for line in proc.stdout.splitlines():
        if "Saved to" in line:
            saved_path = line.split("Saved to",1)[1].strip()
            break
    if not saved_path or not os.path.isfile(saved_path):
        # fallback
        pat = f"results/fixed_shape_c{c_val:.1f}_*.csv"
        found = glob.glob(pat)
        if found:
            saved_path = max(found, key=os.path.getctime)
    if not saved_path or not os.path.isfile(saved_path):
        if verbose:
            print("[WARN] No CSV found for c=%.1f" % c_val)
        return [],[]

    # move to myresults/
    os.makedirs(outfolder, exist_ok=True)
    new_path = os.path.join(outfolder, os.path.basename(saved_path))
    try:
        shutil.move(saved_path, new_path)
    except:
        pass

    # parse reflection
    lam_list, R_list = [], []
    with open(new_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lam = float(row["wavelength_um"])
                R_  = float(row["R"])
                lam_list.append(lam)
                R_list.append(R_)
            except:
                pass
    data = sorted(zip(lam_list, R_list), key=lambda x:x[0])
    lam_list = [d[0] for d in data]
    R_list   = [d[1] for d in data]
    return lam_list, R_list

##############################################################################
# 3) Build shapes: 8 total => 4 "regular" + 4 "random"
##############################################################################
def build_all_8_shapes():
    """
    Return a dict: shapeIndex => (poly_str, points).
     1..4 => regular nQ=1..4
     5..8 => random  nQ=1..4
    """
    shapes = {}
    # regular
    for i,nQ in enumerate([1,2,3,4], start=1):
        poly, pts = generate_c4_regular(nQ)
        shapes[i] = (poly, pts)
    # random
    seeds = [123, 456, 789, 999]
    for i,nQ in enumerate([1,2,3,4], start=5):
        poly, pts = generate_c4_random(nQ, seed=seeds[i-5], rmin=0.1, rmax=0.3)
        shapes[i] = (poly, pts)
    return shapes

##############################################################################
# 4) parallel tasks
##############################################################################
def parallel_tasks(nq_chosen):
    """
    We have 11 c values => for each row c:
      left column => 8 shapes
      middle => 11 rotations for shape 'nq_chosen'
      right => 11 scales for shape 'nq_chosen'
    We'll build tasks => (key, c_val, polygon).
      key is ('left', c_val, shapeIndex) or ('mid', c_val, iRot) or ('right', c_val, iScale).
    """
    c_vals = [round(x,1) for x in np.linspace(0,1,11)]
    tasks = []
    shapes = build_all_8_shapes()  # shapeIndex => (poly_str, pts)

    # build angle array, scale array
    angle_list = np.linspace(0, math.pi/(2*nq_chosen if nq_chosen<=4 else 4), 11)
    # Actually user said "the max rotation is pi/(2*nQ)". If the chosen shape is 5..8 => that nQ is ??? 
    # We must figure out the underlying nQ. If shapeIndex<=4 => nQ= shapeIndex, else shapeIndex-4 => nQ??
    # Actually for random shapes 5->1,6->2,7->3,8->4. 
    # let's define:
    if nq_chosen<=4:
        base_nQ = nq_chosen
    else:
        base_nQ = nq_chosen-4
    angle_list = np.linspace(0, math.pi/(2*base_nQ), 11)

    scale_list = np.linspace(0.5,1.5,11)

    # For the shape 'nq_chosen', we also have the points:
    chosen_poly, chosen_pts = shapes[nq_chosen]

    # helper to rotate or scale chosen shape
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

    # 1) Left col => for each c => 8 shapes
    for c_val in c_vals:
        for shape_idx in range(1,9):
            polyS, _ = shapes[shape_idx]
            tasks.append({
                'key': ('left', c_val, shape_idx),
                'c_val': c_val,
                'poly_str': polyS
            })

    # 2) Middle => for each c => 11 rotations
    for c_val in c_vals:
        for iang,angle in enumerate(angle_list):
            polyS = rotate_polygon(chosen_pts, angle)
            tasks.append({
                'key': ('mid', c_val, iang),
                'c_val': c_val,
                'poly_str': polyS
            })

    # 3) Right => for each c => 11 scales
    for c_val in c_vals:
        for iscl, s_ in enumerate(scale_list):
            polyS = scale_polygon(chosen_pts, s_)
            tasks.append({
                'key': ('right', c_val, iscl),
                'c_val': c_val,
                'poly_str': polyS
            })

    return tasks, angle_list, scale_list


def run_one_task(task):
    key = task['key']
    c_val = task['c_val']
    polyS = task['poly_str']
    lam,R = run_s4_and_get_R(polyS, c_val, outfolder="myresults", verbose=False)
    return (key, lam, R)

##############################################################################
# 5) Main
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nq", type=int, default=4,
                        help="Which shape index [1..8] to use in middle/right columns. 1..4=regular,5..8=random.")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Parallel tasks count.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    nq_chosen = args.nq
    if nq_chosen<1 or nq_chosen>8:
        raise ValueError("Please pick --nq in [1..8].")

    # minimal style
    sns.set_style("white")
    plt.rcParams["axes.grid"]=False
    plt.rcParams["axes.spines.top"]=False
    plt.rcParams["axes.spines.right"]=False
    plt.rcParams["axes.spines.left"]=False
    plt.rcParams["axes.spines.bottom"]=False

    # Build tasks
    tasks, angle_list, scale_list = parallel_tasks(nq_chosen)
    print(f"[INFO] We have {len(tasks)} tasks total. Running in parallel...")

    # run parallel
    results_dict = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as exe:
        fut_map = { exe.submit(run_one_task, t): t for t in tasks }
        for fut in concurrent.futures.as_completed(fut_map):
            t = fut_map[fut]
            try:
                key, lam, R = fut.result()
                results_dict[key] = (lam,R)
            except Exception as e:
                print("[ERROR]", t, e)
                results_dict[t['key']] = ([],[])

    print("[INFO] All tasks done. Building the final figure...")

    # We'll do 11 rows => c in [0..1..0.1], 3 cols => left/mid/right
    c_vals = [round(x,1) for x in np.linspace(0,1,11)]
    fig, axes = plt.subplots(nrows=len(c_vals), ncols=3,
                             figsize=(15,3*len(c_vals)),
                             sharex=False, sharey=False)

    # colors
    # left col => 8 lines => first 4 "regular" one set, next 4 "random" second set
    # let's define two color maps each with 4 distinct colors
    # or we can define a single palette of 8 distinct colors
    colors_left_reg = ["red","green","blue","purple"]
    colors_left_rand = ["orange","darkcyan","magenta","brown"]

    # middle col => len(angle_list) lines => use viridis
    col_mid = plt.cm.viridis(np.linspace(0,1,len(angle_list)))
    # right col => scale => also viridis
    col_right = plt.cm.viridis(np.linspace(0,1,len(scale_list)))

    for row_i, c_val in enumerate(c_vals):
        ax_left = axes[row_i][0]
        ax_mid  = axes[row_i][1]
        ax_right= axes[row_i][2]

        ax_left.set_axis_off()
        ax_mid.set_axis_off()
        ax_right.set_axis_off()

        # left => shapeIndex=1..8
        #  1..4 => use colors_left_reg[i-1]
        #  5..8 => use colors_left_rand[i-5]
        for shape_idx in range(1,9):
            lam,R = results_dict.get(('left', c_val, shape_idx),([],[]))
            if lam:
                if shape_idx<=4:
                    color_ = colors_left_reg[shape_idx-1]
                else:
                    color_ = colors_left_rand[shape_idx-5]
                ax_left.plot(lam,R,color=color_,linewidth=1.5)

        # middle => iRot in 0..len(angle_list)-1
        for iang in range(len(angle_list)):
            lam,R = results_dict.get(('mid', c_val, iang),([],[]))
            if lam:
                ax_mid.plot(lam,R,color=col_mid[iang],linewidth=1.3)

        # right => iScale in 0..len(scale_list)-1
        for iscl in range(len(scale_list)):
            lam,R = results_dict.get(('right', c_val, iscl),([],[]))
            if lam:
                ax_right.plot(lam,R,color=col_right[iscl],linewidth=1.3)

    fig.tight_layout()
    out_fig = "three_col_plot_minimal_axes.png"
    plt.savefig(out_fig, dpi=150)
    print(f"[INFO] Final figure saved to {out_fig}")
    plt.show()

if __name__=="__main__":
    main()

