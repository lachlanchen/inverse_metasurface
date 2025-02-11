#!/usr/bin/env python3
"""
eight_shapes_rotation_scale_4col.py

Generates 8 total C4-symmetric shapes:
  1..4 => "regular" with radius=0.35
  5..8 => "random" with radius in [0.1..0.3], using seeds derived from a base --seed

Prints each shape as shape_i.png in subfolder "shapes_plots/".
Then builds a 4-column figure (with 11 rows for c=0..1):
  - col0 => 4 regular shapes (#1..4)
  - col1 => 4 random shapes (#5..8)
  - col2 => rotation of a single chosen shape (arg --nq in [1..8]) in 11 steps
  - col3 => scale of that same chosen shape in 11 steps [0.5..1.5]

Reflection R vs. wavelength is plotted, no axes/legends, all S4 calls run in parallel.
Final figure => "four_col_plot_minimal_axes.png"
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
# 1) Shape Generators
##############################################################################

def generate_c4_regular(nQ=1):
    """
    "Regular" shape for nQ=1..4, radius=0.35, angles in [0, pi/2] equally spaced.
    Then replicate in all 4 quadrants & shift by (+0.5, +0.5).
    shapeIndex 1..4 calls this.
    """
    r = 0.35
    if   nQ==1: deg_list = [0.0]
    elif nQ==2: deg_list = [0.0, 45.0]
    elif nQ==3: deg_list = [0.0, 30.0, 60.0]
    elif nQ==4: deg_list = [0.0, 22.5, 45.0, 67.5]
    else:
        raise ValueError("nQ must be 1..4 for generate_c4_regular")

    pts_1Q = []
    for deg in deg_list:
        rad = math.radians(deg)
        x   = r*math.cos(rad)
        y   = r*math.sin(rad)
        pts_1Q.append((x,y))

    def rot90(x,y):
        return (-y,x)

    all_pts = list(pts_1Q)
    q2 = [rot90(px,py) for (px,py) in pts_1Q]
    q3 = [rot90(px,py) for (px,py) in q2]
    q4 = [rot90(px,py) for (px,py) in q3]
    all_pts.extend(q2)
    all_pts.extend(q3)
    all_pts.extend(q4)

    shifted = [(xx+0.5, yy+0.5) for (xx,yy) in all_pts]
    poly_str = ";".join(f"{p[0]:.6f},{p[1]:.6f}" for p in shifted)
    return poly_str, shifted

def generate_c4_random(nQ=1, seed=123, rmin=0.1, rmax=0.3):
    """
    "Random" shape for nQ=1..4, angles in [0, pi/2] equally spaced,
    radius random in [rmin..rmax], replicate 4 quadrants, shift by +0.5.
    shapeIndex #5..8 calls this, each with different seeds for reproducibility.
    """
    np.random.seed(seed)
    if   nQ==1: deg_list = [0.0]
    elif nQ==2: deg_list = [0.0, 45.0]
    elif nQ==3: deg_list = [0.0, 30.0, 60.0]
    elif nQ==4: deg_list = [0.0, 22.5, 45.0, 67.5]
    else:
        raise ValueError("nQ must be 1..4 for generate_c4_random")

    radii = np.random.uniform(rmin, rmax, len(deg_list))
    pts_1Q = []
    for i, deg in enumerate(deg_list):
        rad = math.radians(deg)
        r_  = radii[i]
        x   = r_*math.cos(rad)
        y   = r_*math.sin(rad)
        pts_1Q.append((x,y))

    def rot90(x,y):
        return (-y,x)

    all_pts = list(pts_1Q)
    q2 = [rot90(px,py) for (px,py) in pts_1Q]
    q3 = [rot90(px,py) for (px,py) in q2]
    q4 = [rot90(px,py) for (px,py) in q3]
    all_pts.extend(q2)
    all_pts.extend(q3)
    all_pts.extend(q4)

    shifted = [(xx+0.5, yy+0.5) for (xx,yy) in all_pts]
    poly_str = ";".join(f"{p[0]:.6f},{p[1]:.6f}" for p in shifted)
    return poly_str, shifted

##############################################################################
# 2) S4 runner => reflection
##############################################################################
def run_s4_and_get_R(poly_str, c_val, outfolder="myresults", verbose=False):
    """
    Runs S4 => reflection, moves CSV => outfolder, returns (lam[], R[]).
    """
    lua_script = "metasurface_fixed_shape_and_c_value.lua"
    cmd = f'../build/S4 -a "{poly_str} -c {c_val} -v -s" {lua_script}'
    if verbose:
        print("[S4 cmd]", cmd)
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode!=0:
        if verbose:
            print("[ERROR] S4 failed.")
            print("STDOUT:", proc.stdout)
            print("STDERR:", proc.stderr)
        return [],[]

    # parse "Saved to"
    saved_path = None
    for line in proc.stdout.splitlines():
        if "Saved to" in line:
            saved_path = line.split("Saved to",1)[1].strip()
            break

    # fallback
    if not saved_path or not os.path.isfile(saved_path):
        pattern = f"results/fixed_shape_c{c_val:.1f}_*.csv"
        found   = glob.glob(pattern)
        if found:
            saved_path = max(found, key=os.path.getctime)
    if not saved_path or not os.path.isfile(saved_path):
        if verbose:
            print("[WARN] No CSV for c=%.1f" % c_val)
        return [],[]

    os.makedirs(outfolder, exist_ok=True)
    new_path = os.path.join(outfolder, os.path.basename(saved_path))
    try:
        shutil.move(saved_path, new_path)
    except:
        pass

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
    data = sorted(zip(lam_list,R_list), key=lambda x:x[0])
    lam_list = [d[0] for d in data]
    R_list   = [d[1] for d in data]
    return lam_list, R_list

##############################################################################
# 3) Build 8 shapes => shapeIndex=1..8
##############################################################################
def build_8_shapes(seed_base=999):
    """
    shapeIndex=1..4 => generate_c4_regular(nQ=1..4)
    shapeIndex=5..8 => generate_c4_random(nQ=1..4) with seeds = seed_base + offset
    """
    shapes = {}
    # 1..4 => regular
    for i,nQ in enumerate([1,2,3,4], start=1):
        poly, pts = generate_c4_regular(nQ)
        shapes[i] = (poly, pts)

    # 5..8 => random, each with seed= seed_base+(i-5)
    for i,nQ in enumerate([1,2,3,4], start=5):
        seed_ = seed_base + (i-5)
        poly, pts = generate_c4_random(nQ, seed=seed_, rmin=0.1, rmax=0.3)
        shapes[i] = (poly, pts)

    return shapes

##############################################################################
# 4) Plot each shape => shapes_plots/shape_i.png
##############################################################################
def plot_shape(points, shape_idx):
    """
    Save a small figure shapes_plots/shape_{shape_idx}.png
    """
    os.makedirs("shapes_plots", exist_ok=True)
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    x_closed = x + [x[0]]
    y_closed = y + [y[0]]

    plt.figure(figsize=(3,3))
    plt.plot(x_closed, y_closed, 'o-', color='black')
    plt.axis('equal')
    plt.title(f"Shape #{shape_idx}", fontsize=10)
    outname = f"shapes_plots/shape_{shape_idx}.png"
    plt.savefig(outname, dpi=120, bbox_inches="tight")
    plt.close()

##############################################################################
# 5) Build tasks => 4 columns
##############################################################################
def parallel_tasks(nq_chosen, seed_base):
    """
    11 c in [0..1].
    columns:
      col0 => 4 "regular" shapes (#1..4)
      col1 => 4 "random"  shapes (#5..8)
      col2 => rotation => 11 angles for shape #nq_chosen
      col3 => scale => 11 factors for shape #nq_chosen
    """
    c_vals = [round(x,1) for x in np.linspace(0,1,11)]
    shapes = build_8_shapes(seed_base=seed_base)

    # Also, let's plot each shape once:
    for i in range(1,9):
        polyS, pts = shapes[i]
        plot_shape(pts, i)

    # If shapeIndex <=4 => nQ= that index
    # If shapeIndex >=5 => nQ= shapeIndex-4
    if nq_chosen<=4:
        base_nQ = nq_chosen
    else:
        base_nQ = nq_chosen-4

    angle_list = np.linspace(0, math.pi/(2*base_nQ), 11)
    scale_list = np.linspace(0.5,1.5, 11)

    # we'll get the chosen shape:
    chosen_poly, chosen_pts = shapes[nq_chosen]

    def rotate_polygon(pts,theta):
        rpts=[]
        for (x,y) in pts:
            xr = x*math.cos(theta) - y*math.sin(theta)
            yr = x*math.sin(theta) + y*math.cos(theta)
            rpts.append((xr,yr))
        return ";".join(f"{xx:.6f},{yy:.6f}" for (xx,yy) in rpts)

    def scale_polygon(pts,factor):
        spts=[]
        for (x,y) in pts:
            spts.append((factor*x, factor*y))
        return ";".join(f"{xx:.6f},{yy:.6f}" for (xx,yy) in spts)

    tasks=[]
    # col0 => shapeIndex=1..4
    for c_ in c_vals:
        for shape_i in [1,2,3,4]:
            tasks.append({
                'key': (0,c_,shape_i),
                'c_val': c_,
                'poly_str': shapes[shape_i][0]
            })
    # col1 => shapeIndex=5..8
    for c_ in c_vals:
        for shape_i in [5,6,7,8]:
            tasks.append({
                'key': (1,c_,shape_i),
                'c_val': c_,
                'poly_str': shapes[shape_i][0]
            })
    # col2 => rotation => iang => 11 angles
    for c_ in c_vals:
        for iang,theta in enumerate(angle_list):
            polR = rotate_polygon(chosen_pts, theta)
            tasks.append({
                'key': (2,c_, iang),
                'c_val': c_,
                'poly_str': polR
            })
    # col3 => scale => iscl => 11 scale
    for c_ in c_vals:
        for iscl,s_ in enumerate(scale_list):
            polS = scale_polygon(chosen_pts, s_)
            tasks.append({
                'key': (3,c_, iscl),
                'c_val': c_,
                'poly_str': polS
            })

    return tasks, angle_list, scale_list


def run_one_task(task):
    key = task['key']
    lam,R = run_s4_and_get_R(task['poly_str'], task['c_val'],
                             outfolder="myresults", verbose=False)
    return (key, lam, R)

##############################################################################
# 6) Main => produce 11×4 figure
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nq", type=int, default=4,
                        help="Which shape in [1..8] to use for rotation & scale (col2,col3).")
    parser.add_argument("--seed", type=int, default=999,
                        help="Base random seed for shapes #5..8.")
    parser.add_argument("--max-workers",type=int,default=8)
    parser.add_argument("--verbose",action="store_true")
    args = parser.parse_args()

    if args.nq<1 or args.nq>8:
        raise ValueError("--nq must be 1..8")

    sns.set_style("white")
    plt.rcParams["axes.grid"]=False
    plt.rcParams["axes.spines.top"]=False
    plt.rcParams["axes.spines.right"]=False
    plt.rcParams["axes.spines.left"]=False
    plt.rcParams["axes.spines.bottom"]=False

    # build tasks => also prints each of the 8 shapes
    tasks, angle_list, scale_list = parallel_tasks(args.nq, seed_base=args.seed)
    print(f"[INFO] We have {len(tasks)} tasks total. Running in parallel...")

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

    print("[INFO] All tasks done. Building final figure...")

    c_vals = [round(x,1) for x in np.linspace(0,1,11)]
    fig, axes = plt.subplots(nrows=len(c_vals), ncols=4,
                             figsize=(20,2.4*len(c_vals)), # adjust as needed
                             sharex=False, sharey=False)

    # Column0 => 4 "regular" => shapeIdx=1..4
    col0_colors = ["red","green","blue","purple"]
    # Column1 => 4 "random" => shapeIdx=5..8
    col1_colors = ["orange","darkcyan","magenta","brown"]
    # Column2 => rotation => len(angle_list)
    col2_colors = plt.cm.viridis(np.linspace(0,1,len(angle_list)))
    # Column3 => scale => len(scale_list)
    col3_colors = plt.cm.viridis(np.linspace(0,1,len(scale_list)))

    for row_i, c_ in enumerate(c_vals):
        ax0, ax1, ax2, ax3 = axes[row_i]

        ax0.set_axis_off()
        ax1.set_axis_off()
        ax2.set_axis_off()
        ax3.set_axis_off()

        # col0 => shapeIndex=1..4
        for iSh in [1,2,3,4]:
            lam,R = results_dict.get((0,c_,iSh), ([],[]))
            if lam:
                color_ = col0_colors[iSh-1]
                ax0.plot(lam,R,color=color_,linewidth=1.5)

        # col1 => shapeIndex=5..8
        for iSh in [5,6,7,8]:
            lam,R = results_dict.get((1,c_,iSh),([],[]))
            if lam:
                color_ = col1_colors[iSh-5]
                ax1.plot(lam,R,color=color_,linewidth=1.5)

        # col2 => rotation
        for iang in range(len(angle_list)):
            lam,R = results_dict.get((2,c_,iang),([],[]))
            if lam:
                ax2.plot(lam,R,color=col2_colors[iang],linewidth=1.5)

        # col3 => scale
        for iscl in range(len(scale_list)):
            lam,R = results_dict.get((3,c_,iscl),([],[]))
            if lam:
                ax3.plot(lam,R,color=col3_colors[iscl],linewidth=1.5)

    fig.tight_layout()
    outname = "four_col_plot_minimal_axes.png"
    plt.savefig(outname,dpi=150)
    print(f"[INFO] Done. Saved figure => {outname}")
    plt.show()

if __name__=="__main__":
    main()

