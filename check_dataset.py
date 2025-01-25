#!/usr/bin/env python3
"""
check_dataset.py

Example usage:
  python check_dataset.py --c 0.1 --nq 4 --shape-idx 7
    => Runs the S4 command for that c=0.1, with a certain shape
       (hard-coded polygon or from your logic), then reads the output CSV
       to plot Reflection and Transmission.

If --c is not specified, we will iterate over all c-values found in partial_crys_data/.

If --nq or --shape-idx are not specified, we'll parse the merged_s4_shapes_*.csv to find
valid min..max ranges, pick a random valid one, etc.

Then we create a neat folder with datetime to store plots, or we can just show them in a popup.

Make sure you have installed matplotlib for Python. 
"""
import os
import argparse
import glob
import random
import re
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run S4 command for certain c/nq/shape, then plot.")
    parser.add_argument("--c", type=float, default=None,
                        help="Crystallization fraction c. If omitted, we do all c from partial_crys_data.")
    parser.add_argument("--nq", type=int, default=None,
                        help="Number of vertices per quarter, i.e. N_quarter. If omitted, pick random from dataset.")
    parser.add_argument("--shape-idx", type=int, default=None,
                        help="Shape index. If omitted, pick random from dataset.")
    return parser.parse_args()

def find_all_c_values():
    """
    Look in partial_crys_data/ for filenames like partial_crys_C0.1.csv 
    and extract 0.1 as a float. Return them as a sorted list.
    """
    all_csv = glob.glob("partial_crys_data/partial_crys_C*.csv")
    c_vals = []
    pattern = re.compile(r"partial_crys_C([\d\.]+)\.csv$")
    for path in all_csv:
        m = pattern.search(path)
        if m:
            c_str = m.group(1)
            try:
                c_val = float(c_str)
                c_vals.append(c_val)
            except ValueError:
                pass
    return sorted(c_vals)

def find_minmax_shape_nq(merged_csv="merged_s4_shapes_20250119_153038.csv"):
    """
    Very rough example: parse the CSV to see min and max of shape_idx, NQ, etc.
    We'll find the min and max shape_idx, and min and max NQ.
    Return (min_nq, max_nq, min_shape, max_shape).
    """
    # This is simplistic. We read the 'NQ' and 'shape_idx' columns.
    min_nq, max_nq = 999999, -999999
    min_shape, max_shape = 999999, -999999
    
    if not os.path.exists(merged_csv):
        # fallback
        return (1, 10, 1, 100)
    
    import csv
    with open(merged_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nq_str = row.get("NQ", "")
            shape_str = row.get("shape_idx", "")
            try:
                nq_val = int(nq_str)
                shape_val = int(shape_str)
                if nq_val < min_nq: min_nq = nq_val
                if nq_val > max_nq: max_nq = nq_val
                if shape_val < min_shape: min_shape = shape_val
                if shape_val > max_shape: max_shape = shape_val
            except:
                pass
    if min_nq == 999999:  # means we found nothing
        return (1, 10, 1, 100)
    return (min_nq, max_nq, min_shape, max_shape)

def construct_polygon(nq, shape_idx):
    """
    Stub: Return a polygon string for the -a argument
    e.g. "x1,y1;x2,y2;...". 
    For example, we might read from some shape file, or do a fixed shape.
    
    We'll just return a fixed example for simplicity:
    "0.162926,0.189418;-0.189418,0.162926;-0.162926,-0.189418;0.189418,-0.162926"
    """
    # In principle, you could read from "shapes/whatever-nQ{nq}-nS{shape_idx}/outer_shape{shape_idx}.txt"
    # But here is just a static example.
    return "0.162926,0.189418;-0.189418,0.162926;-0.162926,-0.189418;0.189418,-0.162926"

def run_s4_sim(polygon_str, c_val):
    """
    Runs the S4 simulation using:
      ../build/S4 -a "<polygon_str> -c c_val -v -s" metasurface_fixed_shape_and_c_value.lua
    
    We'll capture the stdout to find the line "Saved to X" so we know the CSV path.
    
    Returns: path to the newly saved CSV, or None if not found.
    """
    # Build the command
    # e.g.  ../build/S4 -a "0.162926,0.189418;... -c 0.1 -v -s" metasurface_fixed_shape_and_c_value.lua
    arg_string = f'"{polygon_str} -c {c_val} -v -s"'
    cmd = f'../build/S4 -a {arg_string} metasurface_fixed_shape_and_c_value.lua'
    print("[INFO] Running:", cmd)
    
    # Run with subprocess, capturing stdout
    import subprocess
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    # Check if it ran OK
    if result.returncode != 0:
        print("[ERROR] S4 command failed! Output:")
        print(result.stdout)
        print(result.stderr)
        return None
    
    # We'll parse lines of stdout looking for "Saved to " or something
    saved_csv = None
    for line in result.stdout.splitlines():
        if "Saved to " in line:
            # e.g. "Saved to results/fixed_shape_c0.1_20250123_101112.csv"
            saved_csv = line.split("Saved to",1)[1].strip()
            break
    return saved_csv

def read_lua_results_csv(csv_path):
    """
    Quick parse of the results CSV. 
    Returns (wavelength[], R[], T[]) just for plotting.
    """
    import csv
    wave, Rvals, Tvals = [], [], []
    if not csv_path or not os.path.exists(csv_path):
        print(f"[WARNING] No results CSV found at '{csv_path}'.")
        return wave, Rvals, Tvals
    
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lam = float(row["wavelength_um"])
                rr  = float(row["R"])
                tt  = float(row["T"])
                wave.append(lam)
                Rvals.append(rr)
                Tvals.append(tt)
            except:
                pass
    return wave, Rvals, Tvals

def plot_results(wave, Rvals, Tvals, c_val):
    """
    Plot R and T vs. wavelength, pop up (plt.show), save in a folder with datetime.
    """
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"plot_check_{dt_str}"
    os.makedirs(outdir, exist_ok=True)
    
    import numpy as np
    wave_np = np.array(wave)
    R_np = np.array(Rvals)
    T_np = np.array(Tvals)
    
    # Sort by wavelength just in case
    idx = np.argsort(wave_np)
    wave_np = wave_np[idx]
    R_np = R_np[idx]
    T_np = T_np[idx]
    
    # Reflection
    plt.figure(figsize=(6,4.5))
    plt.plot(wave_np, R_np, 'r-o', label="R")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Reflection")
    plt.title(f"Reflection for c={c_val}")
    plt.legend()
    plt.tight_layout()
    reflect_png = os.path.join(outdir, f"Reflection_c{c_val}.png")
    plt.savefig(reflect_png, dpi=150)
    
    # Transmission
    plt.figure(figsize=(6,4.5))
    plt.plot(wave_np, T_np, 'b-s', label="T")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Transmission")
    plt.title(f"Transmission for c={c_val}")
    plt.legend()
    plt.tight_layout()
    trans_png = os.path.join(outdir, f"Transmission_c{c_val}.png")
    plt.savefig(trans_png, dpi=150)
    
    # Pop up the plots
    plt.show()

def main():
    args = parse_arguments()
    c_arg = args.c
    nq_arg = args.nq
    shape_arg = args.shape_idx
    
    # 1) If c_arg is None => gather all from partial_crys_data
    if c_arg is None:
        c_list = find_all_c_values()
        if not c_list:
            print("[ERROR] No c-values found in partial_crys_data/ . Exiting.")
            return
    else:
        c_list = [c_arg]
    
    # 2) If we need random shape or NQ:
    min_nq, max_nq, min_shape, max_shape = find_minmax_shape_nq()
    
    if nq_arg is None:
        nq_arg = random.randint(min_nq, max_nq)
    if shape_arg is None:
        shape_arg = random.randint(min_shape, max_shape)
    
    print(f"[INFO] Using NQ={nq_arg}, shape_idx={shape_arg}.")
    
    # 3) Construct the polygon from those or load from file
    polygon_str = construct_polygon(nq_arg, shape_arg)
    
    # 4) For each c in c_list, run S4
    for c_val in c_list:
        csv_path = run_s4_sim(polygon_str, c_val)
        if not csv_path:
            print(f"[WARNING] No CSV produced for c={c_val}. Skipping plot.")
            continue
        
        # 5) Read the results CSV
        wave, Rvals, Tvals = read_lua_results_csv(csv_path)
        if not wave:
            print(f"[WARNING] No data read from '{csv_path}'.")
            continue
        
        # 6) Plot and pop up
        plot_results(wave, Rvals, Tvals, c_val)

if __name__ == "__main__":
    main()

