#!/usr/bin/env python3
"""
check_dataset.py

Usage examples:
  1) python check_dataset.py
     => automatically picks the first valid (NQ, shape_idx) from the CSV, sets c=all,
        runs S4 for each c, plots one figure with multiple rows (up to 11 if c=0..1.0).

  2) python check_dataset.py --nq 4 --shape-idx 1705 --c 0.5
     => runs for that NQ=4, shape_idx=1705, only c=0.5, produces 1-row x 3-col plot.

  3) python check_dataset.py --nq 4 --shape-idx 1705 --c all
     => does c in [0.0, 0.1, ..., 1.0] that exist in the CSV for that shape,
        then produces an Nrows x 3 figure.

It will:
 - Parse the big merged CSV (e.g. "merged_s4_shapes_20250119_153038.csv").
 - Find the row(s) that match (NQ, shape_idx) and each c, extracting reflection/transmission.
 - Grab the polygon from "vertices_str" in that row to pass to S4.
 - Run S4 for each c, store results, read them, and compare vs. merged data.
 - Plot them all in a single figure: each row => one c; columns => R, T, (|R|+T).
 - Two lines in each subplot => merged data line & newly-run S4 line.

You can modify file paths, CSV names, or disable 'run_s4_for_c()' if you already have the results.
"""

import os
import sys
import csv
import glob
import random
import argparse
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt

##############################################################################
# 1) PARSE the big merged CSV, find rows for (NQ, shape_idx), get polygons, etc
##############################################################################
def read_merged_csv(merged_csv_path):
    """
    Return a list of dict rows from the CSV:
      each row has keys: folder_key,NQ,nS,shape_idx,c, R@..., T@..., vertices_str, ...
      We'll store them so we can later filter.
    """
    rows = []
    with open(merged_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert some columns to float if convenient
            # For example, row["c"], row["NQ"], row["shape_idx"] if present
            try:
                row["c"] = float(row["c"])
            except:
                pass
            try:
                row["NQ"] = float(row["NQ"])
            except:
                pass
            try:
                row["shape_idx"] = int(row["shape_idx"])
            except:
                pass
            rows.append(row)
    return rows

def find_rows_for_nq_shape(rows, nq_val, shape_idx_val):
    """
    Return all rows that match (NQ==nq_val, shape_idx==shape_idx_val).
    """
    selected = []
    for r in rows:
        if "NQ" in r and "shape_idx" in r:
            if abs(r["NQ"] - nq_val) < 1e-9 and r["shape_idx"] == shape_idx_val:
                selected.append(r)
    return selected

##############################################################################
# 2) EXTRACT reflection/transmission from a single "merged row" dict
##############################################################################
def parse_merged_row_for_RT(row):
    """
    The row has many columns: c, R@..., T@..., etc.
    We want wave_list, R_list, T_list from that single row.
    """
    wave_list = []
    R_list = []
    T_list = []
    
    # gather the columns that look like "R@1.040", "T@1.040", ...
    waves_for_R = {}
    waves_for_T = {}
    
    for k,v in row.items():
        if k.startswith("R@"):
            lam_str = k.split("@")[1]
            try:
                lam_val = float(lam_str)
                # store reflection in waves_for_R
                # parse v into float
                r_val = float(v)
                waves_for_R[lam_val] = r_val
            except:
                pass
        elif k.startswith("T@"):
            lam_str = k.split("@")[1]
            try:
                lam_val = float(lam_str)
                t_val = float(v)
                waves_for_T[lam_val] = t_val
            except:
                pass
    
    # unify the set of wavelengths
    all_lams = sorted(set(waves_for_R.keys()).union(waves_for_T.keys()))
    for lam in all_lams:
        Rv = waves_for_R.get(lam, None)
        Tv = waves_for_T.get(lam, None)
        wave_list.append(lam)
        R_list.append(Rv)
        T_list.append(Tv)
    return wave_list, R_list, T_list

##############################################################################
# 3) RUN S4 for a single c, polygon, store the results CSV path
##############################################################################
def run_s4_for_c(polygon_str, c_val):
    """
    Runs e.g.:
      ../build/S4 -a "<polygon_str> -c <c_val> -v -s" metasurface_fixed_shape_and_c_value.lua
    Then parse stdout to find "Saved to results/fixed_shape_cXYZ_timestamp.csv"
    Return that path or None.
    """
    cmd = (
        f'../build/S4 '
        f'-a "{polygon_str} -c {c_val} -v -s" '
        f'metasurface_fixed_shape_and_c_value.lua'
    )
    print(f"[INFO] Running: {cmd}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[ERROR] S4 run failed!")
        print("=== STDOUT ===")
        print(proc.stdout)
        print("=== STDERR ===")
        print(proc.stderr)
        return None
    
    # parse the line "Saved to X"
    saved_path = None
    for line in proc.stdout.splitlines():
        if "Saved to " in line:
            # e.g. line = "Saved to results/fixed_shape_c0.3_20250123_153038.csv"
            saved_path = line.split("Saved to",1)[1].strip()
            break
    return saved_path

##############################################################################
# 4) READ the newly-run results from that CSV
##############################################################################
def read_results_csv(csv_path):
    """Return (wave_list, R_list, T_list) from the results CSV of the Lua script."""
    wv, Rv, Tv = [], [], []
    if not csv_path or not os.path.exists(csv_path):
        return wv, Rv, Tv
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lam = float(row["wavelength_um"])
                R_ = float(row["R"])
                T_ = float(row["T"])
                wv.append(lam)
                Rv.append(R_)
                Tv.append(T_)
            except:
                pass
    # sort by lam
    data = sorted(zip(wv,Rv,Tv), key=lambda x:x[0])
    wv = [d[0] for d in data]
    Rv = [d[1] for d in data]
    Tv = [d[2] for d in data]
    return wv, Rv, Tv

##############################################################################
# MAIN
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-csv", default="merged_s4_shapes_20250119_153038.csv",
                        help="Path to the big merged CSV file.")
    parser.add_argument("--nq", type=float, default=None,
                        help="NQ value to filter. If not given, pick a random from the CSV.")
    parser.add_argument("--shape-idx", type=int, default=None,
                        help="shape_idx to filter. If not given, pick a random from the CSV.")
    parser.add_argument("--c", default="all",
                        help="Either 'all' or a single float. If not given => 'all'.")
    parser.add_argument("--run-s4", action="store_true",
                        help="If set, we run the S4 command for each c. Otherwise, you might skip.")
    args = parser.parse_args()
    
    merged_csv_path = args.merged_csv
    c_arg = args.c
    run_s4 = True
    
    # Load all rows from the merged CSV
    all_rows = read_merged_csv(merged_csv_path)
    if not all_rows:
        print("[ERROR] No data in merged CSV. Exiting.")
        return
    
    # If user didn't specify NQ or shape_idx, pick from the CSV at random or minimal
    if args.nq is None or args.shape_idx is None:
        # pick a random row
        row0 = random.choice(all_rows)
        chosen_nq = row0["NQ"]
        chosen_shape = row0["shape_idx"]
        print(f"[INFO] Using NQ={chosen_nq}, shape_idx={chosen_shape}.")
    else:
        chosen_nq = float(args.nq)
        chosen_shape = args.shape_idx
    
    # Filter rows that match (NQ=..., shape_idx=...)
    matching_rows = find_rows_for_nq_shape(all_rows, chosen_nq, chosen_shape)
    if not matching_rows:
        print(f"[ERROR] No rows found for NQ={chosen_nq}, shape_idx={chosen_shape}.")
        return
    
    # Each row might have a different c, but the same shape. We assume the shape is the same polygon
    # We'll parse out 'vertices_str' from the first row. (All matching rows presumably have same polygon.)
    polygon_str = matching_rows[0].get("vertices_str", "")
    if not polygon_str:
        print("[ERROR] No vertices_str found in that row. Cannot run S4.")
        return
    
    # Decide which c-values
    if c_arg.lower() == "all":
        # gather all c from these matching_rows
        cvals = sorted({r["c"] for r in matching_rows})
    else:
        # single c
        try:
            cval = float(c_arg)
            # ensure it is in matching rows
            row_cs = sorted({r["c"] for r in matching_rows})
            if cval not in row_cs:
                print(f"[WARN] c={cval} not found in the CSV for that shape. We'll still attempt.")
                # or we can forcibly add it if you want
                row_cs.append(cval)
            cvals = [cval]
        except:
            print("[ERROR] Could not parse --c as float or 'all'. Exiting.")
            return
    
    # We will produce a single figure with len(cvals) rows, each row has 3 columns
    nrows = len(cvals)
    ncols = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,6*nrows),
                             sharex=False, sharey=False)
    if nrows == 1:
        axes = [axes]  # so that axes[0][0], axes[0][1], axes[0][2] are valid

    # For each c, we find the row in matching_rows that has that c (there should be exactly 1),
    # parse the merged data for R/T, run s4, read results, plot.
    for idx, c_val in enumerate(cvals):
        # find the row with c_val
        row_for_c = None
        for rr in matching_rows:
            if abs(rr["c"] - c_val) < 1e-9:
                row_for_c = rr
                break
        if not row_for_c:
            print(f"[WARN] No row for c={c_val} in the CSV. We'll skip.")
            continue
        
        # parse merged data
        wave_m, R_m, T_m = parse_merged_row_for_RT(row_for_c)
        # We'll do abs(R)
        R_m_abs = [abs(x) if x is not None else None for x in R_m]
        RT_m = [ (abs(r) if r else 0)+(t if t else 0) for r,t in zip(R_m,T_m)]
        
        # (optionally) run S4
        results_csv_path = None
        if run_s4:
            # run S4
            results_csv_path = run_s4_for_c(polygon_str, c_val)
        else:
            # or guess we pick the most recent "results/fixed_shape_cX.Y_*.csv"
            # let’s do:
            pat = f"results/fixed_shape_c{c_val:.1f}_*.csv"
            found = glob.glob(pat)
            if found:
                results_csv_path = max(found, key=os.path.getmtime)
        
        # read the newly-run results
        wave_r, R_r, T_r = [], [], []
        if results_csv_path:
            wave_r, R_r, T_r = read_results_csv(results_csv_path)
        # do abs
        R_r_abs = [abs(x) for x in R_r]
        RT_r = [abs(r)+t for r,t in zip(R_r, T_r)]
        
        # Now we plot in row=idx
        ax_ref = axes[idx][0]
        ax_trn = axes[idx][1]
        ax_sum = axes[idx][2]
        
        # Plot reflection
        if wave_m and R_m_abs:
            ax_ref.plot(wave_m, R_m_abs, 'r--', label="Merged |R|")
        if wave_r and R_r_abs:
            ax_ref.plot(wave_r, R_r_abs, 'r-', label="Newly-run |R|")
        ax_ref.set_title(f"c={c_val:.1f} Reflection")
        ax_ref.set_xlabel("Wavelength (um)")
        ax_ref.set_ylabel("|R|")
        
        # Plot transmission
        if wave_m and T_m:
            ax_trn.plot(wave_m, T_m, 'b--', label="Merged T")
        if wave_r and T_r:
            ax_trn.plot(wave_r, T_r, 'b-', label="Newly-run T")
        ax_trn.set_title(f"c={c_val:.1f} Transmission")
        ax_trn.set_xlabel("Wavelength (um)")
        ax_trn.set_ylabel("T")
        
        # Plot (|R| + T)
        if wave_m and RT_m:
            ax_sum.plot(wave_m, RT_m, 'g--', label="Merged |R|+T")
        if wave_r and RT_r:
            ax_sum.plot(wave_r, RT_r, 'g-', label="Newly-run |R|+T")
        ax_sum.set_title(f"c={c_val:.1f} (|R|+T)")
        ax_sum.set_xlabel("Wavelength (um)")
        ax_sum.set_ylabel("|R|+T")
        
        # legend only on the top row
        if idx == 0:
            ax_ref.legend()
            ax_trn.legend()
            ax_sum.legend()
    
    plt.tight_layout()
    
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"check_dataset_plots_{dt_str}"
    os.makedirs(outdir, exist_ok=True)
    outpng = os.path.join(outdir, f"comparison_NQ{chosen_nq}_shape{chosen_shape}.png")
    # plt.tight_layout()
    plt.savefig(outpng, dpi=250)
    print(f"[INFO] Figure saved to {outpng}")
    plt.show()

if __name__ == "__main__":
    main()

