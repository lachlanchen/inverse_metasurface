#!/usr/bin/env python3
"""
compare_all_c_in_single_figure.py

Usage examples:
  1) python compare_all_c_in_single_figure.py --merged merged_s4_shapes_20250119_153038.csv --c all
     => For c in [0.0, 0.1, 0.2, ... 1.0], read the merged data, optionally run S4,
        read the results CSV, and make a single figure with up to 11 rows, 3 columns.

  2) python compare_all_c_in_single_figure.py --merged merged_s4_shapes_20250119_153038.csv --c 0.3
     => Only do c=0.3, produce a single row of subplots (Reflection, Transmission, R+T).

By default, this script:
  - (Optional) runs the S4 simulation for the chosen c-values.  (You can disable if you want.)
  - Reads the "merged" dataset for each c-value.
  - Reads the newly produced CSV from S4's run, named like "results/fixed_shape_cX.Y_YYYYmmdd_HHMMSS.csv"
    (We detect the exact path from the console output, or reconstruct if you prefer.)
  - Plots them in one big figure: each row => a c value, with columns for (R) / (T) / (|R|+T).

Change the code in "run_s4_for_c()" if you want to handle shape indices differently or skip the run.
"""

import os
import argparse
import subprocess
import csv
import glob
import math
from datetime import datetime

import matplotlib.pyplot as plt

###############################################################################
# 1) UTILITY: read the big merged CSV for a given c_value
###############################################################################
def read_merged_csv_for_c(merged_csv_path, c_value):
    """
    Finds the row in merged_s4_shapes_... where 'c' == c_value.
    Extracts reflection columns (R@...) and transmission columns (T@...).
    Returns: wave_m, R_m, T_m (lists).
    """
    wave_m = []
    R_m = []
    T_m = []
    
    with open(merged_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        # Identify which columns are R@..., T@...
        wave_for_R = {}
        wave_for_T = {}
        for h in headers:
            if h.startswith("R@"):
                try:
                    lam = float(h.split("@")[1])
                    wave_for_R[h] = lam
                except:
                    pass
            elif h.startswith("T@"):
                try:
                    lam = float(h.split("@")[1])
                    wave_for_T[h] = lam
                except:
                    pass
        
        found = False
        for row in reader:
            c_str = row.get("c", None)
            if not c_str: 
                continue
            try:
                c_val = float(c_str)
            except:
                continue
            if abs(c_val - c_value) < 1e-9:
                # parse
                # gather (lam, R, T)
                # We only do columns that appear in wave_for_R or wave_for_T
                data_list = []
                all_keys = sorted(set(wave_for_R.keys()).union(wave_for_T.keys()),
                                  key=lambda k: wave_for_R.get(k, wave_for_T.get(k,999999)))
                for k in all_keys:
                    lam = wave_for_R.get(k, None)
                    if lam is None:
                        lam = wave_for_T.get(k, None)
                    # Get R val if k in wave_for_R
                    Rval = None
                    if k in wave_for_R:
                        try:
                            Rval = float(row[k])
                        except:
                            Rval = 0.0
                    Tval = None
                    # We also try the T column with the same wavelength:
                    # Because for the merged CSV, if k is "R@1.040", then T col is "T@1.040"
                    # We'll do a second approach: 'T@%.3f' % lam
                    Tcol = f"T@{lam:.3f}"
                    if Tcol in row:
                        try:
                            Tval = float(row[Tcol])
                        except:
                            Tval = 0.0
                    else:
                        # fallback if the same col is wave_for_T, etc.
                        pass
                    data_list.append((lam, Rval, Tval))
                data_list.sort(key=lambda x: x[0])
                wave_m = [d[0] for d in data_list]
                R_m    = [d[1] for d in data_list]
                T_m    = [d[2] for d in data_list]
                found = True
                break
        if not found:
            print(f"[WARN] Could not find c={c_value} in {merged_csv_path}. Return empty.")
    return wave_m, R_m, T_m

###############################################################################
# 2) UTILITY: run S4 for a single c_value, capturing the path to the results CSV
###############################################################################
def run_s4_for_c(polygon, c_value):
    """
    Example: run the S4 simulation for a single c_value with:
        ../build/S4 -a "<poly> -c <c_val> -v -s" metasurface_fixed_shape_and_c_value.lua

    We'll parse stdout looking for "Saved to X" so we get the CSV path.

    Return: path_to_csv or None
    """
    arg_string = f'"{polygon} -c {c_value} -v -s"'
    cmd = f'../build/S4 -a {arg_string} metasurface_fixed_shape_and_c_value.lua'
    print(f"[INFO] Running: {cmd}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[ERROR] S4 run failed. stdout/stderr below:")
        print(proc.stdout)
        print(proc.stderr)
        return None
    
    saved_csv = None
    for line in proc.stdout.splitlines():
        if "Saved to " in line:
            # e.g. line = "Saved to results/fixed_shape_c0.1_20250123_101112.csv"
            saved_csv = line.split("Saved to",1)[1].strip()
            break
    return saved_csv

###############################################################################
# 3) UTILITY: read the results CSV that was produced by the Lua script
###############################################################################
def read_results_csv(csv_path):
    """
    Return wave_r, R_r, T_r
    """
    wave_r, R_r, T_r = [], [], []
    if not csv_path or not os.path.exists(csv_path):
        print(f"[WARN] results CSV missing? {csv_path}")
        return wave_r, R_r, T_r
    
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lam = float(row["wavelength_um"])
                Rv  = float(row["R"])
                Tv  = float(row["T"])
                wave_r.append(lam)
                R_r.append(Rv)
                T_r.append(Tv)
            except:
                pass
    # Sort by lam
    data = sorted(zip(wave_r, R_r, T_r), key=lambda x: x[0])
    wave_r = [d[0] for d in data]
    R_r    = [d[1] for d in data]
    T_r    = [d[2] for d in data]
    return wave_r, R_r, T_r

###############################################################################
# 4) MAIN: parse arguments, figure out which c-values, run or not run S4,
#    read data, produce single figure with subplots (nrows x 3).
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", required=True,
                        help="Path to merged_s4_shapes.csv (the big wide CSV).")
    parser.add_argument("--c", default="all",
                        help="Either 'all' or a specific float (like 0.3).")
    parser.add_argument("--run-s4", action="store_true",
                        help="If given, we actually run the S4 command for each c. Otherwise we skip.")
    parser.add_argument("--polygon", default="0.162926,0.189418;-0.189418,0.162926;-0.162926,-0.189418;0.189418,-0.162926",
                        help="Polygon string for the shape. Default is a 4-vertex example.")
    args = parser.parse_args()
    
    merged_csv_path = args.merged
    run_s4 = args.run_s4
    polygon_str = args.polygon

    # Decide which c-values
    if args.c.lower() == "all":
        # We'll gather partial_crys_data files
        cvals = []
        for path in glob.glob("partial_crys_data/partial_crys_C*.csv"):
            # parse out the c from partial_crys_CX.csv
            # e.g. partial_crys_data/partial_crys_C0.1.csv
            base = os.path.basename(path)
            # partial_crys_C0.1.csv
            val_str = base.replace("partial_crys_C","").replace(".csv","")
            try:
                cval = float(val_str)
                cvals.append(cval)
            except:
                pass
        cvals = sorted(cvals)
        if not cvals:
            print("[ERROR] No c-values found in partial_crys_data. Exiting.")
            return
    else:
        # single c
        try:
            single_c = float(args.c)
            cvals = [single_c]
        except:
            print("[ERROR] Could not parse '--c' as float or 'all'. Exiting.")
            return
    
    # We'll create a single figure with len(cvals) rows, each row has 3 columns: R / T / R+T
    nrows = len(cvals)
    ncols = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 3*nrows),
                             sharex=False, sharey=False)
    # If only one c => axes is a 1D array if nrows=1, or might be just a 2D anyway. Let's unify logic
    if nrows == 1:
        axes = [axes]  # so axes[0] => the row

    # For each c, optionally run S4, read merged data, read results data, plot
    results_paths = []
    for idx, c_val in enumerate(cvals):
        # 1) run s4 if desired
        csv_result_path = None
        if run_s4:
            csv_result_path = run_s4_for_c(polygon_str, c_val)
        else:
            # We guess the file?  "results/fixed_shape_c0.1_*.csv" => the most recent?
            # Or none. We'll do a quick guess with a wildcard:
            pat = f"results/fixed_shape_c{c_val:.1f}_*.csv"
            found = glob.glob(pat)
            if found:
                csv_result_path = max(found, key=os.path.getmtime)  # pick latest
            else:
                csv_result_path = None
        
        # 2) read merged data
        wave_m, R_m, T_m = read_merged_csv_for_c(merged_csv_path, c_val)
        # We'll do absolute value of R in case negative
        R_m_abs = [abs(x) for x in R_m]
        RTplus_m = [abs(r)+t for r,t in zip(R_m, T_m)]

        # 3) read result data
        wave_r, R_r, T_r = [], [], []
        if csv_result_path:
            wave_r, R_r, T_r = read_results_csv(csv_result_path)
        # also do abs for R
        R_r_abs = [abs(x) for x in R_r]
        RTplus_r = [abs(r)+t for r,t in zip(R_r, T_r)]

        # 4) plot in row=idx, col=0 => reflection
        ax_ref = axes[idx][0] if nrows>1 else axes[0]
        ax_trn = axes[idx][1] if nrows>1 else axes[1]
        ax_sum = axes[idx][2] if nrows>1 else axes[2]

        # Reflection
        if wave_m:
            ax_ref.plot(wave_m, R_m_abs, 'r--', label="Merged |R|")
        if wave_r:
            ax_ref.plot(wave_r, R_r_abs, 'r-', label="New |R|")
        ax_ref.set_title(f"c={c_val:.1f} Reflection")
        ax_ref.set_xlabel("Wavelength (um)")
        ax_ref.set_ylabel("|R|")

        # Transmission
        if wave_m:
            ax_trn.plot(wave_m, T_m, 'b--', label="Merged T")
        if wave_r:
            ax_trn.plot(wave_r, T_r, 'b-', label="New T")
        ax_trn.set_title(f"c={c_val:.1f} Transmission")
        ax_trn.set_xlabel("Wavelength (um)")
        ax_trn.set_ylabel("T")

        # R+T
        if wave_m:
            ax_sum.plot(wave_m, RTplus_m, 'g--', label="Merged |R|+T")
        if wave_r:
            ax_sum.plot(wave_r, RTplus_r, 'g-', label="New |R|+T")
        ax_sum.set_title(f"c={c_val:.1f} (|R|+T)")
        ax_sum.set_xlabel("Wavelength (um)")
        ax_sum.set_ylabel("|R|+T")

        # put legends in the first row if you like
        if idx == 0:
            ax_ref.legend()
            ax_trn.legend()
            ax_sum.legend()

    # overall layout
    plt.tight_layout()

    # Make an output folder for the figure
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"single_figure_{dt_str}"
    os.makedirs(outdir, exist_ok=True)
    outpng = os.path.join(outdir, "comparison.png")
    plt.savefig(outpng, dpi=150)
    print(f"[INFO] Figure saved to {outpng}")

    # show popup
    plt.show()

if __name__ == "__main__":
    main()

