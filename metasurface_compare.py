i#!/usr/bin/env python3
"""
metasurface_compare.py

Compares S4‐simulated R/T to your “merged” CSV results.

Usage example:
  python metasurface_compare.py \
    --merged_csv merged_s4_shapes_20250119_153038.csv \
    --c_val 0.0 \
    --shape_idx 1 \
    --s4cmd /path/to/S4 \
    --out_csv out_temp.csv \
    --debug

Steps:
  1) Loads your merged CSV, finds row(s) with matching (c, shape_idx).
  2) Extracts shape_str => "x1,y1;x2,y2;..." from the "vertices_str" column.
  3) Figures out partial_csv => partial_crys_data/partial_crys_C{c_val}.csv.
  4) Runs "S4 -a partial_csv=... shape_str=... out_csv=... metasurface_check.lua".
  5) Reads the resulting out_csv => columns (wavelength_um, R, T).
  6) Compares those to R@..., T@... columns in the merged CSV, and plots them.
  7) If --debug, prints additional details.
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Hard‐coded wavelengths in your merged CSV
WAVE_STR = [
    "1.040","1.054","1.069","1.084","1.099","1.113","1.128","1.143","1.158","1.173",
    "1.187","1.202","1.217","1.232","1.246","1.261","1.276","1.291","1.305","1.320",
    "1.335","1.350","1.365","1.379","1.394","1.409","1.424","1.438","1.453","1.468",
    "1.483","1.498","1.512","1.527","1.542","1.557","1.571","1.586","1.601","1.616",
    "1.630","1.645","1.660","1.675","1.690","1.704","1.719","1.734","1.749","1.763",
    "1.778","1.793","1.808","1.823","1.837","1.852","1.867","1.882","1.896","1.911",
    "1.926","1.941","1.956","1.970","1.985","2.000","2.015","2.029","2.044","2.059",
    "2.074","2.088","2.103","2.118","2.133","2.148","2.162","2.177","2.192","2.207",
    "2.221","2.236","2.251","2.266","2.281","2.295","2.310","2.325","2.340","2.354",
    "2.369","2.384","2.399","2.414","2.428","2.443","2.458","2.473","2.487","2.502"
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_csv", default="merged_s4_shapes_20250119_153038.csv",
                   help="Path to your merged CSV file containing columns R@..., T@..., etc.")
    p.add_argument("--c_val", type=float, required=True,
                   help="C value (e.g. 0.0, 0.1, 0.2). This picks partial_crys_data/partial_crys_C{c_val}.csv.")
    p.add_argument("--shape_idx", type=int, required=True,
                   help="Shape index to compare, matching the 'shape_idx' column in merged_csv.")
    p.add_argument("--s4cmd", default="S4",
                   help="Path/command to run S4 (default 'S4').")
    p.add_argument("--out_csv", default="out_temp.csv",
                   help="Temporary CSV name to store S4 results.")
    p.add_argument("--debug", action="store_true",
                   help="Print extra debug info.")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Load merged CSV
    if not os.path.isfile(args.merged_csv):
        print(f"ERROR: merged_csv file not found: {args.merged_csv}")
        sys.exit(1)
    df = pd.read_csv(args.merged_csv)

    # 2) Filter row by c_val and shape_idx
    mask = (df["c"]==args.c_val) & (df["shape_idx"]==args.shape_idx)
    sub = df[mask].copy()
    if len(sub)==0:
        print(f"No row found with c={args.c_val}, shape_idx={args.shape_idx} in {args.merged_csv}")
        sys.exit(1)
    row = sub.iloc[0]  # just take the first if multiple

    # shape_str is in e.g. row["vertices_str"]
    shape_str = row["vertices_str"].strip()

    # partial_csv => partial_crys_data/partial_crys_C{c_val}.csv
    c_str = f"{args.c_val:.1f}"
    partial_csv = os.path.join("partial_crys_data", f"partial_crys_C{c_str}.csv")
    if not os.path.isfile(partial_csv):
        print(f"ERROR: partial_csv not found: {partial_csv}")
        sys.exit(1)

    if args.debug:
        print(f"[DEBUG] Merged CSV row index={row.name}")
        print(f"[DEBUG] c={row['c']}, shape_idx={row['shape_idx']}")
        print(f"[DEBUG] shape_str={shape_str}")
        print(f"[DEBUG] partial_csv={partial_csv}")
    
    # 3) Run S4 with metasurface_check.lua
    arg_string = f'partial_csv={partial_csv} shape_str="{shape_str}" out_csv={args.out_csv}'
    cmd = [args.s4cmd, "-a", arg_string, "metasurface_check.lua"]
    if args.debug:
        print("[DEBUG] Running S4 command:", " ".join(cmd))
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        print("ERROR: S4 command failed with code:", ret.returncode)
        print("S4 stdout:\n", ret.stdout)
        print("S4 stderr:\n", ret.stderr)
        sys.exit(1)
    else:
        if args.debug:
            print("[DEBUG] S4 stdout:\n", ret.stdout)
            print("[DEBUG] S4 stderr:\n", ret.stderr)

    # 4) Read the output from metasurface_check.lua => columns (wavelength_um, R, T)
    if not os.path.isfile(args.out_csv):
        print(f"ERROR: Output CSV {args.out_csv} was not created.")
        sys.exit(1)

    df_s4 = pd.read_csv(args.out_csv)
    R_s4 = df_s4["R"].values
    T_s4 = df_s4["T"].values
    lam_s4 = df_s4["wavelength_um"].values  # might differ from WAVE_STR order

    # 5) Extract R_csv, T_csv from the merged row for the same 100 wavelengths
    #    We'll do index-based approach, i.e. R@1.040 => row["R@1.040"], etc.
    R_csv = []
    T_csv = []
    for w_ in WAVE_STR:
        Rcol = f"R@{w_}"
        Tcol = f"T@{w_}"
        if Rcol not in row or Tcol not in row:
            # Possibly the CSV has fewer columns or different names
            R_csv.append(np.nan)
            T_csv.append(np.nan)
            if args.debug:
                print(f"[DEBUG] No columns {Rcol} or {Tcol} in merged CSV row.")
        else:
            R_csv.append(row[Rcol])
            T_csv.append(row[Tcol])
    R_csv = np.array(R_csv, dtype=float)
    T_csv = np.array(T_csv, dtype=float)

    # 6) Plot comparison
    fig, (axR, axT) = plt.subplots(1,2, figsize=(10,4))

    axR.plot(R_csv, "b-", label="R_csv(merged)")
    axR.plot(R_s4, "r--", label="R_s4(sim)")
    axR.set_xlabel("index over WAVE_STR")
    axR.set_ylabel("Reflectance")
    axR.set_title(f"R compare c={args.c_val}, shape={args.shape_idx}")
    axR.legend()

    axT.plot(T_csv, "b-", label="T_csv(merged)")
    axT.plot(T_s4, "r--", label="T_s4(sim)")
    axT.set_xlabel("index over WAVE_STR")
    axT.set_ylabel("Transmittance")
    axT.set_title(f"T compare c={args.c_val}, shape={args.shape_idx}")
    axT.legend()

    out_png = f"compare_c{args.c_val}_shape{args.shape_idx}.png"
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Comparison plot saved to {out_png}")

    # 7) If debug, print a quick side‐by‐side
    if args.debug:
        print("\nIndex | lam_s4     R_s4     T_s4    | R_csv   T_csv")
        for i in range(min(len(lam_s4), len(R_csv))):
            print(f"{i:5d} | {lam_s4[i]:7.4f} {R_s4[i]:7.4f} {T_s4[i]:7.4f} | {R_csv[i]:7.4f} {T_csv[i]:7.4f}")

    print("Done comparison.")

if __name__=="__main__":
    main()

