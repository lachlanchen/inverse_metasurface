#!/usr/bin/env python3
"""
metasurface_compare.py

Example usage:
  python metasurface_compare.py --nq 1 --ns 10000 --shape_idx 1 --c_val 0.0

Flow:
  1) Load the row from merged_s4_shapes_20250119_153038.csv for c=0.0
  2) Parse polygon => shape_str="x0,y0;x1,y1;..."
  3) Decide partial_csv => e.g. partial_crys_data/partial_crys_C0.0.csv
  4) Decide out_csv => e.g. out_c0.0.csv
  5) Run: S4 -a "partial_csv=... out_csv=... shape_str=..." metasurface_check.lua
  6) Read out_csv. Compare with the R@1.040, R@1.054, ... columns if desired.
"""

import os
import sys
import subprocess
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MERGED_CSV = "merged_s4_shapes_20250119_153038.csv"
PARTIAL_DIR = "partial_crys_data"
LUA_FILE    = "metasurface_check.lua"  # the Lua script from above

WAVE_LIST = [
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
    p.add_argument("--nq", type=int, default=1)
    p.add_argument("--ns", type=int, default=10000)
    p.add_argument("--shape_idx", type=int, default=1)
    p.add_argument("--c_val", type=float, default=0.0, help="Which c in [0..1]? e.g. 0.0 or 0.3 etc.")
    p.add_argument("--s4_cmd", default="S4", help="The S4 command or full path. E.g. '../build/S4'")
    return p.parse_args()

def main():
    args = parse_args()

    df = pd.read_csv(MERGED_CSV)
    mask = (df["NQ"]==args.nq) & (df["nS"]==args.ns) & (df["shape_idx"]==args.shape_idx) & (df["c"]==args.c_val)
    sub = df[mask].copy()
    if len(sub)==0:
        print(f"No row found for NQ={args.nq}, nS={args.ns}, shape_idx={args.shape_idx}, c={args.c_val}")
        sys.exit(1)
    row = sub.iloc[0]

    # parse polygon => shape_str
    vertices_str = row["vertices_str"]  # e.g. "0.0,0.1;0.2,0.3;..."
    # let's keep it the same
    # just pass it as-is to the Lua script
    shape_str = vertices_str.strip()

    # partial CSV
    c_str = f"{args.c_val:.1f}"
    partial_csv = os.path.join(PARTIAL_DIR, f"partial_crys_C{c_str}.csv")

    # output results
    out_csv = f"out_c{c_str}.csv"

    # We'll run: S4 -a "partial_csv=... out_csv=... shape_str=..." metasurface_check.lua
    s4_arg = f"partial_csv={partial_csv} out_csv={out_csv} shape_str={shape_str}"

    cmd_list = [
        args.s4_cmd, 
        "-a", s4_arg,
        LUA_FILE
    ]
    print("Running:", " ".join(cmd_list))
    ret = subprocess.run(cmd_list, capture_output=True, text=True)
    if ret.returncode != 0:
        print("S4 returned error code:", ret.returncode)
        print("stdout:\n", ret.stdout)
        print("stderr:\n", ret.stderr)
        sys.exit(1)

    print("S4 done. Reading", out_csv)
    df_s4 = pd.read_csv(out_csv)
    # columns: wavelength_um,R,T

    # We'll gather R_csv, T_csv from the row for plotting
    R_csv = []
    T_csv = []
    for w_ in WAVE_LIST:
        Rcol = f"R@{w_}"
        Tcol = f"T@{w_}"
        R_csv.append(row[Rcol])
        T_csv.append(row[Tcol])
    R_csv = np.array(R_csv)
    T_csv = np.array(T_csv)

    # We'll align them by index. The df_s4 is sorted in ascending lam. 
    # If you want a direct point-by-point comparison, the i-th row of df_s4 might 
    # correspond to wave=the i-th in WAVE_LIST if they match. 
    # But they won't exactly match. 
    # We'll just plot them side-by-side by index. 
    # For a "narrow" or approximate match, you'd do interpolation, etc.

    # For a quick naive approach, let's just plot index vs. value
    lam_s4 = df_s4["wavelength_um"].values
    R_s4 = df_s4["R"].values
    T_s4 = df_s4["T"].values

    import matplotlib.pyplot as plt
    fig, (axR, axT) = plt.subplots(1,2, figsize=(10,4))

    axR.plot(R_csv, color="blue", label="R_csv (index-based)")
    axR.plot(R_s4, color="red", label="R_s4 (index-based)", linestyle="--")
    axR.set_title(f"c={args.c_val}, Reflectance")
    axR.set_ylim([0,1.1])
    axR.legend()

    axT.plot(T_csv, color="blue", label="T_csv (index-based)")
    axT.plot(T_s4, color="red", label="T_s4 (index-based)", linestyle="--")
    axT.set_title(f"c={args.c_val}, Transmittance")
    axT.set_ylim([0,1.1])
    axT.legend()

    plt.tight_layout()
    out_png = f"compare_c{args.c_val}.png"
    plt.savefig(out_png)
    print("Saved plot to", out_png)
    print("Done.")


if __name__=="__main__":
    main()

