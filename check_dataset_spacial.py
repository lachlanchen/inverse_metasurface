#!/usr/bin/env python3
import os
import sys
import csv
import glob
import random
import argparse
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt

def read_merged_csv(merged_csv_path):
    rows = []
    with open(merged_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
    selected = []
    for r in rows:
        if "NQ" in r and "shape_idx" in r:
            if abs(r["NQ"] - nq_val) < 1e-9 and r["shape_idx"] == shape_idx_val:
                selected.append(r)
    return selected

def parse_merged_row_for_RT(row):
    wave_list = []
    R_list = []
    T_list = []
    waves_for_R = {}
    waves_for_T = {}

    for k,v in row.items():
        if k.startswith("R@"):
            lam_str = k.split("@")[1]
            try:
                lam_val = float(lam_str)
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

    all_lams = sorted(set(waves_for_R.keys()).union(waves_for_T.keys()))
    for lam in all_lams:
        Rv = waves_for_R.get(lam, None)
        Tv = waves_for_T.get(lam, None)
        wave_list.append(lam)
        R_list.append(Rv)
        T_list.append(Tv)
    return wave_list, R_list, T_list

def run_s4_for_c(polygon_str, c_val):
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

    saved_path = None
    for line in proc.stdout.splitlines():
        if "Saved to " in line:
            saved_path = line.split("Saved to",1)[1].strip()
            break
    return saved_path

def read_results_csv(csv_path):
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

    data = sorted(zip(wv,Rv,Tv), key=lambda x:x[0])
    wv = [d[0] for d in data]
    Rv = [d[1] for d in data]
    Tv = [d[2] for d in data]
    return wv, Rv, Tv

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
                        help="If set, we run the S4 command for each c. Otherwise, might skip.")
    args = parser.parse_args()

    merged_csv_path = args.merged_csv
    c_arg = args.c

    # Always run_s4 = True if you prefer
    run_s4 = True

    all_rows = read_merged_csv(merged_csv_path)
    if not all_rows:
        print("[ERROR] No data in merged CSV. Exiting.")
        return

    if args.nq is None or args.shape_idx is None:
        row0 = random.choice(all_rows)
        chosen_nq = row0["NQ"]
        chosen_shape = row0["shape_idx"]
        print(f"[INFO] Using NQ={chosen_nq}, shape_idx={chosen_shape}.")
    else:
        chosen_nq = float(args.nq)
        chosen_shape = args.shape_idx

    matching_rows = find_rows_for_nq_shape(all_rows, chosen_nq, chosen_shape)
    if not matching_rows:
        print(f"[ERROR] No rows found for NQ={chosen_nq}, shape_idx={chosen_shape}.")
        return

    polygon_str = matching_rows[0].get("vertices_str", "")
    if not polygon_str:
        print("[ERROR] No vertices_str found in that row. Cannot run S4.")
        return

    if c_arg.lower() == "all":
        cvals = sorted({r["c"] for r in matching_rows})
    else:
        try:
            cval = float(c_arg)
            row_cs = sorted({r["c"] for r in matching_rows})
            if cval not in row_cs:
                print(f"[WARN] c={cval} not found in the CSV for that shape. We'll still attempt.")
                row_cs.append(cval)
            cvals = [cval]
        except:
            print("[ERROR] Could not parse --c as float or 'all'. Exiting.")
            return

    nrows = len(cvals)
    ncols = 3
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, 
        figsize=(10, 6*nrows),
        # sharex=False, sharey=False,
        sharex=True, sharey=True,
        constrained_layout=True  # This helps separate subplots automatically
    )
    # If we still want even more spacing, we can do:
    # fig.subplots_adjust(hspace=0.35, wspace=0.3)

    if nrows == 1:
        # axes is 1D, convert to [axes] so we can index as axes[0][0], etc.
        axes = [axes]

    for idx, c_val in enumerate(cvals):
        row_for_c = None
        for rr in matching_rows:
            if abs(rr["c"] - c_val) < 1e-9:
                row_for_c = rr
                break
        if not row_for_c:
            print(f"[WARN] No row for c={c_val} in the CSV. Skipping.")
            continue

        wave_m, R_m, T_m = parse_merged_row_for_RT(row_for_c)
        R_m_abs = [abs(x) if x is not None else None for x in R_m]
        RT_m = [(abs(r) if r else 0)+(t if t else 0) for r,t in zip(R_m,T_m)]

        results_csv_path = None
        if run_s4:
            results_csv_path = run_s4_for_c(polygon_str, c_val)
        else:
            pat = f"results/fixed_shape_c{c_val:.1f}_*.csv"
            found = glob.glob(pat)
            if found:
                results_csv_path = max(found, key=os.path.getmtime)

        wave_r, R_r, T_r = [], [], []
        if results_csv_path:
            wave_r, R_r, T_r = read_results_csv(results_csv_path)
        R_r_abs = [abs(x) for x in R_r]
        RT_r = [abs(r)+t for r,t in zip(R_r, T_r)]

        ax_ref = axes[idx][0]
        ax_trn = axes[idx][1]
        ax_sum = axes[idx][2]

        # Reflection
        if wave_m and R_m_abs:
            ax_ref.plot(wave_m, R_m_abs, 'r--', label="Merged |R|")
        if wave_r and R_r_abs:
            ax_ref.plot(wave_r, R_r_abs, 'r-', label="Newly-run |R|")
        ax_ref.set_title(f"c={c_val:.1f} Reflection")
        ax_ref.set_xlabel("Wavelength (um)")
        ax_ref.set_ylabel("|R|")

        # Transmission
        if wave_m and T_m:
            ax_trn.plot(wave_m, T_m, 'b--', label="Merged T")
        if wave_r and T_r:
            ax_trn.plot(wave_r, T_r, 'b-', label="Newly-run T")
        ax_trn.set_title(f"c={c_val:.1f} Transmission")
        ax_trn.set_xlabel("Wavelength (um)")
        ax_trn.set_ylabel("T")

        # |R|+T
        if wave_m and RT_m:
            ax_sum.plot(wave_m, RT_m, 'g--', label="Merged |R|+T")
        if wave_r and RT_r:
            ax_sum.plot(wave_r, RT_r, 'g-', label="Newly-run |R|+T")
        ax_sum.set_title(f"c={c_val:.1f} (|R|+T)")
        ax_sum.set_xlabel("Wavelength (um)")
        ax_sum.set_ylabel("|R|+T")

        if idx == 0:
            ax_ref.legend()
            ax_trn.legend()
            ax_sum.legend()

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"check_dataset_plots_{dt_str}"
    os.makedirs(outdir, exist_ok=True)
    outpng = os.path.join(outdir, f"comparison_NQ{chosen_nq}_shape{chosen_shape}.png")
    plt.savefig(outpng, dpi=250)
    print(f"[INFO] Figure saved to {outpng}")

    # Now show the figure with improved spacing from constrained_layout
    plt.show()

if __name__ == "__main__":
    main()

