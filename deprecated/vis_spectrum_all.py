#!/usr/bin/env python3
"""
vis_spectrum.py
Parses an S4 output file that contains lines like:
  "Now processing CSV: partial_crys_data/partial_crys_C0.0.csv"
  and
  "partial_crys_data/partial_crys_C0.0.csv | shape=1, row=2, λ=1.050 µm, freq=0.952, ... => R=..., T=..., R+T=..."

Then generates two sets of plots:
  A) For each CSV: compare all shapes
  B) For each shape: compare all CSVs

Usage:
  python vis_spectrum.py results.txt
The script will create:
  plots/<datetime>/compare_shapes/*.png
  plots/<datetime>/compare_crystallization/*.png
"""

import re
import os
import sys
import datetime
import matplotlib.pyplot as plt
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python vis_spectrum.py <results.txt>")
        sys.exit(1)

    filename = sys.argv[1]

    # Data structure:
    # data[ csv_name ][ shape_number ] = {
    #   "lambda": [ ... ],
    #   "R": [ ... ],
    #   "T": [ ... ],
    #   "RplusT": [ ... ]
    # }
    data = {}

    current_csv = None   # track which CSV we are processing

    # Example line to parse:
    # partial_crys_data/partial_crys_C0.0.csv | shape=1, row=2, λ=1.050 µm, freq=0.952, (n=3.58, k=0.06) => R=0.3495, T=0.6402, R+T=0.9897

    pattern_line = re.compile(
        r"^(.*?)\s*\|\s*shape\s*=\s*(\d+),\s*row=(\d+),\s*λ=([\d.]+)\s*µm.*R=([\-\d.]+),\s*T=([\-\d.]+),\s*R\+T=([\-\d.]+)"
    )

    # We'll also detect lines like:
    # "Now processing CSV: partial_crys_data/partial_crys_C0.0.csv"
    pattern_csv = re.compile(r"^Now processing CSV:\s*(.*\.csv)")

    # 1) Read the file and parse
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # 1A) Detect new CSV lines
            csv_match = pattern_csv.search(line)
            if csv_match:
                current_csv = csv_match.group(1).strip()
                if current_csv not in data:
                    data[current_csv] = {}
                continue

            # 1B) Match main data lines
            match = pattern_line.search(line)
            if match:
                csv_in_line  = match.group(1).strip()  # e.g. partial_crys_data/partial_crys_C0.0.csv
                shape_str    = match.group(2)          # "1"
                row_str      = match.group(3)          # "2"
                lam_str      = match.group(4)          # "1.050"
                r_str        = match.group(5)          # "0.3495"
                t_str        = match.group(6)          # "0.6402"
                rt_str       = match.group(7)          # "0.9897"

                shape_num = int(shape_str)
                lam_val   = float(lam_str)
                r_val     = float(r_str)
                t_val     = float(t_str)
                rt_val    = float(rt_str)

                # In theory, csv_in_line should match current_csv, but if not, we rely on csv_in_line
                if csv_in_line not in data:
                    data[csv_in_line] = {}

                if shape_num not in data[csv_in_line]:
                    data[csv_in_line][shape_num] = {
                        "lambda": [],
                        "R": [],
                        "T": [],
                        "RplusT": []
                    }

                data[csv_in_line][shape_num]["lambda"].append(lam_val)
                data[csv_in_line][shape_num]["R"].append(r_val)
                data[csv_in_line][shape_num]["T"].append(t_val)
                data[csv_in_line][shape_num]["RplusT"].append(rt_val)

    # 2) We want two sets of plots:
    #    A) For each CSV, compare shapes
    #    B) For each shape, compare CSVs

    csv_list = sorted(data.keys())

    # Also gather shape numbers found
    shape_set = set()
    for csvfile in data:
        for shp in data[csvfile]:
            shape_set.add(shp)
    shape_list = sorted(shape_set)

    # Create date/time subfolders
    dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shapes_dir = os.path.join("plots", dt_str, "compare_shapes")
    crys_dir   = os.path.join("plots", dt_str, "compare_crystallization")

    os.makedirs(shapes_dir, exist_ok=True)
    os.makedirs(crys_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # A) For each CSV, compare shapes -> 1 figure, 3 subplots (R, T, R+T)
    # --------------------------------------------------------------------------
    for csvfile in csv_list:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)
        ax_r, ax_t, ax_rt = axes

        # Extract a label for the figure from CSV name
        # e.g. partial_crys_data/partial_crys_C0.3.csv => "C0.3"
        label_for_title = csvfile
        match_c = re.search(r'partial_crys_C([\d.]+)\.csv', csvfile)
        if match_c:
            label_for_title = f"C{match_c.group(1)}"

        shapes_for_this_csv = sorted(data[csvfile].keys())

        for shp in shapes_for_this_csv:
            lam_arr  = np.array(data[csvfile][shp]["lambda"])
            r_arr    = np.array(data[csvfile][shp]["R"])
            t_arr    = np.array(data[csvfile][shp]["T"])
            rt_arr   = np.array(data[csvfile][shp]["RplusT"])

            ax_r.plot(lam_arr, r_arr, label=f"Shape {shp}")
            ax_t.plot(lam_arr, t_arr, label=f"Shape {shp}")
            ax_rt.plot(lam_arr, rt_arr, label=f"Shape {shp}")
            # ax_r.scatter(lam_arr, r_arr, label=f"Shape {shp}")
            # ax_t.scatter(lam_arr, t_arr, label=f"Shape {shp}")
            # ax_rt.scatter(lam_arr, rt_arr, label=f"Shape {shp}")

        ax_r.set_ylabel("R")
        ax_r.set_title(f"Reflection (R) - {label_for_title}")
        ax_t.set_ylabel("T")
        ax_t.set_title(f"Transmission (T) - {label_for_title}")
        ax_rt.set_xlabel("Wavelength (µm)")
        ax_rt.set_ylabel("R + T")
        ax_rt.set_title(f"R + T - {label_for_title}")

        for ax in axes:
            ax.grid(True)
            ax.legend()

        fig.tight_layout()

        # Save
        outname = os.path.join(shapes_dir, f"{label_for_title}.png")
        fig.savefig(outname, dpi=150)
        plt.close(fig)

    # --------------------------------------------------------------------------
    # B) For each shape, compare crystallizations (CSV) -> 1 figure, 3 subplots
    # --------------------------------------------------------------------------
    for shp in shape_list:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)
        ax_r, ax_t, ax_rt = axes

        for csvfile in csv_list:
            if shp in data[csvfile]:
                lam_arr  = np.array(data[csvfile][shp]["lambda"])
                r_arr    = np.array(data[csvfile][shp]["R"])
                t_arr    = np.array(data[csvfile][shp]["T"])
                rt_arr   = np.array(data[csvfile][shp]["RplusT"])

                label_str = csvfile
                match_c = re.search(r'partial_crys_C([\d.]+)\.csv', csvfile)
                if match_c:
                    label_str = f"C{match_c.group(1)}"

                ax_r.plot(lam_arr, r_arr, label=label_str)
                ax_t.plot(lam_arr, t_arr, label=label_str)
                ax_rt.plot(lam_arr, rt_arr, label=label_str)
                # ax_r.scatter(lam_arr, r_arr, label=label_str)
                # ax_t.scatter(lam_arr, t_arr, label=label_str)
                # ax_rt.scatter(lam_arr, rt_arr, label=label_str)

        ax_r.set_ylabel("R")
        ax_r.set_title(f"Reflection (R) - Shape {shp}")
        ax_t.set_ylabel("T")
        ax_t.set_title(f"Transmission (T) - Shape {shp}")
        ax_rt.set_xlabel("Wavelength (µm)")
        ax_rt.set_ylabel("R + T")
        ax_rt.set_title(f"R + T - Shape {shp}")

        for ax in axes:
            ax.grid(True)
            ax.legend()

        fig.tight_layout()

        outname = os.path.join(crys_dir, f"shape_{shp}.png")
        fig.savefig(outname, dpi=150)
        plt.close(fig)

    print("Done!")
    print(f"Plots saved under '{os.path.join('plots', dt_str)}'")

if __name__ == "__main__":
    main()
