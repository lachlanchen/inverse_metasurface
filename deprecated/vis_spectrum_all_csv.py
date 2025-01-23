#!/usr/bin/env python3
"""
vis_spectrum.py
Parses an S4 output file that contains lines like:
  "Now processing CSV: partial_crys_data/partial_crys_C0.0.csv"
  and
  "partial_crys_data/partial_crys_C0.0.csv | shape=1, row=2, λ=1.050 µm, freq=0.952, (n=3.58, k=0.06) => R=..., T=..., R+T=..."

We generate two sets of plots:
  A) For each CSV: compare all shapes
  B) For each shape: compare all CSVs

For each figure, we also save a matching .csv file with columns:
  csv_filename,shape,row,lambda,freq,n,k,R,T,RplusT

Usage:
  python vis_spectrum.py results.txt

The script will create:
  plots/<datetime>/compare_shapes/*.png and *.csv
  plots/<datetime>/compare_crystallization/*.png and *.csv
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

    # We store data in a nested dict:
    # data[csv_filepath][shape_number] = {
    #     "lambda":  [...],
    #     "freq":    [...],
    #     "n":       [...],
    #     "k":       [...],
    #     "R":       [...],
    #     "T":       [...],
    #     "RplusT":  [...],
    #     "row":     [...],  # row index from the original line
    # }
    data = {}

    current_csv = None   # track which CSV we are processing

    # We define a regex that captures:
    #  1) csv_in_line: the path e.g. "partial_crys_data/partial_crys_C0.0.csv"
    #  2) shape_str: e.g. "1"
    #  3) row_str: e.g. "2"
    #  4) lam_str: e.g. "1.050"
    #  5) freq_str: e.g. "0.952"
    #  6) n_str: e.g. "3.58"
    #  7) k_str: e.g. "0.06"
    #  8) r_str: e.g. "0.3495"
    #  9) t_str: e.g. "0.6402"
    # 10) rt_str: e.g. "0.9897"
    #
    # We'll assume the line looks like:
    # "partial_crys_data/partial_crys_C0.0.csv | shape=1, row=2, λ=1.050 µm, freq=0.952, (n=3.58, k=0.06) => R=0.3495, T=0.6402, R+T=0.9897"
    #
    # Adjust if your lines differ.
    pattern_line = re.compile(
        r"^(.*?)\s*\|\s*shape\s*=\s*(\d+),\s*row=(\d+),\s*λ=([\d.]+)\s*µm,\s*freq=([\d.]+).*?\(n=([\d.]+),\s*k=([\d.]+)\).*?R=([\-\d.]+),\s*T=([\-\d.]+),\s*R\+T=([\-\d.]+)"
    )

    # We'll also detect lines:
    # "Now processing CSV: partial_crys_data/partial_crys_C0.0.csv"
    pattern_csv = re.compile(r"^Now processing CSV:\s*(.*\.csv)")

    # --------------------------------------------------------------------------
    # 1) Read the file and parse the data
    # --------------------------------------------------------------------------
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # 1A) Detect "Now processing CSV" lines
            csv_match = pattern_csv.search(line)
            if csv_match:
                current_csv = csv_match.group(1).strip()
                # Initialize if not in data
                if current_csv not in data:
                    data[current_csv] = {}
                continue

            # 1B) Match the main data lines
            match = pattern_line.search(line)
            if match:
                (
                    csv_in_line,
                    shape_str,
                    row_str,
                    lam_str,
                    freq_str,
                    n_str,
                    k_str,
                    r_str,
                    t_str,
                    rt_str,
                ) = match.groups()

                shape_num = int(shape_str)
                row_num   = int(row_str)
                lam_val   = float(lam_str)
                freq_val  = float(freq_str)
                n_val     = float(n_str)
                k_val     = float(k_str)
                r_val     = float(r_str)
                t_val     = float(t_str)
                rt_val    = float(rt_str)

                # If the CSV in the line isn't in data, init
                if csv_in_line not in data:
                    data[csv_in_line] = {}

                if shape_num not in data[csv_in_line]:
                    data[csv_in_line][shape_num] = {
                        "lambda": [],
                        "freq":   [],
                        "n":      [],
                        "k":      [],
                        "R":      [],
                        "T":      [],
                        "RplusT": [],
                        "row":    [],
                    }

                data[csv_in_line][shape_num]["lambda"].append(lam_val)
                data[csv_in_line][shape_num]["freq"].append(freq_val)
                data[csv_in_line][shape_num]["n"].append(n_val)
                data[csv_in_line][shape_num]["k"].append(k_val)
                data[csv_in_line][shape_num]["R"].append(r_val)
                data[csv_in_line][shape_num]["T"].append(t_val)
                data[csv_in_line][shape_num]["RplusT"].append(rt_val)
                data[csv_in_line][shape_num]["row"].append(row_num)

    # --------------------------------------------------------------------------
    # 2) We'll make two sets of plots:
    #    A) For each CSV, compare shapes
    #    B) For each shape, compare CSVs
    # --------------------------------------------------------------------------
    csv_list = sorted(data.keys())

    # Gather all shape numbers found across all CSVs
    shape_set = set()
    for csvfile in data:
        for shp in data[csvfile]:
            shape_set.add(shp)
    shape_list = sorted(shape_set)

    # We'll store the plots under: plots/<datetime>/...
    dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shapes_dir = os.path.join("plots", dt_str, "compare_shapes")
    crys_dir   = os.path.join("plots", dt_str, "compare_crystallization")

    os.makedirs(shapes_dir, exist_ok=True)
    os.makedirs(crys_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # A) For each CSV, compare shapes -> 1 figure, 3 subplots (R, T, R+T)
    #    Also save a CSV with columns:
    #    csv_filename,shape,row,lambda,freq,n,k,R,T,RplusT
    # --------------------------------------------------------------------------
    for csvfile in csv_list:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)
        ax_r, ax_t, ax_rt = axes

        # A short label for the figure from CSV name
        label_for_title = csvfile
        match_c = re.search(r'partial_crys_C([\d.]+)\.csv', csvfile)
        if match_c:
            label_for_title = f"C{match_c.group(1)}"

        shapes_for_this_csv = sorted(data[csvfile].keys())

        # We'll store data lines for the CSV file to be saved
        csv_lines = []
        csv_lines.append("csv_filename,shape,row,lambda,freq,n,k,R,T,RplusT")

        for shp in shapes_for_this_csv:
            lam_arr  = np.array(data[csvfile][shp]["lambda"])
            freq_arr = np.array(data[csvfile][shp]["freq"])
            n_arr    = np.array(data[csvfile][shp]["n"])
            k_arr    = np.array(data[csvfile][shp]["k"])
            r_arr    = np.array(data[csvfile][shp]["R"])
            t_arr    = np.array(data[csvfile][shp]["T"])
            rt_arr   = np.array(data[csvfile][shp]["RplusT"])
            row_arr  = np.array(data[csvfile][shp]["row"])

            # Plot lines
            ax_r.plot(lam_arr, r_arr, label=f"Shape {shp}")
            ax_t.plot(lam_arr, t_arr, label=f"Shape {shp}")
            ax_rt.plot(lam_arr, rt_arr, label=f"Shape {shp}")

            # Accumulate CSV lines (all data columns)
            for i in range(len(lam_arr)):
                csv_line = (
                    f"{csvfile},{shp},{row_arr[i]},"
                    f"{lam_arr[i]:.4f},{freq_arr[i]:.4f},"
                    f"{n_arr[i]:.4f},{k_arr[i]:.4f},"
                    f"{r_arr[i]:.4f},{t_arr[i]:.4f},{rt_arr[i]:.4f}"
                )
                csv_lines.append(csv_line)

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

        # Save figure
        outname_plot = os.path.join(shapes_dir, f"{label_for_title}.png")
        fig.savefig(outname_plot, dpi=150)
        plt.close(fig)

        # Save CSV
        outname_csv = os.path.join(shapes_dir, f"{label_for_title}.csv")
        with open(outname_csv, "w") as fout:
            fout.write("\n".join(csv_lines) + "\n")

    # --------------------------------------------------------------------------
    # B) For each shape, compare crystallizations -> 1 figure, 3 subplots
    #    Also save CSV with columns:
    #    csv_filename,shape,row,lambda,freq,n,k,R,T,RplusT
    # --------------------------------------------------------------------------
    for shp in shape_list:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)
        ax_r, ax_t, ax_rt = axes

        csv_lines = []
        csv_lines.append("csv_filename,shape,row,lambda,freq,n,k,R,T,RplusT")

        for csvfile in csv_list:
            if shp in data[csvfile]:
                lam_arr  = np.array(data[csvfile][shp]["lambda"])
                freq_arr = np.array(data[csvfile][shp]["freq"])
                n_arr    = np.array(data[csvfile][shp]["n"])
                k_arr    = np.array(data[csvfile][shp]["k"])
                r_arr    = np.array(data[csvfile][shp]["R"])
                t_arr    = np.array(data[csvfile][shp]["T"])
                rt_arr   = np.array(data[csvfile][shp]["RplusT"])
                row_arr  = np.array(data[csvfile][shp]["row"])

                # e.g. partial_crys_data/partial_crys_C0.3.csv => "C0.3"
                label_str = csvfile
                match_c = re.search(r'partial_crys_C([\d.]+)\.csv', csvfile)
                if match_c:
                    label_str = f"C{match_c.group(1)}"

                ax_r.plot(lam_arr, r_arr, label=label_str)
                ax_t.plot(lam_arr, t_arr, label=label_str)
                ax_rt.plot(lam_arr, rt_arr, label=label_str)

                # Accumulate lines
                for i in range(len(lam_arr)):
                    csv_line = (
                        f"{csvfile},{shp},{row_arr[i]},"
                        f"{lam_arr[i]:.4f},{freq_arr[i]:.4f},"
                        f"{n_arr[i]:.4f},{k_arr[i]:.4f},"
                        f"{r_arr[i]:.4f},{t_arr[i]:.4f},{rt_arr[i]:.4f}"
                    )
                    csv_lines.append(csv_line)

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

        outname_plot = os.path.join(crys_dir, f"shape_{shp}.png")
        fig.savefig(outname_plot, dpi=150)
        plt.close(fig)

        # Save CSV
        outname_csv = os.path.join(crys_dir, f"shape_{shp}.csv")
        with open(outname_csv, "w") as fout:
            fout.write("\n".join(csv_lines) + "\n")

    print("Done!")
    print(f"Plots & CSVs saved under '{os.path.join('plots', dt_str)}'")

if __name__ == "__main__":
    main()
