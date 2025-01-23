#!/usr/bin/env python3
"""
vis_spectrum_pandas.py

Parses an S4 output file that contains lines like:
  "Now processing CSV: partial_crys_data/partial_crys_C0.0.csv"
  and
  "partial_crys_data/partial_crys_C0.0.csv | shape=1, row=2, λ=1.050 µm, freq=0.952, (n=3.58, k=0.06) => R=..., T=..., R+T=..."

Then generates two sets of plots:
  1) For each CSV file (crystallization): compare all shapes
  2) For each shape: compare all CSV files

Additionally, we save a CSV for each figure with the data used in that figure.

Usage:
  python vis_spectrum_pandas.py results.txt

The script creates:
  plots/<datetime>/compare_shapes/*.png + *.csv
  plots/<datetime>/compare_crystallization/*.png + *.csv
"""

import os
import re
import sys
import datetime
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python vis_spectrum_pandas.py <results.txt>")
        sys.exit(1)

    filename = sys.argv[1]

    # Regex to detect "Now processing CSV: partial_crys_data/partial_crys_C0.0.csv"
    pattern_csv = re.compile(r"^Now processing CSV:\s*(.*\.csv)")

    # Regex to parse lines like:
    # partial_crys_data/partial_crys_C0.0.csv | shape=1, row=2, λ=1.050 µm, freq=0.952, (n=3.58, k=0.06) => R=0.3495, T=0.6402, R+T=0.9897
    # capturing:
    #  (1) csv_file path
    #  (2) shape
    #  (3) row
    #  (4) lam (wavelength)
    #  (5) freq
    #  (6) n
    #  (7) k
    #  (8) R
    #  (9) T
    # (10) R+T
    pattern_line = re.compile(
        r"^(.*?)\s*\|\s*shape\s*=\s*(\d+),\s*row=(\d+),\s*λ=([\d.]+)\s*µm,\s*freq=([\d.]+).*?\(n=([\d.]+),\s*k=([\d.]+)\).*?R=([\-\d.]+),\s*T=([\-\d.]+),\s*R\+T=([\-\d.]+)"
    )

    # We will store the parsed rows in a list of dicts, then convert to a DataFrame.
    rows = []
    current_csv = None  # track current CSV

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            # 1) Check for "Now processing CSV"
            m_csv = pattern_csv.search(line)
            if m_csv:
                current_csv = m_csv.group(1).strip()
                continue

            # 2) Check for data lines
            m_line = pattern_line.search(line)
            if m_line:
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
                ) = m_line.groups()

                # Convert to numeric
                shape_num = int(shape_str)
                row_num   = int(row_str)
                lam_val   = float(lam_str)
                freq_val  = float(freq_str)
                n_val     = float(n_str)
                k_val     = float(k_str)
                r_val     = float(r_str)
                t_val     = float(t_str)
                rt_val    = float(rt_str)

                # We'll store each line as one row in the DataFrame
                rows.append({
                    "csv_file": csv_in_line,  # e.g. partial_crys_data/partial_crys_C0.0.csv
                    "shape": shape_num,
                    "row": row_num,
                    "lambda": lam_val,
                    "freq": freq_val,
                    "n": n_val,
                    "k": k_val,
                    "R": r_val,
                    "T": t_val,
                    "RplusT": rt_val
                })

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    if df.empty:
        print("No data parsed. Check your input file / regex.")
        sys.exit(0)

    # For convenience, we can derive a short "crys_label" from partial_crys_C0.0
    # e.g. partial_crys_data/partial_crys_C0.0.csv => "C0.0"
    # We'll store in a new column "crys_label"
    def extract_crys_label(x):
        # e.g. partial_crys_data/partial_crys_C0.0.csv => match partial_crys_C([\d.]+)
        m = re.search(r'partial_crys_C([\d.]+)\.csv', x)
        if m:
            return f"C{m.group(1)}"
        else:
            return x  # fallback to original
    df["crys_label"] = df["csv_file"].apply(extract_crys_label)

    # Let's create an output folder with a timestamp
    dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("plots", dt_str)
    shapes_dir = os.path.join(base_dir, "compare_shapes")
    crys_dir   = os.path.join(base_dir, "compare_crystallization")
    os.makedirs(shapes_dir, exist_ok=True)
    os.makedirs(crys_dir, exist_ok=True)

    # 1) For each CSV file (unique "csv_file"): compare shapes
    # We'll group by "csv_file"
    for csv_file, subdf_csv in df.groupby("csv_file"):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8,10), sharex=True)
        ax_r, ax_t, ax_rt = axes

        # We'll produce a label for the figure from "crys_label"
        label_for_title = subdf_csv["crys_label"].iloc[0]  # e.g. "C0.0" or fallback

        # We'll store this subdf's data in a CSV: shape, row, lambda, freq, n, k, R, T, RplusT
        # We want to see data for all shapes in this CSV
        out_csv_rows = []
        out_csv_rows.append("csv_file,crys_label,shape,row,lambda,freq,n,k,R,T,RplusT")

        # We'll group subdf_csv by shape
        for shape_num, subdf_shape in subdf_csv.groupby("shape"):
            # Sort by lambda just for consistent plotting
            subdf_shape = subdf_shape.sort_values("lambda")

            lam_arr  = subdf_shape["lambda"].values
            r_arr    = subdf_shape["R"].values
            t_arr    = subdf_shape["T"].values
            rt_arr   = subdf_shape["RplusT"].values

            ax_r.plot(lam_arr, r_arr, label=f"Shape {shape_num}")
            ax_t.plot(lam_arr, t_arr, label=f"Shape {shape_num}")
            ax_rt.plot(lam_arr, rt_arr, label=f"Shape {shape_num}")

            # Append lines to CSV
            for _, rowdat in subdf_shape.iterrows():
                out_csv_rows.append(
                    f"{rowdat['csv_file']},{rowdat['crys_label']},"
                    f"{rowdat['shape']},{rowdat['row']},"
                    f"{rowdat['lambda']},{rowdat['freq']},"
                    f"{rowdat['n']},{rowdat['k']},"
                    f"{rowdat['R']},{rowdat['T']},{rowdat['RplusT']}"
                )

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
        out_png = os.path.join(shapes_dir, f"{label_for_title}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

        # Save CSV
        out_csv_path = os.path.join(shapes_dir, f"{label_for_title}.csv")
        with open(out_csv_path, "w") as fout:
            fout.write("\n".join(out_csv_rows) + "\n")

    # 2) For each shape, compare all CSV files
    for shape_num, subdf_shape in df.groupby("shape"):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8,10), sharex=True)
        ax_r, ax_t, ax_rt = axes

        # We'll store data lines
        out_csv_rows = []
        out_csv_rows.append("csv_file,crys_label,shape,row,lambda,freq,n,k,R,T,RplusT")

        # Group by csv_file inside shape
        for csv_file, subdf_csvfile in subdf_shape.groupby("csv_file"):
            subdf_csvfile = subdf_csvfile.sort_values("lambda")

            lam_arr  = subdf_csvfile["lambda"].values
            r_arr    = subdf_csvfile["R"].values
            t_arr    = subdf_csvfile["T"].values
            rt_arr   = subdf_csvfile["RplusT"].values

            label_for_title = subdf_csvfile["crys_label"].iloc[0]

            ax_r.plot(lam_arr, r_arr, label=label_for_title)
            ax_t.plot(lam_arr, t_arr, label=label_for_title)
            ax_rt.plot(lam_arr, rt_arr, label=label_for_title)

            # Accumulate CSV
            for _, rowdat in subdf_csvfile.iterrows():
                out_csv_rows.append(
                    f"{rowdat['csv_file']},{rowdat['crys_label']},"
                    f"{rowdat['shape']},{rowdat['row']},"
                    f"{rowdat['lambda']},{rowdat['freq']},"
                    f"{rowdat['n']},{rowdat['k']},"
                    f"{rowdat['R']},{rowdat['T']},{rowdat['RplusT']}"
                )

        ax_r.set_ylabel("R")
        ax_r.set_title(f"Reflection (R) - Shape {shape_num}")
        ax_t.set_ylabel("T")
        ax_t.set_title(f"Transmission (T) - Shape {shape_num}")
        ax_rt.set_xlabel("Wavelength (µm)")
        ax_rt.set_ylabel("R + T")
        ax_rt.set_title(f"R + T - Shape {shape_num}")

        for ax in axes:
            ax.grid(True)
            ax.legend()

        fig.tight_layout()

        out_png = os.path.join(crys_dir, f"shape_{shape_num}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

        out_csv_path = os.path.join(crys_dir, f"shape_{shape_num}.csv")
        with open(out_csv_path, "w") as fout:
            fout.write("\n".join(out_csv_rows) + "\n")

    print("Done!")
    print(f"Plots & CSVs saved under '{base_dir}'")


if __name__ == "__main__":
    main()
