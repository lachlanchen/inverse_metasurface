#!/usr/bin/env python3
"""
vis_spectrum_pandas_excel.py

1) Parse an S4 results file (e.g. 'results.txt') into a pandas DataFrame.
2) Save all data to:
   - all_data.xlsx
   - all_data.csv
   in the same base folder as the plots.
3) Generate two sets of plots:
   A) For each CSV file (crystallization): compare all shapes
   B) For each shape: compare all CSV files
   saving them under:
   plots/<datetime>/compare_shapes/
   plots/<datetime>/compare_crystallization/

The DataFrame columns include:
  csv_file, shape, row, lambda, freq, n, k, R, T, RplusT, crys_label

Usage:
  python vis_spectrum_pandas_excel.py results.txt
"""

import os
import re
import sys
import datetime
import pandas as pd
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print("Usage: python vis_spectrum_pandas_excel.py <results.txt>")
        sys.exit(1)

    filename = sys.argv[1]

    # Regex to detect lines like:
    #  "Now processing CSV: partial_crys_data/partial_crys_C0.0.csv"
    pattern_csv = re.compile(r"^Now processing CSV:\s*(.*\.csv)")

    # Regex to parse lines like:
    # partial_crys_data/partial_crys_C0.0.csv | shape=1, row=2, λ=1.050 µm, freq=0.952, (n=3.58, k=0.06) => R=..., T=..., R+T=...
    pattern_line = re.compile(
        r"^(.*?)\s*\|\s*shape\s*=\s*(\d+),\s*row=(\d+),\s*λ=([\d.]+)\s*µm,\s*freq=([\d.]+).*?\(n=([\d.]+),\s*k=([\d.]+)\).*?R=([\-\d.]+),\s*T=([\-\d.]+),\s*R\+T=([\-\d.]+)"
    )

    rows = []
    current_csv = None  # track which CSV is being processed

    # 1) Parse the file
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Detect "Now processing CSV: ...."
            m_csv = pattern_csv.search(line)
            if m_csv:
                current_csv = m_csv.group(1).strip()
                continue

            # Detect main data lines
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

                shape_num = int(shape_str)
                row_num   = int(row_str)
                lam_val   = float(lam_str)
                freq_val  = float(freq_str)
                n_val     = float(n_str)
                k_val     = float(k_str)
                r_val     = float(r_str)
                t_val     = float(t_str)
                rt_val    = float(rt_str)

                # We'll store each line as one row in the final DataFrame
                rows.append({
                    "csv_file": csv_in_line,  # e.g. partial_crys_data/partial_crys_C0.0.csv
                    "shape":    shape_num,
                    "row":      row_num,
                    "lambda":   lam_val,
                    "freq":     freq_val,
                    "n":        n_val,
                    "k":        k_val,
                    "R":        r_val,
                    "T":        t_val,
                    "RplusT":   rt_val
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data parsed. Check your input file / regex pattern.")
        sys.exit(0)

    # 2) We'll derive a "crys_label" from partial_crys_C0.0.csv => "C0.0"
    def extract_crys_label(x):
        m = re.search(r'partial_crys_C([\d.]+)\.csv', x)
        if m:
            return f"C{m.group(1)}"
        else:
            return x
    df["crys_label"] = df["csv_file"].apply(extract_crys_label)

    # 3) We'll create the main output folder (timestamped)
    dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("plots", dt_str)
    shapes_dir = os.path.join(base_dir, "compare_shapes")
    crys_dir   = os.path.join(base_dir, "compare_crystallization")
    os.makedirs(shapes_dir, exist_ok=True)
    os.makedirs(crys_dir, exist_ok=True)

    # 4) Save the entire DataFrame to Excel and CSV in the base_dir
    all_data_excel = os.path.join(base_dir, "all_data.xlsx")
    all_data_csv   = os.path.join(base_dir, "all_data.csv")
    df.to_excel(all_data_excel, index=False)
    df.to_csv(all_data_csv, index=False)
    print(f"Saved full dataset to:\n  {all_data_excel}\n  {all_data_csv}")

    # 5) Now let's do the usual plots:
    #    A) For each CSV (crystallization), compare shapes
    #    B) For each shape, compare CSVs

    # A) For each CSV => compare shapes
    for csv_file, subdf_csv in df.groupby("csv_file"):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8,10), sharex=True)
        ax_r, ax_t, ax_rt = axes

        label_for_title = subdf_csv["crys_label"].iloc[0]  # e.g. "C0.0"

        # We'll build lines for a figure-specific CSV
        out_csv_lines = ["csv_file,crys_label,shape,row,lambda,freq,n,k,R,T,RplusT"]

        # Group by shape
        for shape_num, subdf_shape in subdf_csv.groupby("shape"):
            subdf_shape = subdf_shape.sort_values("lambda")

            lam_arr  = subdf_shape["lambda"].values
            r_arr    = subdf_shape["R"].values
            t_arr    = subdf_shape["T"].values
            rt_arr   = subdf_shape["RplusT"].values

            ax_r.plot(lam_arr, r_arr, label=f"Shape {shape_num}")
            ax_t.plot(lam_arr, t_arr, label=f"Shape {shape_num}")
            ax_rt.plot(lam_arr, rt_arr, label=f"Shape {shape_num}")

            # Add lines to out_csv_lines
            for _, rowdata in subdf_shape.iterrows():
                out_csv_lines.append(
                    f"{rowdata['csv_file']},{rowdata['crys_label']},{rowdata['shape']},{rowdata['row']},"
                    f"{rowdata['lambda']},{rowdata['freq']},{rowdata['n']},{rowdata['k']},"
                    f"{rowdata['R']},{rowdata['T']},{rowdata['RplusT']}"
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

        out_png = os.path.join(shapes_dir, f"{label_for_title}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

        out_csv = os.path.join(shapes_dir, f"{label_for_title}.csv")
        with open(out_csv, "w") as fout:
            fout.write("\n".join(out_csv_lines) + "\n")

    # B) For each shape => compare CSVs
    for shape_num, subdf_shape in df.groupby("shape"):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8,10), sharex=True)
        ax_r, ax_t, ax_rt = axes

        out_csv_lines = ["csv_file,crys_label,shape,row,lambda,freq,n,k,R,T,RplusT"]

        # Group by csv_file
        for csv_file, subdf_csvfile in subdf_shape.groupby("csv_file"):
            subdf_csvfile = subdf_csvfile.sort_values("lambda")

            lam_arr = subdf_csvfile["lambda"].values
            r_arr   = subdf_csvfile["R"].values
            t_arr   = subdf_csvfile["T"].values
            rt_arr  = subdf_csvfile["RplusT"].values

            label_for_legend = subdf_csvfile["crys_label"].iloc[0]  # e.g. "C0.0"

            ax_r.plot(lam_arr, r_arr, label=label_for_legend)
            ax_t.plot(lam_arr, t_arr, label=label_for_legend)
            ax_rt.plot(lam_arr, rt_arr, label=label_for_legend)

            # Add lines
            for _, rowdata in subdf_csvfile.iterrows():
                out_csv_lines.append(
                    f"{rowdata['csv_file']},{rowdata['crys_label']},{rowdata['shape']},{rowdata['row']},"
                    f"{rowdata['lambda']},{rowdata['freq']},{rowdata['n']},{rowdata['k']},"
                    f"{rowdata['R']},{rowdata['T']},{rowdata['RplusT']}"
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

        out_csv = os.path.join(crys_dir, f"shape_{shape_num}.csv")
        with open(out_csv, "w") as fout:
            fout.write("\n".join(out_csv_lines) + "\n")

    print("\nDone!")
    print(f"Full dataset saved in:\n  {all_data_excel}\n  {all_data_csv}")
    print(f"Plots & partial CSVs saved under '{base_dir}'.")


if __name__ == "__main__":
    main()
