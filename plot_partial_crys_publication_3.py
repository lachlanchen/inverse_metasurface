#!/usr/bin/env python3
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Attempt to set a publication-like style; fall back gracefully if unavailable.
    if 'seaborn-whitegrid' in plt.style.available:
        plt.style.use('seaborn-whitegrid')
    elif 'seaborn' in plt.style.available:
        plt.style.use('seaborn')
    else:
        print("Warning: 'seaborn-whitegrid' not found; using default style.")
    
    # Update rcParams for publication quality
    plt.rcParams.update({
        'figure.figsize': (8, 6),
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'axes.edgecolor': 'black',
        'lines.linewidth': 3,
        'legend.fontsize': 14,
        'legend.frameon': False,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'grid.color': 'grey',
        'grid.linestyle': '--',
        'grid.linewidth': 0.75,
        'font.family': 'sans-serif'
    })

    # Folder containing partial_crys_C*.csv
    data_folder = "partial_crys_data"
    # Sibling folder for plots
    out_folder = "partial_crys_plots"
    os.makedirs(out_folder, exist_ok=True)

    # Pattern to capture c value from filename, e.g. partial_crys_C0.0.csv -> 0.0
    file_pattern = os.path.join(data_folder, "partial_crys_C*.csv")
    csv_files = sorted(glob.glob(file_pattern))

    data = {}
    c_regex = re.compile(r"_C([\d.]+)\.csv$")

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        match = c_regex.search(filename)
        if not match:
            print(f"Skipping file with unexpected name: {filename}")
            continue

        c_str = match.group(1)
        try:
            c_val = float(c_str)
        except ValueError:
            print(f"Skipping file {filename}; cannot parse c value from '{c_str}'")
            continue

        df = pd.read_csv(csv_path)
        if not {"Wavelength_um", "n_eff", "k_eff"}.issubset(df.columns):
            print(f"File {filename} missing required columns. Skipping.")
            continue

        data[c_val] = {
            "wavelength": df["Wavelength_um"].values,
            "n_eff": df["n_eff"].values,
            "k_eff": df["k_eff"].values
        }

    c_values = sorted(data.keys())

    # ========== 1) Plot Effective Refractive Index vs. Wavelength ==========
    fig, ax = plt.subplots()
    for c_val in c_values:
        wavelength = data[c_val]["wavelength"]
        n_eff = data[c_val]["n_eff"]
        ax.plot(wavelength, n_eff, label=f"c = {c_val}")
    # ax.set_title("Effective Refractive Index vs. Wavelength")
    ax.set_xlabel("Wavelength (μm)")
    ax.set_ylabel("Effective Refractive Index")
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    out_n_path = os.path.join(out_folder, "n_eff_plot.png")
    plt.savefig(out_n_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_n_path}")

    # ========== 2) Plot Effective Extinction Coefficient vs. Wavelength ==========
    fig, ax = plt.subplots()
    for c_val in c_values:
        wavelength = data[c_val]["wavelength"]
        k_eff = data[c_val]["k_eff"]
        ax.plot(wavelength, k_eff, label=f"c = {c_val}")
    # ax.set_title("Effective Extinction Coefficient vs. Wavelength")
    ax.set_xlabel("Wavelength (μm)")
    ax.set_ylabel("Effective Extinction Coefficient")
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    out_k_path = os.path.join(out_folder, "k_eff_plot.png")
    plt.savefig(out_k_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_k_path}")

    print("Done!")

if __name__ == "__main__":
    main()

