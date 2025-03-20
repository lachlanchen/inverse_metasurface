#!/usr/bin/env python3
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Folder containing partial_crys_C*.csv
    data_folder = "partial_crys_data"
    # Sibling folder for plots
    out_folder = "partial_crys_plots"
    os.makedirs(out_folder, exist_ok=True)

    # Pattern to capture c value from filename, e.g. partial_crys_C0.0.csv -> 0.0
    # We will assume filenames look like: partial_crys_CX.csv
    # Where X is 0.0, 0.1, ... 1.0
    file_pattern = os.path.join(data_folder, "partial_crys_C*.csv")
    csv_files = sorted(glob.glob(file_pattern))

    # Data structure to store everything
    # We'll store in a dict keyed by c, each entry is another dict containing arrays
    # example: data[c_val]["wavelength"], data[c_val]["n_eff"], data[c_val]["k_eff"]
    data = {}

    # Regex to capture c from filename partial_crys_Csomething.csv
    c_regex = re.compile(r"_C([\d.]+)\.csv$")

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        match = c_regex.search(filename)
        if not match:
            print(f"Skipping file with unexpected name: {filename}")
            continue

        c_str = match.group(1)  # e.g. "0.0" or "0.2" etc.
        try:
            c_val = float(c_str)
        except ValueError:
            print(f"Skipping file {filename}; cannot parse c value from '{c_str}'")
            continue

        # Read the CSV
        df = pd.read_csv(csv_path)
        # Expect columns: Wavelength_um, n_eff, k_eff
        if not {"Wavelength_um", "n_eff", "k_eff"}.issubset(df.columns):
            print(f"File {filename} missing required columns. Skipping.")
            continue

        data[c_val] = {
            "wavelength": df["Wavelength_um"].values,
            "n_eff": df["n_eff"].values,
            "k_eff": df["k_eff"].values
        }

    # Sort c values for plotting in ascending order
    c_values = sorted(data.keys())

    # ========== 1) Plot n_eff vs. wavelength (all c lines) ==========
    plt.figure(figsize=(6, 5))
    for c_val in c_values:
        wavelength = data[c_val]["wavelength"]
        n_eff = data[c_val]["n_eff"]
        plt.plot(wavelength, n_eff, label=f"c={c_val}")

    plt.title("n_eff vs. Wavelength")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("n_eff")
    plt.grid(True)
    plt.legend()
    out_n_path = os.path.join(out_folder, "n_eff_plot.png")
    plt.savefig(out_n_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_n_path}")

    # ========== 2) Plot k_eff vs. wavelength (all c lines) ==========
    plt.figure(figsize=(6, 5))
    for c_val in c_values:
        wavelength = data[c_val]["wavelength"]
        k_eff = data[c_val]["k_eff"]
        plt.plot(wavelength, k_eff, label=f"c={c_val}")

    plt.title("k_eff vs. Wavelength")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("k_eff")
    plt.grid(True)
    plt.legend()
    out_k_path = os.path.join(out_folder, "k_eff_plot.png")
    plt.savefig(out_k_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_k_path}")

    print("Done!")

if __name__ == "__main__":
    main()

