import re
import os
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# 1) Read and parse the file: spectrum_results_c4_vshapes.txt
# --------------------------------------------------------------------------

# filename = "spectrum_results_c4_vshapes_morerand.txt"
filename = "spectrum_results_c4_vcylinders.txt"

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

with open(filename, "r") as f:
    for line in f:
        line = line.strip()

        # 1) Detect new CSV lines
        csv_match = pattern_csv.search(line)
        if csv_match:
            current_csv = csv_match.group(1).strip()
            if current_csv not in data:
                data[current_csv] = {}
            continue

        # 2) Match data lines
        match = pattern_line.search(line)
        if match:
            csv_in_line  = match.group(1).strip()  # e.g. partial_crys_data/partial_crys_C0.0.csv
            shape_str    = match.group(2)          # "1"
            row_str      = match.group(3)          # "2" etc
            lam_str      = match.group(4)          # "1.050"
            r_str        = match.group(5)          # "0.3495"
            t_str        = match.group(6)          # "0.6402"
            rt_str       = match.group(7)          # "0.9897"

            shape_num = int(shape_str)
            lam_val   = float(lam_str)
            r_val     = float(r_str)
            t_val     = float(t_str)
            rt_val    = float(rt_str)

            # Just in case the "current_csv" is not set or mismatched:
            # we'll use the 'csv_in_line' as the key:
            # (They should match, but we'll rely on csv_in_line for correctness.)
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

# --------------------------------------------------------------------------
# 2) We want two sets of plots:
#    A) For each CSV, compare shapes
#    B) For each shape, compare CSVs
# --------------------------------------------------------------------------

# Let's gather a list of all unique CSV keys:
csv_list = sorted(data.keys())

# Also gather the set of shape numbers (usually 1..10).
# We'll just look at all shapes found across all CSV files:
shape_set = set()
for csvfile in data:
    for shp in data[csvfile]:
        shape_set.add(shp)
shape_list = sorted(shape_set)

# Create output folders if not exist
os.makedirs("plots/compare_shapes", exist_ok=True)
os.makedirs("plots/compare_crystallization", exist_ok=True)

# --------------------------------------------------------------------------
# A) For each CSV, compare shapes
#    -> 1 figure, 3 subplots (R, T, R+T)
# --------------------------------------------------------------------------

for csvfile in csv_list:
    # We'll make a new figure with 3 subplots:
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)
    # Unpack for convenience
    ax_r, ax_t, ax_rt = axes

    # Extract a label for the figure from CSV name (e.g., partial_crys_data/partial_crys_C0.3.csv => "C0.3")
    label_for_title = csvfile
    match_c = re.search(r'partial_crys_C([\d.]+)\.csv', csvfile)
    if match_c:
        label_for_title = f"C{match_c.group(1)}"

    # Sort shapes in ascending order so we plot shape=1..10 in order
    shapes_for_this_csv = sorted(data[csvfile].keys())

    for shp in shapes_for_this_csv:
        lam_arr  = np.array(data[csvfile][shp]["lambda"])
        r_arr    = np.array(data[csvfile][shp]["R"])
        t_arr    = np.array(data[csvfile][shp]["T"])
        rt_arr   = np.array(data[csvfile][shp]["RplusT"])

        # Plot on each subplot
        ax_r.plot(lam_arr, r_arr, label=f"Shape {shp}")
        ax_t.plot(lam_arr, t_arr, label=f"Shape {shp}")
        ax_rt.plot(lam_arr, rt_arr, label=f"Shape {shp}")

    # Axes settings
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

    # Save to "plots/compare_shapes/<label_for_title>.png"
    outname = f"plots/compare_shapes/{label_for_title}.png"
    fig.savefig(outname, dpi=150)
    plt.close(fig)

# --------------------------------------------------------------------------
# B) For each shape, compare crystallizations (CSV)
#    -> 1 figure, 3 subplots (R, T, R+T)
# --------------------------------------------------------------------------

for shp in shape_list:
    # new figure with 3 subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)
    ax_r, ax_t, ax_rt = axes

    # We'll gather all CSV files that actually have this shape
    # (some might not, but presumably you have shape=1..10 in all).
    for csvfile in csv_list:
        if shp in data[csvfile]:
            lam_arr  = np.array(data[csvfile][shp]["lambda"])
            r_arr    = np.array(data[csvfile][shp]["R"])
            t_arr    = np.array(data[csvfile][shp]["T"])
            rt_arr   = np.array(data[csvfile][shp]["RplusT"])

            # Extract short label from csvfile
            label_str = csvfile
            match_c = re.search(r'partial_crys_C([\d.]+)\.csv', csvfile)
            if match_c:
                label_str = f"C{match_c.group(1)}"

            ax_r.plot(lam_arr, r_arr, label=label_str)
            ax_t.plot(lam_arr, t_arr, label=label_str)
            ax_rt.plot(lam_arr, rt_arr, label=label_str)

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

    # Save to "plots/compare_crystallization/shape_{shp}.png"
    outname = f"plots/compare_crystallization/shape_{shp}.png"
    fig.savefig(outname, dpi=150)
    plt.close(fig)

print("Done! Plots saved in 'plots/compare_shapes' and 'plots/compare_crystallization'.")
