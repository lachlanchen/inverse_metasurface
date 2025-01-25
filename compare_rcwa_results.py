#!/usr/bin/env python3

"""
compare_rcwa_results.py

Reads two types of CSV files:
1) The "merged_s4_shapes_YYYYMMDD_HHMMSS.csv" (a wide CSV) that might contain reflection/transmission data 
   in columns like R@1.040, R@1.054, ..., T@1.040, T@1.054, etc., plus a column 'c'.
2) A "results" CSV that the Lua script (metasurface_fixed_shape_and_c_value.lua) produces. 
   This typically has columns: c_value, wavelength_um, freq_1perum, n_eff, k_eff, R, T, R_plus_T.

We then plot and compare R vs. wavelength (and T vs. wavelength) from the two CSVs for a given 'c' value.

Usage:
    python compare_rcwa_results.py --merged-csv merged_s4_shapes_20250119_153038.csv \
                                   --results-csv fixed_shape_c0.5_20250122_101112.csv \
                                   --c-value 0.5

The script will:
 - Parse the wide "merged" CSV and extract the row matching c=<c_value>.
 - Parse the reflection/transmission columns that match R@..., T@... 
 - Parse the "results" CSV from Lua (where each row has wavelength_um, R, T).
 - Plot them together and save the plots in a timestamped folder.

"""

import os
import sys
import csv
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

def read_merged_s4_csv(merged_csv_path, c_value):
    """
    Reads the wide CSV (e.g. 'merged_s4_shapes_20250119_153038.csv').
    It has columns:
        folder_key,NQ,nS,shape_idx,c,R@1.040,R@1.054,...,T@1.040,T@1.054,...
        plus a final 'vertices_str'.
    
    We want to find the row where c = c_value (float).
    Then we extract the wavelengths from the column names (e.g. "R@1.040" -> 1.040) 
    and the corresponding R, T values.

    Returns:
        wave_list: list of floats (wavelengths)
        R_list: list of floats (reflection)
        T_list: list of floats (transmission)
    """
    # We'll parse the entire CSV, find the row(s) that match c_value, 
    # usually there's only one row per c_value (assuming shape_idx=1 or so).
    
    wave_list = []
    R_list = []
    T_list = []
    found_row = False
    
    with open(merged_csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        # We want the reflection headers like "R@1.040", "R@1.054", ...
        # Then the transmission headers like "T@1.040", etc.
        # We'll systematically extract them if they exist.
        
        # We'll store them in a dict: 
        #   waves_for_R = { "R@1.040": 1.040, "R@1.054": 1.054, ... } 
        # so that we can match columns to numeric wavelength.
        # Similarly for T.
        waves_for_R = {}
        waves_for_T = {}
        
        for h in headers:
            if h.startswith("R@"):
                # parse out the numeric part
                lam_str = h.split("@")[1]  # e.g. "1.040"
                try:
                    lam_val = float(lam_str)
                    waves_for_R[h] = lam_val
                except ValueError:
                    pass
            elif h.startswith("T@"):
                lam_str = h.split("@")[1]
                try:
                    lam_val = float(lam_str)
                    waves_for_T[h] = lam_val
                except ValueError:
                    pass
        
        # Now let's read each row
        for row in reader:
            c_str = row.get("c", None)
            if c_str is not None:
                try:
                    c_float = float(c_str)
                except ValueError:
                    continue
                if abs(c_float - c_value) < 1e-9:
                    # This is the row we want
                    found_row = True
                    # Build wave_list, R_list, T_list in ascending order of wavelength
                    # We'll gather (lam, R, T) in a local list, then sort.
                    data_triplets = []
                    
                    # The sets of column names from waves_for_R and waves_for_T
                    # might not be identical, but presumably they are. We'll do a combined approach.
                    all_keys = sorted(set(waves_for_R.keys()).union(set(waves_for_T.keys())),
                                      key=lambda k: waves_for_R.get(k, waves_for_T.get(k, 999999)))
                    for k in all_keys:
                        if k in waves_for_R:
                            lam_val = waves_for_R[k]
                            R_val_str = row.get(k, "")
                            try:
                                R_val = float(R_val_str)
                            except:
                                R_val = None
                        else:
                            lam_val = waves_for_T[k]
                            R_val = None
                        
                        # T column
                        tcol = "T@" + "{:.3f}".format(lam_val)
                        # but the CSV might have more decimals, so let's do a flexible approach:
                        # We'll search for an exact match first:
                        T_val = None
                        # We can also find the T header by scanning waves_for_T 
                        # (which hopefully has an exact float match).
                        # A simpler approach is to see if k in waves_for_T => that means it's T@something
                        # but let's do a direct check:
                        if k in waves_for_T:
                            lam_check = waves_for_T[k]
                            # that means this is a T column
                            T_val_str = row.get(k, "")
                            try:
                                T_val = float(T_val_str)
                            except:
                                T_val = None
                        else:
                            # maybe the T col name is something else, 
                            # but hopefully there's a parallel. We'll do a best guess approach for brevity.
                            # In practice, if the CSV is consistent, we can rely on k in waves_for_T or not.
                            pass
                        
                        # If it's an R column, let's see if there's a T column with the same wavelength:
                        if k in waves_for_R:
                            # find the parallel T column name
                            lam_str2 = f"{lam_val:.3f}"  # e.g. "1.040"
                            tcol2 = f"T@{lam_str2}"
                            if tcol2 in row:
                                try:
                                    T_val = float(row[tcol2])
                                except:
                                    T_val = None
                                
                            data_triplets.append((lam_val, R_val, T_val))
                    
                    # Sort by lam_val
                    data_triplets.sort(key=lambda x: x[0])
                    
                    wave_list = [x[0] for x in data_triplets]
                    R_list    = [x[1] for x in data_triplets]
                    T_list    = [x[2] for x in data_triplets]
                    
                    break  # we found a matching row, so let's stop
    
    if not found_row:
        print(f"[WARNING] No row found in '{merged_csv_path}' with c={c_value}. Returning empty lists.")
    return wave_list, R_list, T_list


def read_lua_results_csv(results_csv_path):
    """
    Reads the CSV from the Lua program, which has columns:
      c_value,wavelength_um,freq_1perum,n_eff,k_eff,R,T,R_plus_T
    
    Returns lists of wavelength, R, T
    """
    lam_list = []
    R_list = []
    T_list = []
    with open(results_csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # parse
            lam_str = row.get("wavelength_um", None)
            R_str   = row.get("R", None)
            T_str   = row.get("T", None)
            if lam_str is None or R_str is None or T_str is None:
                continue
            try:
                lam_val = float(lam_str)
                R_val   = float(R_str)
                T_val   = float(T_str)
                lam_list.append(lam_val)
                R_list.append(R_val)
                T_list.append(T_val)
            except ValueError:
                pass
    return lam_list, R_list, T_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-csv", required=True,
                        help="Path to merged_s4_shapes_*.csv file.")
    parser.add_argument("--results-csv", required=True,
                        help="Path to the results CSV file from metasurface_fixed_shape_and_c_value.lua.")
    parser.add_argument("--c-value", type=float, required=True,
                        help="Crystallization fraction c to select from the merged CSV.")
    args = parser.parse_args()
    
    merged_csv_path = args.merged_csv
    results_csv_path = args.results_csv
    c_val = args.c_value
    
    # 1) Read the merged CSV, extracting R/T for c_val
    merged_wave, merged_R, merged_T = read_merged_s4_csv(merged_csv_path, c_val)
    
    # 2) Read the results CSV
    results_wave, results_R, results_T = read_lua_results_csv(results_csv_path)
    
    # We'll sort them by wavelength if not already
    # but it's probably sorted. Just in case:
    zipped_res = sorted(zip(results_wave, results_R, results_T), key=lambda x: x[0])
    results_wave = [z[0] for z in zipped_res]
    results_R    = [z[1] for z in zipped_res]
    results_T    = [z[2] for z in zipped_res]
    
    # 3) Create a timestamped folder to save plots
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_folder = f"plots_{dt_str}"
    os.makedirs(out_folder, exist_ok=True)
    
    # 4) Plot R comparison
    plt.figure(figsize=(6,4.5))
    if merged_wave and merged_R:
        plt.plot(merged_wave, merged_R, 'o-', label="Merged CSV R", alpha=0.7)
    if results_wave and results_R:
        plt.plot(results_wave, results_R, 's--', label="Lua Results R", alpha=0.7)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Reflection (R)")
    plt.title(f"Reflection comparison for c={c_val}")
    plt.legend()
    plt.tight_layout()
    rplot_path = os.path.join(out_folder, f"Reflection_comparison_c{c_val}.png")
    plt.savefig(rplot_path, dpi=150)
    plt.close()
    
    # 5) Plot T comparison
    plt.figure(figsize=(6,4.5))
    if merged_wave and merged_T:
        plt.plot(merged_wave, merged_T, 'o-', label="Merged CSV T", alpha=0.7)
    if results_wave and results_T:
        plt.plot(results_wave, results_T, 's--', label="Lua Results T", alpha=0.7)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Transmission (T)")
    plt.title(f"Transmission comparison for c={c_val}")
    plt.legend()
    plt.tight_layout()
    tplot_path = os.path.join(out_folder, f"Transmission_comparison_c{c_val}.png")
    plt.savefig(tplot_path, dpi=150)
    plt.close()
    
    print(f"Plots saved to folder: {out_folder}")
    print("Done.")

if __name__ == "__main__":
    main()

