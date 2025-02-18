#!/usr/bin/env python3
"""
This script splits each large CSV file in the "results" folder (matching a given prefix)
into multiple parts—each containing only the rows for a batch of shapes (e.g. 1–10000, 10001–20000, … up to max_num).
The splits are saved in the folder "split_csvs" (skipping those that already exist with the proper columns).
Then, for each batch, it loads all the split files (across all nQ values), pivots the long-format data
into wide format (with columns "R@...", "T@...", etc.) and attaches shape vertices (as a plain string).
The merged file for that batch is saved into "merged_csvs". For example, with max_num=80000 and batch_num=10000,
each results file is split into 8 pieces (per nQ), and then for each batch interval the splits from all nQ are merged.
Existing merged files are skipped.
"""

import os
import re
import glob
import argparse
import pandas as pd
from tqdm import tqdm

###############################################################################
# UTILITY FUNCTIONS (used in both stages)
###############################################################################

def parse_crys_c(csvfile_path):
    """
    Extract crystallization fraction from the filename.
    Example: 'partial_crys_C0.0.csv' -> 0.0
    """
    match = re.search(r'partial_crys_C([\d\.]+)\.csv', csvfile_path)
    if match:
        return float(match.group(1))
    return None

def read_shape_vertices(shape_file):
    """
    Read a shape file and return a list of (x, y) tuples.
    """
    coords = []
    try:
        with open(shape_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) == 2:
                    try:
                        x_val = float(parts[0])
                        y_val = float(parts[1])
                        coords.append((x_val, y_val))
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"[WARN] Shape file not found: {shape_file}")
    return coords

def gather_run_data(csv_file):
    """
    Load a CSV file using the python engine and inject a new column 'c'
    extracted from the 'csvfile' column. Also, parse and add grouping columns.
    """
    base_name = os.path.basename(csv_file)
    pattern = re.compile(r'(.*?)(_seed\d+)?_nQ(\d+)_nS(\d+).*\.csv$')
    m = pattern.match(base_name)
    if m:
        prefix = m.group(1)
        seed = m.group(2) or ""
        nQ = int(m.group(3))
        nS = int(m.group(4))
    else:
        prefix, seed, nQ, nS = "unknown", "", None, None
        print(f"[WARN] Can't parse filename: {base_name}")
    try:
        df = pd.read_csv(csv_file, engine='python', on_bad_lines='skip')
    except Exception as e:
        print(f"[ERROR] Reading {csv_file} failed: {e}")
        return pd.DataFrame()
    if 'csvfile' in df.columns:
        df['c'] = df['csvfile'].apply(parse_crys_c)
    else:
        df['c'] = None
    # Also ensure grouping columns are present
    df['prefix'] = prefix
    df['seed'] = seed
    df['nQ'] = nQ
    df['nS'] = nS
    return df

def pivot_spectrum(df):
    """
    Pivot the long-format spectrum data to wide format.
    The resulting DataFrame will have an index of [prefix, seed, nQ, nS, shape_idx, c]
    and columns for R@... and T@... (and optionally (R+T)@... if available).
    """
    # Format wavelength to 3 decimals.
    df['wave_str'] = df['wavelength_um'].apply(lambda x: f"{x:.3f}")
    pivot_index = ['prefix', 'seed', 'nQ', 'nS', 'shape_idx', 'c']
    pivot_R = df.pivot_table(
        index=pivot_index,
        columns='wave_str',
        values='R',
        aggfunc='mean'
    ).add_prefix("R@")
    pivot_T = df.pivot_table(
        index=pivot_index,
        columns='wave_str',
        values='T',
        aggfunc='mean'
    ).add_prefix("T@")
    pivoted = pivot_R.merge(pivot_T, left_index=True, right_index=True, how='outer')
    pivoted.reset_index(inplace=True)
    return pivoted

def attach_shape_vertices(df_pivoted, shapes_folder):
    """
    For each row in the pivoted DataFrame, read the corresponding shape file
    from shapes_folder (using filename "outer_shape<shape_idx>.txt") and create a
    single string of coordinates (separated by semicolons). No extra quotes are added.
    """
    vertices_str_list = []
    for _, row in tqdm(df_pivoted.iterrows(), total=df_pivoted.shape[0], desc="Attaching shape vertices"):
        shape_idx = row['shape_idx']
        shape_file = os.path.join(shapes_folder, f"outer_shape{int(shape_idx)}.txt")
        if os.path.isfile(shape_file):
            coords = read_shape_vertices(shape_file)
            coords_str = ";".join(f"{x:.6f},{y:.6f}" for x, y in coords)
        else:
            coords_str = ""
        vertices_str_list.append(coords_str)
    df_pivoted['vertices_str'] = vertices_str_list
    return df_pivoted

def find_shape_folder(prefix, seed, nQ, nS):
    """
    Find the best matching shapes folder based on prefix, seed, nQ, and nS.
    Expected folder pattern: shapes/<prefix>*{seed}*_nQ{nQ}_nS{nS}*
    """
    base_pattern = os.path.join("shapes", f"{prefix}*")
    if seed:
        base_pattern += f"{seed}*"
    base_pattern += f"_nQ{nQ}_nS{nS}*"
    matching = glob.glob(base_pattern)
    if not matching:
        print(f"[WARN] No shape folder found for prefix='{prefix}', seed='{seed}', nQ={nQ}, nS={nS}")
        return None
    return matching[0]

###############################################################################
# SPLITTING STAGE: Produce splits for each CSV file into batches of shape_idx ranges
###############################################################################

def parse_filename_get_nQnS(csv_path):
    """
    Given a filename like:
      myrun_seed12345_g40_nQ1_nS100000_b0.35_r0.30.csv
    returns (prefix, nQ, nS) e.g. ('myrun_seed12345_g40', 1, 100000)
    """
    base = os.path.basename(csv_path)
    pattern = re.compile(r'^(.*?)_nQ(\d+)_nS(\d+).*\.csv$')
    m = pattern.match(base)
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), int(m.group(3))

def need_resplit(outpath):
    """
    Returns True if the file at outpath either does not exist or does not contain
    the required grouping columns: 'prefix', 'nQ', and 'nS'.
    """
    if not os.path.exists(outpath):
        return True
    try:
        with open(outpath, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
        required = {'prefix', 'nQ', 'nS'}
        if required.issubset(set(header)):
            return False
    except Exception:
        pass
    return True

def split_one_csv_into_batches(csv_path, batch_num, max_num, outfolder="split_csvs"):
    """
    Reads the entire CSV from csv_path and for each batch interval:
      lower_bound < shape_idx <= upper_bound,
    (with upper_bound = i * batch_num, for i=1,..., max_num//batch_num)
    saves a subset CSV into outfolder. Each output file name will be:
       <basename>_sub_first<upper_bound>.csv
    Skips if the output file already exists with the required columns.
    """
    if not os.path.exists(outfolder):
        os.makedirs(outfolder, exist_ok=True)
    prefix, nQ, nS = parse_filename_get_nQnS(csv_path)
    if prefix is None:
        print(f"[WARN] Skipping {csv_path} (filename pattern mismatch)")
        return []
    base = os.path.basename(csv_path)
    try:
        df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
    except Exception as e:
        print(f"[ERROR] Reading {csv_path} failed: {e}")
        return []
    if "shape_idx" not in df.columns:
        print(f"[WARN] 'shape_idx' column missing in {csv_path}. Skipping.")
        return []
    # Ensure shape_idx is numeric.
    df['shape_idx'] = pd.to_numeric(df['shape_idx'], errors='coerce')
    # Inject grouping columns.
    df["prefix"] = prefix
    df["nQ"] = nQ
    df["nS"] = nS

    split_paths = []
    max_batches = max_num // batch_num
    for i in range(1, max_batches + 1):
        lower = (i - 1) * batch_num
        upper = i * batch_num
        df_sub = df[(df['shape_idx'] > lower) & (df['shape_idx'] <= upper)]
        if df_sub.empty:
            print(f"[INFO] No rows for shapes in ({lower}, {upper}] in {csv_path}.")
            continue
        outname = f"{os.path.splitext(base)[0]}_sub_first{upper}.csv"
        outpath = os.path.join(outfolder, outname)
        if not need_resplit(outpath):
            print(f"[SPLIT] {outpath} exists with required columns. Skipping.")
            split_paths.append(outpath)
            continue
        df_sub.to_csv(outpath, index=False)
        print(f"[SPLIT] Wrote {outpath} with shape {df_sub.shape}")
        split_paths.append(outpath)
    return split_paths

def split_all_csvs(prefix, batch_num, max_num, results_folder="results", outfolder="split_csvs"):
    """
    For all CSV files in results_folder matching the prefix, split them into batches.
    Returns a list of all produced split file paths.
    """
    pattern = os.path.join(results_folder, f"{prefix}_nQ*_nS*.csv")
    csv_files = glob.glob(pattern)
    if not csv_files:
        print(f"[ERROR] No results files found matching {pattern}.")
        return []
    all_splits = []
    for csv_file in csv_files:
        splits = split_one_csv_into_batches(csv_file, batch_num, max_num, outfolder)
        all_splits.extend(splits)
    return all_splits

###############################################################################
# MERGING STAGE: Merge the split files per batch interval (across all nQ)
###############################################################################

def merge_batch(batch_upper, prefix, split_folder="split_csvs", merged_folder="merged_csvs"):
    """
    For a given batch (i.e. split files whose name ends with _sub_first{batch_upper}.csv),
    load each split file (for different nQ), pivot its data, attach shape vertices,
    then concatenate all and write a merged CSV into merged_folder.
    The output file is named: merged_s4_shapes_<prefix>_first<batch_upper>.csv
    Skips if the output file already exists.
    """
    if not os.path.exists(merged_folder):
        os.makedirs(merged_folder, exist_ok=True)
    pattern = os.path.join(split_folder, f"{prefix}*_sub_first{batch_upper}.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"[MERGE] No split files found for batch {batch_upper} (pattern {pattern})")
        return
    merged_pieces = []
    for fpath in files:
        try:
            df = gather_run_data(fpath)
        except Exception as e:
            print(f"[ERROR] Could not read {fpath}: {e}")
            continue
        if df.empty:
            continue
        df_pivot = pivot_spectrum(df)
        if df_pivot.empty:
            continue
        # Retrieve grouping keys to find shape folder.
        try:
            px = df_pivot['prefix'].iat[0]
            seed = df_pivot['seed'].iat[0]
            nq = df_pivot['nQ'].iat[0]
            ns = df_pivot['nS'].iat[0]
        except Exception as e:
            print(f"[ERROR] Retrieving keys from {fpath}: {e}")
            continue
        shape_folder = find_shape_folder(px, seed, nq, ns)
        df_pivot = attach_shape_vertices(df_pivot, shape_folder)
        merged_pieces.append(df_pivot)
    if not merged_pieces:
        print(f"[MERGE] No data to merge for batch {batch_upper}.")
        return
    final_df = pd.concat(merged_pieces, ignore_index=True)
    outname = f"merged_s4_shapes_{prefix}_first{batch_upper}.csv"
    outpath = os.path.join(merged_folder, outname)
    if os.path.exists(outpath):
        print(f"[MERGE] {outpath} exists. Skipping merge for batch {batch_upper}.")
        return
    final_df.to_csv(outpath, index=False)
    print(f"[MERGE] Wrote merged file {outpath} with shape {final_df.shape}")

def merge_all_batches(prefix, batch_num, max_num, split_folder="split_csvs", merged_folder="merged_csvs"):
    """
    For each batch interval (i=1..max_num//batch_num), merge the corresponding split files.
    """
    max_batches = max_num // batch_num
    for i in range(1, max_batches + 1):
        upper = i * batch_num
        merge_batch(upper, prefix, split_folder, merged_folder)

###############################################################################
# MAIN
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Split results CSVs into batches (up to max_num shapes) and merge each batch across nQ."
    )
    parser.add_argument("--prefix", required=True,
                        help="Prefix to filter files (e.g., 'myrun_seed12345_g40').")
    parser.add_argument("--batch_num", type=int, default=10000,
                        help="Batch size (number of shapes per split).")
    parser.add_argument("--max_num", type=int, default=80000,
                        help="Maximum shape index to process (e.g., 80000 will produce 8 batches).")
    args = parser.parse_args()

    # SPLITTING STAGE: Produce splits for all results files.
    print("[STAGE 1] Splitting CSV files ...")
    all_splits = split_all_csvs(args.prefix, args.batch_num, args.max_num, results_folder="results", outfolder="split_csvs")
    if not all_splits:
        print("[ERROR] No splits produced. Exiting.")
        return

    # MERGING STAGE: For each batch interval, merge the corresponding split files.
    print("[STAGE 2] Merging batches ...")
    merge_all_batches(args.prefix, args.batch_num, args.max_num, split_folder="split_csvs", merged_folder="merged_csvs")
    print("[DONE] All merging complete.")

if __name__ == "__main__":
    main()

