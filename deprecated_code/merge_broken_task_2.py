#!/usr/bin/env python3
import os
import re
import glob
import argparse
import pandas as pd
from tqdm import tqdm

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
        print(f"Shape file not found: {shape_file}")
    return coords

def gather_run_data(csv_file):
    """
    Parse key parameters from the CSV filename and load the CSV data.
    Also apply parse_crys_c to the 'csvfile' column.
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
        print(f"Warning: Can't parse filename: {base_name}")

    # Use the python engine to avoid C engine errors.
    df = pd.read_csv(csv_file, engine='python')
    if 'csvfile' in df.columns:
        df['c'] = df['csvfile'].apply(parse_crys_c)
    else:
        df['c'] = None

    df['prefix'] = prefix
    df['seed'] = seed
    df['nQ'] = nQ
    df['nS'] = nS

    return df

def find_shape_folder(prefix, seed, nQ, nS):
    """
    Find the best matching shapes folder based on prefix, seed, nQ, and nS.
    """
    base_pattern = os.path.join("shapes", f"{prefix}*")
    if seed:
        base_pattern += f"{seed}*"
    base_pattern += f"_nQ{nQ}_nS{nS}*"
    matching_folders = glob.glob(base_pattern)
    if len(matching_folders) == 1:
        return matching_folders[0]
    elif len(matching_folders) > 1:
        print(f"Multiple shape folders match: {matching_folders}")
        return matching_folders[0]
    else:
        print(f"No shape folder found for prefix='{prefix}', seed='{seed}', nQ={nQ}, nS={nS}")
        return None

def pivot_spectrum(df):
    """
    Pivot the spectrum data for easier analysis.
    The wavelength (wavelength_um) is formatted to three decimals.
    R and T values are pivoted with columns prefixed 'R@' and 'T@'.
    """
    df['wave_str'] = df['wavelength_um'].apply(lambda x: f"{x:.3f}")
    pivot_index = ['prefix', 'seed', 'nQ', 'nS', 'shape_idx', 'c']
    pivoted_R = df.pivot_table(
        index=pivot_index,
        columns='wave_str',
        values='R',
        aggfunc='mean'
    ).add_prefix('R@')
    pivoted_T = df.pivot_table(
        index=pivot_index,
        columns='wave_str',
        values='T',
        aggfunc='mean'
    ).add_prefix('T@')
    pivoted = pivoted_R.merge(pivoted_T, left_index=True, right_index=True, how='outer')
    pivoted.reset_index(inplace=True)
    return pivoted

def attach_shape_vertices(df_pivoted, shapes_folder):
    """
    For each row in the pivoted DataFrame, read the corresponding shape file
    from the shapes folder and attach the vertices as a string.
    (No extra quotes are added here.)
    """
    vertices_str_list = []
    for _, row in tqdm(df_pivoted.iterrows(), total=df_pivoted.shape[0], desc="Attaching shape vertices"):
        shape_idx = row['shape_idx']
        shape_file = os.path.join(shapes_folder, f"outer_shape{int(shape_idx)}.txt")
        if os.path.isfile(shape_file):
            coords = read_shape_vertices(shape_file)
            # Build a string with coordinates separated by semicolons
            coords_str = ";".join([f"{x:.6f},{y:.6f}" for x, y in coords])
        else:
            coords_str = ""
        vertices_str_list.append(coords_str)
    df_pivoted['vertices_str'] = vertices_str_list
    return df_pivoted

def split_csv_files(batch_num, max_num, prefix):
    """
    For each CSV file in the results/ folder (optionally filtered by prefix),
    read the file and save only the rows corresponding to the first batch_num shapes.
    The split files are saved into split_csvs/ with the filename appended by _sub_first{batch_num}.
    Files already split are skipped.
    """
    if not os.path.exists("split_csvs"):
        os.makedirs("split_csvs")
    csv_pattern = os.path.join("results", f"*{prefix}*.csv") if prefix else os.path.join("results", "*.csv")
    csv_files = glob.glob(csv_pattern)
    for csv_file in tqdm(csv_files, desc="Splitting CSV files"):
        base_name = os.path.basename(csv_file)
        split_file = os.path.join("split_csvs", f"{os.path.splitext(base_name)[0]}_sub_first{batch_num}.csv")
        if os.path.isfile(split_file):
            print(f"Split file {split_file} exists, skipping.")
            continue
        print(f"Processing {csv_file} ...")
        try:
            df = pd.read_csv(csv_file, engine='python')
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
        if 'shape_idx' in df.columns:
            df['shape_idx'] = pd.to_numeric(df['shape_idx'], errors='coerce')
            # Keep only rows with shape_idx <= batch_num
            df_split = df[df['shape_idx'] <= batch_num]
        else:
            print(f"Column 'shape_idx' not found in {csv_file}. Skipping splitting for this file.")
            continue
        df_split.to_csv(split_file, index=False)
        print(f"Saved split file to {split_file}")

def merge_split_csvs(prefix):
    """
    Merge the CSV files in the split_csvs/ folder (optionally filtered by prefix).
    For each file, pivot the spectrum data from long to wide and attach shape vertices.
    """
    split_pattern = os.path.join("split_csvs", f"*{prefix}*.csv") if prefix else os.path.join("split_csvs", "*.csv")
    split_csv_files_list = glob.glob(split_pattern)
    if not split_csv_files_list:
        print(f"No CSV files found in split_csvs with pattern {split_pattern}")
        return None
    all_pieces = []
    for split_csv in tqdm(split_csv_files_list, desc="Processing split CSV files"):
        df = gather_run_data(split_csv)
        if df.empty:
            continue
        # Pivot the long-format spectrum data to wide format.
        df_pivoted = pivot_spectrum(df)
        # Get key parameters from the pivoted DataFrame.
        prefix_val = df_pivoted['prefix'].iat[0]
        seed_val   = df_pivoted['seed'].iat[0]
        nQ_val     = df_pivoted['nQ'].iat[0]
        nS_val     = df_pivoted['nS'].iat[0]
        shapes_folder = find_shape_folder(prefix_val, seed_val, nQ_val, nS_val)
        if shapes_folder:
            df_pivoted = attach_shape_vertices(df_pivoted, shapes_folder)
        else:
            df_pivoted['vertices_str'] = ""
        all_pieces.append(df_pivoted)
    if not all_pieces:
        print("No data to merge. Exiting.")
        return None
    final_df = pd.concat(all_pieces, ignore_index=True)
    print(f"Merged all runs. final_df.shape = {final_df.shape}")
    return final_df

def main():
    parser = argparse.ArgumentParser(description="Merge S4 results with broken task splitting and merging.")
    parser.add_argument("--batch_num", type=int, default=10000,
                        help="Number of shapes per batch to process (default 10000).")
    parser.add_argument("--max_num", type=int, default=80000,
                        help="Maximum number of shapes (default 80000).")
    parser.add_argument("--prefix", type=str, default="",
                        help="Prefix to filter files (e.g., 'myrun_seed12345_g40').")
    args = parser.parse_args()

    # Step 1: Split the large CSV files in results/ into split_csvs/ with only the first batch of shapes.
    split_csv_files(args.batch_num, args.max_num, args.prefix)

    # Step 2: Merge the split CSV files from split_csvs/ into the final wide-format table.
    merged_df = merge_split_csvs(args.prefix)
    if merged_df is None:
        print("No merged data to save. Exiting.")
        return

    output_name = f"merged_broken_shapes_{args.prefix or 'all'}.csv"
    merged_df.to_csv(output_name, index=False)
    print(f"Saved merged data to '{output_name}'")

if __name__ == "__main__":
    main()

