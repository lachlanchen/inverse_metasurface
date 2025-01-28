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
                    x_val = float(parts[0])
                    y_val = float(parts[1])
                    coords.append((x_val, y_val))
    except FileNotFoundError:
        print(f"Shape file not found: {shape_file}")
    return coords

def gather_run_data(result_csv):
    """
    Parse key parameters (prefix, seed, nQ, nS) from the filename and
    load the CSV data into a DataFrame.
    """
    base_name = os.path.basename(result_csv)
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

    df = pd.read_csv(result_csv)
    # parse c from each row's csvfile
    df['c'] = df['csvfile'].apply(parse_crys_c)

    # store parsed info
    df['prefix'] = prefix
    df['seed'] = seed
    df['nQ'] = nQ
    df['nS'] = nS

    return df

def find_shape_folder(prefix, seed, nQ, nS):
    """
    Find the best matching shape folder based on the prefix, seed, nQ, and nS.
    """
    # Construct a flexible glob pattern
    base_pattern = f"shapes/{prefix}*"
    if seed:
        base_pattern += f"{seed}*"
    base_pattern += f"_nQ{nQ}_nS{nS}*"

    matching_folders = glob.glob(base_pattern)
    if len(matching_folders) == 1:
        return matching_folders[0]
    elif len(matching_folders) > 1:
        print(f"Multiple shape folders match: {matching_folders}")
        # Select the best match or prompt user, etc.
        return matching_folders[0]
    else:
        print(f"No shape folder found for prefix='{prefix}', seed='{seed}', nQ={nQ}, nS={nS}")
        return None

def pivot_spectrum(df):
    """
    Pivot the spectrum data for easier analysis.
    We include prefix, seed, nQ, nS in the index so we can still access them.
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
    Attach shape vertices to the DataFrame by reading shape files.
    """
    vertices_str_list = []
    for _, row in tqdm(df_pivoted.iterrows(), total=df_pivoted.shape[0], desc="Attaching shape vertices"):
        shape_idx = row['shape_idx']
        shape_file = os.path.join(shapes_folder, f"outer_shape{int(shape_idx)}.txt")
        if os.path.isfile(shape_file):
            coords = read_shape_vertices(shape_file)
            coords_str = ";".join([f"{x:.6f},{y:.6f}" for x, y in coords])
        else:
            coords_str = ""
        vertices_str_list.append(coords_str)
    df_pivoted['vertices_str'] = vertices_str_list
    return df_pivoted

def main():
    parser = argparse.ArgumentParser(description="Merge S4 data and attach shape coordinates with robust folder matching.")
    parser.add_argument('--prefix', type=str, default='', help="Prefix to filter files (e.g., 'iccp10kG40NoOv').")
    args = parser.parse_args()

    glob_pattern = f"results/*{args.prefix}*.csv" if args.prefix else "results/*.csv"
    results_csv_list = glob.glob(glob_pattern)
    if not results_csv_list:
        print(f"No matching CSVs found with pattern: {glob_pattern}")
        return

    all_pieces = []
    for result_csv in tqdm(results_csv_list, desc="Processing CSV files"):
        df = gather_run_data(result_csv)
        if df.empty:
            continue
        df_pivoted = pivot_spectrum(df)

        # Now we can get nQ, nS from the pivoted DataFrame
        prefix = df_pivoted['prefix'].iat[0]
        seed   = df_pivoted['seed'].iat[0]
        nQ     = df_pivoted['nQ'].iat[0]
        nS     = df_pivoted['nS'].iat[0]

        # Find the correct shape folder
        shapes_folder = find_shape_folder(prefix, seed, nQ, nS)
        if shapes_folder:
            df_pivoted = attach_shape_vertices(df_pivoted, shapes_folder)
        else:
            df_pivoted['vertices_str'] = ""

        all_pieces.append(df_pivoted)

    if not all_pieces:
        print("No data to merge. Exiting.")
        return

    final_df = pd.concat(all_pieces, ignore_index=True)
    print(f"Merged all runs. final_df.shape = {final_df.shape}")

    # Save output file
    output_name = f"merged_s4_shapes_{args.prefix or 'all'}.csv"
    final_df.to_csv(output_name, index=False)
    print(f"Saved combined data to '{output_name}'")

if __name__ == "__main__":
    main()

