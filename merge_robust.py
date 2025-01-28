#!/usr/bin/env python3
import os
import re
import glob
import argparse
import pandas as pd
from tqdm import tqdm

def parse_crys_c(csvfile_path):
    """
    Given something like 'partial_crys_data/partial_crys_C0.0.csv',
    extract the numeric crystallization fraction: 0.0.
    Returns None if not found.
    """
    match = re.search(r'partial_crys_C([\d\.]+)\.csv', csvfile_path)
    if match:
        return float(match.group(1))
    return None

def read_shape_vertices(shape_file):
    """
    Read a shape file like outer_shape9.txt, which has lines "x,y".
    Return a list of tuples (x, y).
    """
    coords = []
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
    return coords

def gather_run_data(result_csv):
    """
    Read one results CSV (e.g. 'iccp10kG40NoOv_nQ1_nS10000_b0.25_r0.20.csv'),
    parse the numeric 'NQ' and 'nS' from the filename using a more robust regex,
    then return a DataFrame with these columns:

      ['folder_key', 'NQ', 'nS', 'csvfile', 'shape_idx', 'row_idx',
       'wavelength_um', 'freq_1perum', 'n_eff', 'k_eff', 'R', 'T', 'c']
    """
    base_name = os.path.basename(result_csv)
    # More robust pattern captures:
    # (group1)_nQ(\d+)_nS(\d+)(anything).csv
    pattern = re.compile(r'(.*)_nQ(\d+)_nS(\d+).*\.csv$')
    m = pattern.match(base_name)
    if m:
        folder_key = m.group(1)
        NQ = int(m.group(2))
        nS = int(m.group(3))
    else:
        folder_key = "unknown"
        NQ, nS = None, None
        print(f"Warning: can't parse nQ/nS from filename: {base_name}")

    df = pd.read_csv(result_csv)
    # parse c from each row's csvfile
    df['c'] = df['csvfile'].apply(parse_crys_c)

    # add NQ, nS
    df['NQ'] = NQ
    df['nS'] = nS

    # store the folder_key
    df['folder_key'] = folder_key

    return df

def pivot_spectrum(df):
    """
    Convert a DataFrame with columns:
      [folder_key, NQ, nS, shape_idx, c, wavelength_um, R, T, ...]
    into a wide format so each row is (folder_key, NQ, nS, shape_idx, c)
    plus columns R@..., T@... for each wavelength.
    """
    df['wave_str'] = df['wavelength_um'].apply(lambda x: f"{x:.3f}")

    # Pivot R
    pivoted_R = df.pivot_table(
        index=['folder_key','NQ','nS','shape_idx','c'],
        columns='wave_str',
        values='R',
        aggfunc='mean'
    )
    # Pivot T
    pivoted_T = df.pivot_table(
        index=['folder_key','NQ','nS','shape_idx','c'],
        columns='wave_str',
        values='T',
        aggfunc='mean'
    )

    # Rename columns
    pivoted_R = pivoted_R.add_prefix('R@')
    pivoted_T = pivoted_T.add_prefix('T@')

    # Merge
    pivoted = pivoted_R.merge(pivoted_T, left_index=True, right_index=True, how='outer')
    pivoted.reset_index(inplace=True)

    return pivoted

def attach_shape_vertices(df_pivoted, shapes_folder):
    """
    df_pivoted has columns: [folder_key, NQ, nS, shape_idx, c, R@..., T@...].
    For each shape_idx, read 'outer_shape{shape_idx}.txt' in shapes_folder
    and store its vertices as a single string in 'vertices_str'.
    """
    vertices_str_list = []
    for _, row in tqdm(df_pivoted.iterrows(), total=df_pivoted.shape[0], desc="Attaching shape vertices"):
        shape_idx = row['shape_idx']
        shape_file = os.path.join(shapes_folder, f"outer_shape{int(shape_idx)}.txt")
        if os.path.isfile(shape_file):
            coords = read_shape_vertices(shape_file)
            coords_str = ";".join([f"{x:.6f},{y:.6f}" for x,y in coords])
        else:
            coords_str = ""
        vertices_str_list.append(coords_str)

    df_pivoted['vertices_str'] = vertices_str_list
    return df_pivoted

def main():
    parser = argparse.ArgumentParser(
        description="Merge S4 data from results and attach shape coordinates, using a more flexible filename parser."
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help="Specify a prefix (e.g. 'iccp10kG40NoOv'), to glob only CSVs containing that prefix. If omitted, merges all."
    )
    args = parser.parse_args()

    # Construct glob pattern
    # If user provides --prefix 'iccp10kG40NoOv', we look for files in 'results/*iccp10kG40NoOv*.csv'
    # Otherwise, just pick up all CSVs in results/.
    if args.prefix:
        glob_pattern = f"results/*{args.prefix}*.csv"
    else:
        glob_pattern = "results/*.csv"

    # Gather relevant CSVs
    results_csv_list = glob.glob(glob_pattern)
    if not results_csv_list:
        print(f"No matching CSVs found in 'results/' with pattern: {glob_pattern}")
        return

    all_pieces = []
    print(f"Found {len(results_csv_list)} CSV files to merge.")
    
    # Process each CSV with a progress bar
    for result_csv in tqdm(results_csv_list, desc="Processing CSV files"):
        # 1) Read & parse
        df = gather_run_data(result_csv)
        if df.empty:
            continue

        # 2) Pivot spectrum
        df_pivoted = pivot_spectrum(df)

        # 3) Build shape folder name from folder_key, NQ, nS
        #    If the user always has shape folders in the style:
        #        shapes/{folder_key}-poly-wo-hollow-nQ{NQ}-nS{nS}
        #    Then do:
        folder_key = df_pivoted['folder_key'].iat[0]
        nQ_int = df_pivoted['NQ'].iat[0] if df_pivoted['NQ'].notna().any() else None
        nS_int = df_pivoted['nS'].iat[0] if df_pivoted['nS'].notna().any() else None

        if nQ_int is not None and nS_int is not None:
            shapes_folder = f"shapes/{folder_key}-poly-wo-hollow-nQ{nQ_int}-nS{nS_int}"
        else:
            # fallback
            shapes_folder = None

        # 4) Attach shape vertices if folder exists
        if shapes_folder and os.path.isdir(shapes_folder):
            df_pivoted = attach_shape_vertices(df_pivoted, shapes_folder)
        else:
            df_pivoted['vertices_str'] = ""

        all_pieces.append(df_pivoted)

    # 5) Concatenate everything
    if not all_pieces:
        print("No data frames were created. Exiting.")
        return
    final_df = pd.concat(all_pieces, ignore_index=True)
    print("Merged all runs. final_df.shape =", final_df.shape)

    # 6) Write out the combined CSV
    # e.g.: merged_s4_shapes_iccp10kG40NoOv.csv
    if args.prefix:
        outname = f"merged_s4_shapes_{args.prefix}.csv"
    else:
        outname = "merged_s4_shapes.csv"
    final_df.to_csv(outname, index=False)
    print(f"Saved combined data to '{outname}'")

if __name__ == "__main__":
    main()

