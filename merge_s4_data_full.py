#!/usr/bin/env python3
import os
import re
import glob
import argparse
import pandas as pd

# Make sure you have tqdm installed: pip install tqdm
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
    Return a list of lines (verbatim) to preserve original data.
    """
    coords = []
    with open(shape_file, 'r') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            coords.append(line)  # Store the exact line (e.g. "0.123456789,0.987654321")
    return coords

def gather_run_data(result_csv):
    """
    Read one results CSV (like 'results/20250105_181104_output_nQ1_nS1000.csv'),
    parse the numeric 'NQ' and 'nS' from the filename,
    then return a DataFrame that has columns:

      ['folder_key', 'NQ', 'nS', 'csvfile', 'shape_idx', 'row_idx',
       'wavelength_um', 'freq_1perum', 'n_eff', 'k_eff', 'R', 'T', 'c']

    where 'c' is from partial_crys_Cxxx.
    """
    base_name = os.path.basename(result_csv)  # e.g. 20250105_181104_output_nQ1_nS1000.csv

    # parse nQ=? nS=? from the base_name
    m = re.search(r'_nQ(\d+)_nS(\d+)\.csv$', base_name)
    if m:
        NQ = int(m.group(1))
        nS = int(m.group(2))
    else:
        NQ, nS = None, None

    df = pd.read_csv(result_csv)
    # df columns => csvfile, shape_idx, row_idx, wavelength_um, freq_1perum, n_eff, k_eff, R, T, R_plus_T

    # parse c from each row's csvfile
    df['c'] = df['csvfile'].apply(parse_crys_c)

    # add NQ, nS
    df['NQ'] = NQ
    df['nS'] = nS

    # a unique run/folder identifier (like the prefix 20250105_181104)
    folder_key = base_name.split('_output')[0]  # e.g. 20250105_181104
    df['folder_key'] = folder_key

    return df

def pivot_spectrum(df):
    """
    Given a DataFrame with columns:
      shape_idx, c, wavelength_um, R, T, plus maybe more
    We want to pivot so each row is (folder_key, NQ, nS, shape_idx, c) => R@..., T@...
    for each wavelength.

    Returns that pivoted DataFrame, without truncating or rounding numeric data.
    """
    # Convert wavelength to a string without truncation
    df['wave_str'] = df['wavelength_um'].apply(lambda x: str(x))

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
    We store them verbatim to avoid truncation or rounding.
    """
    vertices_str_list = []
    for _, row in tqdm(df_pivoted.iterrows(), total=df_pivoted.shape[0], desc="Attaching shape vertices"):
        shape_idx = row['shape_idx']
        shape_file = os.path.join(shapes_folder, f"outer_shape{int(shape_idx)}.txt")
        if os.path.isfile(shape_file):
            lines = read_shape_vertices(shape_file)
            # Join them into one string
            coords_str = ";".join(lines)
        else:
            coords_str = ""
        vertices_str_list.append(coords_str)

    df_pivoted['vertices_str'] = vertices_str_list
    return df_pivoted

def main():
    parser = argparse.ArgumentParser(
        description="Merge S4 data from results and attach shape coordinates."
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help="Specify a datetime prefix, e.g. 20250114_175110. Only CSVs that start with this prefix will be merged."
    )
    args = parser.parse_args()
    
    # Construct glob pattern
    # If user provides a prefix like '20250114_175110',
    # we match files: results/20250114_175110_output_nQ*_nS*.csv
    # Otherwise, we match results/*_output_nQ*_nS*.csv
    if args.prefix:
        prefix_pattern = f"{args.prefix}_"
    else:
        prefix_pattern = ''
    glob_pattern = f"results/{prefix_pattern}output_nQ*_nS*.csv"

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
        # 2) Pivot spectrum
        df_pivoted = pivot_spectrum(df)

        # 3) Determine the shapes folder
        base_name = os.path.basename(result_csv)
        folder_key = base_name.split('_output')[0]  # e.g. 20250105_181104
        m = re.search(r'_nQ(\d+)_nS(\d+)\.csv$', base_name)
        if not m:
            print(f"Warning: can't parse nQ/nS from {base_name}")
            shapes_folder = None
        else:
            nQ_int = int(m.group(1))
            nS_int = int(m.group(2))
            shapes_folder = f"shapes/{folder_key}-poly-wo-hollow-nQ{nQ_int}-nS{nS_int}"

        # 4) Attach shape vertices if folder exists
        if shapes_folder and os.path.isdir(shapes_folder):
            df_pivoted = attach_shape_vertices(df_pivoted, shapes_folder)
        else:
            df_pivoted['vertices_str'] = ""

        all_pieces.append(df_pivoted)

    # 5) Concatenate everything
    final_df = pd.concat(all_pieces, ignore_index=True)
    print("Merged all runs. final_df.shape =", final_df.shape)

    # 6) Write out the combined CSV
    if args.prefix:
        outname = f"merged_s4_shapes_{args.prefix}.csv"
    else:
        outname = "merged_s4_shapes.csv"
    final_df.to_csv(outname, index=False)
    print(f"Saved combined data to '{outname}'")

if __name__ == "__main__":
    main()
