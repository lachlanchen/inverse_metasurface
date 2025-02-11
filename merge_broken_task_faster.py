#!/usr/bin/env python3
import os
import re
import glob
import csv
import argparse
import pandas as pd
from tqdm import tqdm

###############################################################################
# Helper functions
###############################################################################

def parse_filename_params(csv_path):
    """
    From a file like 'myrun_seed12345_g40_nQ1_nS100000_b0.35_r0.30.csv',
    extract (prefix='myrun_seed12345_g40', seed='_seed12345', nQ=1, nS=100000).
    If not matched, returns (None, None, None, None).
    """
    base_name = os.path.basename(csv_path)
    # Example pattern: (.*?)(_seed\d+)?_nQ(\d+)_nS(\d+)
    pattern = re.compile(r'(.*?)(_seed\d+)?_nQ(\d+)_nS(\d+).*\.csv$')
    m = pattern.match(base_name)
    if m:
        prefix = m.group(1)
        seed   = m.group(2) or ""
        nQ     = int(m.group(3))
        nS     = int(m.group(4))
        return (prefix, seed, nQ, nS)
    return (None, None, None, None)

def parse_crys_c(csvfile_value):
    """
    E.g. 'partial_crys_data/partial_crys_C0.7.csv' => c=0.7
    Return float or None if not found.
    """
    if not csvfile_value:
        return None
    match = re.search(r'partial_crys_C([\d\.]+)\.csv', csvfile_value)
    if match:
        return float(match.group(1))
    return None

def find_shapes_folder(prefix, seed, nQ, nS):
    """
    Attempt to find a folder in './shapes' that matches typical naming:
    e.g. shapes/myrun_seed12345_g40*_nQ1_nS100000*
    Return the first match or None if none found.
    """
    base_pattern = f"shapes/{prefix}*"
    if seed:
        base_pattern += f"{seed}*"
    base_pattern += f"_nQ{nQ}_nS{nS}*"
    matches = glob.glob(base_pattern)
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # If multiple, pick first or do something else
        return matches[0]
    return None

def read_shape_vertices(shapes_folder, shape_idx):
    """
    Attempt to read 'outer_shape{shape_idx}.txt' in shapes_folder.
    Returns a string "x1,y1;x2,y2;..." or "" if not found.
    """
    shape_file = os.path.join(shapes_folder, f"outer_shape{shape_idx}.txt")
    if not os.path.isfile(shape_file):
        return ""
    coords = []
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
                    pass
    return ";".join(f"{x:.6f},{y:.6f}" for (x,y) in coords)

###############################################################################
# Chunk-saving logic
###############################################################################

def save_in_chunks(df, chunk_size, prefix, nQ, out_dir):
    """
    Given a pivoted DataFrame (with columns like 'prefix, seed, nQ, nS, shape_idx, c, ...'),
    write it to multiple CSV files in out_dir, each containing up to chunk_size shapes.
    We'll name each file using the shape range, e.g. prefix_nQ1_00001-10000.csv etc.

    The DataFrame is expected to have a 'shape_idx' column.
    """
    os.makedirs(out_dir, exist_ok=True)

    if df.empty:
        print(f"No data to save for {prefix}_nQ{nQ}.")
        return

    # sort by shape_idx so chunking is consistent
    df = df.sort_values('shape_idx').reset_index(drop=True)

    shape_min = df['shape_idx'].min()
    shape_max = df['shape_idx'].max()

    chunk_start = shape_min
    while chunk_start <= shape_max:
        chunk_end = chunk_start + chunk_size - 1
        mask = (df['shape_idx'] >= chunk_start) & (df['shape_idx'] <= chunk_end)
        chunk_df = df[mask]
        if chunk_df.empty:
            chunk_start = chunk_end + 1
            continue

        s_min = chunk_df['shape_idx'].min()
        s_max = chunk_df['shape_idx'].max()

        chunk_filename = os.path.join(
            out_dir,
            f"{prefix}_nQ{nQ}_{s_min:05d}-{s_max:05d}.csv"
        )
        # We specify quoting so that if 'vertices_str' has commas, it is properly enclosed in quotes
        chunk_df.to_csv(
            chunk_filename,
            index=False,
            quoting=csv.QUOTE_MINIMAL
        )
        print(f"Wrote chunk: {chunk_filename} (shapes={chunk_df.shape[0]})")
        chunk_start = chunk_end + 1

###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Fast in-memory merging of large S4 CSVs for nQ=1..4, "
                    "filtering shape_idx > --max_shape_num, pivoting, and saving in chunks."
    )
    parser.add_argument('--prefix', required=True,
                        help="File prefix in results/ (e.g. 'myrun_seed12345_g40').")
    parser.add_argument('--max_shape_num', type=int, default=80000,
                        help="Ignore any row where shape_idx > this index.")
    parser.add_argument('--chunk_size', type=int, default=10000,
                        help="Number of shapes per output CSV chunk.")
    args = parser.parse_args()

    prefix_filter = args.prefix
    max_shape_num = args.max_shape_num
    chunk_size    = args.chunk_size

    # We'll try to find up to 4 CSV files: nQ=1..4
    # e.g. results/*myrun_seed12345_g40*_nQ1_*.csv
    all_csvs = []
    for nQ in [1,2,3,4]:
        pat = f"results/*{prefix_filter}*_nQ{nQ}_*.csv"
        matches = glob.glob(pat)
        if not matches:
            print(f"No file found for nQ={nQ} pattern={pat}")
            continue
        # pick first or largest if multiple
        chosen_csv = sorted(matches)[0]
        all_csvs.append((nQ, chosen_csv))

    if not all_csvs:
        print("No matching CSV files found for any nQ=1..4. Exiting.")
        return

    out_dir = f"merged_faster_{prefix_filter}"
    os.makedirs(out_dir, exist_ok=True)

    for (nQ, csv_path) in all_csvs:
        print(f"\n=== Processing {os.path.basename(csv_path)} for nQ={nQ} ===")
        pfx, seed, nQ_file, nS = parse_filename_params(csv_path)
        if pfx is None:
            print(f"Skipping {csv_path} (unparsable filename).")
            continue

        # read entire CSV with pandas
        # Use engine='python' + on_bad_lines='skip' to avoid parser crash on lines that have too many/few columns
        try:
            df = pd.read_csv(
                csv_path,
                engine='python',
                on_bad_lines='skip'
            )
        except Exception as e:
            print(f"Failed reading {csv_path} with pandas: {e}")
            continue

        # Filter shape_idx > max_shape_num
        if 'shape_idx' not in df.columns:
            print(f"{csv_path} has no 'shape_idx' column. Skipping.")
            continue

        before_count = df.shape[0]
        df = df[df['shape_idx'] <= max_shape_num].copy()
        after_count = df.shape[0]
        if after_count < before_count:
            print(f"  Filtered out {before_count-after_count} rows with shape_idx > {max_shape_num}.")

        # parse c from csvfile
        if 'csvfile' in df.columns:
            df['c'] = df['csvfile'].apply(parse_crys_c)
        else:
            df['c'] = None

        # We create wave_str from wavelength_um
        if 'wavelength_um' not in df.columns or 'R' not in df.columns or 'T' not in df.columns:
            print(f"{csv_path} is missing needed columns (wavelength_um, R, T). Skipping.")
            continue

        df['wave_str'] = df['wavelength_um'].round(3).astype(str)

        # pivot R
        pivoted_R = df.pivot_table(
            index='shape_idx',
            columns='wave_str',
            values='R',
            aggfunc='mean'
        )
        pivoted_R = pivoted_R.add_prefix('R@')

        # pivot T
        pivoted_T = df.pivot_table(
            index='shape_idx',
            columns='wave_str',
            values='T',
            aggfunc='mean'
        )
        pivoted_T = pivoted_T.add_prefix('T@')

        # merge R and T
        pivoted = pivoted_R.merge(pivoted_T, left_index=True, right_index=True, how='outer')

        # restore shape_idx as a column
        pivoted.reset_index(inplace=True)

        # group c by shape_idx and pick the first non-null
        c_df = df.groupby('shape_idx', as_index=True)['c'].first().to_frame()
        pivoted = pivoted.merge(c_df, left_on='shape_idx', right_index=True, how='left')

        # attach prefix, seed, nQ, nS as columns
        pivoted['prefix'] = pfx
        pivoted['seed']   = seed
        pivoted['nQ']     = nQ_file
        pivoted['nS']     = nS

        # reorder columns to: prefix, seed, nQ, nS, shape_idx, c, [R@...], [T@...]
        # find all R@..., T@... columns
        all_cols = pivoted.columns.tolist()
        # put main fields first
        main_front = ['prefix','seed','nQ','nS','shape_idx','c']
        # the rest are R@..., T@...
        rest_cols = [c for c in all_cols if c not in main_front]
        final_cols = main_front + rest_cols
        pivoted = pivoted[final_cols]

        # attach shape vertices
        shapes_folder = find_shapes_folder(pfx, seed, nQ_file, nS)
        if shapes_folder:
            print(f"  Found shapes folder: {shapes_folder}")
            # We'll do a loop to fill 'vertices_str'
            vertices_list = []
            for sid in tqdm(pivoted['shape_idx'], desc=f"Attaching shapes for nQ={nQ_file}"):
                vs = read_shape_vertices(shapes_folder, sid)
                vertices_list.append(vs)
            pivoted['vertices_str'] = vertices_list
        else:
            print(f"  No shapes folder found for prefix={pfx}, seed={seed}, nQ={nQ_file}, nS={nS}.")
            pivoted['vertices_str'] = ""

        # chunk and save
        save_in_chunks(pivoted, chunk_size, pfx, nQ_file, out_dir)

    print(f"\nAll done. Check the folder: {out_dir}")


if __name__ == "__main__":
    main()

