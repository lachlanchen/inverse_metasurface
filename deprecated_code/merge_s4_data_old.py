#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd

def parse_crys_c(csvfile_path):
    """
    Given something like 'partial_crys_data/partial_crys_C0.0.csv',
    extract the numeric crystallization fraction: 0.0.
    Returns None if not found.
    """
    # e.g. partial_crys_data/partial_crys_C0.0.csv => c=0.0
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
            line=line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) == 2:
                x_str, y_str = parts
                x_val = float(x_str)
                y_val = float(y_str)
                coords.append((x_val, y_val))
    return coords

def gather_run_data(result_csv):
    """
    Read one results CSV (like 'results/20250105_181104_output_nQ1_nS1000.csv'),
    parse the numeric 'NQ' and 'nS' from the filename if you like,
    then return a DataFrame that has columns:

    ['folder_key', 'NQ', 'nS', 'csvfile', 'shape_idx', 'row_idx', 
     'wavelength_um', 'freq_1perum', 'n_eff', 'k_eff', 'R', 'T', 'c']

    where 'c' is from partial_crys_Cxxx. We'll eventually pivot on shape_idx+c.
    """
    base_name = os.path.basename(result_csv)  # e.g. 20250105_181104_output_nQ1_nS1000.csv
    # parse nQ=? nS=? from the base_name
    # example pattern: "..._output_nQ4_nS1000.csv"
    m = re.search(r'_nQ(\d+)_nS(\d+)\.csv$', base_name)
    if m:
        NQ = int(m.group(1))
        nS = int(m.group(2))
    else:
        # fallback
        NQ, nS = None, None

    df = pd.read_csv(result_csv)
    # df columns => csvfile, shape_idx, row_idx, wavelength_um, freq_1perum, n_eff, k_eff, R, T, R_plus_T

    # parse c from each row's csvfile
    df['c'] = df['csvfile'].apply(parse_crys_c)

    # add NQ, nS
    df['NQ'] = NQ
    df['nS'] = nS

    # Optionally, a unique run/folder identifier (like the prefix 20250105_181104)
    folder_key = base_name.split('_output')[0]  # e.g. 20250105_181104
    df['folder_key'] = folder_key

    return df

def pivot_spectrum(df):
    """
    Given a DataFrame with columns:
      shape_idx, c, wavelength_um, R, T, plus maybe more
    We want to pivot so each row is (shape_idx, c) => wave1_R, wave1_T, wave2_R, wave2_T, ...
    Return that pivoted DataFrame.
    """
    # We'll round or keep wavelength as a float? Let's keep it as a string to avoid floating duplication
    df['wave_str'] = df['wavelength_um'].apply(lambda x: f"{x:.3f}")
    # Create pivot keys like wave1_R, wave1_T:
    # We'll do a "long to wide" approach with a multi-level pivot:
    #   index = (shape_idx, c, maybe NQ?), columns = wave_str, values = R, T
    # We'll need to unstack them carefully.

    # Let's just unpivot (R,T) into separate columns, e.g. "R_{wave_str}" and "T_{wave_str}".
    # One approach: create two separate dataframes, pivot them, and merge.
    pivoted_R = df.pivot_table(
        index=['folder_key','NQ','nS','shape_idx','c'],
        columns='wave_str',
        values='R',
        aggfunc='mean'  # or sum, typically there's only one value
    )
    pivoted_T = df.pivot_table(
        index=['folder_key','NQ','nS','shape_idx','c'],
        columns='wave_str',
        values='T',
        aggfunc='mean'
    )
    # rename columns
    pivoted_R = pivoted_R.add_prefix('R@')  # e.g. R@1.040
    pivoted_T = pivoted_T.add_prefix('T@')

    # Merge them
    pivoted = pivoted_R.merge(pivoted_T, left_index=True, right_index=True, how='outer')
    pivoted.reset_index(inplace=True)  # bring the (folder_key,NQ,nS,shape_idx,c) back as columns

    return pivoted

def attach_shape_vertices(df_pivoted, shapes_folder):
    """
    df_pivoted has columns: [folder_key, NQ, nS, shape_idx, c, R@..., T@...].
    We also have a shapes folder like "shapes/20250105_181104-poly-wo-hollow-nQ1-nS1000".
    For each shape_idx, read 'outer_shape{shape_idx}.txt' and store the first quadrant portion
    or the entire polygon as a single text field or multiple numeric columns.

    We'll do something simple: store them in a single string column "vertices_str".
    """
    vertices_str_list = []
    for i, row in df_pivoted.iterrows():
        shape_idx = row['shape_idx']
        # We assume file: shapes_folder/outer_shape{shape_idx}.txt
        shape_file = os.path.join(shapes_folder, f"outer_shape{int(shape_idx)}.txt")
        if os.path.isfile(shape_file):
            coords = read_shape_vertices(shape_file)
            # Suppose we only want the first quadrant coords or just all coords
            # We'll store them as a single string
            coords_str = ";".join([f"{x:.6f},{y:.6f}" for x,y in coords])
        else:
            coords_str = ""
        vertices_str_list.append(coords_str)

    df_pivoted['vertices_str'] = vertices_str_list
    return df_pivoted

def main():
    # 1) Gather all results CSVs that match pattern: results/*_output_nQ*_nS*.csv
    results_csv_list = glob.glob("results/*_output_nQ*_nS*.csv")
    all_pieces = []

    for result_csv in results_csv_list:
        # 2) Read + parse
        df = gather_run_data(result_csv)
        # 3) Pivot spectrum
        df_pivoted = pivot_spectrum(df)

        # 4) Figure out which shapes folder to use
        #    e.g. if result_csv is "results/20250105_181104_output_nQ1_nS1000.csv",
        #    then shapes folder might be "shapes/20250105_181104-poly-wo-hollow-nQ1-nS1000"
        base_name = os.path.basename(result_csv)
        folder_key = base_name.split('_output')[0]  # e.g. 20250105_181104
        # parse nQ, nS again to build the shape folder name
        m = re.search(r'_nQ(\d+)_nS(\d+)\.csv$', base_name)
        if not m:
            # fallback or skip
            print(f"Warning: can't parse nQ/nS from {base_name}")
            shapes_folder = None
        else:
            nQ_int = int(m.group(1))
            nS_int = int(m.group(2))
            shapes_folder = f"shapes/{folder_key}-poly-wo-hollow-nQ{nQ_int}-nS{nS_int}"

        # 5) Attach shape vertices
        if shapes_folder and os.path.isdir(shapes_folder):
            df_pivoted = attach_shape_vertices(df_pivoted, shapes_folder)
        else:
            df_pivoted['vertices_str'] = ""

        all_pieces.append(df_pivoted)

    # 6) Concatenate everything
    if not all_pieces:
        print("No matching CSVs found in 'results/' that match pattern *_output_nQ*_nS*.csv.")
        return

    final_df = pd.concat(all_pieces, ignore_index=True)
    print("Merged all runs. final_df.shape =", final_df.shape)

    # 7) Write out the combined CSV
    outname = "merged_s4_shapes.csv"
    final_df.to_csv(outname, index=False)
    print(f"Saved combined data to '{outname}'")

if __name__ == "__main__":
    main()
