#!/usr/bin/env python3
import os
import sys
import re
import argparse
import pandas as pd
from tqdm import tqdm

def parse_c_value_from_csvfile(csvfile_str):
    """
    Extracts the c value from something like: 'partial_crys_data/partial_crys_C0.0.csv'.
    Returns float(c) or None if not found.
    """
    match = re.search(r'_C(\d+(\.\d+)?)', csvfile_str)
    if match:
        return float(match.group(1))
    return None

def parse_nq_ns_from_filename(fname):
    """
    Given '20250123_155420_output_nQ1_nS10000.csv',
    returns (prefix='20250123_155420', nQ=1, nS=10000)
    or (None,None,None) if not matching.
    """
    base = os.path.basename(fname)
    # Pattern: <prefix>_output_nQ<num>_nS<num>.csv
    m = re.match(r'^(.*?)_output_nQ(\d+)_nS(\d+)\.csv$', base)
    if not m:
        return None, None, None
    prefix = m.group(1)
    nq = int(m.group(2))
    ns = int(m.group(3))
    return prefix, nq, ns

def load_shape_vertices(prefix, nq, ns):
    """
    Reads shape files from shapes/<prefix>-poly-wo-hollow-nQ{nq}-nS{ns}/outer_shape<shape_idx>.txt
    Returns a dict: { shape_idx -> "x1,y1;x2,y2;..." }
    If folder missing or empty, returns {}.
    """
    folder_name = f"{prefix}-poly-wo-hollow-nQ{nq}-nS{ns}"
    shape_dir = os.path.join("shapes", folder_name)
    if not os.path.isdir(shape_dir):
        return {}

    shape_map = {}
    # Each file is outer_shapeXX.txt
    for fname in os.listdir(shape_dir):
        if fname.startswith("outer_shape") and fname.endswith(".txt"):
            m = re.match(r"outer_shape(\d+)\.txt$", fname)
            if not m:
                continue
            shape_idx = int(m.group(1))
            fullpath = os.path.join(shape_dir, fname)
            with open(fullpath, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            # Convert lines -> "x1,y1;x2,y2;..."
            shape_map[shape_idx] = ";".join(lines)
    return shape_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="Prefix, e.g. 20250123_155420")
    args = parser.parse_args()
    prefix = args.prefix

    # 1) Read & merge the four (or however many) spectrum CSV files from results/
    results_dir = "results"
    all_csv = sorted(os.listdir(results_dir))
    dfs = []
    # We'll look specifically for nQ=1..4, but you can adapt as needed.
    for nq in tqdm([1,2,3,4], desc="Gathering CSV files"):
        file_pattern = f"{prefix}_output_nQ{nq}_nS"
        matched_files = [f for f in all_csv if f.startswith(file_pattern)]
        if not matched_files:
            # If none found, just continue
            continue

        for fname in matched_files:
            fullpath = os.path.join(results_dir, fname)
            pfx, real_nq, real_ns = parse_nq_ns_from_filename(fname)
            if pfx is None:
                continue
            df_tmp = pd.read_csv(fullpath)
            # Extract 'c' from the csvfile column
            df_tmp["c"] = df_tmp["csvfile"].apply(parse_c_value_from_csvfile)
            # Add folder_key, nQ, nS
            df_tmp["folder_key"] = pfx
            df_tmp["nQ"] = real_nq
            df_tmp["nS"] = real_ns
            dfs.append(df_tmp)

    if not dfs:
        print(f"[Error] No spectrum CSV found for prefix='{prefix}'. Exiting.")
        sys.exit(1)

    df_spectrum_long = pd.concat(dfs, ignore_index=True)

    # 2) Merge shape vertices: build a dictionary for each (nQ, nS) once.
    #   shape_dict[(nQ, nS)][shape_idx] -> vertices_str
    unique_combos = df_spectrum_long[["nQ","nS"]].drop_duplicates()
    shape_dict = {}
    for row in unique_combos.itertuples(index=False):
        dict_key = (row.nQ, row.nS)
        shape_dict[dict_key] = load_shape_vertices(prefix, row.nQ, row.nS)

    # For each row in df_spectrum_long, find shape_idx => vertices_str
    # We'll use tqdm for a progress bar
    vertices_str_list = []
    for row in tqdm(df_spectrum_long.itertuples(), total=len(df_spectrum_long), desc="Merging shapes"):
        nQ_ = row.nQ
        nS_ = row.nS
        sidx = row.shape_idx
        vmap = shape_dict.get((nQ_, nS_), {})
        vertices_str_list.append(vmap.get(sidx, ""))

    df_spectrum_long["vertices_str"] = vertices_str_list

    # 3) Pivot the long table -> wide columns R@..., T@...
    # We'll keep these as index: [folder_key, nQ, nS, shape_idx, c, vertices_str]
    pivot_index = ["folder_key","nQ","nS","shape_idx","c","vertices_str"]

    # Pivot for R
    df_r = df_spectrum_long.pivot_table(
        index=pivot_index,
        columns="wavelength_um",
        values="R"
    )
    # Pivot for T
    df_t = df_spectrum_long.pivot_table(
        index=pivot_index,
        columns="wavelength_um",
        values="T"
    )

    # Rename columns to R@..., T@...
    df_r.columns = [f"R@{col}" for col in df_r.columns]
    df_t.columns = [f"T@{col}" for col in df_t.columns]

    # Merge them side by side
    df_wide = pd.merge(df_r, df_t, left_index=True, right_index=True, how="outer")

    # Move pivot_index back to normal columns
    df_wide.reset_index(inplace=True)

    # 4) Reorder columns in a user-friendly way
    #   first the pivot_index columns, then all R@..., then T@...
    base_cols = list(pivot_index)  # keep them in the same order
    all_cols = list(df_wide.columns)
    r_cols = [c for c in all_cols if c.startswith("R@")]
    t_cols = [c for c in all_cols if c.startswith("T@")]
    final_cols = base_cols + r_cols + t_cols
    df_final = df_wide[final_cols]

    # 5) Save final
    outname = f"merged_s4_shapes_{prefix}.csv"
    df_final.to_csv(outname, index=False)
    print(f"[Done] Merged wide table saved to: {outname}")

if __name__ == "__main__":
    main()

