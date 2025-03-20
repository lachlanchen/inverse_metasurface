#!/usr/bin/env python3
import os
import re
import glob
import argparse
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
    # Example pattern:  (.*?)(_seed\d+)?_nQ(\d+)_nS(\d+)
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
    # Join them into a string
    return ";".join(f"{x:.6f},{y:.6f}" for (x,y) in coords)

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

###############################################################################
# Main line-by-line merging logic
###############################################################################

def process_large_csv(
    csv_path, 
    shapes_folder,
    max_shape_num,
    chunk_size,
    out_dir
):
    """
    Read the large CSV line by line, grouping lines for each shape_idx,
    then pivoting them into one row: [prefix, seed, nQ, nS, shape_idx, c, R@..., T@..., vertices_str].
    Writes chunked CSVs of up to chunk_size shapes each, ignoring shapes > max_shape_num.
    
    :param csv_path: path to the large S4 results CSV
    :param shapes_folder: path to the matching shapes/ folder (or None if not found)
    :param max_shape_num: only process shapes <= this index
    :param chunk_size: number of shapes per output CSV chunk
    :param out_dir: folder to write the chunked CSVs
    """
    prefix, seed, nQ, nS = parse_filename_params(csv_path)
    if prefix is None:
        print(f"Could not parse nQ/nS from {csv_path}. Skipping.")
        return

    # Create a reusable chunk writer
    chunk_index = 1       # which chunk number we are on
    shape_count_in_chunk = 0
    pivoted_buffer = []   # store pivoted rows (as strings) until we flush

    # We'll need a final sorted list of wavelengths. But typically, we can guess from the first shape or keep them in ascending order.
    # Because we want columns: R@..., T@... . Let's store them in a stable order as we see them.
    # Alternatively, if you know it's always exactly 100 lines per shape with ascending wavelength, just store them in that order.

    # We'll parse the header line first to confirm column indexes
    with open(csv_path, 'r') as f_in:
        header = next(f_in).rstrip('\n')
        cols = header.split(',')
        # Example columns:
        # 0: csvfile
        # 1: shape_idx
        # 2: row_idx
        # 3: wavelength_um
        # 4: freq_1perum
        # 5: n_eff
        # 6: k_eff
        # 7: R
        # 8: T
        # 9: R_plus_T
        try:
            idx_csvfile    = cols.index('csvfile')
            idx_shape_idx  = cols.index('shape_idx')
            idx_row_idx    = cols.index('row_idx')
            idx_wavelength = cols.index('wavelength_um')
            idx_R          = cols.index('R')
            idx_T          = cols.index('T')
        except ValueError as e:
            print(f"Error: CSV {csv_path} missing expected columns. {e}")
            return

        # We'll maintain a stable list of wave_str for pivot
        # (We expect the same set of wavelengths for every shape, e.g. 100 lines.)
        # We'll discover them from the first shape we parse.
        wave_str_list = []

        # We'll store partial lines for the current shape
        current_shape_idx = None
        current_c_value   = None
        # We'll store R/T keyed by wave_str
        current_R_dict = {}
        current_T_dict = {}

        def flush_current_shape():
            """
            Pivot the data for the current shape into one CSV row (string).
            Then add it to pivoted_buffer.
            """
            if current_shape_idx is None or len(current_R_dict)==0:
                return

            # Build a row of: prefix, seed, nQ, nS, shape_idx, c, R@..., T@..., vertices_str
            # We assume wave_str_list is the consistent ordering of wavelengths
            shape_str = str(current_shape_idx)
            c_str = f"{current_c_value:.5g}" if current_c_value is not None else ""

            # If shapes_folder is known, read shape file:
            vertices_str = ""
            if shapes_folder:
                vertices_str = read_shape_vertices(shapes_folder, current_shape_idx)

            row_cells = [
                prefix,
                seed,
                str(nQ),
                str(nS),
                shape_str,
                c_str
            ]
            # Then all R@...
            for wv in wave_str_list:
                val = current_R_dict.get(wv, "")
                row_cells.append(f"{val:.10g}" if val else "")
            # Then all T@...
            for wv in wave_str_list:
                val = current_T_dict.get(wv, "")
                row_cells.append(f"{val:.10g}" if val else "")
            # Then vertices:
            row_cells.append(vertices_str)

            row_str = ",".join(row_cells)
            pivoted_buffer.append(row_str)

        # We'll also define our eventual CSV header row:
        # prefix,seed,nQ,nS,shape_idx,c, (R@ for each wave), (T@ for each wave), vertices_str
        # We'll fill wave_str_list once from the first shape.
        # For chunk writing, we store them in memory. Once we get chunk_size shapes, we flush to a new file.

        # We'll parse lines:
        pbar = tqdm(desc=f"Reading {os.path.basename(csv_path)}", unit="lines")
        for line in f_in:
            pbar.update()
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 10:
                continue  # skip malformed lines

            try:
                shape_idx_val = int(parts[idx_shape_idx])
            except ValueError:
                # skip malformed
                continue

            # If we've gone beyond max_shape_num, we can stop reading the file entirely.
            if max_shape_num is not None and shape_idx_val>max_shape_num:
                # flush last shape
                flush_current_shape()
                # then break
                break

            # If shape_idx changes, flush the old shape, then start a new one
            if current_shape_idx is None:
                # first shape
                current_shape_idx = shape_idx_val
                # parse c from csvfile
                current_c_value = parse_crys_c(parts[idx_csvfile])
                current_R_dict.clear()
                current_T_dict.clear()
            elif shape_idx_val != current_shape_idx:
                # flush the old shape
                flush_current_shape()
                # add to chunk_count
                shape_count_in_chunk += 1
                # if we reached chunk_size, flush chunk to disk
                if shape_count_in_chunk>=chunk_size:
                    write_chunk_csv(
                        wave_str_list,
                        pivoted_buffer,
                        prefix, nQ,
                        chunk_index,
                        out_dir
                    )
                    chunk_index += 1
                    shape_count_in_chunk = 0
                    pivoted_buffer.clear()

                # start a new shape
                current_shape_idx = shape_idx_val
                current_c_value   = parse_crys_c(parts[idx_csvfile])
                current_R_dict.clear()
                current_T_dict.clear()

            # read wavelength, R, T
            wv = float(parts[idx_wavelength])
            Rf = float(parts[idx_R])
            Tf = float(parts[idx_T])

            # We convert wv to a wave_str (like "1.040" if you want 3 decimal places)
            wave_str = f"{wv:.3f}"

            # If we haven't discovered wave_str_list yet, or it's smaller than the total lines (the first shape),
            # we can accumulate wave_str. But let's do that only for the first shape or so.
            if len(wave_str_list)<100:  # or use a set approach
                if wave_str not in wave_str_list:
                    wave_str_list.append(wave_str)

            current_R_dict[wave_str] = Rf
            current_T_dict[wave_str] = Tf

        # Done reading file => flush the last shape in case the file ended in the middle
        flush_current_shape()
        shape_count_in_chunk += 1
        pbar.close()

    # If pivoted_buffer is not empty, we still need to flush it to disk as final chunk
    # but only if shape_count_in_chunk>0 means the last shape was valid
    if shape_count_in_chunk>0 and pivoted_buffer:
        write_chunk_csv(
            wave_str_list,
            pivoted_buffer,
            prefix, nQ,
            chunk_index,
            out_dir
        )
        pivoted_buffer.clear()


def write_chunk_csv(
    wave_str_list,
    pivoted_rows,
    prefix,
    nQ,
    chunk_index,
    out_dir
):
    """
    Write pivoted_rows to a CSV file in out_dir.
    The header is:
      prefix,seed,nQ,nS,shape_idx,c, (R@wave1..waveN), (T@wave1..waveN), vertices_str
    """
    # We'll name it something like:
    # prefix_nQ1_chunk1.csv or so.  Or you can do shape ranges, but we'd need shape min..max.  
    # For simplicity, let's do chunk_{chunk_index:03d}.csv. 
    # If you'd rather do shape ranges, you'd have to track that in process_large_csv. 

    # Here we just do chunk_{chunk_index:03d}.csv. 
    # Or if you prefer "prefix_nQ1_00001-10000.csv" you'd need shape min & max in pivoted_rows.
    # Let's do shape min & max from pivoted_rows so the user can see the shape range:

    shape_indices = []
    for row_str in pivoted_rows:
        # row_str = "prefix,seed,nQ,nS,shape_idx,c,R@...,T@...,vertices_str"
        # shape_idx is at index=4
        # Let's parse it quickly
        parts = row_str.split(',')
        if len(parts)>5:
            try:
                shape_i = int(parts[4])
                shape_indices.append(shape_i)
            except:
                pass
    if shape_indices:
        s_min = min(shape_indices)
        s_max = max(shape_indices)
    else:
        s_min = 1
        s_max = 0  # no data?

    os.makedirs(out_dir, exist_ok=True)
    chunk_filename = os.path.join(
        out_dir,
        f"{prefix}_nQ{nQ}_{s_min:05d}-{s_max:05d}_chunk{chunk_index}.csv"
    )

    # Build the header
    # prefix, seed, nQ, nS, shape_idx, c, R@..., T@..., vertices_str
    R_headers = [f"R@{wv}" for wv in wave_str_list]
    T_headers = [f"T@{wv}" for wv in wave_str_list]
    header_cells = [
        "prefix",
        "seed",
        "nQ",
        "nS",
        "shape_idx",
        "c"
    ] + R_headers + T_headers + ["vertices_str"]

    with open(chunk_filename, 'w') as f_out:
        f_out.write(",".join(header_cells)+"\n")
        for row_str in pivoted_rows:
            f_out.write(row_str+"\n")

    print(f"Wrote chunk: {chunk_filename}  (shapes={len(set(shape_indices))})")


###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Line-by-line merging of large S4 CSVs for nQ1..4, "
                    "up to --max_shape_num, saving in chunks of --chunk_size."
    )
    parser.add_argument('--prefix', required=True,
                        help="File prefix in results/ (e.g. myrun_seed12345_g40).")
    parser.add_argument('--max_shape_num', type=int, default=80000,
                        help="Stop processing shapes above this index.")
    parser.add_argument('--chunk_size', type=int, default=10000,
                        help="Number of shapes per output CSV chunk.")
    args = parser.parse_args()

    # We'll try to find the 4 CSV files for nQ1..4
    # e.g. results/*myrun_seed12345_g40*_nQ1_*.csv
    # Then process them in ascending order (nQ1..nQ4).
    # If your naming has multiple matches, adapt as needed.
    all_csvs = []
    for nQ in [1,2,3,4]:
        pat = f"results/*{args.prefix}*_nQ{nQ}_*.csv"
        matches = glob.glob(pat)
        if not matches:
            print(f"No file found for nQ={nQ} pattern={pat}")
            continue
        # If multiple matches, pick the largest or the first, or prompt user:
        # We'll pick the first for simplicity.
        # Or you can do for m in sorted(matches): ...
        chosen_csv = sorted(matches)[0]
        all_csvs.append(chosen_csv)

    if not all_csvs:
        print("No matching CSV files found for any nQ1..4. Exiting.")
        return

    # We create an output dir for all chunked merges
    out_dir = f"merged_{args.prefix}"
    os.makedirs(out_dir, exist_ok=True)

    # For each CSV, parse prefix, seed, nQ, nS => find shape folder
    # Then line-by-line parse => chunked pivot
    for csv_path in all_csvs:
        prefix, seed, nQ, nS = parse_filename_params(csv_path)
        if prefix is None:
            continue
        # find shape folder
        shapes_folder = find_shapes_folder(prefix, seed, nQ, nS)

        print(f"\n=== Processing {os.path.basename(csv_path)} for nQ={nQ}, shapes_folder={shapes_folder} ===")
        process_large_csv(
            csv_path=csv_path,
            shapes_folder=shapes_folder,
            max_shape_num=args.max_shape_num,
            chunk_size=args.chunk_size,
            out_dir=out_dir
        )

    print("\nAll done. Check the folder:", out_dir)


if __name__ == "__main__":
    main()

