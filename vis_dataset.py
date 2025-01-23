#!/usr/bin/env python3
import sys
import os
import argparse
import datetime
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_arguments():
    """
    Parses command-line arguments.
    
    Usage examples:
      python vis_dataset.py merged_s4_shapes_20250114_175110.csv
      python vis_dataset.py merged_s4_shapes_20250114_175110.csv -list 1,1 1,2 2,10000
      python vis_dataset.py merged_s4_shapes_20250114_175110.csv -N 2

    Returns:
      args.input_csv (str): path to the CSV file
      args.shape_list (list of tuples): e.g. [(NQ, shape_idx), (NQ, shape_idx), ...]
      args.num_random (int): if set, pick up to this many shapes *per distinct NQ*
    """
    parser = argparse.ArgumentParser(
        description="Visualize R/T spectra from an S4 result dataset."
    )
    parser.add_argument("input_csv", type=str,
                        help="Path to input CSV (e.g. merged_s4_shapes_20250114_175110.csv).")

    parser.add_argument("-list", nargs='+', default=None,
                        help="List of shapes in the form NQ,shape_idx (e.g. 1,1 2,10000).")

    parser.add_argument("-N", type=int, default=None,
                        help="Randomly pick up to N shapes per distinct NQ (ignored if -list is used).")

    args = parser.parse_args()

    # Process shape_list
    shape_list = []
    if args.list is not None:
        for item in args.list:
            # Expect "NQ,shape_idx"
            part = item.split(",")
            if len(part) != 2:
                print(f"ERROR parsing shape_id '{item}': must be NQ,shape_idx")
                sys.exit(1)
            try:
                shape_list.append((int(part[0]), int(part[1])))
            except ValueError:
                print(f"ERROR parsing shape_id '{item}': must be integer,int")
                sys.exit(1)

    return args.input_csv, shape_list, args.N

def read_csv_data(csv_path):
    """
    Reads the CSV file into a pandas DataFrame.
    
    Returns:
      df (pd.DataFrame)
    """
    df = pd.read_csv(csv_path)
    return df

def get_available_R_columns(df):
    """
    Returns a sorted list of R columns in the form 'R@X.XXX'.
    Also returns the numeric frequency values as a sorted list.
    """
    rcols = [col for col in df.columns if col.startswith("R@")]
    # Example: 'R@1.040' -> frequency = 1.040
    # Sort them numerically by the part after '@'
    rcols_sorted = sorted(rcols, key=lambda x: float(x.split("@")[1]))
    freqs = [float(x.split("@")[1]) for x in rcols_sorted]
    return rcols_sorted, freqs

def get_available_T_columns(df):
    """
    Same idea as get_available_R_columns but for T.
    """
    tcols = [col for col in df.columns if col.startswith("T@")]
    tcols_sorted = sorted(tcols, key=lambda x: float(x.split("@")[1]))
    return tcols_sorted

def collect_unique_shapes(df):
    """
    Return a sorted list of all unique (NQ, shape_idx) in the data.
    """
    unique_pairs = df[["NQ","shape_idx"]].drop_duplicates()
    # Sort by NQ then shape_idx
    unique_pairs = unique_pairs.sort_values(["NQ","shape_idx"])
    shape_list = list(zip(unique_pairs["NQ"], unique_pairs["shape_idx"]))
    return shape_list

def pick_random_shapes(shape_list, df, N):
    """
    shape_list: list of all (NQ, shape_idx) available
    df: the entire dataset
    N: how many shapes to pick per distinct NQ

    Returns up to N shapes for each NQ, chosen randomly.
    """
    out = []
    # Group shapes by NQ
    from collections import defaultdict
    shapes_by_nq = defaultdict(list)
    for (nq, sid) in shape_list:
        shapes_by_nq[nq].append(sid)

    for nq, sids in shapes_by_nq.items():
        if len(sids) <= N:
            chosen_sids = sids
        else:
            chosen_sids = random.sample(sids, N)
        for sid in chosen_sids:
            out.append((nq, sid))

    return sorted(out)

def plot_spectra_for_shape(df, nq, shape_idx, out_dir):
    """
    For a given shape (identified by NQ and shape_idx),
    gather all rows (one per c), then plot R(ω), T(ω), and R+T(ω).
    
    We create a single figure with 3 subplots or 3 separate plots 
    (here we'll do 1 figure, 3 subplots).
    Then we save as e.g. shape_NQX_shapeIDX.png inside out_dir.
    """
    subset = df[(df["NQ"] == nq) & (df["shape_idx"] == shape_idx)].copy()
    if subset.empty:
        print(f"[WARN] No rows found for shape (NQ={nq}, shape_idx={shape_idx}). Skipping.")
        return

    # Sort subset by c (to have lines in ascending c order)
    subset.sort_values(by="c", inplace=True)

    # Grab column names for R and T
    rcols, freqs = get_available_R_columns(df)
    tcols = get_available_T_columns(df)

    # Prepare figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    fig.suptitle(f"Shape (NQ={nq}, shape_idx={shape_idx}) Spectra")

    # Plot each c as a separate line
    for _, row in subset.iterrows():
        cval = row["c"]
        # R(ω)
        Rvals = row[rcols].values.astype(float)
        # T(ω)
        Tvals = row[tcols].values.astype(float)
        # R+T
        RTvals = Rvals + Tvals

        axs[0].plot(freqs, Rvals, label=f"c={cval:.3f}")
        axs[1].plot(freqs, Tvals, label=f"c={cval:.3f}")
        axs[2].plot(freqs, RTvals, label=f"c={cval:.3f}")

    axs[0].set_title("Reflection (R)")
    axs[1].set_title("Transmission (T)")
    axs[2].set_title("R + T")
    for ax in axs:
        ax.set_xlabel("Frequency (ω)")
        ax.set_ylabel("Magnitude")
        ax.grid(True)
    axs[2].legend(loc="upper left", bbox_to_anchor=(1.0,1.0))

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on right for legend
    outpath = os.path.join(out_dir, f"shape_{nq}_{shape_idx}.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {outpath}")

def main():
    input_csv, shape_list, num_random = parse_arguments()

    # 1) Read data
    df = read_csv_data(input_csv)

    # 2) Decide which shapes to plot
    all_shapes = collect_unique_shapes(df)

    if len(shape_list) > 0:
        # user-specified shapes
        chosen_shapes = shape_list
    else:
        # pick random shapes
        if num_random is None:
            # If neither -list nor -N provided, just pick 1 shape per NQ by default
            num_random = 1
        chosen_shapes = pick_random_shapes(all_shapes, df, num_random)

    # 3) Create output folder with timestamp
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"vis_{timestamp_str}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # 4) Plot for each chosen shape
    #    We ensure at least one shape per distinct NQ if random.
    #    If user-specified list, just loop that.
    done_shapes = set()
    for (nq, sid) in chosen_shapes:
        if (nq, sid) in done_shapes:
            continue
        done_shapes.add((nq, sid))
        plot_spectra_for_shape(df, nq, sid, out_dir)

    print("Done.")

if __name__ == "__main__":
    main()

