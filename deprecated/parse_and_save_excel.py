#!/usr/bin/env python3
"""
parse_and_save_excel.py
Parses an S4 output file (e.g. "results.txt") into a pandas DataFrame,
saves the entire DataFrame to Excel, and demonstrates how to check or plot
each line for a chosen shape + C value.
"""

import re
import sys
import pandas as pd

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_and_save_excel.py <results.txt>")
        sys.exit(1)

    infile = sys.argv[1]

    # Regex to detect lines like:
    #  "Now processing CSV: partial_crys_data/partial_crys_C0.0.csv"
    pattern_csv = re.compile(r"^Now processing CSV:\s*(.*\.csv)")

    # Regex to parse lines like:
    # partial_crys_data/partial_crys_C0.0.csv | shape=1, row=2, λ=1.050 µm, freq=0.952, (n=3.58, k=0.06) => R=0.3495, T=0.6402, R+T=0.9897
    pattern_line = re.compile(
        r"^(.*?)\s*\|\s*shape\s*=\s*(\d+),\s*row=(\d+),\s*λ=([\d.]+)\s*µm,\s*freq=([\d.]+).*?\(n=([\d.]+),\s*k=([\d.]+)\).*?R=([\-\d.]+),\s*T=([\-\d.]+),\s*R\+T=([\-\d.]+)"
    )

    rows = []
    current_csv = None

    with open(infile, "r") as f:
        for line in f:
            line = line.strip()
            # 1) Check if it's "Now processing CSV"
            m_csv = pattern_csv.search(line)
            if m_csv:
                current_csv = m_csv.group(1).strip()
                continue

            # 2) Check if it's a data line
            m_line = pattern_line.search(line)
            if m_line:
                (
                    csv_in_line,  # e.g. partial_crys_data/partial_crys_C0.0.csv
                    shape_str,
                    row_str,
                    lam_str,
                    freq_str,
                    n_str,
                    k_str,
                    r_str,
                    t_str,
                    rt_str,
                ) = m_line.groups()

                # Convert to numeric
                shape_num = int(shape_str)
                row_num   = int(row_str)
                lam_val   = float(lam_str)
                freq_val  = float(freq_str)
                n_val     = float(n_str)
                k_val     = float(k_str)
                r_val     = float(r_str)
                t_val     = float(t_str)
                rt_val    = float(rt_str)

                rows.append({
                    "csv_file": csv_in_line,
                    "shape": shape_num,
                    "row": row_num,
                    "lambda": lam_val,
                    "freq": freq_val,
                    "n": n_val,
                    "k": k_val,
                    "R": r_val,
                    "T": t_val,
                    "RplusT": rt_val
                })

    # Create a DataFrame
    df = pd.DataFrame(rows)

    # If empty, nothing was parsed
    if df.empty:
        print("No data parsed. Check your file or regex.")
        sys.exit(0)

    # Derive a short "crys_label" from partial_crys_C0.0
    # e.g. partial_crys_data/partial_crys_C0.0.csv => "C0.0"
    def extract_crys_label(x):
        m = re.search(r'partial_crys_C([\d.]+)\.csv', x)
        if m:
            return f"C{m.group(1)}"
        else:
            return "UnknownC"

    df["crys_label"] = df["csv_file"].apply(extract_crys_label)

    # Save entire DataFrame to Excel
    # This ensures you have one big table with all lines
    out_excel = "parsed_s4_data.xlsx"
    df.to_excel(out_excel, index=False)
    print(f"Saved entire data to {out_excel}")

    # Example: set a multi-index of (shape, crys_label)
    # so we can see each shape + crystallization as separate groups
    df_indexed = df.set_index(["shape", "crys_label"]).sort_index()

    # Suppose you want to pick shape=1, crys_label="C0.0", and see the data
    # We do a .loc lookup:
    shape_query = 1
    crys_query = "C0.0"

    if (shape_query, crys_query) in df_indexed.index:
        subset = df_indexed.loc[(shape_query, crys_query)]
        # 'subset' is now all lines for shape=1 & C0.0
        print("Data for shape=1, C0.0 =>")
        print(subset[["row", "lambda", "freq", "n", "k", "R", "T", "RplusT"]])
    else:
        print(f"No data found for shape={shape_query}, crys_label={crys_query}")

    # That subset should contain exactly the runs for that shape
    # so you can verify if the data only includes one run of lambda or multiple.

    # If needed, you can do further plotting or checks here.

if __name__ == "__main__":
    main()
