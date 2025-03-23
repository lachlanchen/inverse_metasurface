#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from glob import glob

def analyze_csv_file(csv_path):
    """
    Diagnostic function to analyze a CSV file and identify why records are not being processed
    """
    print(f"\nAnalyzing file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Basic file stats
    print(f"Total rows: {len(df)}")
    print(f"Total unique shape_idx values: {df['shape_idx'].nunique()}")
    
    # Check columns
    r_cols = [c for c in df.columns if c.startswith("R@")]
    t_cols = [c for c in df.columns if c.startswith("T@")]
    print(f"Found {len(r_cols)} R@ columns and {len(t_cols)} T@ columns")
    
    # Check c-values
    c_values = sorted(df['c'].unique())
    print(f"Found {len(c_values)} unique c-values: {c_values}")
    
    # Create shape_uid and verify grouping
    df["shape_uid"] = (df["prefix"].astype(str) + "_" +
                      df["nQ"].astype(str) + "_" +
                      df["nS"].astype(str) + "_" +
                      df["shape_idx"].astype(str))
                      
    print(f"Total unique shape_uid values: {df['shape_uid'].nunique()}")
    
    # Check how many shapes have exactly 11 rows
    shape_counts = df.groupby('shape_uid').size()
    shapes_with_11 = sum(shape_counts == 11)
    print(f"Shapes with exactly 11 rows: {shapes_with_11} out of {len(shape_counts)}")
    
    # Sample a few shapes to analyze their c values
    if shapes_with_11 > 0:
        shapes_to_check = shape_counts[shape_counts == 11].index[:3]  # Get up to 3 shapes with 11 rows
        print("\nSample shapes with 11 rows:")
        for shape in shapes_to_check:
            shape_df = df[df['shape_uid'] == shape]
            print(f"  Shape {shape}: c values = {sorted(shape_df['c'].tolist())}")
    
    # Check for vertices and Q1 points
    if 'vertices_str' in df.columns:
        # Count rows with valid vertices
        sample_vertices = df.iloc[0]['vertices_str'] if len(df) > 0 else "None"
        print(f"\nSample vertices_str: {sample_vertices}")
        
        # Check if any shapes have valid Q1 points after shifting
        valid_count = 0
        for i in range(min(5, len(df))):  # Check up to 5 rows
            row = df.iloc[i]
            v_str = str(row.get("vertices_str", "")).strip()
            if v_str:
                raw_pairs = v_str.split(";")
                all_xy = []
                for pair in raw_pairs:
                    pair = pair.strip()
                    if pair:
                        xy = pair.split(",")
                        if len(xy) == 2:
                            try:
                                x_val = float(xy[0])
                                y_val = float(xy[1])
                                all_xy.append([x_val, y_val])
                            except Exception:
                                pass
                
                if all_xy:
                    all_xy = np.array(all_xy)
                    shifted = all_xy - 0.5
                    q1_points = []
                    for (xx, yy) in shifted:
                        if xx > 0 and yy > 0:
                            q1_points.append([xx, yy])
                    
                    if len(q1_points) > 0:
                        valid_count += 1
        
        print(f"Rows with valid Q1 points (sample of 5): {valid_count}/5")
    
    # SPECIAL DIAGNOSTIC: Check if any groups actually have 11 c-values
    if len(c_values) >= 11:
        # Find a few shapes to check their c-values distribution
        grouped = df.groupby("shape_uid")
        valid_groups = 0
        
        for shape_id, group in list(grouped)[:10]:  # Examine first 10 groups
            if len(group) == 11:
                valid_groups += 1
        
        print(f"\nExamined 10 shape groups, found {valid_groups} with exactly 11 rows")
        
        if valid_groups > 0:
            # Check if any shape actually has all the expected c-values
            for shape_id, group in list(grouped)[:5]:
                if len(group) == 11:
                    group_c = sorted(group['c'].unique())
                    print(f"Shape {shape_id} c-values: {group_c}")
                    break

def main():
    parser = argparse.ArgumentParser(description="Diagnose CSV data issues")
    parser.add_argument("input_file", type=str, help="CSV file to diagnose")
    
    args = parser.parse_args()
    
    # Check if input is a file or directory
    if os.path.isfile(args.input_file):
        analyze_csv_file(args.input_file)
    elif os.path.isdir(args.input_file):
        csv_files = glob(os.path.join(args.input_file, "*.csv"))
        if not csv_files:
            print(f"No CSV files found in {args.input_file}")
            return
        
        # Just analyze the first file as a sample
        analyze_csv_file(csv_files[0])
    else:
        print(f"File or directory not found: {args.input_file}")

if __name__ == "__main__":
    main()
