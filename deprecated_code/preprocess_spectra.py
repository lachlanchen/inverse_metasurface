#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from glob import glob

def process_csv_file(csv_path, mode='T', max_points=4):
    """
    Process a CSV file to extract spectra and shapes.
    Groups rows by shape_uid and ensures each shape has 11 c-values.
    
    Args:
        csv_path: Path to the CSV file
        mode: 'R' for reflectance or 'T' for transmittance
        max_points: Maximum number of points to keep in first quadrant
        
    Returns:
        List of dictionaries with 'uid', 'spectra', and 'shape' keys
    """
    print(f"Processing file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Determine which columns to use based on mode
    if mode == 'R':
        spectrum_cols = [c for c in df.columns if c.startswith("R@")]
    else:  # Default to transmittance
        spectrum_cols = [c for c in df.columns if c.startswith("T@")]
        
    if len(spectrum_cols) == 0:
        # Try the other mode as a fallback
        fallback_mode = 'R' if mode == 'T' else 'T'
        fallback_cols = [c for c in df.columns if c.startswith(f"{fallback_mode}@")]
        if fallback_cols:
            print(f"Using {fallback_mode} columns instead of {mode}")
            spectrum_cols = fallback_cols
            mode = fallback_mode
        else:
            raise ValueError(f"No spectrum columns found in {csv_path}")
    
    # Create unique shape identifier
    df["shape_uid"] = (df["prefix"].astype(str) + "_" +
                      df["nQ"].astype(str) + "_" +
                      df["nS"].astype(str) + "_" +
                      df["shape_idx"].astype(str))
    
    # Group by shape_uid
    grouped = df.groupby("shape_uid", sort=False)
    records = []
    
    # Process each group (shape)
    for uid, group in grouped:
        # Skip if we don't have exactly 11 c-values
        if len(group) != 11:
            continue
            
        # Sort by c value to ensure consistent order
        group_sorted = group.sort_values(by="c")
        
        # Extract the spectrum as 11×100
        spectrum = group_sorted[spectrum_cols].values.astype(np.float32)
        
        # Get vertex information from the first row
        first_row = group_sorted.iloc[0]
        v_str = str(first_row.get("vertices_str", "")).strip()
        if not v_str:
            continue
            
        # Parse vertices
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
                    except Exception:
                        continue
                    all_xy.append([x_val, y_val])
        
        all_xy = np.array(all_xy, dtype=np.float32)
        if len(all_xy) == 0:
            continue
            
        # Shift vertices by (0.5, 0.5)
        shifted = all_xy - 0.5
        
        # Keep only first quadrant vertices (x > 0, y > 0)
        q1_points = []
        for (xx, yy) in shifted:
            if xx > 0 and yy > 0:
                q1_points.append([xx, yy])
                
        q1_points = np.array(q1_points, dtype=np.float32)
        n_q1 = len(q1_points)
        
        # Skip if we don't have at least 1 point in Q1 or too many points
        if n_q1 < 1 or n_q1 > max_points:
            continue
            
        # Create fixed-size shape array (max_points × 3)
        shape_4x3 = np.zeros((max_points, 3), dtype=np.float32)
        for i in range(n_q1):
            shape_4x3[i, 0] = 1.0  # Presence flag
            shape_4x3[i, 1] = q1_points[i, 0]  # X-coordinate
            shape_4x3[i, 2] = q1_points[i, 1]  # Y-coordinate
            
        records.append({
            "uid": uid,
            "spectra": spectrum,
            "shape": shape_4x3
        })
    
    print(f"Found {len(records)} valid records in {csv_path}")
    return records

def process_all_csvs(input_folder, output_file, mode='T', max_points=4, sample=0):
    """
    Process all CSV files in the input folder and save the results to a PyTorch file.
    
    Args:
        input_folder: Folder containing CSV files
        output_file: Path to save the output PyTorch file
        mode: 'R' for reflectance or 'T' for transmittance
        max_points: Maximum number of points to keep in first quadrant
        sample: Number of files to process (0 = all)
    """
    # Find all CSV files in the input folder
    csv_files = glob(os.path.join(input_folder, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_folder}")
    
    csv_files.sort()
    
    # Process only a sample if specified
    if sample > 0 and sample < len(csv_files):
        print(f"Processing {sample} of {len(csv_files)} files")
        csv_files = csv_files[:sample]
    
    all_records = []
    
    # Process each CSV file
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        try:
            records = process_csv_file(csv_file, mode=mode, max_points=max_points)
            all_records.extend(records)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    if not all_records:
        raise ValueError("No valid records found in any of the CSV files")
    
    # Extract UIDs, spectra, and shapes
    uids = [rec["uid"] for rec in all_records]
    spectra = np.array([rec["spectra"] for rec in all_records])  # shape: (N, 11, 100)
    shapes = np.array([rec["shape"] for rec in all_records])     # shape: (N, 4, 3)
    
    # Convert to PyTorch tensors
    spectra_tensor = torch.tensor(spectra, dtype=torch.float32)
    shapes_tensor = torch.tensor(shapes, dtype=torch.float32)
    
    # Save to PyTorch file
    torch.save({
        'uids': uids,
        'spectra': spectra_tensor,
        'shapes': shapes_tensor,
        'mode': mode
    }, output_file)
    
    print(f"Total records processed: {len(uids)}")
    print(f"Data saved to {output_file}")
    print(f"Spectra tensor shape: {spectra_tensor.shape}")
    print(f"Shapes tensor shape: {shapes_tensor.shape}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess spectra data from CSV files")
    parser.add_argument("input_folder", type=str, help="Folder containing CSV files")
    parser.add_argument("output_file", type=str, help="Output PyTorch file (.pt)")
    parser.add_argument("-m", "--mode", type=str, choices=['R', 'T'], default='T',
                      help="Spectra mode: R for reflectance, T for transmittance (default: T)")
    parser.add_argument("--max_points", type=int, default=4,
                      help="Maximum number of points to keep in Q1 (default: 4)")
    parser.add_argument("--sample", type=int, default=0, 
                      help="Only process a sample of N CSV files (0 = all)")
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.isdir(args.input_folder):
        parser.error(f"Input folder does not exist: {args.input_folder}")
    
    # Ensure output file has .pt extension
    if not args.output_file.endswith('.pt'):
        args.output_file += '.pt'
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process the data
    process_all_csvs(args.input_folder, args.output_file, 
                    args.mode, args.max_points, args.sample)

if __name__ == "__main__":
    main()
