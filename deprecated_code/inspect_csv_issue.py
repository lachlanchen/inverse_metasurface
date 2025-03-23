#!/usr/bin/env python3
"""
CSV Inspector - A utility script to diagnose CSV file issues
Run with: python inspect_csv_issue.py /path/to/problematic.csv
"""

import os
import sys
import pandas as pd
import numpy as np

def inspect_csv(filepath):
    """Thoroughly inspect a CSV file for issues"""
    print(f"\n{'='*60}")
    print(f"INSPECTING CSV FILE: {filepath}")
    print(f"{'='*60}")
    
    # Check if file exists and get size
    if not os.path.exists(filepath):
        print(f"ERROR: File does not exist: {filepath}")
        return
        
    file_size = os.path.getsize(filepath) / (1024*1024)  # Convert to MB
    print(f"File size: {file_size:.2f} MB")
    
    # Try different engines and parameters
    engines = ['c', 'python']
    for engine in engines:
        print(f"\n--- Attempting to read with {engine} engine ---")
        try:
            # Try with default parameters first
            df = pd.read_csv(filepath, engine=engine, nrows=5)
            print(f"Success! Read 5 rows with {engine} engine")
            print("Column names:", df.columns.tolist())
            print("Sample data:")
            print(df.head(2))
        except Exception as e:
            print(f"Error with {engine} engine: {str(e)}")
            
            # Try with error handling for the python engine
            if engine == 'python':
                try:
                    print("\nTrying with error_bad_lines=False...")
                    df = pd.read_csv(filepath, engine='python', 
                                     on_bad_lines='skip', # for newer pandas
                                     nrows=5)
                    print("Success with error handling!")
                    print("Column names:", df.columns.tolist())
                    print("Sample data:")
                    print(df.head(2))
                except Exception as e2:
                    print(f"Still failed with error handling: {str(e2)}")
    
    # Try to detect delimiter
    print("\n--- Delimiter Detection ---")
    try:
        with open(filepath, 'r', errors='replace') as f:
            first_few_lines = [next(f) for _ in range(3)]
        
        for line_num, line in enumerate(first_few_lines):
            print(f"Line {line_num} counts:")
            for delim in [',', ';', '\t', '|']:
                count = line.count(delim)
                print(f"  '{delim}': {count} occurrences")
    except Exception as e:
        print(f"Error reading file directly: {str(e)}")
    
    # Check for encoding issues
    print("\n--- Encoding Check ---")
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                # Just try to read a bit
                f.read(1024)
            print(f"File can be read with {encoding} encoding")
        except UnicodeDecodeError:
            print(f"File cannot be read with {encoding} encoding")
    
    # Try custom chunk reading
    print("\n--- Chunk Reading Test ---")
    try:
        chunk_size = 100
        reader = pd.read_csv(filepath, engine='python', chunksize=chunk_size, on_bad_lines='skip')
        for i, chunk in enumerate(reader):
            if i >= 3:  # Just read first few chunks
                break
            print(f"Successfully read chunk {i+1}, shape: {chunk.shape}")
            
            # Check for problematic columns or data types
            for col in chunk.columns:
                null_count = chunk[col].isna().sum()
                pct_null = null_count / len(chunk) * 100
                print(f"Column '{col}': {null_count} nulls ({pct_null:.1f}%)")
    except Exception as e:
        print(f"Error during chunk reading: {str(e)}")
    
    print("\n--- RECOMMENDATIONS ---")
    print("Based on the inspection, try reading the file with:")
    print("1. pd.read_csv(filepath, engine='python', on_bad_lines='skip', encoding='latin1')")
    print("2. If the file is large, use chunk processing: chunksize=1000")
    print("3. If specific columns are causing issues, specify usecols to only read needed columns")
    print("4. Use try/except blocks to handle errors during processing")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_csv_issue.py /path/to/problematic.csv")
        sys.exit(1)
    
    inspect_csv(sys.argv[1])
