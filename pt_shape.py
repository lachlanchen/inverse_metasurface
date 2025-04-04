#!/usr/bin/env python3
# pt_shape.py - Print shape and statistics of a PyTorch tensor file

import torch
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Print info about a PyTorch tensor file')
    parser.add_argument('pt_file', help='Path to the PyTorch tensor file')
    args = parser.parse_args()
    
    print(f"Loading tensor from: {args.pt_file}")
    data = torch.load(args.pt_file)
    
    print(f"Tensor shape: {data.shape}")
    print(f"Number of tiles: {data.shape[0]}")
    print(f"Number of bands: {data.shape[1]}")
    print(f"Tile size: {data.shape[2]}x{data.shape[3]}")
    print(f"Data type: {data.dtype}")
    print(f"Memory usage: {data.element_size() * data.nelement() / (1024*1024):.2f} MB")
    
    # Print some basic statistics
    print("\nStatistics:")
    print(f"Min value: {data.min().item():.6f}")
    print(f"Max value: {data.max().item():.6f}")
    print(f"Mean value: {data.mean().item():.6f}")
    print(f"Standard deviation: {data.std().item():.6f}")
    
    # Check for NaN or Inf values
    if torch.isnan(data).any() or torch.isinf(data).any():
        print("\nWARNING: Dataset contains NaN or Inf values!")
        print(f"NaN values: {torch.isnan(data).sum().item()}")
        print(f"Inf values: {torch.isinf(data).sum().item()}")
    else:
        print("\nNo NaN or Inf values found.")
    
if __name__ == "__main__":
    main()
