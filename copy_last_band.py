#!/usr/bin/env python3
"""
Copy the last band to create a 100-band file from a 99-band file.
"""

import torch
import os
from pathlib import Path

# Paths to the input files
file_paths = [
    "AVIRIS_MOUNT_PROCESSED/aviris_processed_intp_partial_crys_C0.0_128_20250407_164734/aviris_tiles_intp.pt",
    "AVIRIS_FOREST_PROCESSED/aviris_processed_intp_partial_crys_C0.0_128_20250407_143401/aviris_tiles_intp.pt"
]

for file_path in file_paths:
    # Create output path with suffix
    input_path = Path(file_path)
    output_path = input_path.parent / f"{input_path.stem}_100.pt"
    
    print(f"Processing: {input_path}")
    
    # Load the tensor
    data = torch.load(input_path)
    original_shape = data.shape
    print(f"Original shape: {original_shape}")
    
    # Determine the dimensionality of the tensor
    if len(original_shape) == 4:  # [num_tiles, num_bands, height, width]
        # Copy the last band (assuming bands are dimension 1)
        last_band = data[:, -1:, :, :]
        # Append the band to create a new tensor
        new_data = torch.cat([data, last_band], dim=1)
    else:
        raise ValueError(f"Unexpected tensor shape: {original_shape}")
    
    print(f"New shape: {new_data.shape}")
    
    # Save the modified tensor
    torch.save(new_data, output_path)
    print(f"Saved to: {output_path}")
    print("-" * 50)

print("Processing completed!")
