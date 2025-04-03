from filter2shape2filter_pipeline import run_pipeline
import torch

# Create a random filter
filter_11x100 = torch.rand(11, 100)

# Just get the reconstructed filter
recon_filter = run_pipeline(filter_11x100)
print(f"Reconstructed filter shape: {recon_filter.shape}")  # Will print [11, 100]

# Get both the shape and reconstructed filter
shape, recon_filter = run_pipeline(filter_11x100, return_shape=True)
print(f"Shape: {shape.shape}, Filter: {recon_filter.shape}")  # Will print [4, 3], [11, 100]

# Visualize the process
recon_filter = run_pipeline(filter_11x100, visualize=True)