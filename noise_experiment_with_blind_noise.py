#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import numpy.linalg as LA
import shutil
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import math
import random
import json

# Import common functions and classes from the original script
from noise_experiment_with_filter2shape2filter import (
    Shape2FilterModel, Filter2ShapeVarLen, Filter2Shape2FilterFrozen,
    calculate_condition_number, replicate_c4, sort_points_by_angle, plot_shape_with_c4,
    generate_initial_filter, DecoderCNN5Layer
)

# Import train_decoder_with_comparison for consistency 
from noise_experiment_with_filter2shape2filter import train_decoder_with_comparison

from AWAN import AWAN
latent_dim = 11
in_channels = 100

###############################################################################
# IMPROVED DATA LOADING FUNCTIONS
###############################################################################
def is_valid_tile(tile, min_valid_percentage=0.05, max_valid_percentage=0.95, 
                 max_constant_spectral_percentage=0.2, spectral_variance_threshold=1e-6):
    """
    Check if a tile contains valid data (not all 0s or all 1s, and not constant across wavelengths)
    
    Parameters:
    tile: Tensor of shape (C, H, W) or (H, W, C)
    min_valid_percentage: Minimum percentage of non-zero values required
    max_valid_percentage: Maximum percentage of values that can be 1.0
    max_constant_spectral_percentage: Maximum percentage of pixels allowed to have constant spectral values
    spectral_variance_threshold: Threshold for considering spectral values as constant
    
    Returns:
    bool: True if tile is valid, False otherwise
    """
    # Make sure we're working with (C, H, W) format for consistency
    if len(tile.shape) != 3:
        return False
    
    # Determine format and convert to (C, H, W) if needed
    if tile.shape[0] < tile.shape[1] and tile.shape[0] < tile.shape[2]:
        # Already in (C, H, W) format
        C, H, W = tile.shape
    else:
        # Convert from (H, W, C) to (C, H, W)
        H, W, C = tile.shape
        tile = tile.permute(2, 0, 1)
    
    # Check if tile is all zeros or very close to it
    zero_percentage = (tile == 0).float().mean().item()
    if zero_percentage > (1 - min_valid_percentage):
        return False
    
    # Check if tile is all ones or very close to it
    one_percentage = (tile == 1).float().mean().item()
    if one_percentage > max_valid_percentage:
        return False
    
    # Check if there's enough variance in the data
    if torch.var(tile) < 1e-4:
        return False
    
    # Check for constant values across spectral dimension (C)
    # Reshape to (C, H*W) for easier spectral variance calculation
    reshaped = tile.reshape(C, -1)
    # Calculate variance along spectral dimension for each pixel
    spectral_variances = torch.var(reshaped, dim=0)
    # Count pixels with essentially zero spectral variance
    constant_spectral_pixels = (spectral_variances < spectral_variance_threshold).float().mean().item()
    
    # If too many pixels have constant spectral values, reject the tile
    if constant_spectral_pixels > max_constant_spectral_percentage:
        return False
    
    return True

def load_aviris_forest_data(base_path="AVIRIS_FOREST_SIMPLE_SELECT", tile_size=128, cache_file=None, use_cache=False, folder_patterns="all"):
    """
    Load pre-processed AVIRIS forest data from the simplified selection directory
    with improved tile validation.
    
    Parameters:
    base_path: Base path to the AVIRIS_FOREST_SIMPLE_SELECT directory
    tile_size: Size of the tiles (square)
    cache_file: Path to the cache file to save/load the processed data
    use_cache: If True, try to load from cache file first
    folder_patterns: Comma-separated list of folder name patterns to include, or 'all' for all folders
    
    Returns:
    torch.Tensor of shape [num_tiles, tile_size, tile_size, num_bands]
    """
    import torch
    import numpy as np
    import os
    from tqdm import tqdm
    
    # If use_cache is True and cache_file exists, try to load from cache
    if use_cache and cache_file and os.path.exists(cache_file):
        print(f"Loading processed data from cache: {cache_file}")
        try:
            tiles_tensor = torch.load(cache_file)
            
            # Ensure the tensor is in BHWC format
            if tiles_tensor.shape[1] == 100:  # BCHW format
                tiles_tensor = tiles_tensor.permute(0, 2, 3, 1)
                print(f"Loaded data from cache: {tiles_tensor.shape} (converted to BHWC format)")
            else:
                print(f"Loaded data from cache: {tiles_tensor.shape} (BHWC format)")
            
            return tiles_tensor
        except Exception as e:
            print(f"Error loading cache file: {str(e)}")
            print("Falling back to processing raw data.")
    
    # Validate base path
    if not os.path.exists(base_path) or not os.path.isdir(base_path):
        raise FileNotFoundError(f"AVIRIS directory not found at: {os.path.abspath(base_path)}")
    
    print(f"Looking for AVIRIS data in: {os.path.abspath(base_path)}")
    
    # Get list of all subfolders (flight lines)
    all_subfolders = [f for f in os.listdir(base_path) 
                    if os.path.isdir(os.path.join(base_path, f))]
    
    if not all_subfolders:
        raise FileNotFoundError(f"No subfolders found in {base_path}")
    
    # Filter subfolders based on patterns
    if folder_patterns.lower() == "all":
        subfolders = all_subfolders
    else:
        patterns = [p.strip() for p in folder_patterns.split(',')]
        subfolders = []
        for subfolder in all_subfolders:
            if any(pattern in subfolder for pattern in patterns):
                subfolders.append(subfolder)
    
    if not subfolders:
        raise FileNotFoundError(f"No matching subfolders found for patterns: {folder_patterns}")
    
    print(f"Found {len(subfolders)} matching subfolders: {', '.join(subfolders)}")
    
    # List to store all loaded data tensors
    all_data_list = []
    
    # Process each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_path, subfolder)
        print(f"\nProcessing subfolder: {subfolder}")
        
        # Check for torch data
        torch_path = os.path.join(subfolder_path, "torch", "aviris_selected.pt")
        
        if not os.path.exists(torch_path):
            print(f"No torch data found at {torch_path}, skipping...")
            continue
        
        # Load the metadata to get image dimensions
        metadata_path = os.path.join(subfolder_path, "torch", "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Loaded metadata: {metadata.get('image_shape', 'shape info not found')}")
            except Exception as e:
                print(f"Error loading metadata: {str(e)}")
                metadata = None
        else:
            print("No metadata found, will derive dimensions from the data")
            metadata = None
        
        # Load the data
        try:
            data = torch.load(torch_path)
            original_shape = data.shape
            print(f"Loaded data of shape: {original_shape}")
            
            # Check data format (BCHW or image with BHWC)
            if len(original_shape) == 3:  # Single image, HWC format
                # Convert to 4D tensor with batch dimension
                data = data.unsqueeze(0)
                print(f"Converted single image to batch: {data.shape}")
            
            # Normalize if needed
            if torch.max(data) > 1.0 or torch.min(data) < 0.0:
                data_min = torch.min(data)
                data_max = torch.max(data)
                data = (data - data_min) / (data_max - data_min)
                print(f"Normalized data to range [0, 1]. Min: {torch.min(data)}, Max: {torch.max(data)}")
            
            # Ensure correct format - convert to BHWC if in BCHW format
            if data.shape[1] < data.shape[2] and data.shape[1] < data.shape[3]:
                # Data is in BCHW format, convert to BHWC
                data = data.permute(0, 2, 3, 1)
                print(f"Converted data from BCHW to BHWC format: {data.shape}")
            
        except Exception as e:
            print(f"Error loading data from {torch_path}: {str(e)}")
            continue
        
        # Create tiles from the data
        tiles = []
        invalid_tiles = 0
        
        # Get dimensions
        batch_size, height, width, channels = data.shape
        
        # Process each image in the batch
        for b in range(batch_size):
            # Extract the image
            img = data[b]
            
            # Calculate how many tiles we can extract
            h_tiles = height // tile_size
            w_tiles = width // tile_size
            
            print(f"Creating {h_tiles}×{w_tiles} tiles of size {tile_size}×{tile_size} from image {b+1}/{batch_size}")
            
            # Extract tiles
            for i in range(h_tiles):
                for j in range(w_tiles):
                    h_start = i * tile_size
                    h_end = (i + 1) * tile_size
                    w_start = j * tile_size
                    w_end = (j + 1) * tile_size
                    
                    tile = img[h_start:h_end, w_start:w_end, :]
                    
                    # Validate the tile before adding it
                    if is_valid_tile(tile):
                        tiles.append(tile)
                    else:
                        invalid_tiles += 1
        
        # Report statistics
        print(f"Created {len(tiles)} valid tiles, rejected {invalid_tiles} invalid tiles")
        
        # Convert to tensor and add to list
        if tiles:
            subfolder_tensor = torch.stack(tiles)
            print(f"Created tensor of shape {subfolder_tensor.shape} from subfolder {subfolder}")
            all_data_list.append(subfolder_tensor)
    
    # Combine data from all subfolders
    if not all_data_list:
        raise FileNotFoundError("Could not load valid data from any subfolder")
    
    # If we have multiple data tensors, concatenate them
    if len(all_data_list) > 1:
        # Check if all tensors have the same number of spectral bands
        num_bands = all_data_list[0].shape[-1]
        all_same_bands = all(tensor.shape[-1] == num_bands for tensor in all_data_list)
        
        if not all_same_bands:
            print("Warning: Data from different subfolders have different numbers of spectral bands.")
            print("Using only the spectral bands from the first subfolder for consistency.")
            # Only use the first subfolder
            tiles_tensor = all_data_list[0]
        else:
            # Concatenate along the first dimension (batch)
            tiles_tensor = torch.cat(all_data_list, dim=0)
            print(f"Combined data from {len(all_data_list)} subfolders: {tiles_tensor.shape}")
    else:
        # Only one subfolder was processed
        tiles_tensor = all_data_list[0]
    
    # Shuffle the tiles to mix data from different subfolders
    indices = torch.randperm(tiles_tensor.shape[0])
    tiles_tensor = tiles_tensor[indices]
    
    print(f"Final tiles tensor shape: {tiles_tensor.shape} (BHWC format)")
    
    # If cache_file is provided, save the processed data
    if cache_file:
        print(f"Saving processed data to cache: {cache_file}")
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        try:
            # Save in BHWC format for consistency
            torch.save(tiles_tensor, cache_file)
            print(f"Data saved to cache successfully.")
        except Exception as e:
            print(f"Error saving to cache: {str(e)}")
    
    return tiles_tensor

###############################################################################
# MODIFIED AUTOENCODER MODEL WITH RANDOM NOISE
###############################################################################
class HyperspectralAutoencoderRandomNoise(nn.Module):
    def __init__(self, shape2filter_path, filter2shape_path, min_snr=10, max_snr=40, initial_filter_params=None):
        super().__init__()

        # Target SNR range (in dB) as model parameters
        self.min_snr = min_snr
        self.max_snr = max_snr
        
        # Device for model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the pretrained models
        self.shape2filter = Shape2FilterModel()
        self.shape2filter.load_state_dict(torch.load(shape2filter_path, map_location=self.device))
        self.shape2filter = self.shape2filter.to(self.device)
        
        self.filter2shape = Filter2ShapeVarLen()
        self.filter2shape.load_state_dict(torch.load(filter2shape_path, map_location=self.device))
        self.filter2shape = self.filter2shape.to(self.device)
        
        # Freeze both models
        for param in self.shape2filter.parameters():
            param.requires_grad = False
            
        for param in self.filter2shape.parameters():
            param.requires_grad = False
        
        # Create the filter2shape2filter pipeline
        self.pipeline = Filter2Shape2FilterFrozen(self.filter2shape, self.shape2filter, no_grad_frozen=False)
        
        # Initialize learnable filter parameters (11 x 100)
        # Use provided initial parameters if available, otherwise generate new ones
        if initial_filter_params is not None:
            self.filter_params = nn.Parameter(initial_filter_params.clone().to(self.device))
        else:
            # Default to random initialization
            self.filter_params = nn.Parameter(torch.rand(11, 100, device=self.device))
        
        # Decoder: 3-layer convolutional network
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(11, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 100, kernel_size=3, padding=1)
        # )

        self.decoder = AWAN(inplanes=latent_dim, planes=in_channels, channels=128, n_DRBs=2)
    
    def get_current_filter(self):
        """Return the current learnable filter parameters"""
        return self.filter_params
    
    def get_current_shape(self):
        """Get the current shape from filter2shape model"""
        filter = self.get_current_filter().unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            shape = self.filter2shape(filter)[0]  # Remove batch dimension
        return shape
    
    def get_reconstructed_filter(self):
        """Get the reconstructed filter from the full pipeline"""
        filter = self.get_current_filter().unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            _, recon_filter = self.pipeline(filter)
        return recon_filter[0]  # Remove batch dimension

    def get_reconstructed_filter_with_grad(self):
        """Get the reconstructed filter from the full pipeline"""
        filter = self.get_current_filter().unsqueeze(0)  # Add batch dimension
        # with torch.no_grad():
        _, recon_filter = self.pipeline(filter)
        return recon_filter[0]  # Remove batch dimension

    def add_random_noise(self, tensor):
        """
        Add Gaussian white noise to tensor, with noise level randomly chosen
        between self.min_snr and self.max_snr for each batch.
        """
        # Choose a random noise level within the specified range
        target_snr = random.uniform(self.min_snr, self.max_snr)
            
        # Calculate signal power (mean square), using detach() to separate calculation graph
        signal_power = tensor.detach().pow(2).mean()
        # Convert signal power to decibels
        signal_power_db = 10 * torch.log10(signal_power)
        # Calculate noise power (decibels)
        noise_power_db = signal_power_db - target_snr
        # Convert noise power back to linear scale
        noise_power = 10 ** (noise_power_db / 10)
        # Generate Gaussian white noise with the same shape as input tensor
        noise = torch.randn_like(tensor) * torch.sqrt(noise_power)
        # Return tensor with added noise
        return tensor + noise, target_snr
    
    def forward(self, x):
        """
        Forward pass of the autoencoder with random noise level
        Input: x - Hyperspectral data of shape [batch_size, height, width, 100]
                or shape [batch_size, 100, height, width]
        """
        # Ensure input is in BHWC format
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D tensor, got shape {x.shape}")
            
        if x.shape[1] == 100 and x.shape[3] != 100:  # Input is in BCHW format
            x = x.permute(0, 2, 3, 1)  # Convert to BHWC
            
        # Get dimensions
        batch_size, height, width, spectral_bands = x.shape
        
        # Get reconstructed filter (important fix)
        filter = self.get_reconstructed_filter_with_grad()  # Shape: [11, 100]
        
        # Convert input from BHWC to BCHW format for PyTorch convolution
        x_channels_first = x.permute(0, 3, 1, 2)
        
        # Normalize filter for filtering
        filter_normalized = filter / 10.0  # Shape: [11, 100]
        
        # Use efficient tensor operations for spectral filtering
        # Einstein summation: 'bchw,oc->bohw'
        # This performs the weighted sum across spectral dimension for each output band
        encoded_channels_first = torch.einsum('bchw,oc->bohw', x_channels_first, filter_normalized)

        # ----------------- Add random noise ------------------
        encoded_channels_first, applied_snr = self.add_random_noise(encoded_channels_first)
        
        # Convert encoded data back to channels-last format [B,H,W,C]
        encoded = encoded_channels_first.permute(0, 2, 3, 1)
        
        # Decode: use the CNN decoder to expand from 11 to 100 bands
        decoded_channels_first = self.decoder(encoded_channels_first)
        
        # Convert back to original format [B,H,W,C]
        decoded = decoded_channels_first.permute(0, 2, 3, 1)
        
        return decoded, encoded, applied_snr

###############################################################################
# MODIFIED TRAINING FUNCTION WITH RANDOM NOISE
###############################################################################
def train_with_random_noise(shape2filter_path, filter2shape_path, output_dir, min_snr=10, max_snr=40, 
                           initial_filter_params=None, batch_size=10, num_epochs=500, learning_rate=0.001, 
                           cache_file=None, use_cache=False, folder_patterns="all",
                           train_data=None, test_data=None):
    """
    Train and visualize the hyperspectral autoencoder with random noise levels between min_snr and max_snr
    using filter2shape2filter architecture
    
    Parameters:
    shape2filter_path: Path to the pretrained shape2filter model
    filter2shape_path: Path to the pretrained filter2shape model
    output_dir: Directory to save outputs
    min_snr: Minimum SNR level in dB for random noise
    max_snr: Maximum SNR level in dB for random noise
    initial_filter_params: Initial filter parameters (11x100)
    batch_size: Batch size for training
    num_epochs: Number of training epochs
    learning_rate: Learning rate for optimizer
    cache_file: Path to cache file for storing processed data
    use_cache: If True, try to load from cache file first
    folder_patterns: Comma-separated list of folder name patterns to include, or 'all' for all folders
    train_data: Optional training data, if None will be loaded
    test_data: Optional testing data, if None will be loaded
    
    Returns:
    tuple: (initial_shape_path, least_mse_shape_path, lowest_cn_shape_path, final_shape_path, output_dir)
    """
    # Create output directory with noise range
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    noise_dir = f"blind_noise_{min_snr}dB_to_{max_snr}dB"
    output_dir = os.path.join(output_dir, noise_dir + f"_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}/")
    
    # Create subfolder for intermediate visualizations
    viz_dir = os.path.join(output_dir, "intermediate_viz")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load or use provided data
    if train_data is None:
        data = load_aviris_forest_data(base_path="AVIRIS_FOREST_SIMPLE_SELECT", tile_size=128, 
                                      cache_file=cache_file, use_cache=use_cache, folder_patterns=folder_patterns)
        data = data.to(device)
    else:
        data = train_data.to(device)
    
    print("Training data shape:", data.shape)
    
    # Create dataset and dataloader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size//2, shuffle=True)
    
    # If no initial filter parameters were provided, generate them
    if initial_filter_params is None:
        initial_filter_params = generate_initial_filter(device)
    
    # Initialize model with random noise level range and initial filter parameters
    model = HyperspectralAutoencoderRandomNoise(
        shape2filter_path, filter2shape_path, 
        min_snr=min_snr, max_snr=max_snr, 
        initial_filter_params=initial_filter_params
    ).to(device)
    
    print(f"Model initialized with random noise range: {min_snr} to {max_snr} dB and shared initial filter parameters")
    
    # Get initial filter, shape, and filters for visualization
    initial_filter = model.get_current_filter().detach().cpu().numpy()
    initial_shape = model.get_current_shape().detach().cpu().numpy()
    
    # Calculate initial condition number
    initial_condition_number = calculate_condition_number(model.get_reconstructed_filter().detach().cpu())
    print(f"Initial condition number: {initial_condition_number:.4f}")
    
    # Save initial shape with C4 symmetry
    initial_shape_path = f"{output_dir}/initial_shape.png"
    plot_shape_with_c4(initial_shape, f"Initial Shape", initial_shape_path)
    print(f"Initial shape saved to: {os.path.abspath(initial_shape_path)}")
    
    # Save initial filter
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(initial_filter[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Initial Spectral Parameters (SNR: {min_snr}-{max_snr} dB)")
    plt.legend()
    initial_filter_path = f"{output_dir}/initial_filter.png"
    plt.savefig(initial_filter_path)
    plt.close()
    
    # Also save the reconstructed filter from the pipeline
    initial_recon_filter = model.get_reconstructed_filter().detach().cpu().numpy()
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(initial_recon_filter[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Initial Reconstructed Filter (SNR: {min_snr}-{max_snr} dB)")
    plt.legend()
    initial_recon_filter_path = f"{output_dir}/initial_recon_filter.png"
    plt.savefig(initial_recon_filter_path)
    plt.close()
    
    # Before training, get reconstruction of sample from train set
    sample_idx = min(5, len(data) - 1)  # Make sure index is within range
    sample_tensor = data[sample_idx:sample_idx+1]  # Keep as tensor with batch dimension
    
    # If test data is provided, also get test sample
    test_sample_tensor = None
    if test_data is not None:
        test_sample_idx = min(5, len(test_data) - 1)
        test_sample_tensor = test_data[test_sample_idx:test_sample_idx+1].to(device)
    
    with torch.no_grad():
        initial_recon, encoded, applied_snr = model(sample_tensor)
        # Convert to numpy for visualization
        initial_recon_np = initial_recon.detach().cpu().numpy()[0]
        encoded_np = encoded.detach().cpu().numpy()[0]
        sample_np = sample_tensor.detach().cpu().numpy()[0]
        
        # Calculate initial MSE for train
        initial_train_mse = ((initial_recon - sample_tensor) ** 2).mean().item()
        print(f"Initial Train MSE: {initial_train_mse:.6f} (SNR: {applied_snr:.2f} dB)")
        
        # Calculate initial MSE for test if available
        initial_test_mse = None
        if test_sample_tensor is not None:
            test_recon, _, test_applied_snr = model(test_sample_tensor)
            initial_test_mse = ((test_recon - test_sample_tensor) ** 2).mean().item()
            print(f"Initial Test MSE: {initial_test_mse:.6f} (SNR: {test_applied_snr:.2f} dB)")
    
    # Training loop
    losses = []
    condition_numbers = [initial_condition_number]
    train_mse_values = [initial_train_mse]
    test_mse_values = [initial_test_mse] if initial_test_mse is not None else []
    applied_snr_values = []  # Track SNR values applied
    
    # Add variables to track the lowest condition number and corresponding shape
    lowest_condition_number = float('inf')
    lowest_cn_shape = None
    lowest_cn_filter = None
    lowest_cn_epoch = -1
    
    # Add variables to track the lowest MSE and corresponding shape
    lowest_train_mse = float('inf')
    lowest_mse_shape = None
    lowest_mse_filter = None
    lowest_mse_epoch = -1
    
    print(f"Starting training with random noise SNR range: {min_snr} to {max_snr} dB...")
    
    # Create optimizer for just the filter parameters
    optimizer = optim.Adam([model.filter_params], lr=learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        epoch_snr_values = []
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = batch[0]
            
            # Forward pass with random noise
            recon, _, batch_snr = model(x)
            epoch_snr_values.append(batch_snr)
            
            # Calculate loss
            loss = criterion(recon, x)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss and SNR for epoch
        avg_loss = epoch_loss / num_batches
        avg_snr = sum(epoch_snr_values) / len(epoch_snr_values) if epoch_snr_values else 0
        applied_snr_values.append(avg_snr)
        
        losses.append(avg_loss)
        
        # Calculate condition number for current epoch
        current_filter = model.get_reconstructed_filter().detach().cpu()
        current_condition_number = calculate_condition_number(current_filter)
        condition_numbers.append(current_condition_number)
        
        # Get current shape for visualization
        current_shape = model.get_current_shape().detach().cpu().numpy()
        current_filter_raw = model.get_current_filter().detach().cpu()
        
        # Check if this is the lowest condition number so far
        if current_condition_number < lowest_condition_number:
            lowest_condition_number = current_condition_number
            lowest_cn_shape = current_shape.copy()
            lowest_cn_filter = current_filter.clone()
            lowest_cn_epoch = epoch
            print(f"New lowest condition number: {lowest_condition_number:.4f} at epoch {epoch+1}")
        
        # Calculate current MSE for train
        with torch.no_grad():
            current_train_recon, _, train_snr = model(sample_tensor)
            current_train_mse = ((current_train_recon - sample_tensor) ** 2).mean().item()
            train_mse_values.append(current_train_mse)
            
            # Check if this is the lowest MSE so far
            if current_train_mse < lowest_train_mse:
                lowest_train_mse = current_train_mse
                lowest_mse_shape = current_shape.copy()
                lowest_mse_filter = current_filter_raw.clone()
                lowest_mse_epoch = epoch
                print(f"New lowest train MSE: {lowest_train_mse:.6f} at epoch {epoch+1}")
            
            # Calculate current MSE for test if available
            if test_sample_tensor is not None:
                current_test_recon, _, test_snr = model(test_sample_tensor)
                current_test_mse = ((current_test_recon - test_sample_tensor) ** 2).mean().item()
                test_mse_values.append(current_test_mse)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Avg SNR: {avg_snr:.2f} dB, Condition Number: {current_condition_number:.4f}, "
                      f"Train MSE: {current_train_mse:.6f}, Test MSE: {current_test_mse:.6f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Avg SNR: {avg_snr:.2f} dB, Condition Number: {current_condition_number:.4f}, "
                      f"Train MSE: {current_train_mse:.6f}")
        
        # Save intermediate shapes and visualizations
        if (epoch+1) % (num_epochs // 10) == 0 or epoch == 0:
            # Save shape visualization to intermediate directory
            plot_shape_with_c4(
                current_shape, 
                f"Shape at Epoch {epoch+1}", 
                f"{viz_dir}/shape_epoch_{epoch+1}.png"
            )
            
            # Save filter visualization
            plt.figure(figsize=(10, 6))
            for i in range(11):
                plt.plot(current_filter.numpy()[i], label=f'Filter {i}' if i % 3 == 0 else None)
            plt.grid(True)
            plt.xlabel("Wavelength Index")
            plt.ylabel("Filter Value")
            plt.title(f"Filter at Epoch {epoch+1}")
            plt.legend()
            plt.savefig(f"{viz_dir}/filter_epoch_{epoch+1}.png")
            plt.close()
    
    # Get final filter and shape
    final_filter = model.get_current_filter().detach().cpu()
    final_shape = model.get_current_shape().detach().cpu().numpy()
    
    # Calculate final condition number
    final_condition_number = calculate_condition_number(model.get_reconstructed_filter().detach().cpu())
    print(f"Final condition number: {final_condition_number:.4f}")
    
    # Save final shape
    final_shape_path = f"{output_dir}/final_shape.png"
    plot_shape_with_c4(final_shape, f"Final Shape", final_shape_path)
    print(f"Final shape saved to: {os.path.abspath(final_shape_path)}")
    
    # Save final filter
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(final_filter.numpy()[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Final Spectral Parameters (SNR: {min_snr}-{max_snr} dB)")
    plt.legend()
    final_filter_path = f"{output_dir}/final_filter.png"
    plt.savefig(final_filter_path)
    plt.close()
    
    # Also save the reconstructed filter from the pipeline
    final_recon_filter = model.get_reconstructed_filter().detach().cpu().numpy()
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(final_recon_filter[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Final Reconstructed Filter (SNR: {min_snr}-{max_snr} dB)")
    plt.legend()
    final_recon_filter_path = f"{output_dir}/final_recon_filter.png"
    plt.savefig(final_recon_filter_path)
    plt.close()
    
    # After training, get reconstruction of the same sample
    with torch.no_grad():
        final_train_recon, _, final_train_snr = model(sample_tensor)
        final_train_mse = ((final_train_recon - sample_tensor) ** 2).mean().item()
        print(f"Final Train MSE: {final_train_mse:.6f} (SNR: {final_train_snr:.2f} dB)")
        
        final_test_mse = None
        if test_sample_tensor is not None:
            final_test_recon, _, final_test_snr = model(test_sample_tensor)
            final_test_mse = ((final_test_recon - test_sample_tensor) ** 2).mean().item()
            print(f"Final Test MSE: {final_test_mse:.6f} (SNR: {final_test_snr:.2f} dB)")
    
    # Create a directory for the lowest condition number results
    lowest_cn_dir = os.path.join(output_dir, "lowest_cn")
    os.makedirs(lowest_cn_dir, exist_ok=True)
    
    # Save the lowest condition number shape
    lowest_cn_shape_path = f"{lowest_cn_dir}/shape.png"
    plot_shape_with_c4(lowest_cn_shape, f"Shape with Lowest CN: {lowest_condition_number:.4f}", lowest_cn_shape_path)
    
    # Save the lowest condition number filter
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(lowest_cn_filter.numpy()[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Spectral Parameters with Lowest Condition Number: {lowest_condition_number:.4f}")
    plt.legend()
    lowest_cn_filter_path = f"{lowest_cn_dir}/filter.png"
    plt.savefig(lowest_cn_filter_path)
    plt.close()
    
    # Create a directory for the lowest MSE results
    lowest_mse_dir = os.path.join(output_dir, "lowest_mse")
    os.makedirs(lowest_mse_dir, exist_ok=True)
    
    # Save the lowest MSE shape
    lowest_mse_shape_path = f"{lowest_mse_dir}/shape.png"
    plot_shape_with_c4(lowest_mse_shape, f"Shape with Lowest MSE: {lowest_train_mse:.6f}", lowest_mse_shape_path)
    
    # Save the lowest MSE filter
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(lowest_mse_filter.numpy()[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(f"Spectral Parameters with Lowest MSE: {lowest_train_mse:.6f}")
    plt.legend()
    lowest_mse_filter_path = f"{lowest_mse_dir}/filter.png"
    plt.savefig(lowest_mse_filter_path)
    plt.close()
    
    # Save the lowest condition number shape as numpy file
    lowest_cn_shape_npy_path = f"{lowest_cn_dir}/shape.npy"
    np.save(lowest_cn_shape_npy_path, lowest_cn_shape)
    
    # Save the lowest condition number filter as numpy file
    lowest_cn_filter_npy_path = f"{lowest_cn_dir}/filter.npy"
    np.save(lowest_cn_filter_npy_path, lowest_cn_filter.numpy())
    
    # Save the lowest MSE shape as numpy file
    lowest_mse_shape_npy_path = f"{lowest_mse_dir}/shape.npy"
    np.save(lowest_mse_shape_npy_path, lowest_mse_shape)
    
    # Save the lowest MSE filter as numpy file
    lowest_mse_filter_npy_path = f"{lowest_mse_dir}/filter.npy"
    np.save(lowest_mse_filter_npy_path, lowest_mse_filter.numpy())
    
    # Plot condition number during training
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs+1), condition_numbers, 'r-')
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Condition Number")
    plt.title(f"Filter Matrix Condition Number During Training (SNR: {min_snr}-{max_snr} dB)")
    plt.axhline(y=lowest_condition_number, color='g', linestyle='--', 
                label=f'Lowest CN: {lowest_condition_number:.4f} (Epoch {lowest_cn_epoch+1})')
    plt.legend()
    condition_plot_path = f"{output_dir}/condition_number.png"
    plt.savefig(condition_plot_path)
    plt.close()
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Loss (SNR: {min_snr}-{max_snr} dB)")
    loss_plot_path = f"{output_dir}/training_loss.png"
    plt.savefig(loss_plot_path)
    plt.close()
    
    # Plot applied SNR values during training
    plt.figure(figsize=(10, 5))
    plt.plot(applied_snr_values)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Average SNR (dB)")
    plt.title(f"Average Applied SNR During Training (Range: {min_snr}-{max_snr} dB)")
    plt.ylim(min_snr-1, max_snr+1)
    snr_plot_path = f"{output_dir}/applied_snr.png"
    plt.savefig(snr_plot_path)
    plt.close()
    
    # Plot MSE values
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs+1), train_mse_values, 'b-', label='Train MSE')
    plt.axhline(y=lowest_train_mse, color='c', linestyle='--', 
              label=f'Lowest MSE: {lowest_train_mse:.6f} (Epoch {lowest_mse_epoch+1})')
    if test_mse_values:
        plt.plot(range(num_epochs+1), test_mse_values, 'r--', label='Test MSE')
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"MSE of Reconstruction and Input (SNR: {min_snr}-{max_snr} dB)")
    plt.legend()
    mse_plot_path = f"{output_dir}/mse_values.png"
    plt.savefig(mse_plot_path)
    plt.close()
    
    # Save model parameters
    model_save_path = f"{output_dir}/model_state.pt"
    torch.save({
        'filter_params': model.filter_params.detach().cpu(),
        'decoder_state_dict': model.decoder.state_dict()
    }, model_save_path)
    
    # Also save numerical data
    np.save(f"{output_dir}/initial_shape.npy", initial_shape)
    np.save(f"{output_dir}/final_shape.npy", final_shape)
    np.save(f"{output_dir}/initial_filter.npy", initial_filter)
    np.save(f"{output_dir}/final_filter.npy", final_filter.numpy())
    np.save(f"{output_dir}/condition_numbers.npy", np.array(condition_numbers))
    np.save(f"{output_dir}/losses.npy", np.array(losses))
    np.save(f"{output_dir}/train_mse_values.npy", np.array(train_mse_values))
    np.save(f"{output_dir}/applied_snr_values.npy", np.array(applied_snr_values))
    if test_mse_values:
        np.save(f"{output_dir}/test_mse_values.npy", np.array(test_mse_values))
    
    # Create log file to save training parameters
    with open(f"{output_dir}/training_params.txt", "w") as f:
        f.write(f"Noise level range (SNR): {min_snr} to {max_snr} dB\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Cache file: {cache_file}\n")
        f.write(f"Used cache: {use_cache}\n")
        f.write(f"Folder patterns: {folder_patterns}\n")
        f.write("\n")
        
        # Save condition number information
        f.write(f"Initial condition number: {initial_condition_number:.4f}\n")
        f.write(f"Final condition number: {final_condition_number:.4f}\n")
        f.write(f"Lowest condition number: {lowest_condition_number:.4f} at epoch {lowest_cn_epoch+1}\n")
        f.write(f"Condition number change: {final_condition_number - initial_condition_number:.4f}\n\n")
        
        # Save MSE information
        f.write(f"Initial Train MSE: {initial_train_mse:.6f}\n")
        f.write(f"Final Train MSE: {final_train_mse:.6f}\n")
        f.write(f"Lowest Train MSE: {lowest_train_mse:.6f} at epoch {lowest_mse_epoch+1}\n")
        f.write(f"Train MSE improvement: {initial_train_mse - final_train_mse:.6f} ({(1 - final_train_mse/initial_train_mse) * 100:.2f}%)\n")
        
        if initial_test_mse is not None and final_test_mse is not None:
            f.write(f"Initial Test MSE: {initial_test_mse:.6f}\n")
            f.write(f"Final Test MSE: {final_test_mse:.6f}\n")
            f.write(f"Test MSE improvement: {initial_test_mse - final_test_mse:.6f} ({(1 - final_test_mse/initial_test_mse) * 100:.2f}%)\n")
    
    print(f"Training with random noise SNR range {min_snr} to {max_snr} dB completed. All results saved to {output_dir}/")
    
    # Return paths to all shapes for later training
    initial_shape_npy_path = f"{output_dir}/initial_shape.npy"
    final_shape_npy_path = f"{output_dir}/final_shape.npy"
    lowest_cn_shape_npy_path = f"{lowest_cn_dir}/shape.npy"
    lowest_mse_shape_npy_path = f"{lowest_mse_dir}/shape.npy"
    
    return initial_shape_npy_path, lowest_mse_shape_npy_path, lowest_cn_shape_npy_path, final_shape_npy_path, output_dir

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run hyperspectral autoencoder with blind noise experiment.")
    parser.add_argument("--cache", type=str, default="cache_filtered/aviris_tiles_forest.pt", 
                       help="Path to cache file for processed data")
    parser.add_argument("--use-cache", action="store_true", help="Use cached data if available")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (e.g., 0, 1, 2, 3)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for full training")
    parser.add_argument("--decoder-epochs", type=int, default=20, help="Number of epochs for decoder-only training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("-f", "--folders", type=str, default="all", 
                       help="Comma-separated list of folder name patterns to include, or 'all' for all folders")
    parser.add_argument("--min-snr", type=float, default=10, help="Minimum SNR level in dB for random noise")
    parser.add_argument("--max-snr", type=float, default=40, help="Maximum SNR level in dB for random noise")
    
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")
    
    # Modify cache filename based on folders
    folder_patterns = args.folders
    if folder_patterns.lower() == "all":
        folder_suffix = "all"
    else:
        folder_suffix = folder_patterns.replace(",", "_")
    
    # Update the cache path to include folder suffix
    cache_path = args.cache
    if ".pt" in cache_path:
        cache_path = cache_path.replace(".pt", f"_{folder_suffix}.pt")
    else:
        cache_path = f"{cache_path}_{folder_suffix}.pt"
    
    print(f"Using cache path: {cache_path}")
    
    # Check if the model paths exist
    shape2filter_path = "outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt"
    filter2shape_path = "outputs_three_stage_20250322_145925/stageC/spec2shape_stageC.pt"
    
    for model_path in [shape2filter_path, filter2shape_path]:
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            print("Searching for model file in current directory...")
            
            # Try to find the model file in the current directory structure
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.pt'):
                        if "shape2spec" in file and model_path == shape2filter_path:
                            shape2filter_path = os.path.join(root, file)
                            print(f"Using shape2filter model file: {shape2filter_path}")
                        elif "spec2shape" in file and model_path == filter2shape_path:
                            filter2shape_path = os.path.join(root, file)
                            print(f"Using filter2shape model file: {filter2shape_path}")
    
    # Create base output directory with folder patterns
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"blind_noise_forest_experiment_{folder_suffix}_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create cache directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    
    # Set device for generating initial parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate a unified initial filter to use across all experiments
    initial_filter_params = generate_initial_filter(device)
    
    # Save the initial filter for reference
    np.save(f"{base_output_dir}/unified_initial_filter.npy", initial_filter_params.detach().cpu().numpy())
    
    # Load and split data first
    print("Loading and splitting data into train/test sets...")
    data = load_aviris_forest_data(base_path="AVIRIS_FOREST_SIMPLE_SELECT", tile_size=128, 
                                 cache_file=cache_path, use_cache=args.use_cache, folder_patterns=folder_patterns)
    
    # Verify data is in BHWC format
    if data.shape[1] == 100:  # If in BCHW format
        data = data.permute(0, 2, 3, 1)  # Convert to BHWC
        print(f"Converted data to BHWC format: {data.shape}")
    
    # Split data into training and testing sets (80% train, 20% test)
    num_samples = data.shape[0]
    indices = torch.randperm(num_samples)
    train_size = int(0.8 * num_samples)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_data = data[train_indices]
    test_data = data[test_indices]
    
    print(f"Data split into {train_data.shape[0]} training (shape: {train_data.shape}) and {test_data.shape[0]} testing samples")
    
    # Define the range for random noise during blind training from command line arguments
    min_snr = args.min_snr
    max_snr = args.max_snr
    print(f"Using random noise range: {min_snr} to {max_snr} dB")
    
    # STAGE 1: Train with random noise to get shapes
    print(f"\n{'='*50}")
    print(f"STAGE 1: Training with random noise in range {min_snr} to {max_snr} dB")
    print(f"{'='*50}\n")
    
    # Train with random noise to get shapes
    initial_shape_path, lowest_mse_shape_path, lowest_cn_shape_path, final_shape_path, blind_output_dir = train_with_random_noise(
        shape2filter_path=shape2filter_path,
        filter2shape_path=filter2shape_path,
        output_dir=base_output_dir,
        min_snr=min_snr,
        max_snr=max_snr,
        initial_filter_params=initial_filter_params,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        cache_file=cache_path,
        use_cache=args.use_cache,
        folder_patterns=folder_patterns,
        train_data=train_data,
        test_data=test_data
    )
    
    # Store paths from random noise training
    shapes_dict = {
        'initial': initial_shape_path,
        'lowest_mse': lowest_mse_shape_path,
        'lowest_cn': lowest_cn_shape_path,
        'final': final_shape_path
    }
    
    # STAGE 2: Test the shapes from blind training with fixed noise levels
    # Define fixed noise levels to test (in dB)
    fixed_noise_levels = [10, 20, 30, 40]  # Simplified set for faster training
    
    # Create directory for decoder-only results
    decoder_output_dir = os.path.join(base_output_dir, "decoder_comparison_results")
    os.makedirs(decoder_output_dir, exist_ok=True)
    
    # Dictionary to store MSE results for each noise level
    mse_results = {}
    
    # Run decoder training with comparison for each fixed noise level
    for noise_level in fixed_noise_levels:
        print(f"\n{'='*50}")
        print(f"STAGE 2: Testing shapes with fixed noise level: {noise_level} dB")
        print(f"{'='*50}\n")
        
        # Import train_decoder_with_comparison for consistency
        from noise_experiment_with_filter2shape2filter import train_decoder_with_comparison
        
        mse_results[noise_level] = train_decoder_with_comparison(
            shape2filter_path=shape2filter_path,
            filter2shape_path=filter2shape_path,
            shapes_dict=shapes_dict,
            output_dir=decoder_output_dir,
            noise_level=noise_level,
            num_epochs=args.decoder_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            cache_file=cache_path,
            use_cache=args.use_cache,
            folder_patterns=folder_patterns,
            train_data=train_data,
            test_data=test_data
        )
    
    # Collect results for all noise levels - both final and best MSE
    final_train_results = {shape_type: [] for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']}
    final_test_results = {shape_type: [] for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']}
    best_train_results = {shape_type: [] for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']}
    best_test_results = {shape_type: [] for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']}
    
    for noise_level in fixed_noise_levels:
        for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']:
            # Check if shape type exists in the results
            if shape_type in mse_results[noise_level]['final_mse']:
                final_train_results[shape_type].append(mse_results[noise_level]['final_mse'][shape_type]['train'])
                final_test_results[shape_type].append(mse_results[noise_level]['final_mse'][shape_type]['test'])
                best_train_results[shape_type].append(mse_results[noise_level]['best_mse'][shape_type]['train'])
                best_test_results[shape_type].append(mse_results[noise_level]['best_mse'][shape_type]['test'])
    
    # Create summary plots directory
    summary_dir = os.path.join(base_output_dir, "summary_plots")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Plot final MSE comparison across fixed noise levels
    plt.figure(figsize=(12, 10))
    
    # Final Training MSE
    plt.subplot(2, 1, 1)
    for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']:
        if shape_type in final_train_results and len(final_train_results[shape_type]) == len(fixed_noise_levels):
            plt.plot(fixed_noise_levels, final_train_results[shape_type], 'o-', 
                    label=f"{shape_type.capitalize()} Shape")
    plt.grid(True)
    plt.xlabel("Noise Level (SNR in dB)")
    plt.ylabel("MSE after Training")
    plt.title("Final Training MSE Comparison Across Fixed Noise Levels")
    plt.legend()
    plt.gca().invert_xaxis()  # Higher SNR means less noise
    
    # Final Testing MSE
    plt.subplot(2, 1, 2)
    for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']:
        if shape_type in final_test_results and len(final_test_results[shape_type]) == len(fixed_noise_levels):
            plt.plot(fixed_noise_levels, final_test_results[shape_type], 'o-', 
                    label=f"{shape_type.capitalize()} Shape")
    plt.grid(True)
    plt.xlabel("Noise Level (SNR in dB)")
    plt.ylabel("MSE after Training")
    plt.title("Final Testing MSE Comparison Across Fixed Noise Levels")
    plt.legend()
    plt.gca().invert_xaxis()  # Higher SNR means less noise
    
    plt.tight_layout()
    final_comparison_path = os.path.join(summary_dir, "fixed_noise_level_final_mse_comparison.png")
    plt.savefig(final_comparison_path, dpi=300)
    plt.close()
    
    # Plot best MSE comparison across fixed noise levels
    plt.figure(figsize=(12, 10))
    
    # Best Training MSE
    plt.subplot(2, 1, 1)
    for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']:
        if shape_type in best_train_results and len(best_train_results[shape_type]) == len(fixed_noise_levels):
            plt.plot(fixed_noise_levels, best_train_results[shape_type], 'o-', 
                    label=f"{shape_type.capitalize()} Shape")
    plt.grid(True)
    plt.xlabel("Noise Level (SNR in dB)")
    plt.ylabel("Best MSE During Training")
    plt.title("Best Training MSE Comparison Across Fixed Noise Levels")
    plt.legend()
    plt.gca().invert_xaxis()  # Higher SNR means less noise
    
    # Best Testing MSE
    plt.subplot(2, 1, 2)
    for shape_type in ['initial', 'lowest_mse', 'lowest_cn', 'final']:
        if shape_type in best_test_results and len(best_test_results[shape_type]) == len(fixed_noise_levels):
            plt.plot(fixed_noise_levels, best_test_results[shape_type], 'o-', 
                    label=f"{shape_type.capitalize()} Shape")
    plt.grid(True)
    plt.xlabel("Noise Level (SNR in dB)")
    plt.ylabel("Best MSE During Training")
    plt.title("Best Testing MSE Comparison Across Fixed Noise Levels")
    plt.legend()
    plt.gca().invert_xaxis()  # Higher SNR means less noise
    
    plt.tight_layout()
    best_comparison_path = os.path.join(summary_dir, "fixed_noise_level_best_mse_comparison.png")
    plt.savefig(best_comparison_path, dpi=300)
    plt.close()
    
    # Save the numerical results
    np.savez(os.path.join(summary_dir, "blind_fixed_noise_mse_results.npz"),
             fixed_noise_levels=fixed_noise_levels,
             final_train_results=final_train_results,
             final_test_results=final_test_results,
             best_train_results=best_train_results,
             best_test_results=best_test_results)
    
    print(f"\nAll experiments completed! Results saved to: {os.path.abspath(base_output_dir)}")

if __name__ == "__main__":
    main()