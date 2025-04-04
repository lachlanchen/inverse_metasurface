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
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from tqdm import tqdm
import math
import random
import json
from matplotlib.colors import Normalize
from matplotlib import cm

# Import common functions and classes from the original script
from noise_experiment_with_blind_noise import (
    load_aviris_forest_data, is_valid_tile,
    Shape2FilterModel, Filter2ShapeVarLen, Filter2Shape2FilterFrozen,
    calculate_condition_number, replicate_c4, sort_points_by_angle, plot_shape_with_c4,
    generate_initial_filter, DecoderCNN5Layer
)

from noise_experiment_with_blind_noise_and_recon_and_noise_level_comparison_base import (
    calculate_psnr,
    calculate_sam,
    visualize_reconstruction,
    visualize_reconstruction_spectrum,
    HyperspectralAutoencoderRandomNoise,
    FixedShapeModel
)

from AWAN import AWAN
latent_dim = 11
in_channels = 100


###############################################################################
# LAZY DATA LOADING CLASSES
###############################################################################
class LazyAVIRISDataset(Dataset):
    """Dataset that lazily loads AVIRIS data from disk to avoid memory issues."""
    
    def __init__(self, data_path, indices=None, transform=None):
        """
        Initialize the lazy dataset loader.
        
        Args:
            data_path: Path to the numpy array or PyTorch tensor file
            indices: Optional indices to select only specific samples
            transform: Optional transform to apply to the data
        """
        self.data_path = data_path
        self.transform = transform
        self.indices = indices
        
        # Get dataset info without loading all data
        if data_path.endswith('.npy'):
            # Memory map the file to get shape without loading into memory
            with np.load(data_path, mmap_mode='r') as data:
                self.shape = data.shape
                self.length = data.shape[0] if indices is None else len(indices)
        elif data_path.endswith('.pt'):
            # For PyTorch tensor we need to load metadata
            info = torch.load(data_path, map_location='cpu', weights_only=True)
            if isinstance(info, dict) and 'shape' in info:
                # If we saved shape info separately
                self.shape = info['shape']
            else:
                # We need to load the tensor to get its shape
                data = torch.load(data_path, map_location='cpu')
                self.shape = data.shape
                del data  # Free memory
            self.length = self.shape[0] if indices is None else len(indices)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
            
        print(f"Initialized LazyAVIRISDataset with {self.length} samples of shape {self.shape[1:]}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]
            
        # Load only the specific sample
        if self.data_path.endswith('.npy'):
            # Load using numpy memory mapping
            with np.load(self.data_path, mmap_mode='r') as data:
                sample = data[idx].copy()  # Copy to make it writeable
                sample = torch.from_numpy(sample)
        elif self.data_path.endswith('.pt'):
            # For PyTorch we need a custom approach
            data = torch.load(self.data_path, map_location='cpu')
            sample = data[idx]
            del data  # Free memory
            
        # Apply transforms if any
        if self.transform:
            sample = self.transform(sample)
            
        return sample


def create_lazy_data_split(data_path, train_ratio=0.7, batch_size=32, seed=42):
    """
    Create train and test data loaders from a large dataset file without loading all data at once.
    
    Args:
        data_path: Path to the data file
        train_ratio: Ratio of data to use for training
        batch_size: Batch size for DataLoaders
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, test_loader, train_dataset, test_dataset)
    """
    # First, get the size of the dataset
    if data_path.endswith('.npy'):
        # Use memory mapping to check size without loading
        with np.load(data_path, mmap_mode='r') as data:
            num_samples = data.shape[0]
    elif data_path.endswith('.pt'):
        # Quick load to get shape
        info = torch.load(data_path, map_location='cpu', weights_only=True)
        if isinstance(info, dict) and 'shape' in info:
            num_samples = info['shape'][0]
        else:
            data = torch.load(data_path, map_location='cpu')
            num_samples = data.shape[0]
            del data  # Free memory
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # Create train/test split indices
    torch.manual_seed(seed)
    indices = torch.randperm(num_samples).tolist()
    split_idx = int(train_ratio * num_samples)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Create datasets with specific indices
    train_dataset = LazyAVIRISDataset(data_path, indices=train_indices)
    test_dataset = LazyAVIRISDataset(data_path, indices=test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Created data split with {len(train_dataset)} training and {len(test_dataset)} testing samples")
    
    return train_loader, test_loader, train_dataset, test_dataset


###############################################################################
# BATCH METRICS CALCULATION
###############################################################################
def calculate_metrics_in_batches(model, data_loader, device, max_batches=None):
    """
    Calculate MSE, PSNR, and SAM metrics across batches to handle large datasets.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for the dataset
        device: Device to use for computation
        max_batches: Maximum number of batches to process (None for all)
        
    Returns:
        tuple: (avg_mse, avg_psnr, avg_sam, avg_snr)
    """
    model.eval()
    total_mse = 0.0
    total_psnr = 0.0
    total_sam = 0.0
    total_snr = 0.0
    batch_count = 0
    sample_count = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Calculating metrics")):
            if max_batches is not None and i >= max_batches:
                break
                
            # Handle both simple tensors and tuple/list batch formats
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            
            # Forward pass
            recon, _, snr = model(x)
            
            # Calculate MSE
            batch_mse = ((recon - x) ** 2).mean().item()
            
            # Calculate PSNR
            batch_psnr = calculate_psnr(x.cpu(), recon.cpu())
            
            # Calculate SAM
            batch_sam = calculate_sam(x.cpu(), recon.cpu())
            
            # Convert SNR to scalar if it's a tensor
            if torch.is_tensor(snr):
                batch_snr = snr.item()
            else:
                batch_snr = snr
            
            # Accumulate weighted by batch size
            batch_size = x.size(0)
            total_mse += batch_mse * batch_size
            total_psnr += batch_psnr * batch_size
            total_sam += batch_sam * batch_size
            total_snr += batch_snr * batch_size
            
            sample_count += batch_size
            batch_count += 1
            
            # Print progress for large datasets
            if batch_count % 10 == 0:
                print(f"Processed {batch_count} batches, {sample_count} samples")
                print(f"Current averages - MSE: {total_mse/sample_count:.6f}, PSNR: {total_psnr/sample_count:.2f} dB, SAM: {total_sam/sample_count:.6f} rad")
    
    # Calculate averages
    avg_mse = total_mse / sample_count if sample_count > 0 else 0
    avg_psnr = total_psnr / sample_count if sample_count > 0 else 0
    avg_sam = total_sam / sample_count if sample_count > 0 else 0
    avg_snr = total_snr / sample_count if sample_count > 0 else 0
    
    return avg_mse, avg_psnr, avg_sam, avg_snr


###############################################################################
# TRAINING FUNCTION WITH RANDOM NOISE
###############################################################################
def train_with_random_noise(shape2filter_path, filter2shape_path, output_dir, min_snr=10, max_snr=40, 
                           initial_filter_params=None, batch_size=10, num_epochs=500, 
                           encoder_lr=0.001, decoder_lr=0.001, filter_scale_factor=10.0,
                           cache_file=None, use_cache=False, folder_patterns="all",
                           train_data=None, test_data=None, viz_interval_stage1=1,
                           train_loader=None, test_loader=None, eval_batches=None):
    """
    Train and visualize the hyperspectral autoencoder with random noise levels between min_snr and max_snr
    using filter2shape2filter architecture with separate learning rates for encoder and decoder
    
    Parameters:
    shape2filter_path: Path to the pretrained shape2filter model
    filter2shape_path: Path to the pretrained filter2shape model
    output_dir: Directory to save outputs
    min_snr: Minimum SNR level in dB for random noise
    max_snr: Maximum SNR level in dB for random noise
    initial_filter_params: Initial filter parameters (11x100)
    batch_size: Batch size for training
    num_epochs: Number of training epochs
    encoder_lr: Learning rate for encoder (filter parameters)
    decoder_lr: Learning rate for decoder
    filter_scale_factor: Scaling factor for filter normalization
    cache_file: Path to cache file for storing processed data
    use_cache: If True, try to load from cache file first
    folder_patterns: Comma-separated list of folder name patterns to include, or 'all' for all folders
    train_data: Optional training data tensor, if None will be loaded
    test_data: Optional testing data tensor, if None will be loaded
    viz_interval_stage1: Interval for visualization and metrics printing in stage 1
    train_loader: Optional DataLoader for training data
    test_loader: Optional DataLoader for testing data
    eval_batches: Maximum number of batches to use for evaluation (None for all)
    
    Returns:
    tuple: (initial_shape_path, least_mse_shape_path, lowest_cn_shape_path, final_shape_path, output_dir)
    """
    # Create output directory with noise range
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    noise_dir = f"blind_noise_{min_snr}dB_to_{max_snr}dB"
    output_dir = os.path.join(output_dir, noise_dir + f"_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}/")
    
    # Create subfolders
    viz_dir = os.path.join(output_dir, "intermediate_viz")
    recon_dir = os.path.join(output_dir, "reconstructions")
    filter_params_dir = os.path.join(viz_dir, "filter_params")
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(filter_params_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load or use provided data
    if train_loader is None:
        if train_data is None:
            # We need to load the data
            data = load_aviris_forest_data(base_path="AVIRIS_FOREST_SIMPLE_SELECT", tile_size=128, 
                                          cache_file=cache_file, use_cache=use_cache, folder_patterns=folder_patterns)
            print("Training data shape:", data.shape)
            data = data.to(device)
            
            # Create dataset and dataloader
            dataset = TensorDataset(data)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            # Use provided data tensor
            train_data = train_data.to(device)
            print("Training data shape:", train_data.shape)
            
            # Create dataset and dataloader
            dataset = TensorDataset(train_data)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        print(f"Using provided train_loader with batch size {batch_size}")
    
    # Create test dataloader if test data is provided and test_loader not provided
    if test_loader is None:
        if test_data is not None:
            test_dataset = TensorDataset(test_data)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            print("Test data shape:", test_data.shape)
    
    # If no initial filter parameters were provided, generate them
    if initial_filter_params is None:
        initial_filter_params = generate_initial_filter(device)
    
    # Initialize model with random noise level range and initial filter parameters
    model = HyperspectralAutoencoderRandomNoise(
        shape2filter_path, filter2shape_path, 
        min_snr=min_snr, max_snr=max_snr, 
        initial_filter_params=initial_filter_params,
        filter_scale_factor=filter_scale_factor
    ).to(device)
    
    print(f"Model initialized with random noise range: {min_snr} to {max_snr} dB")
    print(f"Filter scale factor: {filter_scale_factor}")
    print(f"Encoder learning rate: {encoder_lr}")
    print(f"Decoder learning rate: {decoder_lr}")
    print(f"Visualization interval: {viz_interval_stage1} epochs")
    
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
    
    # Save initial filter_params
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(initial_filter_params.detach().cpu().numpy()[i], label=f'Filter Param {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Parameter Value")
    plt.title(f"Initial Filter Parameters (SNR: {min_snr}-{max_snr} dB)")
    plt.legend()
    initial_params_path = f"{filter_params_dir}/filter_params_epoch_0.png"
    plt.savefig(initial_params_path)
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
    
    # Get a small sample for visualization (just a few samples)
    sample_tensor = None
    for batch in train_loader:
        if isinstance(batch, (list, tuple)):
            sample_tensor = batch[0][:3].to(device)  # Just take first 5 samples
        else:
            sample_tensor = batch[:3].to(device)  # Just take first 5 samples
        break
    
    # If test data is provided, also get test sample
    test_sample_tensor = None
    if test_loader is not None:
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                test_sample_tensor = batch[0][:3].to(device)  # Just take first 5 samples
            else:
                test_sample_tensor = batch[:3].to(device)  # Just take first 5 samples
            break
    
    # Generate initial reconstruction for visualization
    initial_recon_path = os.path.join(recon_dir, "initial_reconstruction.png")
    initial_mse, initial_psnr, initial_sam = visualize_reconstruction(
        model, sample_tensor, device, initial_recon_path)
    _, _, _ = visualize_reconstruction_spectrum(
        model, sample_tensor, device, initial_recon_path)
    
    # Calculate initial metrics on full dataset using batch processing
    print("Calculating initial metrics on full dataset...")
    initial_full_mse, initial_full_psnr, initial_full_sam, initial_full_snr = calculate_metrics_in_batches(
        model, train_loader, device, max_batches=eval_batches)
    
    print(f"Initial sample metrics - MSE: {initial_mse:.6f}, PSNR: {initial_psnr:.2f} dB, SAM: {initial_sam:.6f} rad")
    print(f"Initial full dataset metrics - MSE: {initial_full_mse:.6f}, PSNR: {initial_full_psnr:.2f} dB, SAM: {initial_full_sam:.6f} rad")
    
    # If test data is available, calculate test metrics
    initial_test_full_mse, initial_test_full_psnr, initial_test_full_sam, initial_test_full_snr = 0, 0, 0, 0
    if test_loader is not None:
        print("Calculating initial test metrics...")
        initial_test_full_mse, initial_test_full_psnr, initial_test_full_sam, initial_test_full_snr = calculate_metrics_in_batches(
            model, test_loader, device, max_batches=eval_batches)
        print(f"Initial test metrics - MSE: {initial_test_full_mse:.6f}, PSNR: {initial_test_full_psnr:.2f} dB, SAM: {initial_test_full_sam:.6f} rad")
    
    # Initialize tracking variables
    losses = []
    condition_numbers = [initial_condition_number]
    train_mse_values = [initial_full_mse]
    train_psnr_values = [initial_full_psnr]
    train_sam_values = [initial_full_sam]
    applied_snr_values = []
    
    # Initialize test metrics tracking if test data is available
    test_mse_values = []
    test_psnr_values = []
    test_sam_values = []
    
    if test_loader is not None:
        test_mse_values.append(initial_test_full_mse)
        test_psnr_values.append(initial_test_full_psnr)
        test_sam_values.append(initial_test_full_sam)
    
    # Variables for tracking best metrics
    lowest_condition_number = float('inf')
    lowest_cn_shape = None
    lowest_cn_filter = None
    lowest_cn_epoch = -1
    
    lowest_train_mse = initial_full_mse
    lowest_mse_shape = initial_shape.copy()
    lowest_mse_filter = initial_filter.copy()
    lowest_mse_epoch = -1
    
    # Create separate optimizers for encoder and decoder
    encoder_optimizer = optim.Adam([model.filter_params], lr=encoder_lr)
    decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=decoder_lr)
    
    criterion = nn.MSELoss()
    
    print(f"Starting training with random noise SNR range: {min_snr} to {max_snr} dB...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        epoch_snr_values = []
        
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Handle both simple tensors and tuple/list batch formats
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            
            # Forward pass with random noise
            recon, _, batch_snr = model(x)
            epoch_snr_values.append(batch_snr)
            
            # Calculate loss
            loss = criterion(recon, x)
            
            # Backward pass and optimize
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            
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
            lowest_cn_filter = current_filter_raw.clone()
            lowest_cn_epoch = epoch
            print(f"New lowest condition number: {lowest_condition_number:.4f} at epoch {epoch+1}")
        
        # Evaluate model on sample for detailed metrics every viz_interval_stage1 epochs or last epoch
        if (epoch + 1) % viz_interval_stage1 == 0 or epoch == num_epochs - 1:
            # Evaluate on visualization sample
            model.eval()
            recon_path = os.path.join(recon_dir, f"reconstruction_epoch_{epoch+1}.png")
            current_mse, current_psnr, current_sam = visualize_reconstruction(
                model, sample_tensor, device, recon_path)
            _, _, _ = visualize_reconstruction_spectrum(
                model, sample_tensor, device, recon_path)
            
            # Calculate metrics on full dataset using batch processing
            print(f"Calculating metrics on full dataset for epoch {epoch+1}...")
            current_full_mse, current_full_psnr, current_full_sam, current_full_snr = calculate_metrics_in_batches(
                model, train_loader, device, max_batches=eval_batches)
            
            # Save metrics
            train_mse_values.append(current_full_mse)
            train_psnr_values.append(current_full_psnr)
            train_sam_values.append(current_full_sam)
            
            # Print detailed training metrics
            print(f"\nEpoch {epoch+1} detailed metrics:")
            print(f"  Sample - MSE: {current_mse:.6f}, PSNR: {current_psnr:.2f} dB, SAM: {current_sam:.6f} rad")
            print(f"  Full Train - MSE: {current_full_mse:.6f}, PSNR: {current_full_psnr:.2f} dB, SAM: {current_full_sam:.6f} rad")
            
            # If test data is available, calculate test metrics
            if test_loader is not None:
                print("Calculating test metrics...")
                test_full_mse, test_full_psnr, test_full_sam, test_full_snr = calculate_metrics_in_batches(
                    model, test_loader, device, max_batches=eval_batches)
                
                # Save test metrics
                test_mse_values.append(test_full_mse)
                test_psnr_values.append(test_full_psnr)
                test_sam_values.append(test_full_sam)
                
                print(f"  Full Test - MSE: {test_full_mse:.6f}, PSNR: {test_full_psnr:.2f} dB, SAM: {test_full_sam:.6f} rad")
            
            # Check if this is the lowest MSE so far
            if current_full_mse < lowest_train_mse:
                lowest_train_mse = current_full_mse
                lowest_mse_shape = current_shape.copy()
                lowest_mse_filter = current_filter_raw.clone()
                lowest_mse_epoch = epoch
                print(f"New lowest MSE: {lowest_train_mse:.6f} at epoch {epoch+1}")
        
        # Save intermediate shapes, filters, and filter_params visualizations
        if (epoch+1) % viz_interval_stage1 == 0 or epoch == 0:
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
            
            # Save filter_params visualization
            current_params = model.filter_params.detach().cpu().numpy()
            plt.figure(figsize=(10, 6))
            for i in range(11):
                plt.plot(current_params[i], label=f'Filter Param {i}' if i % 3 == 0 else None)
            plt.grid(True)
            plt.xlabel("Wavelength Index")
            plt.ylabel("Parameter Value")
            plt.title(f"Filter Parameters at Epoch {epoch+1}")
            plt.legend()
            plt.savefig(f"{filter_params_dir}/filter_params_epoch_{epoch+1}.png")
            plt.close()
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Avg SNR: {avg_snr:.2f} dB, Condition Number: {current_condition_number:.4f}")
    
    # Get final filter and shape
    final_filter = model.get_current_filter().detach().cpu()
    final_shape = model.get_current_shape().detach().cpu().numpy()
    
    # Calculate final condition number
    final_condition_number = calculate_condition_number(model.get_reconstructed_filter().detach().cpu())
    print(f"Final condition number: {final_condition_number:.4f}")
    
    # Get final reconstruction and metrics
    final_recon_path = os.path.join(recon_dir, "final_reconstruction.png")
    final_mse, final_psnr, final_sam = visualize_reconstruction(
        model, sample_tensor, device, final_recon_path)
    _, _, _ = visualize_reconstruction_spectrum(
        model, sample_tensor, device, final_recon_path)
    
    # Calculate final metrics on full dataset using batch processing
    print("Calculating final metrics on full dataset...")
    final_full_mse, final_full_psnr, final_full_sam, final_full_snr = calculate_metrics_in_batches(
        model, train_loader, device, max_batches=eval_batches)
    
    print(f"Final sample metrics - MSE: {final_mse:.6f}, PSNR: {final_psnr:.2f} dB, SAM: {final_sam:.6f} rad")
    print(f"Final full dataset metrics - MSE: {final_full_mse:.6f}, PSNR: {final_full_psnr:.2f} dB, SAM: {final_full_sam:.6f} rad")
    
    # If test data is available, calculate final test metrics
    final_test_full_mse, final_test_full_psnr, final_test_full_sam, final_test_full_snr = 0, 0, 0, 0
    if test_loader is not None:
        print("Calculating final test metrics...")
        final_test_full_mse, final_test_full_psnr, final_test_full_sam, final_test_full_snr = calculate_metrics_in_batches(
            model, test_loader, device, max_batches=eval_batches)
        print(f"Final test metrics - MSE: {final_test_full_mse:.6f}, PSNR: {final_test_full_psnr:.2f} dB, SAM: {final_test_full_sam:.6f} rad")
    
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
    
    # Save final filter_params
    final_params = model.filter_params.detach().cpu().numpy()
    plt.figure(figsize=(12, 8))
    for i in range(11):
        plt.plot(final_params[i], label=f'Filter Param {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Parameter Value")
    plt.title(f"Final Filter Parameters (SNR: {min_snr}-{max_snr} dB)")
    plt.legend()
    final_params_path = f"{filter_params_dir}/filter_params_final.png"
    plt.savefig(final_params_path)
    plt.close()
    
    # Save reconstructed filter
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
    
    # Create directories for lowest CN and MSE results
    lowest_cn_dir = os.path.join(output_dir, "lowest_cn")
    lowest_mse_dir = os.path.join(output_dir, "lowest_mse")
    os.makedirs(lowest_cn_dir, exist_ok=True)
    os.makedirs(lowest_mse_dir, exist_ok=True)
    
    # Save the lowest condition number shape and filter
    lowest_cn_shape_path = f"{lowest_cn_dir}/shape.png"
    plot_shape_with_c4(lowest_cn_shape, f"Shape with Lowest CN: {lowest_condition_number:.4f}", lowest_cn_shape_path)
    
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
    
    # Save the lowest MSE shape and filter
    lowest_mse_shape_path = f"{lowest_mse_dir}/shape.png"
    plot_shape_with_c4(lowest_mse_shape, f"Shape with Lowest MSE: {lowest_train_mse:.6f}", lowest_mse_shape_path)
    
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
    
    # Save as NumPy files
    np.save(f"{output_dir}/initial_shape.npy", initial_shape)
    np.save(f"{output_dir}/final_shape.npy", final_shape)
    np.save(f"{lowest_cn_dir}/shape.npy", lowest_cn_shape)
    np.save(f"{lowest_mse_dir}/shape.npy", lowest_mse_shape)
    
    np.save(f"{output_dir}/initial_filter.npy", initial_filter)
    np.save(f"{output_dir}/final_filter.npy", final_filter.numpy())
    np.save(f"{lowest_cn_dir}/filter.npy", lowest_cn_filter.numpy())
    np.save(f"{lowest_mse_dir}/filter.npy", lowest_mse_filter.numpy())
    
    # Save filter_params as NumPy files
    np.save(f"{output_dir}/initial_filter_params.npy", initial_filter_params.detach().cpu().numpy())
    np.save(f"{output_dir}/final_filter_params.npy", final_params)
    
    # Plot metrics during training
    # 1. Condition Number
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs+1), condition_numbers, 'r-')
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Condition Number")
    plt.title(f"Filter Matrix Condition Number During Training (SNR: {min_snr}-{max_snr} dB)")
    plt.axhline(y=lowest_condition_number, color='g', linestyle='--', 
                label=f'Lowest CN: {lowest_condition_number:.4f} (Epoch {lowest_cn_epoch+1})')
    plt.legend()
    plt.savefig(f"{output_dir}/condition_number.png")
    plt.close()
    
    # 2. Training Loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Loss (SNR: {min_snr}-{max_snr} dB)")
    plt.savefig(f"{output_dir}/training_loss.png")
    plt.close()
    
    # 3. Applied SNR
    plt.figure(figsize=(10, 5))
    plt.plot(applied_snr_values)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Average SNR (dB)")
    plt.title(f"Average Applied SNR During Training (Range: {min_snr}-{max_snr} dB)")
    plt.ylim(min_snr-1, max_snr+1)
    plt.savefig(f"{output_dir}/applied_snr.png")
    plt.close()
    
    # 4. MSE values
    epochs_with_metrics = list(range(0, num_epochs+1, viz_interval_stage1))
    if num_epochs not in epochs_with_metrics:
        epochs_with_metrics.append(num_epochs)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_with_metrics, train_mse_values, 'b-o', label='Train MSE')
    if test_loader is not None:
        plt.plot(epochs_with_metrics, test_mse_values, 'r-o', label='Test MSE')
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"MSE During Training (SNR: {min_snr}-{max_snr} dB)")
    plt.legend()
    plt.savefig(f"{output_dir}/mse_values.png")
    plt.close()
    
    # 5. PSNR values
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_with_metrics, train_psnr_values, 'g-o', label='Train PSNR')
    if test_loader is not None:
        plt.plot(epochs_with_metrics, test_psnr_values, 'm-o', label='Test PSNR')
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title(f"PSNR During Training (SNR: {min_snr}-{max_snr} dB)")
    plt.legend()
    plt.savefig(f"{output_dir}/psnr_values.png")
    plt.close()
    
    # 6. SAM values
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_with_metrics, train_sam_values, 'm-o', label='Train SAM')
    if test_loader is not None:
        plt.plot(epochs_with_metrics, test_sam_values, 'c-o', label='Test SAM')
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("SAM (rad)")
    plt.title(f"SAM During Training (SNR: {min_snr}-{max_snr} dB)")
    plt.legend()
    plt.savefig(f"{output_dir}/sam_values.png")
    plt.close()
    
    # 7. Combined metrics (MSE, PSNR, SAM)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # MSE subplot
    ax1.plot(epochs_with_metrics, train_mse_values, 'b-o', label='Train MSE')
    if test_loader is not None:
        ax1.plot(epochs_with_metrics, test_mse_values, 'r-o', label='Test MSE')
    ax1.set_ylabel("MSE")
    ax1.set_title(f"Image Quality Metrics During Training (SNR: {min_snr}-{max_snr} dB)")
    ax1.grid(True)
    ax1.legend()
    
    # PSNR subplot
    ax2.plot(epochs_with_metrics, train_psnr_values, 'g-o', label='Train PSNR')
    if test_loader is not None:
        ax2.plot(epochs_with_metrics, test_psnr_values, 'm-o', label='Test PSNR')
    ax2.set_ylabel("PSNR (dB)")
    ax2.grid(True)
    ax2.legend()
    
    # SAM subplot
    ax3.plot(epochs_with_metrics, train_sam_values, 'c-o', label='Train SAM')
    if test_loader is not None:
        ax3.plot(epochs_with_metrics, test_sam_values, 'y-o', label='Test SAM')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("SAM (rad)")
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_metrics.png")
    plt.close()
    
    # Save model
    model_save_path = f"{output_dir}/model_state.pt"
    torch.save({
        'filter_params': model.filter_params.detach().cpu(),
        'decoder_state_dict': model.decoder.state_dict(),
        'filter_scale_factor': model.filter_scale_factor
    }, model_save_path)
    
    # Save all numerical data
    np.save(f"{output_dir}/condition_numbers.npy", np.array(condition_numbers))
    np.save(f"{output_dir}/losses.npy", np.array(losses))
    np.save(f"{output_dir}/train_mse_values.npy", np.array(train_mse_values))
    np.save(f"{output_dir}/train_psnr_values.npy", np.array(train_psnr_values))
    np.save(f"{output_dir}/train_sam_values.npy", np.array(train_sam_values))
    np.save(f"{output_dir}/applied_snr_values.npy", np.array(applied_snr_values))
    
    if test_loader is not None:
        np.save(f"{output_dir}/test_mse_values.npy", np.array(test_mse_values))
        np.save(f"{output_dir}/test_psnr_values.npy", np.array(test_psnr_values))
        np.save(f"{output_dir}/test_sam_values.npy", np.array(test_sam_values))
    
    # Create log file to save training parameters
    with open(f"{output_dir}/training_params.txt", "w") as f:
        f.write(f"Noise level range (SNR): {min_snr} to {max_snr} dB\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Encoder learning rate: {encoder_lr}\n")
        f.write(f"Decoder learning rate: {decoder_lr}\n")
        f.write(f"Filter scale factor: {filter_scale_factor}\n")
        f.write(f"Cache file: {cache_file}\n")
        f.write(f"Used cache: {use_cache}\n")
        f.write(f"Folder patterns: {folder_patterns}\n")
        f.write(f"Visualization interval: {viz_interval_stage1} epochs\n")
        f.write(f"Evaluation batches limit: {eval_batches}\n\n")
        
        # Save condition number information
        f.write(f"Initial condition number: {initial_condition_number:.4f}\n")
        f.write(f"Final condition number: {final_condition_number:.4f}\n")
        f.write(f"Lowest condition number: {lowest_condition_number:.4f} at epoch {lowest_cn_epoch+1}\n")
        f.write(f"Condition number change: {final_condition_number - initial_condition_number:.4f}\n\n")
        
        # Save metrics information
        f.write(f"Initial visualization sample metrics:\n")
        f.write(f"  MSE: {initial_mse:.6f}, PSNR: {initial_psnr:.2f} dB, SAM: {initial_sam:.6f} rad\n\n")
        
        f.write(f"Initial full dataset metrics:\n")
        f.write(f"  Train - MSE: {initial_full_mse:.6f}, PSNR: {initial_full_psnr:.2f} dB, SAM: {initial_full_sam:.6f} rad\n")
        if test_loader is not None:
            f.write(f"  Test  - MSE: {initial_test_full_mse:.6f}, PSNR: {initial_test_full_psnr:.2f} dB, SAM: {initial_test_full_sam:.6f} rad\n\n")
        
        f.write(f"Final visualization sample metrics:\n")
        f.write(f"  MSE: {final_mse:.6f}, PSNR: {final_psnr:.2f} dB, SAM: {final_sam:.6f} rad\n\n")
        
        f.write(f"Final full dataset metrics after {num_epochs} epochs:\n")
        f.write(f"  Train - MSE: {final_full_mse:.6f}, PSNR: {final_full_psnr:.2f} dB, SAM: {final_full_sam:.6f} rad\n")
        if test_loader is not None:
            f.write(f"  Test  - MSE: {final_test_full_mse:.6f}, PSNR: {final_test_full_psnr:.2f} dB, SAM: {final_test_full_sam:.6f} rad\n\n")
        
        f.write(f"Lowest MSE: {lowest_train_mse:.6f} at epoch {lowest_mse_epoch+1}\n")
        f.write(f"MSE improvement: {initial_full_mse - final_full_mse:.6f} ({(1 - final_full_mse/initial_full_mse) * 100:.2f}%)\n")
        f.write(f"PSNR improvement: {final_full_psnr - initial_full_psnr:.2f} dB\n")
        f.write(f"SAM improvement: {initial_full_sam - final_full_sam:.6f} rad ({(1 - final_full_sam/initial_full_sam) * 100:.2f}%)\n")
        
        if test_loader is not None:
            f.write("\nTest metrics improvements:\n")
            f.write(f"  Test MSE: {initial_test_full_mse - final_test_full_mse:.6f} ({(1 - final_test_full_mse/initial_test_full_mse) * 100:.2f}%)\n")
            f.write(f"  Test PSNR: {final_test_full_psnr - initial_test_full_psnr:.2f} dB\n")
            f.write(f"  Test SAM: {initial_test_full_sam - final_test_full_sam:.6f} rad ({(1 - final_test_full_sam/initial_test_full_sam) * 100:.2f}%)\n")
    
    print(f"Training with random noise SNR range {min_snr} to {max_snr} dB completed.")
    print(f"All results saved to {output_dir}/")
    
    # Return paths to all shapes for later training
    initial_shape_npy_path = f"{output_dir}/initial_shape.npy"
    final_shape_npy_path = f"{output_dir}/final_shape.npy"
    lowest_cn_shape_npy_path = f"{lowest_cn_dir}/shape.npy"
    lowest_mse_shape_npy_path = f"{lowest_mse_dir}/shape.npy"
    
    return initial_shape_npy_path, lowest_mse_shape_npy_path, lowest_cn_shape_npy_path, final_shape_npy_path, output_dir


###############################################################################
# FIXED SHAPE DECODER TRAINING WITH METRICS - BATCHED VERSION
###############################################################################
def train_with_fixed_shape(shape_name, shape, shape2filter_path, train_loader, test_loader, 
                          noise_level, num_epochs, batch_size, decoder_lr, filter_scale_factor, 
                          output_dir, viz_interval_stage2=1, eval_batches=None):
    """
    Train model with fixed shape encoder and optimize only the decoder
    
    Parameters:
    shape_name: Name of the shape for logs and outputs
    shape: Tensor or array representing the shape
    shape2filter_path: Path to shape2filter model
    train_loader: Training data loader
    test_loader: Testing data loader
    noise_level: SNR level in dB to use
    num_epochs: Number of epochs for training
    batch_size: Batch size for training
    decoder_lr: Learning rate for decoder optimizer
    filter_scale_factor: Scaling factor for filter normalization
    output_dir: Directory to save outputs
    viz_interval_stage2: Interval for visualization in stage 2
    eval_batches: Maximum number of batches to use for evaluation (None for all)
    
    Returns:
    tuple: Dictionary with metrics over epochs
    """
    # Create output directory
    shape_dir = os.path.join(output_dir, f"shape_{shape_name}")
    recon_dir = os.path.join(shape_dir, "reconstructions")
    os.makedirs(shape_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with fixed shape
    model = FixedShapeModel(
        shape=shape,
        shape2filter_path=shape2filter_path,
        noise_level=noise_level,
        filter_scale_factor=filter_scale_factor,
        device=device
    )
    model = model.to(device)
    
    # Create optimizer for decoder only
    optimizer = optim.Adam(model.decoder.parameters(), lr=decoder_lr)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Track metrics
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'train_psnr': [],
        'test_psnr': [],
        'train_sam': [],
        'test_sam': []
    }
    
    # Get a sample batch for visualization (just a few samples)
    train_sample = None
    for batch in train_loader:
        if isinstance(batch, (list, tuple)):
            train_sample = batch[0][:1].to(device)  # Just take first sample
        else:
            train_sample = batch[:1].to(device)  # Just take first sample
        break
    
    test_sample = None
    for batch in test_loader:
        if isinstance(batch, (list, tuple)):
            test_sample = batch[0][:1].to(device)  # Just take first sample
        else:
            test_sample = batch[:1].to(device)  # Just take first sample
        break
    
    # Calculate initial metrics on full datasets
    print(f"Calculating initial metrics for {shape_name} at {noise_level} dB...")
    model.eval()
    
    # Calculate on train set
    train_mse, train_psnr, train_sam, _ = calculate_metrics_in_batches(
        model, train_loader, device, max_batches=eval_batches)
    
    # Calculate on test set
    test_mse, test_psnr, test_sam, _ = calculate_metrics_in_batches(
        model, test_loader, device, max_batches=eval_batches)
    
    # Save initial metrics
    metrics['train_loss'].append(train_mse)
    metrics['test_loss'].append(test_mse)
    metrics['train_psnr'].append(train_psnr)
    metrics['test_psnr'].append(test_psnr)
    metrics['train_sam'].append(train_sam)
    metrics['test_sam'].append(test_sam)
    
    # Initial visualization
    initial_recon_path = os.path.join(recon_dir, "reconstruction_epoch_0.png")
    with torch.no_grad():
        model.eval()
        train_out, _ = model(train_sample, add_noise=False)
        
        # Plot original, reconstruction, and difference
        plt.figure(figsize=(15, 5))
        
        # Select middle band for visualization
        band_idx = train_sample.shape[-1] // 2
        
        # Get data as numpy
        orig = train_sample[0].permute(2, 0, 1)[band_idx].cpu().numpy()
        recon = train_out[0].permute(2, 0, 1)[band_idx].cpu().numpy()
        diff = orig - recon
        
        # Find global min/max for colorbar
        vmin = min(orig.min(), recon.min())
        vmax = max(orig.max(), recon.max())
        
        # Calculate symmetric limits for difference
        diff_abs_max = max(abs(diff.min()), abs(diff.max()))
        
        # Plot original
        plt.subplot(1, 3, 1)
        im1 = plt.imshow(orig, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title('Original')
        plt.colorbar(im1, fraction=0.046, pad=0.04)
        
        # Plot reconstruction
        plt.subplot(1, 3, 2)
        im2 = plt.imshow(recon, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title('Reconstructed')
        plt.colorbar(im2, fraction=0.046, pad=0.04)
        
        # Plot difference
        plt.subplot(1, 3, 3)
        im3 = plt.imshow(diff, cmap='coolwarm', vmin=-diff_abs_max, vmax=diff_abs_max)
        plt.title('Difference')
        plt.colorbar(im3, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(initial_recon_path)
        plt.close()
    
    # Print initial metrics with more detail
    print(f"\nInitial {shape_name} metrics at {noise_level} dB:")
    print(f"  Train - MSE: {train_mse:.6f}, PSNR: {train_psnr:.2f} dB, SAM: {train_sam:.6f} rad")
    print(f"  Test  - MSE: {test_mse:.6f}, PSNR: {test_psnr:.2f} dB, SAM: {test_sam:.6f} rad")
    
    # Calculate and print condition number
    with torch.no_grad():
        condition_number = calculate_condition_number(model.fixed_filter.detach().cpu())
        print(f"  Filter condition number: {condition_number:.4f}")
    
    # Best metrics tracking
    best_test_loss = test_mse
    best_epoch = 0
    best_model_state = model.decoder.state_dict()
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, {shape_name} @ {noise_level}dB"):
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            
            # Forward pass
            recon, _ = model(x, add_noise=True)
            
            # Loss
            loss = criterion(recon, x)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            num_batches += 1
        
        # Calculate average training loss
        avg_train_loss = train_epoch_loss / num_batches
        
        # Evaluation phase - only perform full evaluation at specified intervals
        if (epoch + 1) % viz_interval_stage2 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                # Calculate on train set
                train_mse, train_psnr, train_sam, _ = calculate_metrics_in_batches(
                    model, train_loader, device, max_batches=eval_batches)
                
                # Calculate on test set
                test_mse, test_psnr, test_sam, _ = calculate_metrics_in_batches(
                    model, test_loader, device, max_batches=eval_batches)
                
                # Save metrics (for the epochs without full evaluation, use the training loss)
                metrics['train_loss'].append(train_mse)
                metrics['test_loss'].append(test_mse)
                metrics['train_psnr'].append(train_psnr)
                metrics['test_psnr'].append(test_psnr)
                metrics['train_sam'].append(train_sam)
                metrics['test_sam'].append(test_sam)
                
                # Print detailed metrics
                print(f"\nEpoch {epoch+1}/{num_epochs}, {shape_name} shape at {noise_level} dB:")
                print(f"  Train - MSE: {train_mse:.6f}, PSNR: {train_psnr:.2f} dB, SAM: {train_sam:.6f} rad")
                print(f"  Test  - MSE: {test_mse:.6f}, PSNR: {test_psnr:.2f} dB, SAM: {test_sam:.6f} rad")
                print(f"  Filter condition number: {condition_number:.4f}")
            
                # Check if this is the best model so far
                if test_mse < best_test_loss:
                    best_test_loss = test_mse
                    best_epoch = epoch
                    best_model_state = model.decoder.state_dict()
                    
                    # Save best reconstruction
                    best_recon_path = os.path.join(shape_dir, "best_reconstruction.png")
                    with torch.no_grad():
                        model.eval()
                        train_out, _ = model(train_sample, add_noise=False)
                        
                        # Plot original, reconstruction, and difference
                        plt.figure(figsize=(15, 5))
                        
                        # Select middle band for visualization
                        band_idx = train_sample.shape[-1] // 2
                        
                        # Get data as numpy
                        orig = train_sample[0].permute(2, 0, 1)[band_idx].cpu().numpy()
                        recon = train_out[0].permute(2, 0, 1)[band_idx].cpu().numpy()
                        diff = orig - recon
                        
                        # Find global min/max for colorbar
                        vmin = min(orig.min(), recon.min())
                        vmax = max(orig.max(), recon.max())
                        
                        # Calculate symmetric limits for difference
                        diff_abs_max = max(abs(diff.min()), abs(diff.max()))
                        
                        # Plot original
                        plt.subplot(1, 3, 1)
                        im1 = plt.imshow(orig, cmap='viridis', vmin=vmin, vmax=vmax)
                        plt.title('Original')
                        plt.colorbar(im1, fraction=0.046, pad=0.04)
                        
                        # Plot reconstruction
                        plt.subplot(1, 3, 2)
                        im2 = plt.imshow(recon, cmap='viridis', vmin=vmin, vmax=vmax)
                        plt.title(f'Best Reconstructed (Epoch {epoch+1})')
                        plt.colorbar(im2, fraction=0.046, pad=0.04)
                        
                        # Plot difference
                        plt.subplot(1, 3, 3)
                        im3 = plt.imshow(diff, cmap='coolwarm', vmin=-diff_abs_max, vmax=diff_abs_max)
                        plt.title('Difference')
                        plt.colorbar(im3, fraction=0.046, pad=0.04)
                        
                        plt.tight_layout()
                        plt.savefig(best_recon_path)
                        plt.close()
                    
                    print(f"  New best model at epoch {epoch+1} - Test MSE: {best_test_loss:.6f}")
                
                # Save reconstruction at specified intervals
                recon_path = os.path.join(recon_dir, f"reconstruction_epoch_{epoch+1}.png")
                with torch.no_grad():
                    model.eval()
                    train_out, _ = model(train_sample, add_noise=False)
                    
                    # Plot original, reconstruction, and difference
                    plt.figure(figsize=(15, 5))
                    
                    # Select middle band for visualization
                    band_idx = train_sample.shape[-1] // 2
                    
                    # Get data as numpy
                    orig = train_sample[0].permute(2, 0, 1)[band_idx].cpu().numpy()
                    recon = train_out[0].permute(2, 0, 1)[band_idx].cpu().numpy()
                    diff = orig - recon
                    
                    # Find global min/max for colorbar
                    vmin = min(orig.min(), recon.min())
                    vmax = max(orig.max(), recon.max())
                    
                    # Calculate symmetric limits for difference
                    diff_abs_max = max(abs(diff.min()), abs(diff.max()))
                    
                    # Plot original
                    plt.subplot(1, 3, 1)
                    im1 = plt.imshow(orig, cmap='viridis', vmin=vmin, vmax=vmax)
                    plt.title('Original')
                    plt.colorbar(im1, fraction=0.046, pad=0.04)
                    
                    # Plot reconstruction
                    plt.subplot(1, 3, 2)
                    im2 = plt.imshow(recon, cmap='viridis', vmin=vmin, vmax=vmax)
                    plt.title(f'Reconstructed (Epoch {epoch+1})')
                    plt.colorbar(im2, fraction=0.046, pad=0.04)
                    
                    # Plot difference
                    plt.subplot(1, 3, 3)
                    im3 = plt.imshow(diff, cmap='coolwarm', vmin=-diff_abs_max, vmax=diff_abs_max)
                    plt.title('Difference')
                    plt.colorbar(im3, fraction=0.046, pad=0.04)
                    
                    plt.tight_layout()
                    plt.savefig(recon_path)
                    plt.close()
        else:
            # For epochs where we don't do full evaluation, still track the training loss
            metrics['train_loss'].append(avg_train_loss)
            # For other metrics, duplicate the last values
            metrics['test_loss'].append(metrics['test_loss'][-1])
            metrics['train_psnr'].append(metrics['train_psnr'][-1])
            metrics['test_psnr'].append(metrics['test_psnr'][-1])
            metrics['train_sam'].append(metrics['train_sam'][-1])
            metrics['test_sam'].append(metrics['test_sam'][-1])
            
            # Print basic progress
            print(f"Epoch {epoch+1}/{num_epochs}, {shape_name} @ {noise_level}dB, Train Loss: {avg_train_loss:.6f}")
    
    # Calculate final metrics
    model.eval()
    with torch.no_grad():
        # Calculate on train set
        final_train_mse, final_train_psnr, final_train_sam, _ = calculate_metrics_in_batches(
            model, train_loader, device, max_batches=eval_batches)
        
        # Calculate on test set
        final_test_mse, final_test_psnr, final_test_sam, _ = calculate_metrics_in_batches(
            model, test_loader, device, max_batches=eval_batches)
    
    # Update final metrics to ensure accuracy
    metrics['train_loss'][-1] = final_train_mse
    metrics['test_loss'][-1] = final_test_mse
    metrics['train_psnr'][-1] = final_train_psnr
    metrics['test_psnr'][-1] = final_test_psnr
    metrics['train_sam'][-1] = final_train_sam
    metrics['test_sam'][-1] = final_test_sam
    
    # Save final detailed metrics
    print(f"\nFinal {shape_name} metrics at {noise_level} dB after {num_epochs} epochs:")
    print(f"  Train - MSE: {final_train_mse:.6f}, PSNR: {final_train_psnr:.2f} dB, SAM: {final_train_sam:.6f} rad")
    print(f"  Test  - MSE: {final_test_mse:.6f}, PSNR: {final_test_psnr:.2f} dB, SAM: {final_test_sam:.6f} rad")
    print(f"  Best model at epoch {best_epoch+1} - Test MSE: {best_test_loss:.6f}")
    print(f"  Filter condition number: {condition_number:.4f}")
    
    # Save model
    model_save_path = os.path.join(shape_dir, "decoder_model.pt")
    torch.save(model.decoder.state_dict(), model_save_path)
    
    # Save best model
    best_model_save_path = os.path.join(shape_dir, "best_decoder_model.pt")
    torch.save(best_model_state, best_model_save_path)
    
    # Plot training curves
    # Plot MSE
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs+1), metrics['train_loss'], 'b-', label='Train Loss')
    plt.plot(range(num_epochs+1), metrics['test_loss'], 'r-', label='Test Loss')
    plt.axvline(x=best_epoch, color='g', linestyle='--', 
                label=f'Best Epoch ({best_epoch}): {best_test_loss:.6f}')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title(f'{shape_name} Training/Test MSE at {noise_level} dB')
    plt.legend()
    plt.savefig(os.path.join(shape_dir, 'mse_curves.png'))
    plt.close()
    
    # Plot PSNR
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs+1), metrics['train_psnr'], 'b-', label='Train PSNR')
    plt.plot(range(num_epochs+1), metrics['test_psnr'], 'r-', label='Test PSNR')
    plt.axvline(x=best_epoch, color='g', linestyle='--', 
                label=f'Best Epoch ({best_epoch})')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.title(f'{shape_name} Training/Test PSNR at {noise_level} dB')
    plt.legend()
    plt.savefig(os.path.join(shape_dir, 'psnr_curves.png'))
    plt.close()
    
    # Plot SAM
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs+1), metrics['train_sam'], 'b-', label='Train SAM')
    plt.plot(range(num_epochs+1), metrics['test_sam'], 'r-', label='Test SAM')
    plt.axvline(x=best_epoch, color='g', linestyle='--', 
                label=f'Best Epoch ({best_epoch})')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('SAM (radians)')
    plt.title(f'{shape_name} Training/Test SAM at {noise_level} dB')
    plt.legend()
    plt.savefig(os.path.join(shape_dir, 'sam_curves.png'))
    plt.close()
    
    # Save metrics as numpy arrays
    np.save(os.path.join(shape_dir, 'train_loss.npy'), np.array(metrics['train_loss']))
    np.save(os.path.join(shape_dir, 'test_loss.npy'), np.array(metrics['test_loss']))
    np.save(os.path.join(shape_dir, 'train_psnr.npy'), np.array(metrics['train_psnr']))
    np.save(os.path.join(shape_dir, 'test_psnr.npy'), np.array(metrics['test_psnr']))
    np.save(os.path.join(shape_dir, 'train_sam.npy'), np.array(metrics['train_sam']))
    np.save(os.path.join(shape_dir, 'test_sam.npy'), np.array(metrics['test_sam']))
    
    # Save training summary
    with open(os.path.join(shape_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"Shape: {shape_name}\n")
        f.write(f"Noise level: {noise_level} dB\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Decoder learning rate: {decoder_lr}\n")
        f.write(f"Filter scale factor: {filter_scale_factor}\n")
        f.write(f"Visualization interval: {viz_interval_stage2} epochs\n")
        f.write(f"Evaluation batches limit: {eval_batches}\n\n")
        
        f.write(f"Filter condition number: {condition_number:.4f}\n\n")
        
        f.write(f"Initial metrics:\n")
        f.write(f"  Train - MSE: {metrics['train_loss'][0]:.6f}, PSNR: {metrics['train_psnr'][0]:.2f} dB, SAM: {metrics['train_sam'][0]:.6f} rad\n")
        f.write(f"  Test  - MSE: {metrics['test_loss'][0]:.6f}, PSNR: {metrics['test_psnr'][0]:.2f} dB, SAM: {metrics['test_sam'][0]:.6f} rad\n\n")
        
        f.write(f"Final metrics after {num_epochs} epochs:\n")
        f.write(f"  Train - MSE: {metrics['train_loss'][-1]:.6f}, PSNR: {metrics['train_psnr'][-1]:.2f} dB, SAM: {metrics['train_sam'][-1]:.6f} rad\n")
        f.write(f"  Test  - MSE: {metrics['test_loss'][-1]:.6f}, PSNR: {metrics['test_psnr'][-1]:.2f} dB, SAM: {metrics['test_sam'][-1]:.6f} rad\n\n")
        
        f.write(f"Best model at epoch {best_epoch+1}:\n")
        f.write(f"  Test MSE: {best_test_loss:.6f}\n\n")
        
        f.write(f"Improvements:\n")
        f.write(f"  Train MSE: {metrics['train_loss'][0] - metrics['train_loss'][-1]:.6f} ({(1 - metrics['train_loss'][-1]/metrics['train_loss'][0]) * 100:.2f}%)\n")
        f.write(f"  Test MSE: {metrics['test_loss'][0] - metrics['test_loss'][-1]:.6f} ({(1 - metrics['test_loss'][-1]/metrics['test_loss'][0]) * 100:.2f}%)\n")
        f.write(f"  Train PSNR: {metrics['train_psnr'][-1] - metrics['train_psnr'][0]:.2f} dB\n")
        f.write(f"  Test PSNR: {metrics['test_psnr'][-1] - metrics['test_psnr'][0]:.2f} dB\n")
        f.write(f"  Train SAM: {metrics['train_sam'][0] - metrics['train_sam'][-1]:.6f} rad ({(1 - metrics['train_sam'][-1]/metrics['train_sam'][0]) * 100:.2f}%)\n")
        f.write(f"  Test SAM: {metrics['test_sam'][0] - metrics['test_sam'][-1]:.6f} rad ({(1 - metrics['test_sam'][-1]/metrics['test_sam'][0]) * 100:.2f}%)\n")
    
    return metrics


###############################################################################
# TRAINING WITH FIXED SHAPES AT DIFFERENT NOISE LEVELS - BATCHED VERSION
###############################################################################
def train_multiple_fixed_shapes(shapes_dict, shape2filter_path, output_dir, 
                               noise_levels, num_epochs, batch_size, decoder_lr, 
                               filter_scale_factor, train_loader, test_loader,
                               viz_interval_stage2=1, eval_batches=None):
    """
    Train multiple fixed shapes at different noise levels
    
    Parameters:
    shapes_dict: Dictionary of shapes with paths
    shape2filter_path: Path to shape2filter model
    output_dir: Directory to save results
    noise_levels: List of noise levels (SNR in dB)
    num_epochs: Number of epochs for each training
    batch_size: Batch size for training
    decoder_lr: Learning rate for decoder
    filter_scale_factor: Scale factor for filter normalization
    train_loader: DataLoader for training data
    test_loader: DataLoader for test data
    viz_interval_stage2: Interval for visualization in stage 2
    eval_batches: Maximum number of batches to use for evaluation (None for all)
    
    Returns:
    dict: Dictionary with results for each shape and noise level
    """
    # Create directory for results
    results_dir = os.path.join(output_dir, "fixed_shape_comparison")
    os.makedirs(results_dir, exist_ok=True)
    
    # Results dictionary to store metrics for each shape and noise level
    results = {
        'train_loss': {shape_name: [] for shape_name in shapes_dict},
        'test_loss': {shape_name: [] for shape_name in shapes_dict},
        'train_psnr': {shape_name: [] for shape_name in shapes_dict},
        'test_psnr': {shape_name: [] for shape_name in shapes_dict},
        'train_sam': {shape_name: [] for shape_name in shapes_dict},
        'test_sam': {shape_name: [] for shape_name in shapes_dict},
        'all_metrics': {}
    }
    
    # Load shapes
    shapes = {}
    for shape_name, shape_path in shapes_dict.items():
        if os.path.exists(shape_path):
            shapes[shape_name] = np.load(shape_path)
            print(f"Loaded {shape_name} shape from: {shape_path}")
        else:
            print(f"Warning: {shape_name} shape not found at {shape_path}")
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train each shape with each noise level, organizing by noise level first
    for noise_level in noise_levels:
        print(f"\n{'='*50}")
        print(f"Training with noise level: {noise_level} dB")
        print(f"{'='*50}")
        
        # Create directory for this noise level
        noise_results_dir = os.path.join(results_dir, f"noise_{noise_level}dB")
        os.makedirs(noise_results_dir, exist_ok=True)
        
        # Track metrics for shapes at this noise level for comparison
        noise_metrics = {
            'train_loss': {},
            'test_loss': {},
            'train_psnr': {},
            'test_psnr': {},
            'train_sam': {},
            'test_sam': {}
        }
        
        # List to track condition numbers for each shape
        condition_numbers = {}
        
        # Train each shape
        for shape_name, shape in shapes.items():
            print(f"\nTraining {shape_name} shape with noise level {noise_level} dB")
            
            # Train the model
            metrics = train_with_fixed_shape(
                shape_name=shape_name,
                shape=shape,
                shape2filter_path=shape2filter_path,
                train_loader=train_loader,
                test_loader=test_loader,
                noise_level=noise_level,
                num_epochs=num_epochs,
                batch_size=batch_size,
                decoder_lr=decoder_lr,
                filter_scale_factor=filter_scale_factor,
                output_dir=noise_results_dir,
                viz_interval_stage2=viz_interval_stage2,
                eval_batches=eval_batches
            )
            
            # Save final metrics
            results['train_loss'][shape_name].append(metrics['train_loss'][-1])
            results['test_loss'][shape_name].append(metrics['test_loss'][-1])
            results['train_psnr'][shape_name].append(metrics['train_psnr'][-1])
            results['test_psnr'][shape_name].append(metrics['test_psnr'][-1])
            results['train_sam'][shape_name].append(metrics['train_sam'][-1])
            results['test_sam'][shape_name].append(metrics['test_sam'][-1])
            
            # Save metrics for comparison at this noise level
            noise_metrics['train_loss'][shape_name] = metrics['train_loss']
            noise_metrics['test_loss'][shape_name] = metrics['test_loss']
            noise_metrics['train_psnr'][shape_name] = metrics['train_psnr']
            noise_metrics['test_psnr'][shape_name] = metrics['test_psnr']
            noise_metrics['train_sam'][shape_name] = metrics['train_sam']
            noise_metrics['test_sam'][shape_name] = metrics['test_sam']
            
            # Also store all metrics for this shape and noise level
            results['all_metrics'][(shape_name, noise_level)] = metrics
            
            # Calculate condition number for this shape
            with torch.no_grad():
                # Create temporary model to get the filter
                temp_model = FixedShapeModel(
                    shape=shape,
                    shape2filter_path=shape2filter_path,
                    noise_level=noise_level,
                    filter_scale_factor=filter_scale_factor,
                    device=device
                )
                condition_number = calculate_condition_number(temp_model.fixed_filter.detach().cpu())
                condition_numbers[shape_name] = condition_number
                del temp_model  # Clean up memory
        
        # Create comparison plots for this noise level
        # Define a list of colors for each shape
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        shape_colors = {shape_name: colors[i % len(colors)] for i, shape_name in enumerate(shapes.keys())}

        # 1. MSE comparison
        plt.figure(figsize=(12, 8))
        for shape_name in shapes:
            color = shape_colors[shape_name]
            plt.plot(range(num_epochs+1), noise_metrics['train_loss'][shape_name], 
                    color=color, label=f"{shape_name.capitalize()} Train")
            plt.plot(range(num_epochs+1), noise_metrics['test_loss'][shape_name], 
                    color=color, linestyle='--', label=f"{shape_name.capitalize()} Test")
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.title(f'MSE Comparison at {noise_level} dB')
        plt.legend()
        plt.savefig(os.path.join(noise_results_dir, 'mse_comparison.png'))
        plt.close()
        
        # 2. PSNR comparison
        plt.figure(figsize=(12, 8))
        for shape_name in shapes:
            color = shape_colors[shape_name]
            plt.plot(range(num_epochs+1), noise_metrics['train_psnr'][shape_name], 
                    color=color, label=f"{shape_name.capitalize()} Train")
            plt.plot(range(num_epochs+1), noise_metrics['test_psnr'][shape_name], 
                    color=color, linestyle='--', label=f"{shape_name.capitalize()} Test")
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('PSNR (dB)')
        plt.title(f'PSNR Comparison at {noise_level} dB')
        plt.legend()
        plt.savefig(os.path.join(noise_results_dir, 'psnr_comparison.png'))
        plt.close()
        
        # 3. SAM comparison
        plt.figure(figsize=(12, 8))
        for shape_name in shapes:
            color = shape_colors[shape_name]
            plt.plot(range(num_epochs+1), noise_metrics['train_sam'][shape_name], 
                    color=color, label=f"{shape_name.capitalize()} Train")
            plt.plot(range(num_epochs+1), noise_metrics['test_sam'][shape_name], 
                    color=color, linestyle='--', label=f"{shape_name.capitalize()} Test")
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('SAM (radians)')
        plt.title(f'SAM Comparison at {noise_level} dB')
        plt.legend()
        plt.savefig(os.path.join(noise_results_dir, 'sam_comparison.png'))
        plt.close()
        
        # 4. Create condition number bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(list(condition_numbers.keys()), list(condition_numbers.values()), color='skyblue')
        plt.grid(True, axis='y')
        plt.xlabel('Shape Type')
        plt.ylabel('Condition Number')
        plt.title(f'Filter Condition Number Comparison at {noise_level} dB')
        for i, (shape_name, cn) in enumerate(condition_numbers.items()):
            plt.text(i, cn + 5, f'{cn:.2f}', ha='center')
        plt.savefig(os.path.join(noise_results_dir, 'condition_number_comparison.png'))
        plt.close()
        
        # 5. Create final metrics comparison table
        final_metrics = {
            'Shape': [],
            'Train MSE': [],
            'Test MSE': [],
            'Train PSNR': [],
            'Test PSNR': [],
            'Train SAM': [],
            'Test SAM': [],
            'Condition Number': []
        }
        
        for shape_name in shapes:
            final_metrics['Shape'].append(shape_name)
            final_metrics['Train MSE'].append(noise_metrics['train_loss'][shape_name][-1])
            final_metrics['Test MSE'].append(noise_metrics['test_loss'][shape_name][-1])
            final_metrics['Train PSNR'].append(noise_metrics['train_psnr'][shape_name][-1])
            final_metrics['Test PSNR'].append(noise_metrics['test_psnr'][shape_name][-1])
            final_metrics['Train SAM'].append(noise_metrics['train_sam'][shape_name][-1])
            final_metrics['Test SAM'].append(noise_metrics['test_sam'][shape_name][-1])
            final_metrics['Condition Number'].append(condition_numbers[shape_name])
        
        # Create table and save as CSV
        with open(os.path.join(noise_results_dir, 'final_metrics_comparison.csv'), 'w') as f:
            # Write header
            f.write(','.join(final_metrics.keys()) + '\n')
            
            # Write data
            for i in range(len(final_metrics['Shape'])):
                row = [
                    final_metrics['Shape'][i],
                    f"{final_metrics['Train MSE'][i]:.6f}",
                    f"{final_metrics['Test MSE'][i]:.6f}",
                    f"{final_metrics['Train PSNR'][i]:.2f}",
                    f"{final_metrics['Test PSNR'][i]:.2f}",
                    f"{final_metrics['Train SAM'][i]:.6f}",
                    f"{final_metrics['Test SAM'][i]:.6f}",
                    f"{final_metrics['Condition Number'][i]:.4f}"
                ]
                f.write(','.join(row) + '\n')
        
        # Print summary of results for this noise level
        print(f"\nSummary for noise level {noise_level} dB:")
        print(f"{'Shape':<10} | {'Train MSE':<10} | {'Test MSE':<10} | {'Train PSNR':<10} | {'Test PSNR':<10} | {'Train SAM':<10} | {'Test SAM':<10} | {'Condition #':<10}")
        print('-' * 94)
        
        for i in range(len(final_metrics['Shape'])):
            print(f"{final_metrics['Shape'][i]:<10} | "
                  f"{final_metrics['Train MSE'][i]:<10.6f} | "
                  f"{final_metrics['Test MSE'][i]:<10.6f} | "
                  f"{final_metrics['Train PSNR'][i]:<10.2f} | "
                  f"{final_metrics['Test PSNR'][i]:<10.2f} | "
                  f"{final_metrics['Train SAM'][i]:<10.6f} | "
                  f"{final_metrics['Test SAM'][i]:<10.6f} | "
                  f"{final_metrics['Condition Number'][i]:<10.2f}")
    
    # Create overall comparison plots across noise levels
    # 1. Final MSE comparison
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    for shape_name in shapes:
        plt.plot(noise_levels, results['train_loss'][shape_name], 'o-', 
                label=f"{shape_name.capitalize()}")
    plt.grid(True)
    plt.xlabel('Noise Level (SNR dB)')
    plt.ylabel('Final Train MSE')
    plt.title('Final Training MSE Across Noise Levels')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for shape_name in shapes:
        plt.plot(noise_levels, results['test_loss'][shape_name], 'o-', 
                label=f"{shape_name.capitalize()}")
    plt.grid(True)
    plt.xlabel('Noise Level (SNR dB)')
    plt.ylabel('Final Test MSE')
    plt.title('Final Testing MSE Across Noise Levels')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'final_mse_across_noise.png'))
    plt.close()
    
    # 2. Final PSNR comparison
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    for shape_name in shapes:
        plt.plot(noise_levels, results['train_psnr'][shape_name], 'o-', 
                label=f"{shape_name.capitalize()}")
    plt.grid(True)
    plt.xlabel('Noise Level (SNR dB)')
    plt.ylabel('Final Train PSNR (dB)')
    plt.title('Final Training PSNR Across Noise Levels')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for shape_name in shapes:
        plt.plot(noise_levels, results['test_psnr'][shape_name], 'o-', 
                label=f"{shape_name.capitalize()}")
    plt.grid(True)
    plt.xlabel('Noise Level (SNR dB)')
    plt.ylabel('Final Test PSNR (dB)')
    plt.title('Final Testing PSNR Across Noise Levels')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'final_psnr_across_noise.png'))
    plt.close()
    
    # 3. Final SAM comparison
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    for shape_name in shapes:
        plt.plot(noise_levels, results['train_sam'][shape_name], 'o-', 
                label=f"{shape_name.capitalize()}")
    plt.grid(True)
    plt.xlabel('Noise Level (SNR dB)')
    plt.ylabel('Final Train SAM (radians)')
    plt.title('Final Training SAM Across Noise Levels')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for shape_name in shapes:
        plt.plot(noise_levels, results['test_sam'][shape_name], 'o-', 
                label=f"{shape_name.capitalize()}")
    plt.grid(True)
    plt.xlabel('Noise Level (SNR dB)')
    plt.ylabel('Final Test SAM (radians)')
    plt.title('Final Testing SAM Across Noise Levels')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'final_sam_across_noise.png'))
    plt.close()
    
    # Save numeric results
    for shape_name in shapes:
        np.save(os.path.join(results_dir, f'{shape_name}_train_loss.npy'), np.array(results['train_loss'][shape_name]))
        np.save(os.path.join(results_dir, f'{shape_name}_test_loss.npy'), np.array(results['test_loss'][shape_name]))
        np.save(os.path.join(results_dir, f'{shape_name}_train_psnr.npy'), np.array(results['train_psnr'][shape_name]))
        np.save(os.path.join(results_dir, f'{shape_name}_test_psnr.npy'), np.array(results['test_psnr'][shape_name]))
        np.save(os.path.join(results_dir, f'{shape_name}_train_sam.npy'), np.array(results['train_sam'][shape_name]))
        np.save(os.path.join(results_dir, f'{shape_name}_test_sam.npy'), np.array(results['test_sam'][shape_name]))
    
    # Save noise levels
    np.save(os.path.join(results_dir, 'noise_levels.npy'), np.array(noise_levels))
    
    # Create summary csv
    with open(os.path.join(results_dir, 'summary.csv'), 'w') as f:
        # Write header
        f.write('Shape,Noise_Level,Train_MSE,Test_MSE,Train_PSNR,Test_PSNR,Train_SAM,Test_SAM\n')
        
        # Write data
        for i, noise_level in enumerate(noise_levels):
            for shape_name in shapes:
                f.write(f"{shape_name},{noise_level},{results['train_loss'][shape_name][i]:.6f},"
                        f"{results['test_loss'][shape_name][i]:.6f},{results['train_psnr'][shape_name][i]:.2f},"
                        f"{results['test_psnr'][shape_name][i]:.2f},{results['train_sam'][shape_name][i]:.6f},"
                        f"{results['test_sam'][shape_name][i]:.6f}\n")
    
    return results


###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hyperspectral autoencoder with blind noise, reconstruction visualization, and noise level comparison")
    
    # Data processing arguments
    parser.add_argument('--cache', type=str, default="cache_filtered/aviris_tiles_forest.pt", 
                       help="Path to cache file for processed data")
    parser.add_argument('--use-cache', action='store_true', help="Use cached data if available")
    parser.add_argument('-f', '--folders', type=str, default="all", 
                       help="Comma-separated list of folder name patterns to include, or 'all' for all folders")
    
    # Model arguments
    parser.add_argument('--gpu', type=int, default=None, help="GPU ID to use (e.g., 0, 1, 2, 3)")
    parser.add_argument('--filter-scale', type=float, default=10.0, 
                       help="Scaling factor for filter normalization (default: 10.0)")
    
    # Noise arguments
    parser.add_argument('--min-snr', type=float, default=10, 
                       help="Minimum SNR level in dB for random noise (default: 10)")
    parser.add_argument('--max-snr', type=float, default=40, 
                       help="Maximum SNR level in dB for random noise (default: 40)")
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument('--epochs', type=int, default=50, 
                       help="Number of epochs for full training (default: 50)")
    parser.add_argument('--decoder-epochs', type=int, default=20, 
                       help="Number of epochs for decoder-only training (default: 20)")
    parser.add_argument('--encoder-lr', type=float, default=0.001, 
                       help="Learning rate for encoder (default: 0.001)")
    parser.add_argument('--decoder-lr', type=float, default=0.001, 
                       help="Learning rate for decoder (default: 0.001)")
    
    # Visualization intervals
    parser.add_argument('--viz-interval-stage1', type=int, default=1,
                       help="Interval for visualization in stage 1 (default: 1)")
    parser.add_argument('--viz-interval-stage2', type=int, default=1,
                       help="Interval for visualization in stage 2 (default: 1)")
    
    # Experiment control
    parser.add_argument('--skip-stage1', action='store_true', 
                       help="Skip stage 1 and load shapes from directory")
    parser.add_argument('--load-shapes-dir', type=str, default=None, 
                       help="Directory to load shapes from when skipping stage 1")
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default=None, 
                       help="Output directory (default: blind_noise_experiment_[timestamp])")
    
    # Lazy loading arguments
    parser.add_argument('--lazy-load', action='store_true',
                       help="Use lazy loading for data to minimize memory usage")
    parser.add_argument('--data-file', type=str, default=None,
                       help="Path to pre-processed data file for lazy loading")
    parser.add_argument('--eval-batches', type=int, default=None,
                       help="Maximum number of batches to use for evaluation (None for all)")
    
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
    
    # Create base output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = f"blind_noise_experiment_{folder_suffix}_{timestamp}"
    else:
        base_output_dir = args.output_dir
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create cache directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    
    # Save command-line arguments to output directory
    with open(os.path.join(base_output_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Set device for generating initial parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate a unified initial filter to use across all experiments
    initial_filter_params = generate_initial_filter(device)
    
    # Save the initial filter for reference
    np.save(f"{base_output_dir}/unified_initial_filter.npy", initial_filter_params.detach().cpu().numpy())
    
    # Load and split data - either lazy or regular loading
    print("Setting up data loaders...")
    train_loader = None
    test_loader = None
    train_data = None
    test_data = None
    
    if args.lazy_load:
        # Use lazy loading
        print("Using lazy loading for data to minimize memory usage")
        
        # Determine data file
        data_file = args.data_file if args.data_file else cache_path
        if not os.path.exists(data_file):
            print(f"Data file {data_file} not found. Loading and saving data first...")
            data = load_aviris_forest_data(base_path="AVIRIS_FOREST_SIMPLE_SELECT", tile_size=128, 
                                         cache_file=cache_path, use_cache=args.use_cache, folder_patterns=folder_patterns)
            torch.save(data, cache_path)
            print(f"Data saved to {cache_path}")
            
            # Create lazy loaders from this data now in memory
            num_samples = data.shape[0]
            indices = torch.randperm(num_samples).tolist()
            train_size = int(0.7 * num_samples)
            
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]
            
            # Create train and test datasets
            train_dataset = TensorDataset(data[train_indices])
            test_dataset = TensorDataset(data[test_indices])
            
            # Create loaders
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
        else:
            # Create lazy data loaders using the data file
            train_loader, test_loader, _, _ = create_lazy_data_split(
                data_file, 
                train_ratio=0.7, 
                batch_size=args.batch_size,
                seed=42
            )
    else:
        # Regular loading - load all data into memory
        print("Loading all data into memory...")
        data = load_aviris_forest_data(base_path="AVIRIS_FOREST_SIMPLE_SELECT", tile_size=128, 
                                      cache_file=cache_path, use_cache=args.use_cache, folder_patterns=folder_patterns)
        
        # Verify data is in BHWC format
        if data.shape[1] == 100:  # If in BCHW
            data = data.permute(0, 2, 3, 1)  # Convert to BHWC
            print(f"Converted data to BHWC format: {data.shape}")
        
        # Split data into training and testing sets (70% train, 30% test)
        num_samples = data.shape[0]
        indices = torch.randperm(num_samples)
        train_size = int(0.7 * num_samples)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_data = data[train_indices]
        test_data = data[test_indices]
        
        print(f"Data split into {train_data.shape[0]} training and {test_data.shape[0]} testing samples")
        
        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(train_data), 
            batch_size=args.batch_size, 
            shuffle=True
        )
        
        test_loader = DataLoader(
            TensorDataset(test_data), 
            batch_size=args.batch_size, 
            shuffle=False
        )
    
    # Get shapes either from stage 1 or from provided directory
    shapes_dict = {}
    
    if args.skip_stage1:
        if args.load_shapes_dir is None:
            raise ValueError("Must provide --load-shapes-dir when using --skip-stage1")
        
        print(f"\n=== Skipping Stage 1, loading shapes from {args.load_shapes_dir} ===")
        
        # Try to load shapes from directory
        for shape_name in ['initial', 'lowest_mse', 'lowest_cn', 'final']:
            shape_path = os.path.join(args.load_shapes_dir, f"{shape_name}_shape.npy")
            if not os.path.exists(shape_path):
                # Try alternative path
                shape_path = os.path.join(args.load_shapes_dir, shape_name, "shape.npy")
            
            if os.path.exists(shape_path):
                shapes_dict[shape_name] = shape_path
                print(f"Found {shape_name} shape at: {shape_path}")
            else:
                print(f"Warning: Could not find {shape_name} shape")
        
        if not shapes_dict:
            raise ValueError(f"No shapes found in {args.load_shapes_dir}")
    
    else:
        # STAGE 1: Run training with random noise to generate shapes
        print(f"\n{'='*50}")
        print(f"STAGE 1: Training with random noise in range {args.min_snr} to {args.max_snr} dB")
        print(f"{'='*50}\n")
        
        initial_shape_path, lowest_mse_shape_path, lowest_cn_shape_path, final_shape_path, stage1_output_dir = train_with_random_noise(
            shape2filter_path=shape2filter_path,
            filter2shape_path=filter2shape_path,
            output_dir=base_output_dir,
            min_snr=args.min_snr,
            max_snr=args.max_snr,
            initial_filter_params=initial_filter_params,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            encoder_lr=args.encoder_lr,
            decoder_lr=args.decoder_lr,
            filter_scale_factor=args.filter_scale,
            cache_file=cache_path,
            use_cache=args.use_cache,
            folder_patterns=folder_patterns,
            train_data=train_data,
            test_data=test_data,
            viz_interval_stage1=args.viz_interval_stage1,
            train_loader=train_loader,
            test_loader=test_loader,
            eval_batches=args.eval_batches
        )
        
        # Create dictionary of shape paths for Stage 2
        shapes_dict = {
            'initial': initial_shape_path,
            'lowest_mse': lowest_mse_shape_path,
            'lowest_cn': lowest_cn_shape_path,
            'final': final_shape_path
        }
    
    # STAGE 2: Train with fixed shapes at different noise levels
    print(f"\n{'='*50}")
    print(f"STAGE 2: Training with fixed shapes at different noise levels")
    print(f"{'='*50}\n")
    
    # Define noise levels for fixed shape training
    fixed_noise_levels = [
        10, 
        # 15, 
        20, 
        # 25, 
        30, 
        # 35, 
        40
    ]
    
    # Run training for each shape at each noise level
    results = train_multiple_fixed_shapes(
        shapes_dict=shapes_dict,
        shape2filter_path=shape2filter_path,
        output_dir=base_output_dir,
        noise_levels=fixed_noise_levels,
        num_epochs=args.decoder_epochs,
        batch_size=args.batch_size,
        decoder_lr=args.decoder_lr,
        filter_scale_factor=args.filter_scale,
        train_loader=train_loader,
        test_loader=test_loader,
        viz_interval_stage2=args.viz_interval_stage2,
        eval_batches=args.eval_batches
    )
    
    print(f"\nAll experiments completed! Results saved to: {os.path.abspath(base_output_dir)}")

if __name__ == "__main__":
    main()