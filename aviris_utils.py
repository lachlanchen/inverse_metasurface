#!/usr/bin/env python3
"""
Utility functions for AVIRIS Fixed Shape Experiment
-------------------------------------------------
Common reusable functions for visualization, data processing and analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import random
from tqdm.notebook import tqdm
import numpy.linalg as LA

def set_seed(seed=42):
    """Set random seed for reproducibility across Python, NumPy, PyTorch and CUDA"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")

def calculate_condition_number(filters):
    """Calculate condition number of the spectral filters matrix"""
    # Convert to numpy for condition number calculation
    if isinstance(filters, torch.Tensor):
        filters_np = filters.detach().cpu().numpy()
    else:
        filters_np = filters
    
    # Use singular value decomposition to calculate condition number
    u, s, vh = LA.svd(filters_np)
    
    # Condition number is the ratio of largest to smallest singular value
    # Add small epsilon to prevent division by zero
    condition_number = s[0] / (s[-1] + 1e-10)
    
    return condition_number

def create_tiles(data, tile_size=256, overlap=0):
    """Create tiles from a large image"""
    # Check data shape and convert if necessary
    if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
        # Data is in (C, H, W) format, convert to (H, W, C)
        data = data.permute(1, 2, 0)
    
    H, W, C = data.shape
    tiles = []
    
    stride = tile_size - overlap
    for i in range(0, H - tile_size + 1, stride):
        for j in range(0, W - tile_size + 1, stride):
            tile = data[i:i+tile_size, j:j+tile_size, :]
            # Convert to (C, H, W) format for PyTorch
            tile = tile.permute(2, 0, 1)
            tiles.append(tile)
    
    return tiles

def visualize_reconstruction(model, data_loader, device, save_path=None, num_samples=4, return_fig=False):
    """Visualize original and reconstructed images"""
    model.eval()
    
    # Get samples from data loader
    x = next(iter(data_loader))[:num_samples].to(device)
    
    # Get reconstructions
    with torch.no_grad():
        x_recon, z = model(x, add_noise=False)
    
    # Move to CPU for visualization
    x = x.cpu()
    x_recon = x_recon.cpu()
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    
    for i in range(num_samples):
        # Select a random channel to visualize
        channel = random.randint(0, x.shape[1]-1)
        
        # Original
        im0 = axes[i, 0].imshow(x[i, channel], cmap='viridis')
        axes[i, 0].set_title(f"Original (Ch {channel})")
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Reconstructed
        im1 = axes[i, 1].imshow(x_recon[i, channel], cmap='viridis')
        axes[i, 1].set_title(f"Reconstructed (Ch {channel})")
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Difference
        diff = torch.abs(x[i, channel] - x_recon[i, channel])
        im2 = axes[i, 2].imshow(diff, cmap='hot')
        mse = torch.mean(diff**2).item()
        axes[i, 2].set_title(f"Difference (MSE: {mse:.6f})")
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if return_fig:
        return fig
    
    plt.show()
    plt.close()

def plot_loss_curves(train_losses, test_losses, save_path=None, return_fig=False):
    """Plot training and test loss curves"""
    fig = plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if return_fig:
        return fig
    
    plt.show()
    plt.close()

def plot_filters(filters, save_path=None, return_fig=False):
    """Plot filter shapes"""
    if isinstance(filters, torch.Tensor):
        filters = filters.detach().cpu().numpy()
    
    # Get number of filters
    num_filters = filters.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(1, num_filters, figsize=(2*num_filters, 3))
    
    # Plot each filter
    for i in range(num_filters):
        if num_filters > 1:
            ax = axes[i]
        else:
            ax = axes
            
        im = ax.plot(filters[i])
        ax.set_title(f"Filter {i+1}")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if return_fig:
        return fig
    
    plt.show()
    plt.close()

def visualize_filter_evolution(condition_numbers, save_path=None, return_fig=False):
    """Visualize the evolution of filter condition numbers during training"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Linear scale plot
    ax1.plot(condition_numbers, 'b-')
    ax1.set_title('Filter Condition Number Evolution')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Condition Number')
    ax1.grid(True, alpha=0.3)
    
    # Log scale plot
    ax2.semilogy(condition_numbers, 'r-')
    ax2.set_title('Filter Condition Number Evolution (Log Scale)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Condition Number (log scale)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if return_fig:
        return fig
    
    plt.show()
    plt.close()

def compare_shapes(shapes_dict, metrics_dict=None, save_path=None, return_fig=False):
    """Compare different filter shapes"""
    # Count shapes
    num_shapes = len(shapes_dict)
    shape_names = list(shapes_dict.keys())
    
    # Create figure
    fig, axes = plt.subplots(1, num_shapes, figsize=(4*num_shapes, 4))
    
    # Plot each shape
    for i, (name, shape) in enumerate(shapes_dict.items()):
        if isinstance(shape, torch.Tensor):
            shape = shape.detach().cpu().numpy()
        
        if num_shapes > 1:
            ax = axes[i]
        else:
            ax = axes
        
        # Plot the shape
        if shape.ndim > 1:
            # For 2D shapes (x,y coordinates)
            ax.scatter(shape[:, 0], shape[:, 1], s=10)
            ax.set_aspect('equal')
        else:
            # For 1D shapes
            ax.plot(shape)
        
        # Add title with metrics if available
        title = f"{name}"
        if metrics_dict and name in metrics_dict:
            metrics = metrics_dict[name]
            if 'condition_number' in metrics:
                title += f"\nCond #: {metrics['condition_number']:.2f}"
            if 'test_mse' in metrics:
                title += f"\nMSE: {metrics['test_mse']:.6f}"
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if return_fig:
        return fig
    
    plt.show()
    plt.close()

def compare_stage2_results(results_dict, output_dir=None, return_fig=False):
    """Compare results of Stage 2 training with different shapes"""
    # Extract data
    train_losses = {}
    test_losses = {}
    condition_numbers = {}
    
    for shape_name, results in results_dict.items():
        if 'train_losses' in results:
            train_losses[shape_name] = results['train_losses']
        if 'test_losses' in results:
            test_losses[shape_name] = results['test_losses']
        if 'condition_number' in results:
            condition_numbers[shape_name] = results['condition_number']
    
    # Setup output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Colors for different shapes
    colors = {
        'initial': 'blue',
        'lowest_condition_number': 'green',
        'lowest_test_mse': 'red',
        'final': 'purple',
        'random': 'gray'
    }
    
    # Create train/test × linear/log scale plots
    plot_configs = [
        {'data': train_losses, 'scale': 'linear', 'title': 'Training Loss (Linear Scale)', 'filename': 'train_losses_linear.png'},
        {'data': train_losses, 'scale': 'log', 'title': 'Training Loss (Log Scale)', 'filename': 'train_losses_log.png'},
        {'data': test_losses, 'scale': 'linear', 'title': 'Test Loss (Linear Scale)', 'filename': 'test_losses_linear.png'},
        {'data': test_losses, 'scale': 'log', 'title': 'Test Loss (Log Scale)', 'filename': 'test_losses_log.png'}
    ]
    
    figs = []
    
    # Create each plot
    for config in plot_configs:
        fig = plt.figure(figsize=(12, 8))
        
        # Add each shape's loss curve
        for shape_name, losses in config['data'].items():
            color = colors.get(shape_name, 'orange')
            x = range(1, len(losses) + 1)
            plt.plot(x, losses, label=f"{shape_name}", color=color, linewidth=2)
        
        plt.title(config['title'], fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('MSE Loss', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Set log scale if needed
        if config['scale'] == 'log':
            plt.yscale('log')
        
        # Save the plot
        if output_dir:
            plt.savefig(os.path.join(output_dir, config['filename']), dpi=300, bbox_inches='tight')
        
        figs.append(fig)
        
        if not return_fig:
            plt.show()
            plt.close()
    
    # Save data as numpy files
    if output_dir:
        np.save(os.path.join(output_dir, "train_losses.npy"), train_losses)
        np.save(os.path.join(output_dir, "test_losses.npy"), test_losses)
        if condition_numbers:
            np.save(os.path.join(output_dir, "condition_numbers.npy"), condition_numbers)
    
    if return_fig:
        return figs

def create_timestamp_dir(base_name, output_dir=None):
    """Create a directory with a timestamp suffix"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir:
        dir_name = os.path.join(output_dir, f"{base_name}_{timestamp}")
    else:
        dir_name = f"{base_name}_{timestamp}"
    
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def load_shapes_from_directory(load_shapes_dir):
    """Load recorded shapes from directory"""
    recorded_shapes = {}
    recorded_metrics = {}
    
    # Load shape files
    shape_files = {
        'initial': 'initial_shape.npy',
        'lowest_condition_number': 'lowest_condition_number_shape.npy',
        'lowest_test_mse': 'lowest_test_mse_shape.npy',
        'final': 'final_shape.npy'
    }
    
    for shape_name, file_name in shape_files.items():
        file_path = os.path.join(load_shapes_dir, file_name)
        if os.path.exists(file_path):
            shape_data = np.load(file_path)
            recorded_shapes[shape_name] = torch.tensor(shape_data, dtype=torch.float32)
            print(f"Loaded {shape_name} shape from {file_path}")
    
    # Load metrics if available
    metrics_file = os.path.join(load_shapes_dir, "shape_metrics.txt")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            
        current_shape = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.endswith("shape:"):
                current_shape = line.split()[0]
                if current_shape not in recorded_metrics:
                    recorded_metrics[current_shape] = {}
            elif current_shape and "Condition Number:" in line:
                recorded_metrics[current_shape]['condition_number'] = float(line.split(":")[-1].strip())
            elif current_shape and "Test MSE:" in line:
                recorded_metrics[current_shape]['test_mse'] = float(line.split(":")[-1].strip())
        
        print(f"Loaded metrics for {len(recorded_metrics)} shapes")
    else:
        print(f"Warning: Metrics file {metrics_file} not found. Using default values.")
        for shape_name in recorded_shapes:
            recorded_metrics[shape_name] = {
                'condition_number': 100.0,  # Default value
                'test_mse': 0.001  # Default value
            }
    
    return recorded_shapes, recorded_metrics

def calculate_model_size(model):
    """Calculate the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters())

def measure_inference_time(model, data_loader, device, num_runs=100):
    """Measure average inference time of the model"""
    model.eval()
    
    # Get a sample batch
    x = next(iter(data_loader)).to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            model(x, add_noise=False)
    
    # Measure time
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time.record()
            model(x, add_noise=False)
            end_time.record()
            
            # Synchronize CUDA operations
            torch.cuda.synchronize()
            
            # Append time in milliseconds
            times.append(start_time.elapsed_time(end_time))
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = np.std(times)
    
    results = {
        'avg_time_ms': avg_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'std_time_ms': std_time
    }
    
    return results