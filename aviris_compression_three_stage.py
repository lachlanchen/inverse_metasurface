#!/usr/bin/env python3
"""
AVIRIS Compression with Three-Stage Training Process
---------------------------------------------------------------------------------
This script implements a comprehensive three-stage training process:

Stage 1: Joint training of shape and decoder
- Based on aviris_compression_base.py
- Trains both shape and decoder simultaneously
- Records four key shapes: initial, lowest condition number, lowest MSE, final

Stage 2: Fixed shape training
- Based on aviris_fixed_shape_refactored.py
- For each recorded shape, fixes the shape and trains only the decoder
- Records performance metrics for each shape

Stage 3: Decoder freeze and shape re-optimization (INITIAL shape only)
- Freezes the decoder trained in stage 2
- Makes the shape optimizable again
- Fine-tunes only the shape parameters
- Records and visualizes the full training trajectory across all three stages

Usage:
    # Run all three stages
    python aviris_compression_three_stage.py --use_fsf --model awan --tile_size 100 --epochs 50 \
    --stage2_epochs 50 --stage3_epochs 25 --batch_size 64 --encoder_lr 1e-3 \
    --decoder_lr 5e-4 --min_snr 10 --max_snr 40 \
    --shape2filter_path "outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt" \
    --filter2shape_path "outputs_three_stage_20250322_145925/stageC/spec2shape_stageC.pt" \
    --filter_scale_factor 10.0
    
    # Skip stages as needed
    python aviris_compression_three_stage.py --use_fsf --model awan --tile_size 100 \
    --skip_stage1 --skip_stage2 --stage3_epochs 25 --batch_size 64 --encoder_lr 1e-4 \
    --min_snr 10 --max_snr 40 --shape2filter_path "outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt" \
    --load_dir results_three_stage_20250402_052223/ \
    --filter_scale_factor 10.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import random
from datetime import datetime
import numpy.linalg as LA
from copy import deepcopy

# Import common components from base files
from aviris_compression_base import (
    set_seed, calculate_condition_number, AvirisDataset, LinearEncoder, SimpleCNNDecoder,
    CompressionModel, create_tiles, process_and_cache_data, plot_loss_curves, 
    visualize_filter, visualize_filter_with_shape, visualize_reconstruction, plot_shape_with_c4
)

from aviris_fixed_shape_refactored import (
    FixedShapeEncoder, FixedShapeModel, visualize_shape, save_data_to_csv,
    load_shapes_from_directory
)

# Import AWAN if available
try:
    from AWAN import AWAN
except ImportError:
    print("Warning: Could not import AWAN, only CNN decoder will be available.")

# Import filter2shape2filter models and utilities
try:
    from filter2shape2filter_pipeline import (
        Shape2FilterModel, Filter2ShapeVarLen, create_pipeline, load_models,
        replicate_c4, sort_points_by_angle
    )
except ImportError:
    print("Warning: Could not import filter2shape2filter_pipeline.")


class ShapeOnlyModel(nn.Module):
    """Model with trainable shape encoder and frozen decoder"""
    def __init__(self, shape, in_channels=100, decoder_model=None, filter_scale_factor=50.0, device=None):
        super(ShapeOnlyModel, self).__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create a custom encoder with trainable shape parameters
        self.encoder = LinearEncoder(
            in_dim=in_channels,
            out_dim=shape.shape[0],  # This is the number of rows in the filter matrix
            use_fsf=True,
            shape2filter_path=None,  # Will be set manually
            filter2shape_path=None,  # Will be set manually
            filter_scale_factor=filter_scale_factor
        )
        
        # We need to initialize the filter_H tensor with learnable parameters that will
        # produce a shape similar to the input shape when passed through the pipeline
        # We'll do this after setting up the pipeline
        
        self.encoder.filter_scale_factor = filter_scale_factor
        
        # Set decoder as the pre-trained frozen decoder
        self.decoder = decoder_model
        
        # Freeze decoder weights
        for param in self.decoder.parameters():
            param.requires_grad = False
            
    def setup_pipeline(self, shape2filter, filter2shape, device):
        """Set up the filter2shape2filter pipeline with pretrained models"""
        self.encoder.pipeline = create_pipeline(shape2filter, filter2shape, no_grad_frozen=False)
        self.encoder.use_fsf = True
        
        # Now we can initialize the filter_H parameter
        # We'll use the shape2filter model to get a filter that produces our shape
        # Then we'll backpropagate to find a filter_H that gives this filter when passed through the pipeline
        
        # First, get a filter from the shape using shape2filter
        with torch.no_grad():
            target_filter = shape2filter(self.shape.unsqueeze(0))[0]
            
        # Initialize filter_H with random values
        filter_H = nn.Parameter(torch.randn(target_filter.shape, device=device))
        
        # Optimize filter_H to match target_filter when passed through the pipeline
        optimizer = optim.Adam([filter_H], lr=0.01)
        criterion = nn.MSELoss()
        
        for _ in range(100):  # Few iterations to get closer
            optimizer.zero_grad()
            _, reconstructed_filter = self.encoder.pipeline(filter_H.unsqueeze(0))
            loss = criterion(reconstructed_filter[0], target_filter)
            loss.backward()
            optimizer.step()
        
        # Now assign the optimized filter_H to the encoder
        self.encoder.filter_H = nn.Parameter(filter_H.detach().clone())
        
    def add_noise(self, z, min_snr_db=10, max_snr_db=40):
        """Add random noise with SNR between min_snr_db and max_snr_db"""
        batch_size = z.shape[0]
        # Random SNR for each image in batch
        snr_db = torch.rand(batch_size, 1, 1, 1, device=z.device) * (max_snr_db - min_snr_db) + min_snr_db
        snr = 10 ** (snr_db / 10)
        
        # Calculate signal power
        signal_power = torch.mean(z ** 2, dim=(1, 2, 3), keepdim=True)
        
        # Calculate noise power based on SNR
        noise_power = signal_power / snr
        
        # Generate Gaussian noise
        noise = torch.randn_like(z) * torch.sqrt(noise_power)
        
        # Add noise to signal
        z_noisy = z + noise
        
        return z_noisy
    
    def forward(self, x, add_noise=True, min_snr_db=10, max_snr_db=40):
        # Encode
        z = self.encoder(x)
        
        # Add noise if specified (during training)
        if add_noise:
            z = self.add_noise(z, min_snr_db, max_snr_db)
        
        # Decode with frozen decoder
        with torch.no_grad():
            x_recon = self.decoder(z)
        
        return x_recon, z


def train_model_stage1(model, train_loader, test_loader, args):
    """Train model in stage 1 and record key shapes - similar to base file"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Define separate optimizers for encoder and decoder
    encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=args.encoder_lr)
    decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=args.decoder_lr)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Initialize lists to store losses and condition numbers
    train_losses = []
    test_losses = []
    condition_numbers = []
    
    # Create directories for visualizations and recorded shapes
    stage1_dir = os.path.join(args.output_dir, "stage1")
    filter_dir = os.path.join(stage1_dir, "filter_evolution")
    recon_dir = os.path.join(stage1_dir, "reconstructions")
    shapes_dir = os.path.join(args.output_dir, "recorded_shapes")
    csv_dir = os.path.join(stage1_dir, "csv_data")
    
    os.makedirs(filter_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(shapes_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    print(f"Created output directories for Stage 1")
    
    # Dictionary to store recorded shapes
    recorded_shapes = {}
    recorded_metrics = {
        'initial': {'condition_number': float('inf'), 'test_mse': float('inf')},
        'lowest_condition_number': {'condition_number': float('inf'), 'test_mse': float('inf')},
        'lowest_test_mse': {'condition_number': float('inf'), 'test_mse': float('inf')},
        'final': {'condition_number': float('inf'), 'test_mse': float('inf')}
    }
    
    # Run a dummy forward pass to initialize shape
    if args.use_fsf and model.encoder.pipeline is not None:
        dummy_input = next(iter(train_loader))[:1].to(device)
        with torch.no_grad():
            model.encoder(dummy_input)
        
        # Record initial shape
        initial_shape = model.encoder.current_shape.clone()
        initial_filter_output = model.encoder.filter_output.clone()
        recorded_shapes['initial'] = initial_shape
        
        # Calculate condition number
        condition_number = calculate_condition_number(initial_filter_output)
        condition_numbers.append(condition_number)
        recorded_metrics['initial']['condition_number'] = condition_number
        
        # Save initial shape
        np_save_path = os.path.join(shapes_dir, "initial_shape.npy")
        np.save(np_save_path, initial_shape.detach().cpu().numpy())
        print(f"Saved initial shape to: {np_save_path}")
        
        # Save visualization of initial shape
        viz_save_path = os.path.join(shapes_dir, "initial_shape.png")
        visualize_shape(initial_shape, viz_save_path)
        
        # Visualize initial filter with shape
        filter_viz_path = os.path.join(filter_dir, "filter_initial.png")
        visualize_filter(
            model.encoder.filter_A.detach().cpu(),
            filter_viz_path,
            include_shape=True,
            shape_pred=model.encoder.current_shape,
            filter_output=model.encoder.filter_output
        )
        
        # Detailed initial filter-with-shape visualization
        filter_shape_path = os.path.join(filter_dir, "filter_with_shape_initial.png")
        visualize_filter_with_shape(
            model.encoder.filter_A.detach().cpu(),
            model.encoder.current_shape,
            model.encoder.filter_output,
            filter_shape_path
        )
        
        print(f"Recorded initial shape with condition number: {condition_number:.4f}")
    
    # Train for the specified number of epochs
    best_test_loss = float('inf')
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Stage 1 Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch_idx, x in enumerate(pbar):
                x = x.to(device)
                
                # Forward pass
                x_recon, z = model(x, add_noise=True, min_snr_db=args.min_snr, max_snr_db=args.max_snr)
                
                # Calculate loss
                loss = criterion(x_recon, x)
                
                # Backward pass and optimization
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                
                # Update progress bar
                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": epoch_loss / (batch_idx + 1)})
        
        # Calculate average epoch loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x in test_loader:
                x = x.to(device)
                x_recon, z = model(x, add_noise=False)
                loss = criterion(x_recon, x)
                test_loss += loss.item()
        
        # Calculate average test loss
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # Get updated shape and filter output
        if args.use_fsf and model.encoder.pipeline is not None:
            with torch.no_grad():
                dummy_input = next(iter(train_loader))[:1].to(device)
                model.encoder(dummy_input)
            
            current_shape = model.encoder.current_shape.clone()
            current_filter_output = model.encoder.filter_output.clone()
            
            # Calculate condition number
            current_condition_number = calculate_condition_number(current_filter_output)
            condition_numbers.append(current_condition_number)
            
            # Visualize filter evolution periodically
            if (epoch + 1) % args.viz_interval == 0 or epoch == args.epochs - 1:
                filter_viz_path = os.path.join(filter_dir, f"filter_epoch_{epoch+1}.png")
                visualize_filter(
                    model.encoder.filter_A.detach().cpu(),
                    filter_viz_path,
                    include_shape=True,
                    shape_pred=model.encoder.current_shape,
                    filter_output=model.encoder.filter_output
                )
                
                # Also create a detailed filter-with-shape visualization
                filter_shape_path = os.path.join(filter_dir, f"filter_with_shape_epoch_{epoch+1}.png")
                visualize_filter_with_shape(
                    model.encoder.filter_A.detach().cpu(),
                    model.encoder.current_shape,
                    model.encoder.filter_output,
                    filter_shape_path
                )
            
            # Check for lowest condition number
            if current_condition_number < recorded_metrics['lowest_condition_number']['condition_number']:
                recorded_shapes['lowest_condition_number'] = current_shape
                recorded_metrics['lowest_condition_number']['condition_number'] = current_condition_number
                recorded_metrics['lowest_condition_number']['test_mse'] = avg_test_loss
                
                # Save shape
                np_save_path = os.path.join(shapes_dir, "lowest_condition_number_shape.npy")
                np.save(np_save_path, current_shape.detach().cpu().numpy())
                print(f"Saved lowest condition number shape to: {np_save_path}")
                
                # Save visualization
                viz_save_path = os.path.join(shapes_dir, "lowest_condition_number_shape.png")
                visualize_shape(current_shape, viz_save_path)
                
                print(f"New lowest condition number: {current_condition_number:.4f}")
            
            # Check for lowest test MSE
            if avg_test_loss < recorded_metrics['lowest_test_mse']['test_mse']:
                recorded_shapes['lowest_test_mse'] = current_shape
                recorded_metrics['lowest_test_mse']['condition_number'] = current_condition_number
                recorded_metrics['lowest_test_mse']['test_mse'] = avg_test_loss
                
                # Save shape
                np_save_path = os.path.join(shapes_dir, "lowest_test_mse_shape.npy")
                np.save(np_save_path, current_shape.detach().cpu().numpy())
                print(f"Saved lowest test MSE shape to: {np_save_path}")
                
                # Save visualization
                viz_save_path = os.path.join(shapes_dir, "lowest_test_mse_shape.png")
                visualize_shape(current_shape, viz_save_path)
                
                print(f"New lowest test MSE: {avg_test_loss:.6f}")
        
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            model_save_path = os.path.join(stage1_dir, "best_model.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model with test loss: {best_test_loss:.6f} to: {model_save_path}")
        
        # Visualize reconstruction periodically
        if (epoch + 1) % args.viz_interval == 0 or epoch == args.epochs - 1:
            recon_save_path = os.path.join(recon_dir, f"recon_epoch_{epoch+1}.png")
            visualize_reconstruction(model, test_loader, device, recon_save_path)
    
    # Record final shape
    if args.use_fsf and model.encoder.pipeline is not None:
        final_shape = model.encoder.current_shape.clone()
        final_filter_output = model.encoder.filter_output.clone()
        recorded_shapes['final'] = final_shape
        
        # Calculate condition number
        final_condition_number = calculate_condition_number(final_filter_output)
        recorded_metrics['final']['condition_number'] = final_condition_number
        recorded_metrics['final']['test_mse'] = avg_test_loss
        
        # Save final shape
        np_save_path = os.path.join(shapes_dir, "final_shape.npy")
        np.save(np_save_path, final_shape.detach().cpu().numpy())
        print(f"Saved final shape to: {np_save_path}")
        
        # Save visualization
        viz_save_path = os.path.join(shapes_dir, "final_shape.png")
        visualize_shape(final_shape, viz_save_path)
        
        # Save final filter visualization
        final_filter_path = os.path.join(filter_dir, "filter_final.png")
        visualize_filter(
            model.encoder.filter_A.detach().cpu(),
            final_filter_path,
            include_shape=True,
            shape_pred=final_shape,
            filter_output=final_filter_output
        )
        
        # Detailed final filter-with-shape visualization
        final_filter_shape_path = os.path.join(filter_dir, "filter_with_shape_final.png")
        visualize_filter_with_shape(
            model.encoder.filter_A.detach().cpu(),
            final_shape,
            final_filter_output,
            final_filter_shape_path
        )
        
        print(f"Recorded final shape with condition number: {final_condition_number:.4f}")
    
    # Save metrics for all recorded shapes
    metrics_save_path = os.path.join(shapes_dir, "shape_metrics.txt")
    with open(metrics_save_path, 'w') as f:
        for shape_name, metrics in recorded_metrics.items():
            if shape_name in recorded_shapes:
                f.write(f"{shape_name} shape:\n")
                f.write(f"  Condition Number: {metrics['condition_number']:.4f}\n")
                f.write(f"  Test MSE: {metrics['test_mse']:.6f}\n\n")
    print(f"Saved shape metrics to: {metrics_save_path}")
    
    # Save condition numbers
    condition_path = os.path.join(stage1_dir, "condition_numbers.npy")
    np.save(condition_path, np.array(condition_numbers))
    print(f"Saved condition numbers to: {condition_path}")
    
    # Plot condition numbers
    if condition_numbers:
        # Linear scale plot
        plt.figure(figsize=(10, 6))
        plt.plot(condition_numbers, 'b-')
        plt.title('Filter Condition Number Evolution')
        plt.xlabel('Epochs')
        plt.ylabel('Condition Number')
        plt.grid(True, alpha=0.3)
        cond_plot_path = os.path.join(stage1_dir, "condition_number_evolution.png")
        plt.savefig(cond_plot_path, dpi=300)
        print(f"Saved condition number plot to: {cond_plot_path}")
        plt.close()

        # Log scale plot
        plt.figure(figsize=(10, 6))
        plt.semilogy(condition_numbers, 'r-')
        plt.title('Filter Condition Number Evolution (Log Scale)')
        plt.xlabel('Epochs')
        plt.ylabel('Condition Number (log scale)')
        plt.grid(True, alpha=0.3)
        cond_log_plot_path = os.path.join(stage1_dir, "condition_number_evolution_log.png")
        plt.savefig(cond_log_plot_path, dpi=300)
        print(f"Saved log-scale condition number plot to: {cond_log_plot_path}")
        plt.close()
    
    # Save train and test losses as CSV
    stage1_csv_path = os.path.join(csv_dir, "stage1_losses.csv")
    loss_data = np.column_stack((
        np.arange(1, len(train_losses) + 1),  # Epoch numbers
        np.array(train_losses),               # Train losses
        np.array(test_losses)                 # Test losses
    ))
    save_data_to_csv(loss_data, ["Epoch", "Train_Loss", "Test_Loss"], stage1_csv_path)
    
    # Save condition numbers as CSV
    condition_csv_path = os.path.join(csv_dir, "condition_numbers.csv")
    condition_data = np.column_stack((
        np.arange(len(condition_numbers)),    # Iteration/epoch numbers
        np.array(condition_numbers)           # Condition numbers
    ))
    save_data_to_csv(condition_data, ["Iteration", "Condition_Number"], condition_csv_path)
    
    # Save shape metrics as CSV
    metrics_csv_path = os.path.join(csv_dir, "shape_metrics.csv")
    metrics_data = []
    headers = ["Shape_Type", "Condition_Number", "Test_MSE"]
    
    for shape_name, metrics in recorded_metrics.items():
        if shape_name in recorded_shapes:
            metrics_data.append([
                shape_name,
                metrics['condition_number'],
                metrics['test_mse']
            ])
    
    save_data_to_csv(metrics_data, headers, metrics_csv_path)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(stage1_dir, "final_model.pt"))
    
    # Save the model object itself for later stages
    torch.save(model, os.path.join(stage1_dir, "model_object.pt"))
    
    return recorded_shapes, recorded_metrics, train_losses, test_losses, model


def train_with_fixed_shape(shape_name, shape, train_loader, test_loader, args):
    """Train model with fixed shape, optimizing only the decoder - from fixed shape refactored"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory for this shape
    stage2_dir = os.path.join(args.output_dir, "stage2")
    shape_dir = os.path.join(stage2_dir, f"{shape_name}")
    recon_dir = os.path.join(shape_dir, "reconstructions")
    csv_dir = os.path.join(shape_dir, "csv_data")
    os.makedirs(shape_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    print(f"Created output directory for {shape_name} shape: {shape_dir}")
    
    # Get input dimensions from data
    in_channels = next(iter(train_loader)).shape[1]
    
    # Create model with fixed shape
    model = FixedShapeModel(
        shape=shape,
        in_channels=in_channels,
        decoder_type=args.model,
        filter_scale_factor=args.filter_scale_factor,
        device=device
    )
    
    # Move model to device
    model = model.to(device)
    
    # Only optimize decoder parameters
    optimizer = optim.Adam(model.decoder.parameters(), lr=args.decoder_lr)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Initialize lists to store losses
    train_losses = []
    test_losses = []
    
    # Train for specified number of epochs
    best_test_loss = float('inf')
    print(f"\nTraining Stage 2 model with {shape_name} shape...")
    
    # First let's visualize the initial reconstruction before training
    recon_path = os.path.join(recon_dir, "recon_epoch_0.png")
    visualize_reconstruction(model, test_loader, device, recon_path)
    print(f"Saved initial reconstruction to: {recon_path}")
    
    # Evaluate initial loss
    model.eval()
    with torch.no_grad():
        initial_train_loss = 0
        for x in train_loader:
            x = x.to(device)
            x_recon, _ = model(x, add_noise=False)
            loss = criterion(x_recon, x)
            initial_train_loss += loss.item()
        initial_train_loss /= len(train_loader)
        
        initial_test_loss = 0
        for x in test_loader:
            x = x.to(device)
            x_recon, _ = model(x, add_noise=False)
            loss = criterion(x_recon, x)
            initial_test_loss += loss.item()
        initial_test_loss /= len(test_loader)
    
    train_losses.append(initial_train_loss)
    test_losses.append(initial_test_loss)
    print(f"Initial train loss: {initial_train_loss:.6f}, test loss: {initial_test_loss:.6f}")
    
    for epoch in range(args.stage2_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Stage 2 [{shape_name}] Epoch {epoch+1}/{args.stage2_epochs}") as pbar:
            for batch_idx, x in enumerate(pbar):
                x = x.to(device)
                
                # Forward pass
                x_recon, z = model(x, add_noise=True, min_snr_db=args.min_snr, max_snr_db=args.max_snr)
                
                # Calculate loss
                loss = criterion(x_recon, x)
                
                # Backward pass and optimize decoder only
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": epoch_loss / (batch_idx + 1)})
        
        # Calculate average epoch loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x in test_loader:
                x = x.to(device)
                x_recon, z = model(x, add_noise=False)
                loss = criterion(x_recon, x)
                test_loss += loss.item()
        
        # Calculate average test loss
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        print(f"[{shape_name}] Epoch {epoch+1}/{args.stage2_epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            model_path = os.path.join(shape_dir, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model with test loss: {best_test_loss:.6f} to: {model_path}")
            
            # Save best reconstruction visualization
            best_recon_path = os.path.join(shape_dir, "best_reconstruction.png")
            visualize_reconstruction(model, test_loader, device, best_recon_path)
            print(f"Saved best reconstruction visualization to: {best_recon_path}")
            
            # Save the complete model for stage 3
            if shape_name == 'initial':
                best_model_object_path = os.path.join(shape_dir, "best_model_object.pt")
                torch.save(model, best_model_object_path)
                print(f"Saved best model object for stage 3 to: {best_model_object_path}")
        
        # Visualize reconstruction periodically
        if (epoch + 1) % args.viz_interval == 0 or epoch == args.stage2_epochs - 1:
            recon_path = os.path.join(recon_dir, f"recon_epoch_{epoch+1}.png")
            visualize_reconstruction(model, test_loader, device, recon_path)
            print(f"Saved reconstruction for epoch {epoch+1} to: {recon_path}")
    
    # Save loss values and plots
    loss_path = os.path.join(shape_dir, "loss_values.npz")
    np.savez(loss_path, train_losses=np.array(train_losses), test_losses=np.array(test_losses))
    print(f"Saved loss values to: {loss_path}")
    
    # Save loss values as CSV
    loss_csv_path = os.path.join(csv_dir, "losses.csv")
    loss_data = np.column_stack((
        np.arange(len(train_losses)),  # Epoch numbers (starting at 0 for initial evaluation)
        np.array(train_losses),        # Train losses 
        np.array(test_losses)          # Test losses
    ))
    save_data_to_csv(loss_data, ["Epoch", "Train_Loss", "Test_Loss"], loss_csv_path)
    print(f"Saved loss values to CSV: {loss_csv_path}")
    
    # Plot loss curves
    plot_path = os.path.join(shape_dir, "loss_curves.png")
    plot_loss_curves(train_losses, test_losses, plot_path)
    print(f"Saved loss curves to: {plot_path}")
    
    # Save final model
    final_model_path = os.path.join(shape_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    
    # Save the complete model for stage 3
    if shape_name == 'initial':
        final_model_object_path = os.path.join(shape_dir, "final_model_object.pt")
        torch.save(model, final_model_object_path)
        print(f"Saved final model object for stage 3 to: {final_model_object_path}")
    
    print(f"Saved final model to: {final_model_path}")
    
    # Save summary of training results
    summary_path = os.path.join(shape_dir, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Training summary for {shape_name} shape\n")
        f.write("="*40 + "\n\n")
        f.write(f"Total epochs: {args.stage2_epochs}\n")
        f.write(f"Initial train loss: {train_losses[0]:.6f}\n")
        f.write(f"Initial test loss: {test_losses[0]:.6f}\n")
        f.write(f"Final train loss: {train_losses[-1]:.6f}\n")
        f.write(f"Final test loss: {test_losses[-1]:.6f}\n")
        f.write(f"Best test loss: {best_test_loss:.6f}\n")
        f.write(f"Train loss improvement: {(1 - train_losses[-1]/train_losses[0])*100:.2f}%\n")
        f.write(f"Test loss improvement: {(1 - test_losses[-1]/test_losses[0])*100:.2f}%\n")
    print(f"Saved training summary to: {summary_path}")
    
    # Save training summary as CSV
    summary_csv_path = os.path.join(csv_dir, "training_summary.csv")
    summary_data = [
        ["Shape_Type", shape_name],
        ["Total_Epochs", args.stage2_epochs],
        ["Initial_Train_Loss", train_losses[0]],
        ["Initial_Test_Loss", test_losses[0]],
        ["Final_Train_Loss", train_losses[-1]],
        ["Final_Test_Loss", test_losses[-1]],
        ["Best_Test_Loss", best_test_loss],
        ["Train_Improvement_Pct", (1 - train_losses[-1]/train_losses[0])*100],
        ["Test_Improvement_Pct", (1 - test_losses[-1]/test_losses[0])*100]
    ]
    save_data_to_csv(summary_data, ["Metric", "Value"], summary_csv_path)
    print(f"Saved training summary to CSV: {summary_csv_path}")
    
    print(f"Stage 2 training for {shape_name} shape complete!")
    
    return train_losses, test_losses, model


def train_with_frozen_decoder(shape, train_loader, test_loader, pretrained_model, args):
    """
    Train the model with frozen decoder and trainable shape (Stage 3)
    This is performed only for the initial shape
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory for stage 3
    stage3_dir = os.path.join(args.output_dir, "stage3")
    recon_dir = os.path.join(stage3_dir, "reconstructions")
    filter_dir = os.path.join(stage3_dir, "filter_evolution")
    csv_dir = os.path.join(stage3_dir, "csv_data")
    os.makedirs(stage3_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(filter_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    print(f"Created output directory for Stage 3: {stage3_dir}")
    
    # Extract the pretrained decoder from the input model
    pretrained_decoder = deepcopy(pretrained_model.decoder)
    
    # Get input dimensions from data
    in_channels = next(iter(train_loader)).shape[1]
    
    # For Stage 3, we'll take a different approach
    # Instead of directly using the ShapeOnlyModel, we'll build on top of the CompressionModel
    
    # First, create a fresh model
    stage3_model = CompressionModel(
        in_channels=in_channels, 
        latent_dim=args.latent_dim, 
        decoder_type=args.model,
        use_fsf=args.use_fsf,
        shape2filter_path=args.shape2filter_path,
        filter2shape_path=args.filter2shape_path,
        filter_scale_factor=args.filter_scale_factor
    )
    
    # Now replace the decoder with the pretrained one and freeze it
    stage3_model.decoder = pretrained_decoder
    for param in stage3_model.decoder.parameters():
        param.requires_grad = False
    
    # Initialize the encoder's filter_H with values that will produce the shape we want
    # We need to run a forward pass through the model first
    if args.use_fsf and stage3_model.encoder.pipeline is not None:
        # Initialize with the stage 2 shape as a starting point
        # We'll use a dummy forward pass to get the current shape
        dummy_input = next(iter(train_loader))[:1].to(device)
        
        # First get original filter matrix from pretrained model if available
        if hasattr(pretrained_model, 'encoder') and hasattr(pretrained_model.encoder, 'fixed_filter'):
            # If the model has a fixed_filter attribute, it's from Stage 2 (FixedShapeModel)
            filter_matrix = pretrained_model.encoder.fixed_filter.clone()
            
            # To initialize our filter_H, we need to reverse-engineer it from the filter_matrix
            # For simplicity, we'll just use the filter matrix directly, which should be close enough
            stage3_model.encoder.filter_H = nn.Parameter(filter_matrix.clone())
        else:
            # Otherwise try to generate a filter_H that would produce the shape we want
            # Move model to device for processing
            stage3_model = stage3_model.to(device)
            
            # Force it to have the right shape as a starting point
            # First let's move the shape to the device
            shape = shape.to(device)
            
            # Use shape2filter to get a filter from the shape
            shape2filter = stage3_model.encoder.pipeline.shape2filter
            with torch.no_grad():
                target_filter = shape2filter(shape.unsqueeze(0))[0]
            
            # Initialize filter_H with this target filter
            stage3_model.encoder.filter_H = nn.Parameter(target_filter.clone())
    
    # Move model to device
    stage3_model = stage3_model.to(device)
    
    # Define optimizer for encoder only (since decoder is frozen)
    optimizer = optim.Adam(stage3_model.encoder.parameters(), lr=args.encoder_lr)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Initialize lists to store losses and condition numbers
    train_losses = []
    test_losses = []
    condition_numbers = []
    
    # Evaluate initial model
    stage3_model.eval()
    with torch.no_grad():
        # Run a dummy forward pass to get shape and filter output
        dummy_input = next(iter(train_loader))[:1].to(device)
        _, _ = stage3_model(dummy_input, add_noise=False)
        
        if stage3_model.encoder.pipeline is not None:
            # Save initial filter visualizations
            shape_pred = stage3_model.encoder.current_shape
            filter_output = stage3_model.encoder.filter_output
            
            initial_filter_path = os.path.join(filter_dir, "filter_initial.png")
            visualize_filter(
                stage3_model.encoder.filter_A.detach().cpu(),
                initial_filter_path,
                include_shape=True,
                shape_pred=shape_pred,
                filter_output=filter_output
            )
            
            # Detailed visualization
            filter_shape_path = os.path.join(filter_dir, "filter_with_shape_initial.png")
            visualize_filter_with_shape(
                stage3_model.encoder.filter_A.detach().cpu(),
                shape_pred,
                filter_output,
                filter_shape_path
            )
            
            # Calculate condition number
            condition_number = calculate_condition_number(filter_output)
            condition_numbers.append(condition_number)
            print(f"Initial condition number: {condition_number:.4f}")
        
        # Evaluate initial loss
        initial_train_loss = 0
        for x in train_loader:
            x = x.to(device)
            x_recon, _ = stage3_model(x, add_noise=False)
            loss = criterion(x_recon, x)
            initial_train_loss += loss.item()
        initial_train_loss /= len(train_loader)
        
        initial_test_loss = 0
        for x in test_loader:
            x = x.to(device)
            x_recon, _ = stage3_model(x, add_noise=False)
            loss = criterion(x_recon, x)
            initial_test_loss += loss.item()
        initial_test_loss /= len(test_loader)
    
    train_losses.append(initial_train_loss)
    test_losses.append(initial_test_loss)
    print(f"Initial train loss: {initial_train_loss:.6f}, test loss: {initial_test_loss:.6f}")
    
    # Save initial reconstruction
    recon_path = os.path.join(recon_dir, "recon_epoch_0.png")
    visualize_reconstruction(stage3_model, test_loader, device, recon_path)
    print(f"Saved initial reconstruction to: {recon_path}")
    
    # Train for specified number of epochs
    best_test_loss = initial_test_loss
    for epoch in range(args.stage3_epochs):
        # Training phase
        stage3_model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Stage 3 Epoch {epoch+1}/{args.stage3_epochs}") as pbar:
            for batch_idx, x in enumerate(pbar):
                x = x.to(device)
                
                # Forward pass
                x_recon, z = stage3_model(x, add_noise=True, min_snr_db=args.min_snr, max_snr_db=args.max_snr)
                
                # Calculate loss
                loss = criterion(x_recon, x)
                
                # Backward pass and optimize encoder only
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": epoch_loss / (batch_idx + 1)})
        
        # Calculate average epoch loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        stage3_model.eval()
        test_loss = 0
        with torch.no_grad():
            for x in test_loader:
                x = x.to(device)
                x_recon, z = stage3_model(x, add_noise=False)
                loss = criterion(x_recon, x)
                test_loss += loss.item()
        
        # Calculate average test loss
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # Save and visualize current shape and filter
        if stage3_model.encoder.pipeline is not None:
            # Get updated shape and filter
            dummy_input = next(iter(train_loader))[:1].to(device)
            with torch.no_grad():
                stage3_model.encoder(dummy_input)
            
            shape_pred = stage3_model.encoder.current_shape
            filter_output = stage3_model.encoder.filter_output
            
            # Calculate condition number
            condition_number = calculate_condition_number(filter_output)
            condition_numbers.append(condition_number)
            
            # Visualize periodically
            if (epoch + 1) % args.viz_interval == 0 or epoch == args.stage3_epochs - 1:
                filter_viz_path = os.path.join(filter_dir, f"filter_epoch_{epoch+1}.png")
                visualize_filter(
                    stage3_model.encoder.filter_A.detach().cpu(),
                    filter_viz_path,
                    include_shape=True,
                    shape_pred=shape_pred,
                    filter_output=filter_output
                )
                
                # Detailed visualization
                filter_shape_path = os.path.join(filter_dir, f"filter_with_shape_epoch_{epoch+1}.png")
                visualize_filter_with_shape(
                    stage3_model.encoder.filter_A.detach().cpu(),
                    shape_pred,
                    filter_output,
                    filter_shape_path
                )
        
        print(f"Stage 3 Epoch {epoch+1}/{args.stage3_epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}, CN: {condition_number:.4f}")
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            model_path = os.path.join(stage3_dir, "best_model.pt")
            torch.save(stage3_model.state_dict(), model_path)
            
            # Also save the full model
            best_model_object_path = os.path.join(stage3_dir, "best_model_object.pt")
            torch.save(stage3_model, best_model_object_path)
            
            print(f"Saved new best model with test loss: {best_test_loss:.6f}")
            
            # Save best reconstruction visualization
            best_recon_path = os.path.join(stage3_dir, "best_reconstruction.png")
            visualize_reconstruction(stage3_model, test_loader, device, best_recon_path)
            print(f"Saved best reconstruction visualization")
        
        # Visualize reconstruction periodically
        if (epoch + 1) % args.viz_interval == 0 or epoch == args.stage3_epochs - 1:
            recon_path = os.path.join(recon_dir, f"recon_epoch_{epoch+1}.png")
            visualize_reconstruction(stage3_model, test_loader, device, recon_path)
            print(f"Saved reconstruction for epoch {epoch+1}")
    
    # Save final model
    final_model_path = os.path.join(stage3_dir, "final_model.pt")
    torch.save(stage3_model.state_dict(), final_model_path)
    
    # Also save the full model
    final_model_object_path = os.path.join(stage3_dir, "final_model_object.pt")
    torch.save(stage3_model, final_model_object_path)
    
    # Save loss values as CSV
    loss_csv_path = os.path.join(csv_dir, "losses.csv")
    loss_data = np.column_stack((
        np.arange(len(train_losses)),
        np.array(train_losses),
        np.array(test_losses)
    ))
    save_data_to_csv(loss_data, ["Epoch", "Train_Loss", "Test_Loss"], loss_csv_path)
    
    # Save condition numbers as CSV
    condition_csv_path = os.path.join(csv_dir, "condition_numbers.csv")
    condition_data = np.column_stack((
        np.arange(len(condition_numbers)),
        np.array(condition_numbers)
    ))
    save_data_to_csv(condition_data, ["Epoch", "Condition_Number"], condition_csv_path)
    
    # Plot loss curves
    plot_path = os.path.join(stage3_dir, "loss_curves.png")
    plot_loss_curves(train_losses, test_losses, plot_path)
    
    # Plot condition number curves
    if condition_numbers:
        # Linear scale
        plt.figure(figsize=(10, 6))
        plt.plot(condition_numbers, 'b-')
        plt.title('Stage 3: Filter Condition Number Evolution')
        plt.xlabel('Epochs')
        plt.ylabel('Condition Number')
        plt.grid(True, alpha=0.3)
        cond_plot_path = os.path.join(stage3_dir, "condition_number_evolution.png")
        plt.savefig(cond_plot_path, dpi=300)
        plt.close()
        
        # Log scale
        plt.figure(figsize=(10, 6))
        plt.semilogy(condition_numbers, 'r-')
        plt.title('Stage 3: Filter Condition Number Evolution (Log Scale)')
        plt.xlabel('Epochs')
        plt.ylabel('Condition Number (log scale)')
        plt.grid(True, alpha=0.3)
        cond_log_plot_path = os.path.join(stage3_dir, "condition_number_evolution_log.png")
        plt.savefig(cond_log_plot_path, dpi=300)
        plt.close()
    
    print(f"Stage 3 training complete!")
    
    return train_losses, test_losses, condition_numbers, stage3_model

def plot_combined_stages(stage1_train, stage1_test, stage2_train, stage2_test, 
                       stage3_train, stage3_test, stage1_condition_numbers, 
                       stage3_condition_numbers, args):
    """Create combined plots for all three stages"""
    combined_dir = os.path.join(args.output_dir, "combined_results")
    csv_dir = os.path.join(combined_dir, "csv_data")
    os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    print(f"Created combined results directory: {combined_dir}")
    
    # Calculate total epochs
    stage1_epochs = len(stage1_train)
    stage2_epochs = len(stage2_train)
    stage3_epochs = len(stage3_train)
    
    # Create combined x-axis with vertical lines marking stage transitions
    combined_epochs = stage1_epochs + stage2_epochs + stage3_epochs
    stage1_end = stage1_epochs
    stage2_end = stage1_epochs + stage2_epochs
    
    # Create x-axis points for each stage
    x1 = np.arange(1, stage1_epochs + 1)
    x2 = np.arange(stage1_epochs + 1, stage1_epochs + stage2_epochs + 1)
    x3 = np.arange(stage1_epochs + stage2_epochs + 1, combined_epochs + 1)
    
    # Combined train/test loss plot
    plt.figure(figsize=(15, 8))
    
    # Plot each stage with different colors and markers
    plt.plot(x1, stage1_train, 'b-', label='Stage 1 (Train)', linewidth=2)
    plt.plot(x1, stage1_test, 'b--', label='Stage 1 (Test)', linewidth=2)
    plt.plot(x2, stage2_train, 'g-', label='Stage 2 (Train)', linewidth=2)
    plt.plot(x2, stage2_test, 'g--', label='Stage 2 (Test)', linewidth=2)
    plt.plot(x3, stage3_train, 'r-', label='Stage 3 (Train)', linewidth=2)
    plt.plot(x3, stage3_test, 'r--', label='Stage 3 (Test)', linewidth=2)
    
    # Add vertical lines at stage transitions
    plt.axvline(x=stage1_end, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=stage2_end, color='k', linestyle='--', alpha=0.5)
    
    # Add text annotations for stages
    plt.text(stage1_epochs/2, plt.ylim()[1]*0.9, "Stage 1: Joint Training", 
             horizontalalignment='center', fontsize=12)
    plt.text(stage1_epochs + stage2_epochs/2, plt.ylim()[1]*0.9, "Stage 2: Fixed Shape", 
             horizontalalignment='center', fontsize=12)
    plt.text(stage1_epochs + stage2_epochs + stage3_epochs/2, plt.ylim()[1]*0.9, "Stage 3: Fixed Decoder", 
             horizontalalignment='center', fontsize=12)
    
    plt.title('Three-Stage Training: Loss Evolution', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss (MSE)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=12)
    
    # Save combined loss plot
    combined_loss_path = os.path.join(combined_dir, "combined_loss_evolution.png")
    plt.tight_layout()
    plt.savefig(combined_loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a log-scale version
    plt.figure(figsize=(15, 8))
    
    # Plot each stage with different colors and markers (log scale)
    plt.semilogy(x1, stage1_train, 'b-', label='Stage 1 (Train)', linewidth=2)
    plt.semilogy(x1, stage1_test, 'b--', label='Stage 1 (Test)', linewidth=2)
    plt.semilogy(x2, stage2_train, 'g-', label='Stage 2 (Train)', linewidth=2)
    plt.semilogy(x2, stage2_test, 'g--', label='Stage 2 (Test)', linewidth=2)
    plt.semilogy(x3, stage3_train, 'r-', label='Stage 3 (Train)', linewidth=2)
    plt.semilogy(x3, stage3_test, 'r--', label='Stage 3 (Test)', linewidth=2)
    
    # Add vertical lines at stage transitions
    plt.axvline(x=stage1_end, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=stage2_end, color='k', linestyle='--', alpha=0.5)
    
    # Add text annotations for stages
    plt.text(stage1_epochs/2, plt.ylim()[1]*0.9, "Stage 1: Joint Training", 
             horizontalalignment='center', fontsize=12)
    plt.text(stage1_epochs + stage2_epochs/2, plt.ylim()[1]*0.9, "Stage 2: Fixed Shape", 
             horizontalalignment='center', fontsize=12)
    plt.text(stage1_epochs + stage2_epochs + stage3_epochs/2, plt.ylim()[1]*0.9, "Stage 3: Fixed Decoder", 
             horizontalalignment='center', fontsize=12)
    
    plt.title('Three-Stage Training: Loss Evolution (Log Scale)', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss (MSE)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=12)
    
    # Save combined log-scale loss plot
    combined_log_loss_path = os.path.join(combined_dir, "combined_loss_evolution_log.png")
    plt.tight_layout()
    plt.savefig(combined_log_loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined condition number plot (Stage 1 and 3)
    if stage1_condition_numbers and stage3_condition_numbers:
        plt.figure(figsize=(15, 8))
        
        # Create x-axis points for condition numbers (Stage 1 and 3 only)
        cond_x1 = np.arange(len(stage1_condition_numbers))
        cond_x3 = np.arange(stage1_epochs + stage2_epochs, stage1_epochs + stage2_epochs + len(stage3_condition_numbers))
        
        # Plot condition numbers
        plt.plot(cond_x1, stage1_condition_numbers, 'b-', label='Stage 1', linewidth=2)
        plt.plot(cond_x3, stage3_condition_numbers, 'r-', label='Stage 3', linewidth=2)
        
        # Add vertical lines at stage transitions
        plt.axvline(x=stage1_end, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=stage2_end, color='k', linestyle='--', alpha=0.5)
        
        plt.title('Three-Stage Training: Condition Number Evolution', fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Condition Number', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=12)
        
        # Save combined condition number plot
        combined_cond_path = os.path.join(combined_dir, "combined_condition_evolution.png")
        plt.tight_layout()
        plt.savefig(combined_cond_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a log-scale version
        plt.figure(figsize=(15, 8))
        
        # Plot condition numbers (log scale)
        plt.semilogy(cond_x1, stage1_condition_numbers, 'b-', label='Stage 1', linewidth=2)
        plt.semilogy(cond_x3, stage3_condition_numbers, 'r-', label='Stage 3', linewidth=2)
        
        # Add vertical lines at stage transitions
        plt.axvline(x=stage1_end, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=stage2_end, color='k', linestyle='--', alpha=0.5)
        
        plt.title('Three-Stage Training: Condition Number Evolution (Log Scale)', fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Condition Number', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=12)
        
        # Save combined log-scale condition number plot
        combined_log_cond_path = os.path.join(combined_dir, "combined_condition_evolution_log.png")
        plt.tight_layout()
        plt.savefig(combined_log_cond_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save combined data as CSV
    combined_train_csv = os.path.join(csv_dir, "combined_train_losses.csv")
    combined_test_csv = os.path.join(csv_dir, "combined_test_losses.csv")
    
    # Prepare data
    combined_train_data = []
    combined_test_data = []
    
    # Stage 1
    for i, (train_loss, test_loss) in enumerate(zip(stage1_train, stage1_test)):
        combined_train_data.append([i+1, train_loss, "Stage 1"])
        combined_test_data.append([i+1, test_loss, "Stage 1"])
    
    # Stage 2
    for i, (train_loss, test_loss) in enumerate(zip(stage2_train, stage2_test)):
        epoch = stage1_epochs + i + 1
        combined_train_data.append([epoch, train_loss, "Stage 2"])
        combined_test_data.append([epoch, test_loss, "Stage 2"])
    
    # Stage 3
    for i, (train_loss, test_loss) in enumerate(zip(stage3_train, stage3_test)):
        epoch = stage1_epochs + stage2_epochs + i + 1
        combined_train_data.append([epoch, train_loss, "Stage 3"])
        combined_test_data.append([epoch, test_loss, "Stage 3"])
    
    # Save as CSV
    save_data_to_csv(combined_train_data, ["Epoch", "Loss", "Stage"], combined_train_csv)
    save_data_to_csv(combined_test_data, ["Epoch", "Loss", "Stage"], combined_test_csv)
    
    # Save condition numbers as CSV if available
    if stage1_condition_numbers and stage3_condition_numbers:
        combined_cond_csv = os.path.join(csv_dir, "combined_condition_numbers.csv")
        combined_cond_data = []
        
        # Stage 1
        for i, cond in enumerate(stage1_condition_numbers):
            combined_cond_data.append([i, cond, "Stage 1"])
        
        # Stage 3
        for i, cond in enumerate(stage3_condition_numbers):
            epoch = stage1_epochs + stage2_epochs + i
            combined_cond_data.append([epoch, cond, "Stage 3"])
        
        save_data_to_csv(combined_cond_data, ["Epoch", "Condition_Number", "Stage"], combined_cond_csv)
    
    print(f"Saved combined stage plots and data to: {combined_dir}")

def run_stage2(recorded_shapes, recorded_metrics, train_loader, test_loader, args):
    """Run stage 2 with all recorded shapes"""
    stage2_results = {
        'train_losses': {},
        'test_losses': {},
        'condition_numbers': {},
        'models': {}
    }
    
    # Create directory for combined CSV data
    stage2_dir = os.path.join(args.output_dir, "stage2")
    csv_dir = os.path.join(stage2_dir, "csv_data")
    os.makedirs(stage2_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    # Save shape metrics to CSV before training
    shape_metrics_csv = os.path.join(csv_dir, "shape_metrics_before_training.csv")
    shape_metrics_data = []
    for shape_name, metrics in recorded_metrics.items():
        if shape_name in recorded_shapes:
            shape_metrics_data.append([
                shape_name,
                metrics['condition_number'],
                metrics['test_mse']
            ])
    
    # Save shape metrics CSV
    save_data_to_csv(shape_metrics_data, 
                  ["Shape_Type", "Condition_Number", "Initial_Test_MSE"], 
                  shape_metrics_csv)
    print(f"Saved shape metrics CSV to: {shape_metrics_csv}")
    
    # Train with each recorded shape
    for shape_name, shape in recorded_shapes.items():
        print(f"\n=== Stage 2: Training with fixed {shape_name} shape ===")
        if shape_name in recorded_metrics:
            print(f"Shape condition number: {recorded_metrics[shape_name]['condition_number']:.4f}")
            stage2_results['condition_numbers'][shape_name] = recorded_metrics[shape_name]['condition_number']
        
        try:
            train_losses, test_losses, model = train_with_fixed_shape(
                shape_name, shape, train_loader, test_loader, args)
            
            stage2_results['train_losses'][shape_name] = train_losses
            stage2_results['test_losses'][shape_name] = test_losses
            stage2_results['models'][shape_name] = model
            print(f"Successfully completed training with {shape_name} shape")
        except Exception as e:
            print(f"Error training with {shape_name} shape: {e}")
    
    # Save final comparison metrics after all training
    final_metrics_csv = os.path.join(csv_dir, "shape_metrics_after_training.csv")
    final_metrics_data = []
    headers = ["Shape_Type", "Condition_Number", "Initial_Test_MSE", "Final_Test_MSE", "Best_Test_MSE", "Improvement_Percent"]
    
    for shape_name in recorded_shapes.keys():
        if shape_name in stage2_results['test_losses']:
            test_losses = stage2_results['test_losses'][shape_name]
            if test_losses:
                initial_test_mse = test_losses[0]
                final_test_mse = test_losses[-1]
                best_test_mse = min(test_losses)
                improvement_pct = (1 - final_test_mse/initial_test_mse) * 100
                
                condition_number = stage2_results['condition_numbers'].get(shape_name, float('nan'))
                
                final_metrics_data.append([
                    shape_name,
                    condition_number,
                    initial_test_mse,
                    final_test_mse,
                    best_test_mse,
                    improvement_pct
                ])
    
    # Save final metrics
    save_data_to_csv(final_metrics_data, headers, final_metrics_csv)
    print(f"Saved final comparison metrics to: {final_metrics_csv}")
    
    return stage2_results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AVIRIS Compression with Three-Stage Training Process')
    
    # Data processing arguments
    parser.add_argument('--tile_size', type=int, default=256, help='Tile size (default: 256)')
    parser.add_argument('--use_cache', type=str, default='cache_simple', help='Cache directory (default: cache_simple)')
    parser.add_argument('-f', '--folder', type=str, default='all', 
                        help='Subfolder of AVIRIS_SIMPLE_SELECT to process (or "all")')
    parser.add_argument('--force_cache', action='store_true', help='Force cache recreation even if it exists')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='awan', choices=['awan', 'cnn'],
                        help='Decoder model to use: awan or cnn (default: awan)')
    parser.add_argument('--latent_dim', type=int, default=11, help='Latent dimension (default: 11)')
    parser.add_argument('--min_snr', type=float, default=10, help='Minimum SNR in dB (default: 10)')
    parser.add_argument('--max_snr', type=float, default=40, help='Maximum SNR in dB (default: 40)')
    
    # Filter2Shape2Filter arguments
    parser.add_argument('--use_fsf', action='store_true', help='Use filter2shape2filter pipeline in encoder')
    parser.add_argument('--shape2filter_path', type=str, 
                        default="outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt",
                        help='Path to the shape2filter model weights')
    parser.add_argument('--filter2shape_path', type=str, 
                        default="outputs_three_stage_20250322_145925/stageC/spec2shape_stageC.pt",
                        help='Path to the filter2shape model weights')
    parser.add_argument('--filter_scale_factor', type=float, default=50.0,
                        help='Scale factor to divide FSF pipeline output by (default: 50.0)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for stage 1 (default: 50)')
    parser.add_argument('--stage2_epochs', type=int, default=50, 
                        help='Number of epochs for stage 2 with fixed shapes (default: 50)')
    parser.add_argument('--stage3_epochs', type=int, default=25,
                        help='Number of epochs for stage 3 with fixed decoder (default: 25)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--encoder_lr', type=float, default=1e-3, help='Encoder learning rate (default: 1e-3)')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='Decoder learning rate (default: 1e-4)')
    parser.add_argument('--test_split', type=float, default=0.2, help='Test split ratio (default: 0.2)')
    
    # Experiment control
    parser.add_argument('--skip_stage1', action='store_true', help='Skip stage 1 and load shapes from a directory')
    parser.add_argument('--skip_stage2', action='store_true', help='Skip stage 2 and only run stages 1 and 3')
    parser.add_argument('--skip_stage3', action='store_true', help='Skip stage 3 and only run stages 1 and 2')
    parser.add_argument('--load_dir', type=str, default=None, 
                        help='Directory to load shapes and models from when skipping stages')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory (default: results_three_stage_[model]_[timestamp])')
    parser.add_argument('--viz_interval', type=int, default=5, help='Visualization interval in epochs (default: 5)')
    
    args = parser.parse_args()
    
    # Add timestamp to output directory if not provided
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results_three_stage_{args.model}_{timestamp}"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Created output directory: {args.output_dir}")
    
    # Save arguments to a file for reference
    args_path = os.path.join(args.output_dir, 'args.txt')
    with open(args_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    print(f"Saved arguments to: {args_path}")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Process and cache data
    cache_file = process_and_cache_data(args)
    
    # Load cached data
    print(f"Loading cached tiles from: {cache_file}")
    tiles = torch.load(cache_file)
    print(f"Loaded {tiles.shape[0]} tiles with shape {tiles.shape[1:]} (C×H×W)")
    
    # Create dataset
    dataset = AvirisDataset(tiles)
    
    # Split into train and test sets
    test_size = int(len(dataset) * args.test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"Dataset split: {train_size} training samples, {test_size} test samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize variables for tracking
    recorded_shapes = {}
    recorded_metrics = {}
    stage1_losses = ([], [])  # (train, test)
    stage2_losses = ([], [])  # (train, test) - for initial shape only
    stage3_losses = ([], [])
    stage1_condition_numbers = []
    stage3_condition_numbers = []
    stage1_model = None
    stage2_model = None
    stage3_model = None
    
    # Stage 1: Joint training of shape and decoder
    if not args.skip_stage1:
        # Check if FSF pipeline models exist
        if args.use_fsf:
            if os.path.exists(args.shape2filter_path) and os.path.exists(args.filter2shape_path):
                print("Filter2Shape2Filter integration enabled for Stage 1.")
                print(f"Using filter scale factor: {args.filter_scale_factor}")
            else:
                print("Warning: FSF model paths not found. Disabling FSF integration.")
                args.use_fsf = False
        
        print("\n=== Stage 1: Joint training of shape and decoder ===")
        
        # Create model
        in_channels = tiles.shape[1]
        model = CompressionModel(
            in_channels=in_channels, 
            latent_dim=args.latent_dim, 
            decoder_type=args.model,
            use_fsf=args.use_fsf,
            shape2filter_path=args.shape2filter_path,
            filter2shape_path=args.filter2shape_path,
            filter_scale_factor=args.filter_scale_factor
        )
        
        print(f"Model initialized with:")
        print(f"- {in_channels} input channels")
        print(f"- {args.latent_dim} latent dimensions")
        print(f"- {args.model} decoder")
        print(f"- FSF pipeline: {'Enabled' if args.use_fsf else 'Disabled'}")
        
        # Train model and record shapes
        recorded_shapes, recorded_metrics, train_losses, test_losses, stage1_model = train_model_stage1(
            model, train_loader, test_loader, args)
        
        # Store losses and condition numbers for later visualization
        stage1_losses = (train_losses, test_losses)
        
        # Condition numbers should be calculated during Stage 1
        stage1_condition_numbers = np.load(os.path.join(args.output_dir, "stage1", "condition_numbers.npy")).tolist()
        
        print("\nStage 1 complete!")
        print(f"Recorded shapes: {list(recorded_shapes.keys())}")
        
    else:
        # Skip stage 1 and load shapes from directory
        if args.load_dir is None:
            raise ValueError("Must provide --load_dir when using --skip_stage1")
        
        shapes_dir = os.path.join(args.load_dir, "recorded_shapes")
        print(f"\n=== Skipping Stage 1, loading shapes from {shapes_dir} ===")
        recorded_shapes, recorded_metrics = load_shapes_from_directory(shapes_dir)
        
        # Try to load stage 1 losses and condition numbers if available
        try:
            stage1_losses_csv = os.path.join(args.load_dir, "stage1", "csv_data", "stage1_losses.csv")
            if os.path.exists(stage1_losses_csv):
                losses_data = np.loadtxt(stage1_losses_csv, delimiter=',', skiprows=1)
                train_losses = losses_data[:, 1].tolist()
                test_losses = losses_data[:, 2].tolist()
                stage1_losses = (train_losses, test_losses)
                print(f"Loaded Stage 1 losses from: {stage1_losses_csv}")
            
            condition_numbers_csv = os.path.join(args.load_dir, "stage1", "csv_data", "condition_numbers.csv")
            if os.path.exists(condition_numbers_csv):
                condition_data = np.loadtxt(condition_numbers_csv, delimiter=',', skiprows=1)
                stage1_condition_numbers = condition_data[:, 1].tolist()
                print(f"Loaded Stage 1 condition numbers from: {condition_numbers_csv}")
        except Exception as e:
            print(f"Could not load Stage 1 metrics: {e}")
    
    # Stage 2: Fixed shape training for all recorded shapes
    if not args.skip_stage2 and recorded_shapes:
        print("\n=== Stage 2: Training with fixed shapes ===")
        
        # Run stage 2 with all recorded shapes
        stage2_results = run_stage2(recorded_shapes, recorded_metrics, train_loader, test_loader, args)
        
        # Extract losses for initial shape for later combined visualization
        if 'initial' in stage2_results['train_losses']:
            stage2_losses = (
                stage2_results['train_losses']['initial'],
                stage2_results['test_losses']['initial']
            )
            # Also get the model for stage 3
            stage2_model = stage2_results['models']['initial']
        
        print("\nStage 2 complete!")
    else:
        # If skipping, try to load stage 2 losses for initial shape
        if args.load_dir is not None:
            try:
                stage2_initial_csv = os.path.join(args.load_dir, "stage2", "initial", "csv_data", "losses.csv")
                if os.path.exists(stage2_initial_csv):
                    losses_data = np.loadtxt(stage2_initial_csv, delimiter=',', skiprows=1)
                    train_losses = losses_data[:, 1].tolist()
                    test_losses = losses_data[:, 2].tolist()
                    stage2_losses = (train_losses, test_losses)
                    print(f"Loaded Stage 2 (initial shape) losses from: {stage2_initial_csv}")
                
                # Try to load model for stage 3
                model_path = os.path.join(args.load_dir, "stage2", "initial", "final_model_object.pt")
                if os.path.exists(model_path):
                    stage2_model = torch.load(model_path)
                    print(f"Loaded Stage 2 model from: {model_path}")
                else:
                    # Try best model
                    model_path = os.path.join(args.load_dir, "stage2", "initial", "best_model_object.pt")
                    if os.path.exists(model_path):
                        stage2_model = torch.load(model_path)
                        print(f"Loaded Stage 2 best model from: {model_path}")
            except Exception as e:
                print(f"Could not load Stage 2 metrics: {e}")
    
    # Stage 3: Fixed decoder training (only for initial shape)
    if not args.skip_stage3:
        if 'initial' not in recorded_shapes:
            raise ValueError("Cannot run Stage 3 without 'initial' shape. Make sure it was recorded in Stage 1.")
        
        if stage2_model is None and not args.skip_stage2:
            raise ValueError("Cannot run Stage 3 without a pre-trained Stage 2 model.")
        
        print("\n=== Stage 3: Training with frozen decoder (initial shape only) ===")
        
        # Get the initial shape and pre-trained Stage 2 model
        initial_shape = recorded_shapes['initial']
        
        # If we don't have a Stage 2 model and are skipping Stage 2, try to load from a file
        if stage2_model is None and args.load_dir is not None:
            model_path = os.path.join(args.load_dir, "stage2", "initial", "final_model_object.pt")
            if os.path.exists(model_path):
                try:
                    stage2_model = torch.load(model_path)
                    print(f"Loaded Stage 2 model from: {model_path}")
                except Exception as e:
                    print(f"Error loading Stage 2 model: {e}")
                    # Try best model
                    model_path = os.path.join(args.load_dir, "stage2", "initial", "best_model_object.pt")
                    if os.path.exists(model_path):
                        try:
                            stage2_model = torch.load(model_path)
                            print(f"Loaded Stage 2 best model from: {model_path}")
                        except Exception as e:
                            print(f"Error loading Stage 2 best model: {e}")
        
        if stage2_model is None:
            raise ValueError("Cannot run Stage 3 without a pre-trained Stage 2 model and --load_dir not specified or model not found.")
        
        # Run Stage 3 with frozen decoder
        train_losses, test_losses, condition_numbers, stage3_model = train_with_frozen_decoder(
            initial_shape, train_loader, test_loader, stage2_model, args)
        
        # Store for combined visualization
        stage3_losses = (train_losses, test_losses)
        stage3_condition_numbers = condition_numbers
        
        print("\nStage 3 complete!")
    else:
        # If skipping, try to load stage 3 losses
        if args.load_dir is not None:
            try:
                stage3_csv = os.path.join(args.load_dir, "stage3", "csv_data", "losses.csv")
                if os.path.exists(stage3_csv):
                    losses_data = np.loadtxt(stage3_csv, delimiter=',', skiprows=1)
                    train_losses = losses_data[:, 1].tolist()
                    test_losses = losses_data[:, 2].tolist()
                    stage3_losses = (train_losses, test_losses)
                    print(f"Loaded Stage 3 losses from: {stage3_csv}")
                
                # Try to load condition numbers
                condition_csv = os.path.join(args.load_dir, "stage3", "csv_data", "condition_numbers.csv")
                if os.path.exists(condition_csv):
                    condition_data = np.loadtxt(condition_csv, delimiter=',', skiprows=1)
                    stage3_condition_numbers = condition_data[:, 1].tolist()
                    print(f"Loaded Stage 3 condition numbers from: {condition_csv}")
            except Exception as e:
                print(f"Could not load Stage 3 metrics: {e}")
    

    # Create combined visualization for all three stages
    if stage1_losses[0] and stage2_losses[0] and stage3_losses[0]:
        print("\n=== Creating combined visualization for all three stages ===")
        plot_combined_stages(
            stage1_losses[0],  # Stage 1 train losses
            stage1_losses[1],  # Stage 1 test losses
            stage2_losses[0],  # Stage 2 train losses
            stage2_losses[1],  # Stage 2 test losses
            stage3_losses[0],  # Stage 3 train losses
            stage3_losses[1],  # Stage 3 test losses
            stage1_condition_numbers,
            stage3_condition_numbers,
            args
        )

    print(f"\nThree-stage experiment completed successfully!")
    print(f"All results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()