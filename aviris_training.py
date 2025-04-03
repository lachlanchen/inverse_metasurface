# aviris_training.py
"""
Training functions for AVIRIS Fixed Shape Experiment
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime
from tqdm.notebook import tqdm

from aviris_utils import (
    calculate_condition_number, visualize_reconstruction, 
    plot_loss_curves, CompressionModel, FixedShapeModel
)

def train_model_stage1(model, train_loader, test_loader, config):
    """
    Train model in stage 1 and record key shapes
    
    Args:
        model: The compression model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        config: Dictionary with training parameters
        
    Returns:
        recorded_shapes: Dictionary of recorded shapes
        recorded_metrics: Dictionary of metrics for each shape
        train_losses: List of training losses per epoch
        test_losses: List of test losses per epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Define separate optimizers for encoder and decoder
    encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=config['encoder_lr'])
    decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=config['decoder_lr'])
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Initialize lists to store losses and condition numbers
    train_losses = []
    test_losses = []
    condition_numbers = []
    
    # Create directories for visualizations and recorded shapes
    output_dir = config['output_dir']
    filter_dir = os.path.join(output_dir, "filter_evolution")
    recon_dir = os.path.join(output_dir, "reconstructions")
    shapes_dir = os.path.join(output_dir, "recorded_shapes")
    
    os.makedirs(filter_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(shapes_dir, exist_ok=True)
    
    # Dictionary to store recorded shapes
    recorded_shapes = {}
    recorded_metrics = {
        'initial': {'condition_number': float('inf'), 'test_mse': float('inf')},
        'lowest_condition_number': {'condition_number': float('inf'), 'test_mse': float('inf')},
        'lowest_test_mse': {'condition_number': float('inf'), 'test_mse': float('inf')},
        'final': {'condition_number': float('inf'), 'test_mse': float('inf')}
    }
    
    # Run a dummy forward pass to initialize shape
    if config.get('use_fsf', False) and model.encoder.pipeline is not None:
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
        np.save(os.path.join(shapes_dir, "initial_shape.npy"), initial_shape.detach().cpu().numpy())
        print(f"Recorded initial shape with condition number: {condition_number:.4f}")
    
    # Train for the specified number of epochs
    epochs = config['epochs']
    min_snr = config.get('min_snr', 10)
    max_snr = config.get('max_snr', 40)
    viz_interval = config.get('viz_interval', 5)
    
    best_test_loss = float('inf')
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Stage 1 Epoch {epoch+1}/{epochs}") as pbar:
            for batch_idx, x in enumerate(pbar):
                x = x.to(device)
                
                # Forward pass
                x_recon, z = model(x, add_noise=True, min_snr_db=min_snr, max_snr_db=max_snr)
                
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
        if config.get('use_fsf', False) and model.encoder.pipeline is not None:
            with torch.no_grad():
                dummy_input = next(iter(train_loader))[:1].to(device)
                model.encoder(dummy_input)
            
            current_shape = model.encoder.current_shape.clone()
            current_filter_output = model.encoder.filter_output.clone()
            
            # Calculate condition number
            current_condition_number = calculate_condition_number(current_filter_output)
            condition_numbers.append(current_condition_number)
            
            # Check for lowest condition number
            if current_condition_number < recorded_metrics['lowest_condition_number']['condition_number']:
                recorded_shapes['lowest_condition_number'] = current_shape
                recorded_metrics['lowest_condition_number']['condition_number'] = current_condition_number
                recorded_metrics['lowest_condition_number']['test_mse'] = avg_test_loss
                
                # Save shape
                np.save(os.path.join(shapes_dir, "lowest_condition_number_shape.npy"), 
                      current_shape.detach().cpu().numpy())
                print(f"New lowest condition number: {current_condition_number:.4f}")
            
            # Check for lowest test MSE
            if avg_test_loss < recorded_metrics['lowest_test_mse']['test_mse']:
                recorded_shapes['lowest_test_mse'] = current_shape
                recorded_metrics['lowest_test_mse']['condition_number'] = current_condition_number
                recorded_metrics['lowest_test_mse']['test_mse'] = avg_test_loss
                
                # Save shape
                np.save(os.path.join(shapes_dir, "lowest_test_mse_shape.npy"), 
                      current_shape.detach().cpu().numpy())
                print(f"New lowest test MSE: {avg_test_loss:.6f}")
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"Saved new best model with test loss: {best_test_loss:.6f}")
        
        # Visualize reconstruction periodically
        if (epoch + 1) % viz_interval == 0:
            visualize_reconstruction(model, test_loader, device, 
                                    os.path.join(recon_dir, f"recon_epoch_{epoch+1}.png"))
    
    # Record final shape
    if config.get('use_fsf', False) and model.encoder.pipeline is not None:
        final_shape = model.encoder.current_shape.clone()
        final_filter_output = model.encoder.filter_output.clone()
        recorded_shapes['final'] = final_shape
        
        # Calculate condition number
        final_condition_number = calculate_condition_number(final_filter_output)
        recorded_metrics['final']['condition_number'] = final_condition_number
        recorded_metrics['final']['test_mse'] = avg_test_loss
        
        # Save final shape
        np.save(os.path.join(shapes_dir, "final_shape.npy"), final_shape.detach().cpu().numpy())
        print(f"Recorded final shape with condition number: {final_condition_number:.4f}")
    
    # Save metrics for all recorded shapes
    with open(os.path.join(shapes_dir, "shape_metrics.txt"), 'w') as f:
        for shape_name, metrics in recorded_metrics.items():
            if shape_name in recorded_shapes:
                f.write(f"{shape_name} shape:\n")
                f.write(f"  Condition Number: {metrics['condition_number']:.4f}\n")
                f.write(f"  Test MSE: {metrics['test_mse']:.6f}\n\n")
    
    # Save condition numbers
    np.save(os.path.join(output_dir, "condition_numbers.npy"), np.array(condition_numbers))
    
    # Plot condition numbers
    if condition_numbers:
        plt.figure(figsize=(10, 6))
        plt.plot(condition_numbers, 'b-')
        plt.title('Filter Condition Number Evolution')
        plt.xlabel('Epochs')
        plt.ylabel('Condition Number')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "condition_number_evolution.png"), dpi=300)
        plt.close()

        # Log scale plot
        plt.figure(figsize=(10, 6))
        plt.semilogy(condition_numbers, 'r-')
        plt.title('Filter Condition Number Evolution (Log Scale)')
        plt.xlabel('Epochs')
        plt.ylabel('Condition Number (log scale)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "condition_number_evolution_log.png"), dpi=300)
        plt.close()
    
    return recorded_shapes, recorded_metrics, train_losses, test_losses

def train_with_fixed_shape(shape_name, shape, train_loader, test_loader, config):
    """
    Train model with fixed shape, optimizing only the decoder
    
    Args:
        shape_name: Name of the shape (for logging)
        shape: The fixed shape tensor
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        config: Dictionary with training parameters
        
    Returns:
        train_losses: List of training losses per epoch
        test_losses: List of test losses per epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory for this shape
    output_dir = config['output_dir']
    shape_dir = os.path.join(output_dir, f"stage2_{shape_name}")
    os.makedirs(shape_dir, exist_ok=True)
    os.makedirs(os.path.join(shape_dir, "reconstructions"), exist_ok=True)
    
    # Get input dimensions from data
    in_channels = next(iter(train_loader)).shape[1]
    
    # Create model with fixed shape
    model = FixedShapeModel(
        shape=shape,
        in_channels=in_channels,
        decoder_type=config.get('model', 'awan'),
        filter_scale_factor=config.get('filter_scale_factor', 50.0),
        device=device
    )
    
    # Move model to device
    model = model.to(device)
    
    # Only optimize decoder parameters
    optimizer = optim.Adam(model.decoder.parameters(), lr=config.get('decoder_lr', 1e-4))
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Initialize lists to store losses
    train_losses = []
    test_losses = []
    
    # Train for specified number of epochs
    stage2_epochs = config.get('stage2_epochs', 100)
    min_snr = config.get('min_snr', 10)
    max_snr = config.get('max_snr', 40)
    viz_interval = config.get('viz_interval', 5)
    
    best_test_loss = float('inf')
    print(f"\nTraining Stage 2 model with {shape_name} shape...")
    for epoch in range(stage2_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Stage 2 [{shape_name}] Epoch {epoch+1}/{stage2_epochs}") as pbar:
            for batch_idx, x in enumerate(pbar):
                x = x.to(device)
                
                # Forward pass
                x_recon, z = model(x, add_noise=True, min_snr_db=min_snr, max_snr_db=max_snr)
                
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
        
        print(f"[{shape_name}] Epoch {epoch+1}/{stage2_epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), os.path.join(shape_dir, "best_model.pt"))
            print(f"Saved new best model with test loss: {best_test_loss:.6f}")
        
        # Visualize reconstruction periodically
        if (epoch + 1) % viz_interval == 0 or epoch == stage2_epochs - 1:
            visualize_reconstruction(model, test_loader, device, 
                                   os.path.join(shape_dir, "reconstructions", f"recon_epoch_{epoch+1}.png"))
    
    # Save loss values and plots
    np.savez(os.path.join(shape_dir, "loss_values.npz"), 
            train_losses=np.array(train_losses), 
            test_losses=np.array(test_losses))
    
    # Plot loss curves
    plot_loss_curves(train_losses, test_losses, os.path.join(shape_dir, "loss_curves.png"))
    
    print(f"Stage 2 training for {shape_name} shape complete!")
    
    return train_losses, test_losses

def run_stage2(recorded_shapes, recorded_metrics, train_loader, test_loader, config):
    """
    Run stage 2 with all recorded shapes
    
    Args:
        recorded_shapes: Dictionary of shapes from stage 1
        recorded_metrics: Dictionary of metrics for each shape
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        config: Dictionary with training parameters
        
    Returns:
        stage2_results: Dictionary with results for all shapes
    """
    stage2_results = {
        'train_losses': {},
        'test_losses': {},
        'condition_numbers': {}
    }
    
    # Add random baseline shape if desired
    if config.get('add_random_baseline', False) and recorded_shapes:
        template_shape = next(iter(recorded_shapes.values()))
        random_shape = torch.rand_like(template_shape)
        random_shape[:, 0] = (random_shape[:, 0] > 0.5).float()
        recorded_shapes['random'] = random_shape
        recorded_metrics['random'] = {
            'condition_number': 1000.0,  # Default high value
            'test_mse': 0.1  # Default high value
        }
        print("Added random baseline shape")
    
    # Train with each recorded shape
    for shape_name, shape in recorded_shapes.items():
        print(f"\n=== Stage 2: Training with fixed {shape_name} shape ===")
        if shape_name in recorded_metrics:
            print(f"Shape condition number: {recorded_metrics[shape_name]['condition_number']:.4f}")
            stage2_results['condition_numbers'][shape_name] = recorded_metrics[shape_name]['condition_number']
        
        try:
            train_losses, test_losses = train_with_fixed_shape(
                shape_name, shape, train_loader, test_loader, config)
            
            stage2_results['train_losses'][shape_name] = train_losses
            stage2_results['test_losses'][shape_name] = test_losses
        except Exception as e:
            print(f"Error training with {shape_name} shape: {e}")
    
    return stage2_results