import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from datetime import datetime

# Define the ShapeToSpectraModel class
class ShapeToSpectraModel(nn.Module):
    def __init__(self, d_in=3, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, 11 * 100)
        )
    def forward(self, shape_4x3):
        bsz = shape_4x3.size(0)
        presence = shape_4x3[:, :, 0]
        key_padding_mask = (presence < 0.5)
        x_proj = self.input_proj(shape_4x3)
        x_enc = self.encoder(x_proj, src_key_padding_mask=key_padding_mask)
        pres_sum = presence.sum(dim=1, keepdim=True) + 1e-8
        x_enc_w = x_enc * presence.unsqueeze(-1)
        shape_emb = x_enc_w.sum(dim=1) / pres_sum
        out_flat = self.mlp(shape_emb)
        out_2d = out_flat.view(bsz, 11, 100)
        return out_2d

def straight_through_threshold(x, thresh=0.5):
    """
    Forward: threshold x at `thresh` (returns 1.0 where x>=thresh, 0 otherwise).
    Backward: gradients flow as if the operation were the identity.
    """
    y = (x >= thresh).float()
    return x + (y - x).detach()

def differentiable_legal_shape(raw_params, raw_shift):
    """
    Input:
      raw_params: a 4×2 tensor, each row: [raw_v_pres, raw_radius]
      raw_shift: a scalar tensor representing the raw angle shift.
    
    Output:
      A 4×3 matrix with column 0: binary vertex presence, column 1: x, and column 2: y.
    """
    device = raw_params.device
    dtype = raw_params.dtype

    # --- Vertex Presence ---
    # Apply sigmoid to raw_v_pres (column 0)
    v_pres_prob = torch.sigmoid(raw_params[:, 0])
    # Compute cumulative product: once a value is low, later ones get suppressed.
    v_pres_cum = torch.cumprod(v_pres_prob, dim=0)
    # Force the first vertex to be present:
    v_pres_cum = torch.cat([torch.ones(1, device=device, dtype=dtype), v_pres_cum[1:]], dim=0)
    # Apply straight-through threshold at 0.5:
    v_pres_bin = straight_through_threshold(v_pres_cum, thresh=0.5)

    # --- Count Valid Vertices ---
    n = v_pres_bin.sum()  # differentiable count (should be 4 if all are active)

    # --- Cumulative Indices for Valid Vertices ---
    idx = torch.cumsum(v_pres_bin, dim=0) - 1.0  # indices: first valid gets 0, second gets 1, etc.

    # --- Angle Assignment ---
    # Spacing s = π/(2*n) (avoid division by zero)
    s_spacing = math.pi / (2.0 * torch.clamp(n, min=1.0))
    base_angles = idx * s_spacing  # base angles for each vertex
    # Use the raw_shift parameter to compute delta (shift) in [0, s]
    delta = s_spacing * torch.sigmoid(raw_shift)
    # Final angles for active vertices
    angles = (base_angles + delta) * v_pres_bin

    # --- Radius Mapping ---
    # Map raw_radius (column 1) via sigmoid then linearly to [0.05, 0.65]
    radius = 0.05 + 0.6 * torch.sigmoid(raw_params[:, 1])
    radius = radius * v_pres_bin  # zero out inactive vertices

    # --- Cartesian Coordinates ---
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    coordinates = torch.stack([x, y], dim=1)

    # --- Final Output: 4×3 matrix ---
    # Column 0: binary vertex presence, Column 1 and 2: x and y coordinates.
    final_shape = torch.cat([v_pres_bin.unsqueeze(1), coordinates], dim=1)
    
    return final_shape

def replicate_c4(points):
    """Replicate points with C4 symmetry"""
    c4 = []
    for (x, y) in points:
        c4.append([x, y])       # Q1: original
        c4.append([-y, x])      # Q2: rotate 90°
        c4.append([-x, -y])     # Q3: rotate 180°
        c4.append([y, -x])      # Q4: rotate 270°
    return np.array(c4, dtype=np.float32)

def plot_spectrum_comparison(gt_spectrum, optimized_spectrum, save_path):
    """Plot a comparison of ground truth and optimized spectra"""
    plt.figure(figsize=(14, 10))
    
    # Convert tensors to numpy arrays
    gt_np = gt_spectrum.detach().cpu().numpy()
    opt_np = optimized_spectrum.detach().cpu().numpy()
    
    # Calculate MSE for each wavelength band
    mse_per_band = np.mean((gt_np - opt_np)**2, axis=1)
    total_mse = np.mean(mse_per_band)
    
    # Create subplot grid: 3x4 (to hold 11 plots + 1 legend plot)
    for i in range(11):
        plt.subplot(3, 4, i+1)
        plt.plot(gt_np[i], 'b-', label='Ground Truth')
        plt.plot(opt_np[i], 'r--', label='Optimized')
        plt.title(f'Band {i}: MSE={mse_per_band[i]:.6f}')
        plt.grid(True)
        if i >= 8:  # Add x-label only for bottom row
            plt.xlabel('Wavelength index')
        if i % 4 == 0:  # Add y-label only for leftmost column
            plt.ylabel('Reflectance')
    
    # Use the last subplot for the legend
    plt.subplot(3, 4, 12)
    plt.plot([], [], 'b-', label='Ground Truth')
    plt.plot([], [], 'r--', label='Optimized')
    plt.legend(fontsize=12)
    plt.axis('off')
    plt.text(0.1, 0.5, f'Total MSE: {total_mse:.6f}', fontsize=14)
    
    plt.tight_layout()
    plt.suptitle('Spectrum Comparison: Ground Truth vs Optimized', fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.savefig(save_path)
    plt.close()

def plot_shape_and_spectrum(shape, spectrum, title, save_path):
    """Plot the shape and spectrum in a 1x2 figure"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot shape
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.7, 0.7)
    ax1.set_ylim(-0.7, 0.7)
    ax1.grid(True)
    ax1.set_title("Shape")
    
    # Extract shape coordinates - ensure tensors are detached if they require gradients
    presence = shape[:, 0].detach().cpu().numpy()
    x_coords = shape[:, 1].detach().cpu().numpy()
    y_coords = shape[:, 2].detach().cpu().numpy()
    
    # Extract active points
    active_idx = np.where(presence > 0.5)[0]
    q1_points = np.column_stack([x_coords[active_idx], y_coords[active_idx]])
    
    if len(q1_points) > 0:
        # Replicate points with C4 symmetry
        c4_points = replicate_c4(q1_points)
        
        # Sort points to form a closed polygon
        if len(c4_points) >= 3:
            center = np.mean(c4_points, axis=0)
            angles = np.arctan2(c4_points[:, 1] - center[1], c4_points[:, 0] - center[0])
            sorted_idx = np.argsort(angles)
            sorted_points = c4_points[sorted_idx]
            
            # Close the polygon
            polygon = np.vstack([sorted_points, sorted_points[0]])
            ax1.plot(polygon[:, 0], polygon[:, 1], 'g-', linewidth=2)
            ax1.fill(polygon[:, 0], polygon[:, 1], 'g', alpha=0.3)
    
    # Plot the original Q1 points
    ax1.scatter(x_coords[active_idx], y_coords[active_idx], color='red', s=50)
    
    # Plot spectrum
    spec_np = spectrum.detach().cpu().numpy()
    for i, row in enumerate(spec_np):
        ax2.plot(row, label=f'c={i}' if i % 3 == 0 else "")
    ax2.set_xlabel('Wavelength index')
    ax2.set_ylabel('Reflectance')
    ax2.set_title("Spectrum")
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def optimize_shape_for_target_spectrum():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"shape_optimization_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}/")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load shape2spec model (will be kept frozen)
    model_path = "outputs_three_stage_20250216_180408/stageA/shape2spec_stageA.pt"
    model = ShapeToSpectraModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    
    # Freeze the network - we only optimize the shape
    for param in model.parameters():
        param.requires_grad = False
    print("Shape2spec model loaded and frozen - we'll only optimize the shape parameters")
    
    # Step 1: Create a "ground truth" shape using differentiable_legal_shape
    # Initialize parameters for ground truth shape - make it have 3 vertices
    gt_raw_params = torch.tensor([
        [2.0, 0.5],   # First vertex - high presence prob, medium radius
        [2.0, 0.8],   # Second vertex - high presence prob, larger radius
        [2.0, 0.3],   # Third vertex - high presence prob, smaller radius
        [-2.0, 0.0]   # Fourth vertex - low presence prob, will be inactive
    ], device=device, dtype=torch.float32)
    gt_raw_shift = torch.tensor([0.2], device=device, dtype=torch.float32)
    
    # Generate the ground truth shape
    gt_shape = differentiable_legal_shape(gt_raw_params, gt_raw_shift)
    
    # Print ground truth shape tensor
    print("\nGround Truth Shape Tensor (4x3):")
    print(gt_shape.cpu().numpy())
    
    # Get the ground truth spectrum
    with torch.no_grad():
        gt_spectrum = model(gt_shape.unsqueeze(0))[0]
    
    print("Ground truth shape created with shape:", gt_shape.shape)
    print("Ground truth spectrum created with shape:", gt_spectrum.shape)
    
    # Save the ground truth shape and spectrum
    plot_shape_and_spectrum(gt_shape, gt_spectrum, "Ground Truth", f"{output_dir}/gt_shape_spectrum.png")
    
    # Step 2: Initialize a different shape to optimize
    # Start with different parameters
    opt_raw_params = torch.tensor([
        [2.0, -0.5],  # First vertex - high presence prob, different radius
        [0.0, 0.0],   # Second vertex - 50/50 presence prob
        [-1.0, 0.0],  # Third vertex - low presence prob
        [-2.0, 0.0]   # Fourth vertex - very low presence prob
    ], device=device, dtype=torch.float32, requires_grad=True)
    
    opt_raw_shift = torch.tensor([0.7], device=device, dtype=torch.float32, requires_grad=True)
    
    # Initial shape
    initial_shape = differentiable_legal_shape(opt_raw_params, opt_raw_shift)
    
    # Print initial shape tensor
    print("\nInitial Shape Tensor (4x3):")
    print(initial_shape.detach().cpu().numpy())
    
    with torch.no_grad():
        initial_spectrum = model(initial_shape.unsqueeze(0))[0]
    
    # Save initial shape/spectrum
    plot_shape_and_spectrum(initial_shape, initial_spectrum, "Initial Shape", f"{output_dir}/initial_shape_spectrum.png")
    
    # Step 3: Setup optimization
    optimizer = torch.optim.Adam([opt_raw_params, opt_raw_shift], lr=0.01)
    n_iterations = 1000
    loss_values = []
    
    print("\nStarting optimization...")
    # Optimization loop
    for i in range(n_iterations):
        optimizer.zero_grad()
        
        # Generate the shape from optimizable parameters
        current_shape = differentiable_legal_shape(opt_raw_params, opt_raw_shift)
        
        # Get the current spectrum from the frozen model
        current_spectrum = model(current_shape.unsqueeze(0))[0]
        
        # Calculate loss (MSE between current and target spectra)
        loss = torch.nn.functional.mse_loss(current_spectrum, gt_spectrum)
        loss_values.append(loss.item())
        
        # Backpropagate and update shape parameters only
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (i+1) % 20 == 0 or i == 0:
            print(f"Iteration {i+1}/{n_iterations}, Loss: {loss.item():.6f}")
            
            # Visualize intermediate result
            if (i+1) % 50 == 0:
                plot_shape_and_spectrum(
                    current_shape, 
                    current_spectrum, 
                    f"Optimized Shape (Iteration {i+1})", 
                    f"{output_dir}/optimized_shape_iter_{i+1}.png"
                )
    
    # Final result
    with torch.no_grad():
        final_shape = differentiable_legal_shape(opt_raw_params, opt_raw_shift)
        final_spectrum = model(final_shape.unsqueeze(0))[0]
    
    print("\nOptimization complete!")
    print("Final loss:", loss_values[-1])
    
    # Print final shape tensor
    print("\nFinal Optimized Shape Tensor (4x3):")
    print(final_shape.detach().cpu().numpy())
    
    # Plot final result
    plot_shape_and_spectrum(
        final_shape, 
        final_spectrum, 
        "Final Optimized Shape", 
        f"{output_dir}/final_optimized_shape.png"
    )
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title('Optimization Progress')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(f'{output_dir}/optimization_loss.png')
    plt.close()
    
    # Create spectrum comparison plot
    plot_spectrum_comparison(gt_spectrum, final_spectrum, f"{output_dir}/spectrum_comparison.png")
    
    # Save spectrum data for reference
    np.save(f"{output_dir}/gt_spectrum.npy", gt_spectrum.detach().cpu().numpy())
    np.save(f"{output_dir}/final_spectrum.npy", final_spectrum.detach().cpu().numpy())
    np.save(f"{output_dir}/loss_values.npy", np.array(loss_values))
    
    print(f"\nAll results saved to {output_dir}/")
    print("Spectrum comparison plot saved as spectrum_comparison.png")

if __name__ == "__main__":
    optimize_shape_for_target_spectrum()
