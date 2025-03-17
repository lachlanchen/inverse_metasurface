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

def plot_shape_and_spectrum(shape, spectrum, title, save_path):
    """Plot the shape and spectrum in a 1x2 figure"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot shape
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.7, 0.7)
    ax1.set_ylim(-0.7, 0.7)
    ax1.grid(True)
    ax1.set_title("Shape")
    
    # Extract shape coordinates
    presence = shape[:, 0].cpu().detach().numpy()
    x_coords = shape[:, 1].cpu().detach().numpy()
    y_coords = shape[:, 2].cpu().detach().numpy()
    
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
    spec_np = spectrum.cpu().detach().numpy()
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
    
    # Save spectrum separately
    spectrum_only_path = save_path.replace('.png', '_spectrum.png')
    plt.figure(figsize=(10, 6))
    for i, row in enumerate(spec_np):
        plt.plot(row, label=f'c={i}' if i % 3 == 0 else "")
    plt.xlabel('Wavelength index')
    plt.ylabel('Reflectance')
    plt.title(f"{title} - Spectrum")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(spectrum_only_path)
    plt.close()

def print_shape_details(shape, label):
    """Print detailed information about a shape"""
    print(f"\n=== {label} Shape Details ===")
    # Get shape as numpy for easier printing
    shape_np = shape.detach().cpu().numpy()
    
    # Print the full 4x3 shape matrix
    print("Shape matrix (4x3):")
    print("   Presence      X           Y")
    for i in range(shape_np.shape[0]):
        print(f"[{shape_np[i, 0]:8.4f}  {shape_np[i, 1]:10.6f}  {shape_np[i, 2]:10.6f}]")
    
    # Count active vertices
    active = (shape_np[:, 0] > 0.5)
    num_active = np.sum(active)
    print(f"Number of active vertices: {num_active}")
    
    # Print active vertices only
    if num_active > 0:
        print("Active vertices (x, y):")
        active_points = shape_np[active, 1:]
        for i, (x, y) in enumerate(active_points):
            print(f"  Point {i+1}: ({x:8.6f}, {y:8.6f})")

def optimize_shape_for_target_spectrum():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"shape_optimization_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving results to: {out_dir}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_path = "outputs_three_stage_20250216_180408/stageA/shape2spec_stageA.pt"
    model = ShapeToSpectraModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    
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
    
    # Print details of the ground truth shape
    print_shape_details(gt_shape, "Ground Truth")
    
    # Get the ground truth spectrum
    with torch.no_grad():
        gt_spectrum = model(gt_shape.unsqueeze(0))[0]
    
    print("Ground truth shape created with shape:", gt_shape.shape)
    print("Ground truth spectrum created with shape:", gt_spectrum.shape)
    
    # Save the ground truth shape and spectrum
    plot_shape_and_spectrum(gt_shape, gt_spectrum, "Ground Truth", os.path.join(out_dir, "gt_shape_spectrum.png"))
    
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
    
    # Print details of the initial shape
    print_shape_details(initial_shape, "Initial")
    
    with torch.no_grad():
        initial_spectrum = model(initial_shape.unsqueeze(0))[0]
    
    # Save initial shape/spectrum
    plot_shape_and_spectrum(initial_shape, initial_spectrum, "Initial Shape", 
                           os.path.join(out_dir, "initial_shape_spectrum.png"))
    
    # Step 3: Setup optimization
    optimizer = torch.optim.Adam([opt_raw_params, opt_raw_shift], lr=0.01)
    n_iterations = 350
    loss_values = []
    
    print("\nStarting optimization...")
    # Optimization loop
    for i in range(n_iterations):
        optimizer.zero_grad()
        
        # Generate the shape from optimizable parameters
        current_shape = differentiable_legal_shape(opt_raw_params, opt_raw_shift)
        
        # Get the current spectrum
        current_spectrum = model(current_shape.unsqueeze(0))[0]
        
        # Calculate loss (MSE between current and target spectra)
        loss = torch.nn.functional.mse_loss(current_spectrum, gt_spectrum)
        loss_values.append(loss.item())
        
        # Backpropagate and update
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
                    os.path.join(out_dir, f"optimized_shape_iter_{i+1}.png")
                )
    
    # Final result
    with torch.no_grad():
        final_shape = differentiable_legal_shape(opt_raw_params, opt_raw_shift)
        final_spectrum = model(final_shape.unsqueeze(0))[0]
    
    # Print details of the final optimized shape
    print_shape_details(final_shape, "Final Optimized")
    
    print("\nOptimization complete!")
    print("Final loss:", loss_values[-1])
    
    # Plot final result
    plot_shape_and_spectrum(
        final_shape, 
        final_spectrum, 
        "Final Optimized Shape", 
        os.path.join(out_dir, "final_optimized_shape.png")
    )
    
    # Compare spectra
    plt.figure(figsize=(12, 6))
    
    # Plot ground truth spectrum
    gt_spec_np = gt_spectrum.cpu().numpy()
    for i, row in enumerate(gt_spec_np):
        if i % 3 == 0:  # Only label every 3rd line to avoid clutter
            plt.plot(row, 'b-', label=f'GT c={i}')
        else:
            plt.plot(row, 'b-', alpha=0.5)
    
    # Plot optimized spectrum
    final_spec_np = final_spectrum.cpu().numpy()
    for i, row in enumerate(final_spec_np):
        if i % 3 == 0:  # Only label every 3rd line to avoid clutter
            plt.plot(row, 'r--', label=f'Opt c={i}')
        else:
            plt.plot(row, 'r--', alpha=0.5)
    
    plt.xlabel('Wavelength index')
    plt.ylabel('Reflectance')
    plt.title('Ground Truth vs Optimized Spectrum')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "spectrum_comparison.png"))
    plt.close()
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title('Optimization Progress')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(os.path.join(out_dir, 'optimization_loss.png'))
    plt.close()

if __name__ == "__main__":
    optimize_shape_for_target_spectrum()
