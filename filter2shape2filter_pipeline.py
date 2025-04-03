#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy.linalg as LA
from datetime import datetime

class Shape2FilterModel(nn.Module):
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
        # Apply sigmoid activation to constrain output to [0, 1] range
        out_2d = torch.sigmoid(out_2d)
        return out_2d

class Filter2ShapeVarLen(nn.Module):
    def __init__(self, d_in=100, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.row_preproc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
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
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 12)  # outputs: presence (4) and (x,y) for 4 points
        )
    def forward(self, spec_11x100):
        bsz = spec_11x100.size(0)
        x_r = spec_11x100.view(-1, spec_11x100.size(2))
        x_pre = self.row_preproc(x_r)
        x_pre = x_pre.view(bsz, -1, x_pre.size(-1))
        x_enc = self.encoder(x_pre)
        x_agg = x_enc.mean(dim=1)
        out_12 = self.mlp(x_agg)
        out_4x3 = out_12.view(bsz, 4, 3)
        presence_logits = out_4x3[:, :, 0]
        xy_raw = out_4x3[:, :, 1:]
        presence_list = []
        for i in range(4):
            if i == 0:
                presence_list.append(torch.ones(bsz, device=out_4x3.device, dtype=torch.float32))
            else:
                prob_i = torch.sigmoid(presence_logits[:, i]).clamp(1e-6, 1 - 1e-6)
                prob_chain = prob_i * presence_list[i - 1]
                ste_i = (prob_chain > 0.5).float() + prob_chain - prob_chain.detach()
                presence_list.append(ste_i)
        presence_stack = torch.stack(presence_list, dim=1)
        xy_bounded = torch.sigmoid(xy_raw)
        xy_final = xy_bounded * presence_stack.unsqueeze(-1)
        final_shape = torch.cat([presence_stack.unsqueeze(-1), xy_final], dim=-1)
        return final_shape

class Filter2Shape2FilterFrozen(nn.Module):
    def __init__(self, filter2shape_net, shape2filter_frozen, no_grad_frozen=False):
        """
        no_grad_frozen: if True, the frozen shape2filter network is computed in a no_grad block.
                         For Stage C training, set this to False so gradients can flow.
        """
        super().__init__()
        self.filter2shape = filter2shape_net
        self.shape2filter_frozen = shape2filter_frozen
        self.no_grad_frozen = no_grad_frozen
        for p in self.filter2shape.parameters():
            p.requires_grad = False
        for p in self.shape2filter_frozen.parameters():
            p.requires_grad = False
    
    def forward(self, spec_input):
        if self.no_grad_frozen:
            with torch.no_grad():
                shape_pred = self.filter2shape(spec_input)
                spec_chain = self.shape2filter_frozen(shape_pred)
        else:
            shape_pred = self.filter2shape(spec_input)
            spec_chain = self.shape2filter_frozen(shape_pred)
        return shape_pred, spec_chain

def replicate_c4(points):
    """Replicate points with C4 symmetry"""
    c4 = []
    for (x, y) in points:
        c4.append([x, y])       # Q1: original
        c4.append([-y, x])      # Q2: rotate 90°
        c4.append([-x, -y])     # Q3: rotate 180°
        c4.append([y, -x])      # Q4: rotate 270°
    return np.array(c4, dtype=np.float32)

def sort_points_by_angle(points):
    """Sort points by angle from center for polygon drawing"""
    if len(points) < 3:
        return points
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    idx = np.argsort(angles)
    return points[idx]

def plot_shape_with_c4(shape, title, save_path=None, show=True):
    """Plot shape with C4 symmetry replication in a minimal academic style"""
    plt.figure(figsize=(5, 5))
    plt.xlim(-0.7, 0.7)  # Fixed limits as requested
    plt.ylim(-0.7, 0.7)
    
    # Extract active points
    presence = shape[:, 0] > 0.5
    active_points = shape[presence, 1:3]
    
    # Plot original Q1 points
    plt.scatter(shape[presence, 1], shape[presence, 2], color='red', s=50)
    
    # Apply C4 symmetry and plot the polygon
    if len(active_points) > 0:
        c4_points = replicate_c4(active_points)
        sorted_points = sort_points_by_angle(c4_points)
        
        # If we have enough points for a polygon
        if len(sorted_points) >= 3:
            # Close the polygon
            polygon = np.vstack([sorted_points, sorted_points[0]])
            plt.plot(polygon[:, 0], polygon[:, 1], 'k-', linewidth=1.5)
            plt.fill(polygon[:, 0], polygon[:, 1], 'lightblue', alpha=0.5)
        else:
            # Just plot the points
            plt.scatter(c4_points[:, 0], c4_points[:, 1], color='blue', alpha=0.4, s=30)
    
    plt.title(title, fontsize=12)
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def plot_filter(filter_params, title, save_path=None, show=True):
    """Plot filter parameters"""
    plt.figure(figsize=(12, 8))
    for i in range(filter_params.shape[0]):
        plt.plot(filter_params[i], label=f'Filter {i}' if i % 3 == 0 else None)
    plt.grid(True)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Filter Value")
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def calculate_condition_number(filters):
    """
    Calculate condition number of the spectral filters matrix.
    
    Parameters:
    filters: Tensor of shape [11, 100] representing the spectral filters
    
    Returns:
    float: Condition number
    """
    # Convert to numpy for condition number calculation
    filters_np = filters.detach().cpu().numpy()
    
    # Use singular value decomposition to calculate condition number
    u, s, vh = LA.svd(filters_np)
    
    # Condition number is the ratio of largest to smallest singular value
    # Add small epsilon to prevent division by zero
    condition_number = s[0] / (s[-1] + 1e-10)
    
    return condition_number

def load_models(shape2filter_path, filter2shape_path, device=None):
    """
    Load the pretrained shape2filter and filter2shape models
    
    Parameters:
    shape2filter_path: Path to the pretrained shape2filter model
    filter2shape_path: Path to the pretrained filter2shape model
    device: Device to load the models to (if None, will use CUDA if available)
    
    Returns:
    tuple: (shape2filter, filter2shape) models
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the shape2filter model
    shape2filter = Shape2FilterModel()
    shape2filter.load_state_dict(torch.load(shape2filter_path, map_location=device))
    shape2filter = shape2filter.to(device)
    
    # Load the filter2shape model
    filter2shape = Filter2ShapeVarLen()
    filter2shape.load_state_dict(torch.load(filter2shape_path, map_location=device))
    filter2shape = filter2shape.to(device)
    
    return shape2filter, filter2shape

def create_pipeline(shape2filter, filter2shape, no_grad_frozen=False):
    """
    Create the filter2shape2filter pipeline
    
    Parameters:
    shape2filter: The loaded shape2filter model
    filter2shape: The loaded filter2shape model
    no_grad_frozen: Whether to use no_grad for the frozen shape2filter model
    
    Returns:
    Filter2Shape2FilterFrozen: The pipeline
    """
    return Filter2Shape2FilterFrozen(filter2shape, shape2filter, no_grad_frozen=no_grad_frozen)

def test_pipeline(shape2filter_path, filter2shape_path, output_dir=None):
    """
    Test the filter2shape2filter pipeline with random input filters
    
    Parameters:
    shape2filter_path: Path to the pretrained shape2filter model
    filter2shape_path: Path to the pretrained filter2shape model
    output_dir: Directory to save output visualizations
    
    Returns:
    None
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    shape2filter, filter2shape = load_models(shape2filter_path, filter2shape_path, device)
    print("Models loaded successfully")
    
    # Create pipeline
    pipeline = create_pipeline(shape2filter, filter2shape)
    print("Pipeline created")
    
    # Create a random filter of shape 11x100 with values between 0 and 1
    input_filter = torch.rand(1, 11, 100, device=device)  # Add batch dimension
    print(f"Input filter shape: {input_filter.shape}")
    
    # Pass through pipeline
    with torch.no_grad():
        shape_pred, recon_filter = pipeline(input_filter)
    
    print(f"Shape prediction shape: {shape_pred.shape}")
    print(f"Reconstructed filter shape: {recon_filter.shape}")
    
    # Calculate condition number
    input_condition_number = calculate_condition_number(input_filter[0])
    recon_condition_number = calculate_condition_number(recon_filter[0])
    print(f"Input filter condition number: {input_condition_number:.4f}")
    print(f"Reconstructed filter condition number: {recon_condition_number:.4f}")
    
    # Create output directory if not provided
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"filter2shape2filter_test_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to {output_dir}/")
    
    # Visualize input filter
    input_filter_np = input_filter[0].detach().cpu().numpy()
    plot_filter(
        input_filter_np, 
        f"Input Filter (Condition Number: {input_condition_number:.4f})",
        save_path=f"{output_dir}/input_filter.png",
        show=False
    )
    
    # Visualize shape
    shape_np = shape_pred[0].detach().cpu().numpy()
    plot_shape_with_c4(
        shape_np, 
        "Predicted Shape",
        save_path=f"{output_dir}/predicted_shape.png",
        show=False
    )
    
    # Visualize reconstructed filter
    recon_filter_np = recon_filter[0].detach().cpu().numpy()
    plot_filter(
        recon_filter_np, 
        f"Reconstructed Filter (Condition Number: {recon_condition_number:.4f})",
        save_path=f"{output_dir}/reconstructed_filter.png",
        show=False
    )
    
    # Save numerical data
    np.save(f"{output_dir}/input_filter.npy", input_filter_np)
    np.save(f"{output_dir}/predicted_shape.npy", shape_np)
    np.save(f"{output_dir}/reconstructed_filter.npy", recon_filter_np)
    
    print("Done! Visualizations and data saved successfully.")

def run_pipeline(input_filter, shape2filter_path=None, filter2shape_path=None, device=None, 
               return_shape=False, visualize=False, output_dir=None):
    """
    Simple function to run a filter through the filter2shape2filter pipeline
    
    Parameters:
    input_filter: Tensor of shape [11, 100] - the input filter parameters
    shape2filter_path: Path to the shape2filter model weights (optional)
    filter2shape_path: Path to the filter2shape model weights (optional)
    device: Device to run on (optional)
    return_shape: Whether to also return the intermediate shape (default: False)
    visualize: Whether to visualize the results (default: False)
    output_dir: Directory to save visualizations if visualize=True (optional)
    
    Returns:
    If return_shape=True: (shape, reconstructed_filter)
    If return_shape=False: reconstructed_filter
    Both without batch dimension (shape: [4, 3], filter: [11, 100])
    """
    # Default paths if not provided
    if shape2filter_path is None:
        shape2filter_path = "outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt"
    if filter2shape_path is None:
        filter2shape_path = "outputs_three_stage_20250322_145925/stageC/spec2shape_stageC.pt"
    
    # Set device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure input_filter is a tensor
    if not isinstance(input_filter, torch.Tensor):
        input_filter = torch.tensor(input_filter, dtype=torch.float32)
    
    # Move to device
    input_filter = input_filter.to(device)
    
    # Add batch dimension if needed
    if input_filter.dim() == 2:
        input_filter = input_filter.unsqueeze(0)  # [11, 100] -> [1, 11, 100]
    
    # Load models
    shape2filter, filter2shape = load_models(shape2filter_path, filter2shape_path, device)
    
    # Create pipeline
    pipeline = create_pipeline(shape2filter, filter2shape)
    
    # Run pipeline
    with torch.no_grad():
        shape_pred, recon_filter = pipeline(input_filter)
    
    # Remove batch dimension
    shape_pred = shape_pred[0]  # [1, 4, 3] -> [4, 3]
    recon_filter = recon_filter[0]  # [1, 11, 100] -> [11, 100]
    
    # Visualize if requested
    if visualize:
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"filter2shape2filter_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate condition numbers
        input_cn = calculate_condition_number(input_filter[0])
        recon_cn = calculate_condition_number(recon_filter)
        
        # Plot input filter
        plot_filter(
            input_filter[0].detach().cpu().numpy(),
            f"Input Filter (CN: {input_cn:.4f})",
            save_path=f"{output_dir}/input_filter.png",
            show=True
        )
        
        # Plot predicted shape
        plot_shape_with_c4(
            shape_pred.detach().cpu().numpy(),
            "Predicted Shape",
            save_path=f"{output_dir}/predicted_shape.png",
            show=True
        )
        
        # Plot reconstructed filter
        plot_filter(
            recon_filter.detach().cpu().numpy(),
            f"Reconstructed Filter (CN: {recon_cn:.4f})",
            save_path=f"{output_dir}/reconstructed_filter.png",
            show=True
        )
        
        print(f"Visualizations saved to {output_dir}/")
    
    # Return outputs based on return_shape flag
    if return_shape:
        return shape_pred.detach().cpu(), recon_filter.detach().cpu()
    else:
        return recon_filter.detach().cpu()

# Example usage
if __name__ == "__main__":
    # Update these paths to the location of your model weights
    shape2filter_path = "outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt"
    filter2shape_path = "outputs_three_stage_20250322_145925/stageC/spec2shape_stageC.pt"
    
    # Example 1: Use the test pipeline
    # test_pipeline(shape2filter_path, filter2shape_path)
    
    # Example 2: Use the simple run_pipeline function
    print("Creating random 11x100 filter...")
    random_filter = torch.rand(11, 100)
    
    # Get just the reconstructed filter
    recon_filter = run_pipeline(random_filter, visualize=True)
    print(f"Reconstructed filter shape: {recon_filter.shape}")
    
    # Get both shape and reconstructed filter
    shape, recon_filter = run_pipeline(random_filter, return_shape=True)
    print(f"Shape: {shape.shape}, Reconstructed filter: {recon_filter.shape}")