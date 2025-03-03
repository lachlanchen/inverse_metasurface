import torch
import math

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
    
    Process:
      1. Vertex Presence:
         - Compute p_i = sigmoid(raw_v_pres_i) and then cumulative product.
         - Force the first vertex active and apply a straight–through threshold at 0.5,
           yielding a binary mask b.
      2. Count valid vertices: n = sum(b) and spacing s = π/(2*n).
      3. Angle Assignment:
         - For valid vertices, assign index i (via cumulative sum minus 1) so that
             θ_base = i * s.
         - Compute δ = s * sigmoid(raw_shift) and set final angles θ = θ_base + δ.
      4. Radius Mapping:
         - Map raw_radius via sigmoid to [0,1] then linearly to [0.05,0.65],
           multiplied by the binary mask.
      5. Cartesian Coordinates:
         - x = r * cos(θ) and y = r * sin(θ).
      6. Final Output:
         - Produce a 4×3 matrix with column 0: binary vertex presence, column 1: x, and column 2: y.
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
    # Column 0: binary vertex presence, Columns 1 and 2: x and y coordinates.
    final_shape = torch.cat([v_pres_bin.unsqueeze(1), coordinates], dim=1)
    
    return final_shape, v_pres_bin, n, radius, angles, delta, s_spacing, idx

def main():
    # Initialize raw_params as a random 4×2 tensor.
    # For vertex presence, bias values high so that sigmoid(raw_v_pres) is near 1.
    raw_v_pres = torch.rand(4, 1) + 2.0
    raw_radius = torch.rand(4, 1)  # raw radius values
    raw_params = torch.cat([raw_v_pres, raw_radius], dim=1)
    raw_params.requires_grad_()  # set requires_grad
    
    # Initialize raw_shift as a scalar parameter.
    raw_shift = torch.rand(1, requires_grad=True)
    
    final_shape, v_pres_bin, n, radius, angles, delta, s_spacing, idx = differentiable_legal_shape(raw_params, raw_shift)
    
    print("=== Raw Parameters (4×2) ===")
    print(raw_params)
    print("\nRaw Shift Parameter:")
    print(raw_shift)
    
    print("\n=== Processed Vertex Presence (binary, straight-through) ===")
    print(v_pres_bin)
    print("\nNumber of valid vertices (n):", n)
    print("\nCumulative indices (idx):", idx)
    print("\nSpacing s (π/(2*n)):", s_spacing)
    print("\nDelta (shift) from raw_shift:", delta)
    print("\nFinal angles (radians):", angles)
    print("\nMapped radii (in [0.05, 0.65]):", radius)
    
    print("\n=== Final Shape (4×3 matrix) ===")
    # Column 0: vertex presence, Column 1: x, Column 2: y.
    print(final_shape)
    
    # Test gradient flow
    loss = final_shape.sum()
    loss.backward()
    print("\nGradients on raw_params:")
    print(raw_params.grad)
    print("\nGradients on raw_shift:")
    print(raw_shift.grad)

if __name__ == "__main__":
    main()

