import torch
import math

def straight_through_threshold(x, thresh=0.5):
    """
    Forward: threshold x at `thresh` (returns 1.0 where x>=thresh, 0 otherwise).
    Backward: gradient flows as if the operation were the identity.
    """
    y = (x >= thresh).float()
    return x + (y - x).detach()

def differentiable_legal_shape(raw_params):
    """
    Input:
      raw_params: a 4x3 tensor (each row: [raw_v_pres, raw_radius, raw_angle])
                  provided as a random matrix (before any sigmoid).
    
    Process:
      1. Vertex Presence:
         - Apply sigmoid to column 0 to get probabilities.
         - Compute cumulative product: 
               p₀ = sigmoid(v₀),  pᵢ = pᵢ₋₁ * sigmoid(vᵢ).
         - Force the first vertex to be present by replacing p₀ with 1.
         - Apply a straight–through threshold (thresh=0.5) to yield a binary mask.
      
      2. Count valid vertices: n = sum(binary_mask). Define spacing s = π/(2*n).
      
      3. Angle Assignment:
         - Compute cumulative indices (via cumsum) so that valid vertices get indices 0,1,…, n–1.
         - Base angles: θ_base = index * s.
         - Instead of using torch.rand for the shift, use the third column of raw data.
           Compute the shift as:
             δ = s * sigmoid(raw_params[0,2])
         - Final angle: θ = θ_base + δ (inactive vertices are zeroed).
      
      4. Radius Mapping:
         - Map raw_radius (column 1) via sigmoid to [0,1] then linearly to [0.05, 0.65]:
               r = 0.05 + 0.6 * sigmoid(raw_radius)
         - Multiply by the binary mask so that inactive vertices get zero.
      
      5. Cartesian Coordinates:
         - Compute x = r*cos(θ) and y = r*sin(θ).
      
      6. Final Output:
         - Pack into a 4×3 matrix where:
             • Column 0: binary vertex presence (v_pres)
             • Column 1: x coordinate
             • Column 2: y coordinate
    """
    device = raw_params.device
    dtype = raw_params.dtype
    
    # --- Process Vertex Presence ---
    # Apply sigmoid to raw_v_pres (column 0)
    v_pres_prob = torch.sigmoid(raw_params[:, 0])
    # Compute cumulative product (once a value is low, later ones get suppressed)
    v_pres_cum = torch.cumprod(v_pres_prob, dim=0)
    # Force the first vertex to be present:
    v_pres_cum = torch.cat([torch.ones(1, device=device, dtype=dtype), v_pres_cum[1:]], dim=0)
    # Apply straight-through threshold at 0.5:
    v_pres_bin = straight_through_threshold(v_pres_cum, thresh=0.5)
    
    # --- Count Valid Vertices ---
    n = v_pres_bin.sum()  # differentiable count (if all active, n == 4)
    
    # --- Compute Cumulative Indices for Valid Vertices ---
    # Valid vertices get indices: first valid = 0, next = 1, etc.
    idx = torch.cumsum(v_pres_bin, dim=0) - 1.0  # shape (4,)
    
    # --- Angle Assignment ---
    # Spacing: s = π/(2*n) (avoid division by zero with clamp)
    s = math.pi / (2.0 * torch.clamp(n, min=1.0))
    base_angles = idx * s  # base angle for each row
    # Instead of random shift, use the third column's first entry:
    # Compute delta = s * sigmoid(raw_params[0,2])
    delta = s * torch.sigmoid(torch.sum(raw_params[:, 2]))
    # Final angles (only for active vertices):
    angles = (base_angles + delta) * v_pres_bin
    
    # --- Radius Mapping ---
    # Map raw_radius (column 1) via sigmoid then scale to [0.05, 0.65]
    radius = 0.05 + 0.6 * torch.sigmoid(raw_params[:, 1])
    radius = radius * v_pres_bin  # Zero out inactive vertices
    
    # --- Cartesian Coordinates ---
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    # Stack x and y to get a (4,2) tensor.
    coordinates = torch.stack([x, y], dim=1)
    
    # --- Final 4x3 Output ---
    # Column 0: binary vertex presence, Columns 1 & 2: x and y.
    final_shape = torch.cat([v_pres_bin.unsqueeze(1), coordinates], dim=1)
    
    return final_shape, v_pres_bin, n, radius, angles, delta, s, idx

def main():
    # Initialize raw_params as a random 4x3 tensor.
    # For the vertex presence column, use larger values so that sigmoid yields values near 1.
    raw_v_pres = torch.rand(4, 1) + 2.0
    other_cols = torch.rand(4, 2)
    raw_params = torch.cat([raw_v_pres, other_cols], dim=1)
    raw_params.requires_grad_()  # set requires_grad after creation
    
    final_shape, v_pres_bin, n, radius, angles, delta, s, idx = differentiable_legal_shape(raw_params)
    
    print("=== Raw Parameters (Input, random, before sigmoid) ===")
    print(raw_params)
    
    print("\n=== Processed Vertex Presence (binary, straight-through) ===")
    print(v_pres_bin)
    
    print("\nNumber of valid vertices (n):", n)
    print("\nCumulative indices (idx):", idx)
    print("\nSpacing s (π/(2*n)):", s)
    print("\nDelta (shift) from raw_params[0,2]:", delta)
    print("\nFinal angles (radians):", angles)
    print("\nMapped radii (in [0.05, 0.65]):", radius)
    
    print("\n=== Final Shape (4x3 matrix) ===")
    # Column 0: v_pres, Column 1: x, Column 2: y.
    print(final_shape)
    
    # Test gradient flow with a dummy loss.
    loss = final_shape.sum()
    loss.backward()
    print("\nGradients on raw_params:")
    print(raw_params.grad)

if __name__ == "__main__":
    main()

