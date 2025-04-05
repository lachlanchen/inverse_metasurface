#!/usr/bin/env python3
import os
import sys
import csv
import random
import argparse
from datetime import datetime
import concurrent.futures

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

############################################
# Argument Parsing
############################################

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate transmittance data for metasurface shapes')
    parser.add_argument('--npz_file', type=str, default="preprocessed_t_data.npz",
                        help='Path to the preprocessed NPZ file')
    parser.add_argument('--spec2shape_ckpt', type=str, 
                        default="outputs_three_stage_20250322_145925/stageB/spec2shape_stageB.pt",
                        help='Path to the spec2shape model checkpoint')
    parser.add_argument('--shape2spec_ckpt', type=str, 
                        default="outputs_three_stage_20250322_145925/stageA/shape2spec_stageA.pt",
                        help='Path to the shape2spec model checkpoint')
    parser.add_argument('--n_samples', type=int, default=4,
                        help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=23,
                        help='Random seed for reproducibility')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of worker threads for S4 simulations')
    parser.add_argument('--out_folder', type=str, default=None,
                        help='Output folder name (default: auto-generated with timestamp)')
    return parser.parse_args()

############################################
# S4 and Utility Functions
############################################

def run_s4_for_c(polygon_str, c_val):
    """
    Run the S4 binary with the given polygon string and c value
    using subprocess (blocking). Returns the path to the CSV file with results
    (or None if not found).
    """
    cmd = f'../build/S4 -a "{polygon_str} -c {c_val} -v -s" metasurface_fixed_shape_and_c_value.lua'
    print(f"[DEBUG] S4 command: {cmd}")
    # Using os.popen().read() here is blocking.
    output = os.popen(cmd).read()
    if not output:
        print("[ERROR] S4 produced no output for c =", c_val)
        return None
    saved_path = None
    for line in output.splitlines():
        if "Saved to" in line:
            saved_path = line.split("Saved to", 1)[1].strip()
            break
    return saved_path

def read_results_csv(csv_path):
    """
    Reads the CSV output from S4 and returns lists of wavelengths, reflectance, and transmission.
    """
    wv, Rv, Tv = [], [], []
    if not csv_path or not os.path.exists(csv_path):
        return wv, Rv, Tv
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lam = float(row["wavelength_um"])
                R_ = float(row["R"])
                T_ = float(row["T"])
                wv.append(lam)
                Rv.append(R_)
                Tv.append(T_)
            except Exception:
                continue
    data = sorted(zip(wv, Rv, Tv), key=lambda x: x[0])
    wv = [d[0] for d in data]
    Rv = [d[1] for d in data]
    Tv = [d[2] for d in data]
    return wv, Rv, Tv

def replicate_c4(points):
    """
    Given an array of Q1 vertices (Nx2), replicates them to fill all four quadrants using C4 symmetry.
    """
    replicated = []
    for (x, y) in points:
        replicated.append([x, y])
        replicated.append([-y, x])
        replicated.append([-x, -y])
        replicated.append([y, -x])
    return np.array(replicated, dtype=np.float32)

def sort_points_by_angle(points):
    """
    Sorts a set of 2D points by their polar angle around the centroid.
    """
    if len(points) < 3:
        return points
    cx, cy = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    idx = np.argsort(angles)
    return points[idx]

def polygon_to_string(polygon):
    """
    Converts an array of vertices (Nx2) to a semicolon-separated string.
    """
    return ";".join([f"{v[0]:.6f},{v[1]:.6f}" for v in polygon])

############################################
# Neural Network Models
############################################

class ShapeToSpectraModel(nn.Module):
    """
    Neural network that predicts optical spectra from shape representations.
    """
    def __init__(self, d_in=3, d_model=256, nhead=4, num_layers=4):
        super(ShapeToSpectraModel, self).__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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

        out_2d = torch.sigmoid(out_2d)
        return out_2d

class Spectra2ShapeVarLen(nn.Module):
    """
    Neural network that predicts shape representations from optical spectra.
    """
    def __init__(self, d_in=100, d_model=256, nhead=4, num_layers=4):
        super(Spectra2ShapeVarLen, self).__init__()
        self.row_preproc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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

############################################
# Processing Functions
############################################

def process_sample(idx, uid, gt_spectra, gt_shape, spec2shape_model, shape2spec_model, device):
    """Process a single sample and return data for visualization"""
    # Convert ground-truth spectrum to tensor and add batch dimension
    spec_tensor = torch.tensor(gt_spectra, dtype=torch.float32).unsqueeze(0).to(device)  # (1,11,100)
    
    # Predict shape from spectrum using spec2shape model
    with torch.no_grad():
        pred_shape = spec2shape_model(spec_tensor)  # (1,4,3)
        pred_spec_net = shape2spec_model(pred_shape)  # (1,11,100)
        
        # Also compute network spectrum of GT shape
        gt_shape_tensor = torch.tensor(gt_shape, dtype=torch.float32).unsqueeze(0).to(device)  # (1,4,3)
        gt_spec_net = shape2spec_model(gt_shape_tensor)  # (1,11,100)
    
    pred_shape_np = pred_shape.squeeze(0).cpu().numpy()      # (4, 3)
    pred_spec_net_np = pred_spec_net.squeeze(0).cpu().numpy()  # (11, 100)
    gt_spec_net_np = gt_spec_net.squeeze(0).cpu().numpy()    # (11, 100)
    
    # Process ground-truth shape
    valid_gt = gt_shape[:, 0] > 0.5
    if np.sum(valid_gt) > 0:
        gt_q1 = gt_shape[valid_gt, 1:3]
        gt_full = replicate_c4(gt_q1)
        gt_full = sort_points_by_angle(gt_full)
    else:
        gt_full = None
        gt_q1 = None
    
    # Process predicted shape
    valid_pred = pred_shape_np[:, 0] > 0.5
    if np.sum(valid_pred) > 0:
        pred_q1 = pred_shape_np[valid_pred, 1:3]
        pred_full = replicate_c4(pred_q1)
        pred_full = sort_points_by_angle(pred_full)
    else:
        pred_full = None
        pred_q1 = None
    
    return {
        'uid': uid,
        'gt_spectra': gt_spectra,
        'gt_spec_net_np': gt_spec_net_np,
        'pred_spec_net_np': pred_spec_net_np,
        'gt_full': gt_full,
        'gt_q1': gt_q1,
        'pred_full': pred_full,
        'pred_q1': pred_q1
    }

def run_s4_simulations(polygon_str, uid, max_workers=4):
    """Run S4 simulations for all c values in parallel"""
    s4_spectra_list = []
    c_values = np.linspace(0.0, 1.0, 11)
    
    # Use ThreadPoolExecutor to run up to max_workers S4 processes concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map returns results in order
        results = list(executor.map(lambda c: (c, run_s4_for_c(polygon_str, c)), c_values))
    
    for c, s4_csv in results:
        if s4_csv is not None:
            wv_s4, R_s4, T_s4 = read_results_csv(s4_csv)
            # Store transmittance (T) values instead of reflectance (R)
            s4_spectra_list.append((c, np.array(T_s4)))
        else:
            print(f"[ERROR] S4 failed for c = {c:.1f} for UID {uid}")
            s4_spectra_list.append((c, None))
    
    return s4_spectra_list

def plot_sample(axes, col, sample_data, s4_spectra_list, fig=None):
    """Plot the data for a single sample in a column of the figure"""
    uid = sample_data['uid']
    gt_spectra = sample_data['gt_spectra']
    gt_spec_net_np = sample_data['gt_spec_net_np']
    pred_spec_net_np = sample_data['pred_spec_net_np']
    gt_full = sample_data['gt_full']
    gt_q1 = sample_data['gt_q1']
    pred_full = sample_data['pred_full']
    pred_q1 = sample_data['pred_q1']
    
    x_axis = np.arange(1, 101)
    
    # Define colors for consistency
    c_values = np.linspace(0.0, 1.0, 11)
    viridis_colors = plt.cm.viridis(np.linspace(0, 1, len(c_values)))
    inferno_colors = plt.cm.inferno(np.linspace(0, 1, len(c_values)))
    
    # Row 1: Overlaid shape plot (GT in green, predicted in red)
    ax_shape = axes[0, col]
    if gt_full is not None:
        closed_gt = np.concatenate([gt_full, gt_full[0:1]], axis=0)
        ax_shape.plot(closed_gt[:, 0], closed_gt[:, 1], 'g-', linewidth=2, label="GT")
        ax_shape.scatter(gt_q1[:, 0], gt_q1[:, 1], color='green', s=50)
    if pred_full is not None:
        closed_pred = np.concatenate([pred_full, pred_full[0:1]], axis=0)
        ax_shape.plot(closed_pred[:, 0], closed_pred[:, 1], 'r-', linewidth=2, label="Pred")
        ax_shape.scatter(pred_q1[:, 0], pred_q1[:, 1], color='red', s=50)
    ax_shape.set_title(f"Sample {uid.split('_')[-1]}", fontsize=12, fontweight='bold')
    ax_shape.set_xlabel("X", fontsize=10)
    ax_shape.set_ylabel("Y", fontsize=10)
    ax_shape.axis("equal")
    ax_shape.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend in the best location
    legend = ax_shape.legend(fontsize=9, frameon=True)
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('lightgray')
    
    # Adjust limits to ensure shapes are centered and properly scaled
    x_min, x_max = ax_shape.get_xlim()
    y_min, y_max = ax_shape.get_ylim()
    x_lim = max(0.6, max(abs(x_min), abs(x_max)))
    y_lim = max(0.6, max(abs(y_min), abs(y_max)))
    max_lim = max(x_lim, y_lim)
    ax_shape.set_xlim(-max_lim, max_lim)
    ax_shape.set_ylim(-max_lim, max_lim)
    
    # Row 2: Ground-truth spectrum (all 11 rows) - TRANSMITTANCE
    ax_gt_spec = axes[1, col]
    for i, c in enumerate(c_values):
        ax_gt_spec.plot(x_axis, gt_spectra[i], color=viridis_colors[i], label=f"c={c:.1f}")
    # ax_gt_spec.set_title("GT Spectra (Measured)", fontsize=11)
    ax_gt_spec.set_title("GT Shape → S4 Spectra", fontsize=11)
    ax_gt_spec.set_xlabel("Wavelength Index", fontsize=10)
    ax_gt_spec.set_ylabel("Transmittance", fontsize=10)
    ax_gt_spec.grid(True, linestyle='--', alpha=0.3)
    ax_gt_spec.set_ylim(0, 1)
    
    # Row 3: Network spectrum of GT shape - using inferno colormap
    ax_gt_net_spec = axes[2, col]
    for i, c in enumerate(c_values):
        ax_gt_net_spec.plot(x_axis, gt_spec_net_np[i], color=inferno_colors[i], label=f"c={c:.1f}")
    ax_gt_net_spec.set_title("GT Shape → Network Spectra", fontsize=11)
    ax_gt_net_spec.set_xlabel("Wavelength Index", fontsize=10)
    ax_gt_net_spec.set_ylabel("Transmittance", fontsize=10)
    ax_gt_net_spec.grid(True, linestyle='--', alpha=0.3)
    ax_gt_net_spec.set_ylim(0, 1)
    
    # Row 4: S4 spectra of predicted shape (all c values) - TRANSMITTANCE
    ax_s4_spec = axes[3, col]
    for i, (c, s4_spec) in enumerate(s4_spectra_list):
        if s4_spec is not None:
            ax_s4_spec.plot(x_axis, s4_spec, color=viridis_colors[i], label=f"c={c:.1f}")
    ax_s4_spec.set_title("Pred Shape → S4 Spectra", fontsize=11)
    ax_s4_spec.set_xlabel("Wavelength Index", fontsize=10)
    ax_s4_spec.set_ylabel("Transmittance", fontsize=10)
    ax_s4_spec.grid(True, linestyle='--', alpha=0.3)
    ax_s4_spec.set_ylim(0, 1)
    
    # Row 5: Network output spectrum of predicted shape - using inferno colormap
    ax_pred_net_spec = axes[4, col]
    for i, c in enumerate(c_values):
        ax_pred_net_spec.plot(x_axis, pred_spec_net_np[i], color=inferno_colors[i], label=f"c={c:.1f}")
    ax_pred_net_spec.set_title("Pred Shape → Network Spectra", fontsize=11)
    ax_pred_net_spec.set_xlabel("Wavelength Index", fontsize=10)
    ax_pred_net_spec.set_ylabel("Transmittance", fontsize=10)
    ax_pred_net_spec.grid(True, linestyle='--', alpha=0.3)
    ax_pred_net_spec.set_ylim(0, 1)
    
    # Add a common legend for all spectrum plots on the right side outside the figure
    if col == 3 and fig is not None:  # Only for the last column
        handles, labels = ax_pred_net_spec.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='center right', 
                           bbox_to_anchor=(1.12, 0.5), ncol=1, fontsize=9)
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('lightgray')

############################################
# Main Script
############################################

def main():
    # Parse arguments
    args = parse_args()
    
    # Set fixed seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load preprocessed NPZ file
    if not os.path.exists(args.npz_file):
        print(f"Error: NPZ file not found: {args.npz_file}")
        sys.exit(1)
    data = np.load(args.npz_file, allow_pickle=True)
    uids = data["uids"]
    spectra = data["spectra"]  # shape: (N, 11, 100)
    shapes = data["shapes"]    # shape: (N, 4, 3)
    total_samples = uids.shape[0]
    print(f"[INFO] Loaded {total_samples} samples from {args.npz_file}")
    
    # Choose random sample indices from the entire dataset
    if total_samples < args.n_samples:
        print(f"Error: Dataset has less than {args.n_samples} samples.")
        sys.exit(1)
    # chosen_indices = random.sample(range(total_samples), args.n_samples)
    # chosen_indices = [5000, 80000+15000, 160000+25000, 240000+35000]
    chosen_indices = [8000, 18000, 28000, 38000]
    chosen_indices = [8888, 18888, 28888, 38888]
    print(f"[INFO] Selected {args.n_samples} random samples: {chosen_indices}")
    
    # Load the trained models
    if not os.path.exists(args.spec2shape_ckpt):
        print(f"Error: spec2shape checkpoint not found: {args.spec2shape_ckpt}")
        sys.exit(1)
    if not os.path.exists(args.shape2spec_ckpt):
        print(f"Error: shape2spec checkpoint not found: {args.shape2spec_ckpt}")
        sys.exit(1)
    
    spec2shape_model = Spectra2ShapeVarLen(d_in=100, d_model=256, nhead=4, num_layers=4).to(device)
    spec2shape_model.load_state_dict(torch.load(args.spec2shape_ckpt, map_location=device))
    spec2shape_model.eval()
    
    shape2spec_model = ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4).to(device)
    shape2spec_model.load_state_dict(torch.load(args.shape2spec_ckpt, map_location=device))
    shape2spec_model.eval()
    print(f"[INFO] Loaded models successfully")
    
    # Create an output folder with a datetime stamp
    if args.out_folder:
        out_folder = args.out_folder
    else:
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_folder = f"FilterShapeS4_Evaluator_Transmittance_{dt_str}"
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    print(f"[INFO] Output folder: {out_folder}")
    
    # Group the chosen indices into groups of 4 (each figure will show up to 4 samples)
    groups = [chosen_indices[i:i+4] for i in range(0, len(chosen_indices), 4)]
    
    # For each group, create a 5-row x 4-column figure with optimized layout
    for g_idx, group in enumerate(groups):
        # Create figure with better proportions for publication
        fig, axes = plt.subplots(5, 4, figsize=(14, 16), constrained_layout=True)
        
        # Set overall figure title
        # fig.suptitle("Metasurface Shape and Transmittance Evaluation", 
                    # fontsize=16, fontweight='bold', y=0.98)
        
        # Add timestamp and group info as a subtitle
        dt_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        # fig.text(0.5, 0.955, f"Group {g_idx+1} | Generated: {dt_str}", 
                # fontsize=10, ha='center')
        
        # axes is a 5x4 array; we'll fill one column per sample in this group.
        # If a column is not used, turn off those subplots.
        for col in range(4):
            if col >= len(group):
                for row in range(5):
                    axes[row, col].axis("off")
                continue
            
            idx = group[col]
            uid = uids[idx]
            gt_spectra = spectra[idx]  # (11, 100)
            gt_shape = shapes[idx]     # (4, 3)
            print(f"[INFO] Group {g_idx+1} - Processing sample UID: {uid}")
            
            # Process sample to get shapes and network predictions
            sample_data = process_sample(idx, uid, gt_spectra, gt_shape, spec2shape_model, shape2spec_model, device)
            
            # If we have a valid predicted shape, run S4 simulations
            s4_spectra_list = []
            if sample_data['pred_full'] is not None:
                polygon_str = polygon_to_string(sample_data['pred_full'])
                s4_spectra_list = run_s4_simulations(polygon_str, uid, args.max_workers)
            else:
                print(f"[ERROR] Predicted shape is empty for UID {uid}. Skipping sample.")
                continue
            
            # Plot the sample data
            plot_sample(axes, col, sample_data, s4_spectra_list, fig)
        
        plt.tight_layout()
        outpng = os.path.join(out_folder, f"figure_group_{g_idx+1}.png")
        plt.savefig(outpng, dpi=150)
        
        # Save data alongside figure
        outdata = os.path.join(out_folder, f"figure_group_{g_idx+1}_data.npz")
        group_data = {
            "indices": group,
            "uids": [uids[idx] for idx in group if idx < total_samples],
            "gt_spectra": [spectra[idx] for idx in group if idx < total_samples],
            "gt_shapes": [shapes[idx] for idx in group if idx < total_samples],
            "c_values": np.linspace(0.0, 1.0, 11).tolist(),
            "wavelength_indices": list(range(1, 101))
        }
        np.savez(outdata, **group_data)
        
        # Also save a high-resolution version for publication
        outpng_hires = os.path.join(out_folder, f"figure_group_{g_idx+1}_hires.png")
        plt.savefig(outpng_hires, dpi=300)
        
        # Save in PDF format for vector graphics
        outpdf = os.path.join(out_folder, f"figure_group_{g_idx+1}.pdf")
        plt.savefig(outpdf, format='pdf')
        
        plt.close(fig)
        print(f"[INFO] Saved figure group {g_idx+1} to {outpng}")
        print(f"[INFO] Saved figure data to {outdata}")
    
    print(f"[INFO] All figures saved in folder: {out_folder}")

if __name__ == "__main__":
    main()
