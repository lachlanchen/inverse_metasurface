#!/usr/bin/env python3
import os
import sys
import csv
import random
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

############################################
# S4 and Utility Functions
############################################

def run_s4_for_c(polygon_str, c_val):
    """
    Run the S4 binary with the given polygon string and c value
    using os.popen(). Returns the path to the CSV file with results (or None if not found).
    """
    cmd = f'../build/S4 -a "{polygon_str} -c {c_val} -v -s" metasurface_fixed_shape_and_c_value.lua'
    print(f"[DEBUG] S4 command: {cmd}")
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
        return out_2d

class Spectra2ShapeVarLen(nn.Module):
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
# Main Script: Process 10 Random Samples
############################################

def main():
    # Set fixed seed for reproducibility
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load preprocessed NPZ file (adjust path if needed)
    npz_file = "preprocessed_data.npz"
    if not os.path.exists(npz_file):
        print("Error: NPZ file not found:", npz_file)
        sys.exit(1)
    data = np.load(npz_file, allow_pickle=True)
    uids = data["uids"]
    spectra = data["spectra"]  # shape: (N, 11, 100)
    shapes = data["shapes"]    # shape: (N, 4, 3)
    total_samples = uids.shape[0]
    
    n_samples = 4
    # Choose 10 random sample indices from the entire dataset
    if total_samples < n_samples:
        print("Error: Dataset has less than 10 samples.")
        sys.exit(1)
    chosen_indices = random.sample(range(total_samples), n_samples)
    
    # Load the trained models (adjust paths if necessary)
    spec2shape_ckpt = "outputs_three_stage_20250216_180408/stageB/spec2shape_stageB.pt"
    shape2spec_ckpt = "outputs_three_stage_20250216_180408/stageA/shape2spec_stageA.pt"
    
    if not os.path.exists(spec2shape_ckpt):
        print("Error: spec2shape checkpoint not found:", spec2shape_ckpt)
        sys.exit(1)
    if not os.path.exists(shape2spec_ckpt):
        print("Error: shape2spec checkpoint not found:", shape2spec_ckpt)
        sys.exit(1)
    
    spec2shape_model = Spectra2ShapeVarLen(d_in=100, d_model=256, nhead=4, num_layers=4).to(device)
    spec2shape_model.load_state_dict(torch.load(spec2shape_ckpt, map_location=device))
    spec2shape_model.eval()
    
    shape2spec_model = ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4).to(device)
    shape2spec_model.load_state_dict(torch.load(shape2spec_ckpt, map_location=device))
    shape2spec_model.eval()
    
    # Create an output folder with a datetime stamp
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_folder = f"SpecShapeS4_Evaluator_{dt_str}"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # Loop over each chosen sample
    for idx in chosen_indices:
        uid = uids[idx]
        gt_spectra = spectra[idx]  # (11, 100)
        gt_shape = shapes[idx]     # (4, 3)
        print(f"[INFO] Processing sample UID: {uid}")
        
        # Convert ground-truth spectrum to tensor and add batch dimension.
        spec_tensor = torch.tensor(gt_spectra, dtype=torch.float32).unsqueeze(0).to(device)  # shape: (1,11,100)
        
        # Predict shape from spectrum using spec2shape model.
        with torch.no_grad():
            pred_shape = spec2shape_model(spec_tensor)  # (1, 4, 3)
            pred_spec_net = shape2spec_model(pred_shape)  # (1, 11, 100)
        pred_shape_np = pred_shape.squeeze(0).cpu().numpy()      # (4, 3)
        pred_spec_net_np = pred_spec_net.squeeze(0).cpu().numpy()  # (11, 100)
        
        # Process ground-truth shape: select valid vertices (first column > 0.5)
        valid_gt = gt_shape[:, 0] > 0.5
        if np.sum(valid_gt) > 0:
            gt_q1 = gt_shape[valid_gt, 1:3]
            gt_full = replicate_c4(gt_q1)
            gt_full = sort_points_by_angle(gt_full)
        else:
            gt_full = None
        
        # Process predicted shape similarly.
        valid_pred = pred_shape_np[:, 0] > 0.5
        if np.sum(valid_pred) > 0:
            pred_q1 = pred_shape_np[valid_pred, 1:3]
            pred_full = replicate_c4(pred_q1)
            pred_full = sort_points_by_angle(pred_full)
        else:
            pred_full = None
        
        # Run S4 for all 11 c values (from 0.0 to 1.0 in steps of 0.1) using the predicted shape.
        s4_spectra_list = []
        if pred_full is not None:
            polygon_str = polygon_to_string(pred_full)
            c_values = np.linspace(0.0, 1.0, 11)
            for c in c_values:
                s4_csv = run_s4_for_c(polygon_str, c)
                if s4_csv is not None:
                    wv_s4, R_s4, T_s4 = read_results_csv(s4_csv)
                    s4_spectra_list.append((c, np.array(R_s4)))
                else:
                    print(f"[ERROR] S4 failed for c = {c:.1f} for UID {uid}")
                    s4_spectra_list.append((c, None))
        else:
            print(f"[ERROR] Predicted shape is empty for UID {uid}. Skipping sample.")
            continue
        
        # Prepare x-axis (assumed 1 to 100)
        x_axis = np.arange(1, 101)
        
        # Create a figure with three rows and two columns:
        # Column 1: shape plots; Column 2: spectra plots.
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        
        # Row 1: Ground-truth
        ax_shape_gt = axes[0, 0]
        if gt_full is not None:
            closed_gt = np.concatenate([gt_full, gt_full[0:1]], axis=0)
            ax_shape_gt.plot(closed_gt[:, 0], closed_gt[:, 1], 'g-', linewidth=2)
            ax_shape_gt.scatter(gt_q1[:, 0], gt_q1[:, 1], color='green', s=50, label='GT vertices')
        ax_shape_gt.set_title("Ground-truth Shape (full polygon)")
        ax_shape_gt.set_xlabel("X")
        ax_shape_gt.set_ylabel("Y")
        ax_shape_gt.axis("equal")
        ax_shape_gt.grid(True)
        
        ax_spec_gt = axes[0, 1]
        for i in range(gt_spectra.shape[0]):
            ax_spec_gt.plot(x_axis, gt_spectra[i], label=f"c={i/10:.2f}")
        ax_spec_gt.set_title("Ground-truth Spectra (11 rows)")
        ax_spec_gt.set_xlabel("Wavelength index")
        ax_spec_gt.set_ylabel("Reflectance")
        ax_spec_gt.legend(fontsize=8)
        
        # Row 2: Neural network prediction (shape2spec output)
        ax_shape_pred = axes[1, 0]
        if pred_full is not None:
            closed_pred = np.concatenate([pred_full, pred_full[0:1]], axis=0)
            ax_shape_pred.plot(closed_pred[:, 0], closed_pred[:, 1], 'r-', linewidth=2)
            ax_shape_pred.scatter(pred_q1[:, 0], pred_q1[:, 1], color='red', s=50, label='Predicted vertices')
        ax_shape_pred.set_title("Predicted Shape (from spec2shape)")
        ax_shape_pred.set_xlabel("X")
        ax_shape_pred.set_ylabel("Y")
        ax_shape_pred.axis("equal")
        ax_shape_pred.grid(True)
        
        ax_spec_pred = axes[1, 1]
        for i in range(pred_spec_net_np.shape[0]):
            ax_spec_pred.plot(x_axis, pred_spec_net_np[i], label=f"c={i/10:.2f}")
        ax_spec_pred.set_title("Network Output Spectrum (shape2spec)")
        ax_spec_pred.set_xlabel("Wavelength index")
        ax_spec_pred.set_ylabel("Reflectance")
        ax_spec_pred.legend(fontsize=8)
        
        # Row 3: S4 prediction using the predicted shape for all c values
        ax_shape_s4 = axes[2, 0]
        if pred_full is not None:
            ax_shape_s4.plot(closed_pred[:, 0], closed_pred[:, 1], 'r-', linewidth=2)
            ax_shape_s4.scatter(pred_q1[:, 0], pred_q1[:, 1], color='red', s=50, label='Predicted vertices')
        ax_shape_s4.set_title("Predicted Shape (used for S4)")
        ax_shape_s4.set_xlabel("X")
        ax_shape_s4.set_ylabel("Y")
        ax_shape_s4.axis("equal")
        ax_shape_s4.grid(True)
        
        ax_spec_s4 = axes[2, 1]
        for (c, s4_spec) in s4_spectra_list:
            if s4_spec is not None:
                ax_spec_s4.plot(x_axis, s4_spec, label=f"c={c:.1f}")
        ax_spec_s4.set_title("S4 Spectra (using predicted shape) for all c values")
        ax_spec_s4.set_xlabel("Wavelength index")
        ax_spec_s4.set_ylabel("Reflectance")
        ax_spec_s4.legend(fontsize=8)
        
        plt.tight_layout()
        outpng = os.path.join(out_folder, f"check_npz_with_spec2shape2spec_{uid}.png")
        plt.savefig(outpng)
        # plt.savefig(outpng, dpi=100)
        plt.close(fig)
        print(f"[INFO] Plot saved for UID {uid} to {outpng}")
    
    print(f"[INFO] All figures saved in folder: {out_folder}")

if __name__ == "__main__":
    main()

