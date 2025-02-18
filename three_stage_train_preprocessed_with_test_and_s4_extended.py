#!/usr/bin/env python3
import os
import sys
import csv
import random
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset

############################################
# S4 and Utility Functions (using os.popen)
############################################

def run_s4_for_c(polygon_str, c_val):
    """
    Runs the S4 binary using os.popen() with the given polygon string and c value.
    Returns the CSV output file path as a string (if found) or None.
    """
    cmd = f'../build/S4 -a "{polygon_str} -c {c_val} -v -s" metasurface_fixed_shape_and_c_value.lua'
    print(f"[DEBUG] S4 command: {cmd}")
    output = os.popen(cmd).read()
    if not output:
        print("[ERROR] S4 produced no output.")
        return None
    saved_path = None
    for line in output.splitlines():
        if "Saved to " in line:
            saved_path = line.split("Saved to", 1)[1].strip()
            break
    return saved_path

def read_results_csv(csv_path):
    """
    Reads S4 CSV output and returns lists of wavelengths, reflectance (R), and transmission (T).
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
                pass
    # Sort by wavelength
    data = sorted(zip(wv, Rv, Tv), key=lambda x: x[0])
    wv = [d[0] for d in data]
    Rv = [d[1] for d in data]
    Tv = [d[2] for d in data]
    return wv, Rv, Tv

def replicate_c4(points):
    """
    Given an array of Q1 vertices (Nx2), replicates them using C4 symmetry.
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
    Sorts a set of 2D points by their polar angle about the centroid.
    """
    if len(points) < 3:
        return points
    cx, cy = points.mean(axis=0)
    angles = np.arctan2(points[:,1] - cy, points[:,0] - cx)
    idx = np.argsort(angles)
    return points[idx]

def polygon_to_string(polygon):
    """
    Converts an array of vertices (Nx2) to a semicolon-separated string.
    """
    return ";".join([f"{v[0]:.6f},{v[1]:.6f}" for v in polygon])

############################################
# NPZ Dataset Functions
############################################

def load_npz(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    return data["uids"], data["spectra"], data["shapes"]

############################################
# Dataset Classes
############################################

class PreprocessedSpectraDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file, allow_pickle=True)
        self.uids = data["uids"]
        self.spectra = data["spectra"]
        self.shapes = data["shapes"]
    def __len__(self):
        return len(self.uids)
    def __getitem__(self, idx):
        spec = self.spectra[idx]
        shape = self.shapes[idx]
        uid = self.uids[idx]
        spec_tensor = torch.tensor(spec, dtype=torch.float32)
        shape_tensor = torch.tensor(shape, dtype=torch.float32)
        return spec_tensor, shape_tensor, uid

############################################
# Model Definitions
############################################

class ShapeToSpectraModel(nn.Module):
    def __init__(self, d_in=3, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, 11*100)
        )
    def forward(self, shape_4x3):
        bsz = shape_4x3.size(0)
        presence = shape_4x3[:,:,0]
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
            dim_feedforward=d_model*4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 12)
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
        presence_logits = out_4x3[:,:,0]
        xy_raw = out_4x3[:,:,1:]
        presence_list = []
        for i in range(4):
            if i==0:
                presence_list.append(torch.ones(bsz, device=out_4x3.device, dtype=torch.float32))
            else:
                prob_i = torch.sigmoid(presence_logits[:,i]).clamp(1e-6, 1-1e-6)
                prob_chain = prob_i * presence_list[i-1]
                ste_i = (prob_chain > 0.5).float() + prob_chain - prob_chain.detach()
                presence_list.append(ste_i)
        presence_stack = torch.stack(presence_list, dim=1)
        xy_bounded = torch.sigmoid(xy_raw)
        xy_final = xy_bounded * presence_stack.unsqueeze(-1)
        final_shape = torch.cat([presence_stack.unsqueeze(-1), xy_final], dim=-1)
        return final_shape

class Spec2ShapeFrozen(nn.Module):
    def __init__(self, spec2shape_net, shape2spec_frozen, no_grad_frozen=True):
        super().__init__()
        self.spec2shape = spec2shape_net
        self.shape2spec_frozen = shape2spec_frozen
        self.no_grad_frozen = no_grad_frozen
        for p in self.shape2spec_frozen.parameters():
            p.requires_grad = False
    def forward(self, spec_input):
        shape_pred = self.spec2shape(spec_input)
        if self.no_grad_frozen:
            with torch.no_grad():
                spec_chain = self.shape2spec_frozen(shape_pred)
        else:
            spec_chain = self.shape2spec_frozen(shape_pred)
        return shape_pred, spec_chain

############################################
# Argument Parsing and Main Test Function
############################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test mode: Compare GT spectrum/shape vs. model-predicted shape/spectrum and S4 reflectance from predicted shape."
    )
    parser.add_argument("--test", metavar="ROOT", type=str, help="Folder containing model checkpoints (e.g. outputs_three_stage_20250216_180408)")
    parser.add_argument("--run-s4", action="store_true", help="Run S4 to generate spectrum from predicted shape")
    parser.add_argument("--data_npz", type=str, default="", help="Path to preprocessed NPZ dataset")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to test")
    parser.add_argument("--c", type=float, default=0.1, help="c value for S4 command (base value; individual subplots use c = i/10)")
    return parser.parse_args()

def run_test_mode(args):
    # Load model checkpoints from the test folder.
    if not args.test:
        print("[ERROR] Provide the model folder with --test.")
        sys.exit(1)
    test_root = args.test
    shape2spec_ckpt = os.path.join(test_root, "stageA", "shape2spec_stageA.pt")
    spec2shape_ckpt = os.path.join(test_root, "stageC", "spec2shape_stageC.pt")
    if not os.path.exists(shape2spec_ckpt) or not os.path.exists(spec2shape_ckpt):
        print("[ERROR] Checkpoints not found in the provided test folder.")
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    # Load models
    spec2shape_net = Spectra2ShapeVarLen(d_in=100, d_model=256, nhead=4, num_layers=4).to(device)
    shape2spec_net = ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4).to(device)
    spec2shape_net.load_state_dict(torch.load(spec2shape_ckpt, map_location=device))
    shape2spec_net.load_state_dict(torch.load(shape2spec_ckpt, map_location=device))
    spec2shape_net.eval()
    shape2spec_net.eval()
    pipeline = Spec2ShapeFrozen(spec2shape_net, shape2spec_net, no_grad_frozen=True).to(device)
    pipeline.eval()
    
    # Load dataset
    if args.data_npz and os.path.exists(args.data_npz):
        uids, spectra, shapes = load_npz(args.data_npz)
        dataset = PreprocessedSpectraDataset(args.data_npz)
        print(f"[INFO] Loaded NPZ dataset with {len(dataset)} samples.")
    else:
        print("[ERROR] You must provide a valid --data_npz file.")
        sys.exit(1)
    
    # Randomly select num_samples samples
    total_samples = len(dataset)
    sample_indices = random.sample(range(total_samples), args.num_samples)
    
    # Create output folder with timestamp
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("test_outputs", dt_str)
    os.makedirs(out_dir, exist_ok=True)
    
    # Define a uniform x-axis for plotting (simulate indices 1-100)
    x_axis = np.arange(1, 101)
    
    for idx in sample_indices:
        spec_gt, shape_gt, uid = dataset[idx]
        print(f"\n[INFO] Processing sample UID: {uid}")
        
        # Run pipeline on GT spectrum to get predicted shape and predicted spectrum.
        spec_input = spec_gt.unsqueeze(0).to(device)
        with torch.no_grad():
            shape_pred, spec_pred = pipeline(spec_input)
        shape_pred_np = shape_pred.cpu().numpy()[0]  # (4,3)
        spec_pred_np = spec_pred.cpu().numpy()[0]      # (11,100)
        
        # Reconstruct polygon from predicted shape.
        valid_pred = shape_pred_np[:,0] > 0.5
        if not np.any(valid_pred):
            print(f"[WARN] No valid predicted vertices for UID {uid}. Skipping sample.")
            continue
        pred_q1 = shape_pred_np[valid_pred, 1:3]
        full_pred_polygon = replicate_c4(pred_q1)
        full_pred_polygon = sort_points_by_angle(full_pred_polygon)
        polygon_str_pred = polygon_to_string(full_pred_polygon)
        print(f"[DEBUG] UID {uid} - S4 polygon string (predicted): {polygon_str_pred}")
        
        # (Optionally, you can also reconstruct the GT polygon if desired)
        valid_gt = (shape_gt[:,0] > 0.5).numpy()
        if not np.any(valid_gt):
            print(f"[WARN] No valid GT vertices for UID {uid}.")
            continue
        gt_q1 = shape_gt.numpy()[valid_gt, 1:3]
        full_gt_polygon = replicate_c4(gt_q1)
        full_gt_polygon = sort_points_by_angle(full_gt_polygon)
        polygon_str_gt = polygon_to_string(full_gt_polygon)
        print(f"[DEBUG] UID {uid} - S4 polygon string (GT): {polygon_str_gt}")
        
        # Now, for each of the 11 rows (assume these correspond to different c values: 0.0, 0.1, …, 1.0)
        fig_spec, axes_spec = plt.subplots(11, 1, figsize=(10, 3*11), sharex=True)
        if axes_spec.ndim == 1:
            axes_spec = axes_spec
        for i in range(11):
            c_val = i / 10.0
            ax = axes_spec[i]
            # Plot the ground-truth spectrum (from NPZ) for row i.
            gt_line = spec_gt.numpy()[i]
            ax.plot(x_axis, gt_line, 'b-', label=f"GT Spectrum (c={c_val:.2f})")
            # Plot the predicted spectrum (from model) for row i.
            pred_line = spec_pred_np[i]
            ax.plot(x_axis, pred_line, 'r--', label=f"Predicted Spectrum (c={c_val:.2f})")
            # Run S4 on the predicted polygon with this c value.
            s4_csv_path = None
            if args.run_s4:
                s4_csv_path = run_s4_for_c(polygon_str_pred, c_val)
            if s4_csv_path:
                s4_wv, s4_R, s4_T = read_results_csv(s4_csv_path)
                # For plotting, force a uniform x-axis (1..100)
                ax.plot(x_axis, s4_R, 'g-.', label=f"S4 (Pred shape) R (c={c_val:.2f})")
            else:
                ax.text(0.5, 0.5, "S4 failed", transform=ax.transAxes, color='red')
            ax.set_ylabel("Reflectance")
            ax.set_title(f"c = {c_val:.2f}")
            ax.legend(fontsize=8)
        plt.xlabel("Uniform X-axis (1-100)")
        plt.suptitle(f"UID {uid}: Spectrum Comparison\n(Blue: GT, Red: Model, Green: S4 on Predicted)", fontsize=16)
        outpng_spec = os.path.join(out_dir, f"spectrum_comparison_{uid}.png")
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(outpng_spec, dpi=150)
        plt.close(fig_spec)
        print(f"[INFO] Saved spectrum comparison figure for UID {uid} to {outpng_spec}")
        
        # Also, create a 2x2 figure for shape and separate S4 spectrum plots.
        fig_shape, axes_shape = plt.subplots(2, 2, figsize=(16, 12))
        # Plot GT shape
        axes_shape[0,0].scatter(full_gt_polygon[:,0], full_gt_polygon[:,1], color='blue', label='GT Shape', zorder=5)
        plot_polygon(axes_shape[0,0], full_gt_polygon, color='blue', alpha=0.4, fill=False)
        axes_shape[0,0].set_title(f"GT Shape for UID {uid}")
        axes_shape[0,0].set_xlabel("X")
        axes_shape[0,0].set_ylabel("Y")
        axes_shape[0,0].legend()
        # Plot predicted shape
        axes_shape[0,1].scatter(full_pred_polygon[:,0], full_pred_polygon[:,1], color='red', label='Predicted Shape', zorder=5)
        plot_polygon(axes_shape[0,1], full_pred_polygon, color='red', alpha=0.4, fill=False)
        axes_shape[0,1].set_title(f"Predicted Shape for UID {uid}")
        axes_shape[0,1].set_xlabel("X")
        axes_shape[0,1].set_ylabel("Y")
        axes_shape[0,1].legend()
        # Plot S4 result for GT shape (using args.c for demonstration)
        s4_csv_gt = None
        if args.run_s4:
            s4_csv_gt = run_s4_for_c(polygon_str_gt, args.c)
        if s4_csv_gt:
            wv_gt, R_gt, T_gt = read_results_csv(s4_csv_gt)
            axes_shape[1,0].plot(np.arange(1, len(wv_gt)+1), R_gt, 'g-', label="S4 R (GT)")
            axes_shape[1,0].set_title(f"S4 Spectrum (GT Shape) for UID {uid} at c={args.c:.2f}")
        else:
            axes_shape[1,0].text(0.5,0.5,"S4 GT failed", transform=axes_shape[1,0].transAxes, color='red')
        axes_shape[1,0].set_xlabel("Uniform X-axis")
        axes_shape[1,0].set_ylabel("Reflectance")
        axes_shape[1,0].legend()
        # Plot S4 result for predicted shape (using args.c)
        s4_csv_pred = None
        if args.run_s4:
            s4_csv_pred = run_s4_for_c(polygon_str_pred, args.c)
        if s4_csv_pred:
            wv_pred, R_pred, T_pred = read_results_csv(s4_csv_pred)
            axes_shape[1,1].plot(np.arange(1, len(wv_pred)+1), R_pred, 'g-', label="S4 R (Pred)")
            axes_shape[1,1].set_title(f"S4 Spectrum (Predicted Shape) for UID {uid} at c={args.c:.2f}")
        else:
            axes_shape[1,1].text(0.5,0.5,"S4 Pred failed", transform=axes_shape[1,1].transAxes, color='red')
        axes_shape[1,1].set_xlabel("Uniform X-axis")
        axes_shape[1,1].set_ylabel("Reflectance")
        axes_shape[1,1].legend()
        plt.suptitle(f"UID {uid} Shape and S4 Spectrum Comparison", fontsize=16)
        outpng_shape = os.path.join(out_dir, f"shape_s4_comparison_{uid}.png")
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(outpng_shape, dpi=150)
        plt.close(fig_shape)
        print(f"[INFO] Saved 2x2 shape/spectrum comparison figure for UID {uid} to {outpng_shape}")

def main():
    args = parse_args()
    if args.test:
        run_test_mode(args)
    else:
        print("No test mode specified. Use --test with the model folder.")
        sys.exit(0)

if __name__ == "__main__":
    main()

