#!/usr/bin/env python3
import os
import sys
import csv
import glob
import random
import argparse
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

############################################
# S4 and Utility Functions
############################################

def run_s4_for_c(polygon_str, c_val):
    # Build command string
    cmd = f'../build/S4 -a "{polygon_str} -c {c_val} -v -s" metasurface_fixed_shape_and_c_value.lua'
    print(f"[DEBUG] S4 command: {cmd}")
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=10, preexec_fn=os.setsid)
    except subprocess.TimeoutExpired as e:
        print("[ERROR] S4 command timed out:", e)
        return None
    except Exception as e:
        print("[ERROR] Exception during S4 call:", e)
        return None
    if proc.returncode != 0:
        print("[ERROR] S4 run failed!")
        print("=== STDOUT ===")
        print(proc.stdout)
        print("=== STDERR ===")
        print(proc.stderr)
        return None
    saved_path = None
    for line in proc.stdout.splitlines():
        if "Saved to " in line:
            saved_path = line.split("Saved to",1)[1].strip()
            break
    return saved_path

def read_results_csv(csv_path):
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
    data = sorted(zip(wv,Rv,Tv), key=lambda x: x[0])
    wv = [d[0] for d in data]
    Rv = [d[1] for d in data]
    Tv = [d[2] for d in data]
    return wv, Rv, Tv

def replicate_c4(points):
    replicated = []
    for (x, y) in points:
        replicated.append([x, y])
        replicated.append([-y, x])
        replicated.append([-x, -y])
        replicated.append([y, -x])
    return np.array(replicated, dtype=np.float32)

def sort_points_by_angle(points):
    if len(points) < 3:
        return points
    cx, cy = points.mean(axis=0)
    angles = np.arctan2(points[:,1]-cy, points[:,0]-cx)
    idx = np.argsort(angles)
    return points[idx]

def plot_polygon(ax, points, color='green', alpha=0.4, fill=True):
    from matplotlib.path import Path
    import matplotlib.patches as patches
    if len(points) < 3:
        ax.scatter(points[:,0], points[:,1], c=color)
        return
    closed = np.concatenate([points, points[0:1]], axis=0)
    codes = [Path.MOVETO] + [Path.LINETO]*(len(points)-1) + [Path.CLOSEPOLY]
    path = Path(closed, codes)
    patch = patches.PathPatch(path, facecolor=color if fill else 'none',
                              alpha=alpha, edgecolor=color)
    ax.add_patch(patch)
    ax.autoscale_view()

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

class Q1ShiftedSpectraDataset(Dataset):
    def __init__(self, csv_file, max_points=4):
        self.df = pd.read_csv(csv_file)
        self.r_cols = [c for c in self.df.columns if c.startswith("R@")]
        if len(self.r_cols) == 0:
            raise ValueError("No reflectance columns found (R@...).")
        self.df["shape_uid"] = (self.df["prefix"].astype(str) + "_" +
                                self.df["nQ"].astype(str) + "_" +
                                self.df["nS"].astype(str) + "_" +
                                self.df["shape_idx"].astype(str))
        self.data_list = []
        grouped = self.df.groupby("shape_uid", sort=False)
        for uid, grp in grouped:
            if len(grp) != 11:
                continue
            grp_sorted = grp.sort_values(by="c")
            spec_11x100 = grp_sorted[self.r_cols].values.astype(np.float32)
            first_row = grp_sorted.iloc[0]
            v_str = str(first_row.get("vertices_str", "")).strip()
            if not v_str:
                continue
            raw_pairs = v_str.split(";")
            all_xy = []
            for pair in raw_pairs:
                pair = pair.strip()
                if pair:
                    xy = pair.split(",")
                    if len(xy)==2:
                        try:
                            x_val = float(xy[0])
                            y_val = float(xy[1])
                        except Exception:
                            continue
                        all_xy.append([x_val, y_val])
            all_xy = np.array(all_xy, dtype=np.float32)
            if len(all_xy)==0:
                continue
            shifted = all_xy - 0.5
            q1 = []
            for (xx,yy) in shifted:
                if xx>0 and yy>0:
                    q1.append([xx,yy])
            q1 = np.array(q1, dtype=np.float32)
            n_q1 = len(q1)
            if n_q1 < 1 or n_q1 > max_points:
                continue
            shape_4x3 = np.zeros((max_points,3), dtype=np.float32)
            for i in range(n_q1):
                shape_4x3[i,0] = 1.0
                shape_4x3[i,1] = q1[i,0]
                shape_4x3[i,2] = q1[i,1]
            self.data_list.append({
                "uid": uid,
                "spectra": spec_11x100,
                "shape": shape_4x3
            })
        self.data_len = len(self.data_list)
        if self.data_len==0:
            raise ValueError("No valid shapes found in the dataset.")
    def __len__(self):
        return self.data_len
    def __getitem__(self, idx):
        it = self.data_list[idx]
        spec = it["spectra"]
        shape = it["shape"]
        uid = it["uid"]
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
        description="Test mode: compare GT spectrum/shape, network-predicted spectrum/shape, and S4 spectrum (R only)."
    )
    parser.add_argument("--test", metavar="ROOT", type=str, help="Root folder containing model checkpoints for test mode")
    parser.add_argument("--run-s4", action="store_true", help="Run S4 to generate spectrum from full polygon")
    parser.add_argument("--data_npz", type=str, default="", help="Path to preprocessed NPZ dataset")
    parser.add_argument("--csv_file", type=str, default="", help="Path to CSV dataset")
    parser.add_argument("--c", type=float, default=0.1, help="c value for S4 command")
    return parser.parse_args()

def run_test_mode(args):
    if not args.test:
        print("[ERROR] In test mode, you must provide the root folder with --test.")
        sys.exit(1)
    test_root = args.test
    shape2spec_ckpt = os.path.join(test_root, "stageA", "shape2spec_stageA.pt")
    spec2shape_ckpt = os.path.join(test_root, "stageC", "spec2shape_stageC.pt")
    if not os.path.exists(shape2spec_ckpt):
        print(f"[ERROR] shape2spec checkpoint not found at {shape2spec_ckpt}")
        sys.exit(1)
    if not os.path.exists(spec2shape_ckpt):
        print(f"[ERROR] spec2shape checkpoint not found at {spec2shape_ckpt}")
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    shape2spec_net = ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4).to(device)
    spec2shape_net = Spectra2ShapeVarLen(d_in=100, d_model=256, nhead=4, num_layers=4).to(device)
    shape2spec_net.load_state_dict(torch.load(shape2spec_ckpt, map_location=device))
    spec2shape_net.load_state_dict(torch.load(spec2shape_ckpt, map_location=device))
    shape2spec_net.eval()
    spec2shape_net.eval()
    pipeline = Spec2ShapeFrozen(spec2shape_net, shape2spec_net, no_grad_frozen=True).to(device)
    pipeline.eval()
    if args.data_npz and os.path.exists(args.data_npz):
        dataset = PreprocessedSpectraDataset(args.data_npz)
        print(f"[INFO] Loaded preprocessed dataset from {args.data_npz} with {len(dataset)} samples.")
    elif args.csv_file and os.path.exists(args.csv_file):
        dataset = Q1ShiftedSpectraDataset(args.csv_file)
        print(f"[INFO] Loaded CSV dataset from {args.csv_file} with {len(dataset)} samples.")
    else:
        print("[ERROR] You must provide a valid --data_npz or --csv_file for testing.")
        sys.exit(1)
    # Select one sample for each nQ value in {1,2,3,4} (based on ground-truth shape)
    samples_by_nq = {}
    for i in range(len(dataset)):
        spec_gt, shape_gt, uid = dataset[i]
        gt_presence = (shape_gt[:, 0] > 0.5).numpy()
        n_q = int(np.sum(gt_presence))
        if n_q in [1, 2, 3, 4] and n_q not in samples_by_nq:
            samples_by_nq[n_q] = (spec_gt, shape_gt, uid)
        if len(samples_by_nq) >= 4:
            break
    if len(samples_by_nq) == 0:
        print("[ERROR] No samples with nQ in {1,2,3,4} found.")
        sys.exit(1)
    # Define a uniform x-axis (1 to 100)
    x_axis = np.arange(1, 101)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("test_outputs", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    for n_q, (spec_gt, shape_gt, uid) in samples_by_nq.items():
        print(f"[INFO] Processing UID {uid} with ground-truth nQ = {n_q}")
        spec_input = spec_gt.unsqueeze(0).to(device)
        with torch.no_grad():
            shape_pred, spec_pred = pipeline(spec_input)
        shape_pred_np = shape_pred.cpu().numpy()[0]
        spec_pred_np = spec_pred.cpu().numpy()[0]
        # -- For predicted shape --
        pred_presence = (shape_pred_np[:, 0] > 0.5)
        if not np.any(pred_presence):
            print(f"[WARN] No predicted points for UID {uid}. Skipping sample.")
            continue
        pred_q1 = shape_pred_np[pred_presence, 1:3]
        print(f"[DEBUG] UID {uid} - Predicted Q1 vertices: {pred_q1}")
        full_pred_polygon = replicate_c4(pred_q1)
        sorted_pred_polygon = sort_points_by_angle(full_pred_polygon)
        print(f"[DEBUG] UID {uid} - Full predicted polygon: {sorted_pred_polygon}")
        # Run S4 on predicted shape
        vertex_strs_pred = [f"{v[0]:.6f},{v[1]:.6f}" for v in sorted_pred_polygon]
        polygon_str_pred = ";".join(vertex_strs_pred)
        print(f"[DEBUG] UID {uid} - S4 polygon string (predicted): {polygon_str_pred}")
        s4_wv_pred, s4_R_pred, s4_T_pred = [], [], []
        if args.run_s4:
            results_csv_path_pred = run_s4_for_c(polygon_str_pred, args.c)
            if results_csv_path_pred:
                s4_wv_pred, s4_R_pred, s4_T_pred = read_results_csv(results_csv_path_pred)
                print(f"[DEBUG] UID {uid} - S4 wavelengths (predicted): {s4_wv_pred}")
                print(f"[DEBUG] UID {uid} - S4 reflectance (predicted): {s4_R_pred}")
            else:
                print(f"[WARN] S4 did not return valid results for UID {uid} (predicted).")
        # -- For ground-truth shape --
        gt_presence = (shape_gt[:,0] > 0.5).numpy()
        if not np.any(gt_presence):
            print(f"[WARN] No GT points for UID {uid}.")
            continue
        gt_q1 = shape_gt.numpy()[gt_presence, 1:3]
        full_gt_polygon = replicate_c4(gt_q1)
        sorted_gt_polygon = sort_points_by_angle(full_gt_polygon)
        vertex_strs_gt = [f"{v[0]:.6f},{v[1]:.6f}" for v in sorted_gt_polygon]
        polygon_str_gt = ";".join(vertex_strs_gt)
        print(f"[DEBUG] UID {uid} - S4 polygon string (GT): {polygon_str_gt}")
        s4_wv_gt, s4_R_gt, s4_T_gt = [], [], []
        if args.run_s4:
            results_csv_path_gt = run_s4_for_c(polygon_str_gt, args.c)
            if results_csv_path_gt:
                s4_wv_gt, s4_R_gt, s4_T_gt = read_results_csv(results_csv_path_gt)
                print(f"[DEBUG] UID {uid} - S4 wavelengths (GT): {s4_wv_gt}")
                print(f"[DEBUG] UID {uid} - S4 reflectance (GT): {s4_R_gt}")
            else:
                print(f"[WARN] S4 did not return valid results for UID {uid} (GT).")
        # Average spectra from dataset and network
        gt_spec_mean = spec_gt.numpy().mean(axis=0)
        pred_spec_mean = spec_pred_np.mean(axis=0)
        # Create figure with 2 rows x 2 columns
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        # Row 1: Ground Truth
        axes[0,0].scatter(sorted_gt_polygon[:,0], sorted_gt_polygon[:,1], color='blue', label='GT Shape', zorder=5)
        plot_polygon(axes[0,0], sorted_gt_polygon, color='blue', alpha=0.4, fill=False)
        axes[0,0].set_title(f"GT Shape for UID {uid}")
        axes[0,0].set_xlabel("X")
        axes[0,0].set_ylabel("Y")
        axes[0,0].legend()
        axes[0,1].plot(x_axis, gt_spec_mean, color='blue', linestyle='-', linewidth=2, label='Dataset GT Spectrum')
        if s4_wv_gt and s4_R_gt:
            # For uniform x-axis, we simply plot s4_R_gt using x_axis.
            axes[0,1].plot(x_axis, s4_R_gt, color='green', linestyle='-', linewidth=2, label='S4 (GT shape)')
        axes[0,1].set_title(f"GT Spectrum Comparison for UID {uid}")
        axes[0,1].set_xlabel("Uniform X-axis (1-100)")
        axes[0,1].set_ylabel("Reflectance")
        axes[0,1].legend()
        # Row 2: Predicted
        axes[1,0].scatter(sorted_pred_polygon[:,0], sorted_pred_polygon[:,1], color='red', label='Predicted Shape', zorder=5)
        plot_polygon(axes[1,0], sorted_pred_polygon, color='red', alpha=0.4, fill=False)
        axes[1,0].set_title(f"Predicted Shape for UID {uid}")
        axes[1,0].set_xlabel("X")
        axes[1,0].set_ylabel("Y")
        axes[1,0].legend()
        axes[1,1].plot(x_axis, pred_spec_mean, color='red', linestyle='--', linewidth=2, label='Network Spectrum')
        if s4_wv_pred and s4_R_pred:
            axes[1,1].plot(x_axis, s4_R_pred, color='green', linestyle='-', linewidth=2, label='S4 (Predicted shape)')
        axes[1,1].set_title(f"Spectrum Comparison (Predicted) for UID {uid}")
        axes[1,1].set_xlabel("Uniform X-axis (1-100)")
        axes[1,1].set_ylabel("Reflectance")
        axes[1,1].legend()
        plt.suptitle(f"UID {uid} Comparison", fontsize=16)
        outpng = os.path.join(out_dir, f"test_comparison_{uid}.png")
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(outpng, dpi=250)
        plt.close()
        print(f"[INFO] Saved 2x2 comparison figure for UID {uid} to {outpng}")
        # Additionally, plot separate S4 spectrum figures (for GT and Predicted) if available.
        if s4_wv_gt and s4_R_gt:
            fig_s4_gt, ax_s4_gt = plt.subplots(figsize=(8,6))
            ax_s4_gt.plot(x_axis, s4_R_gt, color='green', linestyle='-', linewidth=2, label='S4 Spectrum (GT)')
            ax_s4_gt.set_title(f"S4 Spectrum (GT) for UID {uid}")
            ax_s4_gt.set_xlabel("Uniform X-axis (1-100)")
            ax_s4_gt.set_ylabel("Reflectance")
            ax_s4_gt.legend()
            s4_gt_file = os.path.join(out_dir, f"s4_spectrum_GT_{uid}.png")
            plt.tight_layout()
            plt.savefig(s4_gt_file, dpi=250)
            plt.close(fig_s4_gt)
            print(f"[INFO] Saved separate S4 GT spectrum plot for UID {uid} to {s4_gt_file}")
        if s4_wv_pred and s4_R_pred:
            fig_s4_pred, ax_s4_pred = plt.subplots(figsize=(8,6))
            ax_s4_pred.plot(x_axis, s4_R_pred, color='green', linestyle='-', linewidth=2, label='S4 Spectrum (Predicted)')
            ax_s4_pred.set_title(f"S4 Spectrum (Predicted) for UID {uid}")
            ax_s4_pred.set_xlabel("Uniform X-axis (1-100)")
            ax_s4_pred.set_ylabel("Reflectance")
            ax_s4_pred.legend()
            s4_pred_file = os.path.join(out_dir, f"s4_spectrum_Pred_{uid}.png")
            plt.tight_layout()
            plt.savefig(s4_pred_file, dpi=250)
            plt.close(fig_s4_pred)
            print(f"[INFO] Saved separate S4 Predicted spectrum plot for UID {uid} to {s4_pred_file}")

def main():
    args = parse_args()
    if args.test:
        run_test_mode(args)
    else:
        print("No test mode specified. Use the --test argument with the root folder.")
        sys.exit(0)

if __name__ == "__main__":
    main()

