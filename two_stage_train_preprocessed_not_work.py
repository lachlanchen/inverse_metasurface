#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import random

###############################################################################
# PREPROCESSING FUNCTIONS
###############################################################################
def process_csv_file(csv_path, max_points=4):
    """
    Read one CSV file, group rows by unique shape (using 'shape_uid'), and
    for each group (which must have exactly 11 rows) process the reflectance
    spectrum (11×100) and parse the polygon (vertices_str). Only polygons whose
    shifted (minus 0.5) Q1-points count is between 1 and max_points are kept.
    Returns a list of dictionaries with keys: 'uid', 'spectra', and 'shape'.
    """
    df = pd.read_csv(csv_path)
    # Find all reflectance columns (they start with "R@")
    r_cols = [c for c in df.columns if c.startswith("R@")]
    if len(r_cols) == 0:
        raise ValueError("No reflectance columns found in " + csv_path)
    # Build a unique ID per shape
    df["shape_uid"] = (df["prefix"].astype(str) + "_" +
                       df["nQ"].astype(str) + "_" +
                       df["nS"].astype(str) + "_" +
                       df["shape_idx"].astype(str))
    records = []
    grouped = df.groupby("shape_uid", sort=False)
    for uid, grp in grouped:
        if len(grp) != 11:
            continue
        grp_sorted = grp.sort_values(by="c")
        spec_11x100 = grp_sorted[r_cols].values.astype(np.float32)
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
                if len(xy) == 2:
                    try:
                        x_val = float(xy[0])
                        y_val = float(xy[1])
                    except Exception:
                        continue
                    all_xy.append([x_val, y_val])
        all_xy = np.array(all_xy, dtype=np.float32)
        if len(all_xy) == 0:
            continue
        # SHIFT: subtract (0.5, 0.5)
        shifted = all_xy - 0.5
        q1 = []
        for (xx, yy) in shifted:
            if xx > 0 and yy > 0:
                q1.append([xx, yy])
        q1 = np.array(q1, dtype=np.float32)
        n_q1 = len(q1)
        if n_q1 < 1 or n_q1 > max_points:
            continue
        # Build a fixed-size (max_points×3) shape array: first column is presence,
        # then x and y coordinates.
        shape_4x3 = np.zeros((max_points, 3), dtype=np.float32)
        for i in range(n_q1):
            shape_4x3[i, 0] = 1.0
            shape_4x3[i, 1] = q1[i, 0]
            shape_4x3[i, 2] = q1[i, 1]
        records.append({"uid": uid, "spectra": spec_11x100, "shape": shape_4x3})
    return records

def preprocess_csv_folder(input_folder, output_npz, max_points=4):
    """
    Iterate over all CSV files (one by one) in the input_folder,
    process each file, and accumulate all valid records. Then save the final
    arrays (uids, spectra, shapes) into a compressed NPZ file.
    """
    all_records = []
    csv_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]
    csv_files.sort()
    for csv_file in csv_files:
        print("Processing file:", csv_file)
        records = process_csv_file(csv_file, max_points=max_points)
        print("Found", len(records), "records in", csv_file)
        all_records.extend(records)
    uids = [rec["uid"] for rec in all_records]
    spectra = np.array([rec["spectra"] for rec in all_records])  # shape: (N, 11, 100)
    shapes = np.array([rec["shape"] for rec in all_records])      # shape: (N, 4, 3)
    print("Total records processed:", len(uids))
    np.savez_compressed(output_npz, uids=uids, spectra=spectra, shapes=shapes)
    print("Preprocessed data saved to", output_npz)

###############################################################################
# DATASET FOR PREPROCESSED DATA
###############################################################################
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

###############################################################################
# UTILS: replicate_c4, sort_points_by_angle, plot_polygon
###############################################################################
# def replicate_c4(points):
#     c4 = []
#     for (x, y) in points:
#         c4.append([ x,  y])
#         c4.append([-y,  x])
#         c4.append([-x, -y])
#         c4.append([ y, -x])
#     return np.array(c4, dtype=np.float32)

def replicate_c4(points):
    # If points is a torch tensor, convert it to a NumPy array.
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    c4 = []
    for (x, y) in points:
        c4.append([ x,  y])
        c4.append([-y,  x])
        c4.append([-x, -y])
        c4.append([ y, -x])
    return np.array(c4, dtype=np.float32)


def sort_points_by_angle(points):
    if len(points) < 3:
        return points
    cx, cy = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    idx = np.argsort(angles)
    return points[idx]

def plot_polygon(ax, points, color='green', alpha=0.4, fill=True):
    import matplotlib.patches as patches
    from matplotlib.path import Path
    if len(points) < 3:
        ax.scatter(points[:, 0], points[:, 1], c=color)
        return
    closed = np.concatenate([points, points[0:1]], axis=0)
    codes = [Path.MOVETO] + [Path.LINETO]*(len(points)-1) + [Path.CLOSEPOLY]
    path = Path(closed, codes)
    patch = patches.PathPatch(path, facecolor=color if fill else 'none',
                              alpha=alpha, edgecolor=color)
    ax.add_patch(patch)
    ax.autoscale_view()

###############################################################################
# MODEL DEFINITION: Stage A – shape->spec
###############################################################################
class ShapeToSpectraModel(nn.Module):
    def __init__(self, d_in=3, d_model=256, nhead=4, num_layers=4):
        super().__init__()
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

###############################################################################
# DATASET CLASS (for original CSV file)
###############################################################################
class Q1ShiftedSpectraDataset(Dataset):
    def __init__(self, csv_file, max_points=4):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.r_cols = [c for c in self.df.columns if c.startswith("R@")]
        if len(self.r_cols)==0:
            raise ValueError("No reflectance columns found (R@...).")
        self.df["shape_uid"] = (self.df["prefix"].astype(str) + "_" +
                                self.df["nQ"].astype(str) + "_" +
                                self.df["nS"].astype(str) + "_" +
                                self.df["shape_idx"].astype(str))
        self.data_list = []
        grouped = self.df.groupby("shape_uid", sort=False)
        for uid, grp in grouped:
            if len(grp)!=11:
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
                        x_val, y_val = float(xy[0]), float(xy[1])
                        all_xy.append([x_val, y_val])
            all_xy = np.array(all_xy, dtype=np.float32)
            if len(all_xy)==0:
                continue
            # SHIFT => minus(0.5,0.5)
            shifted = all_xy - 0.5
            q1 = []
            for (xx, yy) in shifted:
                if xx>0 and yy>0:
                    q1.append([xx, yy])
            q1 = np.array(q1, dtype=np.float32)
            n_q1 = len(q1)
            if n_q1 < 1 or n_q1 > max_points:
                continue
            shape_4x3 = np.zeros((max_points, 3), dtype=np.float32)
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
            raise ValueError("No valid shapes => SHIFT->Q1->UpTo4")
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

###############################################################################
# MODEL DEFINITION: Stage B – spec->shape
###############################################################################
class Spectra2ShapeVarLen(nn.Module):
    def __init__(self, d_in=100, d_model=256, nhead=4, num_layers=4):
        super().__init__()
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
                prob_i = torch.sigmoid(presence_logits[:, i]).clamp(1e-6, 1-1e-6)
                prob_chain = prob_i * presence_list[i - 1]
                ste_i = (prob_chain > 0.5).float() + prob_chain - prob_chain.detach()
                presence_list.append(ste_i)
        presence_stack = torch.stack(presence_list, dim=1)
        xy_bounded = torch.sigmoid(xy_raw)
        xy_final = xy_bounded * presence_stack.unsqueeze(-1)
        final_shape = torch.cat([presence_stack.unsqueeze(-1), xy_final], dim=-1)
        return final_shape

class Spec2ShapeFrozen(nn.Module):
    def __init__(self, spec2shape_net, shape2spec_frozen):
        super().__init__()
        self.spec2shape = spec2shape_net
        self.shape2spec_frozen = shape2spec_frozen
        for p in self.shape2spec_frozen.parameters():
            p.requires_grad = False
    def forward(self, spec_input):
        shape_pred = self.spec2shape(spec_input)
        with torch.no_grad():
            spec_chain = self.shape2spec_frozen(shape_pred)
        return shape_pred, spec_chain

###############################################################################
# VISUALIZATION FUNCTIONS FOR TRAINING
###############################################################################
def visualize_stageA_samples(model, ds_val, device, out_dir, sample_count=4, seed=123):
    random.seed(seed)
    if len(ds_val)==0:
        print("[Stage A] Val set empty => skip visualize.")
        return
    idx_samples = random.sample(range(len(ds_val)), min(sample_count, len(ds_val)))
    fig, axes = plt.subplots(len(idx_samples), 2, figsize=(8, 3*len(idx_samples)))
    if len(idx_samples)==1:
        axes = [axes]
    model.eval()
    with torch.no_grad():
        for i, idx_ in enumerate(idx_samples):
            spec_gt, shape_gt, uid_ = ds_val[idx_]
            axL = axes[i][0]
            pres = (shape_gt[:,0] > 0.5)
            q1 = shape_gt[pres,1:3]
            if len(q1) > 0:
                c4 = replicate_c4(q1)
                c4 = sort_points_by_angle(c4)
                plot_polygon(axL, c4, color='green', alpha=0.4, fill=True)
            axL.set_aspect("equal", "box")
            axL.set_xlim([-0.5,0.5])
            axL.set_ylim([-0.5,0.5])
            axL.grid(True)
            axL.set_title(f"UID={uid_}\n(GT shape polygon)")
            axR = axes[i][1]
            shape_t = torch.tensor(shape_gt, dtype=torch.float32, device=device).unsqueeze(0)
            spec_pd = model(shape_t).cpu().numpy()[0]
            for row_ in spec_gt:
                axR.plot(row_, color='blue', alpha=0.5)
            for row_ in spec_pd:
                axR.plot(row_, color='red', alpha=0.5, linestyle='--')
            axR.set_title("GT spec (blue) vs Pred spec (red)")
            axR.set_xlabel("Wavelength index")
            axR.set_ylabel("Reflectance")
    plt.tight_layout()
    out_fig = os.path.join(out_dir, "samples_2col_stageA.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"[Stage A] sample visualization saved to {out_fig}")

def visualize_stageB_samples(pipeline, shape2spec_frozen, ds_val, device, out_dir, sample_count=4, seed=123):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    if len(ds_val)==0:
        print("[Stage B] Val set empty => skip visualize.")
        return
    idx_samples = random.sample(range(len(ds_val)), min(sample_count, len(ds_val)))
    fig, axes = plt.subplots(len(idx_samples), 3, figsize=(12, 3*len(idx_samples)))
    if len(idx_samples)==1:
        axes = [axes]
    pipeline.eval()
    shape2spec_frozen.eval()
    with torch.no_grad():
        for i, idx_ in enumerate(idx_samples):
            spec_gt, shape_gt, uid_ = ds_val[idx_]
            # Left => GT spectrum
            axL = axes[i][0]
            for row_ in spec_gt:
                axL.plot(row_, color='blue', alpha=0.5)
            axL.set_title(f"UID={uid_}\n(GT spectrum)")
            # Middle => GT shape (green) vs Pred shape (red)
            axM = axes[i][1]
            pres_g = (shape_gt[:,0] > 0.5)
            q1_g = shape_gt[pres_g,1:3]
            if len(q1_g) > 0:
                c4g = replicate_c4(q1_g)
                c4g = sort_points_by_angle(c4g)
                plot_polygon(axM, c4g, color='green', alpha=0.4, fill=True)
            spec_t = torch.tensor(spec_gt, dtype=torch.float32, device=device).unsqueeze(0)
            shape_pd, spec_pd = pipeline(spec_t)
            shape_pd = shape_pd.cpu().numpy()[0]
            pres_p = (shape_pd[:,0] > 0.5)
            q1_p = shape_pd[pres_p,1:3]
            if len(q1_p) > 0:
                c4p = replicate_c4(q1_p)
                c4p = sort_points_by_angle(c4p)
                plot_polygon(axM, c4p, color='red', alpha=0.3, fill=False)
            axM.set_title("GT shape (green) vs Pred shape (red)")
            axM.set_xlim([-0.5,0.5])
            axM.set_ylim([-0.5,0.5])
            axM.set_aspect("equal", "box")
            axM.grid(True)
            # Right => Original spec (blue), GT->spec (green dashed), Pred->spec (red dashed)
            axR = axes[i][2]
            for row_ in spec_gt:
                axR.plot(row_, color='blue', alpha=0.5)
            shape_gt_t = torch.tensor(shape_gt, dtype=torch.float32, device=device).unsqueeze(0)
            spec_gtshape = shape2spec_frozen(shape_gt_t).cpu().numpy()[0]
            for row_ in spec_gtshape:
                axR.plot(row_, color='green', alpha=0.5, linestyle='--')
            for row_ in spec_pd:
                axR.plot(row_, color='red', alpha=0.5, linestyle='--')
            axR.set_title("Original spec (blue), GT->spec (green dashed), Pred->spec (red dashed)")
            axR.set_xlabel("Wavelength index")
            axR.set_ylabel("Reflectance")
    plt.tight_layout()
    out_fig = os.path.join(out_dir, "samples_3col_stageB.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"[Stage B] sample visualization saved to {out_fig}")

###############################################################################
# VISUALIZATION FUNCTIONS FOR TEST MODE
###############################################################################
def visualize_stageA_on_test(model, ds_test, device, out_fig, sample_count=4, seed=123):
    random.seed(seed)
    if len(ds_test)==0:
        print("[Test Stage A] Empty dataset => skip visualization.")
        return
    idx_samples = random.sample(range(len(ds_test)), min(sample_count, len(ds_test)))
    fig, axes = plt.subplots(len(idx_samples), 2, figsize=(8, 3*len(idx_samples)))
    if len(idx_samples)==1:
        axes = [axes]
    model.eval()
    with torch.no_grad():
        for i, idx_ in enumerate(idx_samples):
            spec_gt, shape_gt, uid_ = ds_test[idx_]
            axL = axes[i][0]
            pres = (shape_gt[:,0] > 0.5)
            q1 = shape_gt[pres,1:3]
            if len(q1)>0:
                c4 = replicate_c4(q1)
                c4 = sort_points_by_angle(c4)
                plot_polygon(axL, c4, color='green', alpha=0.4, fill=True)
            axL.set_aspect("equal", "box")
            axL.set_xlim([-0.5,0.5])
            axL.set_ylim([-0.5,0.5])
            axL.grid(True)
            axL.set_title(f"UID={uid_}\nGT shape polygon")
            axR = axes[i][1]
            shape_t = torch.tensor(shape_gt, dtype=torch.float32, device=device).unsqueeze(0)
            spec_pd = model(shape_t).cpu().numpy()[0]
            for row_ in spec_gt:
                axR.plot(row_, color='blue', alpha=0.5)
            for row_ in spec_pd:
                axR.plot(row_, color='red', alpha=0.5, linestyle='--')
            axR.set_title("GT spec (blue) vs Pred spec (red)")
            axR.set_xlabel("Wavelength index")
            axR.set_ylabel("Reflectance")
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()
    print(f"[Test Stage A] Visualization saved to {out_fig}")

def visualize_stageB_on_test(pipeline, shape2spec_frozen, ds_test, device, out_fig, sample_count=4, seed=123):
    random.seed(seed)
    if len(ds_test)==0:
        print("[Test Stage B] Empty dataset => skip visualization.")
        return
    idx_samples = random.sample(range(len(ds_test)), min(sample_count, len(ds_test)))
    fig, axes = plt.subplots(len(idx_samples), 3, figsize=(12, 3*len(idx_samples)))
    if len(idx_samples)==1:
        axes = [axes]
    pipeline.eval()
    shape2spec_frozen.eval()
    with torch.no_grad():
        for i, idx_ in enumerate(idx_samples):
            spec_gt, shape_gt, uid_ = ds_test[idx_]
            axL = axes[i][0]
            for row_ in spec_gt:
                axL.plot(row_, color='blue', alpha=0.5)
            axL.set_title(f"UID={uid_}\n(GT spec)")
            axM = axes[i][1]
            pres_g = (shape_gt[:,0] > 0.5)
            q1_g = shape_gt[pres_g,1:3]
            if len(q1_g)>0:
                c4g = replicate_c4(q1_g)
                c4g = sort_points_by_angle(c4g)
                plot_polygon(axM, c4g, color='green', alpha=0.4, fill=True)
            spec_t = torch.tensor(spec_gt, dtype=torch.float32, device=device).unsqueeze(0)
            shape_pd, spec_pd = pipeline(spec_t)
            shape_pd = shape_pd.cpu().numpy()[0]
            pres_p = (shape_pd[:,0] > 0.5)
            q1_p = shape_pd[pres_p,1:3]
            if len(q1_p)>0:
                c4p = replicate_c4(q1_p)
                c4p = sort_points_by_angle(c4p)
                plot_polygon(axM, c4p, color='red', alpha=0.3, fill=False)
            axM.set_title("GT shape (green) vs Pred shape (red)")
            axM.set_xlim([-0.5,0.5])
            axM.set_ylim([-0.5,0.5])
            axM.set_aspect("equal", "box")
            axM.grid(True)
            axR = axes[i][2]
            for row_ in spec_gt:
                axR.plot(row_, color='blue', alpha=0.5)
            shape_gt_t = torch.tensor(shape_gt, dtype=torch.float32, device=device).unsqueeze(0)
            spec_gtshape = shape2spec_frozen(shape_gt_t).cpu().numpy()[0]
            for row_ in spec_gtshape:
                axR.plot(row_, color='green', alpha=0.5, linestyle='--')
            for row_ in spec_pd.cpu().numpy()[0]:
                axR.plot(row_, color='red', alpha=0.5, linestyle='--')
            axR.set_title("Original spec (blue), GT->spec (green dashed), Pred->spec (red dashed)")
            axR.set_xlabel("Wavelength index")
            axR.set_ylabel("Reflectance")
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()
    print(f"[Test Stage B] Visualization saved to {out_fig}")

###############################################################################
# ARGUMENT PARSING AND MAIN FUNCTION
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage pipeline with preprocessed dataset support and test mode.")
    parser.add_argument("--preprocess", action="store_true",
                        help="Run preprocessing to generate preprocessed dataset from CSVs.")
    parser.add_argument("--input_folder", type=str, default="",
                        help="Folder containing CSV files for preprocessing.")
    parser.add_argument("--output_npz", type=str, default="preprocessed_data.npz",
                        help="Output NPZ file for preprocessed data.")
    parser.add_argument("--data_npz", type=str, default="",
                        help="Preprocessed dataset file to use for training/testing.")
    parser.add_argument("--csv_file", type=str, default="",
                        help="CSV file to use if not using preprocessed dataset.")
    parser.add_argument("--test", action="store_true", help="Run in test mode.")
    parser.add_argument("--model", type=str, default="",
                        help="Model directory (containing stageA/ and stageB/) for test mode.")
    parser.add_argument("--npz_file", type=str, default="",
                        help="(Alternative) Preprocessed NPZ file for test mode (if --data_npz not provided).")
    return parser.parse_args()

def main():
    args = parse_args()
    # Preprocessing mode
    if args.preprocess:
        if not args.input_folder:
            print("Error: --input_folder must be specified for preprocessing.")
            return
        preprocess_csv_folder(args.input_folder, args.output_npz)
        return

    # Decide data source for training/testing
    use_preprocessed = False
    data_source = None
    if args.data_npz:
        if not os.path.isfile(args.data_npz):
            print("Error: Preprocessed dataset file not found:", args.data_npz)
            return
        use_preprocessed = True
        data_source = args.data_npz
    elif args.csv_file:
        data_source = args.csv_file
    else:
        print("Error: Must specify either --data_npz or --csv_file for training/testing.")
        return

    # Test mode
    if args.test:
        if not args.model:
            print("[Error] Must specify --model in test mode.")
            return
        if not os.path.isdir(args.model):
            print("[Error] Model directory does not exist:", args.model)
            return
        test_folder = os.path.join(args.model, "test")
        os.makedirs(test_folder, exist_ok=True)
        shape2spec_path = os.path.join(args.model, "stageA", "shape2spec_stageA.pt")
        spec2shape_path = os.path.join(args.model, "stageB", "spec2shape_stageB.pt")
        if not (os.path.isfile(shape2spec_path) and os.path.isfile(spec2shape_path)):
            print("[Error] Checkpoint files not found in", args.model)
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load Stage A model
        shape2spec_model = ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4)
        shape2spec_model.load_state_dict(torch.load(shape2spec_path))
        shape2spec_model.to(device)
        shape2spec_model.eval()
        # Load Stage B components
        shape2spec_frozen = ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4)
        shape2spec_frozen.load_state_dict(torch.load(shape2spec_path))
        shape2spec_frozen.to(device)
        for p in shape2spec_frozen.parameters():
            p.requires_grad = False
        spec2shape_net = Spectra2ShapeVarLen(d_in=100, d_model=256, nhead=4, num_layers=4)
        spec2shape_net.load_state_dict(torch.load(spec2shape_path))
        spec2shape_net.to(device)
        spec2shape_net.eval()
        final_pipeline = Spec2ShapeFrozen(spec2shape_net, shape2spec_frozen).to(device)
        final_pipeline.eval()
        # Choose test dataset source: priority: data_npz > npz_file > csv_file
        if args.data_npz:
            ds_test = PreprocessedSpectraDataset(args.data_npz)
        elif args.npz_file:
            ds_test = PreprocessedSpectraDataset(args.npz_file)
        elif args.csv_file:
            ds_test = Q1ShiftedSpectraDataset(args.csv_file)
        else:
            print("Error: For test mode, specify one of --data_npz, --npz_file, or --csv_file.")
            return
        print(f"[TEST] Dataset size = {len(ds_test)}")
        # Save test predictions CSV
        out_csv = os.path.join(test_folder, "test_predictions.csv")
        cols = ["uid", "presence_gt", "xy_gt", "presence_pred", "xy_pred",
                "specMSE_shape2spec", "specMSE_finalPipeline"]
        rows = []
        crit_mse = nn.MSELoss(reduction='none')
        loader_test = DataLoader(ds_test, batch_size=1024, shuffle=False)
        with torch.no_grad():
            for (spec_np, shape_np, uid_list) in loader_test:
                bsz = shape_np.size(0)
                spec_t = spec_np.to(device)
                shape_t = shape_np.to(device)
                spec_predA = shape2spec_model(shape_t)
                shape_predB, spec_predB = final_pipeline(spec_t)
                msesA = torch.mean(crit_mse(spec_predA, spec_t).view(bsz, -1), dim=1)
                msesB = torch.mean(crit_mse(spec_predB, spec_t).view(bsz, -1), dim=1)
                shape_np_ = shape_np.cpu().numpy()
                shape_predB_ = shape_predB.cpu().numpy()
                msesA_ = msesA.cpu().numpy()
                msesB_ = msesB.cpu().numpy()
                for i in range(bsz):
                    rowd = {
                        "uid": uid_list[i],
                        "presence_gt": shape_np_[i,:,0].tolist(),
                        "xy_gt": shape_np_[i,:,1:].tolist(),
                        "presence_pred": shape_predB_[i,:,0].tolist(),
                        "xy_pred": shape_predB_[i,:,1:].tolist(),
                        "specMSE_shape2spec": float(msesA_[i]),
                        "specMSE_finalPipeline": float(msesB_[i])
                    }
                    rows.append(rowd)
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(out_csv, index=False)
        print(f"[TEST] Predictions saved to {out_csv}")
        # Save test visualizations
        stageA_fig = os.path.join(test_folder, "samples_2col_test_stageA.png")
        visualize_stageA_on_test(shape2spec_model, ds_test, device, stageA_fig)
        stageB_fig = os.path.join(test_folder, "samples_3col_test_stageB.png")
        visualize_stageB_on_test(final_pipeline, shape2spec_frozen, ds_test, device, stageB_fig)
        print("[TEST] Done.")
    else:
        # Training mode
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_out = "outputs_two_stage_" + timestamp
        os.makedirs(base_out, exist_ok=True)
        # Stage A training
        stageA_dir = os.path.join(base_out, "stageA")
        shape2spec_ckpt, ds_valA, modelA = train_stageA_shape2spec(
            data_source=data_source,
            out_dir=stageA_dir,
            num_epochs=500,
            batch_size=1024,
            lr=1e-4,
            weight_decay=1e-5,
            split_ratio=0.8,
            grad_clip=1.0,
            use_preprocessed=use_preprocessed
        )
        visualize_stageA_samples(modelA, ds_valA, next(modelA.parameters()).device, stageA_dir, sample_count=4)
        # Stage B training
        stageB_dir = os.path.join(base_out, "stageB")
        spec2shape_ckpt, ds_valB, shape2spec_froz, spec2shape_net = train_stageB_spec2shape_frozen(
            data_source=data_source,
            out_dir=stageB_dir,
            shape2spec_ckpt=shape2spec_ckpt,
            num_epochs=500,
            batch_size=1024,
            lr=1e-4,
            weight_decay=1e-5,
            split_ratio=0.8,
            grad_clip=1.0,
            use_preprocessed=use_preprocessed
        )
        visualize_stageB_samples(Spec2ShapeFrozen(spec2shape_net, shape2spec_froz),
                                 shape2spec_froz, ds_valB, next(spec2shape_net.parameters()).device, stageB_dir, sample_count=4)

if __name__ == "__main__":
    main()

