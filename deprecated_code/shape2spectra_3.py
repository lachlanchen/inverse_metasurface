#!/usr/bin/env python3

"""
shape2spectra.py

Modular code for reading SHIFT->Q1 shapes (up to 4 points) and predicting
(11x100) spectra with a small Transformer aggregator.

Added:
  - Middle plot axis range => [-0.5, 0.5] in both x & y.
  - Output results to a folder with a datetime suffix.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from datetime import datetime

###############################################################################
# 1) Dataset Class
###############################################################################
class Q1ShiftedShapeDataset(Dataset):
    """
    Reads each shape from a CSV with 11 lines (c=0..1) per shape.
    Steps:
      1) Group lines by a unique ID => (11,100) reflectances.
      2) From the first row, parse `vertices_str`.
      3) SHIFT => (x-0.5,y-0.5).
      4) Keep points in quadrant 1 => x>0,y>0.
      5) If #Q1 points=1..4, build (4,3) => [presence,x,y]. Otherwise skip.
      6) Return (shape_arr, reflectances, uid).
    """
    def __init__(self, csv_file, max_points=4):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.r_cols = [c for c in self.df.columns if c.startswith("R@")]
        if len(self.r_cols) == 0:
            raise ValueError("No reflectance columns found. Expected 'R@...' columns.")

        # create shape-uid
        self.df["shape_uid"] = (
            self.df["prefix"].astype(str)
            + "_" + self.df["nQ"].astype(str)
            + "_" + self.df["nS"].astype(str)
            + "_" + self.df["shape_idx"].astype(str)
        )

        self.data_list = []
        grouped = self.df.groupby("shape_uid", sort=False)
        for uid, grp in grouped:
            # We want exactly 11 lines => c=0..1
            if len(grp) != 11:
                continue
            grp_sorted = grp.sort_values(by="c")  # ensure c=0..1 ascending

            # reflectances => shape(11,100)
            R_data = grp_sorted[self.r_cols].values.astype(np.float32)

            # parse vertices from first row
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
                        x_val, y_val = float(xy[0]), float(xy[1])
                        all_xy.append([x_val, y_val])
            all_xy = np.array(all_xy, dtype=np.float32)
            if len(all_xy) == 0:
                continue

            # SHIFT => minus (0.5,0.5)
            shifted = all_xy - 0.5

            # keep Q1 => x>0,y>0
            q1_points = []
            for (xx, yy) in shifted:
                if xx>0 and yy>0:
                    q1_points.append([xx, yy])
            q1_points = np.array(q1_points, dtype=np.float32)
            nq1 = len(q1_points)
            if nq1<1 or nq1>max_points:
                continue

            # build (4,3) => presence + x + y
            shape_arr = np.zeros((max_points, 3), dtype=np.float32)
            for i in range(nq1):
                shape_arr[i,0] = 1.0
                shape_arr[i,1] = q1_points[i,0]
                shape_arr[i,2] = q1_points[i,1]

            self.data_list.append({
                "uid"     : uid,
                "shape"   : shape_arr,
                "spectra" : R_data
            })

        self.data_len = len(self.data_list)
        if self.data_len == 0:
            raise ValueError("After SHIFT->Q1->UpTo4 filtering, no valid shapes found.")

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        item = self.data_list[idx]
        return (item["shape"], item["spectra"], item["uid"])


###############################################################################
# 2) The Model: shape -> (11x100) spectra
###############################################################################
class ShapeEncoder(nn.Module):
    """
    A mini Transformer that handles up to 4 tokens (presence,x,y).
    presence=0 => mask out
    """
    def __init__(self, d_in=3, d_model=128, nhead=4, num_layers=2):
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

    def forward(self, shape_tokens):
        """
        shape_tokens => (batch,4,3), shape_tokens[:,:,0] = presence
        """
        bsz, N, _ = shape_tokens.shape
        presence = shape_tokens[:,:,0]  # (bsz,4)
        # True => mask out
        key_padding_mask = (presence < 0.5)

        x_emb = self.input_proj(shape_tokens)  # (bsz,4,d_model)
        x_enc = self.encoder(x_emb, src_key_padding_mask=key_padding_mask)

        # Weighted sum by presence
        presence_sum = presence.sum(dim=1, keepdim=True) + 1e-8
        x_enc_weighted = x_enc * presence.unsqueeze(-1)
        shape_emb = x_enc_weighted.sum(dim=1) / presence_sum
        return shape_emb

class ShapeToSpectraModel(nn.Module):
    """
    shape(4x3) => (11x100) reflectances
    1) Transformer aggregator => shape embedding
    2) MLP => flatten => reshape => (11,100)
    """
    def __init__(self, d_in=3, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.encoder = ShapeEncoder(d_in, d_model, nhead, num_layers)
        self.out_dim = 11 * 100

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, self.out_dim)
        )

    def forward(self, shape_arr):
        bsz = shape_arr.size(0)
        shape_emb = self.encoder(shape_arr)
        out_flat  = self.mlp(shape_emb)           # (bsz,1100)
        out_spec  = out_flat.view(bsz,11,100)     # (bsz,11,100)
        return out_spec


###############################################################################
# 3) Visualization
###############################################################################
def replicate_c4(points):
    """
    Given Q1 points (x>0,y>0),
    replicate them to get symmetrical shape in all 4 quadrants.
    (x,y)->(±x,±y)
    Returns array with 4*N points.
    """
    mirrored = []
    for (xx,yy) in points:
        mirrored.append([ xx,  yy])  # Q1
        mirrored.append([-xx,  yy])  # Q2
        mirrored.append([-xx, -yy])  # Q3
        mirrored.append([ xx, -yy])  # Q4
    return np.array(mirrored, dtype=np.float32)

def sort_points_by_angle(points):
    """
    Sort 2D points by polar angle to create a single closed polygon for fill.
    """
    if len(points)<3:
        return points
    cx, cy = points.mean(axis=0)
    angles = np.arctan2(points[:,1]-cy, points[:,0]-cx)
    idx = np.argsort(angles)
    return points[idx]

def plot_polygon(ax, points, color='green', alpha=0.4, fill=True):
    """
    Draw a closed polygon from points (N,2).
    """
    import matplotlib.patches as patches
    from matplotlib.path import Path

    if len(points) < 3:
        # If fewer than 3, just scatter them
        ax.scatter(points[:,0], points[:,1], c=color)
        return

    # close polygon
    verts = np.concatenate([points, points[0:1]], axis=0)
    codes = [Path.MOVETO] + [Path.LINETO]*(len(points)-1) + [Path.CLOSEPOLY]
    path = Path(verts, codes)
    patch = patches.PathPatch(
        path,
        facecolor=color if fill else 'none',
        alpha=alpha,
        edgecolor=color
    )
    ax.add_patch(patch)
    ax.autoscale_view()

def plot_spectra(ax, spectra_gt, spectra_pred=None, color_gt='blue', color_pred='red'):
    """
    Plot ground-truth spectra, shape=(11,100).
    Optionally overlay predicted in dashed lines.
    """
    n_c = spectra_gt.shape[0]  # typically 11
    for i in range(n_c):
        ax.plot(spectra_gt[i], color=color_gt, alpha=0.5)
    if spectra_pred is not None:
        for i in range(n_c):
            ax.plot(spectra_pred[i], color=color_pred, alpha=0.5, linestyle='--')
    ax.set_xlabel("Wavelength index")
    ax.set_ylabel("Reflectance")

def visualize_4x3_samples(model, ds_val, device, out_dir=".", seed=888888):
    """
    Creates a 4-row, 3-col figure:
      row => shape with 1..4 Q1 points
      col1 => GT spectra
      col2 => shape (c4 filled) with GT in green, placeholder in red
      col3 => GT vs predicted spectra

    Middle plot: fix the axis range to [-0.5,0.5].
    """
    import random
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # find shapes with 1..4 Q1 points
    wanted_counts = [1,2,3,4]
    found_idx = {}
    for idx in range(len(ds_val)):
        shape_np, spec_np, uid_ = ds_val[idx]
        presence = (shape_np[:,0]>0.5)
        n_q1 = int(presence.sum())
        if n_q1 in wanted_counts and n_q1 not in found_idx:
            found_idx[n_q1] = idx
        if len(found_idx)==len(wanted_counts):
            break

    found_counts_sorted = sorted(found_idx.keys())
    if len(found_counts_sorted)==0:
        print("[Warning] No shapes with 1..4 Q1 points found in val set.")
        return

    n_rows = len(found_counts_sorted)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12,3*n_rows))
    if n_rows==1:
        axes = [axes]

    model.eval()
    with torch.no_grad():
        for r_i, n_q1 in enumerate(found_counts_sorted):
            idx_ = found_idx[n_q1]
            shape_np, spec_gt, uid_ = ds_val[idx_]
            # Left col => GT spectra
            axL = axes[r_i][0]
            plot_spectra(axL, spec_gt, spectra_pred=None, color_gt='blue')
            axL.set_title(f"GT Spectrum\nuid={uid_} (#Q1={n_q1})")

            # Middle => shape c4 fill
            axM = axes[r_i][1]
            presence = (shape_np[:,0]>0.5)
            q1_pts = shape_np[presence,1:3]
            if len(q1_pts)>0:
                c4_pts = replicate_c4(q1_pts)
                c4_pts_sorted = sort_points_by_angle(c4_pts)
                plot_polygon(axM, c4_pts_sorted, color='green', alpha=0.4, fill=True)
                # "pred shape" in red = placeholder
                plot_polygon(axM, c4_pts_sorted, color='red', alpha=0.2, fill=False)
            axM.set_xlim([-0.5, 0.5])
            axM.set_ylim([-0.5, 0.5])
            axM.set_aspect("equal","box")
            axM.grid(True)
            axM.set_title("Shape c4 fill (GT green, Pred red)")

            # Right => GT vs predicted spectra
            shape_t = torch.tensor(shape_np, dtype=torch.float32, device=device).unsqueeze(0)
            spec_pred = model(shape_t).cpu().numpy()[0]
            axR = axes[r_i][2]
            plot_spectra(axR, spec_gt, spec_pred, color_gt='blue', color_pred='red')
            axR.set_title("GT vs Pred Spectra")

    plt.tight_layout()
    out_fig = os.path.join(out_dir, "4x3_visualization.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"[Visualization] saved => {out_fig}")


###############################################################################
# 4) Main Training
###############################################################################
def train_shape2spectrum(
    csv_file,
    out_dir="outputs_shape2spectrum",
    num_epochs=100,
    # As requested, always 4096
    batch_size=4096,
    lr=1e-4,
    weight_decay=1e-5,
    split_ratio=0.8,
    d_model=128,
    nhead=4,
    num_layers=2
):
    """
    Train shape->spectrum model on the data. 
    The 'out_dir' will have a datetime suffix appended.
    """
    # Append datetime suffix to out_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"{out_dir}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Dataset
    ds_full = Q1ShiftedShapeDataset(csv_file, max_points=4)
    ds_len = len(ds_full)
    train_len = int(ds_len*split_ratio)
    val_len   = ds_len - train_len
    ds_train, ds_val = random_split(ds_full, [train_len, val_len])
    print(f"[DATA] total={ds_len}, train={train_len}, val={val_len}")

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2) Model
    model = ShapeToSpectraModel(d_in=3, d_model=d_model, nhead=nhead, num_layers=num_layers)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    criterion = nn.MSELoss()

    # 3) Train loop
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        run_loss=0.0
        for shape_np, spec_np, uid_list in train_loader:
            shape_t = shape_np.to(device)
            spec_t  = spec_np.to(device)
            pred = model(shape_t)
            loss = criterion(pred, spec_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run_loss += loss.item()

        avg_train = run_loss/len(train_loader)
        train_losses.append(avg_train)

        # Validation
        model.eval()
        val_sum, val_count=0.0,0
        with torch.no_grad():
            for shape_np, spec_np, uid_list in val_loader:
                bsz_ = shape_np.size(0)
                shape_t = shape_np.to(device)
                spec_t  = spec_np.to(device)
                pred = model(shape_t)
                vloss = criterion(pred, spec_t)*bsz_
                val_sum += vloss.item()
                val_count+=bsz_
        avg_val = val_sum/val_count
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        # print progress
        if (epoch+1) % 20 == 0 or epoch == 0 or epoch == (num_epochs-1):
            print(f"Epoch[{epoch+1}/{num_epochs}] => trainMSE={avg_train:.4f}, valMSE={avg_val:.4f}")

    # 4) Plot training curve
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses,   label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Shape->Spectrum Training")
    plt.legend()
    plt.savefig(os.path.join(out_dir,"training_curve.png"))
    plt.close()

    # 5) Final val MSE
    model.eval()
    val_sum, val_count=0.0, 0
    with torch.no_grad():
        for shape_np, spec_np, uid_list in val_loader:
            bsz_ = shape_np.size(0)
            shape_t = shape_np.to(device)
            spec_t  = spec_np.to(device)
            pred = model(shape_t)
            loss_ = criterion(pred, spec_t)*bsz_
            val_sum += loss_.item()
            val_count+=bsz_
    final_mse = val_sum/val_count
    print(f"[Final Val MSE] => {final_mse:.6f}")

    # 6) Save model
    model_path = os.path.join(out_dir,"shape2spectrum_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[Model saved] => {model_path}")

    # 7) 4x3 Visualization
    if val_len>0:
        visualize_4x3_samples(model, ds_val, device, out_dir=out_dir)


###############################################################################
# if-name-main usage
###############################################################################
def main():
    CSV_FILE = "merged_s4_shapes_iccpOv10kG40_seed88888.csv"
    train_shape2spectrum(
        csv_file=CSV_FILE,
        out_dir="outputs_shape2spectrum",
        num_epochs=100,
        batch_size=4096,  # always 4096
        lr=1e-4,
        weight_decay=1e-5,
        split_ratio=0.8,
        d_model=128,
        nhead=4,
        num_layers=2
    )

if __name__=="__main__":
    main()

