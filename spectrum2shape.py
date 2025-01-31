#!/usr/bin/env python3

"""
spectrum2shape.py

Maps 11x100 spectra -> (4x3) shape with presence bits (0 or 1).
We use a straight-through estimator (STE) for presence bits.

Dataset:
  - Q1ShiftedSpectraDataset: 
    - input_spectra => shape(11,100) float
    - target_shape  => shape(4,3) => [presence,x,y], with x,y in Q1 (shifted by -0.5).
      If #Q1 points not in [1..4], skip.

Model:
  - Transformer aggregator for 11 tokens (each token=100-dim?), ignoring order of c-values if desired 
    or a simpler aggregator (you can adapt). Then MLP -> presence_logit + x,y. 
  - presence -> sigmoid -> STE -> presence_hard. 
  - final shape = presence_hard * [x,y].

Visualization (3 columns):
  1) Left: original GT spectrum (blue).
  2) Middle: GT shape (green) vs predicted shape (red), both replicated via C4 rotation, 
     sorted by angle, then filled. 
  3) Right: same GT spectrum again (blue), to fulfill the “both left & right are the same GT plot.”

Author: ChatGPT
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
class Q1ShiftedSpectraDataset(Dataset):
    """
    For each shape in the CSV, gather:
      - input: spectra(11,100) float
      - target: shape(4,3) => [presence, x, y], after SHIFT->Q1.

    Procedure:
      1) group by shape_uid => get 11 rows for c=0..1 => (11,100) spectra
      2) parse the vertices_str from the first row => SHIFT by -0.5 => keep Q1 => up to 4 points
      3) build (4,3) => [presence,x,y]
      4) if #Q1 not in [1..4], skip
    """
    def __init__(self, csv_file, max_points=4):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.r_cols = [c for c in self.df.columns if c.startswith("R@")]
        if len(self.r_cols) == 0:
            raise ValueError("No reflectance columns found. Expected 'R@...' columns.")

        self.df["shape_uid"] = (
            self.df["prefix"].astype(str)
            + "_" + self.df["nQ"].astype(str)
            + "_" + self.df["nS"].astype(str)
            + "_" + self.df["shape_idx"].astype(str)
        )

        self.data_list = []
        grouped = self.df.groupby("shape_uid", sort=False)

        for uid, grp in grouped:
            # Expect 11 lines => c=0..1
            if len(grp) != 11:
                continue
            grp_sorted = grp.sort_values(by="c")  # ensure c= ascending

            # parse input_spectra => shape(11,100)
            spectra_data = grp_sorted[self.r_cols].values.astype(np.float32)

            # parse shape from first row
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
                        all_xy.append([x_val,y_val])
            all_xy = np.array(all_xy, dtype=np.float32)
            if len(all_xy)==0:
                continue

            # SHIFT => minus (0.5,0.5)
            shifted = all_xy - 0.5

            # keep Q1 => x>0,y>0
            q1_points = []
            for (xx,yy) in shifted:
                if xx>0 and yy>0:
                    q1_points.append([xx,yy])
            q1_points = np.array(q1_points, dtype=np.float32)
            n_q1 = len(q1_points)
            if n_q1<1 or n_q1>max_points:
                continue

            # build (4,3) => [presence, x, y]
            shape_arr = np.zeros((max_points,3),dtype=np.float32)
            for i in range(n_q1):
                shape_arr[i,0] = 1.0
                shape_arr[i,1] = q1_points[i,0]
                shape_arr[i,2] = q1_points[i,1]

            self.data_list.append({
                "uid": uid,
                "spectra": spectra_data,  # (11,100)
                "shape": shape_arr        # (4,3)
            })

        self.data_len = len(self.data_list)
        if self.data_len == 0:
            raise ValueError("No valid shapes found after SHIFT->Q1->UpTo4 filtering.")

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        item = self.data_list[idx]
        return (item["spectra"], item["shape"], item["uid"])


###############################################################################
# 2) Model: (11x100) => (4x3)
###############################################################################
class Spectra2ShapeTransformer(nn.Module):
    """
    Input: spectra => shape(11,100)
    We treat each of the 11 "c-value rows" as a token. dimension=100.
    A small Transformer aggregator => (batch, d_model).
    Then an MLP => presence_logit + x + y for 4 points => shape(4,3).
    We'll do a straight-through presence approach.

    STE presence approach:
      presence_logit => presence = sigmoid
      presence_hard = (presence>0.5).float() + presence - presence.detach()
      final_x = presence_hard * x
      final_y = presence_hard * y
      final => shape(4,3).
    """
    def __init__(self, d_in=100, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model

        # 1) input projection: (100 -> d_model)
        self.input_proj = nn.Linear(d_in, d_model)

        # 2) Transformer aggregator over 11 tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 3) MLP => (4*3) = 12 => we interpret as presence_logit(4) + x(4) + y(4).
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 12)  # => (4*3)
        )

    def forward(self, spectra):
        """
        spectra => shape (B,11,100)
        returns shape => (B,4,3)
        """
        bsz = spectra.size(0)
        # pass each row of shape(11,100) => tokens(11,d_model)
        x_proj = self.input_proj(spectra)        # (B,11,d_model)
        x_enc  = self.encoder(x_proj)            # (B,11,d_model)
        # aggregator => simple mean
        x_agg = x_enc.mean(dim=1)                # (B,d_model)

        # decode => (B,12)
        out_12 = self.mlp(x_agg)                 # (B,12)
        out_reshaped = out_12.view(bsz,4,3)      # (B,4,3)

        # 4) presence_logit => presence => STE => x,y
        presence_logit = out_reshaped[:,:,0]     # (B,4)
        presence_prob  = torch.sigmoid(presence_logit)
        # straight-through approach
        presence_hard = (presence_prob>0.5).float() + presence_prob - presence_prob.detach()

        xy_raw = out_reshaped[:,:,1:]            # (B,4,2)
        # final => presence_hard * x,y
        # broadcast presence_hard => shape(B,4,1)
        xy = xy_raw * presence_hard.unsqueeze(-1)

        # combine presence_hard + xy => (B,4,3)
        final_shape = torch.cat([presence_hard.unsqueeze(-1), xy], dim=-1)
        return final_shape


###############################################################################
# 3) Visualization (3 columns)
###############################################################################
def replicate_c4(points):
    """
    Replicate a set of Q1 points (x,y) into 4 rotations:
       (x,  y)
       (-y, x)
       (-x, -y)
       (y, -x)
    """
    c4 = []
    for (x,y) in points:
        c4.append([ x,  y])    # 0°
        c4.append([-y,  x])    # 90°
        c4.append([-x, -y])    # 180°
        c4.append([ y, -x])    # 270°
    return np.array(c4, dtype=np.float32)

def sort_points_by_angle(points):
    """
    Sort points by polar angle for a consistent closed polygon.
    """
    if len(points)<3:
        return points
    cx, cy = points.mean(axis=0)
    angles = np.arctan2(points[:,1]-cy, points[:,0]-cx)
    idx = np.argsort(angles)
    return points[idx]

def plot_polygon(ax, points, color='green', alpha=0.4, fill=True):
    """
    Draw a closed polygon from Nx2 points
    """
    import matplotlib.patches as patches
    from matplotlib.path import Path

    if len(points) < 3:
        ax.scatter(points[:,0], points[:,1], c=color)
        return
    closed = np.concatenate([points, points[0:1]], axis=0)
    codes = [Path.MOVETO] + [Path.LINETO]*(len(points)-1) + [Path.CLOSEPOLY]
    path = Path(closed, codes)
    patch = patches.PathPatch(
        path, facecolor=color if fill else 'none', alpha=alpha, edgecolor=color
    )
    ax.add_patch(patch)
    ax.autoscale_view()

def plot_spectra(ax, spectra_gt, color='blue'):
    """
    Plot 11x100 ground-truth spectra.
    """
    for row in spectra_gt:
        ax.plot(row, color=color, alpha=0.5)
    ax.set_xlabel("wavelength index")
    ax.set_ylabel("Reflectance")

def visualize_3col(
    model, ds_val, device, out_dir=".",
    sample_count=4, random_seed=123
):
    """
    Creates sample_count rows, 3 columns:
       col1 => GT spectrum
       col2 => GT shape(green) vs predicted shape(red) c4
       col3 => the same GT spectrum again (blue) 
               (as requested: "both left & right are the same GT plots").
    """
    import random
    random.seed(random_seed)
    os.makedirs(out_dir, exist_ok=True)

    if len(ds_val)==0:
        print("[Warning] No data in val set.")
        return

    idx_samples = random.sample(range(len(ds_val)), min(sample_count,len(ds_val)))
    n_rows = len(idx_samples)

    fig, axes = plt.subplots(n_rows, 3, figsize=(12,3*n_rows))
    if n_rows==1:
        axes = [axes]

    model.eval()
    with torch.no_grad():
        for i, idx_ in enumerate(idx_samples):
            spec_np, shape_np, uid_ = ds_val[idx_]
            # col1 => GT spectrum
            axL = axes[i][0]
            plot_spectra(axL, spec_np, color='blue')
            axL.set_title(f"UID={uid_}\n(Left) GT spectrum")

            # col2 => GT shape vs Pred shape
            axM = axes[i][1]
            # GT shape => presence >0 => q1 points
            pres_gt = shape_np[:,0]>0.5
            q1_gt = shape_np[pres_gt,1:3]
            if len(q1_gt)>0:
                c4_gt = replicate_c4(q1_gt)
                c4_gt_sorted = sort_points_by_angle(c4_gt)
                plot_polygon(axM, c4_gt_sorted, color='green', alpha=0.4, fill=True)

            # predicted shape
            spec_t = torch.tensor(spec_np, dtype=torch.float32, device=device).unsqueeze(0) # (1,11,100)
            pred_4x3 = model(spec_t).cpu().numpy()[0] # => (4,3)
            pres_pred = pred_4x3[:,0]>0.5
            q1_pred = pred_4x3[pres_pred,1:3]
            if len(q1_pred)>0:
                c4_pred = replicate_c4(q1_pred)
                c4_pred_sorted = sort_points_by_angle(c4_pred)
                plot_polygon(axM, c4_pred_sorted, color='red', alpha=0.3, fill=False)

            axM.set_xlim([-0.5,0.5])
            axM.set_ylim([-0.5,0.5])
            axM.set_aspect('equal','box')
            axM.grid(True)
            axM.set_title("(Middle)\nGT(green) vs Pred(red) shape (C4)")

            # col3 => the same GT spectrum again
            axR = axes[i][2]
            plot_spectra(axR, spec_np, color='blue')
            axR.set_title("(Right) GT spectrum")

    plt.tight_layout()
    out_fig = os.path.join(out_dir, "samples_3col_plot.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"[Visualization saved] => {out_fig}")


###############################################################################
# 4) Training loop
###############################################################################
def train_spectrum2shape(
    csv_file,
    out_dir="outputs_spectrum2shape",
    num_epochs=100,
    batch_size=4096,  # always 4096
    lr=1e-4,
    weight_decay=1e-5,
    split_ratio=0.8,
    d_model=128,
    nhead=4,
    num_layers=2
):
    # create datetime suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"{out_dir}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # 1) load dataset => (spectra(11,100), shape(4,3))
    ds_full = Q1ShiftedSpectraDataset(csv_file)
    ds_len  = len(ds_full)
    train_len = int(ds_len*split_ratio)
    val_len   = ds_len - train_len
    ds_train, ds_val = random_split(ds_full, [train_len, val_len])
    print(f"[DATA] total={ds_len}, train={train_len}, val={val_len}")

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2) model
    model = Spectra2ShapeTransformer(d_in=100, d_model=d_model, nhead=nhead, num_layers=num_layers)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # reduce LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    # define a shape loss
    # we can do MSE between predicted shape(4,3) and target(4,3)
    # ignoring presence bits means we let the network learn them. We'll do full MSE
    # so presence=1 => we want x,y=gt, presence=0 => we want x,y=0
    criterion = nn.MSELoss()

    # 3) Train loop
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        run_loss=0.0
        for spectra_np, shape_np, uid_list in train_loader:
            spectra_t = spectra_np.to(device)  # (B,11,100)
            shape_t   = shape_np.to(device)    # (B,4,3)

            pred_shape = model(spectra_t)      # (B,4,3)
            loss = criterion(pred_shape, shape_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item()

        avg_train = run_loss/len(train_loader)
        train_losses.append(avg_train)

        # validation
        model.eval()
        val_sum=0.0
        val_count=0
        with torch.no_grad():
            for spectra_np, shape_np, uid_list in val_loader:
                bsz_ = spectra_np.size(0)
                spectra_t = spectra_np.to(device)
                shape_t   = shape_np.to(device)
                pred_shape= model(spectra_t)
                vloss = criterion(pred_shape, shape_t)*bsz_
                val_sum += vloss.item()
                val_count+=bsz_
        avg_val = val_sum/val_count
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if (epoch+1) % 20==0 or epoch==0 or epoch==(num_epochs-1):
            print(f"Epoch[{epoch+1}/{num_epochs}] => trainMSE={avg_train:.4f}, valMSE={avg_val:.4f}")

    # 4) plot training curve
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses,   label='val')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Spectrum->Shape Training (STE presence)")
    plt.legend()
    plt.savefig(os.path.join(out_dir,"training_curve.png"))
    plt.close()

    # 5) final val MSE
    model.eval()
    val_sum, val_count=0.0,0
    with torch.no_grad():
        for spectra_np, shape_np, uid_list in val_loader:
            bsz_ = spectra_np.size(0)
            spectra_t = spectra_np.to(device)
            shape_t   = shape_np.to(device)
            pred_shape= model(spectra_t)
            loss_ = criterion(pred_shape, shape_t)*bsz_
            val_sum+=loss_.item()
            val_count+=bsz_
    final_mse = val_sum/val_count
    print(f"[Final Val MSE] => {final_mse:.6f}")

    # 6) save model
    model_path = os.path.join(out_dir,"spectrum2shape_model.pt")
    torch.save(model.state_dict(), model_path)
    print("[Model saved] =>", model_path)

    # 7) quick sample visualization => 3 columns
    if val_len>0:
        visualize_3col(model, ds_val, device, out_dir=out_dir, sample_count=4, random_seed=42)


###############################################################################
# if-name-main usage
###############################################################################
def main():
    CSV_FILE = "merged_s4_shapes_iccpOv10kG40_seed88888.csv"
    train_spectrum2shape(
        csv_file=CSV_FILE,
        out_dir="outputs_spectrum2shape",
        num_epochs=1000,
        batch_size=4096,
        lr=1e-4,
        weight_decay=1e-5,
        split_ratio=0.8,
        d_model=128,
        nhead=4,
        num_layers=2
    )

if __name__=="__main__":
    main()

