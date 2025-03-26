#!/usr/bin/env python3

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


###############################################################################
# 1) Helper Functions for Polygon Plotting
###############################################################################
def plot_polygon(ax, points, color='green', alpha=0.4, fill=True):
    """
    Plot the polygon given by `points` on axis `ax`.
    Assumes `points` is an Nx2 array. If fewer than 3 points, just scatter them.
    """
    import matplotlib.patches as patches
    from matplotlib.path import Path

    if len(points) < 3:
        # Just scatter if not enough points to form a real polygon
        ax.scatter(points[:,0], points[:,1], c=color)
        return

    # Close the polygon by appending the first point at the end
    verts = np.concatenate([points, points[0:1]], axis=0)
    codes = [Path.MOVETO] + [Path.LINETO]*(len(points)-1) + [Path.CLOSEPOLY]
    path = Path(verts, codes)
    patch = patches.PathPatch(
        path,
        facecolor=color if fill else 'none',
        alpha=alpha,
        edgecolor=color,
        lw=1.5
    )
    ax.add_patch(patch)
    ax.autoscale_view()

def plot_spectra(ax, spectra_gt, spectra_pred=None, color_gt='blue', color_pred='red'):
    """
    Plot ground-truth (GT) spectra, shape = (11,100).
    Optionally plot predicted spectra (same shape) in a different color.
    """
    n_c = spectra_gt.shape[0]  # typically 11
    for i in range(n_c):
        ax.plot(spectra_gt[i], color=color_gt, alpha=0.5)
    if spectra_pred is not None:
        for i in range(n_c):
            ax.plot(spectra_pred[i], color=color_pred, alpha=0.5, linestyle='--')

    ax.set_xlabel("Wavelength index (0..99)")
    ax.set_ylabel("Reflectance")
    ax.set_title("Spectra")


###############################################################################
# 2) Dataset: SHIFT -> Q1 Filter -> Keep up to 4 vertices
###############################################################################
class Q1ShiftedShapeDataset(Dataset):
    """
    Reads shape data from a CSV that has 11 rows for each shape (c=0..1).
    Steps:
      1) Groups by a unique shape ID => parse up to 11 lines of reflectances.
      2) From the first row's `vertices_str`, parse all vertices.
      3) SHIFT them by subtracting (0.5, 0.5).
      4) Keep only quadrant‐1 points (x>0, y>0).
      5) If the result has 1..4 points, store them with presence=1.0. If fewer
         than 4, pad with zeros. If more than 4, skip the shape entirely.
      6) Also gather the reflectances => shape(11, 100). That is the target.
    """
    def __init__(self, csv_file, max_points=4):
        super().__init__()
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.r_cols = [c for c in self.df.columns if c.startswith("R@")]
        if len(self.r_cols) == 0:
            raise ValueError("No reflectance columns found (expected columns named like 'R@1.040').")

        # Create a shape‐ID to group the rows. Adjust these columns as needed:
        self.df["shape_uid"] = (
            self.df["prefix"].astype(str)
            + "_" + self.df["nQ"].astype(str)
            + "_" + self.df["nS"].astype(str)
            + "_" + self.df["shape_idx"].astype(str)
        )

        self.data_list = []
        grouped = self.df.groupby("shape_uid", sort=False)

        for uid, grp in grouped:
            # We expect exactly 11 lines for c=0..1, but your dataset might vary
            if len(grp) != 11:
                continue
            grp_sorted = grp.sort_values(by="c")  # just in case

            # Grab reflectances => shape (11,100)
            R_data = grp_sorted[self.r_cols].values.astype(np.float32)

            # From the first row, parse the polygon:
            first_row = grp_sorted.iloc[0]
            v_str = str(first_row.get("vertices_str", "")).strip()
            if not v_str:
                continue

            # Split into pairs x,y:
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

            # SHIFT => subtract (0.5, 0.5)
            shifted = all_xy - 0.5

            # Keep Q1 => x>0, y>0
            q1_points = []
            for (xx, yy) in shifted:
                if xx>0 and yy>0:
                    q1_points.append([xx, yy])
            q1_points = np.array(q1_points, dtype=np.float32)

            # If no points or >4 points => skip
            n_q1 = len(q1_points)
            if n_q1 == 0 or n_q1 > max_points:
                continue

            # Build a (4,3) array => [presence, x, y]
            shape_arr = np.zeros((max_points, 3), dtype=np.float32)
            for i in range(n_q1):
                shape_arr[i, 0] = 1.0
                shape_arr[i, 1] = q1_points[i,0]
                shape_arr[i, 2] = q1_points[i,1]

            self.data_list.append({
                "uid": uid,
                "shape": shape_arr,    # (4,3)
                "spectra": R_data      # (11,100)
            })

        self.data_len = len(self.data_list)
        if self.data_len == 0:
            raise ValueError(
                f"After SHIFT->Q1->UpTo4, no valid shapes found in {csv_file}!"
            )

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        item = self.data_list[idx]
        return item["shape"], item["spectra"], item["uid"]


###############################################################################
# 3) Transformer Model: shape => spectrum
###############################################################################
class ShapeEncoder(nn.Module):
    """
    A mini-Transformer aggregator that handles up to 4 tokens:
      token = [presence, x, y].
    We use presence=0 to mask out absent tokens.
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
        shape_tokens: (B, 4, 3) => [presence, x, y].
                      Some tokens may have presence=0 => mask them out.
        Returns an aggregated shape embedding => (B, d_model).
        """
        bsz, N, _ = shape_tokens.shape  # typically N=4

        # Create key_padding_mask from presence=0
        presence = shape_tokens[:,:,0]          # (B,4)
        key_padding_mask = (presence < 0.5)     # True => mask out

        x_emb = self.input_proj(shape_tokens)   # (B,4,d_model)
        x_enc = self.encoder(x_emb, src_key_padding_mask=key_padding_mask)
        # Weighted sum by presence
        presence_sum = presence.sum(dim=1, keepdim=True) + 1e-8
        x_enc_weighted = x_enc * presence.unsqueeze(-1)  # broadcast
        shape_emb = x_enc_weighted.sum(dim=1) / presence_sum
        return shape_emb

class ShapeToSpectraModel(nn.Module):
    """
    shape => (11 x 100) reflectance
    1) Transformer aggregator => shape embedding (d_model)
    2) MLP => (11*100) => reshape => (11,100)
    """
    def __init__(self, d_in=3, d_model=128, nhead=4, num_layers=2, out_dim=1100):
        super().__init__()
        self.encoder = ShapeEncoder(d_in, d_model, nhead, num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, out_dim)  # 11*100
        )

    def forward(self, shape_arr):
        """
        shape_arr: (B,4,3)
        Returns (B,11,100)
        """
        bsz = shape_arr.size(0)
        shape_emb = self.encoder(shape_arr)        # (B, d_model)
        out_flat  = self.mlp(shape_emb)            # (B, 1100)
        out_spec  = out_flat.view(bsz, 11, 100)    # (B,11,100)
        return out_spec


###############################################################################
# 4) Training + Visualization
###############################################################################
def visualize_4x3_samples(
    model, ds_test, device, out_dir=".", 
    # We'll try to find shapes with 1..4 Q1 points for a nice 4-row figure
    desired_counts=[1,2,3,4]
):
    os.makedirs(out_dir, exist_ok=True)
    found_idx = {}

    # For each shape in test dataset, see how many Q1 points it has:
    for idx in range(len(ds_test)):
        shape_np, spectra_gt, uid_ = ds_test[idx]
        # presence flags:
        presence = (shape_np[:,0] > 0.5)
        n_q1 = int(presence.sum())
        if n_q1 in desired_counts and (n_q1 not in found_idx):
            found_idx[n_q1] = idx
        if len(found_idx) == len(desired_counts):
            break

    found_counts_sorted = sorted(found_idx.keys())
    if len(found_counts_sorted) == 0:
        print("No test samples found with 1..4 Q1 points. Visualization skipped.")
        return

    n_rows = len(found_counts_sorted)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3*n_rows))
    if n_rows == 1:
        axes = [axes]

    model.eval()
    with torch.no_grad():
        for row_i, n_q1 in enumerate(found_counts_sorted):
            idx_ = found_idx[n_q1]
            shape_np, spectra_gt, uid_ = ds_test[idx_]

            # (1) Original spectrum
            ax_left = axes[row_i][0]
            plot_spectra(ax_left, spectra_gt, spectra_pred=None, color_gt='blue')
            ax_left.set_title(f"GT Spectrum\nuid={uid_}, #Q1={n_q1}")

            # (2) GT shape vs "pred shape" (here we only have the GT shape as input,
            #     so we'll just plot it in 2 colors for demonstration)
            ax_mid = axes[row_i][1]
            ax_mid.set_aspect("equal", "box")
            ax_mid.grid(True)
            # Extract the Q1 points
            presence = shape_np[:,0] > 0.5
            q1_pts = shape_np[presence, 1:3]
            # Plot in green
            plot_polygon(ax_mid, q1_pts, color='green', alpha=0.3, fill=True)
            # Also show "pred shape" in red (since we don't truly predict shape in shape->spectrum,
            # we just replicate the same shape in red to illustrate the "two shapes" idea)
            plot_polygon(ax_mid, q1_pts, color='red', alpha=0.3, fill=False)
            ax_mid.set_title("GT shape (green) & 'pred' (red)")

            # (3) Reconstruction: feed shape => predicted spectrum
            shape_t = torch.tensor(shape_np, dtype=torch.float32, device=device).unsqueeze(0)
            spectra_pred = model(shape_t).cpu().numpy()[0]  # (11,100)
            ax_right = axes[row_i][2]
            plot_spectra(ax_right, spectra_gt, spectra_pred, color_gt='blue', color_pred='red')
            ax_right.set_title("Reconstructed vs GT\nSpectra")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "4x3_visualization.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Visualization] saved => {out_path}")


def train_shape2spectrum(
    csv_file,
    out_dir="outputs_shape2spectrum",
    num_epochs=100,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-5,
    split_ratio=0.8,
    d_model=128,
    nhead=4,
    num_layers=2
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Build dataset
    ds_full = Q1ShiftedShapeDataset(csv_file, max_points=4)
    ds_len = len(ds_full)
    train_len = int(ds_len * split_ratio)
    val_len = ds_len - train_len
    ds_train, ds_val = random_split(ds_full, [train_len, val_len])
    print(f"[DATA] total={ds_len}, train={train_len}, val={val_len}")

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2) Model
    model = ShapeToSpectraModel(
        d_in=3,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        out_dim=11*100
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    criterion = nn.MSELoss()

    # 3) Training loop
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        # === Train ===
        model.train()
        running_loss = 0.0
        for (shape_np, spec_np, uid_list) in train_loader:
            shape_t = shape_np.to(device)      # (B,4,3)
            target  = spec_np.to(device)       # (B,11,100)

            pred = model(shape_t)             # (B,11,100)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_trn = running_loss / len(train_loader)
        train_losses.append(avg_trn)

        # === Validation ===
        model.eval()
        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for (shape_np, spec_np, uid_list) in val_loader:
                bsz_ = shape_np.size(0)
                shape_t = shape_np.to(device)
                target  = spec_np.to(device)
                pred    = model(shape_t)
                vloss   = criterion(pred, target) * bsz_
                val_sum += vloss.item()
                val_count += bsz_
        avg_val = val_sum / val_count
        val_losses.append(avg_val)

        scheduler.step(avg_val)

        if (epoch+1) % 20 == 0 or epoch == 1 or epoch == (num_epochs-1):
            print(f"Epoch[{epoch+1}/{num_epochs}] Train MSE={avg_trn:.4f} Val MSE={avg_val:.4f}")

    # === Plot training curve ===
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Shape->Spectrum Training")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_curve.png"))
    plt.close()

    # === Final test MSE ===
    test_loss = 0.0
    test_count = 0
    model.eval()
    with torch.no_grad():
        for (shape_np, spec_np, uid_list) in val_loader:
            bsz_   = shape_np.size(0)
            shape_t = shape_np.to(device)
            target  = spec_np.to(device)
            pred    = model(shape_t)
            loss_   = criterion(pred, target) * bsz_
            test_loss += loss_.item()
            test_count += bsz_
    test_mse = test_loss / test_count
    print(f"[Final Validation MSE] => {test_mse:.6f}")

    # === Save model ===
    model_path = os.path.join(out_dir, "shape2spectrum_model.pt")
    torch.save(model.state_dict(), model_path)
    print("[Model Saved] =>", model_path)

    # === Visualization of some shapes (4x3) ===
    visualize_4x3_samples(model, ds_val, device, out_dir=out_dir)


###############################################################################
# 5) Main
###############################################################################
def main():
    CSV_FILE = "merged_s4_shapes_iccpOv10kG40_seed88888.csv"  # example
    train_shape2spectrum(
        csv_file=CSV_FILE,
        out_dir="outputs_shape2spectrum",
        num_epochs=100,
        batch_size=32*32,
        lr=1e-4,
        weight_decay=1e-5,
        split_ratio=0.8,
        d_model=128,
        nhead=4,
        num_layers=2
    )

if __name__=="__main__":
    main()

