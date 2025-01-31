import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split


###############################################################################
# 1. Utility functions for visualization
###############################################################################
def c4_polygon(ax, points, color='green', alpha=0.3, fill=True):
    """
    Plot the polygon given by `points` on axis `ax`.
    Assumes points is an Nx2 array.
    """
    import matplotlib.patches as patches
    from matplotlib.path import Path

    if len(points) < 3:
        # Just plot them as scatter if not enough points
        ax.scatter(points[:,0], points[:,1], c=color)
        return

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

def plot_spectra(ax, spectra_gt, spectra_pred, color_gt='blue', color_pred='red'):
    """
    Plot ground-truth vs. predicted spectra.
    spectra_gt, spectra_pred: arrays of shape (11,100).
    Each of the 11 rows corresponds to one c-value's 100 reflectance points.
    """
    n_c = spectra_gt.shape[0]  # should be 11
    for i in range(n_c):
        ax.plot(spectra_gt[i], color=color_gt, alpha=0.5)
        ax.plot(spectra_pred[i], color=color_pred, alpha=0.5, linestyle='--')
    ax.set_xlabel("Wavelength Index (0..99)")
    ax.set_ylabel("Reflectance")
    ax.set_title("Spectra (GT vs Pred)")

###############################################################################
# 2. Dataset
###############################################################################
class ShapeToSpectraDataset(Dataset):
    """
    Reads shape data from a CSV that actually has 11 rows per shape 
    (one for each c=0..1). We group them by a unique ID.

    For each shape, we:
      - Parse up to N=16 vertices (some shapes might have 4,8,12, or 16).
      - Store them as (presence, x, y).
      - Gather the 11 rows of reflectance data into a single target array (11,100).
    
    The dataset returns:
      shape_array: (N, 3) float => [presence, x, y] for each padded vertex (N=16).
      target_spectra: (11,100)
      uid_str: a unique identifier (for logging/plotting).
    """
    def __init__(self, csv_file, max_vertices=16):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        
        # Identify reflectance columns => "R@1.040", "R@1.054", ..., "R@2.502" (100 of them).
        self.r_cols = [c for c in self.df.columns if c.startswith("R@")]

        # Create a UID from certain columns (customize as needed).
        self.df["uid"] = (
            self.df["prefix"].astype(str)
            + "_" + self.df["nQ"].astype(str)
            + "_" + self.df["nS"].astype(str)
            + "_" + self.df["shape_idx"].astype(str)
        )

        self.data_list = []
        grouped = self.df.groupby("uid")

        for uid, grp in grouped:
            # The CSV is known to have 11 distinct c-values per shape, but we will
            # NOT feed c to the model; we just read those 11 lines for the ground truth.
            grp_sorted = grp.sort_values(by="c")
            if len(grp_sorted) != 11:
                continue

            # Parse vertices (from the first row is enough, they are all repeated anyway):
            first_row = grp_sorted.iloc[0]
            v_str = first_row["vertices_str"]
            raw_pairs = v_str.split(";")
            vertices = []
            for pair in raw_pairs:
                pair = pair.strip()
                if pair:
                    xy = pair.split(",")
                    if len(xy) == 2:
                        x_val = float(xy[0])
                        y_val = float(xy[1])
                        vertices.append((x_val, y_val))

            # Possibly you have 4,8,12,16 vertices => pad up to max_vertices=16
            vertices = vertices[:max_vertices]
            num_actual = len(vertices)
            while len(vertices) < max_vertices:
                vertices.append((0.0, 0.0))

            # presence flags:
            presence_vals = [1.0]*num_actual + [0.0]*(max_vertices - num_actual)
            
            shape_array = []
            for i in range(max_vertices):
                shape_array.append([
                    presence_vals[i],
                    vertices[i][0],
                    vertices[i][1]
                ])
            shape_array = np.array(shape_array, dtype=np.float32)  # (16,3)

            # Gather reflectances => shape(11,100)
            R_data = grp_sorted[self.r_cols].values.astype(np.float32)  # (11, 100)

            self.data_list.append({
                "uid": uid,
                "shape": shape_array,
                "spectra": R_data
            })

        self.data_len = len(self.data_list)
        if self.data_len == 0:
            raise ValueError("No valid shapes found. Check the CSV or grouping keys.")

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        item = self.data_list[idx]
        return item["shape"], item["spectra"], item["uid"]

###############################################################################
# 3. Transformer-based aggregator
###############################################################################
class TransformerEncoder(nn.Module):
    """
    A mini-Transformer encoder to handle up to max_vertices (e.g. 16) tokens.
    We embed each token => self-attention => final shape embedding.
    """
    def __init__(self, d_in=3, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(d_in, d_model)

        # TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x_masked):
        """
        x_masked: (bsz, N, 3) => [presence, x, y], with N up to 16 (or 4,8,12).
        We create a key_padding_mask based on presence=0 => ignore those tokens.
        """
        bsz, N, _ = x_masked.shape

        # Linear embedding
        x_emb = self.input_proj(x_masked)  # (bsz, N, d_model)

        # presence => shape (bsz, N)
        presence = x_masked[:,:,0]
        key_padding_mask = (presence < 0.5)  # True => mask out

        # Pass through transformer encoder
        x_enc = self.transformer_encoder(
            x_emb,
            src_key_padding_mask=key_padding_mask
        )  # (bsz, N, d_model)

        # Weighted average pooling over the unmasked tokens
        presence_sum = torch.sum(presence, dim=1, keepdim=True) + 1e-8
        x_enc_weighted = x_enc * presence.unsqueeze(-1)  # (bsz, N, d_model)
        shape_embedding = torch.sum(x_enc_weighted, dim=1) / presence_sum  # (bsz, d_model)

        return shape_embedding


class Shape2SpectraTransformerNet(nn.Module):
    """
    Model:
      - Transformer aggregator => (bsz, d_model)
      - MLP => outputs a flattened vector of size (11*100)
      - Reshape to (bsz, 11, 100)
    """
    def __init__(self, d_in=3, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.transformer_agg = TransformerEncoder(
            d_in=d_in,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )
        self.out_dim = 11 * 100  # We want to produce 11x100 = 1100 total

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, self.out_dim)
        )

    def forward(self, shape_array):
        """
        shape_array: (bsz, N, 3), N up to 16.
        Return => (bsz, 11, 100)
        """
        bsz = shape_array.size(0)
        shape_emb = self.transformer_agg(shape_array)  # (bsz, d_model)
        out_flat = self.mlp(shape_emb)                 # (bsz, 1100)
        out_2d = out_flat.view(bsz, 11, 100)           # (bsz, 11, 100)
        return out_2d

###############################################################################
# 4. Visualization helpers
###############################################################################
def get_num_verts(shape_np):
    """Count how many vertices have presence>0.5 in shape_np."""
    return int((shape_np[:,0] > 0.5).sum())

def visualize_4_samples(model, ds_test, device, out_dir=".", n_rows=4):
    """
    Find up to 4 samples with different vertex counts (like 1..4, or 4..8..12..16),
    generate shape polygon + compare predicted vs GT 11×100 spectra.
    """
    desired_counts = [4, 8, 12, 16]  # or any you want to check
    found_idx = {}

    # Search for shapes with exactly these vertex counts
    # (If you only have certain shapes in your dataset, adjust logic as needed.)
    for idx in range(len(ds_test)):
        shape_np, spectra_np, uid_ = ds_test[idx]
        n_verts = get_num_verts(shape_np)
        if n_verts in desired_counts and (n_verts not in found_idx):
            found_idx[n_verts] = idx
        if len(found_idx) == len(desired_counts):
            break

    if len(found_idx) == 0:
        print("No shapes found in test set for vertex counts 4,8,12,16.")
        return

    # We'll plot as many rows as we actually found
    found_counts_sorted = sorted(found_idx.keys())
    n_rows_plot = len(found_counts_sorted)
    fig, axes = plt.subplots(n_rows_plot, 2, figsize=(10, 3*n_rows_plot))

    if n_rows_plot == 1:
        axes = [axes]

    for row_idx, v_count in enumerate(found_counts_sorted):
        sample_idx = found_idx[v_count]
        shape_np, spectra_gt, uid_ = ds_test[sample_idx]
        shape_t = torch.tensor(shape_np, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            spectra_pred = model(shape_t).cpu().numpy()[0]  # (11,100)

        # 1) Polygon
        ax_shape = axes[row_idx][0]
        shape_points = []
        for i in range(shape_np.shape[0]):
            p = shape_np[i,0]
            x = shape_np[i,1]
            y = shape_np[i,2]
            if p>0.5:
                shape_points.append([x,y])
        shape_points = np.array(shape_points)
        ax_shape.set_aspect('equal', 'box')
        if shape_points.shape[0] > 0:
            c4_polygon(ax_shape, shape_points, color='green', alpha=0.3, fill=True)
        ax_shape.grid(True)
        ax_shape.set_title(f"UID={uid_}, #Vert={v_count}")

        # 2) Spectra
        ax_spec = axes[row_idx][1]
        plot_spectra(ax_spec, spectra_gt, spectra_pred, color_gt='blue', color_pred='red')

    plt.tight_layout()
    save_path = os.path.join(out_dir, "4_rows_visualization.png")
    plt.savefig(save_path)
    plt.close()
    print(f"4-sample visualization saved to {save_path}")

###############################################################################
# 5. Main training function
###############################################################################
def train_shape2spectra_transformer(
    csv_file="merged_s4_shapes.csv",
    save_dir="outputs",
    num_epochs=100,
    batch_size=64,
    lr=1e-4,
    weight_decay=1e-5,
    split_ratio=0.8,
    d_model=128,
    nhead=4,
    num_layers=2
):
    """
    Main training loop. 
    1) Reads CSV via ShapeToSpectraDataset.
    2) Splits into train/test.
    3) Defines a Transformer aggregator model that outputs 11×100.
    4) Trains with MSE loss.
    5) Saves model, plots training loss, and does some final visualizations.
    """
    timestamp = f"{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    out_dir = os.path.join(save_dir, f"shape2spectra_transformer_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load dataset
    ds_full = ShapeToSpectraDataset(csv_file, max_vertices=16)
    print("Total shapes in dataset:", len(ds_full))

    ds_len = len(ds_full)
    train_len = int(ds_len * split_ratio)
    test_len  = ds_len - train_len
    ds_train, ds_test = random_split(ds_full, [train_len, test_len])
    print(f"Train size={len(ds_train)}, Test size={len(ds_test)}")

    if len(ds_train) == 0 or len(ds_test) == 0:
        raise ValueError("Train or Test set is empty. Adjust split_ratio or check data.")

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2) Model
    model = Shape2SpectraTransformerNet(
        d_in=3,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    criterion = nn.MSELoss(reduction='mean')

    # 3) Training loop
    all_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for (shape_np, spectra_np, uid_list) in train_loader:
            shape  = shape_np.to(device)   # (bsz, N, 3)
            target = spectra_np.to(device) # (bsz, 11, 100)

            pred = model(shape)           # (bsz, 11, 100)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        all_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for (shape_np, spectra_np, uid_list) in test_loader:
                shape  = shape_np.to(device)
                target = spectra_np.to(device)
                bsz_   = shape.size(0)
                pred = model(shape)
                vloss = criterion(pred, target) * bsz_
                val_loss_sum += vloss.item()
                val_count += bsz_
        val_mse = val_loss_sum / val_count

        scheduler.step(val_mse)

        if (epoch+1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] TrainLoss={epoch_loss:.4f} ValMSE={val_mse:.4f}")

    # 4) Plot training curve
    plt.figure()
    plt.plot(all_losses, label="Train MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Transformer Aggregator: Shape->(11×100) Spectra")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_loss.png"))
    plt.close()

    # 5) Save model
    model_path = os.path.join(out_dir, "shape2spectra_transformer_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 6) Final test MSE
    model.eval()
    test_loss = 0.0
    test_count = 0
    with torch.no_grad():
        for (shape_np, spectra_np, uid_list) in test_loader:
            shape  = shape_np.to(device)
            target = spectra_np.to(device)
            bsz_   = shape.size(0)
            pred = model(shape)
            loss = criterion(pred, target) * bsz_
            test_loss += loss.item()
            test_count += bsz_
    test_mse = test_loss / test_count
    print(f"Final Test MSE: {test_mse:.6f}")

    # 7) Visualize a few shapes
    visualize_4_samples(model, ds_test, device, out_dir=out_dir, n_rows=4)

    # 8) Single-sample visualize
    import numpy as np
    sample_idx = np.random.randint(0, len(ds_test))
    shape_np, spectra_gt, uid_ = ds_test[sample_idx]
    shape_t = torch.tensor(shape_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        spectra_pred = model(shape_t).cpu().numpy()[0]  # (11,100)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot shape
    shape_points = []
    for i in range(shape_np.shape[0]):
        p = shape_np[i,0]
        x = shape_np[i,1]
        y = shape_np[i,2]
        if p>0.5:
            shape_points.append([x,y])
    shape_points = np.array(shape_points)
    axs[0].set_aspect('equal', 'box')
    if shape_points.size > 0:
        c4_polygon(axs[0], shape_points, color='green', alpha=0.3, fill=True)
    axs[0].grid(True)
    axs[0].set_title(f"Single Sample (UID={uid_})")

    # Plot spectra
    plot_spectra(axs[1], spectra_gt, spectra_pred, color_gt='blue', color_pred='red')

    plt.tight_layout()
    single_vis_path = os.path.join(out_dir, "single_sample_visualization.png")
    plt.savefig(single_vis_path)
    plt.close()
    print(f"Single-sample visualization saved to {single_vis_path} "
          f"(idx={sample_idx}, uid={uid_})")


if __name__ == "__main__":
    # Example usage
    train_shape2spectra_transformer(
        # csv_file="merged_s4_shapes_20250119_153038.csv",
        csv_file="merged_s4_shapes_iccpOv10kG40_seed88888.csv",
        save_dir="outputs",
        num_epochs=1000,
        batch_size=64*64,
        lr=1e-3,
        #weight_decay=1e-5,
        weight_decay=1,
        split_ratio=0.8,
        d_model=256,
        nhead=4,
        num_layers=4
    )
