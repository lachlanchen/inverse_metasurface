import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

###############################################################################
# 1. Utility: c4_polygon & plot_spectra (same as your code if needed)
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

    # close the polygon
    verts = np.concatenate([points, points[0:1]], axis=0)
    codes = [Path.MOVETO] + [Path.LINETO]*(len(points)-1) + [Path.CLOSEPOLY]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor=color if fill else 'none', 
                              alpha=alpha, edgecolor=color, lw=1.5)
    ax.add_patch(patch)
    ax.autoscale_view()

def plot_spectra(ax, spectra_gt, spectra_pred, color_gt='blue', color_pred='red'):
    """
    Plot ground truth vs predicted spectra.
    spectra_gt, spectra_pred: shape (11, 100).
    We'll plot each c as a separate line, or optionally overlay them.
    """
    n_c = spectra_gt.shape[0]
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
    Reads shapes from CSV with up to 4 vertices. 
    Each shape is repeated 11 times with c=0..1 (in increments of 0.1).
    We'll group those 11 rows into one sample. The reflectance columns 
    are R@1.040 ... R@2.502 => 100 columns in your data.

    The dataset returns:
        shape_array: (4,3) float => [presence, x, y] for each of up to 4 vertices
        target_spectra: (11,100) float => reflectance for c=0..1
        uid_str: unique id for debugging
    """
    def __init__(self, csv_file, max_vertices=16):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        
        # You can adjust which columns are used as reflectance
        # Here we assume columns "R@1.040, R@1.054, ... R@2.502" (100 of them).
        self.r_cols = [c for c in self.df.columns if c.startswith("R@")]
        # We also assume c-values are in a column named 'c' with 11 unique values.

        # Make a UID from (folder_key, NQ, nS, shape_idx) or your preference:
        self.df["uid"] = (self.df["folder_key"].astype(str)
                          + "_" + self.df["NQ"].astype(str)
                          + "_" + self.df["nS"].astype(str)
                          + "_" + self.df["shape_idx"].astype(str))

        # Group by UID
        self.data_list = []
        for uid, grp in self.df.groupby("uid"):
            # Sort by c
            grp_sorted = grp.sort_values(by="c")
            # Expect 11 distinct c => 0.0, 0.1, 0.2, ..., 1.0
            if len(grp_sorted) != 11:
                # skip if missing
                continue

            # parse the vertices_str from the first row
            v_str = grp_sorted["vertices_str"].values[0]
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
            
            # Truncate or pad to max_vertices=4
            vertices = vertices[:max_vertices]
            num_actual = len(vertices)
            while len(vertices) < max_vertices:
                vertices.append((0.0, 0.0))
            
            presence = [1.0]*num_actual + [0.0]*(max_vertices - num_actual)
            shape_array = []
            for i in range(max_vertices):
                shape_array.append([presence[i], vertices[i][0], vertices[i][1]])
            shape_array = np.array(shape_array, dtype=np.float32)  # (4,3)

            # get reflectance => shape (11,100)
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
    A mini-Transformer encoder to aggregate sets of up to 4 tokens (vertices).
    We'll just encode each (presence,x,y) as a token vector, 
    then apply self-attention. Finally we either pool or use a CLS token.
    """

    def __init__(self, d_in=3, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        
        # Learnable linear embedding from (presence,x,y) -> d_model
        self.input_proj = nn.Linear(d_in, d_model)

        # Optionally we can add a small positional encoding or 
        # just rely on the self-attention's permutation invariance.
        # We'll keep it simple with no fixed positional encoding:
        self.register_buffer("dummy_pos", torch.zeros(1,1,d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=d_model*4, 
                                                   dropout=0.1, activation='relu',
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=num_layers)
        
    def forward(self, x_masked):
        """
        x_masked: (bsz,4,3) => we only have up to 4 vertices, but some might be presence=0.
        We'll embed them all, but we can create an attention mask to ignore presence=0. 
        """
        bsz = x_masked.shape[0]
        # x_masked => (bsz,4,3)
        # project -> (bsz,4,d_model)
        x_emb = self.input_proj(x_masked)  # (bsz,4,d_model)

        # let's build an attention key-padding mask where presence=0 => masked out
        # presence is x_masked[:,:,0]
        presence = x_masked[:,:,0]  # shape (bsz,4)
        # True => mask out
        key_padding_mask = (presence < 0.5)  # shape (bsz,4), True if no vertex
        # The nn.Transformer uses shape (bsz, seq_len)
        
        # pass through transformer
        x_enc = self.transformer_encoder(x_emb, src_key_padding_mask=key_padding_mask)
        # x_enc => (bsz,4,d_model)

        # we'll average pool over the unmasked tokens
        # make sure to handle the presence for weighting
        # presence sum => shape (bsz,)
        presence_sum = torch.sum(presence, dim=1, keepdim=True) + 1e-8
        # Weighted average
        x_enc_weighted = x_enc * presence.unsqueeze(-1)  # (bsz,4,d_model)
        shape_embedding = torch.sum(x_enc_weighted, dim=1) / presence_sum  # (bsz,d_model)

        return shape_embedding


class Shape2SpectraTransformerNet(nn.Module):
    """
    Overall model:
      - Transformer aggregator to encode up to 4 vertices into a shape embedding (bsz, d_model).
      - For each of 11 c-values, we do final MLP => 100 reflectance.
    """
    def __init__(self, d_in=3, d_model=128, nhead=4, num_layers=2, 
                 out_dim=100, n_c=11):
        super().__init__()
        self.n_c = n_c
        self.d_model = d_model
        
        self.transformer_agg = TransformerEncoder(d_in=d_in, 
                                                  d_model=d_model,
                                                  nhead=nhead, 
                                                  num_layers=num_layers)
        
        # After aggregator, we get (bsz, d_model) per shape
        # Then for each c in [0..1], we do shape_emb + c => MLP => out_dim=100
        # We'll define a deeper MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 1, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, out_dim)
        )

    def forward(self, shape_array):
        """
        shape_array: (bsz,4,3).  presence= shape_array[:,:,0]
        returns (bsz,11,100)
        """
        device = shape_array.device
        bsz = shape_array.size(0)

        # pass to aggregator
        shape_emb = self.transformer_agg(shape_array)  # (bsz, d_model)

        # for each of the 11 c-values => produce reflectance
        c_vals = torch.linspace(0,1,self.n_c, device=device)
        out_list = []
        for i in range(self.n_c):
            c_i = c_vals[i].view(1,1).expand(bsz,1)  # (bsz,1)
            inp = torch.cat([shape_emb, c_i], dim=1) # (bsz, d_model+1)
            pred_i = self.mlp(inp)                   # (bsz,100)
            out_list.append(pred_i.unsqueeze(1))
        out = torch.cat(out_list, dim=1)  # (bsz,11,100)
        return out

###############################################################################
# 4. Train / Evaluate
###############################################################################
def get_num_verts(shape_np):
    """Count how many vertices have presence>0.5"""
    return int((shape_np[:,0] > 0.5).sum())

def visualize_4_samples(model, ds_test, device, out_dir=".", n_rows=4):
    """
    Same logic as your code: find up to 4 shapes with 1..4 vertices 
    and produce a small figure of shape + GT vs Pred spectra.
    """
    # find up to 4 indices
    found_idx = {}
    for idx in range(len(ds_test)):
        shape_np, spectra_np, uid_ = ds_test[idx]
        n_verts = get_num_verts(shape_np)
        if 1 <= n_verts <= 4 and (n_verts not in found_idx):
            found_idx[n_verts] = idx
        if len(found_idx) == 4:
            break
    
    if len(found_idx) == 0:
        print("No shapes with 1..4 vertices found in test set for visualization.")
        return

    n_rows_plot = len(found_idx)
    fig, axes = plt.subplots(n_rows_plot, 2, figsize=(10, 3*n_rows_plot))
    if n_rows_plot==1:
        axes = [axes]

    row_idx = 0
    for v_count in sorted(found_idx.keys()):
        sample_idx = found_idx[v_count]
        shape_np, spectra_gt, uid_ = ds_test[sample_idx]
        shape_t = torch.tensor(shape_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            spectra_pred = model(shape_t).cpu().numpy()[0]  # (11,100)
        
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

        ax_spec = axes[row_idx][1]
        plot_spectra(ax_spec, spectra_gt, spectra_pred, color_gt='blue', color_pred='red')

        row_idx += 1
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, "4_rows_visualization.png")
    plt.savefig(save_path)
    plt.close()
    print(f"4-sample visualization saved to {save_path}")


def train_shape2spectra_transformer(
    csv_file="merged_s4_shapes_20250114_175110.csv",
    save_dir="outputs",
    num_epochs=1000,
    batch_size=1024,
    lr=1e-4,
    weight_decay=1e-5,
    split_ratio=0.8,
    d_model=256,
    nhead=4,
    num_layers=4
):
    """
    Main training function for the transformer-based aggregator.
    Increase num_epochs or tune hyperparams to get better performance.
    """
    timestamp = f"{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    out_dir = os.path.join(save_dir, f"shape2spectra_transformer_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load Dataset
    ds_full = ShapeToSpectraDataset(csv_file, max_vertices=16)
    print("Total shapes in dataset:", len(ds_full))

    ds_len = len(ds_full)
    train_len = int(ds_len * split_ratio)
    test_len  = ds_len - train_len
    ds_train, ds_test = random_split(ds_full, [train_len, test_len])
    print(f"Train size={len(ds_train)}, Test size={len(ds_test)}")

    if len(ds_train)==0 or len(ds_test)==0:
        raise ValueError("Train or Test set is empty. Check CSV or split_ratio.")

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2) Model
    model = Shape2SpectraTransformerNet(
        d_in=3,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        out_dim=100,
        n_c=11
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=0.5, patience=10, 
                                                           verbose=True)
    criterion = nn.MSELoss(reduction='mean')

    # 3) Training loop
    all_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for (shape_np, spectra_np, uid_list) in train_loader:
            shape  = shape_np.to(device)   # (bsz,4,3)
            target = spectra_np.to(device) # (bsz,11,100)

            pred = model(shape)           # (bsz,11,100)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        all_losses.append(epoch_loss)

        # 4) Validation
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for (shape_np, spectra_np, uid_list) in test_loader:
                shape  = shape_np.to(device)
                target = spectra_np.to(device)
                pred = model(shape)
                bsz_ = shape.size(0)
                vloss = criterion(pred, target) * bsz_
                val_loss_sum += vloss.item()
                val_count += bsz_
        val_mse = val_loss_sum / val_count

        # step scheduler
        scheduler.step(val_mse)

        if (epoch+1) % 20 == 0 or epoch==0:
            print(f"Epoch [{epoch+1}/{num_epochs}] TrainLoss={epoch_loss:.4f} ValMSE={val_mse:.4f}")

    # 5) Plot training curve
    plt.figure()
    plt.plot(all_losses, label="Train MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Transformer Aggregator: Shape->Spectra")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_loss.png"))
    plt.close()

    # 6) Save model
    model_path = os.path.join(out_dir, "shape2spectra_transformer_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 7) Final test MSE
    model.eval()
    test_loss = 0.0
    test_count = 0
    with torch.no_grad():
        for (shape_np, spectra_np, uid_list) in test_loader:
            shape  = shape_np.to(device)
            target = spectra_np.to(device)
            bsz_ = shape.size(0)
            pred = model(shape)
            loss = criterion(pred, target) * bsz_
            test_loss += loss.item()
            test_count += bsz_
    test_mse = test_loss / test_count
    print(f"Final Test MSE: {test_mse:.6f}")

    # 8) Visualize a few shapes
    visualize_4_samples(model, ds_test, device, out_dir=out_dir, n_rows=4)

    # single sample visualize
    sample_idx = np.random.randint(0, len(ds_test))
    shape_np, spectra_np, uid_ = ds_test[sample_idx]
    shape_t = torch.tensor(shape_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        spectra_pred = model(shape_t).cpu().numpy()[0]  # (11,100)

    fig, axs = plt.subplots(1,2, figsize=(12,5))
    # shape
    shape_points = []
    for i in range(shape_np.shape[0]):
        p = shape_np[i,0]
        x = shape_np[i,1]
        y = shape_np[i,2]
        if p>0.5:
            shape_points.append([x,y])
    shape_points = np.array(shape_points)
    axs[0].set_aspect('equal', 'box')
    if shape_points.size>0:
        c4_polygon(axs[0], shape_points, color='green', alpha=0.3, fill=True)
    axs[0].grid(True)
    axs[0].set_title(f"Single Sample (UID={uid_})")

    # spectra
    plot_spectra(axs[1], spectra_np, spectra_pred, color_gt='blue', color_pred='red')

    plt.tight_layout()
    single_vis_path = os.path.join(out_dir, "single_sample_visualization.png")
    plt.savefig(single_vis_path)
    plt.close()
    print(f"Single-sample visualization saved to {single_vis_path} (idx={sample_idx}, uid={uid_})")


if __name__=="__main__":
    train_shape2spectra_transformer(
        # csv_file="merged_s4_shapes_20250114_175110.csv",
        csv_file="merged_s4_shapes_20250119_153038.csv",
        save_dir="outputs",
        num_epochs=100,        # Try a larger number of epochs
        batch_size=64*64,        # Adjust based on your GPU memory
        lr=1e-4,                # Often smaller LR can help with transformers
        weight_decay=1e-5,
        split_ratio=0.8,
        d_model=256,           # Larger hidden dimension
        nhead=4//2,
        num_layers=4//2           # Increase for deeper aggregator
    )

