# shape2spectra.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

from vis import c4_polygon, plot_spectra

###############################################################################
# 1. Dataset
###############################################################################

class ShapeToSpectraDataset(Dataset):
    """
    Groups rows by (NQ, shape_idx).
    Each group must have exactly 11 rows (c=0..1, total 11).
    We'll parse up to 4 vertices => shape = (4,3) with (presence,x,y).
    We'll store 11×100 reflectance as the target.
    """

    def __init__(self, csv_file, max_vertices=4):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.df["uid"] = self.df["NQ"].astype(str) + "_" + self.df["shape_idx"].astype(str)

        self.r_cols = [c for c in self.df.columns if c.startswith("R@")]

        self.data_list = []
        for uid, grp in self.df.groupby("uid"):
            grp_sorted = grp.sort_values(by="c")
            if len(grp_sorted) != 11:
                continue

            v_str = grp_sorted["vertices_str"].values[0].strip()
            raw_pairs = v_str.split(";")
            vertices = []
            for pair in raw_pairs:
                xy = pair.split(",")
                if len(xy)==2:
                    x_val = float(xy[0])
                    y_val = float(xy[1])
                    vertices.append((x_val,y_val))

            # up to 4
            vertices = vertices[:max_vertices]
            num_actual = len(vertices)
            while len(vertices) < max_vertices:
                vertices.append((0.0,0.0))
            presence = [1.0]*num_actual + [0.0]*(max_vertices-num_actual)

            shape_array = []
            for i in range(max_vertices):
                shape_array.append([presence[i], vertices[i][0], vertices[i][1]])
            shape_array = np.array(shape_array, dtype=np.float32)  # (4,3)

            # reflectance => (11,100)
            R_data = grp_sorted[self.r_cols].values.astype(np.float32)

            self.data_list.append({
                "uid": uid,
                "shape": shape_array,  # (4,3)
                "spectra": R_data      # (11,100)
            })

        self.data_len = len(self.data_list)
        if self.data_len == 0:
            raise ValueError("No valid shapes found with exactly 11 rows each. Check CSV or grouping columns.")

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        item = self.data_list[idx]
        return item["shape"], item["spectra"], item["uid"]


###############################################################################
# 2. Improved Model: DeepSets Aggregator
###############################################################################

class Shape2SpectraNet(nn.Module):
    """
    (bsz,4,3) => aggregator => shape embedding => for each c in [0..1], produce 100 reflectance.
    We do a "DeepSets" aggregator:
      1) vertex_embed: (3)->(d_embed=64)
      2) pass each vertex through MLPv => (64->64->64)
      3) sum across vertices => shape (bsz,64)
      4) pass sum through MLPagg => final shape embedding (bsz, hidden_dim=128)
      5) for each c => pass [embedding + c] through big MLP => 100 reflectance
    This often yields better performance than a single LSTM aggregator.
    """

    def __init__(self, d_embed=64, hidden_dim=128, out_dim=100, n_c=11):
        super().__init__()
        self.n_c = n_c

        # Vertex embed
        self.vertex_embed = nn.Linear(3, d_embed)
        # MLP for each vertex
        self.mlp_vertex = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed),
            nn.ReLU()
        )
        # aggregator MLP after sum
        self.mlp_agg = nn.Sequential(
            nn.Linear(d_embed, hidden_dim),
            nn.ReLU()
        )

        # final MLP: from hidden_dim + 1 => 256 => 128 => out_dim=100
        self.fc1 = nn.Linear(hidden_dim + 1, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_dim)

    def forward(self, shape_array):
        """
        shape_array: (bsz,4,3)
        returns: (bsz, 11, 100)
        """
        bsz = shape_array.size(0)
        device = shape_array.device

        # 1) embed each vertex => (bsz,4,d_embed)
        v_emb = self.vertex_embed(shape_array)        # (bsz,4,d_embed)
        v_emb = self.mlp_vertex(v_emb)               # (bsz,4,d_embed)
        # 2) sum across 4 vertices => (bsz,d_embed)
        sum_emb = torch.sum(v_emb, dim=1)            # (bsz,d_embed)
        # 3) aggregator MLP => shape_repr
        shape_repr = self.mlp_agg(sum_emb)           # (bsz,hidden_dim)

        # for each c => pass shape_repr + c => final MLP => row of length=100
        c_vals = torch.linspace(0,1,self.n_c, device=device)
        out_rows = []
        for i in range(self.n_c):
            cval = c_vals[i].unsqueeze(0).repeat(bsz,1)      # (bsz,1)
            x_in = torch.cat([shape_repr, cval], dim=1)      # (bsz, hidden_dim+1)
            x = F.relu(self.fc1(x_in))
            x = F.relu(self.fc2(x))
            row = self.fc3(x)  # (bsz,100)
            out_rows.append(row.unsqueeze(1))
        out_spectra = torch.cat(out_rows, dim=1)  # (bsz,11,100)
        return out_spectra


###############################################################################
# 3. Training + Visualization (4-row figure)
###############################################################################

def get_num_verts(shape_np):
    """Count how many vertices have presence>0.5"""
    count = 0
    for i in range(shape_np.shape[0]):
        if shape_np[i,0] > 0.5:
            count+=1
    return count

def visualize_4_samples(model, ds_test, device, out_dir=".", n_rows=4):
    """
    Create a figure with up to 4 rows, each row corresponds to a shape 
    with exactly 1,2,3,4 vertices if possible. For each row:
      left: shape (C4)
      right: spectra comparison
    If we can't find shapes for all 4 vertex counts, we'll show as many as we can.
    """
    # find up to 4 indices with num_verts=1,2,3,4
    # We'll store them in a dict: {1: idx, 2: idx, 3: idx, 4: idx}
    found_idx = {}
    for idx in range(len(ds_test)):
        shape_np, spectra_np, uid_ = ds_test[idx]
        n_verts = get_num_verts(shape_np)
        if n_verts>=1 and n_verts<=4 and (n_verts not in found_idx):
            found_idx[n_verts] = idx
        if len(found_idx)==4:
            break

    if len(found_idx)==0:
        print("No shapes with 1..4 vertices found in test set.")
        return

    # create figure with up to 4 rows, each row= (1) shape (2) spectra
    n_rows_plot = len(found_idx)
    fig, axes = plt.subplots(n_rows_plot, 2, figsize=(10, 3*n_rows_plot))

    # if there's only 1 row, axes might not be a 2D array
    if n_rows_plot==1:
        axes = [axes]

    row_idx = 0
    for v_count in sorted(found_idx.keys()):
        sample_idx = found_idx[v_count]
        shape_np, spectra_gt, uid_ = ds_test[sample_idx]
        shape_t = torch.tensor(shape_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            spectra_pred = model(shape_t).cpu().numpy()[0] # (11,100)

        # shape
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
        if shape_points.size>0:
            c4_polygon(ax_shape, shape_points, color='green', alpha=0.3, fill=True)
        ax_shape.grid(True)
        ax_shape.set_title(f"UID={uid_}, #Vert={v_count}")

        # spectra
        ax_spec = axes[row_idx][1]
        plot_spectra(ax_spec, spectra_gt, spectra_pred, color_gt='blue', color_pred='red')

        row_idx += 1

    plt.tight_layout()
    save_path = os.path.join(out_dir, "4_rows_visualization.png")
    plt.savefig(save_path)
    plt.close()
    print(f"4-sample visualization saved to {save_path}")


def train_shape2spectra(
    csv_file="merged_s4_shapes_20250114_175110.csv",
    save_dir="outputs",
    num_epochs=500,
    batch_size=4096,   # large for big GPU
    lr=1e-3,
    split_ratio=0.8
):
    timestamp = f"{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    out_dir = os.path.join(save_dir, f"shape2spectra_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # dataset
    ds_full = ShapeToSpectraDataset(csv_file, max_vertices=4)
    print("Total shapes in dataset:", len(ds_full))

    ds_len = len(ds_full)
    train_len = int(ds_len*split_ratio)
    test_len  = ds_len - train_len
    ds_train, ds_test = random_split(ds_full, [train_len, test_len])
    if len(ds_train)==0 or len(ds_test)==0:
        raise ValueError("Train or Test set is empty. Check your CSV or split_ratio.")

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # model
    model = Shape2SpectraNet(
        d_embed=64,      # bigger embed
        hidden_dim=128*8,  # bigger aggregator dimension
        out_dim=100,
        n_c=11
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')

    all_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for (shape_np, spectra_np, uid_list) in train_loader:
            shape  = shape_np.to(device)   # (bsz,4,3)
            target = spectra_np.to(device) # (bsz,11,100)

            pred = model(shape)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        all_losses.append(epoch_loss)

        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.5f}")

    # Plot training curve
    plt.figure()
    plt.plot(all_losses, label="Train MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("DeepSets Shape->Spectra Training")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_loss.png"))
    plt.close()

    # Save model
    model_path = os.path.join(out_dir, "shape2spectra_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluate on test
    model.eval()
    test_loss = 0.0
    test_count= 0
    with torch.no_grad():
        for (shape_np, spectra_np, uid_list) in test_loader:
            shape  = shape_np.to(device)
            target = spectra_np.to(device)
            pred = model(shape)
            loss = criterion(pred, target)* shape.size(0)
            test_loss += loss.item()
            test_count+= shape.size(0)
    test_mse = test_loss / test_count
    print(f"Test MSE: {test_mse:.6f}")

    # Visualization: 4 rows
    visualize_4_samples(model, ds_test, device, out_dir=out_dir, n_rows=4)

    # Also do the single-sample old style example if you want
    sample_idx = np.random.randint(0, len(ds_test))
    shape_np, spectra_np, uid_ = ds_test[sample_idx]
    shape_t = torch.tensor(shape_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        spectra_pred = model(shape_t).cpu().numpy()[0]  # (11,100)

    # single figure
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


###############################################################################
# 4. Main
###############################################################################

if __name__=="__main__":
    train_shape2spectra(
        csv_file="merged_s4_shapes_20250114_175110.csv",
        save_dir="outputs",
        num_epochs=5000,
        batch_size=4096,   # large batch for your big GPU
        lr=1e-3,
        split_ratio=0.8
    )

