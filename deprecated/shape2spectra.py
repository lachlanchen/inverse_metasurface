# shape2spectra.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

# We'll import from vis.py
from vis import c4_polygon, plot_spectra

###############################################################################
# 1. Dataset
###############################################################################

class ShapeToSpectraDataset(Dataset):
    """
    - Groups rows by (folder_key, shape_idx).
    - Each group must have 11 rows (c=0..1 stepping).
    - For each shape, parse 'vertices_str' => variable-length vertices in Q1.
    - We store 11×100 reflectance as target.

    We'll define a max_vertices=6 (for example). If a shape has more, we truncate.
    If fewer, we pad with (0.0,0.0) and presence=0. We'll store presence in dimension 0.
    So each vertex => (presence, x, y). shape => (max_vertices, 3).
    """
    def __init__(self, csv_file, max_vertices=6):
        super().__init__()
        self.df = pd.read_csv(csv_file)

        # Let's define a unique ID for grouping
        self.df["uid"] = self.df["NQ"].astype(str) + "_" + self.df["shape_idx"].astype(str)
        # reflectance columns
        self.r_cols = [c for c in self.df.columns if c.startswith("R@")]
        # group
        self.data_list = []
        for uid, grp in self.df.groupby("uid"):
            grp_sorted = grp.sort_values(by="c")
            if len(grp_sorted) != 11:
                continue
            # parse shape from first row (assuming all rows have same 'vertices_str')
            v_str = grp_sorted["vertices_str"].values[0]
            vertices = []
            for pair in v_str.strip().split(";"):
                xy = pair.split(",")
                if len(xy)==2:
                    x_val = float(xy[0])
                    y_val = float(xy[1])
                    # we won't filter quadrant here if user wants full shapes,
                    # but typically we do Q1 => x>0,y>=0
                    vertices.append((x_val, y_val))

            # truncate or pad to max_vertices
            vertices = vertices[:max_vertices]
            num_actual = len(vertices)
            while len(vertices)<max_vertices:
                vertices.append((0.0,0.0))
            # presence
            presence = [1.0]*num_actual + [0.0]*(max_vertices-num_actual)

            # build shape_array => (max_vertices, 3)
            shape_array = []
            for i in range(max_vertices):
                shape_array.append([
                    presence[i],
                    vertices[i][0],
                    vertices[i][1]
                ])
            shape_array = np.array(shape_array, dtype=np.float32)  # (max_vertices,3)

            # gather reflectance => (11, 100)
            R_data = grp_sorted[self.r_cols].values.astype(np.float32)
            # store
            self.data_list.append({
                "uid": uid,
                "shape_array": shape_array,  # (max_vertices,3)
                "spectra": R_data            # (11,100)
            })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        shape = item["shape_array"]   # (max_vertices,3)
        spectra = item["spectra"]     # (11,100)
        uid = item["uid"]
        return shape, spectra, uid


###############################################################################
# 2. Model: shape->spectra
###############################################################################
class Shape2SpectraNet(nn.Module):
    """
    We accept up to 'max_vertices' in shape: (max_vertices,3).
    We do:
      1) Vertex-level embedding => some aggregator => shape embedding
      2) Then for each c in [0..1] (11 steps), produce a 100-length reflectance row.

    The aggregator can be a small Transformer, LSTM, or MLP with sum-pooling. 
    Below is a simple LSTM aggregator for demonstration.
    """
    def __init__(self, max_vertices=6, d_embed=32, hidden_dim=64, out_dim=100, n_c=11):
        super().__init__()
        self.max_vertices = max_vertices
        self.n_c = n_c

        # (3)->(d_embed)
        self.vertex_embed = nn.Linear(3, d_embed)

        # aggregator
        self.agg_lstm = nn.LSTM(input_size=d_embed, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
        # final MLP from aggregator hidden_dim + 1(c) => out_dim(=100)
        self.fc1 = nn.Linear(hidden_dim+1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, shape_array):
        """
        shape_array: (bsz, max_vertices, 3)
        returns: (bsz, n_c, 100)
        """
        bsz = shape_array.size(0)
        device = shape_array.device

        # embed each vertex
        v_emb = self.vertex_embed(shape_array)  # (bsz, max_vertices, d_embed)

        # aggregator => we take the final hidden state
        _, (h_n, c_n) = self.agg_lstm(v_emb)  # h_n: (1, bsz, hidden_dim)
        shape_repr = h_n.squeeze(0)  # (bsz, hidden_dim)

        # For each c in [0..1], produce row
        c_vals = torch.linspace(0,1,self.n_c, device=device)  # (n_c,)
        out_rows = []
        for i in range(self.n_c):
            cval = c_vals[i].view(1,1).repeat(bsz,1)  # (bsz,1)
            x_in = torch.cat([shape_repr, cval], dim=1)  # (bsz, hidden_dim+1)
            x = F.relu(self.fc1(x_in))
            x = F.relu(self.fc2(x))
            row = self.fc3(x)  # (bsz, out_dim=100)
            out_rows.append(row.unsqueeze(1))

        out_spectra = torch.cat(out_rows, dim=1)  # (bsz, n_c, 100)
        return out_spectra


###############################################################################
# 3. Training
###############################################################################
def train_shape2spectra(
    csv_file="merged_s4_shapes.csv",
    save_dir="outputs",
    max_vertices=6,
    num_epochs=50,
    batch_size=64,
    lr=1e-3,
    split_ratio=0.8
):
    """
    We'll train a shape->spectra model by MSE on 11x100 reflectance.
    """
    timestamp = f"{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    out_dir = os.path.join(save_dir, f"shape2spectra_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # dataset
    ds_full = ShapeToSpectraDataset(csv_file, max_vertices=max_vertices)
    ds_len = len(ds_full)
    train_len = int(ds_len*split_ratio)
    test_len  = ds_len - train_len
    ds_train, ds_test = random_split(ds_full, [train_len, test_len])
    
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    model = Shape2SpectraNet(
        max_vertices=max_vertices,
        d_embed=32,
        hidden_dim=64,
        out_dim=100,
        n_c=11
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')

    # training loop
    all_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for (shape_np, spectra_np, uid_list) in train_loader:
            shape  = shape_np.to(device)   # (bsz, max_vertices, 3)
            target = spectra_np.to(device) # (bsz, 11, 100)

            pred = model(shape)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        all_losses.append(epoch_loss)

        if (epoch+1)%5==0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.5f}")

    # plot training curve
    plt.figure()
    plt.plot(all_losses, label="Train MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Shape->Spectra Training")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_loss.png"))
    plt.close()

    # save model
    model_path = os.path.join(out_dir, "shape2spectra_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # test evaluation
    model.eval()
    test_loss = 0.0
    test_count= 0
    with torch.no_grad():
        for (shape_np, spectra_np, uid_list) in test_loader:
            shape  = shape_np.to(device)
            target = spectra_np.to(device)
            pred = model(shape)
            loss = criterion(pred, target)* shape.size(0) # sum
            test_loss += loss.item()
            test_count+= shape.size(0)

    test_mse = test_loss/test_count if test_count>0 else 0.0
    print(f"Test MSE: {test_mse:.6f}")

    # quick visualization
    if len(ds_test)>0:
        sample_idx = np.random.randint(0, len(ds_test))
        shape_np, spectra_np, uid_ = ds_test[sample_idx]
        shape_t = torch.tensor(shape_np, dtype=torch.float32, device=device).unsqueeze(0)
        spectra_gt = spectra_np
        with torch.no_grad():
            spectra_pred = model(shape_t).cpu().numpy()[0]  # (11,100)
        # plot
        # 1) shape
        fig, axs = plt.subplots(1,2, figsize=(12,5))

        # parse shape (presence, x, y)
        # gather points with presence>0.5
        shape_points = []
        for i in range(shape_np.shape[0]):
            p = shape_np[i,0]
            x = shape_np[i,1]
            y = shape_np[i,2]
            if p>0.5:
                shape_points.append([x,y])
        shape_points = np.array(shape_points)
        axs[0].set_aspect('equal', 'box')
        c4_polygon(axs[0], shape_points, color='green', alpha=0.3, fill=True)
        axs[0].set_title(f"Shape (C4) - UID={uid_}")
        axs[0].grid(True)

        # 2) spectra
        plot_spectra(axs[1], spectra_gt, spectra_pred, color_gt='blue', color_pred='red')

        plt.tight_layout()
        vis_path = os.path.join(out_dir, "sample_visualization.png")
        plt.savefig(vis_path)
        plt.close()
        print(f"Sample visualization saved to {vis_path} (idx={sample_idx}, uid={uid_})")


###############################################################################
# 4. Main
###############################################################################

if __name__=="__main__":
    train_shape2spectra(
        csv_file="merged_s4_shapes_20250114_175110.csv",  # adapt to your CSV
        save_dir="outputs",
        max_vertices=4,
        num_epochs=500,
        batch_size=64*64,
        lr=1e-3,
        split_ratio=0.8
    )

