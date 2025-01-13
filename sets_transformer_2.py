import os
import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

################################################################################
# 1. Data Reading and Preprocessing
################################################################################

class ShapeDataset(Dataset):
    """
    Reads the CSV, groups by (folder_key, shape_idx),
    extracts 11 x 100 reflectance arrays, and parses up to 4 vertices in the first quadrant.
    """
    def __init__(self, csv_file="merged_s4_shapes.csv"):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        
        # Combine folder_key and shape_idx to identify each shape uniquely
        self.df["unique_id"] = self.df["folder_key"].astype(str) + "_" + self.df["shape_idx"].astype(str)
        
        # Identify columns that correspond to reflectances R@...
        self.r_columns = [col for col in self.df.columns if col.startswith("R@")]
        self.num_wavelengths = len(self.r_columns)  # typically 100 in your example
        
        self.grouped_data = []
        for uid, grp in self.df.groupby("unique_id"):
            # Sort by "c" if we want consistent ordering of the 11 spectra
            grp_sorted = grp.sort_values(by="c")
            
            # We expect exactly 11 rows (c=0.0 to 1.0 in increments of 0.1).
            if len(grp_sorted) != 11:
                continue
            
            # Build (11, 100) array
            R_data = grp_sorted[self.r_columns].values.astype(np.float32)  # shape (11, 100)
            
            # The polygon string (assumed identical for all 11 rows)
            vertices_str = grp_sorted["vertices_str"].values[0]
            
            # Parse polygon: "x,y;x,y;x,y;..."
            raw_pairs = vertices_str.strip().split(";")
            vertices = []
            for pair in raw_pairs:
                xy = pair.split(",")
                if len(xy) == 2:
                    x_val = float(xy[0])
                    y_val = float(xy[1])
                    # Keep if x>0, y>=0 (first quadrant)
                    if x_val > 0 and y_val >= 0:
                        vertices.append((x_val, y_val))
            
            # Keep up to 4 vertices
            vertices = vertices[:4]
            num_actual = len(vertices)
            while len(vertices) < 4:
                vertices.append((0.0, 0.0))
            
            # presence bits
            presence = [1.0]*num_actual + [0.0]*(4 - num_actual)
            
            # Flatten to (12,) = [p1,x1,y1, p2,x2,y2, p3,x3,y3, p4,x4,y4]
            label_array = []
            for i in range(4):
                label_array.append(presence[i])
                label_array.append(vertices[i][0])
                label_array.append(vertices[i][1])
            
            label_array = np.array(label_array, dtype=np.float32)  # shape (12,)
            
            self.grouped_data.append({
                "unique_id": uid,
                "R_data": R_data,    # (11, 100) float32
                "label": label_array # (12,) float32
            })
        
        self.data_len = len(self.grouped_data)
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        item = self.grouped_data[idx]
        # We'll return numpy arrays here,
        # but typically we want to convert to torch.Tensor in the DataLoader's collate_fn or inside the training loop.
        return item["R_data"], item["label"], item["unique_id"]

################################################################################
# 2. Model Definition
################################################################################

class PhiNetwork(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=64, embed_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, x):
        # x: (batch_size, 11, 100)
        bsz, n_spectra, d_in = x.shape
        x_flat = x.view(bsz*n_spectra, d_in)  # (bsz*11, 100)
        
        out = F.relu(self.fc1(x_flat))
        out = self.fc2(out)  # (bsz*11, embed_dim)
        
        out = out.view(bsz, n_spectra, -1)  # (bsz, 11, embed_dim)
        return out

class RhoNetwork(nn.Module):
    def __init__(self, embed_dim=32, hidden_dim=64, out_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        # x: (batch_size, embed_dim)
        out = F.relu(self.fc1(x))
        out = self.fc2(out)  # (batch_size, out_dim)
        return out

class DeepSetsEncoder(nn.Module):
    """
    Sum aggregator for the 11 embeddings -> out_dim
    """
    def __init__(self, phi_network, rho_network, aggregator="sum"):
        super().__init__()
        self.phi = phi_network
        self.rho = rho_network
        self.aggregator = aggregator
    
    def forward(self, x):
        # x: (batch_size, 11, 100)
        phi_out = self.phi(x)  # (batch_size, 11, embed_dim)
        if self.aggregator == "sum":
            agg = torch.sum(phi_out, dim=1)
        elif self.aggregator == "mean":
            agg = torch.mean(phi_out, dim=1)
        else:
            raise ValueError("Unsupported aggregator.")
        out = self.rho(agg)  # (batch_size, out_dim)
        return out

class LSTMDecoder(nn.Module):
    """
    Decodes the final embedding into up to 4 vertices (presence, x, y).
    We'll feed 4 time steps of dummy input (zeros), each step outputs 3 features.
    We keep input_size=1 to avoid cuDNN issues with input_size=0.
    """
    def __init__(self, input_dim=32, hidden_dim=32, num_vertices=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_vertices = num_vertices
        
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, 
                            num_layers=1, batch_first=True)
        
        # This linear transforms the LSTM hidden state -> (presence, x, y)
        self.fc_out = nn.Linear(hidden_dim, 3)
        
        # We'll have a small linear for h0, c0
        self.fc_init_h = nn.Linear(input_dim, hidden_dim)
        self.fc_init_c = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, encoding):
        """
        encoding: (batch_size, input_dim) from the aggregator
        returns: (batch_size, 4, 3)
        """
        bsz = encoding.size(0)
        device = encoding.device
        
        # Build initial states (1, bsz, hidden_dim)
        h0 = self.fc_init_h(encoding).unsqueeze(0)
        c0 = self.fc_init_c(encoding).unsqueeze(0)
        
        # dummy input to feed LSTM: shape (bsz, 4, 1)
        x_in = torch.zeros(bsz, self.num_vertices, 1, device=device)
        
        lstm_out, (hn, cn) = self.lstm(x_in, (h0, c0))
        # lstm_out: (bsz, 4, hidden_dim)
        
        # project each of the 4 states to 3 values
        preds = self.fc_out(lstm_out)  # (bsz, 4, 3)
        return preds

class FullModel(nn.Module):
    def __init__(self, input_dim=100, embed_dim=32, ds_hidden=64, dec_hidden=32):
        super().__init__()
        phi = PhiNetwork(input_dim, ds_hidden, embed_dim)
        rho = RhoNetwork(embed_dim, ds_hidden, embed_dim)
        self.encoder = DeepSetsEncoder(phi, rho, aggregator="sum")
        self.decoder = LSTMDecoder(input_dim=embed_dim, hidden_dim=dec_hidden, num_vertices=4)
    
    def forward(self, x):
        # x: (batch_size, 11, 100)
        encoding = self.encoder(x)    # (batch_size, embed_dim)
        preds = self.decoder(encoding)  # (batch_size, 4, 3)
        return preds

################################################################################
# 3. Utility Functions: Loss, Accuracy, C4 Symmetry, Saving Polygons
################################################################################

def custom_loss(preds, targets):
    """
    preds: (bsz, 4, 3) = [p, x, y]
    targets: (bsz, 12) -> reshape -> (bsz, 4, 3)
    """
    bsz = preds.shape[0]
    preds_reshape = preds.view(bsz, 4, 3)
    targets_reshape = targets.view(bsz, 4, 3)
    
    p_pred = preds_reshape[:, :, 0]  # (bsz, 4)
    p_tgt  = targets_reshape[:, :, 0]
    
    xy_pred = preds_reshape[:, :, 1:]  # (bsz, 4, 2)
    xy_tgt  = targets_reshape[:, :, 1:]
    
    # presence -> BCE with logits
    presence_loss = F.binary_cross_entropy_with_logits(
        p_pred, p_tgt, reduction='mean'
    )
    
    # coords -> MSE, masked by presence
    mask = (p_tgt > 0.5).unsqueeze(-1).expand_as(xy_pred)  # same shape as xy_pred
    if mask.sum() > 0:
        coords_loss = F.mse_loss(xy_pred[mask], xy_tgt[mask], reduction='mean')
    else:
        coords_loss = 0.0
    
    return presence_loss + coords_loss

def evaluate_model(model, dataloader, device, out_dir):
    """
    Evaluate on the given dataloader, compute presence accuracy, coordinate MSE,
    and also save each predicted polygon (including C4 symmetry) into out_dir/polygons.
    """
    model.eval()
    
    presence_correct = 0
    presence_total = 0
    xy_loss_accum = 0.0
    xy_count = 0
    
    # Folder to save polygons
    poly_dir = os.path.join(out_dir, "predicted_polygons")
    os.makedirs(poly_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (R_data_np, label_np, uid) in enumerate(dataloader):
            # R_data_np: (batch_size, 11, 100) in numpy
            # Convert to tensor
            R_data = torch.tensor(R_data_np, dtype=torch.float32, device=device)
            label  = torch.tensor(label_np,  dtype=torch.float32, device=device)
            
            # Forward
            preds = model(R_data)  # (batch_size, 4, 3)
            
            # Separate presence vs coords
            p_pred = preds[:,:,0]
            xy_pred = preds[:,:,1:]
            
            # GT
            label_reshaped = label.view(-1, 4, 3)
            p_tgt = label_reshaped[:,:,0]
            xy_tgt = label_reshaped[:,:,1:]
            
            # --------- 1) Presence Accuracy ---------
            p_sig = torch.sigmoid(p_pred)              # (bsz, 4)
            p_bin = (p_sig > 0.5).float()               # threshold
            presence_correct += (p_bin == p_tgt).sum().item()
            presence_total   += p_tgt.numel()
            
            # --------- 2) Coordinate MSE (masked) ---------
            mask = (p_tgt > 0.5).unsqueeze(-1).expand_as(xy_pred)
            if mask.sum() > 0:
                # sum of squared error
                sqerr = (xy_pred[mask] - xy_tgt[mask])**2
                xy_loss_accum += sqerr.sum().item()
                xy_count += mask.sum().item()
            
            # --------- 3) Save polygons (with C4 symmetry) ---------
            # We'll do this sample-by-sample
            bsz = preds.size(0)
            for i in range(bsz):
                sample_uid = uid[i]
                # Reconstruct the first-quadrant polygon
                # presence bits
                pvals = p_sig[i].cpu().numpy()
                xyvals = xy_pred[i].cpu().numpy()
                
                # Gather the predicted vertices in Q1
                # We only keep the ones with presence>0.5
                q1_points = []
                for v in range(4):
                    if pvals[v] > 0.5:
                        xval = xyvals[v,0]
                        yval = xyvals[v,1]
                        q1_points.append((xval, yval))
                
                # Now replicate with c4 symmetry:
                # A rotation by +90° around origin: (x, y) -> (-y, x)
                # +180°: (x, y) -> (-x, -y)
                # +270°: (x, y) -> (y, -x)
                # We'll combine them all into a single list
                full_polygon = []
                
                for (xval, yval) in q1_points:
                    # Q1
                    full_polygon.append((xval, yval))
                    # Q2
                    full_polygon.append((-yval, xval))
                    # Q3
                    full_polygon.append((-xval, -yval))
                    # Q4
                    full_polygon.append((yval, -xval))
                
                # Save to a text file or CSV
                out_poly_path = os.path.join(poly_dir, f"{sample_uid}_pred.csv")
                # We'll just append for each batch item or overwrite if we want one file per shape
                with open(out_poly_path, "w") as f:
                    f.write("x,y\n")
                    for (px, py) in full_polygon:
                        f.write(f"{px:.6f},{py:.6f}\n")
    
    presence_acc = presence_correct / (presence_total if presence_total>0 else 1)
    coord_mse = xy_loss_accum / (xy_count if xy_count>0 else 1)
    return presence_acc, coord_mse

################################################################################
# 4. Training Script
################################################################################

def train_model(csv_file="merged_s4_shapes.csv", 
                save_dir="outputs",
                num_epochs=100,
                batch_size=8,
                lr=1e-3,
                split_ratio=0.8):
    # Create output dir with datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(save_dir, f"train_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    # 4.1. Build dataset and split
    full_dataset = ShapeDataset(csv_file=csv_file)
    total_len = len(full_dataset)
    train_len = int(split_ratio*total_len)
    test_len  = total_len - train_len
    
    train_dataset, test_dataset = random_split(full_dataset, [train_len, test_len])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=False)
    
    # 4.2. Model
    model = FullModel(input_dim=100, embed_dim=32, ds_hidden=64, dec_hidden=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 4.3. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4.4. Training loop
    all_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (R_data_np, label_np, uid) in enumerate(train_loader):
            # Convert numpy -> torch tensors
            R_data = torch.tensor(R_data_np, dtype=torch.float32, device=device) # (bsz, 11, 100)
            label  = torch.tensor(label_np,  dtype=torch.float32, device=device) # (bsz, 12)
            
            optimizer.zero_grad()
            preds = model(R_data)        # (bsz, 4, 3)
            loss = custom_loss(preds, label)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        all_losses.append(epoch_loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")
    
    # 4.5. Plot training curve
    plt.figure(figsize=(6,4))
    plt.plot(range(num_epochs), all_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")
    plt.savefig(os.path.join(out_dir, "training_loss.png"))
    plt.close()
    
    # 4.6. Save the model
    model_path = os.path.join(out_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 4.7. Evaluate on test set: presence accuracy & coordinate MSE
    presence_acc, coord_mse = evaluate_model(model, test_loader, device, out_dir)
    print("=== Test Set Results ===")
    print(f"Presence Accuracy: {presence_acc:.4f}")
    print(f"Coordinate MSE:    {coord_mse:.6f}")

################################################################################
# 5. Main
################################################################################

if __name__ == "__main__":
    train_model(
        csv_file="merged_s4_shapes.csv",
        save_dir="outputs",
        num_epochs=100,
        batch_size=8,
        lr=1e-3,
        split_ratio=0.8
    )

