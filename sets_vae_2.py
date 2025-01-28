import os
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

###############################################################################
# 1. Dataset
###############################################################################

class ShapeDataset(Dataset):
    """
    Reads the CSV, groups by (nQ, shape_idx), extracts 11 x 100 reflectance arrays,
    and parses up to 4 vertices in the first quadrant.
    Optionally, we allow semi-supervised: some fraction of shape labels are masked.
    """
    def __init__(self, csv_file, label_mask_ratio=0.3):
        super().__init__()
        self.df = pd.read_csv(csv_file)

        # Unique ID based on (nQ, shape_idx); adapt if needed
        self.df["unique_id"] = (
            self.df["nQ"].astype(str) + "_" + 
            self.df["shape_idx"].astype(str)
        )
        
        self.r_columns = [col for col in self.df.columns if col.startswith("R@")]
        self.num_wavelengths = len(self.r_columns)  # typically 100
        
        self.grouped_data = []
        for uid, grp in self.df.groupby("unique_id"):
            # Sort by "c". We'll let the set-based model ignore order, but let's keep it consistent.
            grp_sorted = grp.sort_values(by="c")
            if len(grp_sorted) != 11:
                continue
            
            # shape (11, 100)
            R_data = grp_sorted[self.r_columns].values.astype(np.float32)
            vertices_str = grp_sorted["vertices_str"].values[0]
            
            # parse up to 4 Q1 vertices
            raw_pairs = vertices_str.strip().split(";")
            vertices = []
            for pair in raw_pairs:
                xy = pair.split(",")
                if len(xy) == 2:
                    x_val = float(xy[0])
                    y_val = float(xy[1])
                    if x_val > 0 and y_val >= 0:
                        vertices.append((x_val, y_val))
            
            # truncate or zero-pad
            vertices = vertices[:4]
            num_actual = len(vertices)
            while len(vertices) < 4:
                vertices.append((0.0, 0.0))
            presence = [1.0]*num_actual + [0.0]*(4-num_actual)
            
            # flatten => 12
            label_array = []
            for i in range(4):
                label_array.append(presence[i])
                label_array.append(vertices[i][0])
                label_array.append(vertices[i][1])
            label_array = np.array(label_array, dtype=np.float32)
            
            self.grouped_data.append({
                "unique_id": uid,
                "R_data": R_data,
                "label": label_array
            })
        
        # semi-supervised: randomly mask shape labels
        self.data_len = len(self.grouped_data)
        self.has_label_list = np.ones(self.data_len, dtype=bool)
        mask_count = int(self.data_len * label_mask_ratio)
        mask_indices = np.random.choice(self.data_len, mask_count, replace=False)
        for idx in mask_indices:
            self.has_label_list[idx] = False

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        item = self.grouped_data[idx]
        R_data = item["R_data"]          # (11,100)
        label  = item["label"]           # (12,)
        has_label = self.has_label_list[idx]
        uid    = item["unique_id"]
        return R_data, label, has_label, uid


###############################################################################
# 2. Set Transformer Encoder (Permutation-Invariant)
###############################################################################
class SABlock(nn.Module):
    """
    Self-Attention Block from the Set Transformer paper:
      SAB: (X) -> (X + MultiHeadSelfAttention(X)) -> (MLP on each element)
    We'll keep it small for demonstration.
    """
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.fc_mlp = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (bsz, n_points, d_model)
        """
        # Self-attention
        x_att, _ = self.mha(x, x, x)  # (bsz, n_points, d_model)
        x = self.norm1(x + x_att)
        # MLP
        x_mlp = self.fc_mlp(x)        # (bsz, n_points, d_model)
        x = self.norm2(x + x_mlp)
        return x


class PMA(nn.Module):
    """
    Pooling by Multihead Attention.
    We'll pool the entire set into 1 vector via attention to a small learned seed vector.
    This is standard in the Set Transformer approach.
    """
    def __init__(self, d_model=64, n_heads=4, n_seeds=1):
        super().__init__()
        self.n_seeds = n_seeds
        self.seed_params = nn.Parameter(torch.randn(n_seeds, d_model))  # (n_seeds, d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (bsz, n_points, d_model)
        return: (bsz, n_seeds, d_model)
        """
        bsz = x.size(0)
        # repeat seeds for each batch
        # seed shape => (bsz, n_seeds, d_model)
        seeds = self.seed_params.unsqueeze(0).repeat(bsz, 1, 1)
        # attention from seeds to x
        out, _ = self.mha(seeds, x, x)  # out shape: (bsz, n_seeds, d_model)
        out = self.norm(out + seeds)
        return out


class SetTransformerEncoder(nn.Module):
    """
    A simple Set Transformer to handle (bsz, 11, 100) as sets of 11 items,
    each with dimension 100. We'll embed each item into d_model, apply a few SAB blocks,
    then pool it to a single embedding of size d_model via PMA with 1 seed.
    """
    def __init__(self, d_model=64, n_heads=4, num_sab_blocks=2):
        super().__init__()
        self.input_embed = nn.Linear(100, d_model)
        self.sabs = nn.ModuleList([SABlock(d_model, n_heads) for _ in range(num_sab_blocks)])
        self.pma  = PMA(d_model, n_heads, n_seeds=1)

    def forward(self, x):
        """
        x: (bsz, 11, 100)
        return: (bsz, d_model)
        """
        h = self.input_embed(x)  # => (bsz, 11, d_model)
        for sab in self.sabs:
            h = sab(h)  # => (bsz, 11, d_model)
        pooled = self.pma(h)     # => (bsz, 1, d_model)
        # remove the "1" dimension
        pooled = pooled.squeeze(1)  # => (bsz, d_model)
        return pooled


###############################################################################
# 3. VAE Encoder & Decoder
###############################################################################
class VAEEncoder(nn.Module):
    """
    Uses a set transformer to produce a single embedding from (11,100),
    then outputs (mu, logvar).
    """
    def __init__(self, d_model=64, z_dim=16):
        super().__init__()
        self.set_trans = SetTransformerEncoder(d_model=d_model, n_heads=4, num_sab_blocks=2)
        self.mu_fc     = nn.Linear(d_model, z_dim)
        self.logvar_fc = nn.Linear(d_model, z_dim)

    def forward(self, x):
        # x: (bsz, 11, 100)
        h = self.set_trans(x)  # => (bsz, d_model)
        mu = self.mu_fc(h)     
        logvar = self.logvar_fc(h)
        return mu, logvar


class GeometricVertexDecoder(nn.Module):
    """
    Decodes z -> up to 4 vertices, each = (presence, x, y).
    Uses an LSTM over 4 time steps with dummy inputs.
    """
    def __init__(self, z_dim=16, hidden_dim=32, num_vertices=4):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_vertices = num_vertices
        
        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=hidden_dim,
            num_layers=1, 
            batch_first=True
        )
        self.fc_init_h = nn.Linear(z_dim, hidden_dim)
        self.fc_init_c = nn.Linear(z_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 3)  # p, x, y
    
    def forward(self, z):
        """
        z: (bsz, z_dim)
        returns: (bsz, 4, 3)
        """
        bsz = z.size(0)
        device = z.device
        
        h0 = self.fc_init_h(z).unsqueeze(0)  # (1, bsz, hidden_dim)
        c0 = self.fc_init_c(z).unsqueeze(0)
        
        dummy_in = torch.zeros(bsz, self.num_vertices, 1, device=device)
        lstm_out, _ = self.lstm(dummy_in, (h0, c0))  # (bsz,4,hidden_dim)
        
        preds = self.fc_out(lstm_out)  # (bsz,4,3)
        return preds


class RCWARenderer(nn.Module):
    """
    shape_flat => 11 reflectance rows (like rcwa).
    We'll feed shape_flat + c => 100 outputs. c in [0..1].
    """
    def __init__(self, shape_dim=12, hidden_dim=64, out_dim=100, n_c=11):
        super().__init__()
        self.fc1 = nn.Linear(shape_dim+1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.n_c = n_c

    def forward(self, shape_flat):
        device = shape_flat.device
        bsz = shape_flat.size(0)
        c_values = torch.linspace(0,1,self.n_c, device=device)
        
        rows = []
        for i in range(self.n_c):
            cval = c_values[i].view(1,1).repeat(bsz,1)  # (bsz,1)
            inp = torch.cat([shape_flat, cval], dim=1)  # (bsz, 13)
            h = F.relu(self.fc1(inp))
            h = F.relu(self.fc2(h))
            out = self.fc3(h)  # (bsz,100)
            rows.append(out.unsqueeze(1))
        # (bsz, 11, 100)
        return torch.cat(rows, dim=1)


class VAENet(nn.Module):
    def __init__(self, d_model=64, z_dim=16, dec_hidden_dim=32, shape_renderer_dim=64, n_c=11):
        super().__init__()
        self.encoder = VAEEncoder(d_model=d_model, z_dim=z_dim)
        self.shape_decoder = GeometricVertexDecoder(z_dim=z_dim, hidden_dim=dec_hidden_dim, num_vertices=4)
        self.renderer = RCWARenderer(shape_dim=12, hidden_dim=shape_renderer_dim, out_dim=100, n_c=n_c)

    def reparam_sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encoder(x)  # => (bsz,z_dim)
        z = self.reparam_sample(mu, logvar)
        shape_pred = self.shape_decoder(z)  # (bsz,4,3)
        shape_flat = shape_pred.view(shape_pred.size(0), -1)  # (bsz,12)
        recon_spectra = self.renderer(shape_flat)  # (bsz,11,100)
        return shape_pred, recon_spectra, mu, logvar


###############################################################################
# 4. Losses
###############################################################################
def kl_divergence(mu, logvar):
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()

def recon_spectra_loss(pred_spectra, tgt_spectra):
    return F.mse_loss(pred_spectra, tgt_spectra, reduction='mean')

def shape_loss(pred_shape, tgt_shape):
    bsz = pred_shape.size(0)
    pred = pred_shape.view(bsz,4,3)
    tgt  = tgt_shape.view(bsz,4,3)
    p_pred = pred[:,:,0]  # (bsz,4)
    p_tgt  = tgt[:,:,0]
    xy_pred = pred[:,:,1:]
    xy_tgt  = tgt[:,:,1:]
    
    # presence BCE
    presence_bce = F.binary_cross_entropy_with_logits(p_pred, p_tgt, reduction='mean')
    
    # coords MSE only where p_tgt=1
    mask = (p_tgt>0.5).unsqueeze(-1).expand_as(xy_pred)
    if mask.sum() > 0:
        coords_mse = F.mse_loss(xy_pred[mask], xy_tgt[mask], reduction='mean')
    else:
        coords_mse = 0.0
    return presence_bce + coords_mse

def geometric_presence_prior(pred_shape, prior_p=0.3):
    bsz = pred_shape.size(0)
    p_logits = pred_shape[:,:,0]
    p_sig = torch.sigmoid(p_logits)
    
    # E[-log prior] = - p_sig log(prior_p) - (1 - p_sig) log(1-prior_p)
    term1 = - p_sig * np.log(prior_p)
    term2 = - (1 - p_sig) * np.log(1.0 - prior_p)
    penalty = term1 + term2
    return penalty.mean()


###############################################################################
# 5. Visualization (C4 Rotations)
###############################################################################
def visualize_shape_c4(ax, shape_points, color, label):
    """
    shape_points: Nx2 array in first quadrant
    We'll rotate it by 90°, 180°, 270° to replicate in 2nd, 3rd, 4th quadrants.
    """
    # original
    if shape_points.size > 0:
        ax.scatter(shape_points[:,0], shape_points[:,1], c=color, marker='o', label=label)

    # rotate 90 deg: (x,y)->(-y,x)
    rot90 = np.array([[-1,0],[0,1]], dtype=np.float32)
    # Actually, let's do it systematically with a function
    def rotate_90_deg(points):
        # (x,y) -> (-y, x)
        rotated = []
        for p in points:
            x = p[0]
            y = p[1]
            rotated.append(np.array([-y, x], dtype=np.float32))
        return np.stack(rotated, axis=0)

    # apply rotate_90_deg multiple times
    shape_90 = rotate_90_deg(shape_points)
    shape_180 = rotate_90_deg(shape_90)
    shape_270 = rotate_90_deg(shape_180)

    if shape_90.size>0:
        ax.scatter(shape_90[:,0], shape_90[:,1], c=color, marker='x')
    if shape_180.size>0:
        ax.scatter(shape_180[:,0], shape_180[:,1], c=color, marker='^')
    if shape_270.size>0:
        ax.scatter(shape_270[:,0], shape_270[:,1], c=color, marker='s')


def visualize_one_test_sample(model, dataset, device, out_dir=".", sample_idx=None):
    """
    We pick one sample from dataset (random or user-specified),
    pass it through the model => predicted shape => replicate shape in C4 quadrants => 
    compare to GT shape (also C4).
    Then compare spectra (GT vs predicted).
    """
    if sample_idx is None:
        sample_idx = np.random.randint(0, len(dataset))

    R_data_np, label_np, has_label, uid = dataset[sample_idx]
    R_data = torch.tensor(R_data_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1,11,100)
    label  = torch.tensor(label_np,  dtype=torch.float32, device=device).unsqueeze(0)  # (1,12)

    model.eval()
    with torch.no_grad():
        shape_pred, recon_spectra, mu, logvar = model(R_data)

    shape_pred_np = shape_pred.squeeze(0).cpu().numpy()  # (4,3)
    recon_spectra_np = recon_spectra.squeeze(0).cpu().numpy()  # (11,100)
    label_reshaped = label.view(1,4,3).squeeze(0).cpu().numpy()  # (4,3)

    # presence threshold
    p_gt  = label_reshaped[:,0]
    xy_gt = label_reshaped[:,1:]
    p_pred = shape_pred_np[:,0]
    xy_pred = shape_pred_np[:,1:]

    # gather GT points in Q1
    gt_points = []
    for i in range(4):
        if p_gt[i] > 0.5:
            gt_points.append(xy_gt[i])
    gt_points = np.array(gt_points)
    # gather predicted points in Q1
    pred_points = []
    for i in range(4):
        if 1/(1+np.exp(-p_pred[i]))>0.5:
            pred_points.append(xy_pred[i])
    pred_points = np.array(pred_points)

    # plot
    fig, axs = plt.subplots(1, 2, figsize=(14,6))

    # 1) shape with C4
    ax1 = axs[0]
    # GT
    if gt_points.size>0:
        visualize_shape_c4(ax1, gt_points, color='blue', label='GT Q1 + C4')
    # Pred
    if pred_points.size>0:
        visualize_shape_c4(ax1, pred_points, color='red', label='Pred Q1 + C4')

    ax1.set_aspect('equal', 'box')
    ax1.grid(True)
    ax1.set_title(f"Shape (C4) - UID={uid}\nHasLabel={has_label}")
    ax1.legend()

    # 2) spectra
    ax2 = axs[1]
    xvals = np.arange(100)
    for c_idx in range(11):
        ax2.plot(xvals, R_data_np[c_idx,:], 'b--', alpha=0.3)
        ax2.plot(xvals, recon_spectra_np[c_idx,:], 'r-', alpha=0.3)
    ax2.set_xlabel("Wavelength Index")
    ax2.set_ylabel("Reflectance")
    ax2.set_title("Spectra (Blue=GT, Red=Pred)")

    plt.tight_layout()
    save_path = os.path.join(out_dir, "sample_visualization.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Sample visualization saved to {save_path}, index={sample_idx}, uid={uid}")


###############################################################################
# 6. Training
###############################################################################
def train_vae(
    csv_file="merged_s4_shapes_20250114_175110.csv",
    save_dir="outputs",
    num_epochs=100,
    batch_size=512*8,  # real: 4096
    lr=3e-4,
    z_dim=16,
    d_model=64,
    dec_hidden_dim=32,
    shape_renderer_dim=64,
    alpha=1.0,
    beta=1.0,
    gamma=1e-3,
    geo_prior_weight=0.1,
    label_mask_ratio=0.3,
    split_ratio=0.8
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(save_dir, f"vae_train_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # dataset
    full_ds = ShapeDataset(csv_file, label_mask_ratio=label_mask_ratio)
    ds_len = len(full_ds)
    train_len = int(ds_len*split_ratio)
    test_len = ds_len - train_len
    train_ds, test_ds = random_split(full_ds, [train_len, test_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAENet(
        d_model=d_model,
        z_dim=z_dim,
        dec_hidden_dim=dec_hidden_dim,
        shape_renderer_dim=shape_renderer_dim,
        n_c=11
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    all_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for (R_data_np, label_np, has_label_np, uid_list) in train_loader:
            R_data = R_data_np.to(device)   # (bsz, 11, 100)
            label  = label_np.to(device)    # (bsz, 12)
            has_label = has_label_np.to(device)  # bool mask

            shape_pred, recon_spectra, mu, logvar = model(R_data)
            
            # always do recon + kl + geo
            L_spectra = recon_spectra_loss(recon_spectra, R_data)
            L_kl      = kl_divergence(mu, logvar)
            L_geo     = geometric_presence_prior(shape_pred, prior_p=0.3)
            
            # shape loss only for labeled
            if has_label.sum() > 0:
                shape_pred_sub = shape_pred[has_label]
                label_sub      = label[has_label]
                L_shape = shape_loss(shape_pred_sub, label_sub)
            else:
                L_shape = torch.tensor(0.0, device=device)

            loss = alpha*L_shape + beta*L_spectra + gamma*L_kl + geo_prior_weight*L_geo

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        all_losses.append(epoch_loss)

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} "
                  f"(Shape={L_shape.item():.4f}, Spectra={L_spectra.item():.4f}, KL={L_kl.item():.4f}, Geo={L_geo.item():.4f})")

    # finalize
    plt.figure()
    plt.plot(all_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Set-Transformer VAE Training Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_loss.png"))
    plt.close()

    model_path = os.path.join(out_dir, "vae_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluate
    model.eval()
    presence_correct = 0
    presence_total   = 0
    spectra_mse_accum = 0.0
    spectra_count = 0
    with torch.no_grad():
        for (R_data_np, label_np, has_label_np, uid_list) in test_loader:
            R_data = R_data_np.to(device)
            label  = label_np.to(device)
            has_label = has_label_np.to(device)
            
            shape_pred, recon_spectra, mu, logvar = model(R_data)
            
            # presence accuracy only on labeled
            idx_labeled = (has_label==True).nonzero(as_tuple=True)[0]
            if idx_labeled.numel()>0:
                shape_pred_l = shape_pred[idx_labeled]
                label_l      = label[idx_labeled]
                p_pred = shape_pred_l[:,:,0]
                p_tgt  = label_l.view(-1,4,3)[:,:,0]
                p_sig  = torch.sigmoid(p_pred)
                p_bin  = (p_sig>0.5).float()
                presence_correct += (p_bin == p_tgt).sum().item()
                presence_total   += p_tgt.numel()
            
            sqerr = (recon_spectra - R_data)**2
            spectra_mse_accum += sqerr.sum().item()
            spectra_count     += sqerr.numel()

    presence_acc = presence_correct / presence_total if presence_total>0 else 0.0
    spectra_mse  = spectra_mse_accum / (spectra_count if spectra_count>0 else 1)
    print("=== Test Results ===")
    print(f"Presence Accuracy (labeled only): {presence_acc:.4f}")
    print(f"Spectra MSE: {spectra_mse:.6f}")

    # visualize one sample
    visualize_one_test_sample(model, test_ds, device, out_dir=out_dir, sample_idx=None)


###############################################################################
# 7. Main
###############################################################################
if __name__ == "__main__":
    train_vae(
        csv_file="merged_s4_shapes_iccpOv10kG40_seed88888.csv",
        save_dir="outputs",
        num_epochs=100,
        batch_size=512*8,  # 4096
        lr=3e-4,
        z_dim=16,
        d_model=64,
        dec_hidden_dim=32,
        shape_renderer_dim=64,
        alpha=1.0,
        beta=1.0,
        gamma=1e-3,
        geo_prior_weight=0.1,
        label_mask_ratio=0.3,
        split_ratio=0.8
    )

