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
    Reads the CSV, groups by (folder_key, shape_idx) or (NQ, shape_idx),
    extracts 11 x 100 reflectance arrays, and parses up to 4 vertices in Q1.
    Optionally, we allow semi-supervised: some fraction of shape labels are masked.
    """
    def __init__(self, csv_file, label_mask_ratio=0.3):
        """
        :param csv_file: path to the CSV
        :param label_mask_ratio: fraction of samples that will have no shape labels
        """
        super().__init__()
        self.df = pd.read_csv(csv_file)

        # Make unique ID
        # Adjust as needed if you prefer grouping by (folder_key, shape_idx).
        self.df["unique_id"] = (
            self.df["NQ"].astype(str) + "_" + 
            self.df["shape_idx"].astype(str)
        )
        
        self.r_columns = [col for col in self.df.columns if col.startswith("R@")]
        self.num_wavelengths = len(self.r_columns)  # typically 100
        
        self.grouped_data = []
        for uid, grp in self.df.groupby("unique_id"):
            grp_sorted = grp.sort_values(by="c")
            # If we don't have exactly 11 rows, skip
            if len(grp_sorted) != 11:
                continue
            
            R_data = grp_sorted[self.r_columns].values.astype(np.float32)  # (11,100)
            vertices_str = grp_sorted["vertices_str"].values[0]

            # Parse up to 4 Q1 vertices
            raw_pairs = vertices_str.strip().split(";")
            vertices = []
            for pair in raw_pairs:
                xy = pair.split(",")
                if len(xy) == 2:
                    x_val = float(xy[0])
                    y_val = float(xy[1])
                    # keep only if x>0, y>=0 (first quadrant)
                    if x_val > 0 and y_val >= 0:
                        vertices.append((x_val, y_val))
            
            # Truncate or zero-pad to 4
            vertices = vertices[:4]
            num_actual = len(vertices)
            while len(vertices) < 4:
                vertices.append((0.0, 0.0))
            presence = [1.0]*num_actual + [0.0]*(4-num_actual)
            
            # Flatten (4 * 3 = 12)
            label_array = []
            for i in range(4):
                label_array.append(presence[i])
                label_array.append(vertices[i][0])
                label_array.append(vertices[i][1])
            label_array = np.array(label_array, dtype=np.float32)
            
            self.grouped_data.append({
                "unique_id": uid,
                "R_data": R_data,          # shape (11,100)
                "label": label_array,      # shape (12,)
            })

        # For semi-supervision: randomly mask shape labels for label_mask_ratio
        # We'll store "has_label" as a boolean for each sample
        self.data_len = len(self.grouped_data)
        self.has_label_list = np.ones(self.data_len, dtype=bool)
        # randomly choose a subset to mask
        mask_count = int(self.data_len * label_mask_ratio)
        mask_indices = np.random.choice(self.data_len, mask_count, replace=False)
        for idx in mask_indices:
            self.has_label_list[idx] = False

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        item = self.grouped_data[idx]
        R_data = item["R_data"]
        label  = item["label"]
        has_label = self.has_label_list[idx]
        uid    = item["unique_id"]
        return R_data, label, has_label, uid


###############################################################################
# 2. Order-Invariant Encoder via Transformer
###############################################################################
class TransformerEncoder(nn.Module):
    """
    A small Transformer-based encoder that reads (11,100) and outputs a single embedding.
    This helps the model be robust to permutations of the 11 reflectance rows.

    We'll do:
      1) Linear embed each row from 100 -> d_model
      2) Add pos. encoding or skip it for actual order invariance
      3) TransformerEncoder
      4) Pool (mean) across the 11 tokens
      5) Final MLP -> (embed_dim)
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.d_model = d_model
        
        self.row_embed = nn.Linear(100, d_model)  # each reflectance row => d_model
        # We won't add a positional encoding so it remains order-agnostic or at least order-independent.

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.final_fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x shape: (batch_size, 11, 100)
        return: (batch_size, d_model)
        """
        # embed each row => shape (batch_size, 11, d_model)
        h = self.row_embed(x)
        # pass through transformer encoder
        h_enc = self.transformer_enc(h)  # (bsz, 11, d_model)
        # pool
        h_mean = torch.mean(h_enc, dim=1)  # shape (bsz, d_model)
        # final
        out = F.relu(self.final_fc(h_mean))
        return out


class VAEEncoder(nn.Module):
    """
    Full encoder: use the Transformer to get an embedding, then produce (mu, logvar).
    """
    def __init__(self, d_model=64, z_dim=16):
        super().__init__()
        self.transformer = TransformerEncoder(
            d_model=d_model, 
            nhead=4, 
            num_layers=2, 
            dim_feedforward=128
        )
        self.mu_fc     = nn.Linear(d_model, z_dim)
        self.logvar_fc = nn.Linear(d_model, z_dim)
    
    def forward(self, x):
        """
        x: (batch_size, 11, 100)
        """
        h = self.transformer(x)  # => (bsz, d_model)
        mu = self.mu_fc(h)       # => (bsz, z_dim)
        logvar = self.logvar_fc(h)
        return mu, logvar


###############################################################################
# 3. Shape Decoder & RCWA-Like Renderer
###############################################################################
class GeometricVertexDecoder(nn.Module):
    """
    A small LSTM-like or GRU-like decoder that sequentially decodes up to 4 vertices.
    We also incorporate a geometric prior that penalizes presence=1 too often.
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
        c0 = self.fc_init_c(z).unsqueeze(0)  # (1, bsz, hidden_dim)
        
        # dummy input: shape (bsz, 4, 1)
        # The decoder sees 4 steps => 4 possible vertices
        dummy_input = torch.zeros(bsz, self.num_vertices, 1, device=device)
        lstm_out, _ = self.lstm(dummy_input, (h0, c0))  # (bsz, 4, hidden_dim)
        
        preds = self.fc_out(lstm_out)  # (bsz, 4, 3)
        return preds


class RCWARenderer(nn.Module):
    """
    A neural 'renderer' that takes a shape (4,3) => (12) plus c in [0..1]
    and outputs a predicted reflectance row (length=100).
    We'll do 11 steps of c from 0..1.
    """
    def __init__(self, shape_dim=12, hidden_dim=64, out_dim=100, n_c=11):
        super().__init__()
        self.fc1 = nn.Linear(shape_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.n_c = n_c
    
    def forward(self, shape_flat):
        """
        shape_flat: (bsz, 12)
        returns: (bsz, n_c, 100)
        """
        device = shape_flat.device
        bsz = shape_flat.size(0)
        
        c_values = torch.linspace(0, 1, self.n_c, device=device)  # (n_c,)
        
        outputs = []
        for i in range(self.n_c):
            cval = c_values[i].view(1,1).repeat(bsz,1)
            inp = torch.cat([shape_flat, cval], dim=1)  # (bsz, 13)
            h = F.relu(self.fc1(inp))
            h = F.relu(self.fc2(h))
            row = self.fc3(h)  # (bsz, 100)
            outputs.append(row.unsqueeze(1))
        out_spectra = torch.cat(outputs, dim=1)  # (bsz, n_c, 100)
        return out_spectra


class VAENet(nn.Module):
    """
    Overall VAE-like pipeline:
      Encoder -> z -> ShapeDecoder -> shape -> Renderer -> spectra
    """
    def __init__(self, 
                 d_model=64,  
                 z_dim=16,
                 dec_hidden_dim=32,
                 shape_renderer_dim=64,
                 n_c=11):
        super().__init__()
        
        self.encoder = VAEEncoder(d_model=d_model, z_dim=z_dim)
        self.shape_decoder = GeometricVertexDecoder(
            z_dim=z_dim, 
            hidden_dim=dec_hidden_dim, 
            num_vertices=4
        )
        self.renderer = RCWARenderer(
            shape_dim=12, 
            hidden_dim=shape_renderer_dim, 
            out_dim=100,
            n_c=n_c
        )
    
    def reparam_sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x):
        # x: (bsz, 11, 100)
        mu, logvar = self.encoder(x)
        z = self.reparam_sample(mu, logvar)
        
        shape_pred = self.shape_decoder(z)  # (bsz,4,3)
        shape_flat = shape_pred.view(shape_pred.size(0), -1)  # (bsz,12)
        
        recon_spectra = self.renderer(shape_flat)  # (bsz,11,100)
        return shape_pred, recon_spectra, mu, logvar


###############################################################################
# 4. Losses
###############################################################################
def kl_divergence(mu, logvar):
    """
    Standard VAE KL divergence
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()

def recon_spectra_loss(pred_spectra, tgt_spectra):
    """
    MSE between predicted and ground-truth reflectance
    """
    return F.mse_loss(pred_spectra, tgt_spectra, reduction='mean')

def shape_loss(pred_shape, tgt_shape):
    """
    pred_shape: (bsz,4,3)
    tgt_shape:  (bsz,12)->(bsz,4,3)
      presence, x, y
    """
    bsz = pred_shape.size(0)
    pred = pred_shape.view(bsz,4,3)
    tgt  = tgt_shape.view(bsz,4,3)
    
    # presence
    p_pred = pred[:,:,0]
    p_tgt  = tgt[:,:,0]   # 0 or 1
    xy_pred = pred[:,:,1:]
    xy_tgt  = tgt[:,:,1:]
    
    # BCE on presence
    presence_bce = F.binary_cross_entropy_with_logits(p_pred, p_tgt, reduction='mean')
    
    # MSE on coords only where presence=1
    mask = (p_tgt > 0.5).unsqueeze(-1).expand_as(xy_pred)
    if mask.sum() > 0:
        coords_mse = F.mse_loss(xy_pred[mask], xy_tgt[mask], reduction='mean')
    else:
        coords_mse = 0.0
    
    return presence_bce + coords_mse

def geometric_presence_prior(pred_shape, prior_p=0.3):
    """
    Extra penalty that presence=1 is 'expensive'.
    We'll treat each presence p_pred (logits) as if there's a geometric prior
    with parameter prior_p => Probability of success (presence=1) = prior_p
    The log-likelihood for presence=1 would be log(prior_p).
    The presence=0 => log(1-prior_p).
    We'll do negative log-likelihood approx:
      if presence=1 => -log(prior_p)
      if presence=0 => -log(1 - prior_p)
    We'll compute the logistic predictions and pick the average penalty.
    """
    # pred_shape: (bsz,4,3)
    bsz = pred_shape.size(0)
    p_logits = pred_shape[:,:,0]  # shape (bsz,4)
    p_sig = torch.sigmoid(p_logits)
    
    # presence=1 => penalty = -log(prior_p)
    # presence=0 => penalty = -log(1 - prior_p)
    # so expected penalty = p_sig*(-log(prior_p)) + (1-p_sig)*(-log(1-prior_p))
    # => - p_sig*log(prior_p) - (1-p_sig)*log(1-prior_p)
    
    term1 = - p_sig * np.log(prior_p)
    term2 = - (1 - p_sig) * np.log(1.0 - prior_p)
    penalty = term1 + term2  # shape (bsz,4)
    
    return penalty.mean()


###############################################################################
# 5. Training
###############################################################################
def train_vae(
    csv_file="merged_s4_shapes.csv",
    save_dir="outputs",
    num_epochs=100,
    batch_size=512,
    lr=3e-4,
    z_dim=16,
    d_model=64,
    dec_hidden_dim=32,
    shape_renderer_dim=64,
    alpha=1.0,       # weight for shape loss
    beta=1.0,        # weight for spectra recon
    gamma=1e-3,      # weight for KL
    geo_prior_weight=0.1,  # weight for presence prior
    label_mask_ratio=0.3,  # fraction of shapes that have no label
    split_ratio=0.8
):
    """
    Semi-supervised VAE training loop.
    - label_mask_ratio% of data won't have shape labels (only spectra).
    - We'll do shape_loss only on labeled samples, plus recon + KL + geometric prior on all.
    """
    # create output folder
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

    # model
    model = VAENet(
        d_model=d_model, 
        z_dim=z_dim,
        dec_hidden_dim=dec_hidden_dim,
        shape_renderer_dim=shape_renderer_dim,
        n_c=11  # we fix 11 c-values
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

            # always do recon loss + kl
            L_spectra = recon_spectra_loss(recon_spectra, R_data)
            L_kl      = kl_divergence(mu, logvar)
            L_geo     = geometric_presence_prior(shape_pred, prior_p=0.3)

            # shape loss only on labeled subset
            # shape_pred: (bsz,4,3)
            # label: (bsz,12)
            # We'll pick only those with has_label=1
            if has_label.sum() > 0:
                # gather the subset
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

    # plot training curve
    plt.figure()
    plt.plot(all_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Semi-Supervised VAE-like Training Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_loss.png"))
    plt.close()

    # save model
    model_path = os.path.join(out_dir, "vae_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # quick test
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
            
            # presence accuracy only on the labeled subset
            idx_labeled = (has_label == True).nonzero(as_tuple=True)[0]
            if idx_labeled.numel() > 0:
                bsz_labeled = idx_labeled.size(0)
                shape_pred_l = shape_pred[idx_labeled]   # (bsz_labeled,4,3)
                label_l      = label[idx_labeled]        # (bsz_labeled,12)
                p_pred = shape_pred_l[:,:,0]
                p_tgt  = label_l.view(bsz_labeled,4,3)[:,:,0]
                p_sig  = torch.sigmoid(p_pred)
                p_bin  = (p_sig>0.5).float()
                presence_correct += (p_bin == p_tgt).sum().item()
                presence_total   += p_tgt.numel()

            # spectra MSE on all
            sqerr = (recon_spectra - R_data)**2
            spectra_mse_accum += sqerr.sum().item()
            spectra_count     += sqerr.numel()

    if presence_total > 0:
        presence_acc = presence_correct / presence_total
    else:
        presence_acc = 0.0
    spectra_mse = spectra_mse_accum / (spectra_count if spectra_count>0 else 1)

    print("=== Test Results ===")
    print(f"Presence Accuracy (only on labeled test): {presence_acc:.4f}")
    print(f"Spectra MSE: {spectra_mse:.6f}")


###############################################################################
# 6. Example Main
###############################################################################
if __name__ == "__main__":
    train_vae(
        csv_file="merged_s4_shapes_20250114_175110.csv",
        save_dir="outputs",
        num_epochs=100,
        batch_size=512*8,
        lr=3e-4,
        z_dim=16,
        d_model=64,
        dec_hidden_dim=32,
        shape_renderer_dim=64,
        alpha=1.0,              # shape loss scale
        beta=1.0,               # spectra recon scale
        gamma=1e-3,             # kl scale
        geo_prior_weight=0.1,   # presence prior scale
        label_mask_ratio=0.3,   # 30% unlabeled
        split_ratio=0.8
    )

