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
# 1. Dataset (Same as Before)
###############################################################################

class ShapeDataset(Dataset):
    """
    Reads the CSV, groups by (folder_key, shape_idx),
    extracts 11 x 100 reflectance arrays, and parses up to 4 vertices in the first quadrant.
    """
    def __init__(self, csv_file="merged_s4_shapes.csv"):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        
        self.df["unique_id"] = self.df["folder_key"].astype(str) + "_" + self.df["shape_idx"].astype(str)
        
        self.r_columns = [col for col in self.df.columns if col.startswith("R@")]
        self.num_wavelengths = len(self.r_columns)  # typically 100
        
        self.grouped_data = []
        for uid, grp in self.df.groupby("unique_id"):
            # Sort by "c"
            grp_sorted = grp.sort_values(by="c")
            if len(grp_sorted) != 11:
                continue
            
            R_data = grp_sorted[self.r_columns].values.astype(np.float32)  # (11,100)
            vertices_str = grp_sorted["vertices_str"].values[0]
            
            # Parse up to 4 Q1 vertices
            raw_pairs = vertices_str.strip().split(";")
            vertices = []
            for pair in raw_pairs:
                xy = pair.split(",")
                if len(xy)==2:
                    x_val = float(xy[0])
                    y_val = float(xy[1])
                    if x_val>0 and y_val>=0:
                        vertices.append((x_val,y_val))
            
            vertices = vertices[:4]
            num_actual = len(vertices)
            while len(vertices)<4:
                vertices.append((0.0,0.0))
            presence = [1.0]*num_actual + [0.0]*(4-num_actual)
            
            # Flatten (12,)
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
        self.data_len = len(self.grouped_data)
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        item = self.grouped_data[idx]
        return item["R_data"], item["label"], item["unique_id"]


###############################################################################
# 2. VAE-Like Model
###############################################################################

class VAEEncoder(nn.Module):
    """
    Encode 11x100 spectra -> latent z ~ N(mu, sigma^2).
    We'll sum (DeepSets) or mean across 11 for order invariance, then produce mu and logvar.
    """
    def __init__(self, input_dim=100, embed_dim=64, z_dim=16):
        """
        input_dim=100 for each spectral row
        embed_dim=64 is hidden dimension after aggregator
        z_dim=16 is latent dimension
        """
        super().__init__()
        self.phi_fc1 = nn.Linear(input_dim, 64)
        self.phi_fc2 = nn.Linear(64, 64)
        self.rho_fc1 = nn.Linear(64, embed_dim)
        
        # final layers for mu, logvar
        self.mu_fc     = nn.Linear(embed_dim, z_dim)
        self.logvar_fc = nn.Linear(embed_dim, z_dim)
    
    def forward(self, x):
        """
        x: (batch_size, 11, 100)
        """
        bsz, n_spectra, d_in = x.shape
        # phi
        x_flat = x.view(bsz*n_spectra, d_in)
        h = F.relu(self.phi_fc1(x_flat))
        h = F.relu(self.phi_fc2(h))
        h = h.view(bsz, n_spectra, 64)
        
        # sum aggregator
        h_sum = torch.sum(h, dim=1)  # (bsz, 64)
        
        # finalize
        emb = F.relu(self.rho_fc1(h_sum))  # (bsz, embed_dim)
        
        mu     = self.mu_fc(emb)      # (bsz, z_dim)
        logvar = self.logvar_fc(emb)  # (bsz, z_dim)
        return mu, logvar

class ShapeDecoderLSTM(nn.Module):
    """
    Given z, decode up to 4 vertices: (presence, x, y).
    Similar to our previous LSTM approach, except input_size=1 with a dummy input,
    but we transform z -> h0, c0.
    """
    def __init__(self, z_dim=16, hidden_dim=32, num_vertices=4):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_vertices = num_vertices
        
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True)
        
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
        
        h0 = self.fc_init_h(z).unsqueeze(0)  # (1,bsz,hidden_dim)
        c0 = self.fc_init_c(z).unsqueeze(0)  # (1,bsz,hidden_dim)
        
        dummy_input = torch.zeros(bsz, self.num_vertices, 1, device=device)
        lstm_out, _ = self.lstm(dummy_input, (h0,c0))  # (bsz, 4, hidden_dim)
        
        preds = self.fc_out(lstm_out)  # (bsz, 4, 3)
        return preds


class ShapeRenderer(nn.Module):
    """
    Given predicted shape -> produce predicted 11x100 spectra.
    We do a small MLP that takes: (c_value, shape_embedding) -> 100-d reflection row
    We'll do presence+x+y -> shape embedding, or simply flatten the 4*(p,x,y).
    
    Implementation detail:
     - We'll flatten the predicted shape to (bsz, 12) = 4*(p,x,y).
     - For each c in [0..10], we pass [c, shape_flat] to a small MLP -> 100 dims.
     - We'll stack results -> (bsz, 11, 100).
    """
    def __init__(self, shape_dim=12, hidden_dim=64, out_dim=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # We'll embed c into a small MLP as well or just treat c as a scalar input
        # We'll do a single MLP that sees shape_flat + c_value
        self.fc1 = nn.Linear(shape_dim+1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, shape_flat):
        """
        shape_flat: (bsz, 12) => [p1,x1,y1, p2,x2,y2, p3,x3,y3, p4,x4,y4]
        We'll produce output_spectra: (bsz, 11, 100)
        """
        device = shape_flat.device
        bsz = shape_flat.size(0)
        
        # We'll build 11 c-values from 0.0..1.0
        c_values = torch.linspace(0,1,11, device=device)  # shape (11,)
        
        outputs = []
        for i in range(11):
            cval = c_values[i].view(1,1).repeat(bsz,1)  # (bsz,1)
            inp = torch.cat([shape_flat, cval], dim=1)  # (bsz, 12+1=13)
            
            h = F.relu(self.fc1(inp))
            h = F.relu(self.fc2(h))
            row = self.fc3(h)  # (bsz, 100)
            
            outputs.append(row.unsqueeze(1))  # (bsz,1,100)
        
        # Stack
        out_spectra = torch.cat(outputs, dim=1)  # (bsz, 11, 100)
        return out_spectra


class VAENet(nn.Module):
    """
    The overall VAE-like model:
      1) Encode spectra -> z
      2) Decode z -> shape
      3) Render shape -> predicted spectra
    We'll optimize shape-likelihood + spectra reconstruction + KL.
    """
    def __init__(self, 
                 input_dim=100,  # each spectral row
                 z_dim=16,
                 enc_embed_dim=64, 
                 dec_hidden_dim=32,
                 shape_renderer_dim=64):
        super().__init__()
        
        # 1) Encoder
        self.encoder = VAEEncoder(input_dim=input_dim, embed_dim=enc_embed_dim, z_dim=z_dim)
        
        # 2) Shape Decoder
        self.shape_decoder = ShapeDecoderLSTM(z_dim=z_dim, hidden_dim=dec_hidden_dim, num_vertices=4)
        
        # 3) Renderer
        self.renderer = ShapeRenderer(shape_dim=12, hidden_dim=shape_renderer_dim, out_dim=input_dim)
    
    def reparam_sample(self, mu, logvar):
        """
        Reparameterization trick: sample z ~ N(mu, sigma^2)
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x):
        """
        x: (bsz, 11, 100) input spectra
        returns:
          shape_pred: (bsz,4,3)
          recon_spectra: (bsz,11,100)
          mu, logvar: for kl
        """
        mu, logvar = self.encoder(x)  # => (bsz, z_dim)
        z = self.reparam_sample(mu, logvar)
        
        shape_pred = self.shape_decoder(z)  # (bsz, 4, 3)
        
        # flatten shape
        shape_flat = shape_pred.view(shape_pred.size(0), -1)  # (bsz, 12)
        recon_spectra = self.renderer(shape_flat)  # (bsz, 11, 100)
        
        return shape_pred, recon_spectra, mu, logvar


###############################################################################
# 3. Losses
###############################################################################

def shape_loss(pred_shape, tgt_shape):
    """
    pred_shape: (bsz,4,3)
    tgt_shape: (bsz,12) -> reshape => (bsz,4,3)
    
    - presence => BCE w/ logits
    - (x,y) => MSE, masked by presence
    """
    bsz = pred_shape.size(0)
    pred = pred_shape.view(bsz,4,3)
    tgt  = tgt_shape.view(bsz,4,3)
    
    p_pred = pred[:,:,0]
    p_tgt  = tgt[:,:,0]
    
    xy_pred = pred[:,:,1:]
    xy_tgt  = tgt[:,:,1:]
    
    presence_bce = F.binary_cross_entropy_with_logits(p_pred, p_tgt, reduction='mean')
    
    mask = (p_tgt>0.5).unsqueeze(-1).expand_as(xy_pred)
    if mask.sum()>0:
        coords_mse = F.mse_loss(xy_pred[mask], xy_tgt[mask], reduction='mean')
    else:
        coords_mse = 0.0
    
    return presence_bce + coords_mse

def recon_spectra_loss(pred_spectra, tgt_spectra):
    """
    pred_spectra: (bsz,11,100)
    tgt_spectra: (bsz,11,100)
    We'll do MSE
    """
    return F.mse_loss(pred_spectra, tgt_spectra, reduction='mean')

def kl_divergence(mu, logvar):
    """
    KL( q(z|x) || N(0,I) ) = 0.5 * sum(1 + logvar - mu^2 - e^logvar )
    for each element in the batch
    """
    # shape: (bsz, z_dim)
    # We'll average across the batch
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


###############################################################################
# 4. Training Script
###############################################################################

def train_vae(csv_file="merged_s4_shapes.csv",
              save_dir="outputs",
              num_epochs=50,
              batch_size=8,
              lr=1e-3,
              z_dim=16,
              alpha=1.0,   # weight for shape loss
              beta=1.0,    # weight for spectra recon
              gamma=1e-3,  # weight for KL
              split_ratio=0.8):
    
    # Output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(save_dir, f"vae_train_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    # Dataset
    full_ds = ShapeDataset(csv_file)
    ds_len = len(full_ds)
    train_len = int(ds_len*split_ratio)
    test_len = ds_len - train_len
    train_ds, test_ds = random_split(full_ds, [train_len, test_len])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model
    model = VAENet(
        input_dim=100,
        z_dim=z_dim,
        enc_embed_dim=64,
        dec_hidden_dim=32,
        shape_renderer_dim=64
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    all_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for (R_data_np, label_np, uid) in train_loader:
            R_data = torch.tensor(R_data_np, dtype=torch.float32, device=device) # (bsz,11,100)
            label  = torch.tensor(label_np,  dtype=torch.float32, device=device)  # (bsz,12)
            
            # Forward
            shape_pred, recon_spectra, mu, logvar = model(R_data)
            
            # Compute losses
            L_shape   = shape_loss(shape_pred, label)
            L_spectra = recon_spectra_loss(recon_spectra, R_data)
            L_kl      = kl_divergence(mu, logvar)
            
            loss = alpha*L_shape + beta*L_spectra + gamma*L_kl
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        all_losses.append(epoch_loss)
        
        if (epoch+1)%5==0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}  (Shape={L_shape.item():.4f}, Spectra={L_spectra.item():.4f}, KL={L_kl.item():.4f})")
    
    # Plot training loss
    plt.figure()
    plt.plot(all_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE-like Training Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_loss.png"))
    plt.close()
    
    # Save model
    model_path = os.path.join(out_dir, "vae_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Quick Test Evaluation: shape presence accuracy & spectra MSE
    model.eval()
    presence_correct = 0
    presence_total   = 0
    spectra_mse_accum = 0.0
    spectra_count = 0
    with torch.no_grad():
        for (R_data_np, label_np, uid) in test_loader:
            R_data = torch.tensor(R_data_np, dtype=torch.float32, device=device)
            label  = torch.tensor(label_np,  dtype=torch.float32, device=device)
            
            shape_pred, recon_spectra, mu, logvar = model(R_data)
            
            # presence bits
            bsz = shape_pred.size(0)
            p_pred = shape_pred[:,:,0]         # (bsz,4)
            p_tgt  = label.view(bsz,4,3)[:,:,0]
            p_sig = torch.sigmoid(p_pred)
            p_bin = (p_sig>0.5).float()
            presence_correct += (p_bin==p_tgt).sum().item()
            presence_total   += p_tgt.numel()
            
            # spectra MSE
            sqerr = (recon_spectra - R_data)**2
            spectra_mse_accum += sqerr.sum().item()
            spectra_count     += sqerr.numel()
    
    presence_acc = presence_correct / (presence_total if presence_total>0 else 1)
    spectra_mse  = spectra_mse_accum / (spectra_count if spectra_count>0 else 1)
    
    print("=== Test Results ===")
    print(f"Presence Accuracy: {presence_acc:.4f}")
    print(f"Spectra MSE:       {spectra_mse:.6f}")

###############################################################################
# 5. Main
###############################################################################

if __name__ == "__main__":
    train_vae(
        csv_file="merged_s4_shapes.csv",
        save_dir="outputs",
        num_epochs=50,
        batch_size=8,
        lr=1e-3,
        z_dim=16,
        alpha=1.0,
        beta=1.0,
        gamma=1e-3,
        split_ratio=0.8
    )

