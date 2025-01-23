#!/usr/bin/env python3

import os
import csv
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

###############################################################################
# 1) Configuration
###############################################################################
CSV_PATH   = "merged_s4_shapes.csv"
BATCH_SIZE = 64
NUM_EPOCHS = 50
LR         = 1e-3
HIDDEN_SIZE= 32  # LSTM hidden size
MAX_VERTS  = 4   # we only track up to 4 vertices
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# 2) Dataset & Parsing
###############################################################################
def parse_vertices_str(vert_str):
    """
    Parse semicolon-delimited "x,y"; keep only x>=0,y>=0
    Return list[(x,y)].
    """
    pts = []
    if not isinstance(vert_str, str):
        return pts
    for token in vert_str.split(';'):
        token = token.strip()
        if not token:
            continue
        sub = token.split(',')
        if len(sub)!=2:
            continue
        try:
            x = float(sub[0]); y = float(sub[1])
            if x>=0 and y>=0:
                pts.append((x,y))
        except:
            pass
    return pts

class SupervisedDataset(Dataset):
    """
    Expects a CSV with columns:
      - c  in [0,1] (if known), else possibly NaN
      - R@1.04, R@1.05, ..., R@2.50  (100 reflection values)
      - vertices_str  => e.g. "0.1,0.2;0.3,0.5;..."
    We'll parse them into:
      - input_seq => shape (100,) of reflection
      - label_c   => float in [0,1], or None
      - label_vp  => [4] presence bits
      - label_vxy => [4,2] coordinates
    Some rows might have fewer than 4 points => fill remainder with zeros, presence=0.
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # Identify reflection columns
        r_cols  = [c for c in df.columns if c.startswith("R@")]
        r_cols  = sorted(r_cols, key=lambda x: float(x.split('@')[1]))
        # We'll store data in lists to convert to torch later
        self.X_seq   = []
        self.y_c     = []
        self.y_vpres = []
        self.y_vxy   = []

        for idx, row in df.iterrows():
            # reflection array => shape (100,)
            r_vals = row[r_cols].values.astype(np.float32)
            if len(r_vals) != 100:
                # skip or handle mismatch
                continue
            
            # c => possibly unknown => store as float; if NaN => set to -1
            c_val = row["c"] if "c" in df.columns else np.nan
            if pd.isna(c_val):
                c_val = -1.0  # indicates "unknown"
            
            # vertices
            vp = np.zeros((MAX_VERTS,), dtype=np.float32)
            vx = np.zeros((MAX_VERTS,), dtype=np.float32)
            vy = np.zeros((MAX_VERTS,), dtype=np.float32)

            pts = parse_vertices_str(row["vertices_str"] if "vertices_str" in df.columns else "")
            # Sort them by angle if needed, or just keep in order
            # Here, let's just keep them as is for simplicity
            for i, (xx,yy) in enumerate(pts[:MAX_VERTS]):
                vp[i] = 1.0
                vx[i] = xx
                vy[i] = yy

            self.X_seq.append(r_vals)
            self.y_c.append(c_val)
            self.y_vpres.append(vp)
            self.y_vxy.append( np.stack([vx,vy], axis=1) ) # shape [4,2]

        self.X_seq   = np.stack(self.X_seq, axis=0)   # [N,100]
        self.y_c     = np.array(self.y_c, dtype=np.float32)  # [N]
        self.y_vpres = np.stack(self.y_vpres, axis=0) # [N,4]
        self.y_vxy   = np.stack(self.y_vxy, axis=0)   # [N,4,2]

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        # Return a dict for clarity
        x_seq    = self.X_seq[idx]
        c_val    = self.y_c[idx]     # in [-1..1], -1 => unknown
        vpres    = self.y_vpres[idx] # [4]
        vxy      = self.y_vxy[idx]   # [4,2]
        return {
            "input_seq": torch.from_numpy(x_seq),   # shape (100,)
            "c_val":     torch.tensor(c_val),        # shape ()
            "v_pres":    torch.from_numpy(vpres),    # shape (4,)
            "v_xy":      torch.from_numpy(vxy),      # shape (4,2)
        }

###############################################################################
# 3) Model Definition (LSTM + MLP heads)
###############################################################################
class LSTMModel(nn.Module):
    """
    We treat the reflection columns as a 100-step time series with 1 feature each.
    Then we take the final LSTM hidden state => feed into an MLP => produce:
      - c_pred in [0,1]
      - v_pres in [0,1] (4 dims)
      - v_xy   in [0,∞) (4,2) dims => we do softplus
    """
    def __init__(self, hidden_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        # LSTM: input_size=1 because each step is a single reflection value
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True)

        # We map from hidden_size => MLP => output
        # Let's define how many outputs we need:
        # c => 1
        # v_pres => 4
        # v_xy => 4*2=8
        # total = 1 + 4 + 8 = 13
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 13)
        )

    def forward(self, x_seq):
        """
        x_seq: [B, 100], but LSTM needs shape [B, 100, 1]
        Return:
          c_pred:   [B]
          vpres_pred:[B,4]
          vxy_pred: [B,4,2]
        """
        B = x_seq.size(0)

        # [B,100] => [B,100,1]
        inp = x_seq.unsqueeze(-1)

        # LSTM
        # out: [B,100,hidden], h_n: [1,B,hidden], c_n: [1,B,hidden]
        out, (h_n, c_n) = self.lstm(inp)
        # We'll use the final hidden state h_n => shape [1,B,hidden]
        h_final = h_n.squeeze(0)  # [B, hidden]

        # MLP => [B,13]
        y_out = self.fc(h_final)

        # parse them
        # index 0 => c
        # index 1..4 => v_pres
        # index 5..12 => v_xy
        c_logit      = y_out[:, 0]                 # [B]
        pres_logits  = y_out[:, 1:1+4]             # [B,4]
        xy_logits    = y_out[:, 5:5+8]             # [B,8]

        c_pred       = torch.sigmoid(c_logit)      # in [0,1]
        vpres_pred   = torch.sigmoid(pres_logits)  # in [0,1]
        xy_reshaped  = xy_logits.view(B, 4, 2)
        vxy_pred     = F.softplus(xy_reshaped)     # in [0,∞)

        return c_pred, vpres_pred, vxy_pred

###############################################################################
# 4) Training Script
###############################################################################
def train_one_epoch(model, loader, optimizer, loss_weights=None):
    """
    loss_weights can be a dict like {'c':1.0, 'pres':1.0, 'xy':1.0}
    We do standard supervised losses (MSE, BCE, etc.) only on known data.
    """
    if loss_weights is None:
        loss_weights = {'c':1.0, 'pres':1.0, 'xy':1.0}
    model.train()
    total_loss = 0.0
    for batch in loader:
        x_seq = batch["input_seq"].to(DEVICE)   # [B,100]
        c_val = batch["c_val"].to(DEVICE)       # [B]
        v_pres= batch["v_pres"].to(DEVICE)      # [B,4]
        v_xy  = batch["v_xy"].to(DEVICE)        # [B,4,2]

        optimizer.zero_grad()
        c_pred, vpres_pred, vxy_pred = model(x_seq)  # [B], [B,4], [B,4,2]

        # 1) c => MSE (or L1). Skip if c_val<0 => unknown
        mask_c    = (c_val >= 0)
        known_c   = c_val[mask_c]
        pred_c    = c_pred[mask_c]
        loss_c    = 0.0
        if known_c.numel() > 0:
            loss_c = F.mse_loss(pred_c, known_c)

        # 2) presence => BCE if we know the vertex. 
        #    Actually in this example, we assume if there's a row for v_xy, we consider that known. 
        #    If you have partial labeling, you might do more logic here.
        loss_pres = F.binary_cross_entropy(vpres_pred, v_pres, reduction='none')  # [B,4]
        # If a row has no vertex info, you might mask out everything. 
        # For simplicity, assume all are known => average
        loss_pres = loss_pres.mean()

        # 3) vertex coords => MSE, but only for vertices that are present in the ground truth
        # v_pres=1 => we apply MSE. 
        # shape => [B,4,2]
        # We'll do elementwise => sum or mean across present coords
        mask_v  = v_pres>0.5  # [B,4]
        mask_v2 = mask_v.unsqueeze(-1).expand(-1,-1,2)  # [B,4,2]
        # gather ground-truth
        v_xy_gt  = v_xy[mask_v2]
        v_xy_pd  = vxy_pred[mask_v2]
        loss_xy  = 0.0
        if v_xy_gt.numel()>0:
            loss_xy = F.mse_loss(v_xy_pd, v_xy_gt)

        loss = (loss_weights['c']*loss_c +
                loss_weights['pres']*loss_pres +
                loss_weights['xy']*loss_xy)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()*x_seq.size(0)

    avg_loss = total_loss/len(loader.dataset)
    return avg_loss

def validate(model, loader):
    """
    Compute the same losses on a validation set (if any).
    Here we just do an overall average to see if we overfit.
    """
    model.eval()
    total_loss=0.0
    n_samples=0
    with torch.no_grad():
        for batch in loader:
            x_seq = batch["input_seq"].to(DEVICE)
            c_val = batch["c_val"].to(DEVICE)
            v_pres= batch["v_pres"].to(DEVICE)
            v_xy  = batch["v_xy"].to(DEVICE)

            c_pred, vpres_pred, vxy_pred = model(x_seq)

            # same computations
            mask_c  = (c_val>=0)
            known_c = c_val[mask_c]
            pred_c  = c_pred[mask_c]
            loss_c  = 0.0
            if known_c.numel() > 0:
                loss_c = F.mse_loss(pred_c, known_c)

            loss_pres = F.binary_cross_entropy(vpres_pred, v_pres, reduction='none').mean()

            mask_v  = (v_pres>0.5)
            mask_v2 = mask_v.unsqueeze(-1).expand(-1,-1,2)
            v_xy_gt = v_xy[mask_v2]
            v_xy_pd = vxy_pred[mask_v2]
            loss_xy = 0.0
            if v_xy_gt.numel()>0:
                loss_xy = F.mse_loss(v_xy_pd, v_xy_gt)

            loss = 1.0*loss_c + 1.0*loss_pres + 1.0*loss_xy
            bs   = x_seq.size(0)
            total_loss += loss.item()*bs
            n_samples  += bs

    return total_loss/n_samples if n_samples>0 else 0.0


###############################################################################
# 5) Main
###############################################################################
def main():
    # 1) load dataset
    ds = SupervisedDataset(CSV_PATH)
    print(f"Dataset size = {len(ds)}")

    # optionally: split train/val
    n_val = int(0.1 * len(ds))
    n_tr  = len(ds) - n_val
    tr_ds, val_ds = torch.utils.data.random_split(ds, [n_tr, n_val],
                                                  generator=torch.Generator().manual_seed(42))

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader= DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 2) build model
    model = LSTMModel(hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 3) training loop
    for epoch in range(1, NUM_EPOCHS+1):
        tr_loss = train_one_epoch(model, tr_loader, optimizer)
        val_loss= validate(model, val_loader)
        print(f"[Epoch {epoch}/{NUM_EPOCHS}] "
              f"TrainLoss={tr_loss:.5f} ValLoss={val_loss:.5f}")

    # 4) Save final
    os.makedirs("checkpoint_simple", exist_ok=True)
    torch.save(model.state_dict(), "checkpoint_simple/model.pt")
    print("Training completed and model saved!")

if __name__=="__main__":
    main()
