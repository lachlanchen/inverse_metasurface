#!/usr/bin/env python3

import os
import math
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline

###############################################################################
# 1) Configuration
###############################################################################
CSV_PATH      = "merged_s4_shapes.csv"
BATCH_SIZE    = 4096
ACCUM_STEPS   = 8       # We'll accumulate gradients over 8 mini-batches
NUM_EPOCHS    = 100
LR            = 1e-3
HIDDEN_SIZE   = 32      # LSTM hidden size
MAX_VERTS     = 4       # we only track up to 4 vertices
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    Reads a CSV with columns:
      - c  in [0,1] (if known), or possibly NaN => treat as unknown
      - R@1.04, R@1.05, ..., R@2.50  (100 reflection values)
      - vertices_str => e.g. "0.1,0.2;0.3,0.5;..."
    We'll parse them into:
      - input_seq => shape (100,) of reflection
      - label_c   => float in [0,1], or -1 if unknown
      - label_vpres => [4] presence bits
      - label_vxy   => [4,2] coordinates
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
        x_seq    = self.X_seq[idx]
        c_val    = self.y_c[idx]      # in [-1..1], -1 => unknown
        vpres    = self.y_vpres[idx]  # [4]
        vxy      = self.y_vxy[idx]    # [4,2]
        return {
            "input_seq": torch.from_numpy(x_seq),   
            "c_val":     torch.tensor(c_val),        
            "v_pres":    torch.from_numpy(vpres),    
            "v_xy":      torch.from_numpy(vxy),      
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
      - v_xy   in [0,∞) (4,2)
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
        # total outputs: c => 1, v_pres => 4, v_xy => 8 => total=13
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 13)
        )

    def forward(self, x_seq):
        """
        x_seq: [B, 100], we reshape to [B,100,1] for LSTM
        Return:
          c_pred:   [B]
          vpres_pred:[B,4]
          vxy_pred: [B,4,2]
        """
        B = x_seq.size(0)
        inp = x_seq.unsqueeze(-1)  # [B,100,1]

        out, (h_n, c_n) = self.lstm(inp)
        # final hidden state => shape [1,B,hidden]
        h_final = h_n.squeeze(0)  # [B,hidden]

        y_out = self.fc(h_final)  # [B,13]

        c_logit      = y_out[:, 0]                 # [B]
        pres_logits  = y_out[:, 1:1+4]             # [B,4]
        xy_logits    = y_out[:, 5:5+8]             # [B,8]

        c_pred       = torch.sigmoid(c_logit)      # in [0,1]
        vpres_pred   = torch.sigmoid(pres_logits)  # in [0,1]
        xy_reshaped  = xy_logits.view(B, 4, 2)
        vxy_pred     = F.softplus(xy_reshaped)     # in [0,∞)

        return c_pred, vpres_pred, vxy_pred

###############################################################################
# 4) Training & Validation
###############################################################################
def train_one_epoch(model, loader, optimizer, loss_weights=None):
    if loss_weights is None:
        loss_weights = {'c':1.0, 'pres':1.0, 'xy':1.0}
    model.train()
    total_loss = 0.0
    # We'll do gradient accumulation
    optimizer.zero_grad()
    step_count= 0

    for i, batch in enumerate(loader):
        x_seq = batch["input_seq"].to(DEVICE)   # [B,100]
        c_val = batch["c_val"].to(DEVICE)       # [B]
        v_pres= batch["v_pres"].to(DEVICE)      # [B,4]
        v_xy  = batch["v_xy"].to(DEVICE)        # [B,4,2]

        c_pred, vpres_pred, vxy_pred = model(x_seq)

        # (1) c => MSE if known
        mask_c    = (c_val >= 0)
        known_c   = c_val[mask_c]
        pred_c    = c_pred[mask_c]
        loss_c    = 0.0
        if known_c.numel() > 0:
            loss_c = F.mse_loss(pred_c, known_c)

        # (2) presence => BCE
        # we'll assume everything is known for presence. Or you can do partial logic.
        loss_pres = F.binary_cross_entropy(vpres_pred, v_pres, reduction='none').mean()

        # (3) vertex coords => MSE only for present vertices
        mask_v  = (v_pres>0.5)  # [B,4]
        mask_v2 = mask_v.unsqueeze(-1).expand(-1,-1,2)
        v_xy_gt = v_xy[mask_v2]
        v_xy_pd = vxy_pred[mask_v2]
        loss_xy = 0.0
        if v_xy_gt.numel()>0:
            loss_xy = F.mse_loss(v_xy_pd, v_xy_gt)

        loss = loss_weights['c']*loss_c + loss_weights['pres']*loss_pres + loss_weights['xy']*loss_xy
        loss.backward()

        step_count += 1
        if step_count % ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * x_seq.size(0)

    # in case the dataset size is not multiple of accum steps
    if step_count % ACCUM_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(loader.dataset)

def validate(model, loader):
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

            loss_batch = (loss_c + loss_pres + loss_xy)
            bs   = x_seq.size(0)
            total_loss += loss_batch.item()*bs
            n_samples  += bs

    return total_loss/n_samples if n_samples>0 else 0.0

###############################################################################
# 5) Generate a smooth random spectrum
###############################################################################
def generate_smooth_spectrum(num_points=100):
    n_ctrl= 8
    x_ctrl= np.linspace(0, 1, n_ctrl)
    y_ctrl= np.random.rand(n_ctrl)*0.8 +0.1
    spline= make_interp_spline(x_ctrl, y_ctrl, k=3)
    x_big= np.linspace(0,1,num_points)
    y_big= spline(x_big)
    y_big= np.clip(y_big, 0,1)
    return torch.tensor(y_big, dtype=torch.float)

###############################################################################
# 6) Polygon plotting (C4 replication)
###############################################################################
def replicate_c4(verts):
    out_list=[]
    angles= [0, math.pi/2, math.pi, 3*math.pi/2]
    for a in angles:
        cosA= math.cos(a)
        sinA= math.sin(a)
        rot= torch.tensor([[cosA, -sinA],[sinA, cosA]],dtype=torch.float)
        chunk= verts @ rot.T
        out_list.append(chunk)
    return torch.cat(out_list, dim=0)

def angle_sort(points):
    px= points[:,0]
    py= points[:,1]
    ang= torch.atan2(py, px)
    idx= torch.argsort(ang)
    return points[idx]

def close_polygon(pts):
    if pts.size(0)>1:
        return torch.cat([pts, pts[:1]], dim=0)
    return pts

def plot_polygon(pts, c_val, out_path, title="C4 polygon"):
    pts_sorted = angle_sort(pts)
    pts_closed = close_polygon(pts_sorted)
    sx = pts_closed[:,0].numpy()
    sy = pts_closed[:,1].numpy()
    plt.figure()
    plt.fill(sx, sy, color='red', alpha=0.3)
    plt.plot(sx, sy, 'ro-')
    plt.title(f"{title}, c={c_val:.3f}")
    plt.axhline(0,color='k',lw=0.5)
    plt.axvline(0,color='k',lw=0.5)
    plt.savefig(out_path)
    plt.close()

###############################################################################
# 7) Testing / Inference
###############################################################################
def test_inference(model, dataset, out_dir):
    """
    - (A) Predict on a random smooth spectrum.
    - (B) Predict on row=0 from dataset, compare with ground-truth.
    - Save everything to out_dir.
    """
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    # (A) random smooth
    random_sp = generate_smooth_spectrum(100).to(DEVICE)
    random_sp_in= random_sp.unsqueeze(0)
    with torch.no_grad():
        c_pred, vpres_pred, vxy_pred = model(random_sp_in)
    c_val = float(c_pred[0].item())
    pres_np = vpres_pred[0].cpu().numpy()
    xy_np   = vxy_pred[0].cpu().numpy()

    # store random spectrum
    np.savetxt(os.path.join(out_dir,"random_spectrum.txt"),
               random_sp.cpu().numpy(), fmt="%.5f")

    # Build polygon
    keep_pts=[]
    for i in range(MAX_VERTS):
        if pres_np[i]>0.5:
            keep_pts.append(xy_np[i])
    if len(keep_pts)==0:
        keep_pts_t = torch.zeros((1,2))
    else:
        keep_pts_t = torch.tensor(keep_pts, dtype=torch.float)
    c4_verts= replicate_c4(keep_pts_t)
    plot_polygon(c4_verts, c_val, os.path.join(out_dir,"smooth_polygon.png"),
                 title="Pred C4 polygon (smooth input)")

    with open(os.path.join(out_dir,"smooth_pred.txt"),"w") as f:
        f.write(f"Pred c={c_val:.3f}\n")
        for i in range(MAX_VERTS):
            f.write(f"v_pres[{i}]={pres_np[i]:.3f}, x={xy_np[i,0]:.3f}, y={xy_np[i,1]:.3f}\n")

    # (B) pick row=0
    if len(dataset)==0:
        print("[WARN] no data => can't do row test.")
        return

    sample= dataset[0]
    real_sp= sample["input_seq"].to(DEVICE) 
    gt_c   = float(sample["c_val"].item())
    gt_vp  = sample["v_pres"].numpy()
    gt_xy  = sample["v_xy"].numpy()

    real_sp_in= real_sp.unsqueeze(0)
    with torch.no_grad():
        c_p2, vp_p2, xy_p2= model(real_sp_in)
    c_val2= float(c_p2[0].item())
    vp2_np= vp_p2[0].cpu().numpy()
    xy2_np= xy_p2[0].cpu().numpy()

    with open(os.path.join(out_dir,"test_row_pred.txt"),"w") as f:
        f.write(f"GT c={gt_c:.3f}, Pred c={c_val2:.3f}\n\n")
        f.write("GT vertices:\n")
        for i in range(MAX_VERTS):
            if gt_vp[i]>0.5:
                f.write(f" i={i}: x={gt_xy[i,0]:.3f}, y={gt_xy[i,1]:.3f}\n")
        f.write("\nPred vertices:\n")
        for i in range(MAX_VERTS):
            f.write(f" i={i}: pres={vp2_np[i]:.3f}, "
                    f"x={xy2_np[i,0]:.3f}, y={xy2_np[i,1]:.3f}\n")

    plt.figure()
    plt.plot(real_sp.cpu().numpy(), 'o-', label="Dataset Row=0 Spectrum")
    plt.title("Row=0 Spectrum")
    plt.ylim([0,1])
    plt.legend()
    plt.savefig(os.path.join(out_dir,"test_row_spectrum.png"))
    plt.close()

    # polygon
    keep_pts2=[]
    for i in range(MAX_VERTS):
        if vp2_np[i]>0.5:
            keep_pts2.append(xy2_np[i])
    if len(keep_pts2)==0:
        keep_pts2_t = torch.zeros((1,2))
    else:
        keep_pts2_t = torch.tensor(keep_pts2, dtype=torch.float)
    c4_verts2= replicate_c4(keep_pts2_t)
    plot_polygon(c4_verts2, c_val2,
                 os.path.join(out_dir,"test_row_polygon.png"),
                 title="Pred polygon row=0")

    # also GT polygon
    keep_pts_gt=[]
    for i in range(MAX_VERTS):
        if gt_vp[i]>0.5:
            keep_pts_gt.append(gt_xy[i])
    if len(keep_pts_gt)==0:
        keep_pts_gt_t= torch.zeros((1,2))
    else:
        keep_pts_gt_t= torch.tensor(keep_pts_gt, dtype=torch.float)
    c4_gt= replicate_c4(keep_pts_gt_t)
    plot_polygon(c4_gt, gt_c if gt_c>=0 else 999.0,
                 os.path.join(out_dir,"test_row_polygon_gt.png"),
                 title="GT polygon row=0")


###############################################################################
# 8) Main
###############################################################################
def main():
    # 1) load dataset
    ds = SupervisedDataset(CSV_PATH)
    print(f"Dataset size = {len(ds)}")

    # 2) split train/val
    n_val = int(0.1 * len(ds))
    n_tr  = len(ds) - n_val
    tr_ds, val_ds = torch.utils.data.random_split(ds, [n_tr, n_val],
                                                  generator=torch.Generator().manual_seed(42))
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader= DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 3) build model
    model = LSTMModel(hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 4) training loop
    for epoch in range(1, NUM_EPOCHS+1):
        tr_loss = train_one_epoch(model, tr_loader, optimizer)
        val_loss= validate(model, val_loader)
        print(f"[Epoch {epoch}/{NUM_EPOCHS}] "
              f"TrainLoss={tr_loss:.5f} ValLoss={val_loss:.5f}")

    # 5) save final
    os.makedirs("checkpoint_simple", exist_ok=True)
    ckpt_path= "checkpoint_simple/model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print("Training completed and model saved!")

    # 6) do a quick inference test
    dt_str= datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir= f"supervised_inference_{dt_str}"

    test_inference(model, ds, out_dir)
    print(f"Inference results saved to {out_dir}/")

if __name__=="__main__":
    main()
