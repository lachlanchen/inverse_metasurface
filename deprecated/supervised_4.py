#!/usr/bin/env python3

import os
import math
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline

##############################################################################
# 1) Configuration
##############################################################################
CSV_PATH      = "merged_s4_shapes.csv"
BATCH_SIZE    = 256    # Adjust to your hardware
NUM_EPOCHS    = 1000
LR            = 1e-3
HIDDEN_SIZE   = 128    # LSTM hidden size
LSTM_LAYERS   = 1      # 2-layer LSTM
MAX_VERTS     = 4      # Up to 6 possible vertices
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################
# 2) Dataset & Parsing
##############################################################################
def parse_vertices_str(vert_str):
    """
    Parse semicolon-delimited "x,y" pairs; keep only x>=0,y>=0.
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
    Reads CSV with columns:
      - c in [0,1], or NaN => unknown (we store as -1).
      - R@... => 100 reflection columns (assuming exactly 100).
      - vertices_str => "x1,y1;x2,y2;..."
    We store up to MAX_VERTS vertices => each has presence + (x,y).
    """
    def __init__(self, csv_path, max_verts=6):
        super().__init__()
        self.max_verts = max_verts

        df = pd.read_csv(csv_path)
        # find reflection columns
        r_cols = [c for c in df.columns if c.startswith("R@")]
        r_cols = sorted(r_cols, key=lambda x: float(x.split('@')[1]))

        self.X_seq   = []
        self.y_c     = []
        self.y_vpres = []  # shape [max_verts]
        self.y_vxy   = []  # shape [max_verts,2]

        for _, row in df.iterrows():
            # get reflection => shape(100,)
            r_vals = row[r_cols].values.astype(np.float32)
            if len(r_vals)!=100:
                continue

            # get c => if nan => -1
            c_val = row["c"] if "c" in df.columns else np.nan
            if pd.isna(c_val):
                c_val = -1.0

            # parse vertices
            pts = parse_vertices_str(row["vertices_str"] if "vertices_str" in df.columns else "")
            # Sort them by angle if you like, but we’ll just keep them as is
            # up to max_verts:
            pts = pts[:max_verts]

            # presence & location
            v_pres = np.zeros((max_verts,), dtype=np.float32)
            v_xy   = np.zeros((max_verts,2), dtype=np.float32)
            for i,(xx,yy) in enumerate(pts):
                v_pres[i] = 1.0
                v_xy[i,0] = xx
                v_xy[i,1] = yy

            self.X_seq.append(r_vals)   # shape(100,)
            self.y_c.append(c_val)
            self.y_vpres.append(v_pres)
            self.y_vxy.append(v_xy)

        self.X_seq   = np.stack(self.X_seq,   axis=0)   # [N,100]
        self.y_c     = np.array(self.y_c,     dtype=np.float32) # [N]
        self.y_vpres = np.stack(self.y_vpres, axis=0)   # [N,max_verts]
        self.y_vxy   = np.stack(self.y_vxy,   axis=0)   # [N,max_verts,2]

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        x_seq = self.X_seq[idx]                     # shape(100,)
        c_val = self.y_c[idx]                       # float
        vp    = self.y_vpres[idx]                   # shape(max_verts,)
        vxy   = self.y_vxy[idx]                     # shape(max_verts,2)
        return {
            "input_seq": torch.from_numpy(x_seq),
            "c_val":     torch.tensor(c_val),
            "v_pres":    torch.from_numpy(vp),
            "v_xy":      torch.from_numpy(vxy),
        }

##############################################################################
# 3) Model (2-layer LSTM => MLP => c + (presence + (x,y)) * max_verts)
##############################################################################
class LSTMModel(nn.Module):
    """
    2-layer LSTM, hidden size=128 => final hidden => MLP => outputs:
      - c (1)
      - for each vertex i in [0..max_verts-1]:
          presence bit p_i
          location (x_i,y_i)
    total dimension => 1 + max_verts*(1+2) = 1 + max_verts*3
    We'll do presence => sigmoid, location => softplus, c => sigmoid
    """
    def __init__(self, max_verts=6, hidden_size=128, num_layers=2):
        super().__init__()
        self.max_verts = max_verts
        self.hidden_size = hidden_size

        # LSTM
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        # MLP output
        out_dim = 1 + max_verts*3  # c + presence + x,y
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x_seq):
        """
        x_seq: shape [B,100]
        return c_pred: shape [B]
               vpres_pred: shape [B,max_verts]
               vxy_pred: shape [B,max_verts,2]
        """
        B = x_seq.size(0)
        inp = x_seq.unsqueeze(-1) # [B,100,1]

        out, (h_n,c_n) = self.lstm(inp)   # out => [B,100, hidden]
        # We'll take the last hidden state from dimension=1 => out[:,-1,:]
        # or equivalently h_n[-1], but let's do out[:,-1,:].
        h_final = out[:,-1,:]   # shape [B,hidden_size]

        # pass to MLP => shape [B, out_dim]
        y_out = self.mlp(h_final)

        # parse
        c_logit = y_out[:,0]  # [B]
        v_data  = y_out[:,1:] # [B, max_verts*3]
        v_data  = v_data.view(B, self.max_verts, 3)  # [B,max_verts,3]

        c_pred       = torch.sigmoid(c_logit)              # [B]
        presence_log = v_data[:,:,0]                       # [B,max_verts]
        xy_log       = v_data[:,:,1:]                      # [B,max_verts,2]

        vpres_pred   = torch.sigmoid(presence_log)         # [B,max_verts]
        vxy_pred     = F.softplus(xy_log)                  # [B,max_verts,2]

        return c_pred, vpres_pred, vxy_pred

##############################################################################
# 4) Training + Evaluate
##############################################################################
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss= 0.0
    n_samples= 0

    for batch in loader:
        x_seq = batch["input_seq"].to(DEVICE)
        c_val = batch["c_val"].to(DEVICE)
        v_pres= batch["v_pres"].to(DEVICE)
        v_xy  = batch["v_xy"].to(DEVICE)

        optimizer.zero_grad()

        c_pred, vpres_pred, vxy_pred = model(x_seq)

        # 1) c => MSE if c_val>=0
        mask_c  = (c_val>=0)
        c_known = c_val[mask_c]
        c_hat   = c_pred[mask_c]
        loss_c  = 0.0
        if c_known.numel()>0:
            loss_c = F.mse_loss(c_hat, c_known)

        # 2) presence => BCE
        #    we assume we always know the presence ground truth
        #    i.e. if v_pres[i]=1 => that vertex is indeed present
        loss_pres = F.binary_cross_entropy(vpres_pred, v_pres, reduction='none')
        # shape => [B,max_verts], average:
        loss_pres = loss_pres.mean()

        # 3) vertex coords => MSE only for those that are present in GT
        #    shape => [B,max_verts,2]
        mask_v   = (v_pres>0.5)          # [B,max_verts]
        mask_v2  = mask_v.unsqueeze(-1)  # [B,max_verts,1]
        mask_v2  = mask_v2.expand(-1,-1,2) # [B,max_verts,2]

        gt_xy    = v_xy[mask_v2]
        pred_xy  = vxy_pred[mask_v2]
        loss_xy  = 0.0
        if gt_xy.numel()>0:
            loss_xy = F.mse_loss(pred_xy, gt_xy)

        loss = loss_c + loss_pres + loss_xy
        loss.backward()
        optimizer.step()

        batch_sz= x_seq.size(0)
        total_loss += loss.item()*batch_sz
        n_samples  += batch_sz

    return total_loss / n_samples if n_samples>0 else 0.0

def eval_epoch(model, loader):
    model.eval()
    total_loss= 0.0
    n_samples= 0
    with torch.no_grad():
        for batch in loader:
            x_seq = batch["input_seq"].to(DEVICE)
            c_val = batch["c_val"].to(DEVICE)
            v_pres= batch["v_pres"].to(DEVICE)
            v_xy  = batch["v_xy"].to(DEVICE)

            c_pred, vpres_pred, vxy_pred = model(x_seq)

            # c => MSE
            mask_c= (c_val>=0)
            c_known= c_val[mask_c]
            c_hat  = c_pred[mask_c]
            loss_c = 0.0
            if c_known.numel()>0:
                loss_c = F.mse_loss(c_hat, c_known)

            # presence => BCE
            loss_pres = F.binary_cross_entropy(vpres_pred, v_pres, reduction='none')
            loss_pres = loss_pres.mean()

            # coords => MSE
            mask_v = (v_pres>0.5)
            mask_v2= mask_v.unsqueeze(-1).expand(-1,-1,2)
            gt_xy  = v_xy[mask_v2]
            pd_xy  = vxy_pred[mask_v2]
            loss_xy= 0.0
            if gt_xy.numel()>0:
                loss_xy= F.mse_loss(pd_xy, gt_xy)

            loss_batch = loss_c + loss_pres + loss_xy
            bs= x_seq.size(0)
            total_loss += loss_batch.item()*bs
            n_samples  += bs

    return total_loss/n_samples if n_samples>0 else 0.0

##############################################################################
# 5) Smooth random input & polygon plot
##############################################################################
def generate_smooth_spectrum(num_points=100):
    n_ctrl= 8
    x_ctrl= np.linspace(0,1,n_ctrl)
    y_ctrl= np.random.rand(n_ctrl)*0.8 +0.1
    spline= make_interp_spline(x_ctrl, y_ctrl, k=3)
    x_big = np.linspace(0,1,num_points)
    y_big = spline(x_big)
    y_big = np.clip(y_big, 0,1)
    return torch.tensor(y_big, dtype=torch.float)

def replicate_c4(verts):
    out_list=[]
    angles=[0, math.pi/2, math.pi, 3*math.pi/2]
    for a in angles:
        cosA= math.cos(a)
        sinA= math.sin(a)
        rot= torch.tensor([[cosA, -sinA],[sinA, cosA]], dtype=torch.float)
        chunk= verts @ rot.T
        out_list.append(chunk)
    return torch.cat(out_list, dim=0)

def angle_sort(pts):
    px= pts[:,0]
    py= pts[:,1]
    ang= torch.atan2(py, px)
    idx= torch.argsort(ang)
    return pts[idx]

def close_polygon(pts):
    if pts.size(0)>1:
        return torch.cat([pts, pts[:1]], dim=0)
    return pts

def plot_polygon(pts, c_val, out_path, title="C4 polygon"):
    pts_sorted= angle_sort(pts)
    pts_closed= close_polygon(pts_sorted)
    sx= pts_closed[:,0].numpy()
    sy= pts_closed[:,1].numpy()
    plt.figure()
    plt.fill(sx, sy, color='red', alpha=0.3)
    plt.plot(sx, sy, 'ro-')
    plt.title(f"{title}, c={c_val:.3f}")
    plt.axhline(0,color='k',lw=0.5)
    plt.axvline(0,color='k',lw=0.5)
    plt.savefig(out_path)
    plt.close()

##############################################################################
# 6) Main
##############################################################################
def main():
    ds = SupervisedDataset(CSV_PATH, max_verts=MAX_VERTS)
    N  = len(ds)
    print(f"Dataset size = {N}")

    # Train/test split => e.g. 80% train, 20% test
    n_test = int(0.2 * N)
    n_train= N - n_test
    tr_ds, te_ds = random_split(ds, [n_train, n_test],
                                generator=torch.Generator().manual_seed(42))
    print(f"Train={len(tr_ds)}, Test={len(te_ds)}")

    tr_loader= DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    te_loader= DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)

    model= LSTMModel(max_verts=MAX_VERTS, hidden_size=HIDDEN_SIZE, num_layers=LSTM_LAYERS).to(DEVICE)
    optimizer= torch.optim.Adam(model.parameters(), lr=LR)

    # train
    for epoch in range(1, NUM_EPOCHS+1):
        tr_loss= train_epoch(model, tr_loader, optimizer)
        te_loss= eval_epoch(model, te_loader)
        print(f"[Epoch {epoch}/{NUM_EPOCHS}] TrainLoss={tr_loss:.4f} TestLoss={te_loss:.4f}")

    # finalize
    os.makedirs("checkpoint_supervised", exist_ok=True)
    ckpt_path= os.path.join("checkpoint_supervised","model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print("Training done => saved model to", ckpt_path)

    # create an inference folder with time
    dt_str= datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir= f"inference_{dt_str}"
    os.makedirs(out_dir, exist_ok=True)

    # (A) Inference on random smooth
    model.eval()
    random_sp= generate_smooth_spectrum(100).to(DEVICE)  # [100]
    random_sp_in= random_sp.unsqueeze(0)                 # [1,100]
    with torch.no_grad():
        c_predA, vpresA, vxyA= model(random_sp_in)
    cA= float(c_predA[0].item())
    vpresA_np= vpresA[0].cpu().numpy()   # shape [max_verts]
    vxyA_np  = vxyA[0].cpu().numpy()     # shape [max_verts,2]

    # save random input
    np.savetxt(os.path.join(out_dir,"smooth_spectrum.txt"),
               random_sp.cpu().numpy(), fmt="%.5f")

    # pick those vertices with presence>0.5
    keepA= []
    for i in range(MAX_VERTS):
        if vpresA_np[i]>0.5:
            keepA.append(vxyA_np[i])
    if len(keepA)==0:
        keepA_t= torch.zeros((1,2))
    else:
        keepA_t= torch.tensor(keepA, dtype=torch.float)
    c4A= replicate_c4(keepA_t)
    plot_polygon(c4A, cA, os.path.join(out_dir,"smooth_polygon.png"),
                 title="Smooth polygon")

    with open(os.path.join(out_dir,"smooth_pred.txt"), "w") as f:
        f.write(f"Predicted c={cA:.3f}\n")
        for i in range(MAX_VERTS):
            f.write(f" Vertex {i}: pres={vpresA_np[i]:.3f}, "
                    f"x={vxyA_np[i,0]:.3f}, y={vxyA_np[i,1]:.3f}\n")

    plt.figure()
    plt.plot(random_sp.cpu().numpy(), 'o-', label="SmoothSpectrum")
    plt.title("Random smooth input")
    plt.ylim([0,1])
    plt.legend()
    plt.savefig(os.path.join(out_dir,"smooth_spectrum_plot.png"))
    plt.close()

    # (B) pick one sample from the test set, e.g. index=0
    if len(te_ds)>0:
        sample = te_ds[0]
        real_sp= sample["input_seq"].to(DEVICE) # [100]
        gt_c   = float(sample["c_val"].item())
        gt_vp  = sample["v_pres"].numpy()
        gt_xy  = sample["v_xy"].numpy()

        with torch.no_grad():
            c_predB, vpresB, vxyB= model(real_sp.unsqueeze(0))
        cB= float(c_predB[0].item())
        presB_np= vpresB[0].cpu().numpy()
        xyB_np  = vxyB[0].cpu().numpy()

        # store
        with open(os.path.join(out_dir,"test_pred.txt"),"w") as f:
            f.write(f"GroundTruth c={gt_c:.3f}, Pred c={cB:.3f}\n\n")
            f.write("GT vertices:\n")
            for i in range(MAX_VERTS):
                if gt_vp[i]>0.5:
                    f.write(f" i={i}, x={gt_xy[i,0]:.3f}, y={gt_xy[i,1]:.3f}\n")
            f.write("\nPred vertices:\n")
            for i in range(MAX_VERTS):
                f.write(f" i={i}: pres={presB_np[i]:.3f}, "
                        f"x={xyB_np[i,0]:.3f}, y={xyB_np[i,1]:.3f}\n")

        # polygon
        keepB= []
        for i in range(MAX_VERTS):
            if presB_np[i]>0.5:
                keepB.append(xyB_np[i])
        if len(keepB)==0:
            keepB_t= torch.zeros((1,2))
        else:
            keepB_t= torch.tensor(keepB, dtype=torch.float)
        c4B= replicate_c4(keepB_t)
        plot_polygon(c4B, cB,
                     os.path.join(out_dir,"test_polygon.png"),
                     title="Test sample polygon")

        # GT polygon
        keep_gt= []
        for i in range(MAX_VERTS):
            if gt_vp[i]>0.5:
                keep_gt.append(gt_xy[i])
        if len(keep_gt)==0:
            keep_gt_t= torch.zeros((1,2))
        else:
            keep_gt_t= torch.tensor(keep_gt, dtype=torch.float)
        c4_gt= replicate_c4(keep_gt_t)
        plot_polygon(c4_gt, gt_c if gt_c>=0 else 999.0,
                     os.path.join(out_dir,"test_polygon_gt.png"),
                     title="Test sample GT polygon")

        # also plot the reflection
        plt.figure()
        plt.plot(real_sp.cpu().numpy(), 'o-', label="Test sample spectrum")
        plt.ylim([0,1])
        plt.title("Test sample reflection")
        plt.legend()
        plt.savefig(os.path.join(out_dir,"test_spectrum.png"))
        plt.close()

    print(f"[INFO] Inference results saved to {out_dir}/.")

if __name__=="__main__":
    main()
