#!/usr/bin/env python3
"""
spectra2shape_1to4.py

Key changes from previous version:
 - Fixed the error when we do "pres_gt = qverts_np[:,0].numpy()": 
   now we just keep them as numpy arrays directly (since qverts_np is already numpy).
 - Provided a more elegant polygon visualization that replicates the quadrant 
   points (predicted or GT) into all four quadrants and draws them as polygons.
 - Show both GT polygon (green) and predicted polygon (red).
"""

import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


###############################################################################
# 1. Utility: replicate quadrant vertices -> full polygon with C4 symmetry
###############################################################################
def replicate_c4_polygon(qpoints):
    """
    qpoints: shape(N,2) in the first quadrant.
    We replicate them into 4 quadrants. 
    For point (x,y):
       Q1 => ( x,  y)
       Q2 => (-y,  x)
       Q3 => (-x, -y)
       Q4 => ( y, -x)
    We'll do them in that order for each of the quadrant points, 
    which might not be perfectly "connected" in a standard polygon sense, 
    but it allows a quick visualization. 
    """
    all_points = []
    for (x, y) in qpoints:
        all_points.append([ x,  y])  # Q1
        all_points.append([-y,  x])  # Q2
        all_points.append([-x, -y])  # Q3
        all_points.append([ y, -x])  # Q4
    return np.array(all_points, dtype=np.float32)

def draw_polygon(ax, points, color='blue', alpha=0.3, fill=True):
    """
    A utility to draw a polygon from an array of shape(N,2).
    We'll close the loop automatically.
    """
    import matplotlib.patches as patches
    from matplotlib.path import Path

    if len(points)<2:
        # Just scatter if <2 points
        ax.scatter(points[:,0], points[:,1], c=color)
        return
    
    # close the polygon
    closed_points = np.concatenate([points, points[:1]], axis=0)
    codes = [Path.MOVETO] + [Path.LINETO]*(len(points)-1) + [Path.CLOSEPOLY]
    path = Path(closed_points, codes)
    patch = patches.PathPatch(path, 
                              facecolor=(color if fill else 'none'), 
                              edgecolor=color,
                              alpha=alpha,
                              lw=1.5)
    ax.add_patch(patch)
    ax.autoscale_view()

###############################################################################
# 2. Dataset
###############################################################################
class Spectra1to4Dataset(Dataset):
    """
    Exactly as before: group by (prefix,nQ,shape_idx), parse 4..16 vertices => quadrant_count
    ...
    """
    def __init__(self, csv_file):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.r_cols = [c for c in self.df.columns if c.startswith("R@")]
        self.df["_groupkey_"] = self.df[["prefix","nQ","shape_idx"]].astype(str).agg("_".join, axis=1)
        self.data_list = []
        grouped = self.df.groupby("_groupkey_")

        for gkey, grp in grouped:
            if len(grp)!=11:
                continue
            grp_sorted = grp.sort_values("c")
            spectra_2d = grp_sorted[self.r_cols].values.astype(np.float32)

            row0 = grp_sorted.iloc[0]
            v_str = str(row0.get("vertices_str","")).strip()
            if not v_str:
                continue
            all_verts = []
            for pair in v_str.split(";"):
                pair=pair.strip()
                if pair:
                    xy=pair.split(",")
                    if len(xy)==2:
                        xval=float(xy[0])
                        yval=float(xy[1])
                        all_verts.append([xval,yval])
            all_verts = np.array(all_verts, dtype=np.float32)
            total_v = all_verts.shape[0]
            if total_v not in [4,8,12,16]:
                continue
            q_count = total_v//4
            q1_verts = all_verts[:q_count]

            out_array = np.zeros((4,3), dtype=np.float32)
            for i in range(q_count):
                out_array[i,0] = 1.0
                out_array[i,1] = q1_verts[i,0]
                out_array[i,2] = q1_verts[i,1]

            self.data_list.append({
                "gkey": gkey,
                "spectra": spectra_2d,
                "qverts": out_array,
                "full_verts": all_verts  # store original for optional GT polygon if needed
            })
        self.data_len = len(self.data_list)
        if self.data_len==0:
            raise ValueError("No valid shapes found. Check CSV or grouping logic")

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        it = self.data_list[idx]
        return it["spectra"], it["qverts"], it["gkey"], it["full_verts"]

###############################################################################
# 3. Model
###############################################################################
class DeepSetsEncoder(nn.Module):
    def __init__(self, d_in=100, d_model=128):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.post_agg = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
    def forward(self,x):
        # x: (bsz,11,100)
        x_emb = self.embed(x)      # (bsz,11,d_model)
        x_sum = x_emb.sum(dim=1)   # (bsz,d_model)
        out   = self.post_agg(x_sum)
        return out

class SpectraToShape1to4(nn.Module):
    def __init__(self, d_in=100, d_model=128):
        super().__init__()
        self.encoder = DeepSetsEncoder(d_in, d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.ReLU(),
            nn.Linear(d_model,4*3)
        )
    def forward(self, x):
        bsz = x.size(0)
        lat = self.encoder(x)
        out_flat = self.decoder(lat)          # (bsz,12)
        out_3d   = out_flat.view(bsz,4,3)     # (bsz,4,3)
        # interpret presence + x + y
        pres_logit = out_3d[:,:,0]
        pres = torch.sigmoid(pres_logit)
        xy_raw = out_3d[:,:,1:]
        xy     = torch.sigmoid(xy_raw)
        return torch.cat([pres.unsqueeze(-1),xy],dim=-1)

def shape_l1_loss(pred, target):
    return F.l1_loss(pred,target,reduction='mean')

def geometric_penalty(pred, alpha=0.7):
    # pred: (bsz,4,3)
    # presence => pred[:,:,0]
    presence = pred[:,:,0]
    bsz,_ = presence.shape
    i_idx = torch.arange(4, device=presence.device).unsqueeze(0)
    weight= alpha**i_idx
    pen_samp = (presence*weight).sum(dim=1)
    return pen_samp.mean()

###############################################################################
# 4. Training + Visualization
###############################################################################
def train_spectra2shape_1to4(
    CSV_FILE,
    out_dir="outputs_spectra1to4",
    num_epochs=100,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-5,
    alpha_geom=0.01,
    use_l1=True,
    split_ratio=0.8
):
    os.makedirs(out_dir, exist_ok=True)
    ds_full = Spectra1to4Dataset(CSV_FILE)
    ds_len = len(ds_full)
    trn_len = int(ds_len*split_ratio)
    val_len = ds_len - trn_len
    ds_train, ds_val = random_split(ds_full, [trn_len, val_len])
    print(f"[Data] total={ds_len}, train={trn_len}, val={val_len}")

    train_loader = DataLoader(ds_train,batch_size=batch_size,shuffle=True,drop_last=True)
    val_loader   = DataLoader(ds_val,batch_size=batch_size,shuffle=False,drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:",device)

    model = SpectraToShape1to4(d_in=100,d_model=128).to(device)
    optimizer= torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=10,verbose=True)

    if use_l1:
        base_loss_fn = shape_l1_loss
        loss_name="L1"
    else:
        base_loss_fn = lambda p,t: F.mse_loss(p,t,reduction='mean')
        loss_name="MSE"

    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        total_loss=0.0
        for (spectra_np,qverts_np,gkeys,fullv) in train_loader:
            spec_t = spectra_np.to(device)
            tgt_t  = qverts_np.to(device)
            pred_t = model(spec_t)
            base_l = base_loss_fn(pred_t,tgt_t)
            if alpha_geom>0:
                pen_l  = geometric_penalty(pred_t,0.7)
                loss   = base_l + alpha_geom*pen_l
            else:
                loss   = base_l

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        avg_trn= total_loss/len(train_loader)
        train_losses.append(avg_trn)

        # val
        model.eval()
        val_sum=0.0
        val_count=0
        with torch.no_grad():
            for (spectra_np,qverts_np,gkeys,fullv) in val_loader:
                bsz_  = spectra_np.size(0)
                spec_t= spectra_np.to(device)
                tgt_t = qverts_np.to(device)
                pred_t= model(spec_t)
                base_l= base_loss_fn(pred_t,tgt_t)
                if alpha_geom>0:
                    pen_l= geometric_penalty(pred_t,0.7)
                    vloss= (base_l + alpha_geom*pen_l)*bsz_
                else:
                    vloss= base_l*bsz_
                val_sum += vloss.item()
                val_count+=bsz_
        avg_val = val_sum/val_count
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if (epoch+1)%20==0 or epoch==0:
            print(f"Epoch[{epoch+1}/{num_epochs}] {loss_name}={avg_trn:.4f} Val={avg_val:.4f}")

    print(f"[Val] final {loss_name} loss: {avg_val:.6f}")

    # save curve
    plt.figure()
    plt.plot(train_losses,label="train")
    plt.plot(val_losses,label="val")
    plt.xlabel("Epoch")
    plt.ylabel(f"{loss_name} Loss")
    plt.title("Spectra->Shape(1to4) training")
    plt.legend()
    plt.savefig(os.path.join(out_dir,"train_curve.png"))
    plt.close()

    # save model
    model_path = os.path.join(out_dir,"spectra2shape_1to4.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Final visualize
    visualize_some(model, ds_val, device, out_dir)

def visualize_some(model, ds_val, device, out_dir, num_samples=4):
    """
    We'll pick a few random samples from ds_val.
    We'll replicate the GT quadrant and predicted quadrant into 4 quadrants 
    to see the full shape polygons:
      - GT polygon => green
      - Pred polygon => red
    """
    import random
    random_idxs = random.sample(range(len(ds_val)), min(num_samples,len(ds_val)))
    model.eval()

    fig, axes = plt.subplots(num_samples,2, figsize=(10,5*num_samples))
    if num_samples==1:
        axes=[axes]
    with torch.no_grad():
        for row_i, idx_ in enumerate(random_idxs):
            spectra_np, qverts_np, gkey, fullv = ds_val[idx_]
            spec_t = torch.tensor(spectra_np, dtype=torch.float32, device=device).unsqueeze(0)
            pred = model(spec_t).cpu().numpy()[0]  # shape(4,3)

            # Qverts => shape(4,3) => presence, x,y
            # GT
            pres_gt = qverts_np[:,0]
            xy_gt   = qverts_np[:,1:]
            # filter out any that has presence>0.5
            Q1_gt = []
            for i in range(4):
                if pres_gt[i]>0.5:
                    Q1_gt.append([xy_gt[i,0],xy_gt[i,1]])
            Q1_gt = np.array(Q1_gt,dtype=np.float32)

            # Pred
            pres_pd= pred[:,0]
            xy_pd  = pred[:,1:]
            Q1_pd  = []
            for i in range(4):
                if pres_pd[i]>0.5:
                    Q1_pd.append([xy_pd[i,0],xy_pd[i,1]])
            Q1_pd = np.array(Q1_pd,dtype=np.float32)

            # replicate to 4 quadrants
            full_gt_polygon = None
            if len(Q1_gt)>0:
                full_gt_polygon = replicate_c4_polygon(Q1_gt)
            full_pd_polygon = None
            if len(Q1_pd)>0:
                full_pd_polygon = replicate_c4_polygon(Q1_pd)

            # left sub-plot => reflectances
            ax_left  = axes[row_i][0]
            for i in range(spectra_np.shape[0]):
                ax_left.plot(spectra_np[i], alpha=0.5)
            ax_left.set_title(f"{gkey}\nInput Spectra(11×100)")
            ax_left.set_xlabel("Wavelength index")
            ax_left.set_ylabel("Reflectance")

            # right sub-plot => polygons
            ax_right = axes[row_i][1]
            ax_right.set_title("GT(green) vs Pred(red)")
            ax_right.set_aspect('equal','box')
            ax_right.grid(True)

            # draw GT polygon
            if full_gt_polygon is not None and len(full_gt_polygon)>=2:
                draw_polygon(ax_right, full_gt_polygon, color='green', alpha=0.3, fill=True)
            # draw Pred polygon
            if full_pd_polygon is not None and len(full_pd_polygon)>=2:
                draw_polygon(ax_right, full_pd_polygon, color='red', alpha=0.3, fill=True)

    plt.tight_layout()
    out_fig= os.path.join(out_dir, "sample_polygons.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"[Visualization] Saved sample polygons to {out_fig}")


def main():
    CSV_FILE = "merged_s4_shapes_iccpOv10kG40_seed88888.csv"  # adjust as needed
    train_spectra2shape_1to4(
        CSV_FILE=CSV_FILE,
        out_dir="outputs_spectra1to4",
        num_epochs=100,
        batch_size=32,
        lr=1e-4,
        weight_decay=1e-5,
        alpha_geom=0.01,
        use_l1=True,
        split_ratio=0.8
    )

if __name__=="__main__":
    main()

