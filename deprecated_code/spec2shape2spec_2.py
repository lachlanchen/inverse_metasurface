#!/usr/bin/env python3

"""
spec2shape2spec_varlen_big.py

A bigger version of the spec->shape->spec pipeline with presence-chaining:

1) Q1ShiftedSpectraDataset => same SHIFT->Q1->UpTo4
2) Spectra2ShapeVarLen => presence-chaining => 1..4 points
   - d_model=256, nhead=8, num_layers=4 => bigger Transformer aggregator
3) ShapeToSpectraModel => also bigger => d_model=256, nhead=8, num_layers=4
4) Combined => Spec2Shape2Spec => shape MSE + spectra MSE
5) We train for 500 epochs with batch_size=4096 by default

Note: This bigger model consumes more GPU memory/time, but may yield better performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from datetime import datetime

###############################################################################
# 1) Dataset
###############################################################################
class Q1ShiftedSpectraDataset(Dataset):
    """
    SHIFT->Q1->UpTo4 => (spectra(11,100), shape(4,3))
    """
    def __init__(self, csv_file, max_points=4):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.r_cols = [c for c in self.df.columns if c.startswith("R@")]
        if len(self.r_cols)==0:
            raise ValueError("No reflectance columns found (R@...).")

        self.df["shape_uid"] = (
            self.df["prefix"].astype(str)
            + "_" + self.df["nQ"].astype(str)
            + "_" + self.df["nS"].astype(str)
            + "_" + self.df["shape_idx"].astype(str)
        )

        self.data_list=[]
        grouped = self.df.groupby("shape_uid", sort=False)
        for uid, grp in grouped:
            if len(grp)!=11:
                continue
            grp_sorted= grp.sort_values(by="c")
            # parse (11,100)
            spectra_11x100= grp_sorted[self.r_cols].values.astype(np.float32)

            # parse shape from first row
            first_row= grp_sorted.iloc[0]
            v_str= str(first_row.get("vertices_str","")).strip()
            if not v_str:
                continue
            raw_pairs= v_str.split(";")
            all_xy=[]
            for pair in raw_pairs:
                pair= pair.strip()
                if pair:
                    xy= pair.split(",")
                    if len(xy)==2:
                        x_val,y_val= float(xy[0]), float(xy[1])
                        all_xy.append([x_val,y_val])
            all_xy= np.array(all_xy,dtype=np.float32)
            if len(all_xy)==0:
                continue
            # SHIFT => minus(0.5,0.5)
            shifted= all_xy-0.5
            q1=[]
            for (xx,yy) in shifted:
                if xx>0 and yy>0:
                    q1.append([xx,yy])
            q1= np.array(q1,dtype=np.float32)
            n_q1= len(q1)
            if n_q1<1 or n_q1>max_points:
                continue

            shape_4x3= np.zeros((max_points,3),dtype=np.float32)
            for i in range(n_q1):
                shape_4x3[i,0]= 1.0
                shape_4x3[i,1]= q1[i,0]
                shape_4x3[i,2]= q1[i,1]

            self.data_list.append({
                "uid": uid,
                "spectra": spectra_11x100,
                "shape": shape_4x3
            })

        self.data_len= len(self.data_list)
        if self.data_len==0:
            raise ValueError("No valid SHIFT->Q1->UpTo4 shapes found.")

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        it= self.data_list[idx]
        return (it["spectra"], it["shape"], it["uid"])


###############################################################################
# 2) Sub-model: presence-chaining
###############################################################################
class Spectra2ShapeVarLen(nn.Module):
    """
    (B,11,100)->(B,4,3)
    bigger: d_model=256, nhead=8, num_layers=4
    presence chaining => forced presence=1 for i=0, subsequent presence[i]= presence[i]*presence[i-1].
    x,y => in [0,1].
    """
    def __init__(self, d_in=100, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.input_proj= nn.Linear(d_in,d_model)

        enc_layer= nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.encoder= nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.mlp= nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 12) # => presence(4)+ x,y(4)= 12
        )

    def forward(self, spec_in):
        bsz= spec_in.size(0)
        x_proj= self.input_proj(spec_in)  # (B,11,d_model)
        x_enc= self.encoder(x_proj)       # (B,11,d_model)
        x_agg= x_enc.mean(dim=1)         # (B,d_model)

        out_12= self.mlp(x_agg)          # (B,12)
        out_4x3= out_12.view(bsz,4,3)    # presence + x + y

        presence_logits= out_4x3[:,:,0]  # (B,4)
        xy_raw         = out_4x3[:,:,1:] # (B,4,2)

        # presence chaining
        presence_list= []
        for i in range(4):
            if i==0:
                # forced=1
                presence_list.append(torch.ones(bsz,device=presence_logits.device,dtype=torch.float32))
            else:
                prob_i= torch.sigmoid(presence_logits[:,i]).clamp(1e-6,1-1e-6)
                prob_i_chained= prob_i * presence_list[i-1]
                presence_ste= (prob_i_chained>0.5).float() + prob_i_chained - prob_i_chained.detach()
                presence_list.append(presence_ste)
        presence_stack= torch.stack(presence_list,dim=1) # (B,4)

        # x,y => in [0,1]
        xy_bounded= torch.sigmoid(xy_raw)
        xy_final= xy_bounded* presence_stack.unsqueeze(-1)

        final_shape= torch.cat([presence_stack.unsqueeze(-1), xy_final], dim=-1) # (B,4,3)
        return final_shape


class ShapeToSpectraModel(nn.Module):
    """
    bigger shape->spectra => d_model=256, nhead=8, num_layers=4
    """
    def __init__(self, d_in=3, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.input_proj= nn.Linear(d_in,d_model)
        enc_layer= nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.encoder= nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.mlp= nn.Sequential(
            nn.Linear(d_model,d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4,d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, 11*100)
        )

    def forward(self, shape_4x3):
        bsz= shape_4x3.size(0)
        presence= shape_4x3[:,:,0]
        key_padding_mask= (presence<0.5)

        x_proj= self.input_proj(shape_4x3)  # (B,4,d_model)
        x_enc= self.encoder(x_proj, src_key_padding_mask=key_padding_mask)

        pres_sum= presence.sum(dim=1,keepdim=True)+1e-8
        x_enc_weighted= x_enc* presence.unsqueeze(-1)
        shape_emb= x_enc_weighted.sum(dim=1)/ pres_sum

        out_flat= self.mlp(shape_emb)   # =>(B,1100)
        out_2d= out_flat.view(bsz,11,100)
        return out_2d


###############################################################################
# 3) Combined => spec->shape->spec
###############################################################################
class Spec2Shape2Spec(nn.Module):
    """
    bigger versions => d_model=256, nhead=8, num_layers=4
    """
    def __init__(self):
        super().__init__()
        self.spec2shape= Spectra2ShapeVarLen(
            d_in=100, d_model=256, nhead=8, num_layers=4
        )
        self.shape2spec= ShapeToSpectraModel(
            d_in=3, d_model=256, nhead=8, num_layers=4
        )

    def forward(self, spectra_11x100):
        shape_pred= self.spec2shape(spectra_11x100)
        spec_recon= self.shape2spec(shape_pred)
        return shape_pred, spec_recon

###############################################################################
# Visualization
###############################################################################
def replicate_c4(points):
    c4=[]
    for (x,y) in points:
        c4.append([ x,  y])
        c4.append([-y,  x])
        c4.append([-x, -y])
        c4.append([ y, -x])
    return np.array(c4,dtype=np.float32)

def sort_points_by_angle(points):
    if len(points)<3:
        return points
    cx, cy= points.mean(axis=0)
    angles= np.arctan2(points[:,1]-cy, points[:,0]-cx)
    idx= np.argsort(angles)
    return points[idx]

def plot_polygon(ax, points, color='green', alpha=0.4, fill=True):
    import matplotlib.patches as patches
    from matplotlib.path import Path
    if len(points)<3:
        ax.scatter(points[:,0], points[:,1], c=color)
        return
    closed= np.concatenate([points, points[:1]], axis=0)
    codes= [Path.MOVETO]+ [Path.LINETO]*(len(points)-1)+ [Path.CLOSEPOLY]
    path= Path(closed, codes)
    patch= patches.PathPatch(path, facecolor=color if fill else 'none',
                             alpha=alpha, edgecolor=color)
    ax.add_patch(patch)
    ax.autoscale_view()

def plot_spectra(ax, spectra_11x100, pred_spectra=None, color_gt='blue', color_pred='red'):
    for row in spectra_11x100:
        ax.plot(row, color=color_gt, alpha=0.5)
    if pred_spectra is not None:
        for row in pred_spectra:
            ax.plot(row, color=color_pred, alpha=0.5, linestyle='--')
    ax.set_xlabel("Wavelength index")
    ax.set_ylabel("Reflectance")

def visualize_3col(
    model, ds_val, device, out_dir=".", sample_count=4, seed=123
):
    import random
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    if len(ds_val)==0:
        print("[Warn] empty val set => skip visualize.")
        return

    idx_samples= random.sample(range(len(ds_val)), min(sample_count,len(ds_val)))
    n_rows= len(idx_samples)
    fig, axes= plt.subplots(n_rows,3, figsize=(12,3*n_rows))
    if n_rows==1:
        axes=[axes]

    model.eval()
    with torch.no_grad():
        for i, idx_ in enumerate(idx_samples):
            spec_np, shape_gt, uid_= ds_val[idx_]
            # left => input spectra
            axL= axes[i][0]
            plot_spectra(axL, spec_np, None, 'blue','red')
            axL.set_title(f"UID={uid_} (Left) Input Spectra")

            # forward => shape_pd, spec_rc
            spec_t= torch.tensor(spec_np, dtype=torch.float32, device=device).unsqueeze(0)
            shape_pd, spec_rc= model(spec_t)
            shape_pd= shape_pd.cpu().numpy()[0]
            spec_rc= spec_rc.cpu().numpy()[0]

            # middle => GT shape vs predicted shape
            axM= axes[i][1]
            pres_gt= (shape_gt[:,0]>0.5)
            q1_gt= shape_gt[pres_gt,1:3]
            if len(q1_gt)>0:
                c4_gt= replicate_c4(q1_gt)
                c4_gt_sorted= sort_points_by_angle(c4_gt)
                plot_polygon(axM, c4_gt_sorted, color='green', alpha=0.4, fill=True)

            pres_pd= (shape_pd[:,0]>0.5)
            q1_pd= shape_pd[pres_pd,1:3]
            if len(q1_pd)>0:
                c4_pd= replicate_c4(q1_pd)
                c4_pd_sorted= sort_points_by_angle(c4_pd)
                plot_polygon(axM, c4_pd_sorted, color='red', alpha=0.3, fill=False)

            axM.set_xlim([-0.5,0.5])
            axM.set_ylim([-0.5,0.5])
            axM.set_aspect("equal","box")
            axM.grid(True)
            axM.set_title("(Middle) GT shape(green) vs Pred shape(red)")

            # right => GT vs recon
            axR= axes[i][2]
            plot_spectra(axR, spec_np, pred_spectra=spec_rc, color_gt='blue', color_pred='red')
            axR.set_title("(Right) GT vs Recon Spectra")

    plt.tight_layout()
    out_fig= os.path.join(out_dir,"samples_3col_plot.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"[Visualization saved] => {out_fig}")


###############################################################################
# 4) Main training function
###############################################################################
def train_spec2shape2spec_varlen_big(
    csv_file,
    out_dir="outputs_spec2shape2spec_varlen_big",
    num_epochs=500,
    batch_size=4096,
    lr=1e-4,
    weight_decay=1e-5,
    split_ratio=0.8
):
    """
    A bigger version:
     - d_model=256, nhead=8, num_layers=4
    for both sub-models => more capacity
    Loss = shape MSE + spec MSE
    """
    timestamp= datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir= f"{out_dir}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    ds_full= Q1ShiftedSpectraDataset(csv_file, max_points=4)
    ds_len= len(ds_full)
    trn_len= int(ds_len*split_ratio)
    val_len= ds_len- trn_len
    ds_train, ds_val= random_split(ds_full, [trn_len, val_len])
    print(f"[DATA] total={ds_len}, train={trn_len}, val={val_len}")

    train_loader= DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader=   DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False)

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model= Spec2Shape2Spec()  # uses bigger dimension=256, nhead=8, num_layers=4
    model.to(device)

    optimizer= torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    crit_shape= nn.MSELoss()
    crit_spec=  nn.MSELoss()

    train_losses, val_losses= [], []
    for epoch in range(num_epochs):
        model.train()
        run_loss= 0.0
        for (spec_np, shape_np, uid_list) in train_loader:
            spec_t= spec_np.to(device)
            shape_t= shape_np.to(device)

            shape_pd, spec_rc= model(spec_t)
            l_shape= crit_shape(shape_pd, shape_t)
            l_spec=  crit_spec(spec_rc, spec_t)
            loss_total= l_shape+ l_spec

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            run_loss+= loss_total.item()

        avg_trn= run_loss/ len(train_loader)
        train_losses.append(avg_trn)

        # val
        model.eval()
        val_sum=0.0
        val_count=0
        with torch.no_grad():
            for (spec_np, shape_np, uid_list) in val_loader:
                bsz_= spec_np.size(0)
                st= spec_np.to(device)
                sh= shape_np.to(device)
                shp_pd, sp_rc= model(st)
                vsh= crit_shape(shp_pd, sh)
                vsp= crit_spec(sp_rc, st)
                vtot= (vsh+vsp)*bsz_
                val_sum+= vtot.item()
                val_count+=bsz_
        avg_val= val_sum/val_count
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if (epoch+1)%20==0 or epoch==0 or epoch==(num_epochs-1):
            print(f"Epoch[{epoch+1}/{num_epochs}] => TrainLoss={avg_trn:.4f}, ValLoss={avg_val:.4f}")

    # final curve
    plt.figure()
    plt.plot(train_losses,label='train')
    plt.plot(val_losses,label='val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (shape+spec MSE)")
    plt.title("Spec->Shape->Spec (VarLen) BIG model")
    plt.legend()
    plt.savefig(os.path.join(out_dir,"training_curve.png"))
    plt.close()

    # final val
    model.eval()
    val_sum=0.0
    val_count=0
    with torch.no_grad():
        for (spec_np, shape_np, uid_list) in val_loader:
            bsz_= spec_np.size(0)
            st= spec_np.to(device)
            sh= shape_np.to(device)
            shp_pd, sp_rc= model(st)
            vsh= crit_shape(shp_pd, sh)
            vsp= crit_spec(sp_rc, st)
            val_sum+= (vsh+vsp).item()*bsz_
            val_count+= bsz_
    final_val= val_sum/val_count
    print(f"[Final Val Loss] => {final_val:.6f}")

    # save
    model_path= os.path.join(out_dir,"spec2shape2spec_varlen_big_model.pt")
    torch.save(model.state_dict(), model_path)
    print("[Model saved] =>", model_path)

    # visualize
    if val_len>0:
        visualize_3col(model, ds_val, device, out_dir=out_dir, sample_count=4, seed=42)


###############################################################################
def main():
    CSV_FILE= "merged_s4_shapes_iccpOv10kG40_seed88888.csv"
    train_spec2shape2spec_varlen_big(
        csv_file= CSV_FILE,
        out_dir= "outputs_spec2shape2spec_varlen_big",
        num_epochs=250,   # typical large # of epochs
        batch_size=4096,
        lr=1e-4,
        # weight_decay=1e-5,
        weight_decay=1,
        split_ratio=0.8
    )

if __name__=="__main__":
    main()

