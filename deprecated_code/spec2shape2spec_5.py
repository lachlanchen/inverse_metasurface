#!/usr/bin/env python3

"""
spec2shape2spec_varlen_enhanced.py

A spec->shape->spec pipeline with presence chaining for variable 1..4 vertices.

**Enhanced training objective**:
  Let shape_pred, spec_pred = model(spectra_input).
  Also let spec_recon_gtShape = model.shape2spec(gt_shape).
  Then total loss = 
    MSE(shape_pred, shape_gt) 
    + MSE(spec_recon_gtShape, spectra_input)  # shape2spec on GT shape
    + MSE(spec_pred, spectra_input)           # chain recon

Visualization (3 columns):
  - Left: GT spectrum (blue)
  - Middle: GT shape(green) vs predicted shape(red)
  - Right:  Original(blue), shape2spec(predicted shape => red dashed),
            shape2spec(gt shape => green dashed)

Author: ChatGPT
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
    SHIFT->Q1->UpTo4 => (spectra(11,100), shape(4,3)), presence + x + y
    If #Q1<1 or #Q1>4 => skip
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

            # input => (11,100)
            spec_11x100= grp_sorted[self.r_cols].values.astype(np.float32)

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
            shifted= all_xy- 0.5
            q1=[]
            for (xx,yy) in shifted:
                if xx>0 and yy>0:
                    q1.append([xx,yy])
            q1= np.array(q1,dtype=np.float32)
            n_q1= len(q1)
            if n_q1<1 or n_q1>max_points:
                continue

            shape_4x3= np.zeros((max_points,3), dtype=np.float32)
            for i in range(n_q1):
                shape_4x3[i,0]= 1.0
                shape_4x3[i,1]= q1[i,0]
                shape_4x3[i,2]= q1[i,1]

            self.data_list.append({
                "uid": uid,
                "spectra": spec_11x100,
                "shape": shape_4x3
            })

        self.data_len= len(self.data_list)
        if self.data_len==0:
            raise ValueError("No valid shapes => SHIFT->Q1->UpTo4")

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        it= self.data_list[idx]
        return (it["spectra"], it["shape"], it["uid"])


###############################################################################
# 2) Sub-model: variable-len shape (presence chaining)
###############################################################################
class Spectra2ShapeVarLen(nn.Module):
    """
    (B,11,100)->(B,4,3)
    presence chaining => forced presence[0]=1, presence[i]= presence[i]*presence[i-1], i=1..3
    uses STE => presence bits
    x,y => in [0,1].
    """
    def __init__(self, d_in=100, d_model=128, nhead=4, num_layers=2):
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
            nn.Linear(d_model, 12) # => presence(4), x,y(4)
        )

    def forward(self, spectra_11x100):
        bsz= spectra_11x100.size(0)
        # aggregator
        x_proj= self.input_proj(spectra_11x100)  # (B,11,d_model)
        x_enc= self.encoder(x_proj)              # (B,11,d_model)
        x_agg= x_enc.mean(dim=1)                 # (B,d_model)

        out_12= self.mlp(x_agg)                  # (B,12)
        out_4x3= out_12.view(bsz,4,3)            # => presence logits + x + y

        presence_logits= out_4x3[:,:,0] # (B,4)
        xy_raw         = out_4x3[:,:,1:]#(B,4,2)

        presence_list= []
        for i in range(4):
            if i==0:
                presence_list.append(torch.ones(bsz,device=presence_logits.device,dtype=torch.float32))
            else:
                prob_i= torch.sigmoid(presence_logits[:,i]).clamp(1e-6,1-1e-6)
                prob_i_chain= prob_i* presence_list[i-1]
                ste_i= (prob_i_chain>0.5).float() + prob_i_chain - prob_i_chain.detach()
                presence_list.append(ste_i)
        presence_stack= torch.stack(presence_list,dim=1) # (B,4)

        # x,y => in [0,1]
        xy_bounded= torch.sigmoid(xy_raw)
        xy_final= xy_bounded* presence_stack.unsqueeze(-1)

        final_shape= torch.cat([presence_stack.unsqueeze(-1), xy_final], dim=-1) # (B,4,3)
        return final_shape


class ShapeToSpectraModel(nn.Module):
    """
    shape(4,3)->(11,100). presence => mask
    Weighted sum => final => mlp => (11,100)
    """
    def __init__(self, d_in=3, d_model=128, nhead=4, num_layers=2):
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

        x_proj= self.input_proj(shape_4x3)
        x_enc= self.encoder(x_proj, src_key_padding_mask=key_padding_mask)

        pres_sum= presence.sum(dim=1,keepdim=True)+1e-8
        x_enc_w= x_enc* presence.unsqueeze(-1)
        shape_emb= x_enc_w.sum(dim=1)/ pres_sum

        out_flat= self.mlp(shape_emb)
        out_2d= out_flat.view(bsz,11,100)
        return out_2d

###############################################################################
# 3) Combined => spec->shape->spec
###############################################################################
class Spec2Shape2Spec(nn.Module):
    """
    sub-model spec->shape var-len
    sub-model shape->spec
    """
    def __init__(self,
                 d_in_spec=100, d_model_spec2shape=128, nhead_s2shape=4, num_layers_s2shape=2,
                 d_in_shape=3,  d_model_shape2spec=128, nhead_shape2spec=4, num_layers_shape2spec=2):
        super().__init__()
        self.spec2shape= Spectra2ShapeVarLen(
            d_in= d_in_spec,
            d_model= d_model_spec2shape,
            nhead= nhead_s2shape,
            num_layers= num_layers_s2shape
        )
        self.shape2spec= ShapeToSpectraModel(
            d_in= d_in_shape,
            d_model= d_model_shape2spec,
            nhead= nhead_shape2spec,
            num_layers= num_layers_shape2spec
        )

    def forward(self, spectra_11x100):
        """
        Usually we do:
          shape_pred, spec_recon = model(spectra)
        """
        shape_pred= self.spec2shape(spectra_11x100)
        spec_recon= self.shape2spec(shape_pred)
        return shape_pred, spec_recon

    def shape2spec_gt(self, shape_gt):
        """
        Utility to run shape->spec with ground-truth shape.
        """
        return self.shape2spec(shape_gt)

###############################################################################
# 4) Visualization
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
    closed= np.concatenate([points,points[0:1]], axis=0)
    codes= [Path.MOVETO]+ [Path.LINETO]*(len(points)-1)+ [Path.CLOSEPOLY]
    path= Path(closed, codes)
    patch= patches.PathPatch(
        path, facecolor=color if fill else 'none',
        alpha=alpha, edgecolor=color
    )
    ax.add_patch(patch)
    ax.autoscale_view()

def plot_spectra_comparison(ax, spectra_gt, spectra_pd=None, color_gt='blue', color_pd='red', spectra_gtShape=None, color_gtShape='green'):
    """
    - Plot original (blue)
    - Plot recon from predicted shape (red dashed)
    - Plot recon from GT shape (green dashed)
    """
    for row in spectra_gt:
        ax.plot(row, color=color_gt, alpha=0.5)
    if spectra_gtShape is not None:
        for row in spectra_gtShape:
            ax.plot(row, color=color_gtShape, alpha=0.5, linestyle='--')
    if spectra_pd is not None:
        for row in spectra_pd:
            ax.plot(row, color=color_pd, alpha=0.5, linestyle='--')
    ax.set_xlabel("Wavelength index")
    ax.set_ylabel("Reflectance")

def visualize_3col(
    model, ds_val, device, out_dir=".",
    sample_count=4, seed=123
):
    """
    3 columns:
     - col1 => GT spectrum
     - col2 => GT shape(green) vs Pred shape(red)
     - col3 => original(blue), recon-from-pred-shape(red dashed), recon-from-GT-shape(green dashed)
    """
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
            spec_gt, shape_gt, uid_= ds_val[idx_]

            # left => original spectrum => (blue lines)
            axL= axes[i][0]
            for row_ in spec_gt:
                axL.plot(row_, color='blue', alpha=0.5)
            axL.set_title(f"UID={uid_}\n(Left) Original GT Spectrum")

            # middle => shape: GT(green) vs Pred(red)
            axM= axes[i][1]
            presG= (shape_gt[:,0]>0.5)
            q1_g= shape_gt[presG,1:3]
            if len(q1_g)>0:
                c4_g= replicate_c4(q1_g)
                c4_g_sorted= sort_points_by_angle(c4_g)
                plot_polygon(axM, c4_g_sorted, color='green', alpha=0.4, fill=True)

            # do forward => predicted shape => predicted recon
            spec_t= torch.tensor(spec_gt, dtype=torch.float32, device=device).unsqueeze(0)
            shape_pd, spec_pd = model(spec_t)
            shape_pd= shape_pd.cpu().numpy()[0]
            spec_pd= spec_pd.cpu().numpy()[0]

            presP= (shape_pd[:,0]>0.5)
            q1_p= shape_pd[presP,1:3]
            if len(q1_p)>0:
                c4_p= replicate_c4(q1_p)
                c4_p_sorted= sort_points_by_angle(c4_p)
                plot_polygon(axM, c4_p_sorted, color='red', alpha=0.3, fill=False)

            axM.set_xlim([-0.5,0.5])
            axM.set_ylim([-0.5,0.5])
            axM.set_aspect("equal","box")
            axM.grid(True)
            axM.set_title("(Middle)\nGT shape(green) vs Pred shape(red)")

            # right => original(blue), shape2spec(pred) => red dashed, shape2spec(gt) => green dashed
            axR= axes[i][2]
            # recon from gt shape
            shape_gt_t= torch.tensor(shape_gt, dtype=torch.float32, device=device).unsqueeze(0)
            spec_gtShape = model.shape2spec(shape_gt_t).cpu().numpy()[0]

            plot_spectra_comparison(
                axR,
                spec_gt,
                spectra_pd= spec_pd,
                color_gt='blue',
                color_pd='red',
                spectra_gtShape= spec_gtShape,
                color_gtShape='green'
            )
            axR.set_title("(Right)\nOriginal(blue), ReconPred(red), ReconGT(green)")

    plt.tight_layout()
    out_fig= os.path.join(out_dir, "samples_3col_plot.png")
    plt.savefig(out_fig)
    plt.close()
    print("[Visualization saved] =>", out_fig)


###############################################################################
# 5) Enhanced Training
###############################################################################
def train_spec2shape2spec_varlen(
    csv_file,
    out_dir="outputs_spec2shape2spec_varlen_enhanced",
    num_epochs=100,
    batch_size=4096,
    lr=1e-4,
    weight_decay=1e-5,
    split_ratio=0.8
):
    """
    Enhanced loss:
      shape_loss = MSE(shape_pred, shape_gt)
      shape2spec_gt_loss = MSE(shape2spec(gt_shape), spectra_gt)
      chain_loss = MSE(shape2spec(shape_pred), spectra_gt)

    total_loss = shape_loss + shape2spec_gt_loss + chain_loss
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

    model= Spec2Shape2Spec(
        d_in_spec=100,
        d_model_spec2shape=128,
        nhead_s2shape=4,
        num_layers_s2shape=2,

        d_in_shape=3,
        d_model_shape2spec=128,
        nhead_shape2spec=4,
        num_layers_shape2spec=2
    ).to(device)

    optimizer= torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    crit_mse= nn.MSELoss()

    train_losses, val_losses= [], []

    for epoch in range(num_epochs):
        model.train()
        run_loss=0.0
        for (spec_np, shape_np, uid_list) in train_loader:
            spec_t= spec_np.to(device)     # (B,11,100)
            shape_t= shape_np.to(device)   # (B,4,3)

            # 1) forward => shape_pred, spec_pred
            shape_pred, spec_pred = model(spec_t)

            # 2) shape2spec on gt shape
            spec_recon_gtShape = model.shape2spec(shape_t)

            # 3) compute losses
            loss_shape = crit_mse(shape_pred, shape_t)
            loss_gt2spec = crit_mse(spec_recon_gtShape, spec_t)
            loss_chain   = crit_mse(spec_pred, spec_t)

            loss_total = loss_shape + loss_gt2spec + loss_chain

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            run_loss+= loss_total.item()

        avg_train= run_loss/len(train_loader)
        train_losses.append(avg_train)

        # validation
        model.eval()
        val_sum=0.0
        val_count=0
        with torch.no_grad():
            for (spec_np, shape_np, uid_list) in val_loader:
                bsz_= spec_np.size(0)
                st= spec_np.to(device)
                sh= shape_np.to(device)

                # forward
                shape_pd, spec_pd = model(st)
                spec_gtShape = model.shape2spec(sh)

                # compute 3 losses
                l_shape = crit_mse(shape_pd, sh)
                l_gt2s  = crit_mse(spec_gtShape, st)
                l_chain = crit_mse(spec_pd, st)

                vtot = (l_shape + l_gt2s + l_chain)*bsz_
                val_sum+= vtot.item()
                val_count+= bsz_
        avg_val= val_sum/ val_count
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if (epoch+1)%20==0 or epoch==0 or epoch==(num_epochs-1):
            print(f"Epoch[{epoch+1}/{num_epochs}] => TrainLoss={avg_train:.4f} ValLoss={avg_val:.4f}")

    # final training curve
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses,   label='val')
    plt.xlabel("Epoch")
    plt.ylabel("TotalLoss= shape + shape2spec(GT) + chain")
    plt.title("Enhanced spec->shape->spec (varlen) Training")
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

            shape_pd, spec_pd= model(st)
            spec_gtShape= model.shape2spec(sh)

            l_shape= crit_mse(shape_pd, sh)
            l_gt2s = crit_mse(spec_gtShape, st)
            l_chain= crit_mse(spec_pd, st)

            l_tot= (l_shape + l_gt2s + l_chain)*bsz_
            val_sum+= l_tot.item()
            val_count+= bsz_
    final_val= val_sum/ val_count
    print(f"[Final Val Loss] => {final_val:.6f}")

    # save
    model_path= os.path.join(out_dir,"spec2shape2spec_varlen_enhanced_model.pt")
    torch.save(model.state_dict(), model_path)
    print("[Model saved] =>", model_path)

    # visualize => same 3-col. 
    # Right => we show original(blue), recon-from-pred-shape(red), recon-from-GT-shape(green).
    if val_len>0:
        visualize_3col(model, ds_val, device, out_dir=out_dir, sample_count=4, seed=42)


###############################################################################
def main():
    CSV_FILE= "merged_s4_shapes_iccpOv10kG40_seed88888.csv"
    train_spec2shape2spec_varlen(
        csv_file= CSV_FILE,
        out_dir= "outputs_spec2shape2spec_varlen_enhanced",
        num_epochs=1000,
        batch_size=4096,
        lr=1e-4,
        weight_decay=1e-5,
        split_ratio=0.8
    )

if __name__=="__main__":
    main()

