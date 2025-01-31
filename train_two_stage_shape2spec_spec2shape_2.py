#!/usr/bin/env python3

"""
two_stage_train.py

A two-stage pipeline to improve shape + spectrum prediction with the
visualization style you used before (polygons in the middle, multi-line
comparison on the right). We do:

STAGE A:
  1. Train shape2spec alone (500 epochs).
  2. We do a simpler 2-col visual:
       * Left: GT shape polygon
       * Right: GT spectrum(blue) vs predicted spec(red dashed)
     Because shape is input, there's no "predicted shape" difference to show.

STAGE B:
  1. Freeze shape2spec from Stage A.
  2. Train spec2shape for 500 epochs.
  3. Do the old 3-col polygon-based visualization:
     (Left) GT spectrum (blue),
     (Middle) GT shape(green) vs predicted shape(red polygon),
     (Right) original spectrum(blue), shape2spec(GT shape => green dashed),
            shape2spec(pred shape => red dashed).

All results go under "outputs_two_stage_<timestamp>/" with subfolders stageA & stageB.

By using d_model=256, num_layers=4, and training for 500 epochs, you can
further enhance shape + spectrum predictions.

Author: Lachlan (ChatGPT code)
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

import random
from datetime import datetime

###############################################################################
# Dataset
###############################################################################
class Q1ShiftedSpectraDataset(Dataset):
    """
    SHIFT->Q1->UpTo4 => (spectra(11,100), shape(4,3)).
    shape(4,3) => presence + x + y.
    If #Q1<1 or #Q1>4 => skip.
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

            spec_11x100= grp_sorted[self.r_cols].values.astype(np.float32)

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
                "spectra": spec_11x100,  # (11,100)
                "shape": shape_4x3      # (4,3)
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
# Stage A: shape->spec
###############################################################################
class ShapeToSpectraModel(nn.Module):
    """
    shape(4,3)->(11,100). presence => mask
    Weighted sum => bigger MLP => (11,100)
    We do a deeper Transformer with d_model=256, num_layers=4 for better performance.
    """
    def __init__(self, d_in=3, d_model=256, nhead=4, num_layers=4):
        super().__init__()

        self.input_proj= nn.Linear(d_in, d_model)

        enc_layer= nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.encoder= nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # bigger MLP
        self.mlp= nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model*2),
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
        x_enc_w= x_enc* presence.unsqueeze(-1)
        shape_emb= x_enc_w.sum(dim=1)/ pres_sum

        out_flat= self.mlp(shape_emb)
        out_2d= out_flat.view(bsz,11,100)
        return out_2d


def train_shape2spec_stageA(
    csv_file,
    out_folder_stageA,
    num_epochs=500,
    batch_size=1024,
    lr=1e-4,
    weight_decay=1e-5,
    split_ratio=0.8,
    grad_clip=1.0
):
    """
    Train shape->spec for 500 epochs.
    We'll produce:
      - training_curve_stageA.png
      - shape2spec_stageA.pt
      - a small 2-col visualization: GT shape polygon + GT spec vs predicted spec
    """
    os.makedirs(out_folder_stageA, exist_ok=True)
    print(f"[Stage A] => {out_folder_stageA}")

    ds_full= Q1ShiftedSpectraDataset(csv_file)
    ds_len= len(ds_full)
    trn_len= int(ds_len*split_ratio)
    val_len= ds_len- trn_len
    ds_train, ds_val= random_split(ds_full, [trn_len, val_len])
    print(f"[DATA: Stage A] total={ds_len}, train={trn_len}, val={val_len}")

    loader_train= DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_val  = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False)

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model= ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4).to(device)
    optim= torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched= torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10, verbose=True)
    crit_mse= nn.MSELoss()

    train_losses, val_losses= [], []

    for epoch in range(num_epochs):
        model.train()
        run_loss= 0.0
        for (spec_np, shape_np, uid_list) in loader_train:
            shape_t= shape_np.to(device)
            spec_gt= spec_np.to(device)

            spec_pd= model(shape_t)
            loss= crit_mse(spec_pd, spec_gt)

            optim.zero_grad()
            loss.backward()
            if grad_clip>0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            run_loss+= loss.item()

        avg_train= run_loss/ len(loader_train)
        train_losses.append(avg_train)

        # val
        model.eval()
        val_sum=0.0
        val_count=0
        with torch.no_grad():
            for (spec_np, shape_np, uid_list) in loader_val:
                bsz= shape_np.size(0)
                shape_t= shape_np.to(device)
                spec_gt= spec_np.to(device)

                spec_pd= model(shape_t)
                v= crit_mse(spec_pd, spec_gt)* bsz
                val_sum+= v.item()
                val_count+= bsz
        avg_val= val_sum/ val_count
        val_losses.append(avg_val)
        sched.step(avg_val)

        if (epoch+1)%50==0 or epoch==0 or epoch==(num_epochs-1):
            print(f"[Stage A] Epoch[{epoch+1}/{num_epochs}] => trainLoss={avg_train:.4f}, valLoss={avg_val:.4f}")

    # plot
    plt.figure()
    plt.plot(train_losses,label='train')
    plt.plot(val_losses,label='val')
    plt.xlabel("Epoch")
    plt.ylabel("MSE( shape->spec )")
    plt.title("Stage A: shape->spec training")
    plt.legend()
    plt.savefig(os.path.join(out_folder_stageA,"training_curve_stageA.png"))
    plt.close()

    final_val= val_losses[-1]
    print(f"[Stage A] final val loss => {final_val:.6f}")

    # save model
    model_path= os.path.join(out_folder_stageA, "shape2spec_stageA.pt")
    torch.save(model.state_dict(), model_path)
    print("[Stage A] model saved =>", model_path)

    # small 2-col visualization
    stageA_visualize(model, ds_val, device, out_folder_stageA)
    return model_path


###############################################################################
# Visualization for Stage A: shape->spec
###############################################################################
def replicate_c4(points):
    """
    For Q1 points => replicate them in all 4 quadrants => for polygon shape
    """
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
    patch= patches.PathPatch(path, facecolor=color if fill else 'none',
                             alpha=alpha, edgecolor=color)
    ax.add_patch(patch)
    ax.autoscale_view()

def stageA_visualize(model, ds_val, device, out_dir, sample_count=4, seed=123):
    """
    For shape->spec alone, let's do a 2-col approach:
      - Left: shape polygon (green)
      - Right: GT spectrum(blue) vs predicted spectrum (red dashed)
    We pick random items from ds_val.
    """
    import random
    random.seed(seed)
    if len(ds_val)==0:
        print("[Stage A] Val set empty => skip visualize.")
        return

    idx_samples= random.sample(range(len(ds_val)), min(sample_count,len(ds_val)))
    fig, axes= plt.subplots(len(idx_samples), 2, figsize=(8,3*len(idx_samples)))
    if len(idx_samples)==1:
        axes= [axes]

    model.eval()
    with torch.no_grad():
        for i, idx_ in enumerate(idx_samples):
            spec_gt, shape_gt, uid_ = ds_val[idx_]

            # left => shape polygon
            axL= axes[i][0]
            pres= (shape_gt[:,0]>0.5)
            q1= shape_gt[pres,1:3]
            if len(q1)>0:
                c4= replicate_c4(q1)
                c4_sorted= sort_points_by_angle(c4)
                plot_polygon(axL, c4_sorted, color='green', alpha=0.4, fill=True)
            axL.set_aspect("equal","box")
            axL.set_xlim([-0.5,0.5])
            axL.set_ylim([-0.5,0.5])
            axL.grid(True)
            axL.set_title(f"UID={uid_}\n(Left) GT shape polygon")

            # right => GT spec(blue) vs predicted spec(red dashed)
            axR= axes[i][1]
            shape_t= torch.tensor(shape_gt, dtype=torch.float32, device=device).unsqueeze(0)
            spec_pd= model(shape_t).cpu().numpy()[0]  # (11,100)

            # plot GT in blue
            for row_ in spec_gt:
                axR.plot(row_, color='blue', alpha=0.5)
            # plot predicted in red dashed
            for row_ in spec_pd:
                axR.plot(row_, color='red', alpha=0.5, linestyle='--')
            axR.set_title("(Right)\nGT spec(blue) vs Pred spec(red dashed)")
            axR.set_xlabel("Wavelength index")
            axR.set_ylabel("Reflectance")

    plt.tight_layout()
    out_fig= os.path.join(out_dir, "samples_2col_stageA.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"[Stage A visualization saved] => {out_fig}")


###############################################################################
# Stage B: spec->shape, using a FROZEN shape2spec
###############################################################################
class Spectra2ShapeVarLen(nn.Module):
    """
    (B,11,100)->(B,4,3)
    presence chaining => forced presence[0]=1, presence[i]= presence[i]*presence[i-1], i=1..3
    x,y => in [0,1].
    We do a bigger transformer with d_model=256, num_layers=4
    plus a small row MLP if we want. 
    """
    def __init__(self, d_in=100, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        # row MLP to get from 100->d_model
        self.row_preproc= nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
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
            nn.Linear(d_model, 12)  # => presence(4), x,y(4)
        )

    def forward(self, spec_11x100):
        bsz= spec_11x100.size(0)
        # row MLP
        x_reshaped= spec_11x100.view(-1, spec_11x100.size(2))  # (B*11,100)
        x_pre= self.row_preproc(x_reshaped)                    # (B*11, d_model)
        x_pre= x_pre.view(bsz, -1, x_pre.size(-1))             # (B,11,d_model)

        x_enc= self.encoder(x_pre)                             # (B,11,d_model)
        x_agg= x_enc.mean(dim=1)                               # set-based aggregator

        out_12= self.mlp(x_agg)                                # (B,12)
        out_4x3= out_12.view(bsz,4,3)

        # presence chain
        presence_logits= out_4x3[:,:,0]
        xy_raw= out_4x3[:,:,1:]

        presence_list=[]
        for i in range(4):
            if i==0:
                presence_list.append(torch.ones(bsz, device=out_4x3.device, dtype=torch.float32))
            else:
                prob_i= torch.sigmoid(presence_logits[:,i]).clamp(1e-6,1-1e-6)
                prob_i_chain= prob_i* presence_list[i-1]
                ste_i= (prob_i_chain>0.5).float() + prob_i_chain - prob_i_chain.detach()
                presence_list.append(ste_i)
        presence_stack= torch.stack(presence_list, dim=1)

        xy_bounded= torch.sigmoid(xy_raw)
        xy_final= xy_bounded* presence_stack.unsqueeze(-1)
        final_shape= torch.cat([presence_stack.unsqueeze(-1), xy_final], dim=-1)
        return final_shape


class Spec2ShapePipeline_FrozenShape2Spec(nn.Module):
    """
    Pipeline: spec->shape (trainable) => shape2spec_frozen => final spec
    We'll freeze shape2spec weights, so no grad flows there.
    """
    def __init__(self, spec2shape_net, shape2spec_frozen):
        super().__init__()
        self.spec2shape= spec2shape_net
        self.shape2spec_frozen= shape2spec_frozen
        # set requires_grad=False for shape2spec_frozen
        for p in self.shape2spec_frozen.parameters():
            p.requires_grad=False

    def forward(self, spec_11x100):
        shape_pred= self.spec2shape(spec_11x100)
        with torch.no_grad():
            spec_chain= self.shape2spec_frozen(shape_pred)
        return shape_pred, spec_chain

def train_spec2shape_stageB(
    csv_file,
    shape2spec_ckpt,
    out_folder_stageB,
    num_epochs=500,
    batch_size=1024,
    lr=1e-4,
    weight_decay=1e-5,
    split_ratio=0.8,
    grad_clip=1.0
):
    """
    Stage B => load shape2spec from stage A (frozen),
    train spec->shape for 500 epochs.
    We'll produce the old 3-col polygon-based visualization at the end.
    """
    os.makedirs(out_folder_stageB, exist_ok=True)
    print(f"[Stage B] => {out_folder_stageB}")

    # load shape2spec
    shape2spec_frozen= ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4)
    shape2spec_frozen.load_state_dict(torch.load(shape2spec_ckpt))
    print("[Stage B] shape2spec loaded from =>", shape2spec_ckpt)

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape2spec_frozen.to(device)
    # freeze
    for p in shape2spec_frozen.parameters():
        p.requires_grad=False

    # build spec2shape
    spec2shape_net= Spectra2ShapeVarLen(d_in=100, d_model=256, nhead=4, num_layers=4).to(device)

    # pipeline
    pipeline= Spec2ShapePipeline_FrozenShape2Spec(spec2shape_net, shape2spec_frozen).to(device)

    # data
    ds_full= Q1ShiftedSpectraDataset(csv_file)
    ds_len= len(ds_full)
    trn_len= int(ds_len*split_ratio)
    val_len= ds_len- trn_len
    ds_train, ds_val= random_split(ds_full, [trn_len, val_len])
    print(f"[DATA: Stage B] total={ds_len}, train={trn_len}, val={val_len}")

    train_loader= DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader=   DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer= torch.optim.AdamW(spec2shape_net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    crit_mse= nn.MSELoss()

    train_losses, val_losses= [], []

    for epoch in range(num_epochs):
        pipeline.train()
        run_loss= 0.0
        for (spec_gt_np, shape_gt_np, uid_list) in train_loader:
            spec_gt= spec_gt_np.to(device)
            shape_gt= shape_gt_np.to(device)

            shape_pred, spec_chain= pipeline(spec_gt)
            # shape MSE
            loss_shape= crit_mse(shape_pred, shape_gt)
            # spec chain MSE
            loss_spec= crit_mse(spec_chain, spec_gt)

            loss= loss_shape+ loss_spec

            optimizer.zero_grad()
            loss.backward()
            if grad_clip>0:
                nn.utils.clip_grad_norm_(spec2shape_net.parameters(), grad_clip)
            optimizer.step()

            run_loss+= loss.item()

        avg_train= run_loss/ len(train_loader)
        train_losses.append(avg_train)

        # val
        pipeline.eval()
        val_sum=0.0
        val_count=0
        with torch.no_grad():
            for (spec_gt_np, shape_gt_np, uid_list) in val_loader:
                bsz= spec_gt_np.size(0)
                spec_gt= spec_gt_np.to(device)
                shape_gt= shape_gt_np.to(device)

                shape_pd, spec_pd= pipeline(spec_gt)
                vs= (crit_mse(shape_pd, shape_gt)+ crit_mse(spec_pd, spec_gt))*bsz
                val_sum+= vs.item()
                val_count+= bsz
        avg_val= val_sum/ val_count
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if (epoch+1)%50==0 or epoch==0 or epoch==(num_epochs-1):
            print(f"[Stage B] Epoch[{epoch+1}/{num_epochs}] => trainLoss={avg_train:.4f}, valLoss={avg_val:.4f}")

    # final
    plt.figure()
    plt.plot(train_losses,label='train')
    plt.plot(val_losses,label='val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss= MSE(shape_pred, shape_gt)+ MSE(spec_chain, spec_gt)")
    plt.title("Stage B: spec->shape (frozen shape2spec) training")
    plt.legend()
    plt.savefig(os.path.join(out_folder_stageB,"training_curve_stageB.png"))
    plt.close()

    final_val= val_losses[-1]
    print(f"[Stage B] final val loss => {final_val:.6f}")

    # save
    spec2shape_path= os.path.join(out_folder_stageB, "spec2shape_stageB.pt")
    torch.save(spec2shape_net.state_dict(), spec2shape_path)
    print("[Stage B] spec2shape saved =>", spec2shape_path)

    # do 3-col polygon-based visualization => 
    # we'll define a function to do the final pipeline's 3-col approach
    visualize_3col_stageB(pipeline, ds_val, shape2spec_frozen, device, out_folder_stageB)


###############################################################################
# 3-col polygon-based visualization for Stage B
###############################################################################
def visualize_3col_stageB(
    pipeline, ds_val, shape2spec_frozen, device, out_dir=".",
    sample_count=4, seed=123
):
    """
    We do the "old" 3-col approach:

     - Left => GT spectrum(blue)
     - Middle => GT shape(green) vs predicted shape(red polygon)
     - Right => original(blue), shape2spec(GT shape => green dashed),
                shape2spec(pred shape => red dashed)

    pipeline: spec->shape->(frozen)shape2spec
    shape2spec_frozen: used to get the shape2spec(GT shape) => green dashed
    """
    import random
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    if len(ds_val)==0:
        print("[Stage B] empty val => skip visualize.")
        return

    idx_samples= random.sample(range(len(ds_val)), min(sample_count,len(ds_val)))
    fig, axes= plt.subplots(len(idx_samples), 3, figsize=(12, 3*len(idx_samples)))
    if len(idx_samples)==1:
        axes= [axes]

    pipeline.eval()
    shape2spec_frozen.eval()
    with torch.no_grad():
        for i, idx_ in enumerate(idx_samples):
            spec_gt, shape_gt, uid_ = ds_val[idx_]

            # left => GT spectrum
            axL= axes[i][0]
            for row_ in spec_gt:
                axL.plot(row_, color='blue', alpha=0.5)
            axL.set_title(f"UID={uid_}\n(Left) GT Spectrum(blue)")

            # middle => GT shape(green) vs predicted shape(red)
            axM= axes[i][1]
            pres_g= (shape_gt[:,0]>0.5)
            q1_g= shape_gt[pres_g,1:3]
            if len(q1_g)>0:
                c4_g= replicate_c4(q1_g)
                c4_g= sort_points_by_angle(c4_g)
                plot_polygon(axM, c4_g, color='green', alpha=0.4, fill=True)

            spec_t= torch.tensor(spec_gt, dtype=torch.float32, device=device).unsqueeze(0)
            shape_pd, spec_pd= pipeline(spec_t)  # shape->(frozen)spec
            shape_pd= shape_pd.cpu().numpy()[0]
            spec_pd= spec_pd.cpu().numpy()[0]

            pres_p= (shape_pd[:,0]>0.5)
            q1_p= shape_pd[pres_p,1:3]
            if len(q1_p)>0:
                c4_p= replicate_c4(q1_p)
                c4_p= sort_points_by_angle(c4_p)
                plot_polygon(axM, c4_p, color='red', alpha=0.3, fill=False)
            axM.set_aspect("equal","box")
            axM.set_xlim([-0.5,0.5])
            axM.set_ylim([-0.5,0.5])
            axM.grid(True)
            axM.set_title("(Middle)\nGT shape(green) vs Pred shape(red)")

            # right => original(blue), shape2spec(GT shape => green dashed),
            #           shape2spec(pred shape => red dashed)
            axR= axes[i][2]
            for row_ in spec_gt:
                axR.plot(row_, color='blue', alpha=0.5)

            # shape2spec(GT shape)
            shape_gt_t= torch.tensor(shape_gt, dtype=torch.float32, device=device).unsqueeze(0)
            spec_gtShape= shape2spec_frozen(shape_gt_t).cpu().numpy()[0]
            for row_ in spec_gtShape:
                axR.plot(row_, color='green', alpha=0.5, linestyle='--')

            # shape2spec(pred shape) => we already have spec_pd
            for row_ in spec_pd:
                axR.plot(row_, color='red', alpha=0.5, linestyle='--')

            axR.set_xlabel("Wavelength index")
            axR.set_ylabel("Reflectance")
            axR.set_title("(Right)\nOriginal(blue), GT->spec(green dash), pred->spec(red dash)")

    plt.tight_layout()
    out_fig= os.path.join(out_dir, "samples_3col_stageB.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"[Stage B visualization saved] => {out_fig}")


###############################################################################
# MAIN => Two Stage
###############################################################################
def main():
    CSV_FILE= "merged_s4_shapes_iccpOv10kG40_seed88888.csv"
    timestamp= datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out= f"outputs_two_stage_{timestamp}"
    os.makedirs(base_out, exist_ok=True)

    # Stage A => shape->spec
    outA= os.path.join(base_out, "stageA")
    shape2spec_pt= train_shape2spec_stageA(
        csv_file= CSV_FILE,
        out_folder_stageA= outA,
        num_epochs=500,       # up to 500
        batch_size=1024,
        lr=1e-4,
        weight_decay=1e-5,
        split_ratio=0.8,
        grad_clip=1.0
    )

    # Stage B => spec->shape with frozen shape2spec
    outB= os.path.join(base_out, "stageB")
    train_spec2shape_stageB(
        csv_file= CSV_FILE,
        shape2spec_ckpt= shape2spec_pt,
        out_folder_stageB= outB,
        num_epochs=500,       # up to 500
        batch_size=1024,
        lr=1e-4,
        weight_decay=1e-5,
        split_ratio=0.8,
        grad_clip=1.0
    )


if __name__=="__main__":
    main()

