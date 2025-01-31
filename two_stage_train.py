#!/usr/bin/env python3

"""
two_stage_train.py

Extended two-stage pipeline to improve shape + spectrum prediction, 
with both training and test modes:

TRAIN MODE (default):
  1) Stage A: Train shape->spec alone (500 epochs).
     - Save training/val losses to CSV and PNG.
     - Save shape2spec_stageA.pt
     - Visualization: 2-col approach (GT shape polygon & GT vs. pred. spectrum).
  2) Stage B: Freeze shape2spec, train spec->shape for 500 epochs.
     - Save training/val losses to CSV and PNG.
     - Save spec2shape_stageB.pt
     - Visualization: 3-col approach 
       (Left= GT spec, Middle= GT shape vs. Pred shape polygon, 
        Right= original spec + shape2spec(GT shape => green dashed) + shape2spec(pred => red dashed)).

TEST MODE (--test):
  - We load shape2spec_stageA.pt, spec2shape_stageB.pt from model_dir's stageA/ & stageB/.
  - We run shape->spec on the dataset if you want 
    (or skip if you only care about the final pipeline).
  - We run final pipeline (spec->shape->(frozen)shape2spec) on the dataset.
  - We save new visualizations and a CSV with predictions vs. GT 
    in "model_dir/test/".

We also store train/val losses in CSV for later figure usage.

Author: Lachlan (ChatGPT code)
"""

import os
import argparse
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
# DATASET
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
# UTILS: replicate_c4, sort_points_by_angle, plot_polygon
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
    patch= patches.PathPatch(path, facecolor=color if fill else 'none',
                             alpha=alpha, edgecolor=color)
    ax.add_patch(patch)
    ax.autoscale_view()


###############################################################################
# STAGE A: shape->spec
###############################################################################
class ShapeToSpectraModel(nn.Module):
    """
    shape(4,3)->(11,100). presence => mask
    Weighted sum => bigger MLP => (11,100)
    Using deeper transformer for better performance.
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

        x_proj= self.input_proj(shape_4x3)
        x_enc= self.encoder(x_proj, src_key_padding_mask=key_padding_mask)

        pres_sum= presence.sum(dim=1,keepdim=True)+1e-8
        x_enc_w= x_enc* presence.unsqueeze(-1)
        shape_emb= x_enc_w.sum(dim=1)/ pres_sum

        out_flat= self.mlp(shape_emb)
        out_2d= out_flat.view(bsz,11,100)
        return out_2d


def train_stageA_shape2spec(
    csv_file,
    out_dir,
    num_epochs=500,
    batch_size=1024,
    lr=1e-4,
    weight_decay=1e-5,
    split_ratio=0.8,
    grad_clip=1.0
):
    """
    Stage A: shape->spec training.
    Returns path to shape2spec model.
    Saves train/val losses in CSV, plus a PNG.
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Stage A] => {out_dir}")

    # dataset
    ds_full= Q1ShiftedSpectraDataset(csv_file)
    ds_len= len(ds_full)
    trn_len= int(ds_len*split_ratio)
    val_len= ds_len- trn_len
    ds_train, ds_val= random_split(ds_full, [trn_len, val_len])
    print(f"[DATA: Stage A] total={ds_len}, train={trn_len}, val={val_len}")

    train_loader= DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader=   DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False)

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Stage A] device:", device)

    model= ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4).to(device)
    optimizer= torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    crit_mse= nn.MSELoss()

    train_losses, val_losses= [], []

    for epoch in range(num_epochs):
        model.train()
        run_loss= 0.0
        for (spec_np, shape_np, uid_list) in train_loader:
            shape_t= shape_np.to(device)
            spec_gt= spec_np.to(device)

            spec_pd= model(shape_t)
            loss= crit_mse(spec_pd, spec_gt)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip>0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            run_loss+= loss.item()

        avg_train= run_loss/len(train_loader)
        train_losses.append(avg_train)

        # validation
        model.eval()
        val_sum=0.0
        val_count=0
        with torch.no_grad():
            for (spec_np, shape_np, uid_list) in val_loader:
                bsz= shape_np.size(0)
                st= shape_np.to(device)
                sg= spec_np.to(device)
                sd= model(st)
                v= crit_mse(sd, sg)* bsz
                val_sum+= v.item()
                val_count+= bsz
        avg_val= val_sum/ val_count
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if (epoch+1)%50==0 or epoch==0 or epoch==(num_epochs-1):
            print(f"[Stage A] Epoch[{epoch+1}/{num_epochs}] => trainLoss={avg_train:.4f}, valLoss={avg_val:.4f}")

    # save curves
    np.savetxt(os.path.join(out_dir, "train_losses_stageA.csv"), np.array(train_losses), delimiter=",")
    np.savetxt(os.path.join(out_dir, "val_losses_stageA.csv"), np.array(val_losses), delimiter=",")

    plt.figure()
    plt.plot(train_losses,label='train')
    plt.plot(val_losses,label='val')
    plt.xlabel("Epoch")
    plt.ylabel("MSE( shape->spec )")
    plt.title("Stage A: shape->spec training")
    plt.legend()
    plt.savefig(os.path.join(out_dir,"training_curve_stageA.png"))
    plt.close()

    final_val= val_losses[-1]
    print(f"[Stage A] final val loss => {final_val:.6f}")

    # save model
    shape2spec_path= os.path.join(out_dir, "shape2spec_stageA.pt")
    torch.save(model.state_dict(), shape2spec_path)
    print("[Stage A] model saved =>", shape2spec_path)

    return shape2spec_path, ds_val, model


def visualize_stageA_samples(model, ds_val, device, out_dir, sample_count=4, seed=123):
    """
    We do a 2-col approach for shape->spec alone:
      - Left => shape polygon
      - Right => GT spec(blue) vs predicted spec(red dashed)
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
                c4= sort_points_by_angle(c4)
                plot_polygon(axL, c4, color='green', alpha=0.4, fill=True)
            axL.set_aspect("equal","box")
            axL.set_xlim([-0.5,0.5])
            axL.set_ylim([-0.5,0.5])
            axL.grid(True)
            axL.set_title(f"UID={uid_}\n(Left) GT shape polygon")

            # right => GT spec(blue) vs predicted spec(red dashed)
            axR= axes[i][1]
            shape_t= torch.tensor(shape_gt, dtype=torch.float32, device=device).unsqueeze(0)
            spec_pd= model(shape_t).cpu().numpy()[0]

            for row_ in spec_gt:
                axR.plot(row_, color='blue', alpha=0.5)
            for row_ in spec_pd:
                axR.plot(row_, color='red', alpha=0.5, linestyle='--')

            axR.set_title("(Right)\nGT spec(blue) vs Pred spec(red)")
            axR.set_xlabel("Wavelength index")
            axR.set_ylabel("Reflectance")

    plt.tight_layout()
    out_fig= os.path.join(out_dir, "samples_2col_stageA.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"[Stage A] sample visualization => {out_fig}")


###############################################################################
# STAGE B: spec->shape using a frozen shape2spec
###############################################################################
class Spectra2ShapeVarLen(nn.Module):
    """
    spec(11,100) -> shape(4,3)
    We do a bigger transformer with row MLP => d_model=256, num_layers=4
    presence chaining => forced presence[0]=1
    """
    def __init__(self, d_in=100, d_model=256, nhead=4, num_layers=4):
        super().__init__()
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
        x_r= spec_11x100.view(-1, spec_11x100.size(2))
        x_pre= self.row_preproc(x_r)
        x_pre= x_pre.view(bsz, -1, x_pre.size(-1))

        x_enc= self.encoder(x_pre)
        x_agg= x_enc.mean(dim=1)

        out_12= self.mlp(x_agg)
        out_4x3= out_12.view(bsz,4,3)

        presence_logits= out_4x3[:,:,0]
        xy_raw= out_4x3[:,:,1:]

        presence_list=[]
        for i in range(4):
            if i==0:
                presence_list.append(torch.ones(bsz, device=out_4x3.device, dtype=torch.float32))
            else:
                prob_i= torch.sigmoid(presence_logits[:,i]).clamp(1e-6,1-1e-6)
                prob_chain= prob_i* presence_list[i-1]
                ste_i= (prob_chain>0.5).float() + prob_chain - prob_chain.detach()
                presence_list.append(ste_i)
        presence_stack= torch.stack(presence_list, dim=1)

        xy_bounded= torch.sigmoid(xy_raw)
        xy_final= xy_bounded* presence_stack.unsqueeze(-1)
        final_shape= torch.cat([presence_stack.unsqueeze(-1), xy_final], dim=-1)
        return final_shape


class Spec2ShapeFrozen(nn.Module):
    """
    spec->shape (trainable),
    shape->spec (frozen).
    """
    def __init__(self, spec2shape_net, shape2spec_frozen):
        super().__init__()
        self.spec2shape= spec2shape_net
        self.shape2spec_frozen= shape2spec_frozen
        for p in self.shape2spec_frozen.parameters():
            p.requires_grad=False

    def forward(self, spec_input):
        shape_pred= self.spec2shape(spec_input)
        # freeze shape2spec
        with torch.no_grad():
            spec_chain= self.shape2spec_frozen(shape_pred)
        return shape_pred, spec_chain


def train_stageB_spec2shape_frozen(
    csv_file,
    out_dir,
    shape2spec_ckpt,
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
    Save train/val losses in CSV and PNG.
    Return path of spec2shape.
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Stage B] => {out_dir}")

    # load shape2spec
    shape2spec_frozen= ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4)
    shape2spec_frozen.load_state_dict(torch.load(shape2spec_ckpt))
    print("[Stage B] shape2spec loaded from =>", shape2spec_ckpt)

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape2spec_frozen.to(device)
    for p in shape2spec_frozen.parameters():
        p.requires_grad=False

    # build spec2shape
    spec2shape_net= Spectra2ShapeVarLen(d_in=100, d_model=256, nhead=4, num_layers=4).to(device)
    pipeline= Spec2ShapeFrozen(spec2shape_net, shape2spec_frozen).to(device)

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

            shape_pd, spec_chain= pipeline(spec_gt)
            loss_shape= crit_mse(shape_pd, shape_gt)
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

        pipeline.eval()
        val_sum=0.0
        val_count=0
        with torch.no_grad():
            for (spec_gt_np, shape_gt_np, uid_list) in val_loader:
                bsz= spec_gt_np.size(0)
                sgt= spec_gt_np.to(device)
                shg= shape_gt_np.to(device)
                sp, sc= pipeline(sgt)
                v= (crit_mse(sp, shg)+ crit_mse(sc, sgt))*bsz
                val_sum+= v.item()
                val_count+= bsz
        avg_val= val_sum/ val_count
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if (epoch+1)%50==0 or epoch==0 or epoch==(num_epochs-1):
            print(f"[Stage B] Epoch[{epoch+1}/{num_epochs}] => trainLoss={avg_train:.4f}, valLoss={avg_val:.4f}")

    # save curves
    np.savetxt(os.path.join(out_dir, "train_losses_stageB.csv"), np.array(train_losses), delimiter=",")
    np.savetxt(os.path.join(out_dir, "val_losses_stageB.csv"), np.array(val_losses), delimiter=",")

    plt.figure()
    plt.plot(train_losses,label='train')
    plt.plot(val_losses,label='val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss= MSE(shape_pred, shape_gt)+ MSE(spec_chain, spec_gt)")
    plt.title("Stage B: spec->shape (frozen shape2spec)")
    plt.legend()
    plt.savefig(os.path.join(out_dir,"training_curve_stageB.png"))
    plt.close()

    final_val= val_losses[-1]
    print(f"[Stage B] final val => {final_val:.6f}")

    spec2shape_path= os.path.join(out_dir, "spec2shape_stageB.pt")
    torch.save(spec2shape_net.state_dict(), spec2shape_path)
    print("[Stage B] spec2shape saved =>", spec2shape_path)

    return spec2shape_path, ds_val, shape2spec_frozen, spec2shape_net


def visualize_stageB_samples(pipeline, shape2spec_frozen, ds_val, device,
                             out_dir, sample_count=4, seed=123):
    """
    3-col approach:
      Left => GT spectrum(blue)
      Middle => GT shape(green) vs predicted shape(red)
      Right => original(blue), shape2spec(GT => green dashed), shape2spec(pred => red dashed)
    """
    import random
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    if len(ds_val)==0:
        print("[Stage B] val set empty => skip visualize.")
        return

    idx_samples= random.sample(range(len(ds_val)), min(sample_count,len(ds_val)))
    fig, axes= plt.subplots(len(idx_samples),3, figsize=(12,3*len(idx_samples)))
    if len(idx_samples)==1:
        axes= [axes]

    pipeline.eval()
    shape2spec_frozen.eval()
    with torch.no_grad():
        for i, idx_ in enumerate(idx_samples):
            spec_gt, shape_gt, uid_ = ds_val[idx_]

            # Left => GT spectrum
            axL= axes[i][0]
            for row_ in spec_gt:
                axL.plot(row_, color='blue', alpha=0.5)
            axL.set_title(f"UID={uid_}\n(Left) GT spec(blue)")

            # Middle => GT shape(green) vs predicted shape(red)
            axM= axes[i][1]
            pres_g= (shape_gt[:,0]>0.5)
            q1_g= shape_gt[pres_g,1:3]
            if len(q1_g)>0:
                c4g= replicate_c4(q1_g)
                c4g= sort_points_by_angle(c4g)
                plot_polygon(axM, c4g, color='green', alpha=0.4, fill=True)

            # forward
            spec_t= torch.tensor(spec_gt, dtype=torch.float32, device=device).unsqueeze(0)
            shape_pd, spec_pd= pipeline(spec_t)
            shape_pd= shape_pd.cpu().numpy()[0]
            spec_pd= spec_pd.cpu().numpy()[0]

            pres_p= (shape_pd[:,0]>0.5)
            q1_p= shape_pd[pres_p,1:3]
            if len(q1_p)>0:
                c4p= replicate_c4(q1_p)
                c4p= sort_points_by_angle(c4p)
                plot_polygon(axM, c4p, color='red', alpha=0.3, fill=False)
            axM.set_title("(Middle)\nGT shape(green) vs Pred shape(red)")
            axM.set_xlim([-0.5,0.5])
            axM.set_ylim([-0.5,0.5])
            axM.set_aspect("equal","box")
            axM.grid(True)

            # Right => original(blue), shape2spec(GT => green dashed),
            #           shape2spec(pred => red dashed)
            axR= axes[i][2]
            for row_ in spec_gt:
                axR.plot(row_, color='blue', alpha=0.5)

            shape_gt_t= torch.tensor(shape_gt, dtype=torch.float32, device=device).unsqueeze(0)
            spec_gtshape= shape2spec_frozen(shape_gt_t).cpu().numpy()[0]
            for row_ in spec_gtshape:
                axR.plot(row_, color='green', alpha=0.5, linestyle='--')

            for row_ in spec_pd:
                axR.plot(row_, color='red', alpha=0.5, linestyle='--')

            axR.set_title("(Right)\nOriginal(blue), GT->spec(green), pred->spec(red)")
            axR.set_xlabel("Wavelength index")
            axR.set_ylabel("Reflectance")

    plt.tight_layout()
    out_fig= os.path.join(out_dir, "samples_3col_stageB.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"[Stage B] sample visualization => {out_fig}")


###############################################################################
# MAIN
###############################################################################
def parse_args():
    parser= argparse.ArgumentParser(description="Two-stage pipeline for shape->spec and spec->shape with optional test mode.")
    parser.add_argument("--csv_file", type=str, default="merged_s4_shapes_iccpOv10kG40_seed88888.csv", help="Path to dataset CSV.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output folder base. If None, auto-named.")
    parser.add_argument("--test", action="store_true", help="If set, we do test mode instead of training.")
    parser.add_argument("--model_dir", type=str, default="", help="Folder with stageA/ shape2spec_stageA.pt and stageB/ spec2shape_stageB.pt, for test mode.")
    return parser.parse_args()

def main():
    args= parse_args()

    if not args.test:
        # Train mode
        timestamp= datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.out_dir is None:
            base_out= f"outputs_two_stage_{timestamp}"
        else:
            base_out= args.out_dir
        os.makedirs(base_out, exist_ok=True)

        # Stage A
        outA= os.path.join(base_out, "stageA")
        shape2spec_path, ds_valA, modelA = train_stageA_shape2spec(
            csv_file= args.csv_file,
            out_dir= outA,
            num_epochs=500,
            batch_size=1024,
            lr=1e-4,
            weight_decay=1e-5,
            split_ratio=0.8,
            grad_clip=1.0
        )
        # Visualize Stage A samples
        deviceA= next(modelA.parameters()).device
        visualize_stageA_samples(modelA, ds_valA, deviceA, outA, sample_count=4)

        # Stage B
        outB= os.path.join(base_out, "stageB")
        spec2shape_path, ds_valB, shape2spec_froz, spec2shape_net = train_stageB_spec2shape_frozen(
            csv_file= args.csv_file,
            out_dir= outB,
            shape2spec_ckpt= shape2spec_path,
            num_epochs=500,
            batch_size=1024,
            lr=1e-4,
            weight_decay=1e-5,
            split_ratio=0.8,
            grad_clip=1.0
        )
        # Visualize Stage B
        deviceB= next(spec2shape_net.parameters()).device
        visualize_stageB_samples(Spec2ShapeFrozen(spec2shape_net, shape2spec_froz),
                                 shape2spec_froz, ds_valB, deviceB, outB, sample_count=4)

    else:
        # TEST mode
        # We'll load the stageA shape2spec, stageB spec2shape from the given model_dir
        model_dir= args.model_dir
        if not os.path.isdir(model_dir):
            print("[Error] --model_dir not found:", model_dir)
            return

        stageA_dir= os.path.join(model_dir, "stageA")
        stageB_dir= os.path.join(model_dir, "stageB")
        shape2spec_ckpt= os.path.join(stageA_dir, "shape2spec_stageA.pt")
        spec2shape_ckpt= os.path.join(stageB_dir, "spec2shape_stageB.pt")
        if not (os.path.isfile(shape2spec_ckpt) and os.path.isfile(spec2shape_ckpt)):
            print("[Error] shape2spec or spec2shape checkpoint not found in", model_dir)
            return

        print(f"[TEST] We load shape2spec={shape2spec_ckpt}, spec2shape={spec2shape_ckpt}")

        # Prepare a test folder
        test_folder= os.path.join(model_dir, "test")
        os.makedirs(test_folder, exist_ok=True)

        # Load dataset from args.csv_file
        ds_test= Q1ShiftedSpectraDataset(args.csv_file)
        print(f"[TEST] dataset size={len(ds_test)} from {args.csv_file}")
        test_loader= DataLoader(ds_test, batch_size=1024, shuffle=False)

        device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1) shape2spec test if you want
        # We'll load shape2spec
        shape2spec_model= ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4)
        shape2spec_model.load_state_dict(torch.load(shape2spec_ckpt))
        shape2spec_model.to(device)
        shape2spec_model.eval()

        # 2) spec2shape + shape2spec (frozen) for final pipeline
        shape2spec_frozen= ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4)
        shape2spec_frozen.load_state_dict(torch.load(shape2spec_ckpt))
        shape2spec_frozen.to(device)
        for p in shape2spec_frozen.parameters():
            p.requires_grad=False
        spec2shape_net= Spectra2ShapeVarLen(d_in=100, d_model=256, nhead=4, num_layers=4)
        spec2shape_net.load_state_dict(torch.load(spec2shape_ckpt))
        spec2shape_net.to(device)
        final_pipeline= Spec2ShapeFrozen(spec2shape_net, shape2spec_frozen).to(device)
        final_pipeline.eval()

        # We'll do a CSV dump of shape2spec (Stage A) predictions and final pipeline predictions
        out_csv= os.path.join(test_folder, "test_predictions.csv")
        cols= ["uid","presence_gt","xy_gt","presence_pred","xy_pred","specMSE_shape2spec","specMSE_finalPipeline"]
        rows= []

        crit_mse= nn.MSELoss(reduction='none')  # so we can compute average ourselves

        # We'll also store some visuals for a random subset
        subset_idx= random.sample(range(len(ds_test)), min(4, len(ds_test)))
        visuals_stageA= []
        visuals_stageB= []

        with torch.no_grad():
            for i, (spec_np, shape_np, uid_) in enumerate(test_loader):
                bsz= shape_np.size(0)
                spec_t= spec_np.to(device)
                shape_t= shape_np.to(device)

                # shape2spec (Stage A)
                spec_predA= shape2spec_model(shape_t)
                # we can measure MSE
                # shape => spec
                msesA= torch.mean(crit_mse(spec_predA, spec_t).view(bsz, -1), dim=1)  # (B,)

                # final pipeline => spec->shape->spec
                shape_predB, spec_predB= final_pipeline(spec_t)
                # measure MSE
                msesB= torch.mean(crit_mse(spec_predB, spec_t).view(bsz, -1), dim=1)  # (B,)

                # record each sample in CSV
                shape_np_ = shape_np.cpu().numpy()  # (B,4,3)
                shape_predB_ = shape_predB.cpu().numpy()  # (B,4,3)
                msesA_ = msesA.cpu().numpy()
                msesB_ = msesB.cpu().numpy()

                for n in range(bsz):
                    rowd= {
                        "uid": uid_[n],
                        "presence_gt": shape_np_[n,:,0].tolist(),
                        "xy_gt": shape_np_[n,:,1:].tolist(),
                        "presence_pred": shape_predB_[n,:,0].tolist(),
                        "xy_pred": shape_predB_[n,:,1:].tolist(),
                        "specMSE_shape2spec": float(msesA_[n]),
                        "specMSE_finalPipeline": float(msesB_[n])
                    }
                    rows.append(rowd)

        # dump CSV
        df_out= pd.DataFrame(rows, columns=cols)
        df_out.to_csv(out_csv, index=False)
        print(f"[TEST] predictions saved => {out_csv}")

        # We'll do final sample visualizations
        # stageA => shape->spec
        # stageB => final pipeline
        # We'll reuse the same approach as we do in training but store in test folder.

        # Stage A visuals
        print("[TEST] Visualizing Stage A shape->spec on subset of test data...")
        stageA_fig= os.path.join(test_folder,"samples_2col_test_stageA.png")
        visualize_stageA_on_test(shape2spec_model, ds_test, device, stageA_fig)

        # Stage B visuals
        print("[TEST] Visualizing Stage B final pipeline on subset of test data...")
        stageB_fig= os.path.join(test_folder,"samples_3col_test_stageB.png")
        visualize_stageB_on_test(final_pipeline, shape2spec_frozen, ds_test, device, stageB_fig)


def visualize_stageA_on_test(model, ds_test, device, out_fig, sample_count=4, seed=123):
    """
    Same approach as visualize_stageA_samples but a single function for test usage.
    2-col plot: (shape polygon) vs (GT vs. predicted spec).
    """
    import random
    random.seed(seed)
    if len(ds_test)==0:
        print("[TEST: Stage A] empty dataset => skip.")
        return

    idx_samples= random.sample(range(len(ds_test)), min(sample_count,len(ds_test)))
    fig, axes= plt.subplots(len(idx_samples), 2, figsize=(8,3*len(idx_samples)))
    if len(idx_samples)==1:
        axes= [axes]

    model.eval()
    with torch.no_grad():
        for i, idx_ in enumerate(idx_samples):
            spec_gt, shape_gt, uid_ = ds_test[idx_]

            # left => shape polygon
            axL= axes[i][0]
            pres= (shape_gt[:,0]>0.5)
            q1= shape_gt[pres,1:3]
            if len(q1)>0:
                c4= replicate_c4(q1)
                c4= sort_points_by_angle(c4)
                plot_polygon(axL, c4, color='green', alpha=0.4, fill=True)
            axL.set_aspect("equal","box")
            axL.set_xlim([-0.5,0.5])
            axL.set_ylim([-0.5,0.5])
            axL.grid(True)
            axL.set_title(f"UID={uid_}\nGT shape polygon")

            # right => GT spec(blue) vs predicted spec(red)
            axR= axes[i][1]
            shape_t= torch.tensor(shape_gt, dtype=torch.float32, device=device).unsqueeze(0)
            spec_pd= model(shape_t).cpu().numpy()[0]

            for row_ in spec_gt:
                axR.plot(row_, color='blue', alpha=0.5)
            for row_ in spec_pd:
                axR.plot(row_, color='red', alpha=0.5, linestyle='--')

            axR.set_title("GT spec(blue) vs Pred spec(red)")
            axR.set_xlabel("Wavelength index")
            axR.set_ylabel("Reflectance")

    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()
    print(f"[TEST: Stage A] saved => {out_fig}")


def visualize_stageB_on_test(pipeline, shape2spec_frozen, ds_test, device, out_fig,
                             sample_count=4, seed=123):
    """
    3-col approach for final pipeline test:
      Left => GT spectrum(blue)
      Middle => GT shape(green) vs predicted shape(red)
      Right => original(blue), shape2spec(GT => green), shape2spec(pred => red).
    """
    import random
    random.seed(seed)
    if len(ds_test)==0:
        print("[TEST: Stage B] empty dataset => skip.")
        return

    idx_samples= random.sample(range(len(ds_test)), min(sample_count,len(ds_test)))
    fig, axes= plt.subplots(len(idx_samples),3, figsize=(12,3*len(idx_samples)))
    if len(idx_samples)==1:
        axes= [axes]

    pipeline.eval()
    shape2spec_frozen.eval()
    with torch.no_grad():
        for i, idx_ in enumerate(idx_samples):
            spec_gt, shape_gt, uid_ = ds_test[idx_]

            axL= axes[i][0]
            for row_ in spec_gt:
                axL.plot(row_, color='blue', alpha=0.5)
            axL.set_title(f"UID={uid_}\n(Left) GT spec(blue)")

            axM= axes[i][1]
            pres_g= (shape_gt[:,0]>0.5)
            q1_g= shape_gt[pres_g,1:3]
            if len(q1_g)>0:
                c4g= replicate_c4(q1_g)
                c4g= sort_points_by_angle(c4g)
                plot_polygon(axM, c4g, color='green', alpha=0.4, fill=True)

            spec_t= torch.tensor(spec_gt, dtype=torch.float32, device=device).unsqueeze(0)
            shape_pd, spec_pd= pipeline(spec_t)
            shape_pd= shape_pd.cpu().numpy()[0]
            spec_pd= spec_pd.cpu().numpy()[0]

            pres_p= (shape_pd[:,0]>0.5)
            q1_p= shape_pd[pres_p,1:3]
            if len(q1_p)>0:
                c4p= replicate_c4(q1_p)
                c4p= sort_points_by_angle(c4p)
                plot_polygon(axM, c4p, color='red', alpha=0.3, fill=False)
            axM.set_xlim([-0.5,0.5])
            axM.set_ylim([-0.5,0.5])
            axM.set_aspect("equal","box")
            axM.grid(True)
            axM.set_title("(Middle)\nGT shape(green) vs Pred shape(red)")

            axR= axes[i][2]
            for row_ in spec_gt:
                axR.plot(row_, color='blue', alpha=0.5)

            shape_gt_t= torch.tensor(shape_gt, dtype=torch.float32, device=device).unsqueeze(0)
            spec_gtshape= shape2spec_frozen(shape_gt_t).cpu().numpy()[0]
            for row_ in spec_gtshape:
                axR.plot(row_, color='green', alpha=0.5, linestyle='--')

            for row_ in spec_pd:
                axR.plot(row_, color='red', alpha=0.5, linestyle='--')
            axR.set_title("(Right)\nOriginal(blue), GT->spec(green), pred->spec(red)")
            axR.set_xlabel("Wavelength index")
            axR.set_ylabel("Reflectance")

    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()
    print(f"[TEST: Stage B] saved => {out_fig}")


if __name__=="__main__":
    args= parse_args()

    if not args.test:
        # TRAIN mode => two stage
        timestamp= datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.out_dir is None:
            base_out= f"outputs_two_stage_{timestamp}"
        else:
            base_out= args.out_dir
        os.makedirs(base_out, exist_ok=True)

        # Stage A
        stageA_dir= os.path.join(base_out, "stageA")
        shape2spec_ckpt, ds_valA, modelA = train_stageA_shape2spec(
            csv_file=args.csv_file,
            out_dir=stageA_dir,
            num_epochs=500,
            batch_size=1024,
            lr=1e-4,
            weight_decay=1e-5,
            split_ratio=0.8,
            grad_clip=1.0
        )
        # Visualize a few from ds_val
        deviceA= next(modelA.parameters()).device
        visualize_stageA_samples(modelA, ds_valA, deviceA, stageA_dir, sample_count=4)

        # Stage B
        stageB_dir= os.path.join(base_out, "stageB")
        spec2shape_ckpt, ds_valB, shape2spec_froz, spec2shape_net = train_stageB_spec2shape_frozen(
            csv_file=args.csv_file,
            out_dir=stageB_dir,
            shape2spec_ckpt=shape2spec_ckpt,
            num_epochs=500,
            batch_size=1024,
            lr=1e-4,
            weight_decay=1e-5,
            split_ratio=0.8,
            grad_clip=1.0
        )
        deviceB= next(spec2shape_net.parameters()).device
        # final pipeline => spec->shape->(frozen) shape2spec
        final_pipeline= Spec2ShapeFrozen(spec2shape_net, shape2spec_froz)
        visualize_stageB_samples(final_pipeline, shape2spec_froz, ds_valB, deviceB, stageB_dir, sample_count=4)

    else:
        # TEST mode
        model_dir= args.model_dir
        if not model_dir:
            print("[Error] Must specify --model_dir in test mode.")
            exit(1)
        if not os.path.isdir(model_dir):
            print("[Error] model_dir does not exist:", model_dir)
            exit(1)

        # We'll place test results in model_dir/test
        test_out= os.path.join(model_dir, "test")
        os.makedirs(test_out, exist_ok=True)

        # StageA => shape2spec
        shape2spec_path= os.path.join(model_dir, "stageA", "shape2spec_stageA.pt")
        # StageB => spec2shape
        spec2shape_path= os.path.join(model_dir, "stageB", "spec2shape_stageB.pt")
        if not (os.path.isfile(shape2spec_path) and os.path.isfile(spec2shape_path)):
            print("[Error] cannot find shape2spec_stageA.pt or spec2shape_stageB.pt in", model_dir)
            exit(1)

        ds_test= Q1ShiftedSpectraDataset(args.csv_file)
        print(f"[TEST] dataset size={len(ds_test)} from => {args.csv_file}")
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1) Load shape2spec
        shape2spec_model= ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4)
        shape2spec_model.load_state_dict(torch.load(shape2spec_path))
        shape2spec_model.to(device)
        shape2spec_model.eval()

        # 2) Load final pipeline => spec->shape->(frozen)shape2spec
        shape2spec_frozen= ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4)
        shape2spec_frozen.load_state_dict(torch.load(shape2spec_path))
        shape2spec_frozen.to(device)
        for p in shape2spec_frozen.parameters():
            p.requires_grad=False

        spec2shape_net= Spectra2ShapeVarLen(d_in=100, d_model=256, nhead=4, num_layers=4)
        spec2shape_net.load_state_dict(torch.load(spec2shape_path))
        spec2shape_net.to(device)
        spec2shape_net.eval()

        final_pipeline= Spec2ShapeFrozen(spec2shape_net, shape2spec_frozen).to(device)
        final_pipeline.eval()

        # We'll do a CSV dump of predictions for each sample
        out_csv= os.path.join(test_out, "test_predictions.csv")
        rows= []
        cols= ["uid",
               "presence_gt","xy_gt",
               "presence_pred","xy_pred",
               "mse_shape2spec","mse_finalPipeline"]
        crit_mse= nn.MSELoss(reduction='none')

        loader_test= DataLoader(ds_test, batch_size=1024, shuffle=False)
        with torch.no_grad():
            for (spec_np, shape_np, uid_list) in loader_test:
                bsz= shape_np.size(0)
                spec_t= spec_np.to(device)
                shape_t= shape_np.to(device)

                # shape2spec => direct
                spec_predA= shape2spec_model(shape_t)
                # final pipeline => spec->shape->spec
                shape_pdB, spec_pdB= final_pipeline(spec_t)

                # compute MSE
                # shape2spec
                tmpA= torch.mean(crit_mse(spec_predA, spec_t).view(bsz, -1), dim=1)  # (B,)
                # final pipeline
                tmpB= torch.mean(crit_mse(spec_pdB, spec_t).view(bsz, -1), dim=1)

                shape_np_ = shape_np.cpu().numpy()
                shape_pdB_= shape_pdB.cpu().numpy()
                tmpA_ = tmpA.cpu().numpy()
                tmpB_ = tmpB.cpu().numpy()

                for i_ in range(bsz):
                    rowd= {
                        "uid": uid_list[i_],
                        "presence_gt": shape_np_[i_,:,0].tolist(),
                        "xy_gt": shape_np_[i_,:,1:].tolist(),
                        "presence_pred": shape_pdB_[i_,:,0].tolist(),
                        "xy_pred": shape_pdB_[i_,:,1:].tolist(),
                        "mse_shape2spec": float(tmpA_[i_]),
                        "mse_finalPipeline": float(tmpB_[i_])
                    }
                    rows.append(rowd)

        df= pd.DataFrame(rows, columns=cols)
        df.to_csv(out_csv, index=False)
        print(f"[TEST] saved predictions => {out_csv}")

        # do visual sample
        visualize_stageA_on_test(shape2spec_model, ds_test, device,
                                 os.path.join(test_out, "samples_2col_test_stageA.png"))
        visualize_stageB_on_test(final_pipeline, shape2spec_frozen, ds_test, device,
                                 os.path.join(test_out, "samples_3col_test_stageB.png"))
        print("[TEST] done.")

