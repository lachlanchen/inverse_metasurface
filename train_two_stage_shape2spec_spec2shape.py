#!/usr/bin/env python3

"""
train_two_stage_shape2spec_spec2shape.py

Two-stage training:

Stage A: Train shape2spec alone (given shape -> predict 11x100 spectrum).
         Save best model as 'shape2spec_only.pt'.

Stage B: Load that shape2spec, freeze it,
         train spec2shape while shape2spec is fixed,
         so the chain's final spec reconstruction is done by the frozen shape2spec.

We log training/validation losses, visualize, and produce final results.

Author: Lachlan (ChatGPT)

--------------------------------------------------------------------
DATASET:
We assume the usual dataset of (spectra(11,100), shape(4,3)) from the CSV.
But for shape2spec training, we treat shape(4,3) as input, spec(11,100) as output.
For spec2shape, we do the reverse, while still using shape2spec in the chain.

--------------------------------------------------------------------
NOTE:
You can incorporate Chamfer distance or additional spectral terms if you like.
Here, we do a simpler approach for demonstration:
 - Stage A: MSELoss(spectrum_pred, spectrum_gt).
 - Stage B: MSELoss(shape_pred, shape_gt) + MSELoss(spectrum_chain, spectrum_gt).
   (where 'spectrum_chain' is from the FROZEN shape2spec).
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
# 1) Dataset
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
            # Must have exactly 11 lines
            if len(grp)!=11:
                continue
            grp_sorted= grp.sort_values(by="c")

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
                "spectra": spec_11x100,  # shape(11,100)
                "shape": shape_4x3      # shape(4,3)
            })

        self.data_len= len(self.data_list)
        if self.data_len==0:
            raise ValueError("No valid shapes => SHIFT->Q1->UpTo4")

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        it= self.data_list[idx]
        # (spectra(11,100), shape(4,3), uid)
        return (it["spectra"], it["shape"], it["uid"])


###############################################################################
# 2) Single Model for shape2spec
###############################################################################
class ShapeToSpectraModel(nn.Module):
    """
    shape(4,3)->(11,100). presence => mask
    Weighted sum => final => bigger MLP => (11,100)
    Using a deeper transformer design if desired.
    """
    def __init__(self, d_in=3, d_model=128, nhead=4, num_layers=2):
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
            nn.Linear(d_model*4, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, 11*100)
        )

    def forward(self, shape_4x3):
        """
        shape_4x3: (B,4,3) => presence + x + y
        returns spec_pred: (B,11,100)
        """
        bsz= shape_4x3.size(0)
        presence= shape_4x3[:,:,0]   # (B,4)
        key_padding_mask= (presence<0.5)

        x_proj= self.input_proj(shape_4x3)  # (B,4,d_model)
        x_enc= self.encoder(x_proj, src_key_padding_mask=key_padding_mask)

        pres_sum= presence.sum(dim=1,keepdim=True)+1e-8
        x_enc_w= x_enc* presence.unsqueeze(-1)
        shape_emb= x_enc_w.sum(dim=1)/ pres_sum

        out_flat= self.mlp(shape_emb)      # => (B, 11*100)
        out_2d= out_flat.view(bsz,11,100)  
        return out_2d

def train_shape2spec_alone(
    csv_file,
    out_dir="outputs_stageA_shape2spec",
    d_model=128,
    nhead=4,
    num_layers=2,
    lr=1e-4,
    weight_decay=1e-5,
    batch_size=1024,
    num_epochs=50,
    split_ratio=0.8,
    grad_clip=1.0
):
    """
    Stage A: Train shape2spec alone. 
    We'll build a small dataset: input= shape(4,3), label= spectra(11,100).
    Minimizing MSE(spec_pred, spec_gt).
    """
    timestamp= datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir= f"{out_dir}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Stage A] Train shape2spec alone => outputs => {out_dir}")

    # Load dataset
    ds_full= Q1ShiftedSpectraDataset(csv_file, max_points=4)
    ds_len= len(ds_full)
    trn_len= int(ds_len*split_ratio)
    val_len= ds_len- trn_len
    ds_train, ds_val= random_split(ds_full, [trn_len, val_len])
    print(f"[DATA] total={ds_len}, train={trn_len}, val={val_len}")

    # We'll create a new wrapper dataset or simply invert the usage:
    # shape->spectrum means input= shape, label= spec
    # We can do that in the training loop or just reorder.
    # We'll do it in the training loop for convenience.

    train_loader= DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader=   DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False)

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model= ShapeToSpectraModel(
        d_in=3,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    ).to(device)

    optimizer= torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    crit_mse= nn.MSELoss()

    train_losses, val_losses= [], []

    for epoch in range(num_epochs):
        model.train()
        run_loss=0.0
        for (spectra_np, shape_np, uid_list) in train_loader:
            shape_t= shape_np.to(device)     # (B,4,3) => input
            spec_gt= spectra_np.to(device)   # (B,11,100) => target

            spec_pred= model(shape_t)
            loss_mse= crit_mse(spec_pred, spec_gt)

            optimizer.zero_grad()
            loss_mse.backward()

            if grad_clip>0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            run_loss+= loss_mse.item()

        avg_train= run_loss/ len(train_loader)
        train_losses.append(avg_train)

        # validation
        model.eval()
        val_sum=0.0
        val_count=0
        with torch.no_grad():
            for (spectra_np, shape_np, uid_list) in val_loader:
                bsz= shape_np.size(0)
                shape_t= shape_np.to(device)
                spec_gt= spectra_np.to(device)

                spec_pd= model(shape_t)
                l_mse= crit_mse(spec_pd, spec_gt)* bsz
                val_sum+= l_mse.item()
                val_count+= bsz
        avg_val= val_sum/ val_count
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if (epoch+1)%10==0 or epoch==0 or epoch==(num_epochs-1):
            print(f"[Stage A] Epoch[{epoch+1}/{num_epochs}] => TrainLoss={avg_train:.4f}, ValLoss={avg_val:.4f}")

    # Plot
    plt.figure()
    plt.plot(train_losses,label='train')
    plt.plot(val_losses,label='val')
    plt.xlabel("Epoch")
    plt.ylabel("MSE(spec_pred, spec_gt)")
    plt.title("Stage A: shape2spec alone")
    plt.legend()
    plt.savefig(os.path.join(out_dir,"training_curve_shape2spec.png"))
    plt.close()

    # final val
    final_val= val_losses[-1]
    print(f"[Stage A, Final Val Loss] => {final_val:.6f}")

    # save
    shape2spec_path= os.path.join(out_dir,"shape2spec_only.pt")
    torch.save(model.state_dict(), shape2spec_path)
    print("[Shape2Spec Only Model saved] =>", shape2spec_path)

    # We can return the final model or its path
    return shape2spec_path, final_val


###############################################################################
# 3) Spec2Shape with a FROZEN shape2spec in the chain
###############################################################################
class Spectra2ShapeVarLen(nn.Module):
    """
    spec(11,100)->shape(4,3)
    presence chaining => forced presence[0]=1, presence[i]= presence[i]*presence[i-1], i=1..3
    x,y => in [0,1]
    We do a simple set-like transformer. 
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

    def forward(self, spec_11x100):
        bsz= spec_11x100.size(0)
        x_proj= self.input_proj(spec_11x100)  # (B,11,d_model)
        x_enc= self.encoder(x_proj)
        x_agg= x_enc.mean(dim=1)              # permutation invariant

        out_12= self.mlp(x_agg)
        out_4x3= out_12.view(bsz,4,3)

        # presence chain
        presence_logits= out_4x3[:,:,0]
        xy_raw= out_4x3[:,:,1:]

        presence_list= []
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

class Spec2Shape2Spec_Frozen(nn.Module):
    """
    We combine:
      spec->shape (trainable)
      shape2spec (frozen)

    We just hold a reference to a shape2spec model that is frozen.
    """
    def __init__(self, spec2shape, shape2spec_frozen):
        super().__init__()
        self.spec2shape= spec2shape        # trainable
        self.shape2spec_frozen= shape2spec_frozen  # NOT trainable

    def forward(self, spec_input):
        shape_pred= self.spec2shape(spec_input)
        # chain => shape2spec
        with torch.no_grad():
            # shape2spec is frozen => no grad
            spec_chain= self.shape2spec_frozen(shape_pred)
        return shape_pred, spec_chain


def train_spec2shape_frozen_shape2spec(
    csv_file,
    shape2spec_ckpt,         # path to the shape2spec weights from Stage A
    out_dir="outputs_stageB_spec2shape_frozenS2S",
    d_model_spec2shape=128,
    nhead_spec2shape=4,
    num_layers_spec2shape=2,
    lr=1e-4,
    weight_decay=1e-5,
    batch_size=1024,
    num_epochs=50,
    split_ratio=0.8,
    grad_clip=1.0
):
    """
    Stage B:
     - We load shape2spec from shape2spec_ckpt, freeze it.
     - Build a pipeline spec->shape->(frozen)shape2spec => spec_pred
     - We train only the spec->shape part
     - Loss = MSE(shape_pred, shape_gt) + MSE(spec_pred, spec_gt)
       (where shape_pred => presence + x,y)
       (spec_pred is the output from frozen shape2spec)

    We'll log training, val, then do some final visualizations.
    """
    timestamp= datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir= f"{out_dir}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Stage B] Train spec->shape while shape2spec is frozen => outputs => {out_dir}")

    # 1) load shape2spec
    shape2spec_frozen= ShapeToSpectraModel(
        d_in=3,
        d_model=128,  # must match Stage A's config or pass them in
        nhead=4,
        num_layers=2
    )
    ckpt= torch.load(shape2spec_ckpt)
    shape2spec_frozen.load_state_dict(ckpt)
    print("[Stage B] shape2spec weights loaded from:", shape2spec_ckpt)

    # freeze
    for p in shape2spec_frozen.parameters():
        p.requires_grad= False

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape2spec_frozen.to(device)

    # 2) build spec2shape
    spec2shape= Spectra2ShapeVarLen(
        d_in=100,
        d_model=d_model_spec2shape,
        nhead=nhead_spec2shape,
        num_layers=num_layers_spec2shape
    ).to(device)

    # 3) combine => spec->shape->(frozen)shape2spec
    model= Spec2Shape2Spec_Frozen(spec2shape, shape2spec_frozen).to(device)

    # dataset
    ds_full= Q1ShiftedSpectraDataset(csv_file, max_points=4)
    ds_len= len(ds_full)
    trn_len= int(ds_len*split_ratio)
    val_len= ds_len- trn_len
    ds_train, ds_val= random_split(ds_full, [trn_len, val_len])
    print(f"[DATA] total={ds_len}, train={trn_len}, val={val_len}")

    train_loader= DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader=   DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer= torch.optim.AdamW(spec2shape.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    crit_mse= nn.MSELoss()

    train_losses, val_losses= [], []

    for epoch in range(num_epochs):
        model.train()
        run_loss=0.0
        for (spec_gt_np, shape_gt_np, uid_list) in train_loader:
            spec_gt= spec_gt_np.to(device)     # input
            shape_gt= shape_gt_np.to(device)

            shape_pred, spec_chain= model(spec_gt)
            # We do shape MSE => presence + x,y
            l_shape= crit_mse(shape_pred, shape_gt)
            # We do spec MSE => compare chain output vs GT
            l_spec= crit_mse(spec_chain, spec_gt)

            loss_total= l_shape + l_spec

            optimizer.zero_grad()
            loss_total.backward()

            if grad_clip>0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            run_loss+= loss_total.item()

        avg_train= run_loss/ len(train_loader)
        train_losses.append(avg_train)

        # validation
        model.eval()
        val_sum=0.0
        val_count=0
        with torch.no_grad():
            for (spec_gt_np, shape_gt_np, uid_list) in val_loader:
                bsz= spec_gt_np.size(0)
                spec_gt= spec_gt_np.to(device)
                shape_gt= shape_gt_np.to(device)

                shape_pd, spec_pd= model(spec_gt)
                vs= crit_mse(shape_pd, shape_gt) + crit_mse(spec_pd, spec_gt)
                val_sum+= vs.item()* bsz
                val_count+= bsz
        avg_val= val_sum/ val_count
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if (epoch+1)%10==0 or epoch==0 or epoch==(num_epochs-1):
            print(f"[Stage B] Epoch[{epoch+1}/{num_epochs}] => TrainLoss={avg_train:.4f}, ValLoss={avg_val:.4f}")

    # Plot
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses,   label='val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss = MSE(shape_pred, shape_gt) + MSE(spec_chain, spec_gt)")
    plt.title("Stage B: spec->shape with frozen shape2spec")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_curve_spec2shape.png"))
    plt.close()

    final_val= val_losses[-1]
    print(f"[Stage B, Final Val Loss] => {final_val:.6f}")

    # save
    # We'll only save the spec2shape since shape2spec is separate/frozen
    spec2shape_ckpt= os.path.join(out_dir, "spec2shape_only.pt")
    torch.save(spec2shape.state_dict(), spec2shape_ckpt)
    print("[Stage B spec2shape Model saved] =>", spec2shape_ckpt)

    # --- Visualization: let's do a final 3-col plot
    # Because shape2spec is frozen, we can do:
    #   pipeline => spec2shape => shape_pred => shape2spec_frozen => spec_pred
    # We'll define a small function below for that visualization.

    # We'll do a basic "samples_3col" style.
    visualize_3col_frozen(model, ds_val, device, out_dir=out_dir, sample_count=4, seed=42)


def visualize_3col_frozen(
    model, ds_val, device, out_dir=".", sample_count=4, seed=123
):
    """
    3 columns:
     - Left => GT spectrum (blue)
     - Middle => GT shape(green) vs Pred shape(red)
     - Right => original(blue) vs chain recon from pred shape (red dashed)
       (Note: we can't do shape2spec(gt_shape) because shape2spec is "frozen" 
        but we can still run it, it just won't have updated parameters. 
        We'll only show recon from predicted shape for clarity.)
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

            # left => GT spectrum
            axL= axes[i][0]
            for row_ in spec_gt:
                axL.plot(row_, color='blue', alpha=0.5)
            axL.set_title(f"UID={uid_}\n(Left) GT Spectrum (blue)")

            # middle => shape(green=gt, red=pred)
            axM= axes[i][1]
            presG= (shape_gt[:,0]>0.5)
            q1_g= shape_gt[presG,1:3]
            if len(q1_g)>0:
                axM.scatter(q1_g[:,0], q1_g[:,1], c='green', label='GT Q1')

            # run model => shape pred => chain spec
            spec_t= torch.tensor(spec_gt, dtype=torch.float32, device=device).unsqueeze(0)
            shape_pd, spec_pd= model(spec_t)
            shape_pd= shape_pd.cpu().numpy()[0]
            spec_pd= spec_pd.cpu().numpy()[0]

            presP= (shape_pd[:,0]>0.5)
            q1_p= shape_pd[presP,1:3]
            if len(q1_p)>0:
                axM.scatter(q1_p[:,0], q1_p[:,1], c='red', marker='x', label='Pred Q1')

            axM.set_xlim([-0.1,1.1])
            axM.set_ylim([-0.1,1.1])
            axM.set_aspect("equal","box")
            axM.grid(True)
            axM.legend()
            axM.set_title("(Middle)\nQ1 coords: GT(green) vs Pred(red)")

            # right => original(blue) vs chain recon(red dashed)
            axR= axes[i][2]
            for row_ in spec_gt:
                axR.plot(row_, color='blue', alpha=0.5)
            for row_ in spec_pd:
                axR.plot(row_, color='red', alpha=0.5, linestyle='--')
            axR.set_title("(Right)\nChain Recon from Pred Shape (red dashed)")
            axR.set_xlabel("Wavelength Index")
            axR.set_ylabel("Reflectance")

    plt.tight_layout()
    out_fig= os.path.join(out_dir,"samples_3col_plot_stageB.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"[Stage B Visualization saved] => {out_fig}")


###############################################################################
# 4) Main: Two-Stage
###############################################################################
def main():
    CSV_FILE= "merged_s4_shapes_iccpOv10kG40_seed88888.csv"

    # Stage A => train shape2spec alone
    shape2spec_ckpt, val_lossA = train_shape2spec_alone(
        csv_file= CSV_FILE,
        out_dir= "outputs_stageA_shape2spec",
        d_model=128,
        nhead=4,
        num_layers=2,
        lr=1e-4,
        weight_decay=1e-5,
        batch_size=1024,
        num_epochs=50,
        split_ratio=0.8,
        grad_clip=1.0
    )

    # Then Stage B => fix shape2spec, train spec2shape
    train_spec2shape_frozen_shape2spec(
        csv_file= CSV_FILE,
        shape2spec_ckpt= shape2spec_ckpt,
        out_dir= "outputs_stageB_spec2shape_frozenS2S",
        d_model_spec2shape=128,
        nhead_spec2shape=4,
        num_layers_spec2shape=2,
        lr=1e-4,
        weight_decay=1e-5,
        batch_size=1024,
        num_epochs=50,
        split_ratio=0.8,
        grad_clip=1.0
    )

if __name__=="__main__":
    main()

