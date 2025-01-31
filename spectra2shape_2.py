#!/usr/bin/env python3

"""
spectra2shape_transformer.py

- Subtract (0.5,0.5) from each shape => keep x>0,y>0 => up to 4 Q1 points => (4,3).
- Use a mini-Transformer encoder aggregator, with no positional embeddings => order-invariant.
- Output (4,3) => presence + x + y in [0,1].
- Train with optional geometric penalty to suppress the number of used points.
- Visualize final Q1 polygons by calling your shape_plot.py => Q1PolygonC4Plotter.

We encourage fewer points by presence ~ 0 if not used => 'geometric_penalty'.

Batch size=4096, as requested.
"""

import os
import sys
import numpy as np
import pandas as pd
import random
import matplotlib
matplotlib.use('Agg')  # to save figures without X
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# Import your shape_plot code:
from shape_plot import Q1PolygonC4Plotter

###############################################################################
# 1) SHIFT dataset => Q1 => up to 4
###############################################################################
class ShiftedQ1Dataset(Dataset):
    """
    For each shape: 
      - parse polygon => subtract (0.5,0.5) => keep x>0,y>0 => up to 4 => (4,3).
      - reflectances => (11,100).
    Must have total_v in {4,8,12,16} => each quadrant => 1..4 Q1 points.
    """

    def __init__(self, csv_file):
        super().__init__()
        self.df= pd.read_csv(csv_file)
        self.r_cols= [c for c in self.df.columns if c.startswith("R@")]
        self.df["gkey"]= (
            self.df[["prefix","nQ","shape_idx"]]
            .astype(str)
            .agg("_".join, axis=1)
        )

        self.data_list=[]
        grouped= self.df.groupby("gkey")
        for gk, grp in grouped:
            if len(grp)!=11:
                continue
            grp_s= grp.sort_values("c")
            # reflectances => shape(11,100)
            specs= grp_s[self.r_cols].values.astype(np.float32)

            row0= grp_s.iloc[0]
            vs= str(row0.get("vertices_str","")).strip()
            if not vs:
                continue

            all_xy=[]
            for pair in vs.split(";"):
                p= pair.strip()
                if p:
                    xy= p.split(",")
                    if len(xy)==2:
                        xval= float(xy[0])
                        yval= float(xy[1])
                        all_xy.append([xval,yval])
            all_xy= np.array(all_xy,dtype=np.float32)
            tot_v= len(all_xy)
            if tot_v not in [4,8,12,16]:
                continue

            # SHIFT => minus 0.5
            shifted= all_xy - 0.5

            # keep Q1 => x>0,y>0
            q1=[]
            for (xx,yy) in shifted:
                if xx>0 and yy>0:
                    q1.append([xx,yy])
            q1= np.array(q1,dtype=np.float32)
            if len(q1)==0 or len(q1)>4:
                # skip
                continue

            arr_43= np.zeros((4,3),dtype=np.float32)
            for i in range(len(q1)):
                arr_43[i,0]=1.0
                arr_43[i,1]= q1[i,0]
                arr_43[i,2]= q1[i,1]

            self.data_list.append({
                "gkey":gk,
                "specs":specs,
                "qverts":arr_43
            })

        self.data_len= len(self.data_list)
        if self.data_len==0:
            raise ValueError("No shapes remain after SHIFT->Q1 => up to 4.")

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        it= self.data_list[idx]
        return (it["specs"], it["qverts"], it["gkey"])

###############################################################################
# 2) Collate
###############################################################################
def my_collate(batch):
    specs_list= [b[0] for b in batch]  # shape(11,100)
    qv_list   = [b[1] for b in batch]  # shape(4,3)
    gk_list   = [b[2] for b in batch]

    specs_t= torch.from_numpy(np.stack(specs_list,axis=0)) # =>(bsz,11,100)
    qv_t   = torch.from_numpy(np.stack(qv_list,axis=0))    # =>(bsz,4,3)
    return specs_t, qv_t, gk_list

###############################################################################
# 3) Mini-Transformer aggregator (order-invariant: no positional embeddings)
###############################################################################
class TransformerSetEncoder(nn.Module):
    """
    We'll embed each reflectance token => (bsz,11,d_model).
    Then pass it through a TransformerEncoder with no positional embedding,
    so shuffling tokens does not change the result => order-invariant aggregator.

    We'll do a final average pooling over the 11 outputs => shape(bsz,d_model).
    """
    def __init__(self, d_in=100, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.d_model= d_model
        # linear embed => (100 -> d_model)
        self.input_proj= nn.Linear(d_in,d_model)

        # a transformer encoder
        enc_layer= nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*2,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.encoder= nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers
        )

    def forward(self,x):
        # x => (bsz,11,100)
        bsz, seq_len, feat_dim= x.size()
        # embed
        x_emb= self.input_proj(x) # => (bsz,11,d_model)

        # no positional embedding => purely set-based
        # We'll not pass any src_mask or key_padding_mask => it's fully accepted
        x_enc= self.encoder(x_emb) # => (bsz,11,d_model)

        # average pool across the 11 tokens => (bsz,d_model)
        lat= x_enc.mean(dim=1)
        return lat

###############################################################################
# 4) Full model => (4,3)
###############################################################################
class SpectraQ1Transformer(nn.Module):
    """
    1) TransformerSetEncoder => (bsz,d_model)
    2) MLP => (4,3) => presence + x + y in [0,1]
    """
    def __init__(self, d_in=100, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.encoder= TransformerSetEncoder(
            d_in=d_in,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )
        self.mlp= nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.ReLU(),
            nn.Linear(d_model,12)  # => 4*3
        )

    def forward(self,x):
        # x => (bsz,11,100)
        lat= self.encoder(x) # => (bsz,d_model)
        out_flat= self.mlp(lat) # => (bsz,12)
        out_3d= out_flat.view(-1,4,3)
        # presence => sigmoid
        pres= torch.sigmoid(out_3d[:,:,0])
        xy= torch.sigmoid(out_3d[:,:,1:])
        return torch.cat([pres.unsqueeze(-1), xy],dim=-1) # =>(bsz,4,3)

###############################################################################
# 5) Loss
###############################################################################
def shape_l1_loss(pred, tgt):
    return F.l1_loss(pred,tgt,reduction='mean')

def geometric_penalty(pred, alpha=0.7):
    presence= pred[:,:,0]  # (bsz,4)
    i_idx= torch.arange(4, device=presence.device).unsqueeze(0)
    weight= alpha**i_idx
    pen_batch= (presence*weight).sum(dim=1)
    return pen_batch.mean()

###############################################################################
# 6) Train + visualize
###############################################################################
def train_spectra2shape_transformer(
    csv_file,
    out_dir="my_s2s_q1_transformer",
    batch_size=4096,
    num_epochs=30,
    lr=1e-4,
    weight_decay=1e-5,
    alpha_geom=0.01,
    split_ratio=0.8,
    use_l1=True,
    d_model=128,
    nhead=4,
    num_layers=2
):
    os.makedirs(out_dir, exist_ok=True)
    ds_full= ShiftedQ1Dataset(csv_file)
    ds_len= len(ds_full)
    trn_len= int(ds_len*split_ratio)
    val_len= ds_len- trn_len
    ds_train, ds_val= random_split(ds_full,[trn_len,val_len])

    print(f"[DATA] total={ds_len},train={trn_len},val={val_len}, SHIFT=(0.5,0.5) sub")
    train_loader= DataLoader(ds_train,batch_size=batch_size,shuffle=True,drop_last=True,collate_fn=my_collate)
    val_loader= DataLoader(ds_val,batch_size=batch_size,shuffle=False,drop_last=False,collate_fn=my_collate)

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model= SpectraQ1Transformer(
        d_in=100,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    ).to(device)
    opt= torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched= torch.optim.lr_scheduler.ReduceLROnPlateau(opt,'min',factor=0.5,patience=5,verbose=True)
    base_loss= shape_l1_loss if use_l1 else lambda p,t: F.mse_loss(p,t,reduction='mean')
    loss_name= "L1" if use_l1 else "MSE"

    train_losses, val_losses= [],[]
    for ep in range(num_epochs):
        model.train()
        run_loss=0.0
        for (spec_t,qv_t,gkeys) in train_loader:
            spec_t= spec_t.to(device)
            qv_t= qv_t.to(device)
            out_t= model(spec_t)
            main_l= base_loss(out_t,qv_t)
            if alpha_geom>0:
                pen= geometric_penalty(out_t,alpha=0.7)
                loss= main_l + alpha_geom*pen
            else:
                loss= main_l

            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss+= loss.item()

        avg_trn= run_loss/len(train_loader)
        train_losses.append(avg_trn)

        # val
        model.eval()
        val_sum=0.0
        val_count=0
        with torch.no_grad():
            for (spec_t,qv_t,gkeys) in val_loader:
                bsz_= spec_t.size(0)
                spec_t= spec_t.to(device)
                qv_t= qv_t.to(device)
                pred_t= model(spec_t)
                main_l= base_loss(pred_t,qv_t)
                if alpha_geom>0:
                    pen= geometric_penalty(pred_t,0.7)
                    vloss= (main_l+ alpha_geom*pen)*bsz_
                else:
                    vloss= main_l*bsz_
                val_sum+= vloss.item()
                val_count+= bsz_
        avg_val= val_sum/val_count
        val_losses.append(avg_val)
        sched.step(avg_val)

        if (ep+1)%10==0 or ep==0:
            print(f"Epoch[{ep+1}/{num_epochs}] {loss_name}={avg_trn:.4f} Val={avg_val:.4f}")

    print(f"[Val] final {loss_name}={avg_val:.6f}")

    # plot train curve
    plt.figure()
    plt.plot(train_losses,label="train")
    plt.plot(val_losses,label="val")
    plt.xlabel("Epoch")
    plt.ylabel(f"{loss_name} Loss")
    plt.title("Spectra->Q1(shift) + 4-lim + presence (Transformer aggregator, no pos enc)")
    plt.legend()
    plt.savefig(os.path.join(out_dir,"train_curve.png"))
    plt.close()

    # save model
    mpath= os.path.join(out_dir,"spectra_transformer_q1.pt")
    torch.save(model.state_dict(), mpath)
    print("Model saved =>", mpath)

    # visualize
    visualize_transformer(model, ds_val, device, out_dir)


def visualize_transformer(model, ds_val, device, out_dir, num_show=4):
    """
    We'll:
      - pick random shapes
      - parse GT => presence + x + y
      - parse Pred => presence + x + y
      - Print them
      - pass them to shape_plot.py => Q1PolygonC4Plotter => replicate => angle-sort => draw
    """
    from shape_plot import Q1PolygonC4Plotter
    plotter= Q1PolygonC4Plotter()

    import random
    random_idxs= random.sample(range(len(ds_val)), min(num_show,len(ds_val)))
    model.eval()

    fig, axes= plt.subplots(len(random_idxs),2, figsize=(10,4*len(random_idxs)))
    if len(random_idxs)==1:
        axes=[axes]

    with torch.no_grad():
        for i, idx_ in enumerate(random_idxs):
            spec_np,qv_np,gkey= ds_val[idx_]
            # forward
            spec_t= torch.tensor(spec_np,dtype=torch.float32,device=device).unsqueeze(0)
            out_ = model(spec_t).cpu().numpy()[0]  # =>(4,3)

            # parse GT
            pres_gt= qv_np[:,0]
            xy_gt= qv_np[:,1:]
            GT_pts=[]
            for k in range(4):
                if pres_gt[k]>0.5:
                    GT_pts.append([xy_gt[k,0], xy_gt[k,1]])
            GT_pts= np.array(GT_pts,dtype=np.float32)

            # parse PD
            pres_pd= out_[:,0]
            xy_pd= out_[:,1:]
            PD_pts=[]
            for k in range(4):
                if pres_pd[k]>0.5:
                    PD_pts.append([xy_pd[k,0], xy_pd[k,1]])
            PD_pts= np.array(PD_pts,dtype=np.float32)

            print("====SHAPE:", gkey)
            print("   GT Q1 =>",GT_pts)
            print("   PD Q1 =>",PD_pts)

            # left => reflectances
            axL= axes[i][0]
            for row_ in spec_np:
                axL.plot(row_, alpha=0.5)
            axL.set_title(f"{gkey}\n(11×100) ignoring order (Transformer aggregator)")
            axL.set_xlabel("Wavelength idx")
            axL.set_ylabel("Reflectance")

            # right => replicate => Q1 => angle sort => polygon
            axR= axes[i][1]
            axR.set_aspect('equal','box')
            axR.grid(True)
            axR.set_title("C4 final: GT(green) vs PD(red)")

            if len(GT_pts)>0:
                plotter.plot_q1_c4(axR, GT_pts, color='green', alpha=0.4)
            if len(PD_pts)>0:
                plotter.plot_q1_c4(axR, PD_pts, color='red', alpha=0.4)

    plt.tight_layout()
    out_fig= os.path.join(out_dir,"sample_polygons.png")
    plt.savefig(out_fig)
    plt.close()
    print("[Visualization] =>", out_fig)


###############################################################################
# main
###############################################################################
def main():
    CSV_FILE= "merged_s4_shapes_iccpOv10kG40_seed88888.csv"
    train_spectra2shape_transformer(
        csv_file=CSV_FILE,
        out_dir="my_s2s_q1_transformer",
        batch_size=4096,
        num_epochs=100,
        lr=1e-4,
        weight_decay=1e-5,
        alpha_geom=0.01,
        split_ratio=0.8,
        use_l1=True,
        d_model=128,
        nhead=4,
        num_layers=2
    )

if __name__=="__main__":
    main()

