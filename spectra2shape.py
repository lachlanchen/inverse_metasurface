#!/usr/bin/env python3

"""
spectra2shape.py

- We SHIFT each shape by subtracting 0.5 (the dataset presumably had +0.5).
- We keep x>0,y>0 => up to 4 "Q1" points => store in (4,3).
- Train a model ignoring the 11-sample order (DeepSets).
- For visualization, we:
   1) Gather the GT Q1 points that have presence=1
   2) Gather the Pred Q1 points that have presence>0.5
   3) Print them to console
   4) Pass them *directly* to shape_plot.py's Q1PolygonC4Plotter -> 
      replicate around (0,0) -> angle-sort -> polygon -> final figure.

Hence the final shapes appear around origin, same as shape_plot.py's standard usage.
"""

import os
import sys
import numpy as np
import pandas as pd
import random
import matplotlib

matplotlib.use('Agg')  # so we can save figures headless
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# 1) Import your "good" shape_plot code
from shape_plot import Q1PolygonC4Plotter

###############################################################################
# 2) SHIFT dataset => Q1 => up to 4
###############################################################################
class ShiftedQ1Dataset(Dataset):
    """
    Subtract (0.5,0.5). Keep x>0,y>0 => up to 4 => (presence,x,y).
    Must have total_v in {4,8,12,16}, so Q1 has 1..4 points.
    Reflectances => (11,100).
    """
    def __init__(self, csv_file):
        super().__init__()
        self.df= pd.read_csv(csv_file)
        self.r_cols= [c for c in self.df.columns if c.startswith("R@")]
        self.df["gkey"]= (
            self.df[["prefix","nQ","shape_idx"]].astype(str)
            .agg("_".join, axis=1)
        )

        self.data_list=[]
        grouped= self.df.groupby("gkey")
        for gk, grp in grouped:
            if len(grp)!=11:
                continue
            # sort by c
            grp_s= grp.sort_values("c")
            # reflectances => shape(11,100)
            specs= grp_s[self.r_cols].values.astype(np.float32)

            row0= grp_s.iloc[0]
            v_str= str(row0.get("vertices_str","")).strip()
            if not v_str:
                continue

            # parse
            all_xy=[]
            for pair in v_str.split(";"):
                pair= pair.strip()
                if pair:
                    xy= pair.split(",")
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

            # Q1 => x>0,y>0
            q1_pts=[]
            for (xx,yy) in shifted:
                if xx>0 and yy>0:
                    q1_pts.append([xx,yy])
            q1_pts= np.array(q1_pts,dtype=np.float32)

            # must have 1..4
            if len(q1_pts)==0 or len(q1_pts)>4:
                continue

            arr_43= np.zeros((4,3),dtype=np.float32)
            for i in range(len(q1_pts)):
                arr_43[i,0]=1.0
                arr_43[i,1]= q1_pts[i,0]
                arr_43[i,2]= q1_pts[i,1]

            self.data_list.append({
                "gkey":gk,
                "specs": specs,
                "qverts": arr_43
            })

        self.data_len= len(self.data_list)
        if self.data_len==0:
            raise ValueError("No shapes remain after SHIFT->Q1.")
    def __len__(self):
        return self.data_len
    def __getitem__(self, idx):
        it= self.data_list[idx]
        return (it["specs"], it["qverts"], it["gkey"])

###############################################################################
# 3) Collate
###############################################################################
def my_collate(batch):
    # (specs(11,100), qverts(4,3), gkey)
    specs_list= [b[0] for b in batch]
    qverts_list= [b[1] for b in batch]
    gkey_list= [b[2] for b in batch]

    specs_t= torch.from_numpy(np.stack(specs_list,axis=0)) # => (bsz,11,100)
    qv_t   = torch.from_numpy(np.stack(qverts_list,axis=0))# => (bsz,4,3)
    return specs_t, qv_t, gkey_list

###############################################################################
# 4) Model
###############################################################################
class DeepSetsEncoder(nn.Module):
    def __init__(self, d_in=100, d_model=128):
        super().__init__()
        self.net= nn.Sequential(
            nn.Linear(d_in,d_model),
            nn.ReLU(),
            nn.Linear(d_model,d_model),
            nn.ReLU()
        )
        self.post= nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.ReLU()
        )
    def forward(self, x):
        # x => (bsz,11,100)
        emb= self.net(x)  # => (bsz,11,d_model)
        summed= emb.sum(dim=1) # => (bsz,d_model)
        return self.post(summed)

class SpectraQ1Net(nn.Module):
    """
    outputs => (4,3) => presence + x + y in [0,1]
    """
    def __init__(self,d_in=100, d_model=128):
        super().__init__()
        self.enc= DeepSetsEncoder(d_in,d_model)
        self.dec= nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.ReLU(),
            nn.Linear(d_model,12) # =>4*3
        )
    def forward(self, x):
        bsz= x.size(0)
        lat= self.enc(x)  # =>(bsz,d_model)
        out_flat= self.dec(lat) # =>(bsz,12)
        out_3d= out_flat.view(bsz,4,3)
        pres_logit= out_3d[:,:,0]
        pres= torch.sigmoid(pres_logit)
        xy_raw= out_3d[:,:,1:]
        xy= torch.sigmoid(xy_raw)
        return torch.cat([pres.unsqueeze(-1),xy],dim=-1) # =>(bsz,4,3)

###############################################################################
# 5) Loss
###############################################################################
def shape_l1_loss(pred, tgt):
    return F.l1_loss(pred,tgt,reduction='mean')

def geometric_penalty(pred, alpha=0.7):
    # presence => pred[:,:,0]
    presence= pred[:,:,0]
    i_idx= torch.arange(4, device=presence.device).unsqueeze(0)
    weight= alpha**i_idx
    pen_batch= (presence*weight).sum(dim=1)
    return pen_batch.mean()

###############################################################################
# 6) Train + visualize
###############################################################################
def train_spectra2shape(
    csv_file,
    out_dir="my_s2s_q1c4_final",
    batch_size=4096,
    num_epochs=30,
    lr=1e-4,
    weight_decay=1e-5,
    alpha_geom=0.01,
    split_ratio=0.8,
    use_l1=True
):
    os.makedirs(out_dir, exist_ok=True)
    ds_full= ShiftedQ1Dataset(csv_file)
    ds_len= len(ds_full)
    trn_len= int(ds_len*split_ratio)
    val_len= ds_len-trn_len
    ds_train, ds_val= random_split(ds_full,[trn_len,val_len])
    print(f"[DATA] total={ds_len},train={trn_len},val={val_len}, SHIFT=Yes")

    train_loader= DataLoader(ds_train,batch_size=batch_size,shuffle=True,drop_last=True,collate_fn=my_collate)
    val_loader=   DataLoader(ds_val,batch_size=batch_size,shuffle=False,drop_last=False,collate_fn=my_collate)

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model= SpectraQ1Net(d_in=100,d_model=128).to(device)
    opt= torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched= torch.optim.lr_scheduler.ReduceLROnPlateau(opt,'min',factor=0.5,patience=5,verbose=True)

    base_loss= shape_l1_loss if use_l1 else lambda p,t: F.mse_loss(p,t,reduction='mean')
    loss_name= "L1" if use_l1 else "MSE"

    train_losses, val_losses=[],[]
    for ep in range(num_epochs):
        model.train()
        run_loss=0.0
        for (spec_t,qv_t,gkeys) in train_loader:
            spec_t= spec_t.to(device)
            qv_t= qv_t.to(device)
            out_t= model(spec_t)
            base_l= base_loss(out_t,qv_t)
            if alpha_geom>0:
                pen= geometric_penalty(out_t,0.7)
                loss= base_l + alpha_geom*pen
            else:
                loss= base_l

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
                out_v= model(spec_t)
                b_l= base_loss(out_v,qv_t)
                if alpha_geom>0:
                    pen= geometric_penalty(out_v,0.7)
                    vloss= (b_l+ alpha_geom*pen)*bsz_
                else:
                    vloss= b_l*bsz_
                val_sum+= vloss.item()
                val_count+= bsz_
        avg_val= val_sum/val_count
        val_losses.append(avg_val)
        sched.step(avg_val)

        if (ep+1)%10==0 or ep==0:
            print(f"Epoch[{ep+1}/{num_epochs}] {loss_name}={avg_trn:.4f} Val={avg_val:.4f}")

    print(f"[Val] final {loss_name}={avg_val:.6f}")
    # plot
    plt.figure()
    plt.plot(train_losses,label="train")
    plt.plot(val_losses,label="val")
    plt.xlabel("Epoch")
    plt.ylabel(f"{loss_name} Loss")
    plt.title("Spectra -> SHIFT( -0.5 ) => Q1 => up to 4 corners => replicateC4")
    plt.legend()
    plt.savefig(os.path.join(out_dir,"train_curve.png"))
    plt.close()

    # save
    mp= os.path.join(out_dir,"spectra_q1_model.pt")
    torch.save(model.state_dict(), mp)
    print("Model saved:", mp)

    # visualize
    visualize_results(model, ds_val, device, out_dir)

def visualize_results(model, ds_val, device, out_dir, num_show=4):
    """
    We'll pick random shapes from ds_val, parse GT Q1 => parse Pred => print them,
    then call Q1PolygonC4Plotter(...) to replicate + angle-sort => polygon => done.
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
            spec_np, qv_np, gkey= ds_val[idx_]
            # forward
            spec_t= torch.tensor(spec_np, dtype=torch.float32, device=device).unsqueeze(0)
            out_ = model(spec_t).cpu().numpy()[0]  # => shape(4,3)

            # parse GT
            pres_gt= qv_np[:,0]
            xy_gt= qv_np[:,1:]
            GT_points=[]
            for k in range(4):
                if pres_gt[k]>0.5:
                    GT_points.append([xy_gt[k,0], xy_gt[k,1]])
            GT_points= np.array(GT_points, dtype=np.float32)

            # parse Pred
            pres_pd= out_[:,0]
            xy_pd= out_[:,1:]
            PD_points=[]
            for k in range(4):
                if pres_pd[k]>0.5:
                    PD_points.append([xy_pd[k,0], xy_pd[k,1]])
            PD_points= np.array(PD_points, dtype=np.float32)

            print("==== SHAPE:", gkey)
            print("  GT Q1 points =>", GT_points)
            print("  PD Q1 points =>", PD_points)

            # left => reflectances
            axL= axes[i][0]
            for row_ in spec_np:
                axL.plot(row_, alpha=0.5)
            axL.set_title(f"{gkey}\n(11×100) ignoring order")
            axL.set_xlabel("Wave idx")
            axL.set_ylabel("Reflectance")

            # right => replicate + angle sort => polygon => from shape_plot
            axR= axes[i][1]
            axR.set_title("C4 final: GT(green) vs Pred(red)")
            axR.set_aspect('equal','box')
            axR.grid(True)

            if len(GT_points)>0:
                # The Q1Plotter expects all x>0,y>0
                # We'll just call: plotter.plot_q1_c4(axR, GT_points, color='green')
                plotter.plot_q1_c4(axR, GT_points, color='green', alpha=0.4)

            if len(PD_points)>0:
                plotter.plot_q1_c4(axR, PD_points, color='red', alpha=0.4)

    plt.tight_layout()
    out_png= os.path.join(out_dir,"sample_polygons.png")
    plt.savefig(out_png)
    plt.close()
    print(f"[Visualization] => {out_png}")

###############################################################################
# if main
###############################################################################
def main():
    CSV_FILE= "merged_s4_shapes_iccpOv10kG40_seed88888.csv"
    train_spectra2shape(
        csv_file=CSV_FILE,
        out_dir="my_s2s_q1c4_final",
        batch_size=4096,
        num_epochs=30,
        lr=1e-4,
        weight_decay=1e-5,
        alpha_geom=0.01,
        split_ratio=0.8,
        use_l1=True
    )

if __name__=="__main__":
    main()

