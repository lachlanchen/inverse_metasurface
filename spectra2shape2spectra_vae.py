#!/usr/bin/env python3

"""
Spectra->Shape->Spectra with a VAE approach:

1) We have two sub-models:
   - (A) Spectra2Latent => sample z => decode => Q1 shape => presence + x,y
   - (B) Shape2Spectra => reconstruct spectra from shape

2) We'll define a small VAE training loop:
   - Input: original spectra (11×100)
   - Encoder => (mu, logvar) => sample z
   - Decode z => shape(4,3)
   - (Optionally c4 replicate => or if the shape2spectra net expects full shape, we unify it)
   - shape2spectra => reconstructed spectra => recon loss vs. original

3) We'll add a geometric penalty on shape presence to gently suppress extra vertices.

4) We'll define "ShiftedQ1" logic if you have *both* spectra and shape in your CSV. 
   If not, just comment out the lines that handle GT shape or do a shorter pipeline.

5) Visualization: triple-plot for 1 random sample:
   - (Left) Original spectra vs. Reconstructed
   - (Center) GT shape vs. predicted shape in Q1
   - (Right) the c4-replicated polygons for GT vs. pred

We store final logs in "outputs/spectra2shape2spectra_<YYYYmmdd_HHMMSS>/" folder.
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# We assume your shape_plot.py has Q1PolygonC4Plotter or similar
from shape_plot import Q1PolygonC4Plotter

###############################################################################
# 1) Dataset example
###############################################################################
class SpectraShapeVAEDataset(Dataset):
    """
    Example dataset that has, for each shape:
      - spectra(11,100)
      - GT shape? => SHIFT by -0.5 => keep x>0,y>0 => up to 4 => (4,3)
    We'll store that shape for visualization or for shape-supervision (if you want).
    But here we mainly do an unsupervised approach: we won't do shape-based losses,
    just keep it for final plotting. 
    If you do not have GT shape in your CSV, remove references to 'qverts'.

    CSV structure assumption:
      - 11 rows per shape => c=0..1, sorted
      - "vertices_str"
      - reflectances => columns "R@..."
      - total v in [4,8,12,16]
    """
    def __init__(self, csv_file):
        super().__init__()
        self.df= pd.read_csv(csv_file)
        self.r_cols= [c for c in self.df.columns if c.startswith("R@")]
        # group key
        self.df["gkey"]= (
            self.df[["prefix","nQ","shape_idx"]]
            .astype(str).agg("_".join, axis=1)
        )

        self.data_list=[]
        grouped= self.df.groupby("gkey")
        for gk, grp in grouped:
            if len(grp)!=11:
                continue
            grp_s= grp.sort_values("c")
            # input spectra => shape(11,100)
            specs= grp_s[self.r_cols].values.astype(np.float32)

            # parse shape => SHIFT => Q1 => (4,3)
            row0= grp_s.iloc[0]
            vs= str(row0.get("vertices_str","")).strip()
            if not vs:
                # no shape data => you can skip or store None
                # here we skip
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

            # SHIFT => minus 0.5 => keep x>0,y>0 => up to 4
            shifted= all_xy - 0.5
            q1=[]
            for (xx,yy) in shifted:
                if xx>0 and yy>0:
                    q1.append([xx,yy])
            q1= np.array(q1,dtype=np.float32)
            if len(q1)==0 or len(q1)>4:
                # skip
                continue

            qv_43= np.zeros((4,3),dtype=np.float32)
            for i in range(len(q1)):
                qv_43[i,0]=1.0
                qv_43[i,1]= q1[i,0]
                qv_43[i,2]= q1[i,1]

            self.data_list.append({
                "gkey": gk,
                "specs": specs,   # (11,100)
                "qverts": qv_43  # (4,3)
            })

        self.data_len= len(self.data_list)
        if self.data_len==0:
            raise ValueError("No shapes remain after SHIFT->Q1 => up to 4 points")

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        it= self.data_list[idx]
        return (it["specs"], it["qverts"], it["gkey"])


def my_collate(batch):
    specs_list= [b[0] for b in batch]  # shape(11,100)
    qv_list   = [b[1] for b in batch]  # shape(4,3)
    gk_list   = [b[2] for b in batch]

    specs_t= torch.from_numpy(np.stack(specs_list, axis=0)) # =>(bsz,11,100)
    qv_t   = torch.from_numpy(np.stack(qv_list, axis=0))    # =>(bsz,4,3)
    return specs_t, qv_t, gk_list

###############################################################################
# 2) Sub-model (B) => shape->spectra
###############################################################################
class Shape2SpectraNet(nn.Module):
    """
    We do a simple aggregator:
      shape => presence + x + y => pass them via a small Transformer or MLP => produce (11,100).
    For simplicity, let's do a small MLP aggregator ignoring token order (like a deep sets).
    Or you can use a mini-Transformer aggregator for shape->spectra as well.
    We'll do a mini MLP ignoring order for brevity.
    """
    def __init__(self, d_in=3, max_verts=4, d_model=128):
        super().__init__()
        # We'll flatten the 4*(presence,x,y)=4*3=12 => mlp => (11*100=1100)
        self.out_dim= 11*100
        self.mlp= nn.Sequential(
            nn.Linear(max_verts*d_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model,d_model),
            nn.ReLU(),
            nn.Linear(d_model,self.out_dim)
        )
    def forward(self, shape_43):
        """
        shape_43 => (bsz,4,3)
        We'll flatten => (bsz,12)
        => MLP => (bsz,1100) => reshape =>(bsz,11,100)
        """
        bsz= shape_43.size(0)
        x_flat= shape_43.view(bsz,-1) # =>(bsz,12)
        out_flat= self.mlp(x_flat)
        out_2d= out_flat.view(bsz,11,100)
        return out_2d

###############################################################################
# 3) Sub-model (A) => spectra->latent z => decode shape
###############################################################################
class SpectraEncoderNet(nn.Module):
    """
    We'll embed each reflectance => d_model
    Then do an order-invariant aggregator => produce (mu, logvar) => z dimension.
    """
    def __init__(self, d_in=100, n_tokens=11, d_model=128, z_dim=16):
        super().__init__()
        self.z_dim= z_dim
        self.input_proj= nn.Linear(d_in,d_model)

        # aggregator = sum or average ignoring order
        self.post_enc= nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        # final => (mu, logvar)
        self.mu_head= nn.Linear(d_model, z_dim)
        self.logvar_head= nn.Linear(d_model, z_dim)

    def forward(self, specs_2d):
        """
        specs_2d => (bsz, 11,100)
        We'll do a simple sum aggregator ignoring order. 
        For a more advanced approach, you can do a mini-Transformer aggregator here.
        """
        bsz= specs_2d.size(0)
        # embed => (bsz,11,d_model)
        emb= self.input_proj(specs_2d)
        # sum => (bsz,d_model)
        summed= emb.sum(dim=1)
        lat= self.post_enc(summed)
        mu= self.mu_head(lat)
        logvar= self.logvar_head(lat)
        return mu, logvar

class ShapeDecoderNet(nn.Module):
    """
    z => shape(4,3) => presence + x + y in [0,1]
    We'll do a simple MLP from z_dim => (4*3)
    """
    def __init__(self, z_dim=16, d_hidden=128):
        super().__init__()
        self.mlp= nn.Sequential(
            nn.Linear(z_dim,d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden,d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden,4*3)
        )
    def forward(self,z):
        bsz= z.size(0)
        out_flat= self.mlp(z)  # =>(bsz,12)
        out_3d= out_flat.view(bsz,4,3)
        pres_logit= out_3d[:,:,0]        # =>(bsz,4)
        pres= torch.sigmoid(pres_logit)
        xy_raw= out_3d[:,:,1:]          # =>(bsz,4,2)
        xy= torch.sigmoid(xy_raw)       # =>(bsz,4,2)
        return torch.cat([pres.unsqueeze(-1), xy],dim=-1) # =>(bsz,4,3)

###############################################################################
# 4) The VAE
###############################################################################
class Spectra2Shape2SpectraVAE(nn.Module):
    """
    Full pipeline:
      - encoder(spectra)->(mu,logvar)-> sample z
      - shape decoder => shape(4,3)
      - shape2spectra => reconstruct(11,100)

    We do a forward pass returning (recon_spectra, mu, logvar, shape_43).
    """
    def __init__(self, 
                 d_in=100,        # reflectance dimension
                 d_model=128,
                 z_dim=16,
                 # aggregator tokens=11
                 shape2spectra_dim=128
                 ):
        super().__init__()
        self.encoder= SpectraEncoderNet(d_in=100, n_tokens=11, d_model=d_model, z_dim=z_dim)
        self.decoder= ShapeDecoderNet(z_dim=z_dim, d_hidden=d_model)
        self.shape2spectra= Shape2SpectraNet(d_in=3, max_verts=4, d_model=shape2spectra_dim)

    def forward(self, specs_2d):
        """
        specs_2d => (bsz,11,100)
        Return => recon_spectra, mu, logvar, shape(4,3)
        """
        mu, logvar= self.encoder(specs_2d)
        z= self._reparam(mu, logvar)
        shape_43= self.decoder(z)              # =>(bsz,4,3)
        recon= self.shape2spectra(shape_43)    # =>(bsz,11,100)
        return recon, mu, logvar, shape_43

    @staticmethod
    def _reparam(mu, logvar):
        """
        sample z ~ N(mu, exp(logvar))
        """
        eps= torch.randn_like(mu)
        std= (0.5*logvar).exp()
        return mu + eps*std

###############################################################################
# 5) VAE losses
###############################################################################
def kl_divergence(mu, logvar):
    """
    KL( q(z|x) || p(z) ), p(z)=N(0,I).
    => 0.5 * sum( exp(logvar) + mu^2 -1 - logvar ) over z_dim
    We do mean over batch => no dimension-based scaling.
    """
    # shape(bsz, z_dim)
    kl= 0.5* torch.sum(torch.exp(logvar)+ mu**2 -1.0 - logvar, dim=1)
    return kl.mean()

def recon_mse_loss(recon, target):
    """
    MSE over (bsz,11,100)
    """
    return F.mse_loss(recon, target, reduction='mean')

def shape_geometric_penalty(shape_43, alpha=0.7):
    """
    shape_43 =>(bsz,4,3). presence => shape_43[:,:,0].
    Encourage fewer vertices.
    """
    p= shape_43[:,:,0]
    i_idx= torch.arange(4, device=p.device).unsqueeze(0)
    weight= alpha**i_idx
    pen_batch= (p*weight).sum(dim=1)
    return pen_batch.mean()

###############################################################################
# 6) The training loop
###############################################################################
def train_vae_spectra2shape2spectra(
    csv_file,
    out_dir="outputs/spectra2shape2spectra",
    batch_size=4096,
    num_epochs=500,
    lr=1e-4,
    weight_decay=1e-5,
    alpha_geom=0.001,
    beta_kl=1.0,
    split_ratio=0.8,
    d_model=128,
    z_dim=16,
    shape2spectra_dim=128
):
    # unique folder
    from datetime import datetime
    stamp= datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir= f"{out_dir}_{stamp}"
    os.makedirs(out_dir, exist_ok=True)

    # dataset
    ds_full= SpectraShapeVAEDataset(csv_file)
    ds_len= len(ds_full)
    trn_len= int(ds_len*split_ratio)
    val_len= ds_len-trn_len
    ds_train, ds_val= random_split(ds_full, [trn_len,val_len])
    print(f"[DATA] total={ds_len},train={trn_len},val={val_len}, SHIFT=Yes(Q1)")

    train_loader= DataLoader(ds_train,batch_size=batch_size,shuffle=True,drop_last=True,collate_fn=my_collate)
    val_loader=   DataLoader(ds_val,batch_size=batch_size,shuffle=False,drop_last=False,collate_fn=my_collate)

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # define model
    model= Spectra2Shape2SpectraVAE(
        d_in=100,
        d_model=d_model,
        z_dim=z_dim,
        shape2spectra_dim=shape2spectra_dim
    ).to(device)

    opt= torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # no scheduler or you can do so
    # e.g. sched= torch.optim.lr_scheduler.ReduceLROnPlateau(...)

    # logs
    recon_losses, kl_losses, train_losses, val_losses= [],[],[],[]

    def compute_loss(spec_t, qv_t):
        # Forward pass => recon, mu, logvar, shape_43
        recon, mu, logvar, shape_43= model(spec_t)
        # recon => (bsz,11,100)
        # 1) recon MSE
        rec_mse= recon_mse_loss(recon, spec_t)
        # 2) kl
        kl= kl_divergence(mu, logvar)
        # 3) shape presence penalty
        if alpha_geom>0:
            pen= shape_geometric_penalty(shape_43, alpha=0.7)
        else:
            pen= torch.tensor(0.0, device=spec_t.device)
        # total
        total= rec_mse + beta_kl*kl + alpha_geom*pen
        return total, rec_mse, kl, pen, shape_43, recon

    # training loop
    for ep in range(num_epochs):
        model.train()
        run_recon=0.0
        run_kl=0.0
        run_tot=0.0
        num_batch=0
        for (spec_t, qv_t, gk) in train_loader:
            bsz_= spec_t.size(0)
            spec_t= spec_t.to(device)
            qv_t= qv_t.to(device) # only for debugging or shape-based losses if you want
            total, rec_mse, kl, pen, shape_out, recon_sp= compute_loss(spec_t,qv_t)

            opt.zero_grad()
            total.backward()
            opt.step()

            run_recon+= rec_mse.item()
            run_kl+= kl.item()
            run_tot+= total.item()
            num_batch+= 1

        avg_recon= run_recon/num_batch
        avg_kl= run_kl/num_batch
        avg_tot= run_tot/num_batch
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)
        train_losses.append(avg_tot)

        # validation
        model.eval()
        val_sum=0.0
        val_count=0
        with torch.no_grad():
            for (spec_t, qv_t, gk) in val_loader:
                bsz_= spec_t.size(0)
                spec_t= spec_t.to(device)
                total, rec_mse_, kl_, pen_, shape_out_, recon_sp_ = compute_loss(spec_t,qv_t)
                val_sum+= total.item()*bsz_
                val_count+= bsz_
        val_losses.append(val_sum/val_count)

        if (ep+1)%10==0 or ep==0:
            print(f"Epoch[{ep+1}/{num_epochs}] tot={avg_tot:.4f} rec={avg_recon:.4f} kl={avg_kl:.4f} val={val_losses[-1]:.4f}")

    # finalize
    # plot
    plt.figure()
    plt.plot(train_losses,label="train total")
    plt.plot(val_losses,label="val total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("VAE total Loss")
    plt.savefig(os.path.join(out_dir,"train_loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(recon_losses,label="train recon MSE")
    plt.plot(kl_losses,label="train KL")
    plt.legend()
    plt.title("Train recon & KL")
    plt.savefig(os.path.join(out_dir,"train_recon_kl.png"))
    plt.close()

    # save model
    model_path= os.path.join(out_dir,"vae_model.pt")
    torch.save(model.state_dict(), model_path)
    print("Model saved =>", model_path)

    # final visualize
    visualize_vae(model, ds_val, device, out_dir)

def visualize_vae(model, ds_val, device, out_dir, num_show=4):
    """
    We'll do triple-plot for each sample:
      1) (Left) Original vs recon spectra
      2) (Center) GT shape vs pred shape in Q1
      3) (Right) c4 replicate polygons
    """
    from shape_plot import Q1PolygonC4Plotter
    plotter= Q1PolygonC4Plotter()

    import random
    random_idxs= random.sample(range(len(ds_val)), min(num_show,len(ds_val)))
    model.eval()

    fig, axes= plt.subplots(len(random_idxs), 3, figsize=(15,5*len(random_idxs)))
    if len(random_idxs)==1:
        axes=[axes]

    with torch.no_grad():
        for i, idx_ in enumerate(random_idxs):
            spec_np, qv_np, gk= ds_val[idx_]
            # forward
            spec_t= torch.tensor(spec_np,dtype=torch.float32,device=device).unsqueeze(0)
            recon_sp, mu_, logvar_, shape_out= model(spec_t)
            recon_sp= recon_sp.cpu().numpy()[0]   # =>(11,100)
            shape_out= shape_out.cpu().numpy()[0] # =>(4,3)

            # parse GT shape
            pres_gt= qv_np[:,0]
            xy_gt= qv_np[:,1:]
            GT_used=[]
            for kk in range(4):
                if pres_gt[kk]>0.5:
                    GT_used.append([xy_gt[kk,0], xy_gt[kk,1]])
            GT_used= np.array(GT_used,dtype=np.float32)

            # parse PD shape
            pres_pd= shape_out[:,0]
            xy_pd= shape_out[:,1:]
            PD_used=[]
            for kk in range(4):
                if pres_pd[kk]>0.5:
                    PD_used.append([xy_pd[kk,0], xy_pd[kk,1]])
            PD_used= np.array(PD_used,dtype=np.float32)

            print("=== Sample idx=", idx_, "gk=", gk)
            print("   GT Q1 =>", GT_used)
            print("   PD Q1 =>", PD_used)

            # subplot(0) => left => original vs recon spectra
            axL= axes[i][0]
            # original => spec_np => shape(11,100)
            # recon => recon_sp => shape(11,100)
            # plot them
            for j in range(11):
                axL.plot(spec_np[j], color='blue', alpha=0.3)
                axL.plot(recon_sp[j], color='red', alpha=0.3, linestyle='--')
            axL.set_title("Original vs Recon Spectra")
            axL.set_xlabel("Wave idx")
            axL.set_ylabel("Reflectance")

            # subplot(1) => center => GT shape vs pred shape in Q1
            axC= axes[i][1]
            axC.set_title("Q1 shapes: GT(green), PD(red)")
            axC.set_aspect('equal','box')
            axC.grid(True)
            # just scatter them or do a small polygon if you want
            # we can do a polygon if we angle-sort
            from shape_plot import PointPolygonPlotter
            # GT
            if len(GT_used)>1:
                sorted_gt= PointPolygonPlotter.angle_sort(GT_used)
                PointPolygonPlotter.plot_polygon(axC, sorted_gt, color='green', alpha=0.3)
            elif len(GT_used)==1:
                axC.scatter(GT_used[0,0], GT_used[0,1], c='g')
            # PD
            if len(PD_used)>1:
                sorted_pd= PointPolygonPlotter.angle_sort(PD_used)
                PointPolygonPlotter.plot_polygon(axC, sorted_pd, color='red', alpha=0.3)
            elif len(PD_used)==1:
                axC.scatter(PD_used[0,0], PD_used[0,1], c='r')

            # subplot(2) => right => c4 replicate polygons
            axR= axes[i][2]
            axR.set_title("C4 polygons: GT(green), PD(red)")
            axR.set_aspect('equal','box')
            axR.grid(True)

            # replicate c4 => angle-sort => polygon
            if len(GT_used)>0:
                plotter.plot_q1_c4(axR, GT_used, color='green', alpha=0.4)
            if len(PD_used)>0:
                plotter.plot_q1_c4(axR, PD_used, color='red', alpha=0.4)

    plt.tight_layout()
    fig_out= os.path.join(out_dir,"sample_triple_plot.png")
    plt.savefig(fig_out)
    plt.close()
    print("Wrote triple-plot to", fig_out)

###############################################################################
def main():
    CSV_FILE= "merged_s4_shapes_iccpOv10kG40_seed88888.csv"
    from datetime import datetime
    train_vae_spectra2shape2spectra(
        csv_file=CSV_FILE,
        out_dir="outputs/spectra2shape2spectra",
        batch_size=4096,
        num_epochs=500,
        lr=1e-4,
        weight_decay=1e-5,
        alpha_geom=0.001,   # gentle
        beta_kl=1.0,        # standard VAE
        split_ratio=0.8,
        d_model=128,
        z_dim=16,
        shape2spectra_dim=128
    )

if __name__=="__main__":
    main()

