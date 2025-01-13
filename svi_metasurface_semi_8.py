#!/usr/bin/env python3

"""
Semi-Supervised AIR with Forced First Vertex in 1st Quadrant
-----------------------------------------------------------
- Reflection columns: R@... => 100 points
- c in [0,1] partially labeled => keep 70%
- Up to 4 vertices, each forced into first quadrant (x>=0,y>=0) by softplus transform
- The first vertex presence is forced=1
- We fix the shape mismatch by .mask(...) with shape [B], not [B,1].
- Trains 100 epochs, then does inference on a random 100-pt reflection
- Saves data & plots to a date-stamped folder
"""

import os
import math
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceGraph_ELBO
import pyro.optim as optim
from pyro.poutine import trace
from tqdm import trange

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------ Config ------------------
MAX_STEPS   = 4
N_WAVES     = 100
MASK_FRAC   = 0.7
NUM_EPOCHS  = 100
BATCH_SIZE  = 512
LR          = 1e-3

# We'll create a time-stamped subfolder:
TIMESTAMP   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR    = f"results_forcedFirstVtx_{TIMESTAMP}"
os.makedirs(SAVE_DIR, exist_ok=True)

###############################################################################
# 1) Data Loading + Semi-Supervision
###############################################################################
def load_data(csv_path):
    """
    CSV expected with:
      R@... => 100 columns reflection
      c => [0..1] optional partial
      v_pres_0..3 => presence
      v_where_0..3_{x,y} => location in first quadrant
    Returns:
      spectra: [N,100]
      c_vals: [N,1]
      is_c_known: bool [N]
      v_pres: [N,4]
      v_where: [N,4,2]
      is_v_known: bool [N]
    """
    df = pd.read_csv(csv_path)

    # Reflection
    r_cols = [col for col in df.columns if col.startswith("R@")]
    r_cols = sorted(r_cols, key=lambda x: float(x.split('@')[1]))
    spectra_np = df[r_cols].values.astype(np.float32)
    spectra     = torch.from_numpy(spectra_np)
    N, n_waves  = spectra.shape
    if n_waves != N_WAVES:
        print(f"[WARN] found {n_waves} reflection columns, expected {N_WAVES}.")

    # c
    c_np = np.full((N,1), np.nan, dtype=np.float32)
    if "c" in df.columns:
        c_np = df["c"].values.reshape(-1,1).astype(np.float32)
    is_c_known_np = ~np.isnan(c_np).reshape(-1)
    c_np[np.isnan(c_np)] = 0.0
    c_vals     = torch.from_numpy(c_np)
    is_c_known = torch.from_numpy(is_c_known_np)

    # v
    pres_cols = [f"v_pres_{t}" for t in range(MAX_STEPS)]
    wx_cols   = [f"v_where_{t}_x" for t in range(MAX_STEPS)]
    wy_cols   = [f"v_where_{t}_y" for t in range(MAX_STEPS)]
    have_pres = all(pc in df.columns for pc in pres_cols)
    have_where= all((wx in df.columns and wy in df.columns) for wx,wy in zip(wx_cols, wy_cols))

    if have_pres and have_where:
        v_pres_np= df[pres_cols].values.astype(np.float32) # [N,4]
        vx_np    = df[wx_cols].values.astype(np.float32)   # [N,4]
        vy_np    = df[wy_cols].values.astype(np.float32)   # [N,4]
        row_nan  = (np.isnan(v_pres_np).any(axis=1)|
                    np.isnan(vx_np).any(axis=1)|
                    np.isnan(vy_np).any(axis=1))
        is_v_known_np= ~row_nan
        v_pres_np[np.isnan(v_pres_np)] = 0
        vx_np[np.isnan(vx_np)] = 0
        vy_np[np.isnan(vy_np)] = 0
        v_pres = torch.from_numpy(v_pres_np)
        v_where= torch.stack([torch.from_numpy(vx_np),
                              torch.from_numpy(vy_np)], dim=-1) # [N,4,2]
        is_v_known= torch.from_numpy(is_v_known_np)
    else:
        v_pres    = torch.zeros(N, MAX_STEPS)
        v_where   = torch.zeros(N, MAX_STEPS,2)
        is_v_known= torch.zeros(N, dtype=torch.bool)

    return spectra, c_vals, is_c_known, v_pres, v_where, is_v_known

def apply_semi_mask(is_c_known, is_v_known, fraction=MASK_FRAC, seed=123):
    """
    Keep 'fraction' of known => True, rest => False
    """
    N = len(is_c_known)
    np.random.seed(seed)
    rand_vals= np.random.rand(N)
    keep = rand_vals < fraction
    final_c= is_c_known.numpy() & keep
    final_v= is_v_known.numpy() & keep
    return torch.from_numpy(final_c), torch.from_numpy(final_v)

###############################################################################
# 2) Model + Guide
###############################################################################
class Decoder(nn.Module):
    """
    c + up to 4 first-quadrant vertices => reflection(100)
    We'll embed each vertex(2D) -> 64D, sum, cat with c => final MLP => 100D
    """
    def __init__(self, n_waves=100, hidden_dim=64):
        super().__init__()
        self.vert_embed= nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.final_net= nn.Sequential(
            nn.Linear(hidden_dim+1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_waves)
        )

    def forward(self, c, v_pres, v_where):
        """
        c: [B,1], v_pres: [B,4], v_where: [B,4,2]
        Returns predicted reflection: [B,100]
        """
        B, hidden_dim= c.size(0), 64
        accum= torch.zeros(B, hidden_dim, device=c.device)
        for t in range(MAX_STEPS):
            feat= self.vert_embed(v_where[:,t,:])  # [B,64]
            pres= v_pres[:,t]                      # shape [B], not [B,1]
            # We fix shape mismatch by unsqueezing feat if needed or pres if needed:
            # feat: [B,64], pres: [B], so let's broadcast manually:
            #  accum += feat * pres[:,None]
            accum+= feat* pres.unsqueeze(-1)  # => shape [B,64]
        cat_in= torch.cat([accum, c], dim=-1) # [B, 64+1]
        return self.final_net(cat_in)         # => [B, 100]

class ForcedFirstVertexModel(nn.Module):
    """
    - c => Beta(2,2) or Delta if known
    - The first vertex presence => forced=1 in model
    - location => raw ~ Normal(0,1) => softplus => x>=0,y>=0
    - subsequent vertices => Bernoulli(0.5 * prev)
    - decode => Normal(...,0.01)
    """
    def __init__(self, n_waves=100):
        super().__init__()
        self.n_waves= n_waves
        self.decoder= Decoder(n_waves,64)

    def model(self, 
              spectrum,    # [B,100]
              c_vals,      # [B,1]
              is_c_known,  # [B]
              v_pres,      # [B,4]
              v_where,     # [B,4,2]
              is_v_known): # [B]
        pyro.module("Forced1stVtxModel", self)
        B= spectrum.size(0)

        with pyro.plate("data", B):
            # c
            alpha,beta= 2.0,2.0
            raw_c= pyro.sample("raw_c",
                               dist.Beta(alpha*torch.ones(B,1),
                                         beta*torch.ones(B,1)).to_event(1))
            c_mask= is_c_known.float().unsqueeze(-1)
            c_final= c_mask*c_vals + (1-c_mask)*raw_c

            # presence t=0 => forced=1
            # if known => delta( known ), else => delta(1)
            known_mask0= (is_v_known.float()* (v_pres[:,0]>=0.5).float()) # shape [B]
            forced_1= torch.ones(B, device=spectrum.device)
            pres_0= torch.where(known_mask0>0.5, v_pres[:,0], forced_1) # shape [B]

            # location t=0 => raw ~ Normal(0,1) => softplus => x>=0,y>=0
            # if known => delta, else => the sample
            raw_w0= pyro.sample("raw_where_0",
                                dist.Normal(torch.zeros(B,2, device=spectrum.device),
                                            torch.ones(B,2, device=spectrum.device))
                                    .to_event(1)
                                    .mask(pres_0)) # we want mask shape [B], not [B,1]
            w0_unclamped= F.softplus(raw_w0)
            # if known => use v_where
            where_known0= (is_v_known.float()* pres_0) # [B]
            w0_final= torch.where(where_known0.unsqueeze(-1)>0.5,
                                  v_where[:,0,:],
                                  w0_unclamped)

            v_pres_list= [pres_0]
            v_where_list= [w0_final]
            prev_pres= pres_0

            for t in range(1,MAX_STEPS):
                # presence
                nm_p= f"raw_pres_{t}"
                p_prob= 0.5*prev_pres
                raw_p= pyro.sample(nm_p,
                                   dist.Bernoulli(p_prob).mask(prev_pres)) # shape [B]
                # if known => delta
                known_mask_t= (is_v_known.float() *(v_pres[:,t]>=0.5).float()) # [B]
                p_final= torch.where(known_mask_t>0.5,
                                     v_pres[:,t],
                                     raw_p)

                # location
                w_name= f"raw_where_{t}"
                raw_w= pyro.sample(w_name,
                                   dist.Normal(torch.zeros(B,2, device=spectrum.device),
                                               torch.ones(B,2, device=spectrum.device))
                                       .to_event(1)
                                       .mask(raw_p)) # shape [B,2]
                w_unclamped= F.softplus(raw_w)
                where_mask= (is_v_known.float()* p_final)
                w_final= torch.where(where_mask.unsqueeze(-1)>0.5,
                                     v_where[:,t,:],
                                     w_unclamped)

                v_pres_list.append(p_final)
                v_where_list.append(w_final)
                prev_pres= p_final

            # stack
            v_pres_cat= torch.stack(v_pres_list, dim=1)   # shape [B,4]
            v_where_cat= torch.stack(v_where_list, dim=1) # shape [B,4,2]

            # decode
            mean_sp= self.decoder(c_final, v_pres_cat, v_where_cat)
            pyro.sample("obs_spectrum",
                        dist.Normal(mean_sp,0.01).to_event(1),
                        obs=spectrum)

    def guide(self,
              spectrum,
              c_vals, is_c_known,
              v_pres, v_where, is_v_known):
        pyro.module("Forced1stVtxModel", self)
        B= spectrum.size(0)

        if not hasattr(self,"c_net"):
            self.c_net= nn.Sequential(nn.Linear(self.n_waves,2))

        with pyro.plate("data", B):
            # c
            c_out= self.c_net(spectrum) # [B,2]
            alpha_= F.softplus(c_out[:,0:1])+1
            beta_ = F.softplus(c_out[:,1:2]) +1
            raw_c= pyro.sample("raw_c", dist.Beta(alpha_,beta_).to_event(1))

            # presence t=0 => forced=1 => no matching sample => so let's do a dummy sample with is_aux
            forced_1= torch.ones(B, device=spectrum.device)
            dummy0= pyro.sample("dummy_pres_0",
                                dist.Delta(forced_1),
                                infer={"is_auxiliary": True})
            # combine if known
            known_mask0= (is_v_known.float()* (v_pres[:,0]>=0.5).float()) # [B]
            pres_0= torch.where(known_mask0>0.5,
                                v_pres[:,0],
                                dummy0)

            # location t=0 => param net
            if not hasattr(self,"w0_net"):
                self.w0_net= nn.Sequential(nn.Linear(self.n_waves,4))
            w0_out= self.w0_net(spectrum)
            loc0= w0_out[:,0:2]
            sc0 = F.softplus(w0_out[:,2:4])+1e-4
            raw_w0= pyro.sample("raw_where_0",
                                dist.Normal(loc0, sc0).to_event(1)
                                    .mask(pres_0))
            w0_unclamped= F.softplus(raw_w0)
            where_mask0= (is_v_known.float()* pres_0)
            w0_final= torch.where(where_mask0.unsqueeze(-1)>0.5,
                                  v_where[:,0,:],
                                  w0_unclamped)

            prev_pres= pres_0
            for t in range(1,MAX_STEPS):
                # presence
                nm_p= f"raw_pres_{t}"
                if not hasattr(self, f"pres_net_{t}"):
                    setattr(self,f"pres_net_{t}",
                            nn.Sequential(
                                nn.Linear(self.n_waves,1),
                                nn.Sigmoid()
                            ))
                pres_net= getattr(self, f"pres_net_{t}")
                raw_prob= pres_net(spectrum).squeeze(-1) # shape [B]
                p_prob= raw_prob*prev_pres
                raw_p= pyro.sample(nm_p,
                                   dist.Bernoulli(p_prob).mask(prev_pres)) # shape [B]

                known_mask_t= (is_v_known.float()* (v_pres[:,t]>=0.5).float()) # [B]
                p_final= torch.where(known_mask_t>0.5,
                                     v_pres[:,t],
                                     raw_p)

                # location
                w_nm= f"raw_where_{t}"
                if not hasattr(self, f"where_net_{t}"):
                    setattr(self, f"where_net_{t}",
                            nn.Sequential(nn.Linear(self.n_waves,4)))
                w_net= getattr(self, f"where_net_{t}")
                w_out= w_net(spectrum)
                loc= w_out[:,0:2]
                sc = F.softplus(w_out[:,2:4])+1e-4
                r_w= pyro.sample(w_nm,
                                 dist.Normal(loc, sc).to_event(1)
                                     .mask(raw_p)) # shape [B,2]
                w_unclamped= F.softplus(r_w)
                where_mask= (is_v_known.float()* p_final)
                w_final= torch.where(where_mask.unsqueeze(-1)>0.5,
                                     v_where[:,t,:],
                                     w_unclamped)
                prev_pres= p_final


###############################################################################
# 3) Train
###############################################################################
def train_model(csv_path):
    pyro.clear_param_store()

    # 1) load
    spectra, c_vals, is_c_known, v_pres, v_where, is_v_known= load_data(csv_path)
    N,n_waves= spectra.shape
    # 2) partial mask
    final_c, final_v= apply_semi_mask(is_c_known, is_v_known, fraction=MASK_FRAC, seed=123)

    net= ForcedFirstVertexModel(n_waves=n_waves)
    opt= optim.Adam({"lr": LR})
    svi= SVI(net.model, net.guide, opt, loss=TraceGraph_ELBO())

    idx_arr= np.arange(N)
    for ep in trange(NUM_EPOCHS, desc="Training"):
        np.random.shuffle(idx_arr)
        total_loss=0.0
        for start_i in range(0, N, BATCH_SIZE):
            end_i= start_i+BATCH_SIZE
            sb= idx_arr[start_i:end_i]
            sp_b= spectra[sb]
            c_b = c_vals[sb]
            ic_b= final_c[sb]
            vp_b= v_pres[sb]
            vw_b= v_where[sb]
            iv_b= final_v[sb]

            loss= svi.step(sp_b, c_b, ic_b, vp_b, vw_b, iv_b)
            total_loss+= loss
        avg_elbo= total_loss/N
        print(f"[Epoch {ep+1}/{NUM_EPOCHS}] ELBO={avg_elbo:.2f}")

        # save
        st= pyro.get_param_store().get_state()
        torch.save(st, os.path.join(SAVE_DIR, f"ckpt_epoch{ep+1}.pt"))
    print("Training completed!")


###############################################################################
# 4) Inference + Plot
###############################################################################
@torch.no_grad()
def replicate_c4(verts):
    """
    replicate points [N,2] in 4 quadrants
    """
    out_list=[]
    angles= [0, math.pi/2, math.pi, 3*math.pi/2]
    for ang in angles:
        cosA= math.cos(ang)
        sinA= math.sin(ang)
        rot= torch.tensor([[cosA, -sinA],[sinA, cosA]],dtype=torch.float)
        chunk= verts@rot.T
        out_list.append(chunk)
    return torch.cat(out_list,dim=0)

def angle_sort(points):
    px= points[:,0]
    py= points[:,1]
    ang= torch.atan2(py, px)
    idx= torch.argsort(ang)
    return points[idx]

def close_polygon(pts):
    if pts.size(0)>1:
        pts= torch.cat([pts, pts[:1]],dim=0)
    return pts

@torch.no_grad()
def infer_latents(net, spectrum):
    B=1
    c_b  = torch.zeros(B,1)
    ic_b = torch.zeros(B,dtype=torch.bool)
    vp_b = torch.zeros(B,MAX_STEPS)
    vw_b = torch.zeros(B,MAX_STEPS,2)
    iv_b = torch.zeros(B,dtype=torch.bool)

    gtr= trace(net.guide).get_trace(spectrum, c_b, ic_b, vp_b, vw_b, iv_b)
    raw_c_val= gtr.nodes["raw_c"]["value"]
    c_final= raw_c_val

    # presence t=0 => dummy_pres_0
    # location t=0 => raw_where_0
    dummy0= gtr.nodes["dummy_pres_0"]["value"]
    v_pres_list=[dummy0]
    r_w0= gtr.nodes["raw_where_0"]["value"]
    w0= F.softplus(r_w0)
    v_where_list=[w0]
    prev= dummy0
    for t in range(1,MAX_STEPS):
        nm_p= f"raw_pres_{t}"
        nm_w= f"raw_where_{t}"
        rp= gtr.nodes[nm_p]["value"]
        rw= gtr.nodes[nm_w]["value"]
        rw= F.softplus(rw)
        v_pres_list.append(rp)
        v_where_list.append(rw)
        prev= rp

    v_pres_cat= torch.stack(v_pres_list, dim=1)  # shape [1,4]
    v_where_cat= torch.stack(v_where_list, dim=1) # shape [1,4,2]
    return c_final, v_pres_cat, v_where_cat

def angle_sort_polygon(points):
    px= points[:,0]
    py= points[:,1]
    ang= torch.atan2(py, px)
    idx= torch.argsort(ang)
    return points[idx]

def plot_polygon(pts, c_val, out_path):
    if pts.size(0)<2:
        plt.figure()
        plt.scatter(pts[:,0].numpy(), pts[:,1].numpy(), c='r')
        plt.title(f"Single Vertex. c={c_val:.3f}")
        plt.savefig(out_path)
        plt.close()
        return
    sorted_pts= angle_sort_polygon(pts)
    sorted_pts= torch.cat([sorted_pts, sorted_pts[:1]], dim=0)
    sx= sorted_pts[:,0].numpy()
    sy= sorted_pts[:,1].numpy()
    plt.figure()
    plt.fill(sx, sy, c='r', alpha=0.3)
    plt.plot(sx, sy, 'ro-', label=f"N={pts.size(0)}")
    plt.axhline(0,c='k',lw=0.5)
    plt.axvline(0,c='k',lw=0.5)
    plt.title(f"C4 polygon (angle-sorted). c={c_val:.3f}")
    plt.legend()
    plt.savefig(out_path)
    plt.close()

def test_inference():
    ckpt_path= os.path.join(SAVE_DIR, f"ckpt_epoch{NUM_EPOCHS}.pt")
    net= ForcedFirstVertexModel(N_WAVES)
    pyro.clear_param_store()

    if os.path.isfile(ckpt_path):
        st= torch.load(ckpt_path, map_location="cpu")
        pyro.get_param_store().set_state(st)
        print(f"[INFO] Loaded param store from {ckpt_path}")
    else:
        print(f"[WARN] {ckpt_path} not found. Using random init...")

    # random spec
    random_spec= (torch.rand(N_WAVES)*0.5 +0.25).unsqueeze(0)  # [1,100]
    c_pred, v_pres_pred, v_where_pred= infer_latents(net, random_spec)
    # decode
    recon= net.decoder(c_pred, v_pres_pred, v_where_pred).squeeze(0).detach()
    random_1d= random_spec.squeeze(0).detach()
    c_val= float(c_pred[0,0])

    # create subfolder
    dt_str= datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir= f"inference_plots_{dt_str}"
    os.makedirs(out_dir, exist_ok=True)

    # save data
    np.savetxt(os.path.join(out_dir,"random_spectrum.txt"), random_1d.numpy(), fmt="%.5f")
    np.savetxt(os.path.join(out_dir,"reconstructed_spectrum.txt"), recon.numpy(), fmt="%.5f")
    with open(os.path.join(out_dir,"predicted_latents.txt"),"w") as f:
        f.write(f"c={c_val:.5f}\n")
        for t in range(MAX_STEPS):
            ps= float(v_pres_pred[0,t])
            xx= float(v_where_pred[0,t,0])
            yy= float(v_where_pred[0,t,1])
            f.write(f"Vertex {t}: pres={ps:.3f}, x={xx:.3f}, y={yy:.3f}\n")

    # plot spectrum
    x_axis= np.arange(N_WAVES)
    plt.figure()
    plt.plot(x_axis, random_1d.numpy(), label="RandomSpec", marker='o')
    plt.plot(x_axis, recon.numpy(), label="Reconstructed", marker='x')
    plt.legend()
    plt.title("Random vs. Reconstructed Spectrum")
    plt.savefig(os.path.join(out_dir,"spectra_compare.png"))
    plt.close()

    # gather present vertices
    keep_pts=[]
    # forced first
    keep_pts.append(v_where_pred[0,0,:].detach())
    for t in range(1,MAX_STEPS):
        if float(v_pres_pred[0,t])>0.5:
            keep_pts.append(v_where_pred[0,t,:].detach())

    if len(keep_pts)==0:
        print("[WARN] no predicted vertices? Should have at least the forced one.")
    else:
        first_quad= torch.stack(keep_pts,dim=0)
        # replicate c4
        c4_verts= replicate_c4(first_quad)
        # filter out negative? we can clamp if you want:
        c4_clamped= torch.clamp(c4_verts, min=0)
        # plot polygon
        out_poly= os.path.join(out_dir,"c4_polygon.png")
        plot_polygon(c4_clamped, c_val, out_poly)

    print(f"[INFO] Inference results saved in {out_dir}/")


def main():
    csv_path= "merged_s4_shapes.csv"
    # 1) train
    train_model(csv_path)
    # 2) inference
    test_inference()

if __name__=="__main__":
    main()
