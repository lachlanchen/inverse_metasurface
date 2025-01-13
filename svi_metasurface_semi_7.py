#!/usr/bin/env python3

"""
Semi-Supervised AIR with Forced First Vertex and First-Quadrant Vertices
-----------------------------------------------------------------------
- We load reflection columns R@... => n_waves=100.
- c in [0,1] partially labeled (mask=70%).
- The first vertex presence is forced = 1 => at least one vertex.
- Up to 4 total vertices, each in the first quadrant (x>=0, y>=0).
- We replicate the predicted first-quadrant vertices with C4 symmetry for plotting.
- Train for 100 epochs with a progress bar.
- Then do inference on a random 100-pt reflection, plot the polygon as a
  connected shape in ascending angle.

Author: ChatGPT
"""

import os
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceGraph_ELBO
import pyro.optim as optim
from tqdm import trange
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------- CONFIG -------------
MAX_STEPS = 4
MASK_FRAC = 0.7
NUM_EPOCHS = 100
BATCH_SIZE = 512
LR = 1e-3
SAVE_DIR = "results_forcedFirstVertex"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------- 1) Data Loading & Semi Mask -------------

def load_data(csv_path):
    """
    Expects a CSV with columns:
      R@... (100 columns) => reflection
      c => optional partial [0..1]
      v_pres_0..3 => presence
      v_where_0_x, v_where_0_y, ... => up to 4 sets of (x,y).
    Returns:
      spectra: [N, n_waves]
      c_vals: [N,1], 0 if unknown
      is_c_known: bool [N]
      v_pres: [N,4], v_where: [N,4,2]
      is_v_known: bool[N]
    """
    df = pd.read_csv(csv_path)
    # reflection
    r_cols = [col for col in df.columns if col.startswith("R@")]
    r_cols = sorted(r_cols, key=lambda x: float(x.split('@')[1]))
    spectra_np = df[r_cols].values.astype(np.float32)
    spectra = torch.from_numpy(spectra_np)
    N, n_waves = spectra.shape
    if n_waves != 100:
        print(f"[WARN] CSV has {n_waves} reflection columns, not 100. Adjust code if needed.")

    # c
    c_np = np.full((N,1), np.nan, dtype=np.float32)
    if "c" in df.columns:
        c_np = df["c"].values.reshape(-1,1).astype(np.float32)
    is_c_known_np = ~np.isnan(c_np).reshape(-1)
    c_np[np.isnan(c_np)] = 0.0
    c_vals = torch.from_numpy(c_np)
    is_c_known = torch.from_numpy(is_c_known_np)

    # v
    pres_cols= [f"v_pres_{t}" for t in range(MAX_STEPS)]
    wx_cols  = [f"v_where_{t}_x" for t in range(MAX_STEPS)]
    wy_cols  = [f"v_where_{t}_y" for t in range(MAX_STEPS)]
    have_pres = all(pc in df.columns for pc in pres_cols)
    have_where= all((wx in df.columns and wy in df.columns) for wx,wy in zip(wx_cols,wy_cols))

    if have_pres and have_where:
        v_pres_np= df[pres_cols].values.astype(np.float32)
        vx_np= df[wx_cols].values.astype(np.float32)
        vy_np= df[wy_cols].values.astype(np.float32)
        row_nan= (np.isnan(v_pres_np).any(axis=1)|
                  np.isnan(vx_np).any(axis=1)|
                  np.isnan(vy_np).any(axis=1))
        is_v_known_np= ~row_nan
        # fill 0
        v_pres_np[np.isnan(v_pres_np)] = 0
        vx_np[np.isnan(vx_np)] = 0
        vy_np[np.isnan(vy_np)] = 0
        v_pres = torch.from_numpy(v_pres_np)
        v_where= torch.stack([torch.from_numpy(vx_np),
                              torch.from_numpy(vy_np)], dim=-1) # [N,4,2]
        is_v_known= torch.from_numpy(is_v_known_np)
    else:
        v_pres = torch.zeros(N, MAX_STEPS)
        v_where= torch.zeros(N, MAX_STEPS,2)
        is_v_known= torch.zeros(N, dtype=torch.bool)

    return spectra, c_vals, is_c_known, v_pres, v_where, is_v_known

def apply_semi_mask(is_c_known, is_v_known, fraction=MASK_FRAC, seed=123):
    """
    Keep 'fraction' of known samples. The rest => unknown => 0
    """
    N= len(is_c_known)
    np.random.seed(seed)
    r= np.random.rand(N)
    keep= r< fraction
    final_c= is_c_known.numpy() & keep
    final_v= is_v_known.numpy() & keep
    return torch.from_numpy(final_c), torch.from_numpy(final_v)

# ------------- 2) Model & Guide (Force First Vertex in 1st Quadrant) -------------

class Decoder(nn.Module):
    """
    c + up to 4 first-quadrant vertices => reflect(100)
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
        B, hidden_dim= c.size(0), 64
        accum= torch.zeros(B, hidden_dim, device=c.device)
        for t in range(MAX_STEPS):
            # we assume x>0, y>=0 from data side, but the network can predict negative?
            # We'll trust the model can learn to keep them in quadrant if data encourages that.
            feat= self.vert_embed(v_where[:,t,:])
            pres= v_pres[:,t:t+1]
            accum= accum+ feat*pres
        x= torch.cat([accum, c], dim=-1)
        return self.final_net(x)

class ForcedFirstVertexModel(nn.Module):
    """
    The model forcibly sets presence of first vertex=1 => at least 1 vertex.

    c => Beta(2,2) or Delta if known
    presence => (0) forced 1, (1..3) => Bernoulli(0.5 * prev)
    location => Normal(0,1) or Delta
    decode => Normal(...,0.01)
    """
    def __init__(self, n_waves=100):
        super().__init__()
        self.n_waves= n_waves
        self.decoder= Decoder(n_waves,64)

    def model(self, spectrum, c_vals, is_c_known, v_pres, v_where, is_v_known):
        pyro.module("forcedFVModel", self)
        B= spectrum.size(0)
        with pyro.plate("data", B):
            # c
            alpha,beta= 2.0,2.0
            raw_c= pyro.sample("raw_c",
                               dist.Beta(alpha*torch.ones(B,1),
                                         beta*torch.ones(B,1)).to_event(1))
            c_mask= is_c_known.float().unsqueeze(-1)
            c = c_mask*c_vals + (1-c_mask)*raw_c

            # presence t=0 => forced=1
            if_b_known0= (is_v_known.float()*(v_pres[:,0]>=0.5).float()).unsqueeze(-1)
            raw_p0= torch.ones(B,1, device=spectrum.device)
            pres_0= if_b_known0*v_pres[:,0:1] + (1-if_b_known0)*raw_p0

            # location t=0 => normal(0,1) or delta
            loc0= torch.zeros(B,2, device=spectrum.device)
            sc0 = torch.ones(B,2, device=spectrum.device)
            raw_w0= pyro.sample("raw_where_0",
                                dist.Normal(loc0, sc0)
                                    .mask(pres_0)
                                    .to_event(1))
            w0_mask= is_v_known.float().unsqueeze(-1)*pres_0
            w0_val= w0_mask*v_where[:,0,:] + (1-w0_mask)*raw_w0

            v_pres_list= [pres_0]
            v_where_list=[w0_val]
            prev_pres= pres_0

            for t in range(1,MAX_STEPS):
                nm_p= f"raw_pres_{t}"
                p_prob= 0.5*prev_pres
                raw_p= pyro.sample(nm_p,
                                   dist.Bernoulli(p_prob).to_event(1))
                if_known= (is_v_known.float()*(v_pres[:,t]>=0.5).float()).unsqueeze(-1)
                p_val= if_known*v_pres[:,t:t+1] + (1-if_known)*raw_p

                nm_w= f"raw_where_{t}"
                loc= torch.zeros(B,2, device=spectrum.device)
                sc = torch.ones(B,2, device=spectrum.device)
                raw_w= pyro.sample(nm_w,
                                   dist.Normal(loc, sc)
                                       .mask(raw_p)
                                       .to_event(1))
                w_mask= is_v_known.float().unsqueeze(-1)*p_val
                w_val= w_mask*v_where[:,t,:] + (1-w_mask)*raw_w

                v_pres_list.append(p_val)
                v_where_list.append(w_val)
                prev_pres= p_val

            v_pres_cat= torch.cat(v_pres_list, dim=1)   # [B,4]
            v_where_cat= torch.stack(v_where_list, dim=1)# [B,4,2]
            mean_sp= self.decoder(c, v_pres_cat, v_where_cat)
            pyro.sample("obs_spectrum",
                        dist.Normal(mean_sp,0.01).to_event(1),
                        obs=spectrum)

    def guide(self, spectrum, c_vals, is_c_known, v_pres, v_where, is_v_known):
        pyro.module("forcedFVModel", self)
        B= spectrum.size(0)

        # c param
        if not hasattr(self,"c_net"):
            self.c_net= nn.Sequential(
                nn.Linear(self.n_waves,2),
            )

        with pyro.plate("data", B):
            out_c= self.c_net(spectrum)
            alpha_ = F.softplus(out_c[:,0:1])+1
            beta_  = F.softplus(out_c[:,1:2]) +1
            raw_c= pyro.sample("raw_c",
                               dist.Beta(alpha_,beta_).to_event(1))

            # presence t=0 => forced=1
            prob0= torch.ones(B,1, device=spectrum.device)
            # We'll define "dummy_pres_0" for shape consistency
            raw_p0= pyro.sample("dummy_pres_0",
                                dist.Bernoulli(prob0).to_event(1))
            if_known0= (is_v_known.float()*(v_pres[:,0]>=0.5).float()).unsqueeze(-1)
            pres_0= if_known0*v_pres[:,0:1] + (1-if_known0)*raw_p0

            # location t=0 => net
            if not hasattr(self,"where_0_net"):
                self.where_0_net= nn.Sequential(nn.Linear(self.n_waves,4))
            w0_out= self.where_0_net(spectrum)
            w0_loc= w0_out[:,0:2]
            w0_scale= F.softplus(w0_out[:,2:4])+1e-4
            raw_w0= pyro.sample("raw_where_0",
                                dist.Normal(w0_loc,w0_scale)
                                    .mask(pres_0)
                                    .to_event(1))
            w0_mask= is_v_known.float().unsqueeze(-1)*pres_0
            w0_val= w0_mask*v_where[:,0,:]+ (1-w0_mask)*raw_w0

            prev_pres= pres_0
            for t in range(1,MAX_STEPS):
                nm_p= f"raw_pres_{t}"
                if not hasattr(self,f"pres_net_{t}"):
                    setattr(self, f"pres_net_{t}",
                            nn.Sequential(
                                nn.Linear(self.n_waves,1),
                                nn.Sigmoid()
                            ))
                pres_net= getattr(self, f"pres_net_{t}")
                p_prob0= pres_net(spectrum)
                p_prob = p_prob0*prev_pres
                raw_p= pyro.sample(nm_p,
                                   dist.Bernoulli(p_prob).to_event(1))
                if_known= (is_v_known.float()*(v_pres[:,t]>=0.5).float()).unsqueeze(-1)
                p_val= if_known*v_pres[:,t:t+1] + (1-if_known)*raw_p

                nm_w= f"raw_where_{t}"
                if not hasattr(self, f"where_net_{t}"):
                    setattr(self, f"where_net_{t}",
                            nn.Sequential(nn.Linear(self.n_waves,4)))
                w_net= getattr(self, f"where_net_{t}")
                w_out= w_net(spectrum)
                loc= w_out[:,0:2]
                sc=  F.softplus(w_out[:,2:4])+1e-4
                raw_w= pyro.sample(nm_w,
                                   dist.Normal(loc,sc)
                                       .mask(raw_p)
                                       .to_event(1))
                w_mask= is_v_known.float().unsqueeze(-1)*p_val
                w_val= w_mask*v_where[:,t,:]+ (1-w_mask)*raw_w
                prev_pres= p_val

# ------------- 3) Train -------------
def train_model(csv_path, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    # load
    spectra, c_vals, is_c_known, v_pres, v_where, is_v_known= load_data(csv_path)
    N, n_waves= spectra.shape
    # partial mask
    final_c, final_v= apply_semi_mask(is_c_known, is_v_known, fraction=MASK_FRAC, seed=123)

    net= ForcedFirstVertexModel(n_waves)
    pyro.clear_param_store()
    opt= optim.Adam({"lr":lr})
    svi= SVI(net.model, net.guide, opt, loss=TraceGraph_ELBO())

    idx_arr= np.arange(N)
    from tqdm import trange
    for ep in trange(num_epochs, desc="Training"):
        np.random.shuffle(idx_arr)
        total_loss= 0.0
        for start_i in range(0,N,batch_size):
            end_i= start_i+batch_size
            sb= idx_arr[start_i:end_i]
            spec_b= spectra[sb]
            c_b   = c_vals[sb]
            ic_b  = final_c[sb]
            vp_b  = v_pres[sb]
            vw_b  = v_where[sb]
            iv_b  = final_v[sb]
            loss= svi.step(spec_b, c_b, ic_b, vp_b, vw_b, iv_b)
            total_loss+=loss
        avg_elbo= total_loss/N
        print(f"[Epoch {ep+1}/{num_epochs}] ELBO={avg_elbo:.2f}")
        st= pyro.get_param_store().get_state()
        torch.save(st, os.path.join(SAVE_DIR, f"ckpt_epoch{ep+1}.pt"))
    print("Training completed!")

# ------------- 4) Inference + Plot -------------
@torch.no_grad()
def replicate_c4(verts):
    """
    replicate points in [n,2] in 4 quadrants for symmetrical shape
    We'll only do it if the model is designed to predict first quadrant.
    """
    out_list=[]
    angles= [0, math.pi/2, math.pi, 3*math.pi/2]
    for ang in angles:
        cosA= math.cos(ang)
        sinA= math.sin(ang)
        rot= torch.tensor([[cosA, -sinA],
                           [sinA,  cosA]],dtype=torch.float)
        chunk= verts@rot.T
        out_list.append(chunk)
    return torch.cat(out_list, dim=0)

@torch.no_grad()
def infer_latents(net, spectrum):
    """
    net: ForcedFirstVertexModel
    spectrum: [1, n_waves]
    Return c, v_pres, v_where
    """
    from pyro.poutine import trace
    B=1
    n_waves= spectrum.size(1)
    c_b  = torch.zeros(B,1)
    ic_b = torch.zeros(B,dtype=torch.bool)
    vp_b = torch.zeros(B,MAX_STEPS)
    vw_b = torch.zeros(B,MAX_STEPS,2)
    iv_b = torch.zeros(B,dtype=torch.bool)

    gtr= trace(net.guide).get_trace(spectrum, c_b, ic_b, vp_b, vw_b, iv_b)
    raw_c_val= gtr.nodes["raw_c"]["value"]
    c_final= raw_c_val

    # presence t=0 => "dummy_pres_0"
    # location t=0 => "raw_where_0"
    v_pres_list=[]
    v_where_list=[]

    dummy0= gtr.nodes["dummy_pres_0"]["value"]
    v_pres_list.append(dummy0)
    w0_val= gtr.nodes["raw_where_0"]["value"]
    v_where_list.append(w0_val)

    prev= dummy0
    for t in range(1,MAX_STEPS):
        nm_p= f"raw_pres_{t}"
        nm_w= f"raw_where_{t}"
        rp= gtr.nodes[nm_p]["value"]
        rw= gtr.nodes[nm_w]["value"]
        v_pres_list.append(rp)
        v_where_list.append(rw)
        prev= rp

    v_pres_cat= torch.cat(v_pres_list, dim=1)
    v_where_cat= torch.stack(v_where_list, dim=1)
    return c_final, v_pres_cat, v_where_cat


def angle_sort_polygon(points):
    """
    points: [N,2], presumably in the first quadrant
    We'll sort them by angle in ascending order, then connect in a polygon.
    If N=1 or 2, it's a small shape. We'll just do a line or a minimal polygon.
    """
    # compute angles
    px= points[:,0]
    py= points[:,1]
    angles= torch.atan2(py, px)  # shape [N]
    # sort by ascending angle
    sorted_indices= torch.argsort(angles)
    sorted_pts= points[sorted_indices]
    return sorted_pts

def plot_polygon(points, c_val, out_path):
    """
    points: [N,2], we connect them in ascending angle + close the polygon.
    We'll do a fill or line2D.
    """
    N= points.size(0)
    if N<2:
        # trivial
        xx= points[:,0].tolist()
        yy= points[:,1].tolist()
        plt.figure()
        plt.scatter(xx,yy,c='r')
        plt.title(f"Only 1 vertex. c={c_val:.3f}")
        plt.savefig(out_path)
        plt.close()
        return
    # sort by angle
    sorted_pts= angle_sort_polygon(points)
    # close
    sorted_pts= torch.cat([sorted_pts, sorted_pts[0:1,:]], dim=0)
    sx= sorted_pts[:,0].tolist()
    sy= sorted_pts[:,1].tolist()
    plt.figure()
    plt.fill(sx, sy, c='r', alpha=0.3)
    plt.plot(sx, sy, c='r', marker='o', label=f"N={N}")
    plt.title(f"Polygon in ascending angle, c={c_val:.3f}")
    plt.axhline(0,c='k',lw=0.5)
    plt.axvline(0,c='k',lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def test_inference(ckpt_path, n_waves=100, out_dir="inference_plots"):
    net= ForcedFirstVertexModel(n_waves)
    pyro.clear_param_store()
    if os.path.isfile(ckpt_path):
        st= torch.load(ckpt_path, map_location="cpu")
        pyro.get_param_store().set_state(st)
        print(f"[INFO] Loaded param store from {ckpt_path}")
    else:
        print(f"[WARN] {ckpt_path} not found, using random init.")

    # create random spec of shape [1, n_waves]
    random_spec= (torch.rand(n_waves)*0.5 + 0.25).unsqueeze(0)
    c_pred, v_pres_pred, v_where_pred= infer_latents(net, random_spec)

    # decode
    recon= net.decoder(c_pred, v_pres_pred, v_where_pred).squeeze(0).detach()
    random_spec_1d= random_spec.squeeze(0).detach()

    c_val= float(c_pred[0,0])
    # choose only vertices with presence>0.5 plus the first forced one
    # first is forced => presence=1, others => check presence
    keep_verts= []
    # definitely keep first
    keep_verts.append(v_where_pred[0,0,:].detach())
    for t in range(1,MAX_STEPS):
        if float(v_pres_pred[0,t])>0.5:
            keep_verts.append(v_where_pred[0,t,:].detach())

    # 1) plot spectra
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    x_axis= np.arange(n_waves)
    plt.plot(x_axis, random_spec_1d.numpy(), label="Input random spec", marker='o')
    plt.plot(x_axis, recon.numpy(), label="Reconstructed", marker='x')
    plt.legend()
    plt.title("Random vs. Reconstructed Spectrum")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"spectra_compare.png"))
    plt.close()

    # 2) polygon in first quadrant => connect them
    # if we have at least 2 points
    # replicate them in c4 for final plot
    keep_t= torch.stack(keep_verts, dim=0)
    # ensure x>=0,y>=0. The model might produce negative, but you wanted them in first quadrant:
    # we can clamp them
    keep_t= torch.clamp(keep_t, min=0)
    c4_verts= replicate_c4(keep_t)
    # sort angle, fill
    out_path= os.path.join(out_dir,"c4_polygon.png")
    plot_polygon(c4_verts, c_val, out_path)
    print(f"[INFO] Inference results saved in {out_dir}")


def main():
    # 1) Train
    csv_path= "merged_s4_shapes.csv"
    train_model(csv_path, num_epochs=100, batch_size=512, lr=1e-3)

    # 2) Inference
    ckpt_100= os.path.join(SAVE_DIR,"ckpt_epoch100.pt")
    test_inference(ckpt_100, n_waves=100, out_dir="inference_plots")

if __name__=="__main__":
    main()
