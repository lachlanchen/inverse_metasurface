#!/usr/bin/env python3

"""
Semi-Supervised AIR with Forced First Vertex & vertices_str
------------------------------------------------------------
- 100 reflection columns R@...
- c in [0,1] partially labeled => keep 70%
- vertices_str => semicolon-delimited (x,y) pairs. Only keep x>=0,y>=0. Sort them by angle, keep up to 4 => v_where,v_pres
- The first vertex presence is forced=1 in the model, subsequent => Bernoulli(0.5 * prev)
- Vertex locations => raw ~ Normal(0,1), softplus => x>=0,y>=0
- Trains 100 epochs, then inference on a random reflection
- **Updated** final polygon plotting to rotate the valid first-quadrant points
  by 0°, 90°, 180°, 270° => negative coords in other quadrants => angle-sort => fill
- Also updated test_inference() to generate a smoother random spectrum 
  (via cubic spline interpolation) rather than purely random values.
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

###############################################################################
# Extra import for smooth interpolation
###############################################################################
from scipy.interpolate import make_interp_spline

###############################################################################
# Configuration
###############################################################################
MAX_STEPS   = 4
N_WAVES     = 100
MASK_FRAC   = 0.7
NUM_EPOCHS  = 100
BATCH_SIZE  = 4096  # Using a large batch size per your request
LR          = 1e-3

TIMESTAMP   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR    = f"results_forced1stVtx_{TIMESTAMP}"
os.makedirs(SAVE_DIR, exist_ok=True)

###############################################################################
# 1) Data Loading (R@..., c, vertices_str)
###############################################################################
def parse_vertices_str(vert_str):
    """
    Parse a semicolon-delimited string of "x,y"
    Keep only x>=0,y>=0
    Return list of (x,y).
    """
    points = []
    if not isinstance(vert_str, str):
        return points
    tokens = vert_str.strip().split(';')
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        sub = tok.split(',')
        if len(sub)!=2:
            continue
        try:
            x= float(sub[0])
            y= float(sub[1])
            if x>=0 and y>=0:
                points.append((x,y))
        except ValueError:
            continue
    return points

def angle_sort_q1(points):
    """
    Given a list of (x,y) in first quadrant, sort by ascending angle.
    Return numpy array shape [N,2].
    """
    if not points:
        return points
    arr= np.array(points,dtype=np.float32)
    angles= np.arctan2(arr[:,1], arr[:,0])
    idx= np.argsort(angles)
    return arr[idx]

def load_data(csv_path):
    """
    - reflection R@... => [N,100]
    - c => [N,1]
    - parse vertices_str => up to 4 first-quadrant points => v_where, v_pres
    - is_v_known => True if we found >=1
    """
    df= pd.read_csv(csv_path)

    # reflection
    r_cols= [col for col in df.columns if col.startswith("R@")]
    r_cols= sorted(r_cols, key=lambda x: float(x.split('@')[1]))
    spectra_np= df[r_cols].values.astype(np.float32)
    N, n_waves= spectra_np.shape
    spectra= torch.from_numpy(spectra_np)
    if n_waves!=N_WAVES:
        print(f"[WARN] found {n_waves} reflection columns, expected {N_WAVES}.")

    # c
    c_np= np.full((N,1), np.nan, dtype=np.float32)
    if "c" in df.columns:
        c_np= df["c"].values.reshape(-1,1).astype(np.float32)
    is_c_known_np= ~np.isnan(c_np).reshape(-1)
    c_np[np.isnan(c_np)] = 0
    c_vals    = torch.from_numpy(c_np)
    is_c_known= torch.from_numpy(is_c_known_np)

    # parse vertices_str
    v_pres = np.zeros((N,MAX_STEPS), dtype=np.float32)
    v_where= np.zeros((N,MAX_STEPS,2), dtype=np.float32)
    is_v_known = np.zeros((N,), dtype=bool)

    if "vertices_str" in df.columns:
        for i in range(N):
            vs_str = df["vertices_str"].iloc[i]
            pts    = parse_vertices_str(vs_str)
            if not pts:
                is_v_known[i]= False
                continue
            # sort by angle, keep up to 4
            arr_sorted= angle_sort_q1(pts)
            arr_sorted= arr_sorted[:MAX_STEPS]
            npts= arr_sorted.shape[0]
            if npts>0:
                is_v_known[i]= True
            for t in range(npts):
                v_pres[i,t]= 1
                v_where[i,t,0]= arr_sorted[t,0]
                v_where[i,t,1]= arr_sorted[t,1]

    v_pres_t    = torch.from_numpy(v_pres)
    v_where_t   = torch.from_numpy(v_where)
    is_v_known_t= torch.from_numpy(is_v_known)

    return spectra, c_vals, is_c_known, v_pres_t, v_where_t, is_v_known_t

def apply_semi_mask(is_c_known, is_v_known, fraction=MASK_FRAC, seed=123):
    N= len(is_c_known)
    np.random.seed(seed)
    keep= np.random.rand(N) < fraction
    c_final= is_c_known.numpy() & keep
    v_final= is_v_known.numpy() & keep
    return torch.from_numpy(c_final), torch.from_numpy(v_final)

###############################################################################
# 2) Model + Guide
###############################################################################
class Decoder(nn.Module):
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
        B= c.size(0)
        hidden_dim= 64
        accum= torch.zeros(B, hidden_dim, device=c.device)
        for t in range(MAX_STEPS):
            feat= self.vert_embed(v_where[:,t,:])  # [B,64]
            pres= v_pres[:,t]                      # [B]
            accum+= feat * pres.unsqueeze(-1)
        cat_in= torch.cat([accum, c], dim=-1)      # [B,64+1]
        return self.final_net(cat_in)             # => [B,n_waves]

class ForcedFirstVertexModel(nn.Module):
    """
    - first vertex presence => forced=1 in model
    - c => Beta(2,2) or delta if known
    - location => raw ~ Normal(0,1), then softplus => x>=0,y>=0
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
            known_mask0= (is_v_known.float() * (v_pres[:,0]>=0.5).float()) # [B]
            forced_1= torch.ones(B, device=spectrum.device)
            pres_0= torch.where(known_mask0>0.5,
                                v_pres[:,0],
                                forced_1) # shape [B]

            # location t=0 => raw => Normal => softplus
            raw_w0= pyro.sample("raw_where_0",
                                dist.Normal(torch.zeros(B,2, device=spectrum.device),
                                            torch.ones(B,2, device=spectrum.device))
                                    .to_event(1)
                                    .mask(pres_0))
            w0_unclamped= F.softplus(raw_w0)
            w0_mask= (is_v_known.float()* pres_0)
            w0_final= torch.where(w0_mask.unsqueeze(-1)>0.5,
                                  v_where[:,0,:],
                                  w0_unclamped)

            v_pres_list= [pres_0]
            v_where_list= [w0_final]
            prev= pres_0

            for t in range(1,MAX_STEPS):
                nm_p= f"raw_pres_{t}"
                p_prob= 0.5*prev
                raw_p= pyro.sample(nm_p,
                                   dist.Bernoulli(p_prob)
                                       .mask(prev))
                known_mask_t= (is_v_known.float()* (v_pres[:,t]>=0.5).float())
                p_final= torch.where(known_mask_t>0.5,
                                     v_pres[:,t],
                                     raw_p)

                nm_w= f"raw_where_{t}"
                raw_w= pyro.sample(nm_w,
                                   dist.Normal(torch.zeros(B,2, device=spectrum.device),
                                               torch.ones(B,2, device=spectrum.device))
                                       .to_event(1)
                                       .mask(raw_p))
                w_unclamp= F.softplus(raw_w)
                w_mask= (is_v_known.float()* p_final)
                w_final= torch.where(w_mask.unsqueeze(-1)>0.5,
                                     v_where[:,t,:],
                                     w_unclamp)

                v_pres_list.append(p_final)
                v_where_list.append(w_final)
                prev= p_final

            v_pres_cat= torch.stack(v_pres_list, dim=1)   # [B,4]
            v_where_cat= torch.stack(v_where_list, dim=1) # [B,4,2]

            mean_sp= self.decoder(c_final, v_pres_cat, v_where_cat)

            # We model the reflection in [0,1] still with a Normal( mean, 0.01 ).
            pyro.sample("obs_spectrum",
                        dist.Normal(mean_sp, 0.01).to_event(1),
                        obs=spectrum)

    def guide(self, 
              spectrum, c_vals, is_c_known,
              v_pres, v_where, is_v_known):
        pyro.module("Forced1stVtxModel", self)
        B= spectrum.size(0)

        if not hasattr(self,"c_net"):
            self.c_net= nn.Sequential(nn.Linear(self.n_waves,2))

        with pyro.plate("data", B):
            # c
            c_out= self.c_net(spectrum)
            alpha_= F.softplus(c_out[:,0:1])+1
            beta_ = F.softplus(c_out[:,1:2]) +1
            raw_c= pyro.sample("raw_c", dist.Beta(alpha_, beta_).to_event(1))

            # presence t=0 => forced=1 => no direct model sample => do dummy
            forced_1= torch.ones(B, device=spectrum.device)
            dummy0= pyro.sample("dummy_pres_0",
                                dist.Delta(forced_1),
                                infer={"is_auxiliary": True})
            known_mask0= (is_v_known.float()*(v_pres[:,0]>=0.5).float())
            pres_0= torch.where(known_mask0>0.5,
                                v_pres[:,0],
                                dummy0)

            if not hasattr(self,"w0_net"):
                self.w0_net= nn.Sequential(nn.Linear(self.n_waves,4))
            w0_out= self.w0_net(spectrum)
            loc0= w0_out[:,0:2]
            sc0= F.softplus(w0_out[:,2:4])+1e-4
            r_w0= pyro.sample("raw_where_0",
                              dist.Normal(loc0, sc0).to_event(1)
                                  .mask(pres_0))
            w0_unclamp= F.softplus(r_w0)
            w0_mask= (is_v_known.float()* pres_0)
            w0_final= torch.where(w0_mask.unsqueeze(-1)>0.5,
                                  v_where[:,0,:],
                                  w0_unclamp)

            prev= pres_0
            for t in range(1,MAX_STEPS):
                nm_p= f"raw_pres_{t}"
                if not hasattr(self,f"pres_net_{t}"):
                    setattr(self,f"pres_net_{t}",
                            nn.Sequential(nn.Linear(self.n_waves,1),
                                          nn.Sigmoid()))
                pres_net= getattr(self,f"pres_net_{t}")
                p_prob0= pres_net(spectrum).squeeze(-1)
                raw_p= pyro.sample(nm_p,
                                   dist.Bernoulli(p_prob0*prev)
                                       .mask(prev))
                known_mask_t= (is_v_known.float()* (v_pres[:,t]>=0.5).float())
                p_final= torch.where(known_mask_t>0.5,
                                     v_pres[:,t],
                                     raw_p)

                nm_w= f"raw_where_{t}"
                if not hasattr(self,f"where_net_{t}"):
                    setattr(self, f"where_net_{t}",
                            nn.Sequential(nn.Linear(self.n_waves,4)))
                w_net= getattr(self, f"where_net_{t}")
                w_out= w_net(spectrum)
                loc= w_out[:,0:2]
                sc=  F.softplus(w_out[:,2:4])+1e-4
                r_w= pyro.sample(nm_w,
                                 dist.Normal(loc, sc).to_event(1)
                                     .mask(raw_p))
                w_unclamp= F.softplus(r_w)
                w_mask= (is_v_known.float()* p_final)
                w_final= torch.where(w_mask.unsqueeze(-1)>0.5,
                                     v_where[:,t,:],
                                     w_unclamp)

                prev= p_final

###############################################################################
# 3) Train
###############################################################################
def train_model(csv_path):
    pyro.clear_param_store()
    spectra, c_vals, is_c_known, v_pres, v_where, is_v_known= load_data(csv_path)
    N, n_waves= spectra.shape
    c_mask, v_mask= apply_semi_mask(is_c_known, is_v_known, fraction=MASK_FRAC)

    net= ForcedFirstVertexModel(n_waves)
    opt= optim.Adam({"lr": LR})
    svi= SVI(net.model, net.guide, opt, loss=TraceGraph_ELBO())

    idx_arr= np.arange(N)
    for ep in trange(NUM_EPOCHS, desc="Training"):
        np.random.shuffle(idx_arr)
        total_loss=0.0
        for start_i in range(0,N,BATCH_SIZE):
            end_i= start_i+BATCH_SIZE
            sb= idx_arr[start_i:end_i]

            sp_b= spectra[sb]
            c_b = c_vals[sb]
            ic_b= c_mask[sb]
            vp_b= v_pres[sb]
            vw_b= v_where[sb]
            iv_b= v_mask[sb]

            loss= svi.step(sp_b, c_b, ic_b, vp_b, vw_b, iv_b)
            total_loss+= loss
        avg_elbo= total_loss/N
        print(f"[Epoch {ep+1}/{NUM_EPOCHS}] ELBO={avg_elbo:.2f}")
        st= pyro.get_param_store().get_state()
        torch.save(st, os.path.join(SAVE_DIR, f"ckpt_epoch{ep+1}.pt"))
    print("Training completed!")

###############################################################################
# 4) Smoother random spectrum generation + Inference + Plot
###############################################################################
def generate_smooth_spectrum(num_points=100):
    """
    Generate a more realistic random reflection in [0,1] by:
      - picking ~8 random control points in [0,1] x [0.0..1.0]
      - do a cubic spline -> 1D
      - clamp to [0,1]
    Requires SciPy for 'make_interp_spline'
    """
    n_ctrl= 8
    x_ctrl= np.linspace(0, 1, n_ctrl)
    y_ctrl= np.random.rand(n_ctrl)*0.8 + 0.1  # in [0.1..0.9]
    spline= make_interp_spline(x_ctrl, y_ctrl, k=3)
    x_big= np.linspace(0,1, num_points)
    y_big= spline(x_big)
    y_big= np.clip(y_big, 0, 1)
    return torch.tensor(y_big, dtype=torch.float)

@torch.no_grad()
def replicate_c4(verts):
    """
    Given points [N,2] in the first quadrant, replicate them by rotating
    0°, 90°, 180°, 270°, yielding negative coords for other quadrants.
    Returns shape [4N,2].
    """
    out_list=[]
    angles= [0, math.pi/2, math.pi, 3*math.pi/2]
    for a in angles:
        cosA= math.cos(a)
        sinA= math.sin(a)
        rot= torch.tensor([[cosA, -sinA],
                           [sinA,  cosA]],dtype=torch.float)
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
    from pyro.poutine import trace
    B=1
    c_b  = torch.zeros(B,1)
    ic_b = torch.zeros(B,dtype=torch.bool)
    vp_b = torch.zeros(B,MAX_STEPS)
    vw_b = torch.zeros(B,MAX_STEPS,2)
    iv_b = torch.zeros(B,dtype=torch.bool)

    g= trace(net.guide).get_trace(spectrum, c_b, ic_b, vp_b, vw_b, iv_b)
    raw_c_val= g.nodes["raw_c"]["value"]
    c_final= raw_c_val

    dummy0= g.nodes["dummy_pres_0"]["value"]
    v_pres_list=[dummy0]
    r_w0= g.nodes["raw_where_0"]["value"]
    w0= F.softplus(r_w0)
    v_where_list=[w0]
    prev= dummy0
    for t in range(1,MAX_STEPS):
        nm_p= f"raw_pres_{t}"
        nm_w= f"raw_where_{t}"
        rp= g.nodes[nm_p]["value"]
        rw= g.nodes[nm_w]["value"]
        rw= F.softplus(rw)
        v_pres_list.append(rp)
        v_where_list.append(rw)
        prev= rp

    v_pres_cat= torch.stack(v_pres_list, dim=1)   # [1,4]
    v_where_cat= torch.stack(v_where_list, dim=1) # [1,4,2]
    return c_final, v_pres_cat, v_where_cat

def plot_polygon(pts, c_val, out_path):
    """
    Angle-sort, close, fill => polygon in all quadrants
    """
    if pts.size(0)<2:
        plt.figure()
        plt.scatter(pts[:,0].numpy(), pts[:,1].numpy(), c='r')
        plt.title(f"Single Vertex c={c_val:.3f}")
        plt.savefig(out_path)
        plt.close()
        return
    sorted_pts= angle_sort(pts)
    closed_pts= close_polygon(sorted_pts)
    sx= closed_pts[:,0].numpy()
    sy= closed_pts[:,1].numpy()
    plt.figure()
    plt.fill(sx, sy, color='red', alpha=0.3)
    plt.plot(sx, sy, 'ro-')
    plt.title(f"C4 polygon, c={c_val:.3f}")
    plt.axhline(0,color='k',lw=0.5)
    plt.axvline(0,color='k',lw=0.5)
    plt.savefig(out_path)
    plt.close()

def test_inference():
    ckpt_path= os.path.join(SAVE_DIR, f"ckpt_epoch{NUM_EPOCHS}.pt")
    net= ForcedFirstVertexModel(N_WAVES)
    pyro.clear_param_store()
    if os.path.isfile(ckpt_path):
        st= torch.load(ckpt_path, map_location="cpu")
        pyro.get_param_store().set_state(st)
        print(f"[INFO] Loaded state from {ckpt_path}")
    else:
        print(f"[WARN] {ckpt_path} not found, using random init...")

    # Instead of random_sp= (torch.rand(N_WAVES)*0.5 +0.25) => we do a smooth curve:
    random_sp= generate_smooth_spectrum(N_WAVES).unsqueeze(0)  # shape [1,100]

    c_pred, v_pres_pred, v_where_pred= infer_latents(net, random_sp)
    recon= net.decoder(c_pred, v_pres_pred, v_where_pred).squeeze(0).detach()

    dt_str= datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir= f"inference_{dt_str}"
    os.makedirs(out_dir, exist_ok=True)

    np.savetxt(os.path.join(out_dir,"random_spectrum.txt"),
               random_sp.squeeze(0).numpy(), fmt="%.5f")
    np.savetxt(os.path.join(out_dir,"reconstructed_spectrum.txt"),
               recon.numpy(), fmt="%.5f")

    c_val= float(c_pred[0,0])
    with open(os.path.join(out_dir,"predicted_vertices.txt"),"w") as f:
        f.write(f"c={c_val:.3f}\n")
        for t in range(MAX_STEPS):
            ps= float(v_pres_pred[0,t])
            xx= float(v_where_pred[0,t,0])
            yy= float(v_where_pred[0,t,1])
            f.write(f"Vertex {t}: pres={ps:.3f}, x={xx:.3f}, y={yy:.3f}\n")

    # Plot spectrums
    x_axis= np.arange(N_WAVES)
    plt.figure()
    plt.plot(x_axis, random_sp.squeeze(0).numpy(), marker='o', label="Smooth InputSpec")
    plt.plot(x_axis, recon.numpy(), marker='x', label="Reconstructed")
    plt.legend()
    plt.title("Random (Smooth) vs. Reconstructed Spectrum")
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
        print("[WARN] No predicted vertices? That conflicts with forced first.")
    else:
        first_quad= torch.stack(keep_pts,dim=0)
        # replicate in all 4 quadrants
        c4_verts= replicate_c4(first_quad)
        # no clamp => negative x/y in other quadrants
        out_poly= os.path.join(out_dir,"c4_polygon.png")
        plot_polygon(c4_verts, c_val, out_poly)

    print(f"[INFO] Inference results saved to {out_dir}/")

def main():
    csv_path= "merged_s4_shapes.csv"
    # train
    train_model(csv_path)
    # inference
    test_inference()

if __name__=="__main__":
    main()
