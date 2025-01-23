#!/usr/bin/env python3

"""
Semi-Supervised AIR with Forced First Vertex (At Least One Vertex)
------------------------------------------------------------------
- c in [0,1] partially labeled (mask=0.7).
- The first vertex presence v_pres_0 = 1 forced in the model. 
- Up to 4 vertices total. 
- Reflection has 100 points. 
- Training for 100 epochs, printing progress via tqdm.trange.
- Post-training inference on a random 100-point reflection, with polygon plotting.

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
from pyro.poutine import trace
from tqdm import trange

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_STEPS = 4
SAVE_DIR = "results_forcedFirstVertex"
os.makedirs(SAVE_DIR, exist_ok=True)

###############################################################################
# 1) Data Loading + Semi-Supervision
###############################################################################
def load_data(csv_path):
    """
    We parse:
      - reflection columns R@... (100 points)
      - c in [0,1]
      - v_pres_0..3, v_where_0..3_x,y if available
    Returns:
      spectra: [N,100]
      c_vals: [N,1]
      is_c_known: bool[N]
      v_pres: [N,4]
      v_where: [N,4,2]
      is_v_known: bool[N]
    """
    df = pd.read_csv(csv_path)
    # reflection
    r_cols = [c for c in df.columns if c.startswith("R@")]
    r_cols = sorted(r_cols, key=lambda x: float(x.split('@')[1]))
    spectra_np = df[r_cols].values.astype(np.float32)
    spectra = torch.from_numpy(spectra_np)
    N, n_waves = spectra.shape
    if n_waves != 100:
        print(f"[WARN] data has {n_waves} reflection points, not 100.")

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
    have_where= all((wx in df.columns and wy in df.columns) 
                    for wx,wy in zip(wx_cols,wy_cols))

    if have_pres and have_where:
        v_pres_np= df[pres_cols].values.astype(np.float32)
        vx_np= df[wx_cols].values.astype(np.float32)
        vy_np= df[wy_cols].values.astype(np.float32)
        row_nan= (np.isnan(v_pres_np).any(axis=1)|
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
        v_pres = torch.zeros(N, MAX_STEPS)
        v_where= torch.zeros(N, MAX_STEPS,2)
        is_v_known= torch.zeros(N, dtype=torch.bool)

    return spectra, c_vals, is_c_known, v_pres, v_where, is_v_known

def apply_semi_mask(is_c_known, is_v_known, fraction=0.7, seed=123):
    """
    Keep 'fraction' of known labels as known, random with fixed seed. 
    Others => unknown => 0
    """
    N= len(is_c_known)
    np.random.seed(seed)
    r= np.random.rand(N)
    keep= r< fraction
    final_c= is_c_known.numpy() & keep
    final_v= is_v_known.numpy() & keep
    return torch.from_numpy(final_c), torch.from_numpy(final_v)

###############################################################################
# 2) Model + Guide (Force 1st Vertex)
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
        # c: [B,1], v_pres: [B,4], v_where: [B,4,2]
        B, hidden_dim= c.size(0), 64
        accum= torch.zeros(B, hidden_dim, device=c.device)
        for t in range(MAX_STEPS):
            feat= self.vert_embed(v_where[:,t,:])
            pres= v_pres[:,t:t+1]
            accum= accum+ feat*pres
        x= torch.cat([accum, c], dim=-1)
        return self.final_net(x)

class ForcedFirstVertexModel(nn.Module):
    """
    We always set the first vertex presence = 1 or known in model & guide,
    allowing up to 3 more with geometric approach.
    c => Beta(2,2) or Delta if known
    decode => Normal( mean, 0.01)
    """
    def __init__(self, n_waves=100):
        super().__init__()
        self.n_waves= n_waves
        self.decoder= Decoder(n_waves, 64)

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

            # presence
            # forced first => 1 
            # if known => Delta, else => Delta(1)
            if_b_known0= (is_v_known.float()*(v_pres[:,0]>=0.5).float()).unsqueeze(-1)
            raw_p0= torch.ones(B,1, device=spectrum.device)
            pres_0= if_b_known0*v_pres[:,0:1] + (1-if_b_known0)*raw_p0

            # location for t=0
            loc0= torch.zeros(B,2, device=spectrum.device)
            sc0= torch.ones(B,2, device=spectrum.device)
            raw_w0= pyro.sample("raw_where_0",
                                dist.Normal(loc0, sc0)
                                    .mask(pres_0)
                                    .to_event(1))
            w0_mask= is_v_known.float().unsqueeze(-1)*pres_0
            w0_val= w0_mask*v_where[:,0,:] + (1-w0_mask)*raw_w0

            v_pres_list= [pres_0]
            v_where_list= [w0_val]
            prev_pres= pres_0

            # t=1..3
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

            v_pres_cat= torch.cat(v_pres_list, dim=1)
            v_where_cat= torch.stack(v_where_list, dim=1)
            mean_sp= self.decoder(c, v_pres_cat, v_where_cat)
            pyro.sample("obs_spectrum",
                        dist.Normal(mean_sp, 0.01).to_event(1),
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
            # c
            out_c= self.c_net(spectrum)
            alpha_ = F.softplus(out_c[:,0:1])+1
            beta_  = F.softplus(out_c[:,1:2]) +1
            raw_c= pyro.sample("raw_c",
                               dist.Beta(alpha_, beta_).to_event(1))

            # presence t=0 => forced=1
            # guide => sample => raw? We'll do the same as model, but no randomness. 
            # We'll define a sample "raw_pres_0" with Bernoulli(prob=1)
            # Actually we used "raw_p0" in the model, let's keep naming consistent:
            prob0= torch.ones(B,1, device=spectrum.device)
            raw_p0= pyro.sample("dummy_pres_0", 
                                dist.Bernoulli(prob0).to_event(1)) # shape [B,1] => always 1
            if_known0= (is_v_known.float()*(v_pres[:,0]>=0.5).float()).unsqueeze(-1)
            pres_0= if_known0*v_pres[:,0:1] + (1-if_known0)*raw_p0

            # location t=0 => param net
            if not hasattr(self,"where_0_net"):
                self.where_0_net= nn.Sequential(
                    nn.Linear(self.n_waves,4)
                )
            w0_out= self.where_0_net(spectrum)
            w0_loc= w0_out[:,0:2]
            w0_scale= F.softplus(w0_out[:,2:4])+1e-4
            raw_w0= pyro.sample("raw_where_0",
                                dist.Normal(w0_loc,w0_scale)
                                    .mask(pres_0)
                                    .to_event(1))
            w0_mask= is_v_known.float().unsqueeze(-1)*pres_0
            w0_val= w0_mask*v_where[:,0,:] + (1-w0_mask)*raw_w0

            prev_pres= pres_0

            for t in range(1,MAX_STEPS):
                nm_p= f"raw_pres_{t}"
                if not hasattr(self, f"pres_net_{t}"):
                    setattr(self, f"pres_net_{t}",
                            nn.Sequential(
                                nn.Linear(self.n_waves,1),
                                nn.Sigmoid()
                            ))
                pres_net= getattr(self, f"pres_net_{t}")
                p_prob_raw= pres_net(spectrum)
                p_prob= p_prob_raw*prev_pres
                raw_p= pyro.sample(nm_p,
                                   dist.Bernoulli(p_prob).to_event(1))
                if_known= (is_v_known.float()*(v_pres[:,t]>=0.5).float()).unsqueeze(-1)
                p_val= if_known*v_pres[:,t:t+1] + (1-if_known)*raw_p

                nm_w= f"raw_where_{t}"
                if not hasattr(self, f"where_net_{t}"):
                    setattr(self, f"where_net_{t}",
                            nn.Sequential(
                                nn.Linear(self.n_waves,4)
                            ))
                w_net= getattr(self, f"where_net_{t}")
                w_out= w_net(spectrum)
                loc= w_out[:,0:2]
                sc = F.softplus(w_out[:,2:4])+1e-4
                raw_w= pyro.sample(nm_w,
                                   dist.Normal(loc, sc)
                                       .mask(raw_p)
                                       .to_event(1))
                w_mask= is_v_known.float().unsqueeze(-1)*p_val
                w_val= w_mask*v_where[:,t,:] + (1-w_mask)*raw_w
                prev_pres= p_val

###############################################################################
# 3) Train
###############################################################################
def train(csv_path="my_100point_data.csv", 
          num_epochs=100, 
          batch_size=512, 
          lr=1e-3, 
          mask_frac=0.7):
    # load
    spectra, c_vals, is_c_known, v_pres, v_where, is_v_known= load_data(csv_path)
    N,n_waves= spectra.shape

    # semi mask
    final_c, final_v= apply_semi_mask(is_c_known, is_v_known, fraction=mask_frac, seed=123)

    # model
    net= ForcedFirstVertexModel(n_waves)
    pyro.clear_param_store()
    optimizer= optim.Adam({"lr": lr})
    svi= SVI(net.model, net.guide, optimizer, loss=TraceGraph_ELBO())

    idx_arr= np.arange(N)
    from tqdm import trange
    for ep in trange(num_epochs, desc="Training"):
        np.random.shuffle(idx_arr)
        total_loss= 0.0
        for start_i in range(0,N,batch_size):
            end_i= start_i+batch_size
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
        print(f"[Epoch {ep+1}/{num_epochs}] ELBO={avg_elbo:.2f}")

        # save param store
        st= pyro.get_param_store().get_state()
        torch.save(st, os.path.join(SAVE_DIR, f"ckpt_epoch{ep+1}.pt"))

    print("Training complete!")

###############################################################################
# 4) Inference
###############################################################################
@torch.no_grad()
def replicate_c4(verts):
    """
    replicate points in [n,2] for 4 quadrants
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

@torch.no_grad()
def infer_latents(net, spectrum):
    """
    net: ForcedFirstVertexModel
    spectrum: [1,100]
    Returns c, v_pres, v_where
    """
    from pyro.poutine import trace
    B=1
    c_b= torch.zeros(B,1)
    ic_b= torch.zeros(B,dtype=torch.bool)
    vp_b= torch.zeros(B,MAX_STEPS)
    vw_b= torch.zeros(B,MAX_STEPS,2)
    iv_b= torch.zeros(B,dtype=torch.bool)

    g_tr= trace(net.guide).get_trace(spectrum,c_b,ic_b,vp_b,vw_b,iv_b)
    raw_c_val= g_tr.nodes["raw_c"]["value"]
    c_final= raw_c_val

    # t=0 => presence => "raw_p0"? we used "dummy_pres_0" or "raw_where_0_pres" 
    # in the code we used "dummy_pres_0"? let's see. Actually we used
    #   pyro.sample("raw_where_0", ...) for location
    #   pyro.sample("raw_pres_0", ...) is not in the code. We used "dummy_pres_0"? let's double-check the code.
    # We named it "raw_p0" for the model, but in the guide we do "dummy_pres_0"? 
    # Let's see if it shows up in the trace.
    # We'll parse them out carefully:
    v_pres_list=[]
    v_where_list=[]

    # guide: we do pyro.sample("raw_where_0", ...) => location
    #         pyro.sample("dummy_pres_0", Bernoulli(1)) => presence
    dummy_p0_val= g_tr.nodes["dummy_pres_0"]["value"]
    v_pres_list.append(dummy_p0_val) # shape [B,1]
    w0_val= g_tr.nodes["raw_where_0"]["value"]
    v_where_list.append(w0_val)

    prev= dummy_p0_val
    for t in range(1,MAX_STEPS):
        nm_p= f"raw_pres_{t}"
        raw_p_val= g_tr.nodes[nm_p]["value"]
        v_pres_list.append(raw_p_val)
        nm_w= f"raw_where_{t}"
        raw_w_val= g_tr.nodes[nm_w]["value"]
        v_where_list.append(raw_w_val)
        prev= raw_p_val

    v_pres_cat= torch.cat(v_pres_list,dim=1)
    v_where_cat= torch.stack(v_where_list,dim=1)
    return c_final, v_pres_cat, v_where_cat

def plot_inference(random_spec, recon_spec, c_pred, v_pres_pred, v_where_pred, out_dir="inference_plots"):
    os.makedirs(out_dir, exist_ok=True)
    random_spec_np = random_spec.detach().cpu().numpy()
    recon_np       = recon_spec.detach().cpu().numpy()
    # 1) spectra
    n_waves= random_spec.size(0)
    x_axis= np.arange(n_waves)
    plt.figure()
    plt.plot(x_axis, random_spec_np, label="Input Spectrum", marker='o')
    plt.plot(x_axis, recon_np, label="Reconstructed", marker='x')
    plt.legend()
    plt.title("Random vs. Reconstructed Spectrum")
    plt.xlabel("Index")
    plt.ylabel("Reflectance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"spectra_compare.png"))
    plt.close()

    # 2) polygon => show forced vertex + others if presence>0.5
    c_val= float(c_pred[0,0])
    keep_verts = []
    # the first vertex is always present
    first_xy= v_where_pred[0,0,:].detach()
    keep_verts.append(first_xy)
    for t in range(1,MAX_STEPS):
        if float(v_pres_pred[0,t])>0.5:
            keep_verts.append(v_where_pred[0,t,:].detach())
    if len(keep_verts)==0:
        print("[WARN] no predicted vertices?! (Should at least have the first though.)")
        return
    vert_stack= torch.stack(keep_verts, dim=0)
    c4 = replicate_c4(vert_stack)
    c4_np= c4.cpu().numpy()
    plt.figure()
    plt.scatter(c4_np[:,0], c4_np[:,1], c='r', marker='o')
    plt.axhline(0,c='k',lw=0.5)
    plt.axvline(0,c='k',lw=0.5)
    plt.title(f"C4 polygon from predicted vertices, c={c_val:.3f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"c4_polygon.png"))
    plt.close()
    print(f"Plots saved in {out_dir}/")

###############################################################################
# 5) Main
###############################################################################
def main():
    # 1) Train for 100 epochs
    # Suppose your CSV is "mydata_100.csv"
    csv_path= "merged_s4_shapes.csv"
    train(csv_path=csv_path, num_epochs=100, batch_size=512, lr=1e-3, mask_frac=0.7)

    # 2) Inference test
    # Let's load epoch100
    ckpt_path = os.path.join(SAVE_DIR,"ckpt_epoch100.pt")
    net= ForcedFirstVertexModel(n_waves=100)
    pyro.clear_param_store()
    if os.path.isfile(ckpt_path):
        st= torch.load(ckpt_path, map_location="cpu")
        pyro.get_param_store().set_state(st)
        print(f"[INFO] loaded param store from {ckpt_path}")
    else:
        print(f"[WARN] no {ckpt_path}, using random init...")

    # 3) Generate random 100-pt reflection
    random_spec= (torch.rand(100)*0.5 +0.25).unsqueeze(0) # [1,100]
    # 4) Infer
    c_pred, v_pres_pred, v_where_pred= infer_latents(net, random_spec)
    # 5) Reconstruct
    rec= net.decoder(c_pred, v_pres_pred, v_where_pred).squeeze(0).detach()
    # 6) Print latents
    c_val= float(c_pred[0,0])
    print(f"Predicted c={c_val:.3f}")
    for t in range(MAX_STEPS):
        pres_v= float(v_pres_pred[0,t])
        loc_x= float(v_where_pred[0,t,0])
        loc_y= float(v_where_pred[0,t,1])
        print(f" Vertex {t}: presence={pres_v:.3f}, location=({loc_x:.3f}, {loc_y:.3f})")
    # 7) Plot
    out_dir= "inference_forced_first_vertex_plots"
    random_spec_1d= random_spec.squeeze(0).detach()
    plot_inference(random_spec_1d, rec, c_pred, v_pres_pred, v_where_pred, out_dir=out_dir)

    print("Done!")

if __name__=="__main__":
    main()
