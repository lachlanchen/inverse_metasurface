#!/usr/bin/env python3

"""
Semi-Supervised AIR with Guaranteed First Vertex:
 - We always set v_pres_0 = 1 in the model => at least 1 vertex present.
 - We allow up to 4 vertices total: t=0 forced, t=1..3 geometric approach.
 - c in [0,1] partially labeled. 
 - v partially labeled (presence & location).
 - 100 reflection points.

We demonstrate training plus an inference script that:
 - Loads a checkpoint
 - Infers latents from a random 100-point spectrum
 - Reconstructs the spectrum
 - Plots everything
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_STEPS = 4
SAVE_DIR = "results_forcedFirstVertex"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# 1) Data loading
# -------------------------
def load_data(csv_path):
    """
    We parse columns:
      R@... => reflection (100 points)
      c => maybe partial known in [0,1]
      v_pres_{t}, v_where_{t}_x,y => up to 4 vertices
    Return:
      spectra: [N,100]
      c_vals: [N,1], 0 if unknown
      is_c_known: bool[N]
      v_pres: [N,4], v_where: [N,4,2], is_v_known: bool[N]
    """
    df = pd.read_csv(csv_path)
    # reflection
    r_cols = [c for c in df.columns if c.startswith("R@")]
    r_cols = sorted(r_cols, key=lambda x: float(x.split('@')[1]))
    spectra_np = df[r_cols].values.astype(np.float32)
    spectra = torch.from_numpy(spectra_np)
    N, n_waves = spectra.shape
    if n_waves != 100:
        print(f"[WARN] The data has {n_waves} reflection points, not 100.")
    # c
    c_np = np.full((N,1), np.nan, dtype=np.float32)
    if "c" in df.columns:
        c_np = df["c"].values.reshape(-1,1).astype(np.float32)
    is_c_known_np = ~np.isnan(c_np).reshape(-1)
    c_np[np.isnan(c_np)] = 0.0
    c_vals = torch.from_numpy(c_np)
    is_c_known = torch.from_numpy(is_c_known_np)

    # v
    pres_cols = [f"v_pres_{t}" for t in range(MAX_STEPS)]
    wx_cols   = [f"v_where_{t}_x" for t in range(MAX_STEPS)]
    wy_cols   = [f"v_where_{t}_y" for t in range(MAX_STEPS)]
    have_pres = all(pc in df.columns for pc in pres_cols)
    have_where= all((wx in df.columns and wy in df.columns) for wx,wy in zip(wx_cols,wy_cols))

    if have_pres and have_where:
        v_pres_np = df[pres_cols].values.astype(np.float32)
        vx_np     = df[wx_cols].values.astype(np.float32)
        vy_np     = df[wy_cols].values.astype(np.float32)
        row_has_nan = (np.isnan(v_pres_np).any(axis=1)|
                       np.isnan(vx_np).any(axis=1)|
                       np.isnan(vy_np).any(axis=1))
        is_v_known_np = ~row_has_nan
        v_pres_np[np.isnan(v_pres_np)] = 0
        vx_np[np.isnan(vx_np)] = 0
        vy_np[np.isnan(vy_np)] = 0
        v_pres = torch.from_numpy(v_pres_np)
        v_where= torch.stack([torch.from_numpy(vx_np),
                              torch.from_numpy(vy_np)],dim=-1) # [N,4,2]
        is_v_known= torch.from_numpy(is_v_known_np)
    else:
        v_pres = torch.zeros(N, MAX_STEPS)
        v_where= torch.zeros(N, MAX_STEPS,2)
        is_v_known= torch.zeros(N,dtype=torch.bool)

    return spectra, c_vals, is_c_known, v_pres, v_where, is_v_known

def apply_semi_mask(is_c_known, is_v_known, fraction=0.7, seed=123):
    """
    Keep 'fraction' of known labels as known, random with fixed seed. 
    Others => unknown => 0
    """
    N = len(is_c_known)
    np.random.seed(seed)
    r = np.random.rand(N)
    keep = r < fraction
    final_c = is_c_known.numpy() & keep
    final_v = is_v_known.numpy() & keep
    return torch.from_numpy(final_c), torch.from_numpy(final_v)

# -------------------------
# 2) Model & Guide
#    Force v_pres_0=1 => first vertex always present
# -------------------------
class Decoder(nn.Module):
    """
    c + up to 4 vertices => reflect(100)
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
        B, hidden_dim = c.size(0), 64
        accum = torch.zeros(B, hidden_dim, device=c.device)
        for t in range(MAX_STEPS):
            feat= self.vert_embed(v_where[:,t,:])
            pres= v_pres[:,t:t+1]
            accum += feat*pres
        x = torch.cat([accum, c], dim=-1)
        return self.final_net(x)

class ForcedFirstVertexModel(nn.Module):
    def __init__(self, n_waves=100):
        super().__init__()
        self.decoder= Decoder(n_waves,64)
        self.n_waves= n_waves

    def model(self, 
              spectrum,
              c_vals, is_c_known,
              v_pres, v_where, is_v_known):
        """
        c => Beta(2,2) or Delta if known
        v_pres_0 => forced 1
        v_pres_{t>0} => Bernoulli(0.5 * prev)
        v_where => Normal(0,1) or Delta
        decode => Normal
        """
        pyro.module("ForcedFVModel", self)
        B = spectrum.size(0)
        with pyro.plate("data", B):
            # c
            # if known => Delta, else Beta(2,2)
            alpha,beta= 2.0,2.0
            raw_c= pyro.sample("raw_c", dist.Beta(alpha*torch.ones(B,1),
                                                  beta*torch.ones(B,1)).to_event(1))
            # combine if known
            c_mask= is_c_known.float().unsqueeze(-1)
            c = c_mask*c_vals + (1-c_mask)*raw_c

            # presence
            # force v_pres_0 => 1 or known
            v_pres_list=[]
            v_where_list=[]
            # t=0
            # if known => Delta, else Delta(1)
            if_b_known = (is_v_known.float()*(v_pres[:,0]>=0.5).float()).unsqueeze(-1)
            # raw_0 => Delta(1)
            raw_p0 = torch.ones(B,1, device=spectrum.device)
            pres_0 = if_b_known*v_pres[:,0:1] + (1-if_b_known)*raw_p0
            v_pres_list.append(pres_0)

            # v_where_0 => if known => Delta, else => Normal(0,1) masked by pres_0
            loc0 = torch.zeros(B,2, device=spectrum.device)
            sc0  = torch.ones(B,2, device=spectrum.device)
            raw_w0= pyro.sample("raw_where_0",
                                dist.Normal(loc0, sc0)
                                    .mask(pres_0)
                                    .to_event(1))
            w0_mask= is_v_known.float().unsqueeze(-1)*pres_0
            w_val0= w0_mask*v_where[:,0,:] + (1-w0_mask)*raw_w0
            v_where_list.append(w_val0)

            prev_pres = pres_0
            # t=1..3 => Bernoulli(0.5 * prev)
            for t in range(1,MAX_STEPS):
                nm_p= f"raw_pres_{t}"
                p_prob= 0.5*prev_pres
                raw_p= pyro.sample(nm_p,
                                   dist.Bernoulli(p_prob).to_event(1))
                # combine if known
                if_known= (is_v_known.float()*(v_pres[:,t]>=0.5).float()).unsqueeze(-1)
                p_val = if_known*v_pres[:,t:t+1] + (1-if_known)*raw_p

                nm_w= f"raw_where_{t}"
                loc= torch.zeros(B,2, device=spectrum.device)
                sc = torch.ones(B,2, device=spectrum.device)
                raw_w= pyro.sample(nm_w,
                                   dist.Normal(loc,sc).mask(raw_p).to_event(1))
                w_mask= is_v_known.float().unsqueeze(-1)*p_val
                w_val= w_mask*v_where[:,t,:] + (1-w_mask)*raw_w

                v_pres_list.append(p_val)
                v_where_list.append(w_val)
                prev_pres= p_val

            # decode
            v_pres_cat= torch.cat(v_pres_list, dim=1)   # [B,4]
            v_where_cat= torch.stack(v_where_list, dim=1) # [B,4,2]
            mean_sp= self.decoder(c, v_pres_cat, v_where_cat)
            pyro.sample("obs_spectrum",
                        dist.Normal(mean_sp,0.01).to_event(1),
                        obs=spectrum)

    def guide(self, 
              spectrum,
              c_vals, is_c_known,
              v_pres, v_where, is_v_known):
        pyro.module("ForcedFVModel", self)
        B= spectrum.size(0)

        # param net for c
        if not hasattr(self,"c_net"):
            self.c_net= nn.Sequential(
                nn.Linear(self.n_waves, 2),
            )

        with pyro.plate("data", B):
            # c param => alpha,beta => we do a small transform
            out_c= self.c_net(spectrum)
            alpha_ = F.softplus(out_c[:,0:1])+1
            beta_  = F.softplus(out_c[:,1:2]) +1
            raw_c= pyro.sample("raw_c", dist.Beta(alpha_, beta_).to_event(1))

            # presence, location => forced first is pres=1 => no randomness 
            # so we do Delta(1) or if known => Delta( known )
            # Actually we can't have a "Delta(1)" in the guide if we do the same shape as the model
            # We'll do the same approach: but that presence has prob=1 => Bernoulli(prob=1).
            # if known => Delta( known ), else => Bernoulli(1)
            # and location => if known => Delta, else => Normal( param from net? ) ?

            # We keep it simple: we won't define large param nets for each step. We'll do a single approach for demonstration:

            # t=0 => presence=1
            # in the guide, do the same to keep shapes consistent
            # We'll do raw_pres_0 => Bernoulli(prob=1)
            prob0= torch.ones(B,1, device=spectrum.device)
            raw_p0= pyro.sample("raw_where_0_pres",
                                dist.Bernoulli(prob0).to_event(1)) # shape [B,1], always 1
            # if known => Delta
            if_known0= (is_v_known.float()*(v_pres[:,0]>=0.5).float()).unsqueeze(-1)
            pres_0 = if_known0*v_pres[:,0:1] + (1-if_known0)*raw_p0

            # location t=0
            # if known => Delta, else => Normal( param )
            if not hasattr(self, "where_net_0"):
                self.where_net_0 = nn.Sequential(
                    nn.Linear(self.n_waves,4)
                )
            w0_out= self.where_net_0(spectrum)
            w0_loc= w0_out[:,0:2]
            w0_scale= F.softplus(w0_out[:,2:4])+1e-4
            raw_w0= pyro.sample("raw_where_0",
                                dist.Normal(w0_loc, w0_scale)
                                    .mask(pres_0)
                                    .to_event(1))
            # combine if known
            w0_mask= is_v_known.float().unsqueeze(-1)*pres_0
            w0_val= w0_mask*v_where[:,0,:] + (1-w0_mask)*raw_w0

            prev_pres= pres_0

            # t=1..3
            for t in range(1,MAX_STEPS):
                # presence
                nm_p= f"raw_pres_{t}"
                # define param net
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

                # location
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
                                   dist.Normal(loc,sc)
                                   .mask(raw_p)
                                   .to_event(1))
                w_mask= is_v_known.float().unsqueeze(-1)*p_val
                w_val= w_mask*v_where[:,t,:] + (1-w_mask)*raw_w

                prev_pres= p_val

# -------------------------
# 3) Train
# -------------------------
def train(csv_path="mydata.csv", num_epochs=5, batch_size=512, lr=1e-3):
    spectra, c_vals, is_c_known, v_pres, v_where, is_v_known= load_data(csv_path)
    N, n_waves= spectra.shape

    # random partial
    final_c, final_v= apply_semi_mask(is_c_known, is_v_known, fraction=0.7, seed=123)

    net= ForcedFirstVertexModel(n_waves)
    pyro.clear_param_store()
    optimizer= optim.Adam({"lr":lr})
    svi= SVI(net.model, net.guide, optimizer, loss=TraceGraph_ELBO())

    idx= np.arange(N)
    for ep in range(num_epochs):
        np.random.shuffle(idx)
        total_loss=0.0
        for start_i in range(0,N,batch_size):
            end_i= start_i+batch_size
            sb= idx[start_i:end_i]
            sp_b= spectra[sb]
            c_b = c_vals[sb]
            ic_b= final_c[sb]
            vp_b= v_pres[sb]
            vw_b= v_where[sb]
            iv_b= final_v[sb]
            loss= svi.step(sp_b, c_b, ic_b, vp_b, vw_b, iv_b)
            total_loss+= loss
        avg_loss= total_loss/N
        print(f"[Epoch {ep+1}/{num_epochs}] ELBO={avg_loss:.2f}")
        # save
        st= pyro.get_param_store().get_state()
        torch.save(st, os.path.join(SAVE_DIR,f"ckpt_epoch{ep+1}.pt"))
    print("Training done.")

# -------------------------
# 4) Inference / Plot
#    We'll do forced first vertex presence => the model is trained that way
# -------------------------
@torch.no_grad()
def infer_latents(net, spectrum):
    """
    net: ForcedFirstVertexModel
    spectrum: [1,100]
    Return: c, v_pres, v_where
    """
    from pyro.poutine import trace
    B=1
    c_b= torch.zeros(B,1)
    ic_b= torch.zeros(B,dtype=torch.bool)
    vp_b= torch.zeros(B,MAX_STEPS)
    vw_b= torch.zeros(B,MAX_STEPS,2)
    iv_b= torch.zeros(B,dtype=torch.bool)

    g_tr= trace(net.guide).get_trace(spectrum, c_b, ic_b, vp_b, vw_b, iv_b)

    # parse c => we have "raw_c"
    raw_c_val= g_tr.nodes["raw_c"]["value"] # [B,1]
    # presence => "raw_where_0_pres"? actually we might have "raw_where_0"? 
    # We see we forcibly sample "raw_where_0_pres"? let's check we named it "raw_where_0" for location, "raw_where_0_pres" for presence?
    # Actually the code has "raw_where_0_pres" for presence t=0
    # But let's unify the logic with the model. The model has no "raw_pres_0", it has forced presence=1 => no sample. 
    # So let's read from the guide:
    #  "raw_where_0_pres" => shape [B,1]
    #  for t=1..3 => "raw_pres_t"
    #  for location => "raw_where_t"
    c_final= raw_c_val

    v_pres_list=[]
    v_where_list=[]
    # t=0 => presence
    # we named => "raw_where_0_pres" 
    # let's check if it actually is in the trace or if it might be absent if there's no pyro.sample
    # We see we do pyro.sample("raw_where_0_pres", Bernoulli(prob0)) in the guide => so yes it should exist
    name_p0= "raw_where_0_pres"
    raw_p0_val = g_tr.nodes[name_p0]["value"] # shape [B,1]
    # combine with known? Actually in the guide we do if_known0 => we do that in model. Let's replicate. 
    # We'll just read out the final presence = raw_p0_val, because there's no partial known in inference.
    pres_0= raw_p0_val
    v_pres_list.append(pres_0)

    # location t=0 => "raw_where_0"
    raw_w0= g_tr.nodes["raw_where_0"]["value"] # shape [B,2]
    v_where_list.append(raw_w0)

    prev_pres= pres_0
    for t in range(1,MAX_STEPS):
        nm_p= f"raw_pres_{t}"
        raw_p_val= g_tr.nodes[nm_p]["value"]
        v_pres_list.append(raw_p_val)
        nm_w= f"raw_where_{t}"
        raw_w_val= g_tr.nodes[nm_w]["value"]
        v_where_list.append(raw_w_val)
        prev_pres= raw_p_val

    v_pres_cat= torch.cat(v_pres_list,dim=1)
    v_where_cat= torch.stack(v_where_list,dim=1)
    return c_final, v_pres_cat, v_where_cat

def replicate_c4(verts):
    """
    replicate points in verts [N,2] for 4 quadrants
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

def test_inference(ckpt_path="results_forcedFirstVertex/ckpt_epoch5.pt"):
    n_waves=100
    device="cpu"
    net= ForcedFirstVertexModel(n_waves).to(device)
    pyro.clear_param_store()
    if os.path.isfile(ckpt_path):
        st= torch.load(ckpt_path, map_location=device)
        pyro.get_param_store().set_state(st)
        print(f"[INFO] Loaded from {ckpt_path}")
    else:
        print(f"[WARN] no {ckpt_path}, random init...")

    # generate random spectrum [1,100]
    # let's do random. If you want something more realistic, do e.g. smooth function
    random_spec= torch.rand(n_waves, device=device)*0.5 +0.25
    random_spec= random_spec.unsqueeze(0) # shape [1,100]

    # do inference
    c_pred, v_pres_pred, v_where_pred= infer_latents(net, random_spec)
    # decode
    recon= net.decoder(c_pred, v_pres_pred, v_where_pred)
    recon= recon.squeeze(0).detach()

    c0= float(c_pred[0,0])
    print(f"Predicted c={c0:.3f}")

    # Print presence & location
    pres_np= v_pres_pred.squeeze(0).detach().numpy()
    where_np= v_where_pred.squeeze(0).detach().numpy()
    for t in range(MAX_STEPS):
        print(f" Vertex {t}: presence={pres_np[t]:.3f}, location=({where_np[t,0]:.3f},{where_np[t,1]:.3f})")

    n_present= int((v_pres_pred>0.5).sum().item())
    print(f" [INFO] predicted # vertices present={n_present}")

    # plot
    out_dir= "inference_forcedFirstVertex"
    os.makedirs(out_dir, exist_ok=True)
    # 1) spectra
    x_axis= np.arange(n_waves)
    plt.figure()
    plt.plot(x_axis, random_spec.squeeze(0).cpu().detach().numpy(), label="Input Spectrum", marker='o')
    plt.plot(x_axis, recon.cpu().numpy(), label="Reconstructed", marker='x')
    plt.title("Random Spectrum vs Reconstructed")
    plt.xlabel("Index")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"spectra_compare.png"))
    plt.close()

    # 2) c4 polygon if we have any presence>0.5
    keep_verts=[]
    pres_thr= (v_pres_pred>0.5).squeeze(0).cpu().numpy()
    for t in range(MAX_STEPS):
        if pres_thr[t]>0.5:
            keep_verts.append(v_where_pred[0,t,:].detach())
    if len(keep_verts)==0:
        print("[WARN] no predicted vertices.")
    else:
        keep_stack= torch.stack(keep_verts,dim=0)
        c4_verts= replicate_c4(keep_stack)
        c4_np= c4_verts.cpu().detach().numpy()
        plt.figure()
        plt.scatter(c4_np[:,0], c4_np[:,1], c='r', marker='o', label="C4 polygon points")
        plt.axhline(0,c='k',lw=0.5)
        plt.axvline(0,c='k',lw=0.5)
        plt.title(f"C4 polygon from predicted vertices (forced 1st). c={c0:.3f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,"c4_polygon.png"))
        plt.close()

    # 3) Save data
    np.savetxt(os.path.join(out_dir,"random_spectrum.txt"), random_spec.squeeze(0).cpu().detach().numpy(), fmt="%.5f")
    np.savetxt(os.path.join(out_dir,"reconstructed_spectrum.txt"), recon.cpu().numpy(), fmt="%.5f")
    with open(os.path.join(out_dir,"predicted_latents.txt"),"w") as f:
        f.write(f"Predicted c={c0:.5f}\n")
        for t in range(MAX_STEPS):
            ps= float(v_pres_pred[0,t])
            wx= float(v_where_pred[0,t,0])
            wy= float(v_where_pred[0,t,1])
            f.write(f" t={t}: presence={ps:.3f}, (x,y)=({wx:.3f},{wy:.3f})\n")
    print(f"[INFO] Plots + data saved in {out_dir}/")

# -------------------------
# 5) Main
# -------------------------
def main():
    # Example usage:
    # 1) train => "results_forcedFirstVertex/ckpt_epochX.pt"
    # 2) test_inference => loads epoch5
    # (comment or reorder as needed)

    # Example train call:
    # train("my_100point_data.csv", num_epochs=5, batch_size=512, lr=1e-3)

    # Example inference:
    test_inference("results_forcedFirstVertex/ckpt_epoch5.pt")

if __name__=="__main__":
    main()
