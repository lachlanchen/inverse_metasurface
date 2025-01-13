#!/usr/bin/env python3

"""
Semi-Supervised AIR Metasurface with Extra Visualization and Performance Print
------------------------------------------------------------------------------

We assume a CSV with columns:
  - c (float, possibly 0..1)
  - R@... reflection columns
  - v_pres_0..3, v_where_0_x..3, v_where_0_y..3 (for 4 vertices) 
    fully labeled, but we demonstrate partial random masking.

We do:
 - Random mask on c and v with a fixed RANDOM_SEED to simulate partial-labeled data.
 - The model uses Delta(...) for known c or v, else from prior.
 - The guide uses a param net for unknown c or v.
 - After each epoch, we measure:
    1) "Spectrum->(c,v)": sampling from guide, compare with ground-truth c & v
       => MSE of c, presence, location. 
    2) "(c,v)->Spectrum": decode the known c & v to see MSE vs. real spectrum.

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

SAVE_DIR = "svi_results"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_STEPS = 4
RANDOM_SEED = 1234  # for reproducibility of random masking
MASK_FRAC = 0.7  # fraction of data that remains labeled

###############################################################################
# 1) Data Loading
###############################################################################
def load_data(csv_path):
    """
    We parse reflection (R@...), c, v_pres, v_where if present.
    Return: 
      spectra: [N, n_waves]
      c_vals: [N,1] float
      is_c_known: bool [N]
      v_pres: [N,4], v_where: [N,4,2]
      is_v_known: bool [N]
    """
    df = pd.read_csv(csv_path)
    # reflection
    r_cols = [c for c in df.columns if c.startswith("R@")]
    r_cols = sorted(r_cols, key=lambda x: float(x.split('@')[1]))
    spectra_np = df[r_cols].values.astype(np.float32)
    spectra = torch.from_numpy(spectra_np)

    N, n_waves = spectra.shape

    # c
    c_np = np.full((N,1), np.nan, dtype=np.float32)
    if "c" in df.columns:
        c_np = df["c"].values.reshape(-1,1).astype(np.float32)
    is_c_known_np = ~np.isnan(c_np).reshape(-1)
    c_np[np.isnan(c_np)] = 0.0
    c_vals = torch.from_numpy(c_np)
    is_c_known = torch.from_numpy(is_c_known_np)

    # v_pres / v_where
    pres_cols = [f"v_pres_{t}" for t in range(MAX_STEPS)]
    wx_cols   = [f"v_where_{t}_x" for t in range(MAX_STEPS)]
    wy_cols   = [f"v_where_{t}_y" for t in range(MAX_STEPS)]
    have_pres = all(c in df.columns for c in pres_cols)
    have_where= all((wx in df.columns and wy in df.columns) for wx,wy in zip(wx_cols,wy_cols))

    if have_pres and have_where:
        v_pres_np = df[pres_cols].values.astype(np.float32)  # [N,4]
        vx_np = df[wx_cols].values.astype(np.float32)        # [N,4]
        vy_np = df[wy_cols].values.astype(np.float32)        # [N,4]
        # find rows with any nan
        row_has_nan = np.any(np.isnan(v_pres_np), axis=1) | np.any(np.isnan(vx_np),axis=1) | np.any(np.isnan(vy_np),axis=1)
        is_v_known_np = ~row_has_nan
        # fill unknown with 0
        v_pres_np[np.isnan(v_pres_np)] = 0
        vx_np[np.isnan(vx_np)] = 0
        vy_np[np.isnan(vy_np)] = 0
        v_pres = torch.from_numpy(v_pres_np)
        v_where = torch.stack([torch.from_numpy(vx_np), torch.from_numpy(vy_np)], dim=-1) # [N,4,2]
        is_v_known = torch.from_numpy(is_v_known_np)
    else:
        v_pres = torch.zeros(N, MAX_STEPS)
        v_where= torch.zeros(N, MAX_STEPS,2)
        is_v_known= torch.zeros(N,dtype=torch.bool)

    return spectra, c_vals, is_c_known, v_pres, v_where, is_v_known


def apply_random_mask(is_c_known, is_v_known, frac=MASK_FRAC, seed=RANDOM_SEED):
    """
    We keep 'frac' of labeled data, turn the rest unknown => is_c_known=0, is_v_known=0
    This simulates partial supervision. 
    """
    N = len(is_c_known)
    np.random.seed(seed)
    rand_vals= np.random.rand(N)
    keep_mask= rand_vals < frac
    final_c = is_c_known.numpy() & keep_mask
    final_v = is_v_known.numpy() & keep_mask
    return torch.from_numpy(final_c), torch.from_numpy(final_v)

###############################################################################
# 2) Model / Guide
###############################################################################
class SpectrumDecoder(nn.Module):
    def __init__(self, n_waves=50, hidden_dim=64):
        super().__init__()
        self.vert_embed = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.final = nn.Sequential(
            nn.Linear(hidden_dim+1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_waves)
        )
    def forward(self, c, v_pres, v_where):
        # c: [B,1], v_pres:[B,4], v_where:[B,4,2]
        B, hidden_dim = c.size(0), 64
        accum = torch.zeros(B, hidden_dim, device=c.device)
        for t in range(MAX_STEPS):
            feat = self.vert_embed(v_where[:,t,:])
            pres= v_pres[:,t:t+1]
            accum += feat*pres
        x = torch.cat([accum, c],dim=-1)
        return self.final(x)

class SemiAirModel(nn.Module):
    def __init__(self, n_waves=50):
        super().__init__()
        self.n_waves= n_waves
        self.decoder= SpectrumDecoder(n_waves, 64)

    def model(self, spectrum, 
              c_vals, is_c_known,
              v_pres, v_where, is_v_known):
        """
        c => Normal(0,1) or Delta
        v_pres => Bernoulli(0.5*prev) or Delta
        v_where => Normal(0,1) or Delta
        decode => Normal(spectrum,0.01)
        """
        pyro.module("SemiAirModel", self)
        B = spectrum.size(0)

        with pyro.plate("data", B):
            c0   = torch.zeros(B,1, device=spectrum.device)
            c_std= torch.ones(B,1, device=spectrum.device)
            raw_c= pyro.sample("raw_c",
                               dist.Normal(c0, c_std).to_event(1)) # [B,1]
            # combine
            c_mask= is_c_known.float().unsqueeze(-1)
            c = c_mask*c_vals + (1-c_mask)*raw_c

            # presence / location
            prev_pres = torch.ones(B,1, device=spectrum.device)
            v_pres_collect=[]
            v_where_collect=[]
            for t in range(MAX_STEPS):
                # presence
                name_p = f"raw_pres_{t}"
                p_prob = 0.5*prev_pres
                raw_p  = pyro.sample(name_p,
                                     dist.Bernoulli(p_prob).to_event(1)) # [B,1]
                # if known => Delta
                pres_mask= is_v_known.float().unsqueeze(-1)*(v_pres[:,t]>=0.5).float().unsqueeze(-1)
                pres_val = pres_mask*v_pres[:,t:t+1] + (1-pres_mask)*raw_p

                # location
                name_w= f"raw_where_{t}"
                loc0= torch.zeros(B,2, device=spectrum.device)
                sc0= torch.ones(B,2, device=spectrum.device)
                raw_w= pyro.sample(name_w,
                                   dist.Normal(loc0, sc0).mask(raw_p).to_event(1)) # [B,2]
                where_mask= is_v_known.float().unsqueeze(-1)*pres_val
                w_val= where_mask*v_where[:,t,:] + (1-where_mask)*raw_w

                v_pres_collect.append(pres_val)
                v_where_collect.append(w_val)
                prev_pres= pres_val

            # stack
            v_pres_cat= torch.cat(v_pres_collect, dim=1)   # [B,4]
            v_where_cat= torch.stack(v_where_collect,dim=1)# [B,4,2]

            mean_sp= self.decoder(c, v_pres_cat, v_where_cat)
            pyro.sample("obs_spectrum",
                        dist.Normal(mean_sp, 0.01).to_event(1),
                        obs=spectrum)

    def guide(self, spectrum,
              c_vals, is_c_known,
              v_pres, v_where, is_v_known):
        pyro.module("SemiAirModel", self)
        B = spectrum.size(0)

        # param net for c
        if not hasattr(self,"enc_c"):
            self.enc_c = nn.Sequential(
                nn.Linear(self.n_waves,64),
                nn.ReLU(),
                nn.Linear(64,2)
            )

        with pyro.plate("data", B):
            out_c= self.enc_c(spectrum) # [B,2]
            c_loc= out_c[:,0:1]
            c_scale= F.softplus(out_c[:,1:2])+1e-4
            raw_c= pyro.sample("raw_c",
                               dist.Normal(c_loc, c_scale).to_event(1))

            # presence/where
            prev_pres = torch.ones(B,1, device=spectrum.device)
            for t in range(MAX_STEPS):
                # presence
                if not hasattr(self, f"pres_net_{t}"):
                    setattr(self, f"pres_net_{t}",
                            nn.Sequential(
                                nn.Linear(self.n_waves,64),
                                nn.ReLU(),
                                nn.Linear(64,1)
                            ))
                pres_net= getattr(self, f"pres_net_{t}")
                p_logit= pres_net(spectrum)
                p_prob= torch.sigmoid(p_logit)*prev_pres

                raw_p = pyro.sample(f"raw_pres_{t}",
                                    dist.Bernoulli(p_prob).to_event(1))

                # location
                if not hasattr(self, f"where_net_{t}"):
                    setattr(self, f"where_net_{t}",
                            nn.Sequential(
                                nn.Linear(self.n_waves,4)
                            ))
                where_net= getattr(self, f"where_net_{t}")
                w_out= where_net(spectrum) # [B,4] => loc(2), scale(2)
                w_loc= w_out[:,0:2]
                w_scale= F.softplus(w_out[:,2:4])+1e-4

                raw_w= pyro.sample(f"raw_where_{t}",
                                   dist.Normal(w_loc, w_scale).mask(raw_p).to_event(1))

                prev_pres= raw_p


###############################################################################
# 3) EVALUATION
###############################################################################
@torch.no_grad()
def evaluate(net, 
             spectrum, 
             c_vals, is_c_known,
             v_pres, v_where, is_v_known):
    """
    We measure:
     1) Spectrum->(c, v): sample from guide, compare c, v with known => c/v MSE
     2) (c, v)->Spectrum: decode known c, v => compare with real spectrum => spectrum MSE
    Returns a dict
    """
    B = spectrum.size(0)
    device= spectrum.device

    # 1) guide trace => c, v
    from pyro.poutine import trace
    guide_tr = trace(net.guide).get_trace(
        spectrum, c_vals, is_c_known, v_pres, v_where, is_v_known
    )

    # parse out raw_c, raw_pres_{t}, raw_where_{t}
    # but we also do the final c, pres, where => as in model we do mask
    # For simplicity, let's replicate the logic here.

    c_ex = torch.zeros(B,1, device=device)
    # in the guide, we only have "raw_c" => shape [B,1]
    raw_c_val= guide_tr.nodes["raw_c"]["value"]
    # combine with mask
    c_mask= is_c_known.float().unsqueeze(-1)
    c_final= c_mask*c_vals + (1-c_mask)*raw_c_val

    # presence, location
    pres_list= []
    where_list=[]
    prev_pres_est= torch.ones(B,1, device=device)
    for t in range(MAX_STEPS):
        # presence
        raw_name= f"raw_pres_{t}"
        raw_p_val= guide_tr.nodes[raw_name]["value"] # shape [B,1]
        # combine with known
        pres_mask= is_v_known.float().unsqueeze(-1)*(v_pres[:,t]>=0.5).float().unsqueeze(-1)
        pres_val = pres_mask*v_pres[:,t:t+1] + (1-pres_mask)*raw_p_val

        # location
        raw_w_name= f"raw_where_{t}"
        raw_w_val= guide_tr.nodes[raw_w_name]["value"] # shape [B,2]
        where_mask= is_v_known.float().unsqueeze(-1)*pres_val
        wh_val= where_mask*v_where[:,t,:] + (1-where_mask)*raw_w_val

        pres_list.append(pres_val)
        where_list.append(wh_val)
        prev_pres_est= pres_val

    pres_cat= torch.cat(pres_list, dim=1) # [B,4]
    where_cat= torch.stack(where_list, dim=1) # [B,4,2]

    # measure c MSE (only for is_c_known=1)
    c_err= (c_final - c_vals).pow(2)
    c_err[~is_c_known.unsqueeze(-1)] = 0
    c_count= is_c_known.sum().item()
    c_mse= float(c_err.sum().item()/(c_count+1e-9))

    # measure presence, location MSE for known
    # presence => compare pres_cat, v_pres
    # location => only if v_pres>0.5
    v_err= 0.0
    v_count= 0
    for i in range(B):
        if is_v_known[i]:
            for t in range(MAX_STEPS):
                p_gt= float(v_pres[i,t]>=0.5)
                p_est=float(pres_cat[i,t])
                v_err+= (p_gt - p_est)**2
                v_count+= 1
                if p_gt>0.5:  # location
                    wh_gt= v_where[i,t]
                    wh_es= where_cat[i,t]
                    v_err+= float(torch.sum((wh_gt - wh_es)**2))
                    v_count+= 2  # x,y dims
    v_mse= v_err/(v_count+1e-9)

    # 2) decode => spectrum MSE with known c, v
    # in model => c= c_vals if known else prior, but let's just do full c, v from data
    # Because you say fully labeled, so let's do that:
    # Actually let's do partial: c if known else 0, v if known else pres=0
    c_mask= is_c_known.float().unsqueeze(-1)
    c_true= c_mask*c_vals # if not known => 0
    pres_true= torch.zeros_like(pres_cat)
    where_true= torch.zeros_like(where_cat)
    for i in range(B):
        if is_v_known[i]:
            pres_true[i] = v_pres[i]
            where_true[i] = v_where[i]

    # decode => predicted spectrum
    pred_spectrum= net.decoder(c_true, pres_true, where_true)
    # measure MSE
    spec_mse= float(F.mse_loss(pred_spectrum, spectrum).item())

    return {"c_mse": c_mse, "v_mse": v_mse, "spec_mse": spec_mse}

###############################################################################
# 4) Training Loop
###############################################################################
def train(csv_path, num_epochs=5, batch_size=4096, lr=1e-3):
    # 1) load
    spectra, c_vals, is_c_known, v_pres, v_where, is_v_known = load_data(csv_path)
    N, n_waves= spectra.shape

    # 2) random partial labeling
    # If fully labeled, skip. But we do it here as example
    final_c, final_v = apply_random_mask(is_c_known, is_v_known, frac=MASK_FRAC, seed=RANDOM_SEED)

    net= SemiAirModel(n_waves)
    pyro.clear_param_store()
    optimizer= optim.Adam({"lr": lr})
    svi= SVI(net.model, net.guide, optimizer, loss=TraceGraph_ELBO())

    idx_all= np.arange(N)

    for epoch in range(num_epochs):
        np.random.shuffle(idx_all)
        total_loss=0.0
        for start_i in range(0,N,batch_size):
            end_i= start_i+batch_size
            sub_idx= idx_all[start_i:end_i]
            spec_b= spectra[sub_idx]
            c_b   = c_vals[sub_idx]
            ic_b  = final_c[sub_idx]
            vp_b  = v_pres[sub_idx]
            vw_b  = v_where[sub_idx]
            iv_b  = final_v[sub_idx]

            loss= svi.step(spec_b, c_b, ic_b, vp_b, vw_b, iv_b)
            total_loss+= loss
        avg_loss= total_loss/N

        # Evaluate
        eval_dict= evaluate(net, spectra, c_vals, is_c_known, v_pres, v_where, is_v_known)
        print(f"[Epoch {epoch+1}/{num_epochs}] ELBO={avg_loss:.2f} | "
              f"cMSE={eval_dict['c_mse']:.4f} | vMSE={eval_dict['v_mse']:.4f} | specMSE={eval_dict['spec_mse']:.4f}")

        # Save checkpoint
        torch.save(pyro.get_param_store().get_state(), os.path.join(SAVE_DIR, f"ckpt_epoch{epoch+1}.pt"))
        print(f"  [Saved checkpoint: ckpt_epoch{epoch+1}.pt]")

    print("Training Completed.")


###############################################################################
# 5) Main
###############################################################################
if __name__=="__main__":
    import sys
    csv_path="merged_s4_shapes.csv"
    if len(sys.argv)>1:
        csv_path= sys.argv[1]
    print(f"Using CSV={csv_path}")
    train(csv_path, num_epochs=5, batch_size=4096, lr=1e-3)
