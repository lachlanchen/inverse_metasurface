#!/usr/bin/env python3

"""
Semi-Supervised AIR Metasurface
-------------------------------
Fully vectorized approach with partial labeling for c, v_pres, v_where.

If c or vertex is known (semi-supervised), the model uses Delta(...).
If unknown, it samples from prior. The guide similarly: if known => Delta, else sample from approximate posterior.

We do up to 4 vertices. We do Bernoulli(0.5 * prev_pres) for presence
and Normal(0,1) for location.

To run:
  python semi_supervised_air.py [csv_path]

We assume you have a CSV with columns:
  c (maybe partial known), 
  R@..., 
  v_pres_0..3, v_where_0_x..3, v_where_0_y..3 (maybe partial known).
"""

import os
import math
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceGraph_ELBO
import pyro.optim as optim
from tqdm import trange

MAX_STEPS = 4
SAVE_DIR = "svi_results"
os.makedirs(SAVE_DIR, exist_ok=True)

###############################################################################
# 1) Data Loading
###############################################################################
def load_data(csv_path):
    """
    Returns:
      spectra: [N, n_waves]
      c_vals: [N,1], with unknown=0.0
      is_c_known: bool [N]
      v_pres: [N,4], v_where: [N,4,2], unknown => 0, is_v_known: bool [N]
    """
    df = pd.read_csv(csv_path)
    # reflection columns
    r_cols = [col for col in df.columns if col.startswith("R@")]
    r_cols = sorted(r_cols, key=lambda x: float(x.split('@')[1]))
    spectra_np = df[r_cols].values.astype(np.float32)
    spectra = torch.from_numpy(spectra_np)

    N, n_waves = spectra.shape

    # c
    c_np = np.full((N,1), np.nan, dtype=np.float32)
    if "c" in df.columns:
        c_np = df["c"].values.reshape(-1,1).astype(np.float32)
    is_c_known = ~np.isnan(c_np).reshape(-1)
    c_np[np.isnan(c_np)] = 0.0
    c_vals = torch.from_numpy(c_np)
    is_c_known = torch.from_numpy(is_c_known)

    # v_pres, v_where
    pres_cols = [f"v_pres_{t}" for t in range(MAX_STEPS)]
    where_x_cols= [f"v_where_{t}_x" for t in range(MAX_STEPS)]
    where_y_cols= [f"v_where_{t}_y" for t in range(MAX_STEPS)]
    have_pres = all(pc in df.columns for pc in pres_cols)
    have_where= all(wx in df.columns and wy in df.columns 
                    for wx,wy in zip(where_x_cols, where_y_cols))

    if have_pres and have_where:
        v_pres_np = df[pres_cols].values.astype(np.float32)   # [N,4]
        v_where_x_np= df[where_x_cols].values.astype(np.float32) # [N,4]
        v_where_y_np= df[where_y_cols].values.astype(np.float32) # [N,4]
        nan_pres  = np.isnan(v_pres_np)
        nan_wx    = np.isnan(v_where_x_np)
        nan_wy    = np.isnan(v_where_y_np)
        row_has_nan = np.any(nan_pres, axis=1) | np.any(nan_wx, axis=1) | np.any(nan_wy, axis=1)
        is_v_known_np = ~row_has_nan
        # fill with 0
        v_pres_np[nan_pres] = 0
        v_where_x_np[nan_wx] = 0
        v_where_y_np[nan_wy] = 0
        v_pres = torch.from_numpy(v_pres_np)
        v_where = torch.stack([
            torch.from_numpy(v_where_x_np),
            torch.from_numpy(v_where_y_np)
        ], dim=-1) # [N,4,2]
        is_v_known = torch.from_numpy(is_v_known_np)
    else:
        v_pres = torch.zeros(N, MAX_STEPS)
        v_where= torch.zeros(N, MAX_STEPS, 2)
        is_v_known = torch.zeros(N, dtype=torch.bool)

    return spectra, c_vals, is_c_known, v_pres, v_where, is_v_known

###############################################################################
# 2) Model / Guide
###############################################################################
class SpectrumDecoder(nn.Module):
    """
    Combine c + up to 4 vertices => predict reflection
    c: [B,1]; v_pres: [B,4]; v_where: [B,4,2]
    """
    def __init__(self, n_waves=50, hidden_dim=64):
        super().__init__()
        self.vert_embed = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.final_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_waves)
        )

    def forward(self, c, v_pres, v_where):
        B, hidden_dim = c.size(0), 64
        accum = torch.zeros(B, hidden_dim, device=c.device)
        for t in range(MAX_STEPS):
            feat = self.vert_embed(v_where[:, t, :])  # [B,hidden_dim]
            pres = v_pres[:, t:t+1]                   # [B,1]
            accum += feat*pres
        out = torch.cat([accum, c], dim=-1)
        return self.final_net(out)

class AirSemiModel(nn.Module):
    def __init__(self, n_waves=50):
        super().__init__()
        self.n_waves = n_waves
        self.decoder = SpectrumDecoder(n_waves, 64)

    def model(self, 
              spectrum, 
              c_vals, is_c_known,
              v_pres, v_where, is_v_known):
        """
        c ~ Normal(0,1) or Delta
        v_pres ~ Bernoulli(0.5*prev_pres) or Delta
        v_where ~ Normal(0,1) or Delta
        Then decode => Normal(spectrum,0.01)
        """
        pyro.module("AirSemiModel", self)
        B = spectrum.size(0)

        with pyro.plate("data", B):
            # c
            c0 = torch.zeros(B,1, device=spectrum.device)
            s0 = torch.ones(B,1, device=spectrum.device)

            # If c known => Delta, else Normal(0,1)
            # We'll do a single sample statement for c. We create a "raw_c"
            raw_c = pyro.sample("raw_c", dist.Normal(c0, s0).to_event(1))

            # Then we combine with mask
            mask_c = is_c_known.float().unsqueeze(-1) # [B,1]
            c = mask_c*c_vals + (1-mask_c)*raw_c

            # v
            prev_pres = torch.ones(B,1, device=spectrum.device)
            v_pres_collect = []
            v_where_collect= []

            for t in range(MAX_STEPS):
                name_p = f"raw_pres_{t}"
                name_w = f"raw_where_{t}"

                # raw presence
                p_pres = 0.5*prev_pres
                raw_pres = pyro.sample(name_p,
                                       dist.Bernoulli(p_pres).to_event(1))  # shape [B,1]
                # combine with known
                # if known => v_pres[i,t]
                pres_mask = (is_v_known.float().unsqueeze(-1) * (v_pres[:,t]>=0.5).float().unsqueeze(-1))
                pres_val  = pres_mask*v_pres[:,t:t+1] + (1-pres_mask)*raw_pres

                # location
                loc0   = torch.zeros(B,2, device=spectrum.device)
                scale0 = torch.ones(B,2, device=spectrum.device)
                raw_where= pyro.sample(name_w,
                                       dist.Normal(loc0, scale0).mask(raw_pres).to_event(1)) # shape [B,2]

                # if known => Delta
                where_mask = is_v_known.float().unsqueeze(-1)*pres_val
                wh_val = where_mask*v_where[:,t,:] + (1-where_mask)*raw_where

                v_pres_collect.append(pres_val)
                v_where_collect.append(wh_val)

                prev_pres = pres_val

            # stack
            v_pres_cat = torch.cat(v_pres_collect, dim=1)    # [B,4]
            v_where_cat= torch.stack(v_where_collect, dim=1) # [B,4,2]

            # decode
            mean_spectrum = self.decoder(c, v_pres_cat, v_where_cat)
            pyro.sample("obs_spectrum",
                        dist.Normal(mean_spectrum, 0.01).to_event(1),
                        obs=spectrum)

    def guide(self,
              spectrum, 
              c_vals, is_c_known,
              v_pres, v_where, is_v_known):
        """
        We do same logic, but param:
          raw_c ~ Normal(loc,scale)
          raw_pres ~ Bernoulli(prob)
          raw_where ~ Normal(...).mask(...)
        If known => we do same approach: combine known with raw sample
        """
        pyro.module("AirSemiModel", self)
        B = spectrum.size(0)

        # Let's define param net for c
        # or an MLP from spectrum => (loc, scale)
        if not hasattr(self, "enc_c"):
            self.enc_c = nn.Sequential(
                nn.Linear(self.n_waves, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )

        # param net for presence => we do a small param
        # for location => we do another param net or just direct ?

        with pyro.plate("data", B):
            # c param
            c_out = self.enc_c(spectrum) # [B,2]
            c_loc = c_out[:,0:1]
            c_scale= F.softplus(c_out[:,1:2]) +1e-4

            raw_c = pyro.sample("raw_c", dist.Normal(c_loc, c_scale).to_event(1))

            # presence & location
            prev_pres_est = torch.ones(B,1, device=spectrum.device)

            for t in range(MAX_STEPS):
                # presence
                if not hasattr(self, f"pres_net_{t}"):
                    # each step: param net => prob
                    setattr(self, f"pres_net_{t}",
                            nn.Sequential(
                                nn.Linear(self.n_waves, 64),
                                nn.ReLU(),
                                nn.Linear(64, 1)
                            ))
                pres_net = getattr(self, f"pres_net_{t}")
                p_logits = pres_net(spectrum) # [B,1]
                p_probs  = torch.sigmoid(p_logits)*prev_pres_est

                raw_pres= pyro.sample(f"raw_pres_{t}",
                                      dist.Bernoulli(p_probs).to_event(1))
                # combine known
                pres_mask= (is_v_known.float().unsqueeze(-1)*(v_pres[:,t]>=0.5).float().unsqueeze(-1))
                pres_val = pres_mask*v_pres[:,t:t+1] + (1-pres_mask)*raw_pres

                # location
                if not hasattr(self, f"where_net_{t}"):
                    setattr(self, f"where_net_{t}",
                            nn.Sequential(
                                nn.Linear(self.n_waves, 4), # => loc(2), scale(2)
                            ))
                where_net = getattr(self, f"where_net_{t}")
                w_out = where_net(spectrum) # [B,4]
                w_loc = w_out[:,0:2]
                w_scale= F.softplus(w_out[:,2:4]) +1e-4

                raw_where = pyro.sample(f"raw_where_{t}",
                                        dist.Normal(w_loc, w_scale).mask(raw_pres).to_event(1))

                where_mask = is_v_known.float().unsqueeze(-1)*pres_val
                wh_val = where_mask*v_where[:,t,:] + (1-where_mask)*raw_where

                prev_pres_est = pres_val

###############################################################################
# 3) Training
###############################################################################
def run_training(csv_path, num_epochs=5, batch_size=256, lr=1e-3):
    # 1) load
    spectra, c_vals, is_c_known, v_pres, v_where, is_v_known = load_data(csv_path)
    N, n_waves = spectra.shape

    net = AirSemiModel(n_waves)
    pyro.clear_param_store()
    opt = optim.Adam({"lr": lr})
    svi = SVI(net.model, net.guide, opt, loss=TraceGraph_ELBO())

    idx_all = np.arange(N)
    for epoch in range(num_epochs):
        np.random.shuffle(idx_all)
        total_loss=0.0
        for start_i in range(0,N,batch_size):
            end_i = start_i+batch_size
            sub_idx= idx_all[start_i:end_i]
            spec_b = spectra[sub_idx]
            c_b    = c_vals[sub_idx]
            ic_b   = is_c_known[sub_idx]
            vp_b   = v_pres[sub_idx]
            vw_b   = v_where[sub_idx]
            iv_b   = is_v_known[sub_idx]

            loss = svi.step(spec_b, c_b, ic_b, vp_b, vw_b, iv_b)
            total_loss+=loss
        avg_loss = total_loss/N
        print(f"Epoch {epoch+1}/{num_epochs}, ELBO={avg_loss:.4f}")
        # Save
        torch.save(pyro.get_param_store().get_state(), os.path.join(SAVE_DIR,f"ckpt_epoch{epoch+1}.pt"))

    print("Training Done.")

###############################################################################
# 4) Main
###############################################################################
if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        csvp = sys.argv[1]
    else:
        csvp = "merged_s4_shapes.csv"
    run_training(csvp, num_epochs=5, batch_size=4096, lr=1e-3)
