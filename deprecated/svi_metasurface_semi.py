#!/usr/bin/env python3

"""
AIR-style Inverse Metasurface Model with up to 4 vertices and semi-supervision.

We assume a CSV with columns:
 - c (float in [0,1], partial or fully known)
 - R@... reflection columns
 - Possibly ground-truth vertex presence & location data, if available.
   For instance, we store them as:
     v_pres_0, v_pres_1, v_pres_2, v_pres_3
     v_where_0_x, v_where_0_y, ...
   If not present or if blank, we treat them as unknown.

We then do a pyro Model/Guide:
 - c ~ Beta(...) or Delta(observed_c)
 - v_pres_t ~ Bernoulli(...) or Delta if observed
 - v_where_t ~ Normal(...) or Delta if observed
Then decode into the reflection spectrum. We use a Normal likelihood.

We also demonstrate an AIR-like LSTM approach in the guide, with a data-dependent baseline for
the discrete Bernoulli variables to reduce gradient variance.

At the bottom is a Lua snippet for generating a C4-symmetric polygon if you want to *simulate*
some training data. That code is not invoked by Python but offered as reference for data creation.

Usage:
  pip install torch pyro-ppl tqdm pandas numpy
  python air_metasurface_semi.py

You can customize the `MAX_STEPS=4` approach, or adapt further as needed.
"""

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


###############################################################################
# -- 0) Example Lua Snippet for generating C4-symmetric polygons (reference) --
###############################################################################
lua_snippet = r"""
--------------------------------------------------------------------------------
-- This snippet shows how you might generate a C4-symmetric polygon in 2D.
-- We'll produce N vertices, where N is multiple of 4. We generate the radius
-- for the first quadrant, then replicate for other quadrants, ensuring
-- symmetrical shape. Finally, we rotate by some random angle in [0..π/2].
--------------------------------------------------------------------------------

function generate_c4_polygon(N, base_radius, rand_amt)
    if (N % 4) ~= 0 then
        error("generate_c4_polygon: N must be divisible by 4 for perfect C4 symmetry.")
    end

    local verts = {}
    local two_pi = 2 * math.pi
    local quarter = N / 4

    -- Generate radius array for quadrant 1 only
    local radii = {}
    for i=1, quarter do
        -- random radius around base_radius ± rand_amt
        local r = base_radius + rand_amt * (2*math.random() - 1)
        table.insert(radii, r)
    end

    -- We'll accumulate angles from 0..2π in steps of 2π/N
    for i=0, N-1 do
        local angle = i * (two_pi / N)
        local idx = (i % quarter) + 1
        local r   = radii[idx]
        local x   = r * math.cos(angle)
        local y   = r * math.sin(angle)
        table.insert(verts, x)
        table.insert(verts, y)
    end

    -- Rotate entire polygon by a random angle in [0..π/2]
    local angle_offset = math.random() * (math.pi / 2)
    local cosA = math.cos(angle_offset)
    local sinA = math.sin(angle_offset)
    for i=1, #verts, 2 do
        local xx = verts[i]
        local yy = verts[i+1]
        local rx = xx*cosA - yy*sinA
        local ry = xx*sinA + yy*cosA
        verts[i]   = rx
        verts[i+1] = ry
    end

    return verts
end
"""

###############################################################################
# -- 1) Data Loading
###############################################################################

MAX_STEPS = 4  # at most 4 vertices

def load_data(csv_path):
    """
    This function loads:
      - c in [0,1], can be partially known or unknown (if empty or NaN).
      - R@... columns for reflection.
      - Possibly v_pres_0..3, v_where_0_x, v_where_0_y, etc. if provided.

    Returns:
      spectra: [N, n_waves]
      c_vals: [N, 1] or None if unknown
      v_pres: [N, MAX_STEPS] or None if unknown
      v_where: [N, MAX_STEPS, 2] or None
      is_c_known: bool mask
      is_v_known: bool mask
    """
    df = pd.read_csv(csv_path)

    # reflection columns
    r_cols = [col for col in df.columns if col.startswith("R@")]
    r_cols = sorted(r_cols, key=lambda x: float(x.split('@')[1]))  # sort by wavelength numeric
    spectra_np = df[r_cols].values.astype(np.float32)
    spectra = torch.from_numpy(spectra_np)

    # c in [0,1], or unknown => store as NaN in CSV if unknown
    c_np = np.full((len(df),1), np.nan, dtype=np.float32)
    if "c" in df.columns:
        c_np = df["c"].values.reshape(-1,1).astype(np.float32)
    c_vals = torch.from_numpy(c_np)
    is_c_known = ~torch.isnan(c_vals)  # True if c is not NaN
    c_vals[torch.isnan(c_vals)] = 0.0  # fill unknown with 0 temporarily

    # We also see if we have v_pres_0..3, v_where_0_x, etc.
    # We'll attempt to parse:
    #   v_pres_t => in columns "v_pres_0", "v_pres_1", ...
    #   v_where_t => in columns "v_where_0_x", "v_where_0_y", etc.
    pres_cols = [f"v_pres_{t}" for t in range(MAX_STEPS)]
    where_x_cols = [f"v_where_{t}_x" for t in range(MAX_STEPS)]
    where_y_cols = [f"v_where_{t}_y" for t in range(MAX_STEPS)]
    have_pres = all(c in df.columns for c in pres_cols)
    have_where = all((cx in df.columns and cy in df.columns) for cx,cy in zip(where_x_cols, where_y_cols))

    if have_pres and have_where:
        v_pres_np = df[pres_cols].values.astype(np.float32)  # [N,4]
        v_where_x_np = df[where_x_cols].values.astype(np.float32)  # [N,4]
        v_where_y_np = df[where_y_cols].values.astype(np.float32)  # [N,4]
        # If any entry is NaN => unknown
        is_v_known = ~np.isnan(v_pres_np).any(axis=1) & ~np.isnan(v_where_x_np).any(axis=1) & ~np.isnan(v_where_y_np).any(axis=1)
        # We'll fill any NaN with 0
        v_pres_np[np.isnan(v_pres_np)] = 0
        v_where_x_np[np.isnan(v_where_x_np)] = 0
        v_where_y_np[np.isnan(v_where_y_np)] = 0
        v_pres = torch.from_numpy(v_pres_np)
        v_where = torch.stack([torch.from_numpy(v_where_x_np), torch.from_numpy(v_where_y_np)], dim=-1)  # [N,4,2]
        is_v_known = torch.from_numpy(is_v_known)
    else:
        # No columns => unknown for all
        v_pres = None
        v_where = None
        is_v_known = torch.zeros(len(df), dtype=torch.bool)

    return spectra, c_vals, is_c_known, v_pres, v_where, is_v_known


###############################################################################
# -- 2) Model/Guide Modules
###############################################################################

class SpectrumDecoder(nn.Module):
    """
    Combine c + up to 4 vertices => predict reflection spectrum
    We'll do a small MLP: each present vertex is embedded, summed, appended c => final MLP => spectrum
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

    def forward(self, c, v_pres_list, v_where_list):
        """
        c: [B,1]
        v_pres_list: list of length 4, each [B,1]
        v_where_list: list of length 4, each [B,2]
        Return: mean_spectrum [B, n_waves]
        """
        B = c.shape[0]
        hidden_dim = 64
        accum = torch.zeros(B, hidden_dim, device=c.device)
        for pres, w in zip(v_pres_list, v_where_list):
            e = self.vert_embed(w)  # [B, hidden_dim]
            accum = accum + pres * e
        combined = torch.cat([accum, c], dim=-1)  # [B, hidden_dim+1]
        out = self.final_net(combined)            # [B, n_waves]
        return out


class GuideRNN(nn.Module):
    """
    LSTM that outputs presence(prob), location(mean+scale) for each step.
    We'll do 4 steps unrolled.
    """
    def __init__(self, input_dim=64, hidden_dim=64):
        super().__init__()
        self.rnn = nn.LSTMCell(input_dim, hidden_dim)
        self.pres_head = nn.Linear(hidden_dim, 1)
        self.where_loc = nn.Linear(hidden_dim, 2)
        self.where_scale = nn.Linear(hidden_dim, 2)

    def forward(self, x_embed, h, c_):
        """
        x_embed: [B, input_dim]
        (h, c_): hidden + cell
        returns: (h_next, c_next), pres_probs, loc, scale
        """
        h_next, c_next = self.rnn(x_embed, (h, c_))
        pres_logits = self.pres_head(h_next)
        pres_probs = torch.sigmoid(pres_logits)
        loc = self.where_loc(h_next)
        scale = F.softplus(self.where_scale(h_next)) + 1e-4
        return (h_next, c_next), pres_probs, loc, scale


class BaselineLSTM(nn.Module):
    """
    For data-dependent baseline for discrete presence variables.
    We'll output a single scalar baseline for each step.
    """
    def __init__(self, input_dim=64, hidden_dim=32):
        super().__init__()
        self.rnn = nn.LSTMCell(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_embed, h, c_):
        """
        returns (h_next, c_next), baseline_value
        """
        h_next, c_next = self.rnn(x_embed, (h, c_))
        val = self.head(h_next)
        return (h_next, c_next), val


###############################################################################
# -- 3) Full Model + Guide
###############################################################################

class AirMetasurface(nn.Module):
    def __init__(self, n_waves=50, hidden_dim=64):
        super().__init__()
        self.n_waves = n_waves
        self.hidden_dim = hidden_dim

        # The decoder
        self.decoder = SpectrumDecoder(n_waves=n_waves, hidden_dim=hidden_dim)

        # For the guide
        self.guide_rnn = GuideRNN(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.baseline_lstm = BaselineLSTM(input_dim=hidden_dim, hidden_dim=32)

        # We'll have an encoder MLP from spectrum -> embedding for the LSTM
        self.encoder_spec = nn.Sequential(
            nn.Linear(n_waves, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # For c, we do a small MLP: c_loc, c_logit for Beta or Delta.
        self.encoder_c = nn.Sequential(
            nn.Linear(n_waves, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def model(
        self, 
        spectrum, 
        c_vals=None,    # [B,1], might be observed or None if unknown
        is_c_known=None,# [B] bool
        v_pres=None,    # [B,4], might be observed or None
        v_where=None,   # [B,4,2], might be observed or None
        is_v_known=None # [B] bool
    ):
        """
        c ~ Beta(2,2) or Delta(c_vals) if known
        presence ~ Bernoulli(...) or Delta
        location ~ Normal(0,1) or Delta
        Then decode => normal likelihood of spectrum
        """
        pyro.module("air_net", self)
        B = spectrum.size(0)

        with pyro.plate("data_plate", B) as idx:
            # If c is known => Delta(c_vals[i]), else Beta(2,2)
            c_prior_a = 2.0
            c_prior_b = 2.0
            c_samples = []
            for i in range(B):
                name_c = f"c_{idx[i].item()}"  # unique name
                if is_c_known[i]:
                    # observe c
                    pyro.sample(name_c, dist.Delta(c_vals[i:i+1,:]).to_event(1))
                    c_samples.append(c_vals[i:i+1,:])
                else:
                    # sample from Beta
                    s = pyro.sample(name_c,
                                    dist.Beta(torch.tensor([c_prior_a]), torch.tensor([c_prior_b]))
                                        .to_event(1))
                    c_samples.append(s)
            c_cat = torch.cat(c_samples, dim=0)  # [B,1]

            # We'll do up to 4 steps
            prev_pres = torch.ones(B,1, device=spectrum.device)
            v_pres_list = []
            v_where_list = []

            for t in range(MAX_STEPS):
                pres_list = []
                where_list = []
                for i in range(B):
                    name_pres = f"v_pres_{t}_{idx[i].item()}"
                    name_where= f"v_where_{t}_{idx[i].item()}"
                    # if v is known for this sample, we do Delta
                    # else sample from Bernoulli/Normal
                    if is_v_known[i]:
                        # we have ground truth for v_pres[i,t], v_where[i,t]
                        pres_val = v_pres[i, t].reshape(1,1)
                        pyro.sample(name_pres, dist.Delta(pres_val).to_event(1))
                        where_val = v_where[i, t].reshape(1,2)
                        # mask by pres_val
                        pyro.sample(name_where,
                                    dist.Delta(where_val)
                                        .mask(pres_val)
                                        .to_event(1))
                        pres_list.append(pres_val)
                        where_list.append(where_val)
                    else:
                        # presence
                        p_pres = 0.8 * prev_pres[i:i+1,:]  # or 0.5 or 0.8
                        s_pres = pyro.sample(name_pres,
                                             dist.Bernoulli(p_pres).to_event(1))
                        # location
                        loc0 = torch.zeros(1,2, device=spectrum.device)
                        scale0 = torch.ones(1,2, device=spectrum.device)
                        s_where = pyro.sample(name_where,
                                              dist.Normal(loc0, scale0)
                                                  .mask(s_pres)
                                                  .to_event(1))
                        pres_list.append(s_pres)
                        where_list.append(s_where)

                # stack
                pres_t = torch.cat(pres_list, dim=0)
                where_t= torch.cat(where_list, dim=0)
                v_pres_list.append(pres_t)
                v_where_list.append(where_t)
                prev_pres = pres_t  # next step

            # decode
            mean_spectrum = self.decoder(c_cat, v_pres_list, v_where_list)
            pyro.sample("obs_spectrum",
                        dist.Normal(mean_spectrum, 0.01).to_event(1),
                        obs=spectrum)

    def guide(
        self, 
        spectrum, 
        c_vals=None,
        is_c_known=None,
        v_pres=None,
        v_where=None,
        is_v_known=None
    ):
        """
        We'll do:
         - if c known => Delta
         - else use an MLP to produce c's alpha,beta, or param => sample from Beta
         - presence & location => LSTM approach w/ data-dependent baseline
        """
        pyro.module("air_net", self)
        B = spectrum.size(0)

        # Encode spectrum
        spec_embed = self.encoder_spec(spectrum)  # [B, hidden_dim]

        with pyro.plate("data_plate", B) as idx:
            # c
            c_out = self.encoder_c(spectrum)  # [B,2]
            c_alpha_logits = c_out[:,0:1]
            c_beta_logits  = c_out[:,1:2]
            # For numeric stability, we do alpha = softplus(...)+1, etc.
            alpha = F.softplus(c_alpha_logits) + 1
            beta  = F.softplus(c_beta_logits)  + 1

            # sample c if unknown
            for i in range(B):
                name_c = f"c_{idx[i].item()}"
                if is_c_known[i]:
                    pyro.sample(name_c, dist.Delta(c_vals[i:i+1,:]).to_event(1))
                else:
                    pyro.sample(name_c, dist.Beta(alpha[i:i+1,:], beta[i:i+1,:]).to_event(1))

            # LSTM hidden
            h = torch.zeros(B, self.hidden_dim, device=spectrum.device)
            c_ = torch.zeros(B, self.hidden_dim, device=spectrum.device)
            # baseline hidden
            bl_h = torch.zeros(B, 32, device=spectrum.device)
            bl_c = torch.zeros(B, 32, device=spectrum.device)

            # We do 4 steps
            prev_pres_approx = torch.ones(B,1, device=spectrum.device)
            for t in range(MAX_STEPS):
                # We feed spec_embed as input. (Optionally cat with t or prev pres).
                (h, c_), pres_probs, loc, scale = self.guide_rnn(spec_embed, h, c_)

                # baseline
                (bl_h, bl_c), baseline_val = self.baseline_lstm(spec_embed, bl_h, bl_c)
                baseline_val = baseline_val * prev_pres_approx  # once off, no need to refine

                for i in range(B):
                    name_pres = f"v_pres_{t}_{idx[i].item()}"
                    name_where= f"v_where_{t}_{idx[i].item()}"

                    if is_v_known[i]:
                        # known => Delta
                        pres_val = v_pres[i, t].reshape(1,1)
                        pyro.sample(name_pres, dist.Delta(pres_val).to_event(1))
                        # location
                        where_val= v_where[i, t].reshape(1,2)
                        pyro.sample(name_where,
                                    dist.Delta(where_val)
                                        .mask(pres_val)
                                        .to_event(1))
                        # in a real approach, might skip updating LSTM state. We'll approximate
                    else:
                        # presence
                        # multiply by prev pres
                        p_pres = pres_probs[i:i+1,:] * prev_pres_approx[i:i+1,:]
                        pyro.sample(name_pres,
                                    dist.Bernoulli(p_pres).to_event(1),
                                    infer={"baseline": {"baseline_value": baseline_val[i:i+1].squeeze(-1)}})
                        # location
                        pyro.sample(name_where,
                                    dist.Normal(loc[i:i+1,:], scale[i:i+1,:])
                                        .mask(p_pres)
                                        .to_event(1))
                # update prev_pres_approx
                prev_pres_approx = pres_probs * prev_pres_approx


###############################################################################
# -- 4) Training
###############################################################################

def run_training(csv_path, num_epochs=5, batch_size=32, lr=1e-3):
    # 1) Load data
    spectra, c_vals, is_c_known, v_pres, v_where, is_v_known = load_data(csv_path)
    N, n_waves = spectra.shape

    # 2) Create model
    net = AirMetasurface(n_waves=n_waves, hidden_dim=64)

    # 3) SVI
    pyro.clear_param_store()
    optimizer = optim.Adam({"lr": lr})
    svi = SVI(net.model, net.guide, optimizer, loss=TraceGraph_ELBO())

    # 4) Training loop
    idx_all = np.arange(N)
    for epoch in trange(num_epochs, desc="Training"):
        np.random.shuffle(idx_all)
        total_loss = 0.0
        for start_i in range(0, N, batch_size):
            end_i = start_i + batch_size
            batch_idx = idx_all[start_i:end_i]
            spec_batch = spectra[batch_idx]
            c_batch    = c_vals[batch_idx]
            ic_batch   = is_c_known[batch_idx]
            if v_pres is not None:
                vp_batch = v_pres[batch_idx]
            else:
                vp_batch = None
            if v_where is not None:
                vw_batch = v_where[batch_idx]
            else:
                vw_batch = None
            iv_batch   = is_v_known[batch_idx]

            loss = svi.step(
                spec_batch,
                c_vals=c_batch,
                is_c_known=ic_batch,
                v_pres=vp_batch,
                v_where=vw_batch,
                is_v_known=iv_batch
            )
            total_loss += loss

        avg_loss = total_loss / N
        print(f"Epoch {epoch+1}/{num_epochs}, ELBO: {avg_loss:.4f}")


###############################################################################
# -- 5) Main Entry
###############################################################################

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "merged_s4_shapes.csv"

    print(f"Using CSV = {csv_path}")
    run_training(csv_path, num_epochs=5, batch_size=32, lr=1e-3)

