#!/usr/bin/env python3

"""
AIR-style Metasurface Inverse Design
------------------------------------
An illustrative semi-supervised SVI model in Pyro that:
 - Inputs: R spectrum
 - Latent: c, (v_pres, v_where) repeated up to MAX_STEPS
 - Decoder: predict reconstructed R spectrum
 - We use an LSTM to handle the vertex presence and location at each step.
 - We attach a data-dependent baseline to reduce variance for discrete presence variables.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, TraceGraph_ELBO
import pandas as pd
import numpy as np
from tqdm import tqdm, trange

###############################################################################
# 1) Data Loading
###############################################################################

def parse_vertices_str(vertices_str):
    """
    Given a line like 'x1,y1;x2,y2;...', parse into a Python list of (x,y).
    We filter only points in the first quadrant (x>=0, y>0).
    """
    pts = []
    if not isinstance(vertices_str, str):
        return pts
    for token in vertices_str.strip().split(';'):
        if not token:
            continue
        xy = token.split(',')
        if len(xy) == 2:
            x, y = float(xy[0]), float(xy[1])
            if x >= 0 and y > 0:
                pts.append([x, y])
    return pts

def load_metasurface_csv(csv_path):
    """
    Expects columns:
      - 'c' (float)
      - Reflection columns 'R@1.040', 'R@1.054', ...
      - 'vertices_str'
    Returns:
      spectra: tensor [N, n_wavelengths]
      c_vals: tensor [N, 1]
      vertices_list: list of length N, each is a variable-length list of (x,y)
    """
    df = pd.read_csv(csv_path)
    # Grab reflection columns
    r_cols = [c for c in df.columns if c.startswith("R@")]
    r_cols = sorted(r_cols, key=lambda x: float(x.split('@')[1]))
    spectra_np = df[r_cols].values.astype(np.float32)  # shape [N, #waves]
    c_np = df['c'].values.reshape(-1, 1).astype(np.float32)
    vertices_list = [parse_vertices_str(vs) for vs in df['vertices_str'].values]

    spectra = torch.from_numpy(spectra_np)
    c_vals = torch.from_numpy(c_np)
    return spectra, c_vals, vertices_list

###############################################################################
# 2) Networks
###############################################################################

MAX_STEPS = 4  # max # of vertices we allow in the "geometric" sense

class VertexEncoderLSTM(nn.Module):
    """
    This LSTM (used in the guide) will produce distribution parameters
    for each step's vertex presence and location.
    """
    def __init__(self, input_dim=128, hidden_dim=128):
        super().__init__()
        self.rnn = nn.LSTMCell(input_dim, hidden_dim)
        # For presence
        self.pres_head = nn.Linear(hidden_dim, 1)
        # For where
        self.where_loc_head = nn.Linear(hidden_dim, 2)
        self.where_scale_head = nn.Linear(hidden_dim, 2)

    def forward(self, x_embed, h_prev, c_prev):
        """
        x_embed: [batch_size, input_dim] input
        (h_prev, c_prev): hidden, cell states from previous step
        Returns: (h, c), pres_probs, where_loc, where_scale
        """
        h, c = self.rnn(x_embed, (h_prev, c_prev))
        pres_logits = self.pres_head(h)  # shape [batch_size, 1]
        pres_probs = torch.sigmoid(pres_logits)

        where_loc = self.where_loc_head(h)       # [batch_size, 2]
        where_scale = F.softplus(self.where_scale_head(h)) + 1e-4
        return (h, c), pres_probs, where_loc, where_scale


class SpectrumDecoderNN(nn.Module):
    """
    Given the latent info (c, the set of vertices), produce the mean of R-spectrum.
    For simplicity, we embed each vertex location in a small MLP, sum them, also incorporate c.
    """
    def __init__(self, n_wavelengths, hidden_dim=128):
        super().__init__()
        # We'll embed each vertex's (x,y) into a small feature
        self.vert_embed = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        # Combine c and aggregated vertex features to produce R
        self.final_mlp = nn.Sequential(
            nn.Linear(16 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_wavelengths)
        )

    def forward(self, c, v_pres_list, v_where_list):
        """
        c: [batch_size, 1]
        v_pres_list: list of length MAX_STEPS, each a [batch_size, 1] presence
        v_where_list: list of length MAX_STEPS, each a [batch_size, 2] location
        We embed each vertex, multiply by presence, sum, then pass with c into final MLP.
        Returns mean_R: [batch_size, n_wavelengths]
        """
        batch_size = c.shape[0]
        accum = torch.zeros(batch_size, 16, device=c.device)
        for pres, wh in zip(v_pres_list, v_where_list):
            # embed wh
            feat = self.vert_embed(wh)  # [batch, 16]
            accum = accum + pres * feat
        # cat c
        cat_in = torch.cat([accum, c], dim=-1)
        mean_r = self.final_mlp(cat_in)
        return mean_r


class SpectrumEncoderNN(nn.Module):
    """
    A small MLP that encodes the input R spectrum into a latent embedding.
    Used in the guide to help produce hidden states for the LSTM that
    infers presence+where at each step.
    """
    def __init__(self, n_wavelengths, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_wavelengths, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
    def forward(self, spec):
        return self.net(spec)


class CBaselineLSTM(nn.Module):
    """
    A baseline network that helps reduce variance for the presence Bernoulli discrete variables.
    We'll feed:
      (encoded_spectrum, step_id, etc.)
    and produce a single scalar baseline.
    """
    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, embed, h_prev, c_prev):
        """
        embed: [batch, input_dim]
        h_prev, c_prev: [batch, hidden_dim]
        returns baseline_value: [batch, 1]
        """
        h, c = self.lstm(embed, (h_prev, c_prev))
        val = self.head(h)
        return (h, c), val


###############################################################################
# 3) Model + Guide
###############################################################################
class AIRMetasurfaceModel(nn.Module):
    """
    We'll put everything in one class for clarity.
    """
    def __init__(self, n_wavelengths=60, hidden_dim=128, max_steps=MAX_STEPS):
        super().__init__()
        self.n_wavelengths = n_wavelengths
        self.max_steps = max_steps

        # Decoder (model) that goes from latent -> predicted spectrum
        self.decoder = SpectrumDecoderNN(n_wavelengths, hidden_dim)

        # We'll define prior loc/scale for c ~ Normal(0,1)
        # We'll define prior for presence ~ Bernoulli(prob), e.g. prob=0.9 * previous
        # We'll define prior for v_where ~ Normal(0,1) (2D)

    def model(self, spectrum, mask_c=None):
        """
        `spectrum`: [batch_size, n_wavelengths] 
        `mask_c`: if you want to treat c as observed (semi-supervised),
                  pass a Tensor of shape [batch_size,1].  If None, c is latent.
        """
        pyro.module("air_metasurface_model", self)

        batch_size = spectrum.shape[0]

        # Plate for data
        with pyro.plate("data_plate", batch_size):
            # Sample c
            if mask_c is None:
                c = pyro.sample("c", dist.Normal(torch.zeros(batch_size,1), 
                                                 torch.ones(batch_size,1)).to_event(1))
            else:
                # Observed c
                c = pyro.sample("c", dist.Delta(mask_c).to_event(1))

            # We'll keep track of a presence indicator from the previous step
            prev_pres = torch.ones(batch_size,1, device=spectrum.device)

            v_pres_list = []
            v_where_list = []

            for t in range(self.max_steps):
                # Sample presence
                # geometry approach: p = 0.5 for each step, but multiplied by prev_pres
                # (so once we fail, subsequent steps are "off")
                p_pres = 0.5 * prev_pres
                v_pres = pyro.sample(f"v_pres_{t}",
                                     dist.Bernoulli(p_pres)
                                         .to_event(1))

                # Sample the vertex location
                # We mask by v_pres so that if v_pres=0, we zero out log prob
                loc0 = torch.zeros(batch_size, 2, device=spectrum.device)
                scale0 = torch.ones(batch_size, 2, device=spectrum.device)
                v_where = pyro.sample(f"v_where_{t}",
                                      dist.Normal(loc0, scale0)
                                          .mask(v_pres)
                                          .to_event(1))

                v_pres_list.append(v_pres)
                v_where_list.append(v_where)
                prev_pres = v_pres  # next step depends on the presence

            # Combine c + all vertices -> reconstruct
            mean_r = self.decoder(c, v_pres_list, v_where_list)

            # Observe w.r.t. the real spectrum
            pyro.sample("obs_spectrum",
                        dist.Normal(mean_r, 0.01).to_event(1),
                        obs=spectrum)

    def guide(self, spectrum, mask_c=None):
        """
        Guide network: use an LSTM approach to infer presence + where at each step,
        plus we either treat c as known (Delta) or approximate it with a Normal.
        """
        pyro.module("air_metasurface_model", self)
        batch_size = spectrum.shape[0]

        # Let's define the networks we need here:
        #  - An MLP to produce c_loc, c_scale from input spectrum (if c not observed).
        #  - An LSTM that unrolls for steps, each step producing presence + where.
        #  - A baseline LSTM for presence.

        # For simplicity, define them as global Pyro params or small modules:
        # But a single-coded approach is often easier with separate classes.

        # 1) c ~ q(c|spectrum)
        if mask_c is None:
            c_loc_param = pyro.param("c_loc_param", torch.zeros((1,)), constraint=dist.constraints.real)
            c_log_scale_param = pyro.param("c_log_scale_param", torch.zeros((1,)))
        # We'll also do an encoder for the entire spectrum
        # so that c can depend on the actual input.
        # This can be more expressive than a single param.
        # Let's do that:
        if not hasattr(self, "encoder_c"):
            self.encoder_c = nn.Sequential(
                nn.Linear(self.n_wavelengths, 64),
                nn.ReLU(),
                nn.Linear(64, 2)  # outputs (loc, log_scale)
            )

        # 2) LSTM for presence + where
        if not hasattr(self, "lstm_guide"):
            self.lstm_guide = VertexEncoderLSTM(input_dim=128, hidden_dim=128)

        if not hasattr(self, "baseline_lstm"):
            self.baseline_lstm = CBaselineLSTM(input_dim=128, hidden_dim=64)

        # 3) Another encoder for the spectrum -> hidden embedding
        if not hasattr(self, "spectrum_encoder"):
            self.spectrum_encoder = SpectrumEncoderNN(self.n_wavelengths, 128)

        pyro.module("encoder_c", self.encoder_c)
        pyro.module("lstm_guide", self.lstm_guide)
        pyro.module("baseline_lstm", self.baseline_lstm)
        pyro.module("spectrum_encoder", self.spectrum_encoder)

        with pyro.plate("data_plate", batch_size):
            # Encode the input spectrum
            spec_embed = self.spectrum_encoder(spectrum)  # [B, 128]

            # c
            if mask_c is None:
                # param from net:
                c_params = self.encoder_c(spectrum)  # shape [B, 2] => loc, log_scale
                c_loc, c_log_scale = c_params[:, 0:1], c_params[:, 1:2]
                c_scale = F.softplus(c_log_scale) + 1e-4
                pyro.sample("c", dist.Normal(c_loc, c_scale).to_event(1))
            else:
                pyro.sample("c", dist.Delta(mask_c).to_event(1))

            # LSTM hidden for presence + where
            h = torch.zeros(batch_size, 128, device=spectrum.device)
            c_ = torch.zeros(batch_size, 128, device=spectrum.device)
            # LSTM hidden for baseline
            bl_h = torch.zeros(batch_size, 64, device=spectrum.device)
            bl_c = torch.zeros(batch_size, 64, device=spectrum.device)

            # We'll maintain the "active" presence from previous step
            prev_pres = torch.ones(batch_size,1, device=spectrum.device)

            for t in range(self.max_steps):
                # At each step, we combine the spec_embed + maybe the previous step info
                # A simpler approach: just feed spec_embed, ignoring step info for brevity
                # In a real design, you might cat with prev presence + location, etc.
                rnn_input = spec_embed  # [B, 128]

                (h, c_), pres_probs, w_loc, w_scale = self.lstm_guide(rnn_input, h, c_)

                # Baseline net
                (bl_h, bl_c), baseline_val = self.baseline_lstm(rnn_input, bl_h, bl_c)
                # Multiply baseline by prev_pres so once it's "off," we skip
                baseline_val = baseline_val * prev_pres  # shape [B, 1]

                # presence
                pyro.sample(
                    f"v_pres_{t}",
                    dist.Bernoulli(pres_probs * prev_pres).to_event(1),
                    infer={"baseline": {"baseline_value": baseline_val.squeeze(-1)}}
                )

                # location
                pyro.sample(
                    f"v_where_{t}",
                    dist.Normal(w_loc, w_scale)
                        .mask(prev_pres)  # if pres=0, remove log prob
                        .to_event(1)
                )

                # update prev_pres
                # We can't do "prev_pres = sample(...)" in the guide like the model does,
                # because sample() returns actual random draws. We can do a hack:
                # We approximate the *expected presence* or something simpler. We'll do:
                #   E[v_pres] = pres_probs * prev_pres
                # That is a typical "mean-field" style approach so we keep forward consistency.
                prev_pres = pres_probs * prev_pres

###############################################################################
# 4) Training Script
###############################################################################

def train_air(csv_path, num_epochs=10, batch_size=32, lr=1e-3):
    # 1) Load data
    spectra, c_vals, vertices_list = load_metasurface_csv(csv_path)
    N, n_waves = spectra.shape

    # 2) Create model
    model_obj = AIRMetasurfaceModel(n_wavelengths=n_waves, hidden_dim=128, max_steps=MAX_STEPS)

    # 3) Setup SVI
    pyro.clear_param_store()
    optimizer = optim.Adam({"lr": lr})
    svi = SVI(model_obj.model, model_obj.guide, optimizer, loss=TraceGraph_ELBO())

    # 4) Training loop
    indices = np.arange(N)
    for epoch in trange(num_epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        for start_i in trange(0, N, batch_size):
            end_i = start_i + batch_size
            idx = indices[start_i:end_i]
            spec_batch = spectra[idx]
            # For demonstration, let's treat c as *observed*, so we pass c_vals (semi-supervised).
            # If you want c as fully latent, pass mask_c=None.
            c_batch = c_vals[idx]

            loss = svi.step(spec_batch, mask_c=c_batch)
            epoch_loss += loss
        
        avg_elbo = epoch_loss / N
        print(f"[epoch {epoch+1}]  avg_elbo: {avg_elbo:.4f}")

    print("Training complete!")

###############################################################################
# 5) Main
###############################################################################
if __name__ == "__main__":
    import sys

    csv_path = "merged_s4_shapes.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    print(f"Using CSV = {csv_path}")
    train_air(csv_path, num_epochs=5, batch_size=4096, lr=1e-3)
