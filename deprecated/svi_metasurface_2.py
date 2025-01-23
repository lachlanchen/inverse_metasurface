#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import math

#####################################################################
# 1) Data Loading
#####################################################################

def parse_vertices_str(vertices_str):
    """
    Example function to parse your 'vertices_str' from the CSV.
    Each row is something like: "x1,y1;x2,y2;x3,y3;..."
    
    We only keep those in the first quadrant (x>=0, y>0).
    Or you can keep them all, but let's illustrate a filter.
    Returns a list of 2D points.
    """
    points = []
    if not isinstance(vertices_str, str) or len(vertices_str.strip()) == 0:
        return points
    
    pairs = vertices_str.split(';')
    for p in pairs:
        xy = p.split(',')
        if len(xy) == 2:
            x, y = float(xy[0]), float(xy[1])
            if x >= 0 and y > 0:
                points.append([x, y])
    return points

def load_merged_s4_shapes(csv_path="merged_s4_shapes.csv"):
    """
    Loads the merged CSV. We assume columns:
      - c  (float, crystallization parameter)
      - R@1.040, R@1.054, ..., R@2.502  (reflection values)
      - vertices_str  (string describing polygon vertices)
    plus maybe folder_key, shape_idx, etc. We'll just parse out what we need.
    
    Returns:
      spectra_tensor: shape [N, n_wavelengths]
      c_tensor: shape [N, 1]
      vertices_list: list of length N, each an arbitrary number of 2D points
    """
    df = pd.read_csv(csv_path)
    
    # 1) Gather reflection columns
    #    (You might have columns named R@1.040, T@1.040, etc. We'll focus on R@ only.)
    r_cols = [col for col in df.columns if col.startswith("R@")]
    r_cols = sorted(r_cols, key=lambda x: float(x.split('@')[1]))  # sort by numeric wavelength
    # Make sure we have a consistent set
    spectra = df[r_cols].values.astype(np.float32)
    
    # 2) c-values
    c = df['c'].values.astype(np.float32).reshape(-1, 1)
    
    # 3) vertices
    vertices_strs = df['vertices_str'].values
    vertices_list = [parse_vertices_str(vs) for vs in vertices_strs]
    
    # Convert to torch
    spectra_tensor = torch.from_numpy(spectra)
    c_tensor = torch.from_numpy(c)
    
    return spectra_tensor, c_tensor, vertices_list

#####################################################################
# 2) LSTM-based Vertex Distribution
#####################################################################

class VertexLSTMDecoder(PyroModule):
    """
    Example of a module that can decode a latent embedding into a
    distribution over variable-length polygon vertices (in the first quadrant).
    
    We'll model the polygon as a sequence of up to `max_steps` vertices,
    each step sampling a 'continue flag' from a Bernoulli( p_continue )
    and (x, y) ~ some distribution. We let the LSTM hidden state track
    the progression.
    """
    def __init__(self, latent_dim, hidden_dim=64, max_steps=20):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        
        # We project latent => initial LSTM hidden state
        self.fc_h = PyroModule[nn.Linear](latent_dim, hidden_dim)
        self.fc_c = PyroModule[nn.Linear](latent_dim, hidden_dim)
        
        # The LSTM cell
        self.lstm_cell = PyroModule[nn.LSTMCell](input_size=hidden_dim, hidden_size=hidden_dim)
        
        # Predict (x, y) distribution parameters from hidden
        self.fc_loc = PyroModule[nn.Linear](hidden_dim, 2)   # loc of Normal
        self.fc_scale = PyroModule[nn.Linear](hidden_dim, 2) # log-scale of Normal
        
        # Predict the "continuation" probability from hidden
        self.fc_continue = PyroModule[nn.Linear](hidden_dim, 1)
        
        # For numerical stability
        self.softplus = nn.Softplus()
    
    def forward(self, z, prefix="vertex_seq", obs_seq=None):
        """
        z: [batch_size, latent_dim]
        obs_seq: if training (guide), we may pass the observed list of vertices,
                 each is variable length. We'll do a single-sample approach
                 for illustration: we'll enumerate steps up to max_steps.

        We return the entire *list of random draws (x,y)* as well as a 
        list of *continue_flag*.
        
        If obs_seq is not None, we condition on the provided data (for the guide).
        In the model we *sample* from the distribution.
        """
        
        batch_size = z.shape[0]
        # Initialize LSTM hidden state
        h_0 = self.fc_h(z)
        c_0 = self.fc_c(z)
        
        h_t, c_t = torch.tanh(h_0), torch.tanh(c_0)
        
        # Keep track of samples
        vertex_samples = []
        cont_samples = []
        
        for step in range(self.max_steps):
            # Sample continuation
            p_continue = torch.sigmoid(self.fc_continue(h_t))  # [batch_size, 1]
            
            # For pyro, we need to shape things carefully
            cont_name = f"{prefix}_cont_{step}"
            cont_dist = dist.Bernoulli(probs=p_continue.squeeze(-1))
            
            if obs_seq is not None and step < len(obs_seq[0]):  # we assume batch_size=1 for simplicity or we index
                # We'll do an extremely simplified approach for demonstration
                # (In a real scenario you'd handle batching carefully.)
                cont_val = 1.0 if step < len(obs_seq[0]) else 0.0
                cont_sample = pyro.sample(cont_name, cont_dist.to_event(0), obs=torch.tensor([cont_val]))
            else:
                cont_sample = pyro.sample(cont_name, cont_dist.to_event(0))
            
            cont_samples.append(cont_sample)
            
            # If cont_sample=0 => no more vertices from now on, but let's keep sampling 
            # up to max_steps for pyro correctness, ignoring them in the final polygon.
            
            # Now sample (x, y)
            loc = self.fc_loc(h_t)
            log_scale = self.fc_scale(h_t)
            scale = self.softplus(log_scale)
            
            xy_dist = dist.Normal(loc, scale).to_event(1)
            
            xy_name = f"{prefix}_xy_{step}"
            
            if obs_seq is not None and step < len(obs_seq[0]):
                xy_val = torch.tensor(obs_seq[0][step], dtype=torch.float32)
                xy_sample = pyro.sample(xy_name, xy_dist, obs=xy_val)
            else:
                xy_sample = pyro.sample(xy_name, xy_dist)
            
            vertex_samples.append(xy_sample)
            
            # Update LSTM hidden
            # Use xy_sample as input (or zero if cont=0).
            # We'll do a simple approach here.
            input_t = h_t  # or torch.cat([xy_sample, h_t], dim=-1) if you want
            h_t, c_t = self.lstm_cell(input_t, (h_t, c_t))
        
        return vertex_samples, cont_samples

#####################################################################
# 3) Full VAE Model
#####################################################################

class MetasurfaceVAE(PyroModule):
    """
    We'll define a generative model p(spectrum | z) p(z).
    The latent z includes both the shape-vertex sequence AND the c-value.
    Actually, we can break it up:
      - z_main: some continuous embedding of dimension z_dim
      - c: we can treat c as part of the latent or treat it as partially observed
    Then the decoder tries to reconstruct the reflection spectrum
    from (z_main, c) AND also generates the polygon vertices via LSTM.
    """
    def __init__(self, n_wavelengths, z_dim=16, hidden_dim=64, max_verts=20):
        super().__init__()
        self.n_wavelengths = n_wavelengths
        self.z_dim = z_dim
        
        # Prior p(z_main) ~ N(0,I)
        # We also treat c ~ N(0,1) or you can treat it as known in some fraction of data
        # For simplicity, we do c as part of the latent with normal prior,
        # or if we have c observed, we condition in the model() vs. guide() for semi-supervision.
        
        # Define a small MLP to decode (z_main, c) => reflection spectrum
        self.decoder_spectrum = PyroModule[nn.Sequential](
            nn.Linear(z_dim+1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_wavelengths)
        )
        
        self.vertex_decoder = VertexLSTMDecoder(latent_dim=z_dim+1,
                                                hidden_dim=hidden_dim,
                                                max_steps=max_verts)
    
    def model(self, spectrum, c_obs=None, vertices_obs=None):
        """
        model(...) is the *generative* story:
        
        1) Sample z_main ~ N(0,I).
        2) Sample c ~ N(0,1) unless c is observed (semi-supervised).
        3) Using (z_main, c), decode polygon (variable-length).
        4) Using (z_main, c), decode reflection spectrum => likelihood p(spectrum | z,c).
        """
        pyro.module("metasurface_vae", self)
        
        batch_size = spectrum.shape[0]
        
        # 1) z_main prior
        with pyro.plate("data_plate", batch_size):
            z_main = pyro.sample("z_main", dist.Normal(torch.zeros(batch_size, self.z_dim),
                                                       torch.ones(batch_size, self.z_dim))
                                 .to_event(1))
            
            # 2) c prior or observed
            if c_obs is None:
                c_latent = pyro.sample("c_latent",
                                       dist.Normal(torch.zeros(batch_size,1),
                                                   torch.ones(batch_size,1))
                                       .to_event(1))
            else:
                # Condition on observed c
                c_latent = pyro.sample("c_latent",
                                       dist.Delta(c_obs).to_event(1))
            
            # 3) decode polygon
            # For simplicity, let's do 1-sample per item in batch.
            # In real usage, you'd handle variable-len lists carefully.
            if vertices_obs is not None:
                # we store the list for each item in batch, here just do item=0 for demonstration
                for i in range(batch_size):
                    self.vertex_decoder.forward(torch.cat([z_main[i], c_latent[i]]).unsqueeze(0),
                                                prefix=f"vertex_seq_{i}",
                                                obs_seq=[vertices_obs[i]])  # pass as a list-of-lists
            else:
                for i in range(batch_size):
                    self.vertex_decoder.forward(torch.cat([z_main[i], c_latent[i]]).unsqueeze(0),
                                                prefix=f"vertex_seq_{i}",
                                                obs_seq=None)
            
            # 4) decode reflection => Gaussian or some distribution around the measured spectrum
            #    We get mean from the MLP, set some fixed scale or param
            zc = torch.cat([z_main, c_latent], dim=-1)  # [batch_size, z_dim+1]
            pred_mean = self.decoder_spectrum(zc)
            
            # We pick a scale for the distribution. Let's assume a small fixed sigma
            sigma = 0.01
            pyro.sample("obs_spectrum",
                        dist.Normal(pred_mean, sigma).to_event(1),
                        obs=spectrum)
    
    def guide(self, spectrum, c_obs=None, vertices_obs=None):
        """
        The *approximate posterior* q(z_main, c | spectrum, possibly c_obs, vertices_obs).
        
        We'll define networks that encode the reflection => (z_main_mean, z_main_std).
        For c, if c is observed, then we just do Delta, else we do a param net.
        For the polygon, in a full hierarchical approach, you might do an inverse LSTM,
        but let's do a simpler approach where we directly condition on the observed vertices 
        if they're given. We'll *not* do a big complicated network for them here, 
        but you can add one if needed.
        """
        pyro.module("metasurface_vae", self)
        batch_size = spectrum.shape[0]
        
        # A simple MLP encoder for z_main
        # We'll define it inline (though typically you'd define a separate PyroModule).
        hidden_dim = 64
        # Let's get or define global param for the encoder
        encoder = pyro.param("encoder_weights", torch.randn(self.n_wavelengths, hidden_dim))
        encoder_bias = pyro.param("encoder_bias", torch.zeros(hidden_dim))
        out_w = pyro.param("encoder_out_w", torch.randn(hidden_dim, self.z_dim))
        out_b = pyro.param("encoder_out_b", torch.zeros(self.z_dim))
        
        def encode_spectrum(sp):
            h = torch.tanh(sp @ encoder + encoder_bias)  # shape [batch_size, hidden_dim]
            mean = h @ out_w + out_b                      # shape [batch_size, z_dim]
            # log_std param
            log_std_param = pyro.param("encoder_log_std", torch.zeros(self.z_dim))
            std = torch.exp(log_std_param)
            return mean, std
        
        with pyro.plate("data_plate", batch_size):
            mean, std = encode_spectrum(spectrum)
            z_main = pyro.sample("z_main", dist.Normal(mean, std).to_event(1))
            
            # c: if c_obs is known, Delta
            # else we learn a param
            if c_obs is None:
                c_loc = pyro.param("c_loc", torch.zeros(batch_size,1))
                c_scale = torch.exp(pyro.param("c_log_scale", torch.zeros(batch_size,1)))
                c_latent = pyro.sample("c_latent", dist.Normal(c_loc, c_scale).to_event(1))
            else:
                c_latent = pyro.sample("c_latent", dist.Delta(c_obs).to_event(1))
            
            # Condition on vertices if available
            # A more sophisticated approach is to define a reversed LSTM that 
            # takes the observed sequence and outputs distributions. 
            # For brevity, we'll just do a "nothing" step if we have obs.
            if vertices_obs is not None:
                for i in range(batch_size):
                    self.vertex_decoder.forward(torch.cat([z_main[i], c_latent[i]]).unsqueeze(0),
                                                prefix=f"vertex_seq_{i}",
                                                obs_seq=[vertices_obs[i]])
            else:
                for i in range(batch_size):
                    self.vertex_decoder.forward(torch.cat([z_main[i], c_latent[i]]).unsqueeze(0),
                                                prefix=f"vertex_seq_{i}",
                                                obs_seq=None)


#####################################################################
# 4) Training Loop
#####################################################################

def train_vae(csv_path, num_epochs=10, batch_size=4, lr=1e-3):
    # 1) Load data
    spectra_tensor, c_tensor, vertices_list = load_merged_s4_shapes(csv_path)
    
    N = spectra_tensor.shape[0]
    n_wavelengths = spectra_tensor.shape[1]
    
    # 2) Construct model
    vae = MetasurfaceVAE(n_wavelengths=n_wavelengths,
                         z_dim=16,
                         hidden_dim=64,
                         max_verts=20)
    
    # 3) SVI setup
    optimizer = Adam({"lr": lr})
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())
    
    # Just do a naive data loader
    indices = np.arange(N)
    
    for epoch in range(num_epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        for start_i in range(0, N, batch_size):
            end_i = start_i + batch_size
            idx = indices[start_i:end_i]
            sp_batch = spectra_tensor[idx]
            c_batch = c_tensor[idx]
            # Build a python list of observed vertices for each item
            # e.g. vertices_list[idx[i]] => we pass them as is 
            # (Be mindful of the shape for the decoding)
            vert_batch = [vertices_list[ii] for ii in idx]
            
            # Because of how we wrote code, let's do single-element mini-batches or adjust:
            # We'll do a single-batch approach that can handle it if you coded your LSTM carefully.
            # If the code assumes batch_size=1 for the LSTM, you might have to loop or re-write.
            # We'll assume the example works in a bigger batch for demonstration, 
            # but in practice, you might need to do some indexing carefully in vertex_decoder.
            
            loss = svi.step(sp_batch, c_obs=c_batch, vertices_obs=vert_batch)
            epoch_loss += loss
        
        print(f"Epoch {epoch+1}/{num_epochs}, ELBO loss={epoch_loss:.2f}")
    
    print("Training complete. You can now sample or do inference with the learned model.")


#####################################################################
# 5) Main
#####################################################################
if __name__ == "__main__":
    # Example usage
    csv_path = "merged_s4_shapes.csv"
    if not os.path.isfile(csv_path):
        print(f"CSV file '{csv_path}' not found! Please adjust the path.")
    else:
        train_vae(csv_path, num_epochs=10, batch_size=2, lr=1e-3)
