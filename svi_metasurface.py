import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceGraph_ELBO
import pyro.optim as optim

# -------------------------
# 1. Load and Preprocess Data
# -------------------------

# Load merged data
data_file = "merged_s4_shapes.csv"  # Adjust path as needed
data = pd.read_csv(data_file)

# Extract the R spectrum as input features
spectrum_cols = [col for col in data.columns if col.startswith("R@")]
spectra = data[spectrum_cols].values  # Shape: (num_samples, num_wavelengths)

# Extract first-quadrant vertices
def extract_first_quadrant_vertices(vertices_str):
    vertices = []
    for vertex in vertices_str.split(";"):
        x, y = map(float, vertex.split(","))
        if x >= 0 and y > 0:  # First quadrant
            vertices.append((x, y))
    return vertices

data["first_quadrant_vertices"] = data["vertices_str"].apply(extract_first_quadrant_vertices)

# Convert vertices into tensors
max_vertices = 10  # Maximum possible vertices in the first quadrant
vertex_tensors = []
vertex_presence = []
for vertices in data["first_quadrant_vertices"]:
    tensor = torch.zeros((max_vertices, 2))
    presence = torch.zeros(max_vertices)
    for i, (x, y) in enumerate(vertices[:max_vertices]):
        tensor[i] = torch.tensor([x, y])
        presence[i] = 1.0
    vertex_tensors.append(tensor)
    vertex_presence.append(presence)

vertex_tensors = torch.stack(vertex_tensors)  # Shape: (num_samples, max_vertices, 2)
vertex_presence = torch.stack(vertex_presence)  # Shape: (num_samples, max_vertices)

# Crystallization factor (c)
crystallization = torch.tensor(data["c"].values, dtype=torch.float32)  # Shape: (num_samples,)

# Convert R spectrum to tensor
spectra_tensor = torch.tensor(spectra, dtype=torch.float32)  # Shape: (num_samples, num_wavelengths)

# -------------------------
# 2. Define Model and Guide
# -------------------------

# Number of wavelengths and max vertices
num_wavelengths = spectra_tensor.size(1)

# Decoder network to map latent variables to R spectrum
class Decoder(nn.Module):
    def __init__(self, num_wavelengths, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, num_wavelengths)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))

# Initialize the decoder
decoder = Decoder(num_wavelengths, latent_dim=1 + 2 * max_vertices)  # c + (x, y) for each vertex

# Generative model
def model(spectra, crystallization=None, vertices=None, vertex_presence=None):
    pyro.module("decoder", decoder)
    batch_size = spectra.size(0)
    
    # Prior for crystallization factor (c)
    c_prior = dist.Normal(0.5, 0.2).expand([batch_size]).to_event(1)
    c = pyro.sample("c", c_prior) if crystallization is None else pyro.sample("c", dist.Delta(crystallization).to_event(1))
    
    # Prior for vertex presence
    v_pres = pyro.sample("v_pres", dist.Geometric(0.5).expand([batch_size, max_vertices]).to_event(2)) \
        if vertex_presence is None else pyro.sample("v_pres", dist.Delta(vertex_presence).to_event(2))
    
    # Prior for vertices (x, y)
    z_where_loc = torch.zeros(batch_size, max_vertices, 2)
    z_where_scale = torch.ones(batch_size, max_vertices, 2)
    z_where = pyro.sample("z_where", dist.Normal(z_where_loc, z_where_scale).mask(v_pres).to_event(2)) \
        if vertices is None else pyro.sample("z_where", dist.Delta(vertices).to_event(2))
    
    # Combine latent variables
    z = torch.cat([c.unsqueeze(-1), z_where.view(batch_size, -1)], dim=1)
    # Decode latent variables into R spectrum
    pred_spectra = decoder(z)
    
    # Likelihood
    pyro.sample("obs", dist.Normal(pred_spectra, 0.1).to_event(1), obs=spectra)

# Guide (Variational Inference)
class Guide(nn.Module):
    def __init__(self, num_wavelengths, max_vertices):
        super().__init__()
        self.fc1 = nn.Linear(num_wavelengths, 128)
        self.fc2_c = nn.Linear(128, 1)
        self.fc2_vertices = nn.Linear(128, 2 * max_vertices)
        self.fc2_presence = nn.Linear(128, max_vertices)

    def forward(self, spectra):
        h = torch.relu(self.fc1(spectra))
        c_loc = torch.sigmoid(self.fc2_c(h)).squeeze(-1)
        v_pres_logits = self.fc2_presence(h)
        z_where_loc = self.fc2_vertices(h).view(-1, max_vertices, 2)
        return c_loc, v_pres_logits, z_where_loc

guide_nn = Guide(num_wavelengths, max_vertices)

def guide(spectra, crystallization=None, vertices=None, vertex_presence=None):
    pyro.module("guide_nn", guide_nn)
    c_loc, v_pres_logits, z_where_loc = guide_nn(spectra)
    
    # Guide for crystallization factor
    pyro.sample("c", dist.Normal(c_loc, 0.1).to_event(1))
    
    # Guide for vertex presence
    pyro.sample("v_pres", dist.Geometric(torch.sigmoid(v_pres_logits)).to_event(2))
    
    # Guide for vertices
    z_where_scale = 0.1 * torch.ones_like(z_where_loc)
    pyro.sample("z_where", dist.Normal(z_where_loc, z_where_scale).to_event(2))

# -------------------------
# 3. Training
# -------------------------

# Optimizer
adam = optim.Adam({"lr": 1e-3})

# SVI
svi = SVI(model, guide, adam, loss=TraceGraph_ELBO())

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    loss = svi.step(spectra_tensor, crystallization, vertex_tensors, vertex_presence)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss / len(spectra_tensor):.4f}")

