#!/usr/bin/env python3

"""
Test script: Use a trained semi-supervised AIR model to:
1) Generate a random "fake" reflection spectrum
2) Infer {c, v_pres, v_where} from the random spectrum
3) Reconstruct the spectrum from the latents
4) Plot both spectra + the predicted polygon (with C4 symmetry)
5) Save everything in "inference_results/" folder
"""

import os
import math
import numpy as np
import torch
import pyro

import matplotlib
matplotlib.use("Agg")  # so it can run headless
import matplotlib.pyplot as plt

from pyro.poutine import trace

# ----------------------------------------------------------------
# 1) We'll import the same model code from your "svi_metasurface_semi_3.py"
#    or place that code here for convenience.
#    We'll just replicate the essential parts: the model + guide + load state.
# ----------------------------------------------------------------

MAX_STEPS = 4

class SpectrumDecoder(torch.nn.Module):
    def __init__(self, n_waves=50, hidden_dim=64):
        super().__init__()
        self.vert_embed = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim+1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_waves)
        )

    def forward(self, c, v_pres, v_where):
        B, hidden_dim = c.size(0), 64
        accum = torch.zeros(B, hidden_dim, device=c.device)
        for t in range(MAX_STEPS):
            feat = self.vert_embed(v_where[:,t,:])
            pres = v_pres[:,t:t+1]
            accum += feat*pres
        x = torch.cat([accum, c], dim=-1)
        return self.final(x)

class SemiAirModel(torch.nn.Module):
    def __init__(self, n_waves=50):
        super().__init__()
        self.n_waves = n_waves
        self.decoder = SpectrumDecoder(n_waves, 64)

    def model(self, spectrum, c_vals, is_c_known, v_pres, v_where, is_v_known):
        pyro.module("SemiAirModel", self)
        B = spectrum.size(0)
        with pyro.plate("data", B):
            c0   = torch.zeros(B,1, device=spectrum.device)
            c_std= torch.ones(B,1, device=spectrum.device)
            raw_c= pyro.sample("raw_c",
                               pyro.distributions.Normal(c0, c_std).to_event(1))
            c_mask= is_c_known.float().unsqueeze(-1)
            c = c_mask*c_vals + (1-c_mask)*raw_c

            prev_pres = torch.ones(B,1, device=spectrum.device)
            v_pres_collect= []
            v_where_collect= []
            for t in range(MAX_STEPS):
                name_p = f"raw_pres_{t}"
                p_prob = 0.5*prev_pres
                raw_p  = pyro.sample(name_p,
                                     pyro.distributions.Bernoulli(p_prob).to_event(1))
                pres_mask= is_v_known.float().unsqueeze(-1)*(v_pres[:,t]>=0.5).float().unsqueeze(-1)
                pres_val = pres_mask*v_pres[:,t:t+1] + (1-pres_mask)*raw_p

                name_w= f"raw_where_{t}"
                loc0= torch.zeros(B,2, device=spectrum.device)
                sc0= torch.ones(B,2, device=spectrum.device)
                raw_w= pyro.sample(name_w,
                                   pyro.distributions.Normal(loc0, sc0)
                                   .mask(raw_p)
                                   .to_event(1))
                where_mask= is_v_known.float().unsqueeze(-1)*pres_val
                w_val= where_mask*v_where[:,t,:] + (1-where_mask)*raw_w

                v_pres_collect.append(pres_val)
                v_where_collect.append(w_val)
                prev_pres= pres_val

            v_pres_cat= torch.cat(v_pres_collect, dim=1)
            v_where_cat= torch.stack(v_where_collect, dim=1)

            mean_sp= self.decoder(c, v_pres_cat, v_where_cat)
            pyro.sample("obs_spectrum",
                        pyro.distributions.Normal(mean_sp, 0.01).to_event(1),
                        obs=spectrum)

    def guide(self, spectrum, c_vals, is_c_known, v_pres, v_where, is_v_known):
        pyro.module("SemiAirModel", self)
        B = spectrum.size(0)

        if not hasattr(self,"enc_c"):
            self.enc_c = torch.nn.Sequential(
                torch.nn.Linear(self.n_waves,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,2)
            )

        with pyro.plate("data", B):
            out_c= self.enc_c(spectrum)
            c_loc= out_c[:,0:1]
            c_scale= torch.nn.functional.softplus(out_c[:,1:2])+1e-4
            raw_c= pyro.sample("raw_c",
                               pyro.distributions.Normal(c_loc, c_scale).to_event(1))

            prev_pres = torch.ones(B,1, device=spectrum.device)
            for t in range(MAX_STEPS):
                if not hasattr(self, f"pres_net_{t}"):
                    setattr(self, f"pres_net_{t}",
                            torch.nn.Sequential(
                                torch.nn.Linear(self.n_waves,64),
                                torch.nn.ReLU(),
                                torch.nn.Linear(64,1)
                            ))
                pres_net= getattr(self, f"pres_net_{t}")
                p_logit= pres_net(spectrum)
                p_prob = torch.sigmoid(p_logit)*prev_pres

                raw_p= pyro.sample(f"raw_pres_{t}",
                                   pyro.distributions.Bernoulli(p_prob).to_event(1))

                if not hasattr(self, f"where_net_{t}"):
                    setattr(self, f"where_net_{t}",
                            torch.nn.Sequential(
                                torch.nn.Linear(self.n_waves,4)
                            ))
                where_net= getattr(self, f"where_net_{t}")
                w_out= where_net(spectrum)
                w_loc= w_out[:,0:2]
                w_scale= torch.nn.functional.softplus(w_out[:,2:4])+1e-4

                raw_w= pyro.sample(f"raw_where_{t}",
                                   pyro.distributions.Normal(w_loc, w_scale)
                                   .mask(raw_p)
                                   .to_event(1))

                prev_pres= raw_p


# ------------------------------------------------------------------------
# 2) We'll define a function to load the pyro param store from checkpoint.
# ------------------------------------------------------------------------
def load_trained_model(ckpt_path, n_waves=50):
    """Load param store and return a SemiAirModel instance with loaded params."""
    model = SemiAirModel(n_waves)
    pyro.clear_param_store()
    if os.path.isfile(ckpt_path):
        st = torch.load(ckpt_path, map_location="cpu")
        pyro.get_param_store().set_state(st)
        print(f"[INFO] Loaded Param Store from {ckpt_path}")
    else:
        print(f"[WARN] Checkpoint {ckpt_path} not found, using random init.")
    return model


# ------------------------------------------------------------------------
# 3) A function to do inference on a single spectrum
#    returning c, v_pres, v_where
# ------------------------------------------------------------------------
@torch.no_grad()
def infer_latents(model, spectrum, device="cpu"):
    """
    model: an instance of SemiAirModel
    spectrum: [1, n_waves] or [n_waves]
    Return: c, v_pres, v_where
    We'll pass dummy placeholders for c_vals, v_pres, etc. as all unknown.
    """
    if spectrum.dim()==1:
        spectrum = spectrum.unsqueeze(0)  # [1, n_waves]
    B = spectrum.size(0)

    # We'll create placeholders for c_vals, is_c_known, etc.
    c_vals = torch.zeros(B,1, device=device)
    is_c_known = torch.zeros(B,dtype=torch.bool)
    v_pres = torch.zeros(B, MAX_STEPS)
    v_where= torch.zeros(B, MAX_STEPS,2)
    is_v_known= torch.zeros(B,dtype=torch.bool)

    # trace the guide
    guide_tr = trace(model.guide).get_trace(
        spectrum, c_vals, is_c_known,
        v_pres, v_where, is_v_known
    )

    # parse out latents
    #  c = maskC*c_vals + (1-maskC)*raw_c in model, but let's do the same logic here
    raw_c_val = guide_tr.nodes["raw_c"]["value"]  # shape [B,1]
    c_final = raw_c_val  # since c unknown => we used raw_c

    v_pres_list=[]
    v_where_list=[]
    prev_pres_est= torch.ones(B,1, device=device)
    for t in range(MAX_STEPS):
        name_pres= f"raw_pres_{t}"
        name_where= f"raw_where_{t}"
        raw_pres_val = guide_tr.nodes[name_pres]["value"]  # [B,1]
        raw_where_val= guide_tr.nodes[name_where]["value"] # [B,2]

        v_pres_list.append(raw_pres_val)
        v_where_list.append(raw_where_val)
        prev_pres_est= raw_pres_val

    v_pres_cat = torch.cat(v_pres_list, dim=1)    # [B,4]
    v_where_cat= torch.stack(v_where_list, dim=1) # [B,4,2]

    return c_final, v_pres_cat, v_where_cat

# ------------------------------------------------------------------------
# 4) Let's define a C4 symmetry function to replicate predicted vertices
# ------------------------------------------------------------------------
def replicate_c4(verts):
    """
    Suppose 'verts' is shape [N,2], points in the first quadrant.
    We'll replicate them for quadrants 2,3,4 to get a c4-symmetric shape.
    This is a simplistic approach:
     angle_offset = [0, pi/2, pi, 3pi/2]
    We'll just replicate each point. If you only have up to 4 vertices, 
    we can't guarantee a smooth polygon, but let's illustrate.
    Return all_verts: shape [4*N, 2].
    """
    out_list = []
    n = verts.size(0)
    angles = [0, math.pi/2, math.pi, 3*math.pi/2]
    for ang in angles:
        cosA= math.cos(ang)
        sinA= math.sin(ang)
        rot = torch.stack([
            torch.tensor([cosA, -sinA]),
            torch.tensor([sinA,  cosA])
        ], dim=0).float()  # 2x2
        chunk = verts @ rot.T  # shape [n,2]
        out_list.append(chunk)
    all_pts = torch.cat(out_list, dim=0)
    return all_pts


# ------------------------------------------------------------------------
# 5) Visualization code
# ------------------------------------------------------------------------
def plot_inference(random_spec, rec_spec, c_pred, v_pres_pred, v_where_pred, out_dir="inference_results"):
    os.makedirs(out_dir, exist_ok=True)

    # Plot the random_spec vs rec_spec
    x_axis = np.arange(random_spec.size(0))
    plt.figure(figsize=(6,5))
    plt.plot(x_axis, random_spec.cpu().detach().numpy(), label="Input Spectrum", marker='o')
    plt.plot(x_axis, rec_spec.detach().cpu().numpy(), label="Reconstructed", marker='x')
    plt.title("Random Spectrum vs Reconstructed")
    plt.xlabel("Wavelength index")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spectra_compare.png"))
    plt.close()

    # Next, let's do a quick shape plot with c4 symmetry from predicted vertices
    # We'll collect all the vertices that are present (where v_pres>0.5) in the first quadrant
    # Then replicate them
    # For simplicity, we do a threshold on presence => 0.5
    present_mask = (v_pres_pred>0.5)
    keep_verts = []
    for t in range(MAX_STEPS):
        if float(present_mask[0,t])>0.5:
            keep_verts.append(v_where_pred[0,t]) # shape [2]
    if len(keep_verts)==0:
        # no vertices
        print("No predicted vertices.")
        return
    keep_verts_t = torch.stack(keep_verts, dim=0) # shape [n,2]
    # replicate c4
    c4_verts = replicate_c4(keep_verts_t)
    c4_np = c4_verts.cpu().numpy()

    # Plot
    plt.figure(figsize=(5,5))
    plt.scatter(c4_np[:,0], c4_np[:,1], c='r', marker='o', label="C4 polygon points")
    plt.axhline(0, c='k', lw=0.5)
    plt.axvline(0, c='k', lw=0.5)
    plt.title(f"C4 polygon from predicted vertices; c={float(c_pred[0]):.3f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "c4_polygon.png"))
    plt.close()

    print(f"[INFO] Plots saved in {out_dir}/")


# ------------------------------------------------------------------------
# 6) Main script
# ------------------------------------------------------------------------
def main():
    # 6.1) Load model & param store
    # For example, after finishing training with "svi_metasurface_semi_3.py",
    # you might have "ckpt_epoch5.pt" in "svi_results/"
    ckpt_path = "svi_results/ckpt_epoch5.pt"
    n_waves = 60  # Adjust if your data had a different dimension
    device = "cpu"

    model = SemiAirModel(n_waves).to(device)
    pyro.clear_param_store()
    if os.path.isfile(ckpt_path):
        st = torch.load(ckpt_path, map_location=device)
        pyro.get_param_store().set_state(st)
        print(f"[INFO] Loaded checkpoint {ckpt_path}")
    else:
        print(f"[WARN] {ckpt_path} not found, using random init param store...")

    # 6.2) Generate a random "reasonable" spectrum: e.g. 60 points in [0..1]
    # You might want something smoother. We'll just do a random shape for example:
    random_spec = torch.rand(n_waves)*0.5 + 0.25  # in [0.25..0.75], for instance
    random_spec = random_spec.to(device)

    # 6.3) Infer latents from this random spec
    c_pred, v_pres_pred, v_where_pred = infer_latents(model, random_spec, device=device)

    # 6.4) Reconstruct the spectrum
    # We'll do a single-batch decode
    # shape => [1, n_waves]
    c_input = c_pred.clone().view(1,1)
    v_pres_input = v_pres_pred.clone().view(1, MAX_STEPS)
    v_where_input= v_where_pred.clone().view(1, MAX_STEPS, 2)
    rec_spec = model.decoder(c_input, v_pres_input, v_where_input).squeeze(0) # shape [n_waves]

    # 6.5) Print out the results
    print(f"Predicted c: {float(c_pred[0]):.3f}")
    for t in range(MAX_STEPS):
        print(f" Vertex {t}: presence={float(v_pres_pred[0,t]):.3f}, location=({float(v_where_pred[0,t,0]):.3f}, {float(v_where_pred[0,t,1]):.3f})")

    # 6.6) Compare the geometry distribution of presence
    n_vertices_pred = int((v_pres_pred>0.5).sum().item())
    print(f"Predicted total vertices present: {n_vertices_pred}")

    # 6.7) Plot and save
    out_dir = "inference_results"
    plot_inference(random_spec, rec_spec, c_pred, v_pres_pred, v_where_pred, out_dir=out_dir)

    # 6.8) Save the data to a CSV or text
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir,"random_spectrum.txt"), random_spec.cpu().detach().numpy(), fmt="%.5f")
    np.savetxt(os.path.join(out_dir,"reconstructed_spectrum.txt"), rec_spec.cpu().detach().numpy(), fmt="%.5f")
    with open(os.path.join(out_dir,"predicted_latents.txt"),"w") as f:
        f.write(f"Predicted c = {float(c_pred[0]):.5f}\n")
        for t in range(MAX_STEPS):
            pres_val = float(v_pres_pred[0,t])
            wx       = float(v_where_pred[0,t,0])
            wy       = float(v_where_pred[0,t,1])
            f.write(f"vertex {t}: presence={pres_val:.3f}, location=({wx:.3f}, {wy:.3f})\n")

    print(f"[INFO] Inference data saved to {out_dir}/")


if __name__=="__main__":
    main()
