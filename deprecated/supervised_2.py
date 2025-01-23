#!/usr/bin/env python3

import os
import math
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import pandas as pd

from torch.utils.data import DataLoader
from scipy.interpolate import make_interp_spline

# 1) Import or define the same classes from supervised.py
#    (Dataset, parse_vertices_str, etc.) so we can load the data in the same way.
#    If you've put them into a separate module, you can do:
#
# from supervised import SupervisedDataset, LSTMModel, parse_vertices_str
#
# Otherwise, copy them here. For brevity, assume we do:
from supervised import SupervisedDataset, LSTMModel, HIDDEN_SIZE, DEVICE

###############################################################################
# 2) Generate a smooth random spectrum in [0,1]
###############################################################################
def generate_smooth_spectrum(num_points=100):
    """
    Generate a more realistic random reflection in [0,1] by:
      - picking ~8 random control points in [0,1]
      - cubic spline interpolation
    """
    n_ctrl= 8
    x_ctrl= np.linspace(0, 1, n_ctrl)
    y_ctrl= np.random.rand(n_ctrl)*0.8 +0.1  # in [0.1..0.9]
    spline= make_interp_spline(x_ctrl, y_ctrl, k=3)
    x_big= np.linspace(0,1,num_points)
    y_big= spline(x_big)
    y_big= np.clip(y_big, 0,1)
    return torch.tensor(y_big, dtype=torch.float)

###############################################################################
# 3) Helper for polygon plotting (similar to the SVI code)
###############################################################################
def replicate_c4(verts):
    """
    Replicate a first-quadrant polygon in C4 symmetry.
    Input: verts shape [N,2]
    Output: shape [4*N,2]
    """
    out_list=[]
    angles= [0, math.pi/2, math.pi, 3*math.pi/2]
    for a in angles:
        cosA= math.cos(a)
        sinA= math.sin(a)
        rot= torch.tensor([[cosA, -sinA],[sinA, cosA]],dtype=torch.float)
        chunk= verts @ rot.T
        out_list.append(chunk)
    return torch.cat(out_list, dim=0)

def angle_sort(points):
    px= points[:,0]
    py= points[:,1]
    ang= torch.atan2(py, px)
    idx= torch.argsort(ang)
    return points[idx]

def close_polygon(pts):
    if pts.size(0)>1:
        return torch.cat([pts, pts[:1]], dim=0)
    return pts

def plot_polygon(pts, c_val, out_path, title="C4 polygon"):
    """
    angle-sort, close, fill
    """
    pts_sorted = angle_sort(pts)
    pts_closed = close_polygon(pts_sorted)
    sx = pts_closed[:,0].numpy()
    sy = pts_closed[:,1].numpy()
    plt.figure()
    plt.fill(sx, sy, color='red', alpha=0.3)
    plt.plot(sx, sy, 'ro-')
    plt.title(f"{title}, c={c_val:.3f}")
    plt.axhline(0,color='k',lw=0.5)
    plt.axvline(0,color='k',lw=0.5)
    plt.savefig(out_path)
    plt.close()

###############################################################################
# 4) The inference script for the supervised model
###############################################################################
def test_supervised_inference(checkpoint_path, csv_path):
    """
    1) Load model
    2) Inference on (A) random smooth spectrum
                   (B) a real row from the dataset
    3) Save results in a timestamped folder
    """
    # 1) Load model
    model = LSTMModel(hidden_size=HIDDEN_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    # Prepare output dir
    dt_str= datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir= f"supervised_inference_{dt_str}"
    os.makedirs(out_dir, exist_ok=True)

    ###########################################################################
    # (A) Inference on a random smooth spectrum
    ###########################################################################
    random_sp = generate_smooth_spectrum(num_points=100).to(DEVICE)  # [100]
    # The model expects shape [B,100], so add batch dimension
    random_sp_input= random_sp.unsqueeze(0)                           # [1,100]
    with torch.no_grad():
        c_pred, vpres_pred, vxy_pred = model(random_sp_input)
        # c_pred => shape [1], vpres_pred => [1,4], vxy_pred => [1,4,2]
    c_val= float(c_pred[0].cpu().item())

    # Save random input & predictions
    np.savetxt(os.path.join(out_dir,"smooth_input_spectrum.txt"),
               random_sp.cpu().numpy(),
               fmt="%.5f")

    # Reconstruct "polygon" from predicted presence bits
    v_pres_np = vpres_pred[0].cpu().numpy()  # shape (4,)
    v_xy_np   = vxy_pred[0].cpu().numpy()    # shape (4,2)

    # We'll keep the first-quadrant points that the model "thinks" are present 
    # if v_pres>0.5
    keep_pts = []
    for i in range(4):
        if v_pres_np[i]>0.5:
            keep_pts.append(v_xy_np[i])
    if len(keep_pts)==0:
        print("[WARN] No predicted vertices from random. Possibly v_pres<0.5 for all.")
    keep_pts_t= torch.tensor(keep_pts, dtype=torch.float) if len(keep_pts)>0 else torch.zeros((1,2))

    # replicate in C4, plot
    c4_verts = replicate_c4(keep_pts_t)
    poly_path= os.path.join(out_dir,"smooth_polygon.png")
    plot_polygon(c4_verts, c_val, poly_path, title="Predicted C4 polygon (smooth input)")

    # also let's store the presence and coords
    with open(os.path.join(out_dir,"smooth_pred.txt"),"w") as f:
        f.write(f"Predicted c={c_val:.3f}\n")
        for i in range(4):
            f.write(f"v_pres[{i}]={v_pres_np[i]:.3f}, v_xy=({v_xy_np[i,0]:.3f},{v_xy_np[i,1]:.3f})\n")

    # Plot the smooth input vs. "some predicted reconstruction"?
    # Unlike the semi-supervised approach, we don't have a direct "decoder" that
    # reconstructs the spectrum. We only have predictions for c, v, x,y.
    # So we can't directly do "reconstructed_spectrum" from the model. 
    # We'll just plot the input spectrum alone.
    x_axis= np.arange(100)
    plt.figure()
    plt.plot(x_axis, random_sp.cpu().numpy(), marker='o', label="Smooth Input")
    plt.ylim([0,1])
    plt.legend()
    plt.title("Smooth random input spectrum")
    plt.savefig(os.path.join(out_dir,"spectrum_smooth.png"))
    plt.close()

    ###########################################################################
    # (B) Inference on a real row from the dataset
    ###########################################################################
    # Let's pick row=0 from the dataset for demonstration (or you can pick any).
    ds = SupervisedDataset(csv_path)
    if len(ds)==0:
        print("[WARN] CSV data is empty? Can't do dataset test.")
        return

    row_idx= 0  # pick row
    sample= ds[row_idx]
    real_sp= sample["input_seq"].to(DEVICE)    # shape [100]
    gt_c   = float(sample["c_val"].item())     # might be -1 if unknown
    gt_pres= sample["v_pres"].numpy()          # shape [4]
    gt_xy  = sample["v_xy"].numpy()            # shape [4,2]

    real_sp_in= real_sp.unsqueeze(0)           # [1,100]
    with torch.no_grad():
        c_pred2, vpres_pred2, vxy_pred2 = model(real_sp_in)
    c_val2   = float(c_pred2[0].cpu().item())
    v_pres2n = vpres_pred2[0].cpu().numpy() 
    v_xy2n   = vxy_pred2[0].cpu().numpy()

    # Save results
    with open(os.path.join(out_dir,"masked_inference.txt"), "w") as f:
        f.write(f"Row idx={row_idx}\n")
        f.write(f"GroundTruth c={gt_c:.3f}\n")
        f.write(f"Inferred c={c_val2:.3f}\n")
        f.write("\n---GroundTruth Vertices---\n")
        for i in range(4):
            if gt_pres[i]>0.5:
                f.write(f" GT Vertex {i}: x={gt_xy[i,0]:.3f}, y={gt_xy[i,1]:.3f}\n")
        f.write("\n---Predicted---\n")
        for i in range(4):
            f.write(f" Pred Vertex {i}: pres={v_pres2n[i]:.3f}, "
                    f"x={v_xy2n[i,0]:.3f}, y={v_xy2n[i,1]:.3f}\n")

    # Plot real row's spectrum
    real_sp_np = real_sp.cpu().numpy()
    plt.figure()
    plt.plot(np.arange(100), real_sp_np, marker='o', label="CSV row spectrum")
    plt.ylim([0,1])
    plt.legend()
    plt.title("Row-0 Real Spectrum")
    plt.savefig(os.path.join(out_dir,"row_spectrum.png"))
    plt.close()

    # Build polygon from predicted presence
    keep_pts2= []
    for i in range(4):
        if v_pres2n[i]>0.5:
            keep_pts2.append(v_xy2n[i])
    if len(keep_pts2)==0:
        keep_pts2_t= torch.zeros((1,2))
    else:
        keep_pts2_t= torch.tensor(keep_pts2, dtype=torch.float)
    c4_verts2= replicate_c4(keep_pts2_t)
    out_poly= os.path.join(out_dir,"row_polygon_pred.png")
    plot_polygon(c4_verts2, c_val2, out_poly, title="Pred C4 polygon from dataset row")

    # Also we can plot the GT polygon if c>=0
    # We'll do the same for GT presence bits
    keep_pts_gt= []
    for i in range(4):
        if gt_pres[i]>0.5:
            keep_pts_gt.append(gt_xy[i])
    if len(keep_pts_gt)==0:
        keep_pts_gt_t= torch.zeros((1,2))
    else:
        keep_pts_gt_t= torch.tensor(keep_pts_gt, dtype=torch.float)
    c4_verts_gt= replicate_c4(keep_pts_gt_t)
    out_poly_gt= os.path.join(out_dir,"row_polygon_gt.png")
    plot_polygon(c4_verts_gt, gt_c if gt_c>=0 else 999.0,
                 out_poly_gt, title="GT C4 polygon from dataset row")

    print(f"[INFO] Inference results saved to {out_dir}/")

###############################################################################
# 5) Main
###############################################################################
if __name__=="__main__":
    # Example usage:
    # python supervised_inference.py --ckpt checkpoint_simple/model.pt --csv merged_s4_shapes.csv
    import argparse
    parser= argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoint_simple/model.pt")
    parser.add_argument("--csv",  type=str, default="merged_s4_shapes.csv")
    args= parser.parse_args()

    test_supervised_inference(args.ckpt, args.csv)
