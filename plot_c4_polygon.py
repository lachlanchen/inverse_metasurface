#!/usr/bin/env python3

"""
c4_polygon.py
-------------
Reads predicted_vertices.txt lines, replicates each point (with presence>0.5) 
in all four quadrants (0, 90, 180, 270 degrees), angle-sorts, closes the polygon,
and plots it. If pres=0.0 (or <=0.5), we do NOT include that point at all.

Usage:
  python c4_polygon.py predicted_vertices.txt [out.png]

Example 'predicted_vertices.txt' lines:
  c=0.172
  Vertex 0: pres=1.000, x=0.453, y=0.440
  Vertex 1: pres=1.000, x=0.536, y=0.450
  Vertex 2: pres=0.000, x=0.388, y=1.146
  Vertex 3: pres=0.000, x=0.409, y=0.959
We skip any line where pres <= 0.5.

This produces a single polygon that extends into all four quadrants if
the original (x>0,y>0) points are replicated via rotations of 90, 180, 270 deg.
No clamping is performed, so negative coordinates in 2nd/3rd/4th quadrants will appear.
"""

import sys
import math
import numpy as np

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def replicate_c4(verts):
    """
    Given points [N,2] from the first quadrant, replicate them
    by rotating angles 0, 90, 180, 270. No clamping is done,
    so negative coordinates will appear in the other quadrants.
    Returns shape [4N,2].
    """
    out_list = []
    angles   = [0, math.pi/2, math.pi, 3*math.pi/2]
    for ang in angles:
        cosA = math.cos(ang)
        sinA = math.sin(ang)
        rot  = torch.tensor([[cosA, -sinA],
                             [sinA,  cosA]], dtype=torch.float)
        chunk= verts @ rot.T  # shape [N,2]
        out_list.append(chunk)
    return torch.cat(out_list, dim=0)

def angle_sort(points):
    """
    Sort points [N,2] by ascending polar angle in [-π, π].
    We'll use torch.atan2(y,x).
    Returns a new tensor with points in sorted order.
    """
    px  = points[:,0]
    py  = points[:,1]
    ang = torch.atan2(py, px)
    idx = torch.argsort(ang)
    return points[idx]

def close_polygon(pts):
    """
    If we have >=2 points, replicate the first point at the end
    so that we can fill(...) for a closed polygon.
    """
    if pts.size(0)>1:
        pts = torch.cat([pts, pts[:1]], dim=0)
    return pts

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} predicted_vertices.txt [out.png]")
        sys.exit(1)

    infile  = sys.argv[1]
    outfile = "c4_polygon.png"
    if len(sys.argv)>2:
        outfile = sys.argv[2]

    c_val   = None
    keep_pts= []  # will store (x,y) for lines with pres>0.5

    # 1) Parse the input file
    with open(infile,'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("c="):
                # parse c
                try:
                    c_val = float(line.split('=')[1])
                except:
                    c_val = None
            elif line.startswith("Vertex"):
                # e.g. "Vertex 0: pres=1.000, x=0.536, y=0.450"
                sub= line.split(',')
                if len(sub)<3:
                    continue
                try:
                    pres_str= sub[0].split('=')[1].strip()  # e.g. "1.000"
                    x_str   = sub[1].split('=')[1].strip()  # e.g. "0.536"
                    y_str   = sub[2].split('=')[1].strip()  # e.g. "0.450"
                    pres= float(pres_str)
                    x   = float(x_str)
                    y   = float(y_str)
                    # Only keep if pres>0.5
                    print("pres: ", pres)
                    if pres>0.5:
                        keep_pts.append((x,y))
                except:
                    continue

    if not keep_pts:
        print("[WARN] No presence>0.5 points found => No polygon plotted.")
        sys.exit(0)

    # Convert to torch shape [N,2]
    first_quad = torch.tensor(keep_pts, dtype=torch.float)

    # 2) Replicate in four quadrants
    c4_points  = replicate_c4(first_quad)

    # 3) Sort by angle
    sorted_pts = angle_sort(c4_points)
    # 4) Close polygon
    closed_pts = close_polygon(sorted_pts)

    # 5) Plot
    sx= closed_pts[:,0].numpy()
    sy= closed_pts[:,1].numpy()

    plt.figure()
    plt.fill(sx, sy, color='red', alpha=0.3)
    plt.plot(sx, sy, 'ro-',
             label=f"{len(keep_pts)} first-quadrant points => {c4_points.size(0)} total")
    title_c= c_val if c_val is not None else 0.0
    plt.title(f"C4 polygon, c={title_c:.3f}")
    plt.axhline(0,color='k',lw=0.5)
    plt.axvline(0,color='k',lw=0.5)
    plt.legend()
    plt.savefig(outfile)
    plt.close()

    print(f"[INFO] Plotted polygon with {len(keep_pts)} Q1 points => {c4_points.size(0)} total vertices. Saved to {outfile}.")

if __name__=="__main__":
    main()
