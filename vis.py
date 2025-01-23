# vis.py

import numpy as np
import matplotlib.pyplot as plt

def c4_polygon(ax, shape_points, color='blue', alpha=0.3, fill=True):
    """
    Plots the polygon in Q1 and replicates it by rotating ±90°, 180°.
    shape_points: (N,2) array of vertices in the *first quadrant* (x>0, y>=0).
    We'll treat them as a closed polygon in Q1, then replicate in Q2,Q3,Q4.
    """

    if shape_points.shape[0] < 3:
        # Not enough vertices to form a polygon; just scatter points
        ax.scatter(shape_points[:,0], shape_points[:,1], c=color, marker='o')
        return

    def rotate_90(points):
        # (x,y) -> (-y,x)
        rotated = []
        for (x,y) in points:
            rotated.append([-y, x])
        return np.array(rotated)

    # Q1 polygon
    q1 = shape_points
    # replicate
    q2 = rotate_90(q1)
    q3 = rotate_90(q2)
    q4 = rotate_90(q3)

    if fill:
        def fill_poly(pts):
            if len(pts) >= 3:
                ax.fill(pts[:,0], pts[:,1], color=color, alpha=alpha)
        fill_poly(q1)
        fill_poly(q2)
        fill_poly(q3)
        fill_poly(q4)
    else:
        # Just draw lines
        ax.plot(q1[:,0], q1[:,1], color=color)
        ax.plot(q2[:,0], q2[:,1], color=color)
        ax.plot(q3[:,0], q3[:,1], color=color)
        ax.plot(q4[:,0], q4[:,1], color=color)


def plot_spectra(ax, spectra_gt, spectra_pred, color_gt='blue', color_pred='red'):
    """
    Plots the 11×100 ground truth (blue) vs. predicted (red) reflectances.
    spectra_gt, spectra_pred: shape (11,100)
    """
    xvals = np.arange(spectra_gt.shape[1])
    for i in range(spectra_gt.shape[0]):
        ax.plot(xvals, spectra_gt[i], color=color_gt, alpha=0.3)
        ax.plot(xvals, spectra_pred[i], color=color_pred, alpha=0.3)
    ax.set_xlabel("Wavelength Index (0..99)")
    ax.set_ylabel("Reflectance")
    ax.set_title("Spectra (Blue=GT, Red=Pred)")
    ax.grid(True)

