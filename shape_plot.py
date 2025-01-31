#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Base class: for angle-sorting and polygon plotting
###############################################################################
class PointPolygonPlotter:
    """
    Base class providing:
      - angle_sort(points): returns points sorted by ascending polar angle
      - plot_polygon(ax, points, color='blue', alpha=0.3): draws a closed polygon
    """

    @staticmethod
    def angle_sort(points):
        """
        Sort Nx2 array by ascending polar angle, angle = atan2(y,x).
        Returns Nx2 in CCW order.
        """
        angles = np.arctan2(points[:,1], points[:,0])
        idx_order = np.argsort(angles)
        return points[idx_order]

    @staticmethod
    def plot_polygon(ax, points, color='blue', alpha=0.3):
        """
        Draw a closed polygon from Nx2 points in (presumably) CCW order.
        If N<2, draw scatter.
        """
        import matplotlib.patches as patches
        from matplotlib.path import Path

        points = np.asarray(points)
        n = len(points)
        if n < 2:
            ax.scatter(points[:,0], points[:,1], c=color)
            return

        # close
        closed = np.concatenate([points, points[:1]], axis=0)
        codes  = [Path.MOVETO] + [Path.LINETO]*(n-1) + [Path.CLOSEPOLY]
        path   = Path(closed, codes)
        patch  = patches.PathPatch(path, facecolor=color, edgecolor=color, alpha=alpha, lw=1.5)
        ax.add_patch(patch)
        ax.autoscale_view()

###############################################################################
# Derived class (or a helper) for Q1 -> C4 usage
###############################################################################
class Q1PolygonC4Plotter(PointPolygonPlotter):
    """
    Provides:
      - check_q1(points): ensure all points in Q1 (x>0,y>0)
      - replicate_c4(points): replicate Nx2 points around origin
      - plot_q1_c4(ax, q1_points, color): a single call that checks Q1,
                                          replicates c4,
                                          sorts ccw,
                                          and calls base plot.
    """

    @staticmethod
    def check_q1(points):
        """Check that all x>0,y>0. If not, raise ValueError."""
        if not np.all(points>0):
            raise ValueError("Some points are not strictly in the first quadrant (x>0,y>0).")

    @staticmethod
    def replicate_c4(points):
        """
        Given Nx2 in Q1 => replicate by C4 about the origin:
          ( x,  y)
          (-y,  x)
          (-x, -y)
          ( y, -x)
        Returns shape(4*N,2).
        """
        all_pts = []
        for (x,y) in points:
            # Q1
            all_pts.append([ x,  y])
            # Q2
            all_pts.append([-y,  x])
            # Q3
            all_pts.append([-x, -y])
            # Q4
            all_pts.append([ y, -x])
        return np.array(all_pts, dtype=np.float32)

    def plot_q1_c4(self, ax, q1_points, color='blue', alpha=0.3):
        """
        1) Check Q1
        2) replicate C4
        3) angle-sort => CCW
        4) plot polygon
        """
        self.check_q1(q1_points)
        c4_pts = self.replicate_c4(q1_points)
        c4_sorted = self.angle_sort(c4_pts)
        self.plot_polygon(ax, c4_sorted, color=color, alpha=alpha)

###############################################################################
# if __name__ == "__main__": test/demo
###############################################################################
if __name__ == "__main__":
    import random

    np.random.seed(42)
    random.seed(42)

    # We'll create 5 random shapes:
    num_shapes = 5
    colors = ['red','green','blue','magenta','orange','cyan','purple','brown']

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal','box')
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    ax.set_title("Q1 -> C4 replicate => CCW plot demonstration")
    ax.grid(True)

    plotter = Q1PolygonC4Plotter()

    for i in range(num_shapes):
        n = random.randint(1,4)
        # generate Nx2 in Q1 => [0,1]^2 with x>0,y>0
        q1 = np.random.rand(n,2)  # random in [0,1]
        # We can ensure strictly x>0,y>0 if needed:
        # but rand() won't produce exactly 0. We'll trust that is fine.

        col = colors[i % len(colors)]
        plotter.plot_q1_c4(ax, q1, color=col, alpha=0.4)

    plt.show()

