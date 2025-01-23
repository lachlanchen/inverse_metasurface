#!/usr/bin/env python3
"""
vis_circles.py
Visualize circles stored in a text file where each line has:
    cx, cy, radius
(e.g. "0.699963,0.706795,0.174960")

Usage:
  python vis_circles.py shapes/20250104_115733/circles_csvPartialGSST_C0.0_shape1.txt
"""
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    if len(sys.argv) < 2:
        print("Usage: python vis_circles.py <circle_file.txt>")
        sys.exit(1)

    circle_file = sys.argv[1]

    circles = []
    with open(circle_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 3:
                cx, cy, r = map(float, parts)
                circles.append((cx, cy, r))

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6,6))

    # Optionally draw the 1×1 unit cell boundary
    rect = patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='gray', linewidth=1)
    ax.add_patch(rect)

    # Add each circle to the plot
    for (cx, cy, r) in circles:
        circle_patch = patches.Circle((cx, cy), r, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(circle_patch)

    # Format the axis
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Circles from: {circle_file}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

