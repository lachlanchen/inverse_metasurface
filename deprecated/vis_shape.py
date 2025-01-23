#!/usr/bin/env python3
"""
shape_vis.py

Usage:
  python shape_vis.py shapes/outer_f0.50_20240101_123456.txt

This script:
  1) Reads (x,y) lines from a text file.
  2) Plots them as a polygon in matplotlib.
"""

import sys
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python shape_vis.py <polygon_file.txt>")
        sys.exit(1)

    filename = sys.argv[1]

    # Read the polygon file
    xs, ys = [], []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Expect lines "x,y"
            parts = line.split(',')
            if len(parts) != 2:
                continue
            x, y = float(parts[0]), float(parts[1])
            xs.append(x)
            ys.append(y)

    # Optionally close the polygon by repeating the first point
    if xs and ys:
        xs.append(xs[0])
        ys.append(ys[0])

    # Plot
    plt.figure()
    plt.plot(xs, ys, '-o', markersize=3)
    plt.title(f"Polygon from {filename}")
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()

