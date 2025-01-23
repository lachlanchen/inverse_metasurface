#!/usr/bin/env python3
"""
plot_polygons.py

Usage:
  python plot_polygons.py <folder_with_txt_files>

Example:
  python plot_polygons.py shapes/20250104_121237-polygon-wo-hollow

What it does:
  1) Looks for all *.txt files in <folder_with_txt_files>.
  2) For each *.txt:
     - Reads lines, each containing "x,y".
     - Stores coordinates into lists.
     - (Optionally closes the polygon by re-appending the first point).
     - Plots the polygon in a new figure.
     - Saves figure as images/<filename>.png within the same folder.

Requirements:
  pip install matplotlib
"""

import os
import sys
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_polygons.py <folder_with_txt_files>")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid directory.")
        sys.exit(1)

    # Create a subfolder named "images" inside 'folder'
    images_folder = os.path.join(folder, "images")
    os.makedirs(images_folder, exist_ok=True)

    # Loop over all .txt files in the folder
    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue  # skip non-txt files

        txt_path = os.path.join(folder, fname)

        # Read polygon coordinates
        xvals = []
        yvals = []
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # skip empty lines
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        xvals.append(x)
                        yvals.append(y)
                    except ValueError:
                        # skip lines that don't parse as floats
                        pass

        if len(xvals) < 2:
            # Not enough points to plot a polygon
            print(f"Warning: '{fname}' has insufficient points. Skipping.")
            continue

        # Optionally close the polygon by re-appending the first point
        xvals.append(xvals[0])
        yvals.append(yvals[0])

        # Plot
        fig, ax = plt.subplots()
        ax.plot(xvals, yvals, 'o-', color='blue')
        ax.set_aspect('equal', 'box')
        ax.set_title(fname)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Save figure to images/ subfolder with the same base name
        base, _ = os.path.splitext(fname)
        out_png = os.path.join(images_folder, base + ".png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

        print(f"Plotted '{fname}' -> '{out_png}'")

if __name__ == "__main__":
    main()

