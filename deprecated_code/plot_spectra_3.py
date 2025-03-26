#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon

def gradient_fill(ax, x, y, alpha=0.5, linewidth=1.5):
    """
    Plot a spectrum curve with a fixed BGR gradient fill under the curve on the given axes.
    """
    # Plot the spectrum curve in black.
    ax.plot(x, y, color="black", linewidth=linewidth)
    
    # Create a horizontal gradient image (1x256 pixels).
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    # Define a fixed colormap that goes from blue to green to red.
    bgr_cmap = mcolors.LinearSegmentedColormap.from_list("bgr", ["blue", "green", "red"])
    
    # Set the extent of the gradient image.
    xmin, xmax = x[0], x[-1]
    y0 = 0
    y1 = np.max(y)
    im = ax.imshow(grad, extent=[xmin, xmax, y0, y1], aspect="auto",
                   origin="lower", cmap=bgr_cmap, alpha=alpha)
    
    # Define a polygon for clipping: the area under the curve.
    xy = np.column_stack([x, y])
    xy = np.vstack([[x[-1], y0], [x[0], y0], xy])
    clip_path = Polygon(xy, closed=True, transform=ax.transData)
    im.set_clip_path(clip_path)

def plot_single_spectrum(x, y, index, output_folder="spectra_plots/subfolder"):
    """
    Plot a single spectrum as a figure with size (11,2) inches.
    Removes axes and texts, and saves the figure to the output folder with a transparent background.
    """
    os.makedirs(output_folder, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 2))
    
    gradient_fill(ax, x, y, alpha=0.5, linewidth=1.5)
    
    # Remove all axes, ticks, and labels.
    ax.axis("off")
    
    save_path = os.path.join(output_folder, f"spectrum_{index}.png")
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    print(f"Spectrum {index} saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Plot each of 11 spectra as an individual 11x2 inch figure (transparent bg) saved into a subfolder of spectra_plots."
    )
    parser.add_argument("--npz_file", type=str, default="data.npz",
                        help="Path to the NPZ file containing the spectra dataset")
    parser.add_argument("--idx", type=int, default=1,
                        help="Group index to plot (default: 1)")
    parser.add_argument("--output_folder", type=str, default="spectra_plots/11x2",
                        help="Subfolder (relative to current folder) where the plots will be saved")
    args = parser.parse_args()
    
    # Load the NPZ file.
    data = np.load(args.npz_file)
    if "spectra" in data:
        spectra_all = data["spectra"]
    else:
        # If key "spectra" is not found, use the first array.
        spectra_all = data[list(data.keys())[0]]
    
    # If the data is a 3D array, select the group specified by --idx.
    if spectra_all.ndim == 3:
        group_index = args.idx - 1  # convert to zero-indexed.
        try:
            spectra = spectra_all[group_index, :, :]
        except IndexError:
            raise ValueError(f"Group index {args.idx} is out of range for data with shape {spectra_all.shape}")
    elif spectra_all.ndim == 2:
        spectra = spectra_all
    else:
        raise ValueError("Unexpected data shape in NPZ file. Expecting a 2D or 3D array.")
    
    # Ensure we have 11 spectra.
    if spectra.shape[0] != 11:
        raise ValueError(f"Expected 11 spectra, but got {spectra.shape[0]}")
    
    # Create a common x-axis (normalized to [0, 1]).
    x = np.linspace(0, 1, spectra.shape[1])
    
    # Plot each of the 11 spectra as a separate image.
    for i in range(spectra.shape[0]):
        plot_single_spectrum(x, spectra[i, :], index=i+1, output_folder=args.output_folder+"-".args.idx)

if __name__ == "__main__":
    main()

