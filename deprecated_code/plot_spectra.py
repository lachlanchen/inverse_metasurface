#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon

def gradient_fill(x, y, ax=None, alpha=0.5, linewidth=2, **plot_kwargs):
    if ax is None:
        ax = plt.gca()
    # Plot the spectrum curve.
    line, = ax.plot(x, y, linewidth=linewidth, **plot_kwargs)
    # Create a horizontal gradient: a 1x256 array varying from 0 to 1.
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    # Define a fixed colormap from blue to green to red.
    bgr_cmap = mcolors.LinearSegmentedColormap.from_list("bgr", ["blue", "green", "red"])
    xmin = x[0]
    xmax = x[-1]
    y0 = 0
    y1 = np.max(y)
    # Draw the gradient image over the region [xmin, xmax] and [y0, y1].
    im = ax.imshow(grad, extent=[xmin, xmax, y0, y1], aspect='auto',
                   origin='lower', cmap=bgr_cmap, alpha=alpha)
    # Create a polygon that covers the area under the curve.
    xy = np.column_stack([x, y])
    xy = np.vstack([[x[-1], y0], [x[0], y0], xy])
    clip_path = Polygon(xy, closed=True, transform=ax.transData)
    im.set_clip_path(clip_path)
    return line, im

def plot_single_spectrum(x, y, index, output_folder="spectra_plots"):
    os.makedirs(output_folder, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    gradient_fill(x, y, ax=ax, alpha=0.5, linewidth=2)
    ax.set_title(f"Spectrum {index}")
    ax.set_xlabel("Normalized Index")
    ax.set_ylabel("Intensity")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, np.max(y)*1.1)
    save_path = os.path.join(output_folder, f"spectrum_{index}.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Spectrum {index} saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Plot each spectrum from an NPZ dataset with a fixed BGR gradient fill under the curve."
    )
    parser.add_argument("--npz_file", type=str, default="data.npz",
                        help="Path to the NPZ file containing the spectra dataset")
    parser.add_argument("--idx", type=int, default=1,
                        help="Index of the group to plot (default: 1)")
    parser.add_argument("--output_folder", type=str, default="spectra_plots",
                        help="Folder where the individual plots will be saved")
    args = parser.parse_args()
    
    data = np.load(args.npz_file)
    if "spectra" in data:
        spectra_all = data["spectra"]
    else:
        spectra_all = data[list(data.keys())[0]]
    
    if spectra_all.ndim == 3:
        group_index = args.idx - 1
        try:
            spectra = spectra_all[group_index, :, :]
        except IndexError:
            raise ValueError(f"Group index {args.idx} is out of range for the dataset with shape {spectra_all.shape}")
    elif spectra_all.ndim == 2:
        spectra = spectra_all
    else:
        raise ValueError("Unexpected data shape in NPZ file. Expecting a 2D or 3D array.")
    
    x = np.linspace(0, 1, spectra.shape[1])
    num_spectra = spectra.shape[0]
    
    for i in range(num_spectra):
        plot_single_spectrum(x, spectra[i, :], index=i+1, output_folder=args.output_folder)

if __name__ == "__main__":
    main()

