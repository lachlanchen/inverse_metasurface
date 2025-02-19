#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon

def gradient_fill(ax, x, y, alpha=0.5, linewidth=1.5):
    """
    Plot the spectrum as a black line and fill the area under the curve
    with a custom color gradient representing wavelengths from 1 µm (navy)
    to 2.5 µm (red) via blue, cyan, green, yellow, and orange.
    """
    # Plot the spectrum curve in black.
    ax.plot(x, y, color="black", linewidth=linewidth)
    
    # Create a horizontal gradient array.
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    
    # Define a custom colormap.
    # This colormap is designed to roughly represent a progression from 1 µm to 2.5 µm.
    wavelength_cmap = mcolors.LinearSegmentedColormap.from_list(
        "wavelength", ["navy", "blue", "cyan", "green", "yellow", "orange", "red"]
    )
    
    # Determine the extent for the gradient image.
    xmin, xmax = x[0], x[-1]
    y0 = 0
    y1 = np.max(y)
    
    # Plot the gradient image.
    im = ax.imshow(grad, extent=[xmin, xmax, y0, y1], aspect="auto",
                   origin="lower", cmap=wavelength_cmap, alpha=alpha)
    
    # Create a polygon corresponding to the area under the curve.
    xy = np.column_stack([x, y])
    xy = np.vstack([[x[-1], y0], [x[0], y0], xy])
    clip_path = Polygon(xy, closed=True, transform=ax.transData)
    im.set_clip_path(clip_path)

def plot_single_spectrum(x, y, index, output_folder):
    """
    Plot a single spectrum in a figure 11 inches wide by 2 inches tall.
    Remove all texts and axes, then save into the specified output folder.
    """
    fig, ax = plt.subplots(figsize=(11, 2))
    gradient_fill(ax, x, y, alpha=0.5, linewidth=1.5)
    
    # Remove axes, ticks, and labels.
    ax.axis("off")
    
    # Save the figure.
    save_path = os.path.join(output_folder, f"spectrum_{index}.png")
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Spectrum {index} saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Plot 11 spectra (each 11x2 inches) with a wavelength (1 µm to 2.5 µm) color gradient under the curve, and save into a subfolder."
    )
    parser.add_argument("--npz_file", type=str, default="data.npz",
                        help="Path to the NPZ file containing the spectra dataset")
    parser.add_argument("--idx", type=int, default=1,
                        help="Group index to plot (default: 1)")
    parser.add_argument("--output_subfolder", type=str, default="wavelength_gradient",
                        help="Subfolder (relative to current folder) where the images will be saved")
    args = parser.parse_args()
    
    # Load the NPZ file.
    data = np.load(args.npz_file)
    if "spectra" in data:
        spectra_all = data["spectra"]
    else:
        spectra_all = data[list(data.keys())[0]]
    
    # If the data is 3D, select the specified group; if 2D, use it directly.
    if spectra_all.ndim == 3:
        group_index = args.idx - 1  # convert to zero-indexed
        try:
            spectra = spectra_all[group_index, :, :]
        except IndexError:
            raise ValueError(f"Group index {args.idx} is out of range for data with shape {spectra_all.shape}")
    elif spectra_all.ndim == 2:
        spectra = spectra_all
    else:
        raise ValueError("Unexpected data shape in NPZ file. Expecting a 2D or 3D array.")
    
    if spectra.shape[0] != 11:
        raise ValueError(f"Expected 11 spectra, but got {spectra.shape[0]}")
    
    # Create output subfolder relative to the current working directory.
    output_folder = os.path.join(os.getcwd(), args.output_subfolder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a common x-axis (normalized).
    x = np.linspace(0, 1, spectra.shape[1])
    
    # Plot each of the 11 spectra.
    for i in range(spectra.shape[0]):
        plot_single_spectrum(x, spectra[i, :], index=i+1, output_folder=output_folder)

if __name__ == "__main__":
    main()

