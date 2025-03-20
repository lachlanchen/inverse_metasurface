import numpy as np
import matplotlib.pyplot as plt
import spectral
from spectral import envi
import spectral.io.envi as envi
import os

# Set the path to the new folder
base_folder = 'AVIRIS/AV320231008t173943_L2A_OE_main_98b13fff'
img_file = 'AV320231008t173943_L2A_OE_main_98b13fff_RFL_ORT'
hdr_file = f'{base_folder}/{img_file}.hdr'

print(f"Opening AVIRIS image from: {hdr_file}")
img = spectral.open_image(hdr_file)

# Print basic information about the dataset
print(f"Image dimensions: {img.shape}")
print(f"Number of bands: {img.nbands}")

if hasattr(img, 'bands') and hasattr(img.bands, 'centers'):
    print(f"Wavelengths available: {len(img.bands.centers)}")
    print(f"Wavelength range: {min(img.bands.centers)} to {max(img.bands.centers)} nm")

# Choose a band in the green spectrum
wavelengths = np.array(img.bands.centers)
target_wavelength = 550  # nm (green light)
band_idx = np.abs(wavelengths - target_wavelength).argmin()
print(f"Using band {band_idx} with wavelength {wavelengths[band_idx]:.2f} nm")

# Load the entire band data
print("Loading full image band data...")
band_data = img.read_band(band_idx)

# Find nodata values (assuming -9999.0 as before, adjust if different)
nodata_value = -9999.0
valid_mask = band_data != nodata_value
valid_data = band_data[valid_mask]

# Calculate stats of valid data
print(f"Valid data statistics:")
print(f"  Min value: {np.min(valid_data)}")
print(f"  Max value: {np.max(valid_data)}")
print(f"  Mean value: {np.mean(valid_data):.4f}")
print(f"  Std deviation: {np.std(valid_data):.4f}")
print(f"  Number of valid pixels: {np.sum(valid_mask)}")
print(f"  Number of nodata pixels: {np.sum(~valid_mask)}")

# Replace nodata values with the mean of valid values
mean_value = np.mean(valid_data)
print(f"Replacing nodata values with the mean: {mean_value:.4f}")
processed_data = np.copy(band_data)
processed_data[~valid_mask] = mean_value

# Plot the entire processed image
plt.figure(figsize=(12, 10))
plt.imshow(processed_data, cmap='viridis')
plt.colorbar(label='Reflectance Value')
plt.title(f'AVIRIS Band {band_idx} ({wavelengths[band_idx]:.2f} nm)')
plt.xlabel('Column')
plt.ylabel('Row')

# Add a histogram of valid values only
plt.figure(figsize=(10, 6))
plt.hist(valid_data, bins=50)
plt.title(f'Histogram of Band {band_idx} ({wavelengths[band_idx]:.2f} nm) - Valid Values Only')
plt.xlabel('Reflectance Value')
plt.ylabel('Pixel Count')

# Create a downsampled version for easier viewing
print("Creating downsampled version for easier viewing...")
downsample_factor = 10  # Adjust as needed
rows, cols = band_data.shape
downsampled = processed_data[::downsample_factor, ::downsample_factor]

plt.figure(figsize=(12, 10))
plt.imshow(downsampled, cmap='viridis')
plt.colorbar(label='Reflectance Value')
plt.title(f'Downsampled AVIRIS Band {band_idx} ({wavelengths[band_idx]:.2f} nm)')
plt.xlabel('Column')
plt.ylabel('Row')

# Try to visualize an RGB composite using bands close to R, G, B wavelengths
try:
    print("Attempting to create RGB composite...")
    red_idx = np.abs(wavelengths - 650).argmin()  # Red ~650nm
    green_idx = np.abs(wavelengths - 550).argmin()  # Green ~550nm
    blue_idx = np.abs(wavelengths - 450).argmin()  # Blue ~450nm
    
    print(f"Using bands: R={red_idx} ({wavelengths[red_idx]:.2f}nm), "
          f"G={green_idx} ({wavelengths[green_idx]:.2f}nm), "
          f"B={blue_idx} ({wavelengths[blue_idx]:.2f}nm)")
    
    # Create RGB array
    rgb = np.zeros((band_data.shape[0], band_data.shape[1], 3), dtype=np.float32)
    
    # Read each band and handle nodata values
    for i, idx in enumerate([red_idx, green_idx, blue_idx]):
        band = img.read_band(idx)
        # Replace nodata values with mean
        valid_band_mask = band != nodata_value
        valid_band_data = band[valid_band_mask]
        band_mean = np.mean(valid_band_data)
        processed_band = np.copy(band)
        processed_band[~valid_band_mask] = band_mean
        rgb[:,:,i] = processed_band
    
    # Normalize each channel for display
    for i in range(3):
        p2, p98 = np.percentile(rgb[:,:,i], (2, 98))
        rgb[:,:,i] = np.clip((rgb[:,:,i] - p2) / (p98 - p2) if p98 > p2 else 0, 0, 1)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(rgb)
    plt.title('AVIRIS RGB Composite')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Create downsampled RGB
    downsampled_rgb = rgb[::downsample_factor, ::downsample_factor, :]
    plt.figure(figsize=(12, 10))
    plt.imshow(downsampled_rgb)
    plt.title('Downsampled AVIRIS RGB Composite')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
except Exception as e:
    print(f"Error creating RGB composite: {e}")

plt.tight_layout()
plt.show()

print("Visualization complete!")
