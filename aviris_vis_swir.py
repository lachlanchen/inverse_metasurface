import numpy as np
import matplotlib.pyplot as plt
import spectral
import pandas as pd

# Set the path to the folder
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

    # Generate wavelength table
    wavelengths = np.array(img.bands.centers)
    
    # Create a table showing band index and corresponding wavelength
    # Display every 10th band to keep the output manageable
    step = 10
    band_info = []
    for i in range(0, len(wavelengths), step):
        band_info.append({
            'Band Index': i,
            'Wavelength (nm)': wavelengths[i],
            'Wavelength (μm)': wavelengths[i]/1000
        })
    
    # Display as a table
    df = pd.DataFrame(band_info)
    print("\nWavelength correspondence (showing every 10th band):")
    print(df.to_string(index=False))
    
    # Find bands closest to target wavelengths
    target_wavelengths = [1000, 1500, 2000, 2500]  # in nm
    target_bands = []
    
    print("\nTarget wavelength bands:")
    for target_nm in target_wavelengths:
        band_idx = np.abs(wavelengths - target_nm).argmin()
        actual_nm = wavelengths[band_idx]
        target_bands.append({
            'Target (nm)': target_nm,
            'Target (μm)': target_nm/1000,
            'Closest Band': band_idx,
            'Actual (nm)': actual_nm,
            'Actual (μm)': actual_nm/1000
        })
        print(f"Target: {target_nm} nm ({target_nm/1000} μm) → Band {band_idx}: {actual_nm:.2f} nm ({actual_nm/1000:.2f} μm)")
    
    # Create nodata masks and visualize each target wavelength
    nodata_value = -9999.0  # Assuming same nodata value as before
    
    # Create subplots for each target wavelength
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Create a downsampled RGB composite for reference
    red_idx = np.abs(wavelengths - 650).argmin()
    green_idx = np.abs(wavelengths - 550).argmin()
    blue_idx = np.abs(wavelengths - 450).argmin()
    
    print(f"\nCreating RGB composite for reference using bands:")
    print(f"  R={red_idx} ({wavelengths[red_idx]:.2f} nm)")
    print(f"  G={green_idx} ({wavelengths[green_idx]:.2f} nm)")
    print(f"  B={blue_idx} ({wavelengths[blue_idx]:.2f} nm)")
    
    rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    
    for i, idx in enumerate([red_idx, green_idx, blue_idx]):
        band = img.read_band(idx)
        valid_mask = band != nodata_value
        valid_data = band[valid_mask]
        band_mean = np.mean(valid_data)
        rgb[:,:,i] = np.copy(band)
        rgb[~valid_mask,i] = band_mean
    
    # Normalize for display
    for i in range(3):
        p2, p98 = np.percentile(rgb[:,:,i], (2, 98))
        rgb[:,:,i] = np.clip((rgb[:,:,i] - p2) / (p98 - p2) if p98 > p2 else 0, 0, 1)
    
    # Create a reference RGB figure
    plt.figure(figsize=(12, 10))
    plt.imshow(rgb)
    plt.title('RGB Composite (Reference)')
    plt.colorbar(label='Normalized Reflectance')
    
    # Downsample for easier viewing
    downsample_factor = 5
    downsampled_rgb = rgb[::downsample_factor, ::downsample_factor, :]
    
    plt.figure(figsize=(12, 10))
    plt.imshow(downsampled_rgb)
    plt.title('Downsampled RGB Composite (Reference)')
    plt.colorbar(label='Normalized Reflectance')
    
    # Plot each target wavelength
    for i, target in enumerate(target_bands):
        band_idx = target['Closest Band']
        band_data = img.read_band(band_idx)
        
        # Handle nodata values
        valid_mask = band_data != nodata_value
        valid_data = band_data[valid_mask]
        mean_value = np.mean(valid_data)
        
        processed_data = np.copy(band_data)
        processed_data[~valid_mask] = mean_value
        
        # Print statistics for this band
        print(f"\nBand {band_idx} ({wavelengths[band_idx]:.2f} nm, {wavelengths[band_idx]/1000:.2f} μm) statistics:")
        print(f"  Min value: {np.min(valid_data)}")
        print(f"  Max value: {np.max(valid_data)}")
        print(f"  Mean value: {np.mean(valid_data):.4f}")
        print(f"  Std deviation: {np.std(valid_data):.4f}")
        
        # Create normalized version for display
        p2, p98 = np.percentile(processed_data, (2, 98))
        normalized = np.clip((processed_data - p2) / (p98 - p2) if p98 > p2 else 0, 0, 1)
        
        # Plot on the corresponding subplot
        ax = axes[i]
        im = ax.imshow(normalized, cmap='inferno')
        ax.set_title(f'Band {band_idx}: {wavelengths[band_idx]:.2f} nm ({wavelengths[band_idx]/1000:.2f} μm)')
        plt.colorbar(im, ax=ax, label='Normalized Reflectance')
        
        # Create individual figure for each band
        plt.figure(figsize=(12, 10))
        plt.imshow(normalized, cmap='inferno')
        plt.colorbar(label='Normalized Reflectance')
        plt.title(f'Band {band_idx}: {wavelengths[band_idx]:.2f} nm ({wavelengths[band_idx]/1000:.2f} μm)')
        
        # Create histogram of valid values
        plt.figure(figsize=(10, 6))
        plt.hist(valid_data, bins=50)
        plt.title(f'Histogram for Band {band_idx}: {wavelengths[band_idx]:.2f} nm ({wavelengths[band_idx]/1000:.2f} μm)')
        plt.xlabel('Reflectance Value')
        plt.ylabel('Pixel Count')
    
    # Adjust the subplot layout
    fig.tight_layout()
    
    # Create a side-by-side comparison of all four wavelengths for easier comparison
    # Downsample for this comparison to save memory and make it easier to view
    plt.figure(figsize=(16, 12))
    
    # Use a consistent colormap for all four bands for easier comparison
    cmap = 'viridis'
    
    for i, target in enumerate(target_bands):
        band_idx = target['Closest Band']
        band_data = img.read_band(band_idx)
        
        # Handle nodata values
        valid_mask = band_data != nodata_value
        processed_data = np.copy(band_data)
        processed_data[~valid_mask] = np.mean(band_data[valid_mask])
        
        # Downsample
        downsampled = processed_data[::downsample_factor, ::downsample_factor]
        
        # Normalize for display
        p2, p98 = np.percentile(downsampled, (2, 98))
        normalized = np.clip((downsampled - p2) / (p98 - p2) if p98 > p2 else 0, 0, 1)
        
        plt.subplot(2, 2, i+1)
        plt.imshow(normalized, cmap=cmap)
        plt.colorbar(label='Normalized Reflectance')
        plt.title(f'Band {band_idx}: {wavelengths[band_idx]:.2f} nm ({wavelengths[band_idx]/1000:.2f} μm)')
    
    plt.tight_layout()
    plt.show()

else:
    print("No wavelength information found in the image header.")

print("Visualization complete!")
