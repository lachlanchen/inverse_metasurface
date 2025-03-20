import os
import numpy as np
import pandas as pd
import spectral
from scipy.interpolate import interp1d
import shutil

def process_file(hdr_file, partial_wavelengths):
    """
    Process one AVIRIS dataset:
      - Load the hyperspectral data from the given ENVI header.
      - Convert wavelengths from nm to µm.
      - Truncate the data to the range covered by partial_wavelengths.
      - Interpolate each pixel's spectrum to the new wavelengths.
      - Update the ENVI header metadata.
      
    Returns:
      (interpolated_data, new_metadata) or None if no wavelengths fall in range.
    """
    try:
        # Open the hyperspectral image using spectral
        img = spectral.open_image(hdr_file)
        data = img.load()  # shape: (lines, samples, bands)

        # Get original wavelengths (assumed in nm) and convert to µm
        aviris_wavelengths_nm = np.array(img.bands.centers)
        aviris_wavelengths_um = aviris_wavelengths_nm / 1000.0

        print(f"Original data shape: {data.shape}")
        print(f"Original wavelength range: {aviris_wavelengths_um.min():.4f} - {aviris_wavelengths_um.max():.4f} µm")

        # Truncate wavelengths to the range of partial_wavelengths
        min_wl = partial_wavelengths.min()
        max_wl = partial_wavelengths.max()
        mask = (aviris_wavelengths_um >= min_wl) & (aviris_wavelengths_um <= max_wl)
        if not np.any(mask):
            print(f"Warning: No wavelengths in {hdr_file} fall within the partial range ({min_wl:.4f} - {max_wl:.4f} µm). Skipping.")
            return None

        truncated_aviris_wavelengths = aviris_wavelengths_um[mask]
        truncated_data = data[:, :, mask]

        print(f"Truncated data shape: {truncated_data.shape}")
        print(f"Truncated wavelength range: {truncated_aviris_wavelengths.min():.4f} - {truncated_aviris_wavelengths.max():.4f} µm")

        # Interpolate the spectral data to match the wavelengths from the partial CSV.
        height, width, _ = truncated_data.shape
        num_pixels = height * width
        spectra = truncated_data.reshape(num_pixels, -1)
        num_partial = len(partial_wavelengths)
        interp_spectra = np.zeros((num_pixels, num_partial), dtype=spectra.dtype)

        # Perform linear interpolation for each pixel (spectral axis)
        for i in range(num_pixels):
            f = interp1d(truncated_aviris_wavelengths, spectra[i, :],
                         kind='linear', bounds_error=False, fill_value="extrapolate")
            interp_spectra[i, :] = f(partial_wavelengths)
        interpolated_data = interp_spectra.reshape(height, width, num_partial)

        print(f"Interpolated data shape: {interpolated_data.shape}")

        # Read original ENVI header metadata
        hdr_metadata = spectral.envi.read_envi_header(hdr_file)
        
        # Update metadata: update bands count and wavelength info (as a string list in curly braces)
        hdr_metadata['bands'] = str(num_partial)
        hdr_metadata['wavelength'] = "{" + ", ".join([str(w) for w in partial_wavelengths]) + "}"
        hdr_metadata['wavelength units'] = 'um'  # Ensure this is set to micrometers
        hdr_metadata['description'] = hdr_metadata.get('description', '') + ' (Processed to match SHIR wavelengths)'
        
        return interpolated_data, hdr_metadata
        
    except Exception as e:
        print(f"Error processing {hdr_file}: {str(e)}")
        return None

def main():
    # Base directories
    input_base = 'AVIRIS'
    output_base = 'AVIRIS_SHIR'
    os.makedirs(output_base, exist_ok=True)

    # Read the partial CSV file to get the wavelengths (assumed column name "Wavelength_um")
    partial_csv_path = 'partial_crys_data/partial_crys_C0.0.csv'
    partial_df = pd.read_csv(partial_csv_path)
    partial_wavelengths = np.sort(partial_df['Wavelength_um'].values)
    
    print(f"Loaded {len(partial_wavelengths)} wavelengths from {partial_csv_path}")
    print(f"Wavelength range: {partial_wavelengths.min():.4f} - {partial_wavelengths.max():.4f} µm")

    # Walk through the AVIRIS folder recursively
    for root, dirs, files in os.walk(input_base):
        for file in files:
            # Process only ENVI header files (assumed *.hdr)
            if file.lower().endswith('.hdr'):
                hdr_file = os.path.join(root, file)
                
                # Determine relative path to maintain folder structure
                rel_dir = os.path.relpath(root, input_base)
                output_dir = os.path.join(output_base, rel_dir)
                os.makedirs(output_dir, exist_ok=True)
                
                output_hdr_file = os.path.join(output_dir, file)
                
                # Get the corresponding data file (file without .hdr extension)
                data_file = os.path.splitext(hdr_file)[0]
                output_data_file = os.path.splitext(output_hdr_file)[0]
                
                # Skip if output files already exist
                if os.path.exists(output_hdr_file) and os.path.exists(output_data_file):
                    print(f"Output already exists for {hdr_file}, skipping.")
                    continue

                print(f"\nProcessing {hdr_file} ...")
                result = process_file(hdr_file, partial_wavelengths)
                
                if result is None:
                    continue
                    
                interpolated_data, new_metadata = result

                # Save the processed data (ENVI format) to the output directory
                spectral.envi.save_image(output_hdr_file, interpolated_data, metadata=new_metadata, force=True)
                print(f"Saved processed file to {output_hdr_file}")

if __name__ == '__main__':
    main()
