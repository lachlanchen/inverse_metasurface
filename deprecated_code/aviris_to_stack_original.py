import os
import numpy as np
import spectral
import cv2

def extract_original_wavelength_images(hdr_file, output_dir):
    """
    Read the original AVIRIS hyperspectral data file and save each wavelength band
    as a full-resolution grayscale image without resizing.
    
    Parameters:
    -----------
    hdr_file : str
        Path to the ENVI header file
    output_dir : str
        Directory where to save the output images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Open the hyperspectral image
        print(f"Opening {hdr_file}...")
        img = spectral.open_image(hdr_file)
        data = img.load()  # shape: (lines, samples, bands)
        
        # Get wavelength information (assumed to be in nm)
        wavelengths_nm = np.array(img.bands.centers)
        wavelengths_um = wavelengths_nm / 1000.0  # Convert to micrometers
        
        print(f"Data shape: {data.shape}")
        print(f"Number of wavelength bands: {len(wavelengths_um)}")
        print(f"Wavelength range: {wavelengths_um.min():.4f} - {wavelengths_um.max():.4f} µm")
        
        # Calculate global min/max across ALL bands for consistent scaling
        global_min = np.min(data)
        global_max = np.max(data)
        print(f"Global data range: {global_min:.4f} - {global_max:.4f}")
        
        # For each wavelength channel
        for i, wavelength in enumerate(wavelengths_um):
            # Extract the channel
            channel = data[:, :, i].astype(np.float32)
            
            # Scale to 0-255 range using global min/max
            # This preserves the relative brightness between different wavelengths
            scaled = 255 * (channel - global_min) / (global_max - global_min)
            
            # Convert to 8-bit image
            image_8bit = np.clip(scaled, 0, 255).astype(np.uint8)
            
            # Save grayscale image
            output_file = os.path.join(output_dir, f"band_{i:03d}_{wavelength:.4f}um.png")
            cv2.imwrite(output_file, image_8bit)
            
            # Progress indicator
            if i % 10 == 0 or i == len(wavelengths_um) - 1:
                print(f"Processed {i+1}/{len(wavelengths_um)} bands...")
        
        print(f"All wavelength bands processed and saved to {output_dir}")
        
    except Exception as e:
        print(f"Error processing {hdr_file}: {str(e)}")

def main():
    # Base directory for original AVIRIS data
    base_dir = 'AVIRIS'
    
    # Walk through the AVIRIS folder recursively
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Process only ENVI header files
            if file.lower().endswith('.hdr'):
                hdr_file = os.path.join(root, file)
                
                # Create output directory for wavelength images
                parent_dir = os.path.dirname(hdr_file)
                output_dir = os.path.join(parent_dir, 'fullres_wavelength_images')
                
                print(f"\nProcessing {hdr_file}...")
                extract_original_wavelength_images(hdr_file, output_dir)

if __name__ == '__main__':
    main()
