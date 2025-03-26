import os
import numpy as np
import spectral
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2

def extract_resized_wavelength_images(hdr_file, output_dir, resize_shape=(500, 500)):
    """
    Read a processed AVIRIS hyperspectral data file, extract each wavelength channel,
    resize to the specified dimensions, and save as separate images.
    
    Parameters:
    -----------
    hdr_file : str
        Path to the ENVI header file
    output_dir : str
        Directory where to save the output images
    resize_shape : tuple
        Target image dimensions as (height, width)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the hyperspectral image
    print(f"Opening {hdr_file}...")
    img = spectral.open_image(hdr_file)
    data = img.load()  # shape: (lines, samples, bands)
    
    # Get wavelength information
    wavelengths = np.array(img.bands.centers)
    
    print(f"Data shape: {data.shape}")
    print(f"Number of wavelength bands: {len(wavelengths)}")
    print(f"Wavelength range: {wavelengths.min():.4f} - {wavelengths.max():.4f} µm")
    
    # For each wavelength channel
    for i, wavelength in enumerate(wavelengths):
        # Extract the channel
        channel = data[:, :, i].astype(np.float32)
        
        # Normalize to 0-1 range for visualization
        if np.max(channel) > np.min(channel):
            norm_channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
        else:
            norm_channel = np.zeros_like(channel)
        
        # Resize to target dimensions
        resized = resize(norm_channel, resize_shape, anti_aliasing=True)
        
        # Convert to 8-bit image (0-255)
        image_8bit = (resized * 255).astype(np.uint8)
        
        # Save grayscale version
        output_file = os.path.join(output_dir, f"band_{i:03d}_{wavelength:.4f}um_gray.png")
        cv2.imwrite(output_file, image_8bit)
        
        # Create pseudocolor version using matplotlib
        plt.figure(figsize=(6, 6))
        plt.imshow(resized, cmap='viridis')
        plt.colorbar(label='Normalized Reflectance')
        plt.title(f"Band {i}: {wavelength:.4f} µm")
        plt.axis('off')
        
        # Save pseudocolor version
        output_file_color = os.path.join(output_dir, f"band_{i:03d}_{wavelength:.4f}um_color.png")
        plt.savefig(output_file_color, bbox_inches='tight', dpi=100)
        plt.close()
        
        if i % 10 == 0:
            print(f"Processed {i+1}/{len(wavelengths)} bands...")
    
    print(f"All {len(wavelengths)} wavelength bands processed and saved to {output_dir}")

    # Create a sample RGB composite using three bands (near beginning, middle, and end)
    if len(wavelengths) >= 3:
        try:
            idx1 = 0  # First band
            idx2 = len(wavelengths) // 2  # Middle band
            idx3 = len(wavelengths) - 1  # Last band
            
            # Extract original sized data for each channel
            r_channel = data[:, :, idx1].astype(np.float32)
            g_channel = data[:, :, idx2].astype(np.float32)
            b_channel = data[:, :, idx3].astype(np.float32)
            
            # Normalize each channel individually
            if np.max(r_channel) > np.min(r_channel):
                r_norm = (r_channel - np.min(r_channel)) / (np.max(r_channel) - np.min(r_channel))
            else:
                r_norm = np.zeros_like(r_channel)
                
            if np.max(g_channel) > np.min(g_channel):
                g_norm = (g_channel - np.min(g_channel)) / (np.max(g_channel) - np.min(g_channel))
            else:
                g_norm = np.zeros_like(g_channel)
                
            if np.max(b_channel) > np.min(b_channel):
                b_norm = (b_channel - np.min(b_channel)) / (np.max(b_channel) - np.min(b_channel))
            else:
                b_norm = np.zeros_like(b_channel)
            
            # Stack to create RGB
            rgb = np.stack([r_norm, g_norm, b_norm], axis=2)
            
            # Resize the stacked RGB image
            rgb_resized = resize(rgb, resize_shape + (3,), anti_aliasing=True)
            
            # Ensure proper shape and range for display
            rgb_display = np.clip(rgb_resized, 0, 1)
            
            # Save RGB composite using matplotlib
            plt.figure(figsize=(8, 8))
            plt.imshow(rgb_display)
            plt.title(f"RGB Composite ({wavelengths[idx1]:.4f}, {wavelengths[idx2]:.4f}, {wavelengths[idx3]:.4f} µm)")
            plt.axis('off')
            
            rgb_file = os.path.join(output_dir, "rgb_composite.png")
            plt.savefig(rgb_file, bbox_inches='tight', dpi=100)
            plt.close()
            
            # Also save as OpenCV image
            rgb_cv = (rgb_display[:, :, ::-1] * 255).astype(np.uint8)  # BGR for OpenCV
            cv2_rgb_file = os.path.join(output_dir, "rgb_composite_cv.png")
            cv2.imwrite(cv2_rgb_file, rgb_cv)
            
            print(f"RGB composite saved to {rgb_file}")
            
        except Exception as e:
            print(f"Error creating RGB composite: {str(e)}")
            print("Continuing with individual band processing...")

def main():
    # Base directory for processed AVIRIS data
    base_dir = 'AVIRIS_SHIR'
    
    # Walk through the processed AVIRIS folder recursively
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Process only ENVI header files
            if file.lower().endswith('.hdr'):
                hdr_file = os.path.join(root, file)
                
                # Create output directory for wavelength images (inside the same folder as the HDR)
                output_dir = os.path.join(root, os.path.splitext(file)[0] + '_wavelength_images')
                
                print(f"\nProcessing {hdr_file}...")
                extract_resized_wavelength_images(hdr_file, output_dir)

if __name__ == '__main__':
    main()
