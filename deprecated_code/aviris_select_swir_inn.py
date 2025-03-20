import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import spectral
import os
import pandas as pd
from spectral.io import envi
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import time

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths
base_folder = 'AVIRIS/AV320231008t173943_L2A_OE_main_98b13fff'
img_file = 'AV320231008t173943_L2A_OE_main_98b13fff_RFL_ORT'
hdr_file = f'{base_folder}/{img_file}.hdr'
csv_file = 'partial_crys_data/partial_crys_C0.0.csv'

# Create output directory
output_folder = 'AVIRIS_SWIR_INN'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_img_file = f'{output_folder}/{img_file}'
output_hdr_file = f'{output_folder}/{img_file}.hdr'
output_npy_file = f'{output_folder}/{img_file}.npy'
output_pt_file = f'{output_folder}/{img_file}.pt'
output_model_file = f'{output_folder}/implicit_model.pt'

# Define the neural network architecture
class ImplicitNeuralNetwork(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4):
        super(ImplicitNeuralNetwork, self).__init__()
        
        # Input layer: x, y, wavelength -> hidden_dim
        self.input_layer = nn.Linear(3, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Output layer: hidden_dim -> 1 (reflectance value)
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Activation function
        self.activation = nn.ReLU()
    
    def forward(self, x, y, wavelength):
        # Combine inputs
        x_combined = torch.stack([x, y, wavelength], dim=1)
        
        # Forward pass through the network
        h = self.activation(self.input_layer(x_combined))
        
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        
        # Output layer (no activation - we want raw values)
        output = self.output_layer(h)
        
        return output.squeeze(1)  # Remove the channel dimension


def normalize_coordinates(coords, shape):
    """Normalize x, y coordinates to range [-1, 1]"""
    return 2 * (coords / (shape - 1)) - 1

def normalize_wavelengths(wavelengths, min_wl, max_wl):
    """Normalize wavelengths to range [-1, 1]"""
    return 2 * ((wavelengths - min_wl) / (max_wl - min_wl)) - 1

def denormalize_wavelengths(norm_wavelengths, min_wl, max_wl):
    """Convert normalized wavelengths back to original range"""
    return ((norm_wavelengths + 1) / 2) * (max_wl - min_wl) + min_wl

def train_implicit_network(img, aviris_wavelengths, csv_wavelengths, 
                           batch_size=10000, epochs=10, learning_rate=0.001, 
                           subsample_factor=4):
    """
    Train an implicit neural network on the AVIRIS data.
    subsample_factor: Use every Nth pixel to reduce training data size.
    """
    print("Preparing training data...")
    rows, cols, bands = img.shape
    
    # Subsample the image to reduce training data size
    row_indices = np.arange(0, rows, subsample_factor)
    col_indices = np.arange(0, cols, subsample_factor)
    
    # Create coordinate grids for the subsampled image
    Y, X = np.meshgrid(row_indices, col_indices, indexing='ij')
    coords = np.stack([Y.flatten(), X.flatten()], axis=1)
    
    # Normalize spatial coordinates to [-1, 1]
    norm_x = normalize_coordinates(torch.tensor(coords[:, 1], dtype=torch.float32), cols)
    norm_y = normalize_coordinates(torch.tensor(coords[:, 0], dtype=torch.float32), rows)
    
    # Get min and max wavelengths for normalization
    min_wl = np.min(aviris_wavelengths)
    max_wl = np.max(aviris_wavelengths)
    
    # Create the neural network
    model = ImplicitNeuralNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Calculate total number of training examples
    num_pixels = len(coords)
    print(f"Training with {num_pixels} pixels (subsampled by factor of {subsample_factor})")
    
    # Create a dataset of random samples
    num_batches = (num_pixels * bands) // batch_size + 1
    
    # Pre-load all bands to memory for faster access
    print("Pre-loading bands for faster access...")
    all_bands = []
    for b in tqdm(range(bands), desc="Loading bands"):
        all_bands.append(img.read_band(b))
    
    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0
        start_time = time.time()
        
        # Process data in batches
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}"):
            # Randomly sample pixel coordinates
            pixel_indices = np.random.randint(0, num_pixels, batch_size)
            band_indices_batch = np.random.randint(0, bands, batch_size)
            
            # Get the pixel coordinates and wavelengths for this batch
            x_batch = norm_x[pixel_indices].to(device)
            y_batch = norm_y[pixel_indices].to(device)
            
            # Get the wavelengths for this batch and normalize them
            wavelengths_batch = torch.tensor(
                aviris_wavelengths[band_indices_batch], 
                dtype=torch.float32
            )
            norm_wavelengths = normalize_wavelengths(wavelengths_batch, min_wl, max_wl).to(device)
            
            # Get the actual pixel values for these coordinates
            pixel_values = []
            for i in range(batch_size):
                y_coord = coords[pixel_indices[i], 0]
                x_coord = coords[pixel_indices[i], 1]
                band = band_indices_batch[i]
                
                # Access the pre-loaded band data directly
                pixel_values.append(all_bands[band][y_coord, x_coord])
            
            pixel_values = torch.tensor(pixel_values, dtype=torch.float32).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(x_batch, y_batch, norm_wavelengths)
            
            # Calculate loss
            loss = criterion(predictions, pixel_values)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Calculate average loss for this epoch
        avg_loss = epoch_loss / num_batches
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Time: {elapsed_time:.2f}s")
    
    # Save the trained model
    torch.save(model.state_dict(), output_model_file)
    print(f"Model saved to {output_model_file}")
    
    # Return model and wavelength normalization parameters
    return model, min_wl, max_wl

def sample_network_at_wavelengths(model, min_wl, max_wl, aviris_img, csv_wavelengths):
    """
    Sample the trained network at specific wavelengths from the CSV.
    """
    print("Sampling network at specified wavelengths...")
    rows, cols, _ = aviris_img.shape
    num_wavelengths = len(csv_wavelengths)
    
    # Create output array
    new_data = np.zeros((rows, cols, num_wavelengths), dtype=np.float32)
    
    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    
    # Normalize spatial coordinates to [-1, 1]
    norm_x = normalize_coordinates(torch.tensor(x_coords.flatten(), dtype=torch.float32), cols)
    norm_y = normalize_coordinates(torch.tensor(y_coords.flatten(), dtype=torch.float32), rows)
    
    # Process each wavelength
    for w_idx, wavelength_nm in enumerate(tqdm(csv_wavelengths, desc="Processing wavelengths")):
        # Normalize the wavelength
        norm_wavelength = normalize_wavelengths(
            torch.tensor([wavelength_nm], dtype=torch.float32),
            min_wl, max_wl
        ).expand(rows * cols)
        
        # Process in chunks to avoid memory issues
        chunk_size = 100000  # Adjust based on available memory
        num_chunks = (rows * cols) // chunk_size + 1
        
        predictions = []
        
        with torch.no_grad():
            for chunk in range(num_chunks):
                start_idx = chunk * chunk_size
                end_idx = min((chunk + 1) * chunk_size, rows * cols)
                
                if start_idx >= end_idx:
                    break
                
                # Get the inputs for this chunk
                x_chunk = norm_x[start_idx:end_idx].to(device)
                y_chunk = norm_y[start_idx:end_idx].to(device)
                wavelength_chunk = norm_wavelength[start_idx:end_idx].to(device)
                
                # Get predictions
                pred_chunk = model(x_chunk, y_chunk, wavelength_chunk)
                predictions.append(pred_chunk.cpu().numpy())
        
        # Combine chunks and reshape to image dimensions
        band_predictions = np.concatenate(predictions).reshape(rows, cols)
        new_data[:, :, w_idx] = band_predictions
    
    return new_data

def main():
    # Step 1: Load the AVIRIS data and check the wavelength range
    print(f"Opening AVIRIS image from: {hdr_file}")
    img = spectral.open_image(hdr_file)
    print(f"Image dimensions: {img.shape}")
    print(f"Number of bands: {img.nbands}")
    
    # Get wavelength information
    if hasattr(img, 'bands') and hasattr(img.bands, 'centers'):
        aviris_wavelengths = np.array(img.bands.centers)
        print(f"AVIRIS wavelength range: {np.min(aviris_wavelengths):.2f} to {np.max(aviris_wavelengths):.2f} nm")
        
        # Step 2: Read the wavelengths from the CSV file
        print(f"\nReading wavelengths from CSV: {csv_file}")
        csv_data = pd.read_csv(csv_file)
        csv_wavelengths = csv_data['Wavelength_um'].to_numpy() * 1000  # Convert μm to nm
        print(f"Found {len(csv_wavelengths)} wavelengths in CSV file")
        print(f"CSV wavelength range: {np.min(csv_wavelengths):.2f} to {np.max(csv_wavelengths):.2f} nm")
        
        # Step 3: Train the implicit neural network
        model, min_wl, max_wl = train_implicit_network(
            img, aviris_wavelengths, csv_wavelengths, 
            batch_size=10000*300, epochs=10, subsample_factor=4
        )
        
        # Step 4: Sample the network at the desired wavelengths
        sampled_data = sample_network_at_wavelengths(
            model, min_wl, max_wl, img, csv_wavelengths
        )
        
        # Step 5: Save the results in various formats
        
        # Save as NumPy array
        np.save(output_npy_file, sampled_data)
        print(f"Saved NumPy array to: {output_npy_file}")
        
        # Save as PyTorch tensor
        torch.save(torch.tensor(sampled_data), output_pt_file)
        print(f"Saved PyTorch tensor to: {output_pt_file}")
        
        # Create a new header dictionary based on the original
        original_header = envi.read_envi_header(hdr_file)
        
        # Update the header for the new file
        new_header = original_header.copy()
        new_header['bands'] = len(csv_wavelengths)
        
        # Update wavelength information
        if 'wavelength' in new_header:
            new_header['wavelength'] = [str(wl) for wl in csv_wavelengths]
        
        # Write the new data to an ENVI file
        print(f"Writing sampled data to ENVI file: {output_img_file}")
        envi.save_image(output_hdr_file, sampled_data, metadata=new_header, force=True)
        
        # Check if we need to rename the file to match the original format
        if os.path.exists(f"{output_img_file}.img") and not os.path.exists(f"{base_folder}/{img_file}.img"):
            print(f"Renaming output file to match original format")
            shutil.move(f"{output_img_file}.img", output_img_file)
        
        # Step 6: Create visualizations of the results
        print(f"\nCreating visualizations of selected frames...")
        
        # Load the newly created file to verify it worked
        new_img = spectral.open_image(output_hdr_file)
        
        # Plot specific frames
        frames_to_plot = [0, 24, 49, 99]  # 0-indexed, so these are frames 1, 25, 50, 100
        frames_to_plot = [idx for idx in frames_to_plot if idx < len(csv_wavelengths)]
        
        # Create a figure with consistent scaling
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Calculate consistent scaling across all data
        all_valid_data = sampled_data.flatten()
        p2, p98 = np.percentile(all_valid_data[~np.isnan(all_valid_data)], (2, 98))
        vmin, vmax = p2, p98
        
        print(f"Using consistent scale for visualization: vmin={vmin:.4f}, vmax={vmax:.4f}")
        
        for i, frame_idx in enumerate(frames_to_plot):
            if i < len(axes):
                # Get statistics for this frame
                band_data = new_img.read_band(frame_idx)
                valid_data = band_data[~np.isnan(band_data)]
                
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)
                
                print(f"\nFrame {frame_idx+1} (Wavelength: {csv_wavelengths[frame_idx]:.2f} nm):")
                print(f"  Min: {min_val:.6f}, Max: {max_val:.6f}")
                print(f"  Mean: {mean_val:.6f}, Std: {std_val:.6f}")
                
                # Plot on the corresponding subplot with consistent scale
                im = axes[i].imshow(band_data, cmap='viridis', vmin=vmin, vmax=vmax)
                
                axes[i].set_title(f'Frame {frame_idx+1}: {csv_wavelengths[frame_idx]:.2f} nm '
                                 f'({csv_wavelengths[frame_idx]/1000:.2f} μm)')
                plt.colorbar(im, ax=axes[i], label='Reflectance')
        
        # Hide any unused subplots
        for i in range(len(frames_to_plot), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_folder}/implicit_sampled_frames.png', dpi=300)
        
        # Save the wavelength mapping to a CSV file for reference
        wavelength_mapping = pd.DataFrame({
            'CSV_Index': range(len(csv_wavelengths)),
            'Wavelength_nm': csv_wavelengths,
            'Wavelength_um': csv_wavelengths / 1000
        })
        wavelength_mapping.to_csv(f'{output_folder}/implicit_wavelength_mapping.csv', index=False)
        print(f"Saved wavelength mapping to: {output_folder}/implicit_wavelength_mapping.csv")
        
        print("Processing complete!")
        
    else:
        print("No wavelength information found in the image header.")
        exit(1)

if __name__ == "__main__":
    main()
