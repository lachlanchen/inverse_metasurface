import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the ShapeToSpectraModel class
class ShapeToSpectraModel(nn.Module):
    def __init__(self, d_in=3, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, 11 * 100)
        )
    def forward(self, shape_4x3):
        bsz = shape_4x3.size(0)
        presence = shape_4x3[:, :, 0]
        key_padding_mask = (presence < 0.5)
        x_proj = self.input_proj(shape_4x3)
        x_enc = self.encoder(x_proj, src_key_padding_mask=key_padding_mask)
        pres_sum = presence.sum(dim=1, keepdim=True) + 1e-8
        x_enc_w = x_enc * presence.unsqueeze(-1)
        shape_emb = x_enc_w.sum(dim=1) / pres_sum
        out_flat = self.mlp(shape_emb)
        out_2d = out_flat.view(bsz, 11, 100)
        return out_2d

# Load the model and run it with a test shape
def test_shape2spec():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_path = "outputs_three_stage_20250216_180408/stageA/shape2spec_stageA.pt"
    model = ShapeToSpectraModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create a random shape in Q1 (1-4 points)
    max_points = 4
    shape = np.zeros((max_points, 3), dtype=np.float32)
    
    # Let's say we want 2 points active
    num_points = 2
    for i in range(num_points):
        shape[i, 0] = 1.0  # Point is present
        shape[i, 1] = np.random.uniform(0.1, 0.5)  # x-coordinate in Q1
        shape[i, 2] = np.random.uniform(0.1, 0.5)  # y-coordinate in Q1
    
    print("Input shape:")
    print(shape)
    
    # Convert to tensor and add batch dimension
    shape_tensor = torch.tensor(shape, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get spectrum prediction
    with torch.no_grad():
        spectrum = model(shape_tensor).cpu().numpy()[0]
    
    print(f"Output spectrum shape: {spectrum.shape}")  # Should be (11, 100)
    
    # Plot the spectrum
    plt.figure(figsize=(10, 6))
    for i, row in enumerate(spectrum):
        plt.plot(row, label=f'c={i}')
    plt.xlabel('Wavelength index')
    plt.ylabel('Reflectance')
    plt.title('Predicted Spectrum')
    plt.legend()
    plt.grid(True)
    plt.savefig('predicted_spectrum.png')
    print("Spectrum plot saved to 'predicted_spectrum.png'")
    
    return spectrum

if __name__ == "__main__":
    test_shape2spec()
