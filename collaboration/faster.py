import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# Load data
data_path = 'merged_s4_shapes_20250114_175110.csv'
data = pd.read_csv(data_path)

# Data preprocessing
def preprocess_data(data):
    # Extract feature columns
    feature_cols = [col for col in data.columns if '@' in col]
    target_col = data.columns[-1]

    # Group by shape_idx and aggregate features
    grouped = data.groupby(['NQ','shape_idx'])
    processed_data = []

    for shape_idx, group in grouped:
        features = group[feature_cols].values.reshape(-1)
        target = group[target_col].iloc[0]

        # Process target: keep only positive coordinates and sort them
        points = target.split(';')
        filtered_points = [
            list(map(float, point.split(',')))
            for point in points
            if all(float(coord) > 0 for coord in point.split(','))
        ]
        sorted_points = sorted(filtered_points, key=lambda x: (x[0], x[1]))

        processed_data.append({
            'features': features,
            'targets': sorted_points[:4]  # Keep up to 4 points
        })

    return processed_data

processed_data = preprocess_data(data)

# Split data into training and testing sets
train_data, test_data = train_test_split(processed_data, test_size=0.02, random_state=42)

# Define Dataset class
class ShapeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        features = torch.tensor(sample['features'], dtype=torch.float32)

        # Pad targets to fixed length (4 points)
        targets = sample['targets']
        coords = torch.zeros((4, 2))
        flags = torch.zeros(4)

        for i, point in enumerate(targets):
            coords[i] = torch.tensor(point)
            flags[i] = 1
        # print(targets)
        # print(sum(flags))

        return features, coords, flags

# Prepare DataLoader
train_dataset = ShapeDataset(train_data)
test_dataset = ShapeDataset(test_data)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define Transformer Model
class ShapePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(ShapePredictor, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8), num_layers=4
        )
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.coord_head = nn.Linear(hidden_dim, 8)  # 4 points x 2 coords
        self.flag_head = nn.Linear(hidden_dim, 4)   # 4 flags

    def forward(self, x):
        x = self.fc(x).unsqueeze(0)  # Add sequence dimension
        x = self.encoder(x).squeeze(0)
        coords = self.coord_head(x).view(-1, 4, 2)  # Reshape to (batch_size, 4, 2)
        flags = torch.sigmoid(self.flag_head(x))
        return coords, flags

# Instantiate model
input_dim = len(train_dataset[0][0])
model = ShapePredictor(input_dim=input_dim)

# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss and optimizer
criterion_coords = nn.MSELoss()
criterion_flags = nn.BCELoss()
def loss_valid_count(pred_flags, true_flags):
    pred_count = torch.sum(torch.round(pred_flags))
    true_count = torch.sum(true_flags)
    return (pred_count - true_count).abs()

def regularization_invalid(pred_flags, pred_coords):
    invalid_flags = 1 - pred_flags  # Focus on invalid points (flag=0)
    penalty = torch.mean((invalid_flags * torch.clamp(pred_coords, max=0)) ** 2)
    return penalty

optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Visualization function
def visualize_predictions(pred_coords, pred_flags, true_coords, out_dir="visualizations-faster", sample_idx=None, epoch=None):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot ground truth
    for i in range(4):
        if true_coords[i, 0] > 0 and true_coords[i, 1] > 0:
            ax.scatter(true_coords[i, 0], true_coords[i, 1], color='blue', label='Ground Truth' if i == 0 else "")
            # ax.scatter(true_coords[i, 0], true_coords[i, 1], color='blue', label='Ground Truth')

    # Plot predictions
    for i in range(4):
        if pred_flags[i] > 0.5 and all(pred_coords[i] > 0):  # Only visualize valid points with positive coords
        # if all(pred_coords[i] > 0):
            ax.scatter(pred_coords[i, 0], pred_coords[i, 1], color='red', marker='x', label='Prediction' if i == 0 else "")
            # ax.scatter(pred_coords[i, 0], pred_coords[i, 1], color='red', marker='x', label='Prediction')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Epoch {epoch} - Sample {sample_idx}')
    ax.legend()
    plt.savefig(f"{out_dir}/epoch_{epoch}_sample_{sample_idx}.png")
    plt.close()

def compute_loss(pred_coords, pred_flags, true_coords, true_flags):
    """
    Compute total loss:
    - MSE for valid points (flag=1).
    - Regularization for invalid points (flag=0).
    """
    # Separate valid and invalid points
    valid_mask = true_flags > 0.5  # Mask for valid points
    invalid_mask = ~valid_mask  # Mask for invalid points

    # Loss for valid points
    if valid_mask.sum() > 0:
        mse_loss = nn.MSELoss()(pred_coords[valid_mask], true_coords[valid_mask])
    else:
        mse_loss = 0.0

    # Regularization for invalid points
    reg_loss = torch.mean((invalid_mask.unsqueeze(-1) * torch.clamp(pred_coords, max=0)) ** 2)

    return mse_loss, reg_loss

# Training loop
save_dir = "model_outputs"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(10000):
    model.train()
    total_loss_coords = 0
    total_loss_flags = 0
    total_loss_count = 0
    total_regularization = 0

    for features, coords, flags in train_dataloader:
        features, coords, flags = features.to(device), coords.to(device), flags.to(device)

        optimizer.zero_grad()

        pred_coords, pred_flags = model(features)

        # Compute losses
        loss_valid, reg_loss = compute_loss(pred_coords, pred_flags, coords, flags)
        loss_flags = criterion_flags(pred_flags, flags)
        loss_count = loss_valid_count(pred_flags, flags)

        # Combine all losses
        loss = loss_valid + reg_loss + loss_flags + loss_count

        loss.backward()
        optimizer.step()

        total_loss_coords += loss_valid.item()
        total_loss_flags += loss_flags.item()
        total_loss_count += loss_count.item()
        total_regularization += reg_loss.item()

    print(f"Epoch {epoch + 1}, Total Loss: {loss.item():.4f}, Coords Loss: {total_loss_coords:.4f}, Flags Loss: {total_loss_flags:.4f}, Count Loss: {total_loss_count:.4f}, Reg Loss: {total_regularization:.4f}")

    # Validation and visualization every 100 epochs
    if (epoch + 1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            for i, (features, coords, flags) in enumerate(test_dataloader):
                features, coords, flags = features.to(device), coords.to(device), flags.to(device)
                pred_coords, pred_flags = model(features)

                # Check if all flags are 1
                if torch.all(pred_flags[0] > 0.5):
                # if 1:
                    visualize_predictions(
                        pred_coords[-1].cpu().numpy(),
                        pred_flags[-1].cpu().numpy(),
                        coords[-1].cpu().numpy(),
                        sample_idx=i,
                        epoch=epoch + 1
                    )

# Save final model
torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
