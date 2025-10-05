import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import logging
import os
from tqdm import tqdm
import copy

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_SAVE_PATH = "humanity_model_resnet_v4.pth" # Saved new version
# --- FIX: Corrected typo in variable name ---
TRAIN_H5_PATH = "train_humanity.h5" 
VALIDATION_H5_PATH = "validation_humanity.h5"

# --- Hyperparameters ---
LEARNING_RATE = 0.0005      # Slightly increased LR to work with ReduceLROnPlateau
WEIGHT_DECAY = 1e-5         
BATCH_SIZE = 256
NUM_EPOCHS = 30             # Increased epochs for the new scheduler
EARLY_STOPPING_PATIENCE = 5 # Increased patience slightly

# --- MODIFICATION: Exponentially Weighted Loss Configuration ---
# This scaling factor heavily penalizes errors on high-divergence samples.
# weight = 1.0 + exp(target_score * WEIGHT_SCALING_FACTOR)
WEIGHT_SCALING_FACTOR = 3.0 

# --- Custom PyTorch Dataset for the Humanity Model ---
class HumanityDataset(Dataset):
    """
    Loads the HDF5 dataset with board tensors (inputs) and 
    the continuous Humanity Score (outputs) into RAM for fast training.
    """
    def __init__(self, h5_path):
        self.h5_path = h5_path
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Dataset file not found: {h5_path}.")
        
        logging.info(f"Loading '{h5_path}' into RAM...")
        with h5py.File(self.h5_path, 'r') as f:
            self.inputs_mem = f['inputs'][:]
            self.outputs_mem = f['outputs'][:]
        self.length = len(self.inputs_mem)
        logging.info("Dataset loaded into RAM successfully.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(self.inputs_mem[idx].astype(np.float32))
        humanity_score = torch.tensor(self.outputs_mem[idx], dtype=torch.float32)
        return input_tensor, humanity_score

# --- Humanity Model Architecture ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class HumanityResNet(nn.Module):
    """
    Uses the same powerful ResNet body as the Personality Model, but attaches a new
    Regression Head to predict the single, continuous Humanity Score.
    """
    def __init__(self, in_channels=16, num_blocks=8, num_filters=128):
        super(HumanityResNet, self).__init__()
        # The "Body" - a powerful feature extractor
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        self.residual_tower = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_blocks)])
        
        # The "Head" - a regressor to predict the Humanity Score
        self.regression_head = nn.Sequential(
            nn.Conv2d(num_filters, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5), # Dropout layer for regularization
            nn.Linear(256, 1),
            nn.Sigmoid() # Crucial: Ensures the output is always between 0.0 and 1.0
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.residual_tower(x)
        humanity_score = self.regression_head(x)
        return humanity_score

# --- MODIFICATION: Custom Exponentially Weighted Loss Function ---
def weighted_mse_loss(predictions, targets, weight_factor):
    """
    Calculates MSE loss with weights that grow exponentially with the target values.
    This forces the model to pay much more attention to rare, high-divergence samples.
    """
    # New exponential weighting to aggressively penalize errors on high-score samples
    weights = 1.0 + torch.exp(targets * weight_factor)
    squared_errors = (predictions - targets) ** 2
    weighted_squared_errors = weights * squared_errors
    return torch.mean(weighted_squared_errors)

# --- MODIFICATION: Helper Function for Binned Accuracy Metric ---
def get_binned_accuracy(predictions, targets):
    """Calculates the accuracy based on more balanced bins."""
    def get_bin(score):
        if score < 0.1: return 0  # Low divergence
        if score < 0.4: return 1  # Medium divergence
        return 2                  # High divergence

    pred_bins = np.vectorize(get_bin)(predictions.cpu().numpy())
    target_bins = np.vectorize(get_bin)(targets.cpu().numpy())
    
    correct = np.sum(pred_bins == target_bins)
    total = len(pred_bins)
    return 100 * correct / total if total > 0 else 0

# --- Main Training Function ---
def train():
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")
    
    num_workers = 0 if device.type == 'mps' else 4

    # --- 1. Load Data ---
    try:
        train_dataset = HumanityDataset(TRAIN_H5_PATH)
        val_dataset = HumanityDataset(VALIDATION_H5_PATH)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=num_workers)
    except FileNotFoundError as e:
        logging.critical(f"ERROR: Dataset file not found. {e}\nNOTE: You may need to split 'humanity_dataset.h5' into train/validation sets first.")
        return

    # --- 2. Initialize Model, Loss, and Optimizer ---
    model = HumanityResNet().to(device)
    # criterion is now called inside the loop
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # --- FIX: Removed 'verbose' argument which is not supported in all torch versions ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # --- Variables for Early Stopping logic ---
    best_val_mae = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    # --- 3. Training & Validation Loop ---
    logging.info("Starting training with EXPONENTIALLY-WEIGHTED LOSS...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", unit="batch")
        
        for inputs, targets in train_progress_bar:
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = weighted_mse_loss(predictions, targets, WEIGHT_SCALING_FACTOR)
            loss.backward()
            optimizer.step()
            
            train_progress_bar.set_postfix(loss=loss.item())
        
        # --- Validation Loop ---
        model.eval()
        total_mae = 0.0
        all_preds, all_targets = [], []
        
        val_progress_bar = tqdm(validation_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", unit="batch")
        with torch.no_grad():
            for inputs, targets in val_progress_bar:
                inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                predictions = model(inputs)
                
                mae = torch.abs(predictions - targets).mean()
                total_mae += mae.item()
                
                all_preds.append(predictions)
                all_targets.append(targets)

        # --- Calculate and Report Metrics ---
        avg_mae = total_mae / len(validation_loader)
        all_preds_tensor = torch.cat(all_preds)
        all_targets_tensor = torch.cat(all_targets)
        binned_acc = get_binned_accuracy(all_preds_tensor, all_targets_tensor)
        
        logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Summary | "
                     f"Validation MAE: {avg_mae:.4f} | "
                     f"Binned Accuracy (New Bins): {binned_acc:.2f}%")
        
        scheduler.step(avg_mae)

        # --- Early Stopping Logic ---
        if avg_mae < best_val_mae:
            best_val_mae = avg_mae
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            logging.info(f"  -> New best validation MAE: {best_val_mae:.4f}. Saving model state.")
        else:
            epochs_no_improve += 1
            logging.info(f"  -> Validation MAE did not improve. Patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            logging.info(f"Stopping early. Validation MAE has not improved for {EARLY_STOPPING_PATIENCE} epochs.")
            break
            
    logging.info("Finished Training.")
    
    # --- 4. Save the Model ---
    if best_model_state:
        torch.save(best_model_state, MODEL_SAVE_PATH)
        logging.info(f"Best Humanity Model saved to '{MODEL_SAVE_PATH}' with MAE: {best_val_mae:.4f}")
    else:
        logging.warning("No best model state was saved. This can happen if training is interrupted before the first epoch completes.")


if __name__ == "__main__":
    train()

