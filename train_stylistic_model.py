import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import logging
import os
from tqdm import tqdm
import time

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Final Training Configuration ---
MODEL_SAVE_PATH = "chess_imitator_final.pth"
TRAIN_H5_PATH = "train.h5"
VALIDATION_H5_PATH = "validation.h5"

# --- Winning Hyperparameters from HPO Search ---
LEARNING_RATE = 0.003990
# Effective Batch Size = 2048. Since BASE_BATCH_SIZE is 1024, we need 2 accumulation steps.
ACCUMULATION_STEPS = 2 
BASE_BATCH_SIZE = 1024
NUM_EPOCHS = 20 # A longer run for the final model

# --- Custom PyTorch Dataset (Optimized for In-Memory Speed) ---
class ChessDataset(Dataset):
    """
    Loads the entire HDF5 dataset into RAM for maximum training speed.
    """
    def __init__(self, h5_path, fraction=1.0):
        self.h5_path = h5_path
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Dataset file not found: {h5_path}")

        logging.info(f"Loading '{h5_path}' entirely into RAM...")
        start_time = time.time()
        
        with h5py.File(self.h5_path, 'r') as f:
            total_samples = len(f['inputs'])
            # For the final model, we always load 100% of the data.
            num_to_load = int(total_samples * fraction)
            
            inputs_np = f['inputs'][:num_to_load]
            outputs_raw_np = f['outputs'][:num_to_load]

        # Convert inputs to float32 tensor
        self.inputs_mem = torch.from_numpy(inputs_np.astype(np.float32))
        
        # Pre-calculate 'from' and 'to' targets
        from_targets_np = outputs_raw_np // 64
        to_targets_np = outputs_raw_np % 64
        
        # Convert targets to LongTensors
        self.from_targets_mem = torch.from_numpy(from_targets_np).long()
        self.to_targets_mem = torch.from_numpy(to_targets_np).long()

        self.length = len(self.inputs_mem)
        end_time = time.time()
        logging.info(f"Dataset with {self.length} samples loaded into RAM in {end_time - start_time:.2f} seconds.")


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Data is already in RAM, so this is just a quick tensor slice
        return self.inputs_mem[idx], self.from_targets_mem[idx], self.to_targets_mem[idx]


# --- Model Architecture ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out

class ImitationResNet(nn.Module):
    def __init__(self, in_channels=16, num_blocks=8, num_filters=128):
        super(ImitationResNet, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_filters, num_filters) for _ in range(num_blocks)]
        )

        self.from_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 64)
        )
        self.to_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 64)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.residual_tower(x)
        from_logits = self.from_head(x)
        to_logits = self.to_head(x)
        return from_logits, to_logits

# --- Main Training Function ---
def train_final_model():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.set_float32_matmul_precision('high')
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision('high')
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    num_workers = 0 
    logging.info(f"Using {num_workers} data loader workers for in-memory dataset.")

    # --- 1. Load Full Data ---
    try:
        train_dataset = ChessDataset(TRAIN_H5_PATH, fraction=1.0)
        val_dataset = ChessDataset(VALIDATION_H5_PATH, fraction=1.0)

        train_loader = DataLoader(train_dataset, batch_size=BASE_BATCH_SIZE, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(val_dataset, batch_size=BASE_BATCH_SIZE, num_workers=num_workers)
    except FileNotFoundError as e:
        logging.critical(f"ERROR: Dataset file not found. {e}")
        return

    # --- 2. Initialize Model, Loss, Optimizer & Scheduler ---
    model = ImitationResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Add a learning rate scheduler to improve convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    use_amp = device.type in ['cuda', 'mps']
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    
    best_val_accuracy = 0.0

    # --- 3. Full Training & Validation Loop ---
    logging.info("Starting final model training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", unit="batch")

        optimizer.zero_grad(set_to_none=True)
        for i, (inputs, from_targets, to_targets) in enumerate(train_progress_bar):
            inputs, from_targets, to_targets = inputs.to(device, non_blocking=True), from_targets.to(device, non_blocking=True), to_targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                from_logits, to_logits = model(inputs)
                loss_from = criterion(from_logits, from_targets)
                loss_to = criterion(to_logits, to_targets)
                total_loss = (loss_from + loss_to)

            scaled_loss = total_loss / ACCUMULATION_STEPS
            scaler.scale(scaled_loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += total_loss.item()
            train_progress_bar.set_postfix(loss=total_loss.item())

        # --- Validation Loop ---
        model.eval()
        total_from_correct, total_to_correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for inputs, from_targets, to_targets in validation_loader:
                inputs, from_targets, to_targets = inputs.to(device, non_blocking=True), from_targets.to(device, non_blocking=True), to_targets.to(device, non_blocking=True)
                from_logits, to_logits = model(inputs)
                
                _, from_predicted = torch.max(from_logits.data, 1)
                _, to_predicted = torch.max(to_logits.data, 1)
                
                total_samples += from_targets.size(0)
                total_from_correct += (from_predicted == from_targets).sum().item()
                total_to_correct += (to_predicted == to_targets).sum().item()

        from_accuracy = 100 * total_from_correct / total_samples
        to_accuracy = 100 * total_to_correct / total_samples
        avg_loss = running_loss / len(train_loader)
        total_accuracy = from_accuracy + to_accuracy

        logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Summary | Avg Loss: {avg_loss:.4f} | "
                     f"Val From Acc: {from_accuracy:.2f}% | Val To Acc: {to_accuracy:.2f}%")
        
        # --- 4. Save the Best Model ---
        if total_accuracy > best_val_accuracy:
            best_val_accuracy = total_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logging.info(f"✅ New best model saved to '{MODEL_SAVE_PATH}' with total accuracy: {total_accuracy:.2f}%")
            
        scheduler.step()

    logging.info(f"Finished Training. Best model saved to '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    train_final_model()
