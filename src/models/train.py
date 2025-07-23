"""
Train the ResNet18 model. Supports weighted loss, early stopping, checkpointing, hash verification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import hashlib
import os
from src.models.model import get_model

def sha256_checksum(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def train():
    # Load config (stub)
    config = {
        'epochs': 30,
        'batch_size': 32,
        'lr': 1e-4,
        'patience': 5,
        'checkpoint_dir': 'checkpoints/',
        'train_data': 'data/processed/train/',
        'val_data': 'data/processed/val/'
    }
    # TODO: load actual config from YAML

    # TODO: Implement dataset and DataLoader
    train_loader = DataLoader(...)
    val_loader = DataLoader(...)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=device)) # Example weighting
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        # TODO: Training loop
        train_loss = 0.0
        for batch in train_loader:
            # images, labels = batch
            # ...
            pass
        # TODO: Validation loop
        val_loss = 0.0
        # ...
        print(f"Epoch {epoch}: train_loss={train_loss}, val_loss={val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(config['checkpoint_dir'], exist_ok=True)
            ckpt_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
            torch.save(model.state_dict(), ckpt_path)
            hash_val = sha256_checksum(ckpt_path)
            with open(ckpt_path + ".sha256", "w") as f:
                f.write(hash_val)
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print("Early stopping.")
                break
        scheduler.step(val_loss)

if __name__ == "__main__":
    train()
