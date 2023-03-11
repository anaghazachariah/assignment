import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List

class TranscriptionDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        transcription = row["transcription"]
        action = row["action"]
        object = row["object"]
        location = row["location"]

        return {
            "transcription": torch.Tensor(transcription),
            "action": torch.Tensor(action),
            "object": torch.Tensor(object),
            "location": torch.Tensor(location)
        }


import torch
import torch.nn as nn
from typing import Dict

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc(out[:, -1, :])

        return {
            "action": out[:, :3],
            "object": out[:, 3:6],
            "location": out[:, 6:]
        }


import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


import logging
import os

def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    return logger

def train(config):
    # Setup logging
    logger = setup_logging(config["training"]["log_dir"])

    # Set device (GPU or CPU)
    device = get_device(config["training"]["device"])

    # Load data
    train_dataset = TranscriptionDataset(config["training"]["train_csv"])
    valid_dataset = TranscriptionDataset(config["training"]["valid_csv"])
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["testing"]["batch_size"], shuffle=False)

    # Load model
    model = LSTMModel(
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        bidirectional=config["model"]["bidirectional"],
        dropout=config["model"]["dropout"],
        output_dim=config["model"]["output_dim"],
    ).to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["optimizer"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Train loop
    for epoch in range(1, config["training"]["epochs"] + 1):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = batch["transcription"].to(device)
            targets = batch["target"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(inputs)
        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                inputs = batch["transcription"].to(device)
                targets = batch["target"].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item() * len(inputs)
        valid_loss /= len(valid_dataset)

        # Log epoch statistics
        logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f}")

        # Save model checkpoint
        checkpoint_dir = config["training"]["checkpoint_dir"]
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)

        # Save tensorboard log
        log_dir = config["training"]["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("valid_loss", valid_loss, epoch)
        writer.close()

    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, "final.pt")
    torch.save(model.state_dict(), final_checkpoint_path)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="path to configuration file")
args = parser.parse_args()
# Load configuration file
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Train model
train(config)

       
