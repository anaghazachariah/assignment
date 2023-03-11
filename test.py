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
from sklearn.metrics import f1_score

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


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def load_model(config):
    # Set device (GPU or CPU)
    device = get_device()

    # Load saved model checkpoint
    checkpoint_path = config["testing"]["checkpoint_path"]
    model.load_state_dict(torch.load(checkpoint_path))

    return model, device


def predict_single_file(model, device, file_path):
    # Load data
    dataset = TranscriptionDataset(file_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Predict on data
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["transcription"].to(device)
            outputs = model(inputs)
            action_preds = torch.argmax(outputs["action"], dim=1).tolist()
            object_preds = torch.argmax(outputs["object"], dim=1).tolist()
            location_preds = torch.argmax(outputs["location"], dim=1).tolist()
            all_predictions.extend([action_preds[0], object_preds[0], location_preds[0]])

    return all_predictions


def predict_single_text(model, device, text):
    # Convert text to tensor
    tensor = torch.Tensor(text).unsqueeze(0)

    # Predict on tensor
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        action_preds = torch.argmax(outputs["action"], dim=1).item()
        object_preds = torch.argmax(outputs["object"],
