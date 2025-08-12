from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Prediction:
    pred_idx: int
    probs: Tuple[float, float]
    class_names: Tuple[str, str] = ("NORMAL", "PNEUMONIA")

class ImageClassifier:
    def __init__(self, weights_path: Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = models.resnet34(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        x = self.t(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
        pred_idx = int(torch.argmax(logits, dim=1).item())
        return Prediction(pred_idx=pred_idx, probs=(probs[0], probs[1]))