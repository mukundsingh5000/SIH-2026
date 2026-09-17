import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# EXACT class mapping per SIH specifications
# 0 = background, 1 = aircraft, 2 = bottle, 3 = cylinder, 4 = human, 5 = net, 6 = pipe, 7 = wreck
CLASS_LIST = [
    "background",
    "aircraft",
    "bottle",
    "cylinder",
    "human",
    "net",
    "pipe",
    "wreck"
]

IMG_SIZE = 64


class GhostNetScorer(nn.Module):
    """Lightweight Sonar CNN Architecture for Edge Deployment"""
    def __init__(self, n_classes: int = len(CLASS_LIST)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),   # 64->32
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 16->8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Identity(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier[2](x)


class SonarPredictor:
    """Handles PyTorch model initialization and real-time sonar image classification."""
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Predictor] Initializing GhostNet Scorer on device: {self.device}")
        
        self.model = GhostNetScorer(n_classes=len(CLASS_LIST)).to(self.device)
        
        state_dict = torch.load(model_path, map_location=self.device)
        if isinstance(state_dict, dict) and "model_state" in state_dict:
            state_dict = state_dict["model_state"]
            
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[Predictor] Model successfully loaded from {model_path}")

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """Loads byte stream into OpenCV grayscale array, resizes to 64x64, normalizes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid image file or unreadable format")
        
        # Resize to model input size (64x64)
        crop = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] range as done in crops_to_batch demo_infer logic
        batch = crop.astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
        return tensor

    def predict(self, image_bytes: bytes) -> dict:
        """Executes model inference and returns formatted detection result."""
        tensor = self.preprocess(image_bytes).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            
            class_id = int(pred.item())
            confidence = float(conf.item())
            class_name = CLASS_LIST[class_id]
            detected = (class_id != 0)

        return {
            "success": True,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "detected": detected
        }
