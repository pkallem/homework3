from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=6):
        """
        A simple convolutional network for image classification.
        Input shape: (B, 3, 64, 64) -> Output shape: (B, 6)
        """
        super().__init__()

        # Register normalization buffers (used for standardizing the input)
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Example architecture:
        #  (3, 64, 64) -> conv -> bn -> relu -> pool -> ...
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (16, 32, 32)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (64, 1, 1)
        )

        # Final linear layer to produce 6 logits
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1) Optional normalize
          2) Convolutions
          3) Global pooling
          4) Linear -> logits
        """
        # 1) normalize input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # 2) apply CNN feature extractor
        feats = self.features(z)
        # feats has shape (B, 64, 1, 1) after AdaptiveAvgPool2d

        # 3) flatten + pass to linear
        feats = feats.view(feats.size(0), -1)  # (B,64)
        logits = self.classifier(feats)        # (B,6)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference (the grader calls this).
        Returns class labels in {0..5}.
        """
        return self(x).argmax(dim=1)


class Detector(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        """
        A single model that performs segmentation + depth prediction.

        Output:
          - segmentation logits: (B, num_classes, H, W)
          - depth: (B, H, W)
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Example small "U-Net" style architecture
        # Down
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Up
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # Final heads
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)
        self.depth_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Takes an image (B,3,H,W) -> returns (segmentation logits, depth map).
          - segmentation logits: (B,3,H,W)
          - depth: (B,H,W)
        """
        # 1) normalize
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # 2) downsample
        d1 = self.down1(z)   # (B,16,H/2,W/2)
        d2 = self.down2(d1)  # (B,32,H/4,W/4)

        # 3) upsample
        u1 = self.up1(d2)    # (B,16,H/2,W/2)
        u2 = self.up2(u1)    # (B,16,H,W)

        # 4) produce final outputs
        logits = self.seg_head(u2)            # (B,3,H,W)
        raw_depth = self.depth_head(u2)       # (B,1,H,W)

        # For convenience, reduce to (B,H,W) for the depth
        raw_depth = raw_depth[:, 0, :, :]     # -> (B,H,W)

        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference (the grader calls this).
        Returns (predicted_segmentation, predicted_depth).
          - predicted_segmentation: (B,H,W) in {0..2}
          - predicted_depth: (B,H,W) in [0,1]
        """
        logits, raw_depth = self(x)
        pred_seg = logits.argmax(dim=1)
        pred_depth = raw_depth  # you could clamp or post-process if needed
        return pred_seg, pred_depth


# The grader looks for these function names
MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}

def load_model(model_name: str, with_weights: bool = False, **model_kwargs) -> nn.Module:
    """
    Called by the grader to load a pre-trained model by name.
    Do not remove or rename this function.
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        m.load_state_dict(torch.load(model_path, map_location="cpu"))

    # optional size check, etc.
    return m

def save_model(model: nn.Module) -> str:
    """
    Use this function to save your model in train_*.py
    """
    for name, cls_ in MODEL_FACTORY.items():
        if isinstance(model, cls_):
            save_path = HOMEWORK_DIR / f"{name}.th"
            torch.save(model.state_dict(), save_path)
            return str(save_path)

    raise ValueError(f"Unknown model class: {type(model)}")