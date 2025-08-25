"""Feature extraction wrapper using a pre-trained ReID model."""
from __future__ import annotations

from typing import Optional

import numpy as np

import torch
from torchvision import models, transforms


class FeatureExtractor:
    """Wrapper around a person re-identification model."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialize and load the underlying model.

        Parameters
        ----------
        model_path:
            Optional path to a model checkpoint containing weights for the
            512‑dimensional embedding layer.  If omitted the embedding layer
            remains randomly initialised which is sufficient for testing
            purposes.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load an ImageNet pre‑trained ResNet‑50 backbone and replace the final
        # classification layer with a 512‑D embedding layer.
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, 512)

        if model_path is not None:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet preprocessing pipeline.
        self._preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    # ------------------------------------------------------------------
    def extract(self, cropped: np.ndarray) -> np.ndarray:
        """Generate a feature vector for the provided image.

        The image is preprocessed, passed through the ResNet‑50 backbone and
        normalised to unit length.

        Parameters
        ----------
        cropped:
            Image array containing the detected object (H x W x C in RGB).

        Returns
        -------
        np.ndarray
            Normalised 512‑D embedding representing the object.
        """

        # Convert to tensor and move to the appropriate device
        tensor = self._preprocess(cropped).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)

        # L2 normalise to obtain the final descriptor
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.squeeze(0).cpu().numpy()
