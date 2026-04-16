"""Model definitions: encoders, fusion networks, and model registry."""

from biometric.models.base import BaseEncoder, BaseFusionModel
from biometric.models.iris_encoder import IrisEncoder
from biometric.models.fingerprint_encoder import FingerprintEncoder
from biometric.models.fusion import MultimodalFusionNet

__all__ = [
    "BaseEncoder",
    "BaseFusionModel",
    "IrisEncoder",
    "FingerprintEncoder",
    "MultimodalFusionNet",
]
