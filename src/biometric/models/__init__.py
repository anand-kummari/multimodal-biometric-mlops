"""Model definitions: encoders, fusion networks, and model registry."""

from biometric.models.base import BaseEncoder, BaseFusionModel
from biometric.models.fingerprint_encoder import FingerprintEncoder
from biometric.models.fusion import MultimodalFusionNet
from biometric.models.iris_encoder import IrisEncoder

__all__ = [
    "BaseEncoder",
    "BaseFusionModel",
    "FingerprintEncoder",
    "IrisEncoder",
    "MultimodalFusionNet",
]
