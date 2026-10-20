"""
MorphicCV: Next-Generation Lightweight Computer Vision Library (<8GB RAM).
Combines Differentiable Mathematical Morphology, Dynamic Sparse Training,
GigaPixel Saliency Routing, and Foundation Model Adapters.
"""

__version__ = "0.1.1"

from .morphology.ops import SoftDilation2d, SoftErosion2d, MorphologicalGradient2d
from .morphology.rigl import RigLScheduler
from .routing.router import SaliencyRouter
from .adapters.msa import MorphicAdapter
from .models.dinov2 import MorphicDINOv2
from .streaming.quantizer import ProductQuantizer
from .streaming.shard import TokenShardWriter, StreamingTokenDataset
from .inference.engine import MorphicInferenceEngine

__all__ = [
    "__version__",
    "SoftDilation2d",
    "SoftErosion2d",
    "MorphologicalGradient2d",
    "RigLScheduler",
    "SaliencyRouter",
    "MorphicAdapter",
    "MorphicDINOv2",
    "ProductQuantizer",
    "TokenShardWriter",
    "StreamingTokenDataset",
    "MorphicInferenceEngine",
]
