"""
Image Search Package
====================

A modular image search system using ResNet50 embeddings and FAISS.
"""

from .config import *
from .dataset import *
from .model import *
from .indexer import *
from .search import *
from .pipeline import *

__version__ = "1.0.0"
__all__ = [
    "config",
    "dataset", 
    "model",
    "indexer",
    "search",
    "pipeline",
    "ImageSearchPipeline",
    "EmbeddingModel",
    "FAISSIndex",
    "FashionDataset",
]