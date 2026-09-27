"""Inference kernels for merged qpeft models."""
from .tinygemm import TinyGemmLinear, to_tinygemm

__all__ = ["TinyGemmLinear", "to_tinygemm"]
