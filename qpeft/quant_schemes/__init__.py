"""Quantization schemes: the (fake_quant, fuse) contract and its backends.

  base.py       QuantScheme contract, shared int_uniform machinery
  reference.py  pure-torch backend (always available)
  torchao.py    torchao backend (optional; imported lazily by the registry)
  registry.py   qat_scheme -> factory, build_scheme
"""
from .base import (
    SCALE_MAX, SCALE_MIN, FakeQuantizeConfig, QuantScheme, UnsupportedSchemeError,
)
from .reference import ReferenceIntUniformScheme
from .registry import build_scheme, register_scheme

__all__ = [
    "SCALE_MAX", "SCALE_MIN", "FakeQuantizeConfig", "QuantScheme", "UnsupportedSchemeError",
    "ReferenceIntUniformScheme", "build_scheme", "register_scheme",
]
