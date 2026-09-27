"""Scheme name (config.qat_scheme) -> the function that builds the scheme for a backend."""
from __future__ import annotations

from typing import Callable

from ..config import QuantTuningConfig
from .base import FakeQuantizeConfig, QuantScheme, UnsupportedSchemeError
from .reference import ReferenceIntUniformScheme

_SCHEMES: dict[str, Callable[[FakeQuantizeConfig, str], QuantScheme]] = {}


def register_scheme(name: str):
    def register(build):
        _SCHEMES[name] = build
        return build
    return register


def build_scheme(config: QuantTuningConfig) -> QuantScheme:
    """The scheme for `config`, or UnsupportedSchemeError if it cannot guarantee
    fake_quant == merge for it."""
    if config.qat_scheme not in _SCHEMES:
        raise UnsupportedSchemeError(
            f"unknown qat_scheme {config.qat_scheme!r}; registered: {sorted(_SCHEMES)}. "
            f"Refusing rather than approximating.")
    fq = FakeQuantizeConfig(dtype=f"int{config.bits}", group_size=config.group_size)
    scheme = _SCHEMES[config.qat_scheme](fq, config.backend)
    scheme.assert_supported(config)
    return scheme


# The backend names who implements the primitives, not a device; the device follows the tensors.
@register_scheme("int_uniform")
def _int_uniform(fq: FakeQuantizeConfig, backend: str) -> QuantScheme:
    if backend in ("auto", "torch"):
        return ReferenceIntUniformScheme(fq, backend="torch", supports_adapter=True)
    if backend == "torchao":
        try:
            from .torchao import TorchaoIntUniformScheme
        except ImportError as e:
            raise NotImplementedError(
                "backend 'torchao' needs torchao: pip install 'qpeft[torchao]'.") from e
        return TorchaoIntUniformScheme(fq, backend="torchao", supports_adapter=True)
    if backend == "mlx":
        raise NotImplementedError(
            "the 'mlx' backend is not built yet; it implements the same "
            "(fake_quant, merge) contract and is measured at the same gate.")
    raise UnsupportedSchemeError(
        f"unknown backend {backend!r}; choose from 'auto', 'torch', 'torchao', 'mlx'. "
        f"Refusing rather than approximating.")
