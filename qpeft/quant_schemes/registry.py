"""qat_scheme -> scheme factory. The contract is chosen by name, the backend
(provider) inside the factory; build_scheme refuses what it cannot guarantee."""
from __future__ import annotations

from typing import Callable

from ..config import QuantTuningConfig
from .base import FakeQuantizeConfig, QuantScheme, UnsupportedSchemeError
from .reference import ReferenceIntUniformScheme


_SCHEMES: dict[str, Callable[[FakeQuantizeConfig, str], QuantScheme]] = {}


def register_scheme(name: str):             # ~ peft's peft_type -> tuner registry
    def deco(fn):
        _SCHEMES[name] = fn
        return fn
    return deco


def build_scheme(cfg: QuantTuningConfig) -> QuantScheme:
    fq = FakeQuantizeConfig(dtype=f"int{cfg.bits}", group_size=cfg.group_size)
    try:
        factory = _SCHEMES[cfg.qat_scheme]
    except KeyError:
        raise UnsupportedSchemeError(              # refuse loudly, don't KeyError
            f"unknown qat_scheme {cfg.qat_scheme!r}; registered: {sorted(_SCHEMES)}. "
            f"Refusing rather than approximating.") from None
    scheme = factory(fq, cfg.backend)
    scheme.assert_supported(cfg)                   # refuse rather than approximate
    return scheme


# Backend names a PROVIDER (who implements the primitives), never a device:
#   "auto" | "torch"  -> pure-torch reference, always available
#   "torchao"         -> torchao's stable affine primitives (optional dependency)
#   "mlx"             -> planned, refused until built
# The device is orthogonal and follows the model's tensors.
@register_scheme("int_uniform")
def _int_uniform(fq: FakeQuantizeConfig, backend: str) -> QuantScheme:
    if backend in ("auto", "torch"):
        return ReferenceIntUniformScheme(fq, backend="torch", supports_adapter=True)
    if backend == "torchao":
        try:
            from .torchao import TorchaoIntUniformScheme
        except ImportError as e:                    # torchao is an optional dependency
            raise NotImplementedError(
                "backend 'torchao' needs torchao: pip install 'qpeft[torchao]'.") from e
        return TorchaoIntUniformScheme(fq, backend="torchao", supports_adapter=True)
    if backend == "mlx":                            # known provider, not built yet
        raise NotImplementedError(
            "the 'mlx' backend is not built yet; it implements the same "
            "(fake_quant, merge) contract and is measured at the same gate.")
    raise UnsupportedSchemeError(                   # typo / unknown provider: refuse loudly
        f"unknown backend {backend!r}; choose from 'auto', 'torch', 'torchao', 'mlx'. "
        f"Refusing rather than approximating.")
