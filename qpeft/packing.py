"""Integer codes packed into int32 words: the GPTQ qweight layout.

codes (out, in) -> qweight (in * bits // 32, out). Along the input dimension the codes of one output
column form a little-endian bit stream: code k sits at bit offset k * bits, a code that crosses a word
boundary continues in the next word. For 2 / 4 / 8 bits this is GPTQ's "32 // bits codes per word",
for 3 bits it is GPTQ's 32-codes-in-3-words scheme; other widths follow the same stream.
"""
from __future__ import annotations

import torch

_MASK32 = (1 << 32) - 1


def _offsets(in_features: int, bits: int, device):
    if (in_features * bits) % 32:
        raise ValueError(f"in_features * bits must be a multiple of 32 to pack {bits}-bit codes "
                         f"(in_features={in_features}).")
    off = torch.arange(in_features, device=device, dtype=torch.int64) * bits
    return off // 32, off % 32


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """(out, in) integer codes in [0, 2**bits - 1] -> (in * bits // 32, out) int32."""
    out_f, in_f = codes.shape
    word, shift = _offsets(in_f, bits, codes.device)
    c = codes.t().to(torch.int64)
    if int(c.min()) < 0 or int(c.max()) > (1 << bits) - 1:
        raise ValueError(f"codes must be in [0, {(1 << bits) - 1}] for {bits} bits.")
    n_words = in_f * bits // 32
    words = torch.zeros(n_words + 1, out_f, dtype=torch.int64, device=codes.device)
    words.index_add_(0, word, c << shift[:, None])    # bit ranges are disjoint, so add == or
    words[1:] += words[:-1] >> 32                     # the part of a code that crossed into the next word
    words = words[:n_words] & _MASK32
    return torch.where(words > 0x7FFFFFFF, words - (1 << 32), words).to(torch.int32)


def unpack_codes(packed: torch.Tensor, bits: int, in_features: int) -> torch.Tensor:
    """(in * bits // 32, out) int32 -> (out, in) int32 codes."""
    word, shift = _offsets(in_features, bits, packed.device)
    w = packed.to(torch.int64) & _MASK32
    nxt = torch.cat([w[1:], torch.zeros_like(w[:1])])
    v = (w[word] >> shift[:, None]) | (nxt[word] << (32 - shift)[:, None])
    return (v & ((1 << bits) - 1)).t().to(torch.int32).contiguous()
