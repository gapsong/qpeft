"""Integer codes packed into int32 words: the GPTQ qweight layout.

codes (out, in) -> qweight (in * bits // 32, out).
For each output column, the codes along the input dimension are written as one bit stream,
lowest bit first: code k starts at bit offset k * bits, and the stream is cut into 32-bit words.
So code k lands in word (k * bits) // 32 at shift (k * bits) % 32; the bits that do not fit
spill over into the next word.
For 2 / 4 / 8 bits this is GPTQ's "32 // bits codes per word"; for 3 bits it is GPTQ's
32-codes-in-3-words scheme.

Layout of AutoGPTQ's QuantLinear.pack (https://github.com/AutoGPTQ/AutoGPTQ, MIT License).
"""
from __future__ import annotations

import torch

LOW_32_BITS = (1 << 32) - 1


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """(out, in) integer codes in [0, 2**bits - 1] -> (in * bits // 32, out) int32."""
    out_features, in_features = codes.shape
    word, shift = _word_and_shift(in_features, bits, codes.device)
    codes = codes.t().to(torch.int64)                            # (in, out)
    if int(codes.min()) < 0 or int(codes.max()) > (1 << bits) - 1:
        raise ValueError(f"codes must be in [0, {(1 << bits) - 1}] for {bits} bits.")

    shifted = codes << shift[:, None]
    n_words = in_features * bits // 32
    words = torch.zeros(n_words + 1, out_features, dtype=torch.int64, device=codes.device)
    # The codes' bit ranges do not overlap, so adding them is the same as OR-ing them.
    words.index_add_(0, word, shifted & LOW_32_BITS)             # the part that fits in its word
    words.index_add_(0, word + 1, shifted >> 32)                 # the part that spills into the next word
    return _to_signed_int32(words[:n_words])


def unpack_codes(packed: torch.Tensor, bits: int, in_features: int) -> torch.Tensor:
    """(in * bits // 32, out) int32 -> (out, in) int32 codes."""
    word, shift = _word_and_shift(in_features, bits, packed.device)
    words = packed.to(torch.int64) & LOW_32_BITS                 # the same 32 bits, as a positive number
    next_words = torch.cat([words[1:], torch.zeros_like(words[:1])])

    low_part = words[word] >> shift[:, None]
    spilled_part = next_words[word] << (32 - shift)[:, None]
    codes = (low_part | spilled_part) & ((1 << bits) - 1)
    return codes.t().to(torch.int32).contiguous()


def _word_and_shift(in_features: int, bits: int, device):
    """Where each code starts: word index and bit shift inside that word."""
    if (in_features * bits) % 32:
        raise ValueError(f"in_features * bits must be a multiple of 32 to pack {bits}-bit codes "
                         f"(in_features={in_features}).")
    offset = torch.arange(in_features, device=device, dtype=torch.int64) * bits
    return offset // 32, offset % 32


def _to_signed_int32(words):
    """0 .. 2**32 - 1 -> the int32 with the same 32 bits."""
    return torch.where(words >= 1 << 31, words - (1 << 32), words).to(torch.int32)
