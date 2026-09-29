"""Why does the bf16 merge check fail at a high QA-LoRA lr?

One bf16 QA-LoRA layer with a growing adapter. The training forward is compared with the merged
output, once with the folded zero-point z' in bf16 (as qpeft stores it) and once with z' in fp32.
Both are also compared with the exact fp32 result of what was trained.
"""
import copy

import torch
import torch.nn.functional as F
from torch import nn

from qpeft import QALoraConfig, get_quant_model
from qpeft.tuners.tuners_utils import quant_layers
from qpeft.utils import _merge_tolerance

FEATURES, GROUP_SIZE = 960, 64


def max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def expand(per_group):
    return per_group.repeat_interleave(GROUP_SIZE, dim=-1)


@torch.no_grad()
def compare(adapter_std):
    base = nn.Sequential(nn.Linear(FEATURES, FEATURES)).to("cuda", torch.bfloat16)
    layer = quant_layers(get_quant_model(base, QALoraConfig(bits=4, group_size=GROUP_SIZE, r=16)))[0].eval()
    layer.adapter.A.normal_(0, adapter_std)
    layer.adapter.B.normal_(0, adapter_std)
    x = torch.randn(4, FEATURES, device="cuda", dtype=torch.bfloat16)

    trained = layer(x)
    merged = copy.deepcopy(layer)
    merged.merge()
    merged_bf16_z = merged(x)

    codes = layer.codes.float()
    scale = layer.scheme.clamp_scale(layer.scale.float())
    zero_point = layer._zero_point_used().float()
    folded_fp32 = zero_point - layer.adapter.folded_delta().float() / scale
    weight = ((codes - expand(folded_fp32)) * expand(scale)).to(torch.bfloat16)
    merged_fp32_z = F.linear(x, weight, layer.bias)

    exact_weight = (codes - expand(zero_point)) * expand(scale)
    exact = F.linear(x.float(), exact_weight, layer.bias.float()) + layer.adapter(x.float())

    print(f"adapter std {adapter_std:4}: tolerance {_merge_tolerance(torch.bfloat16, trained):.3f} | "
          f"trained vs merged: z' bf16 {max_diff(trained, merged_bf16_z):.4f}, "
          f"z' fp32 {max_diff(trained, merged_fp32_z):.4f} | vs exact: trained {max_diff(trained, exact):.4f}, "
          f"merged z' bf16 {max_diff(merged_bf16_z, exact):.4f}, merged z' fp32 {max_diff(merged_fp32_z, exact):.4f}")


def main():
    torch.manual_seed(0)
    for adapter_std in (0.05, 0.1, 0.2, 0.4, 0.8):
        compare(adapter_std)


if __name__ == "__main__":
    main()
