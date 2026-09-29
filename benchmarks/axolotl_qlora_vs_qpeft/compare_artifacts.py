"""Are two saved qpeft artifacts bit-identical? (the Triton and the torch-path training runs)"""
import sys

import torch

a, b = (torch.load(f"{path}/qpeft_model.pt", weights_only=True) for path in sys.argv[1:3])
assert a.keys() == b.keys()
different = [name for name in a if not torch.equal(a[name], b[name])]
print(f"artifacts: {len(a)} tensors, {len(different)} differ" + (f" (first: {different[:3]})" if different else ""))
