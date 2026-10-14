"""Session 03 — Calculator to check conv numericals: output size, params, MACs, receptive field.

Usage:
  python conv_output_and_receptive_field_calc.py                         # runs the built-in examples
  python conv_output_and_receptive_field_calc.py --input 224 3 \
      --layers "conv:7:2:3:64 pool:3:2:1 conv:3:1:1:64 conv:3:1:1:64:2"
Layer syntax:  conv:K:S:P:C_out[:D]   pool:K:S:P   (D = dilation, default 1; square inputs)
Formulas (same notation as the notes):
  W_out = floor((W - K_eff + 2P)/S) + 1,  K_eff = D(K-1) + 1
  params = (K*K*C_in + 1) * C_out  (with bias);  pooling has 0 params
  MACs   = W_out^2 * K^2 * C_in * C_out
  RF: r_l = r_{l-1} + (K_eff - 1) * j_{l-1},  j_l = j_{l-1} * S,  r_0 = j_0 = 1
"""
import argparse
import math


def conv_out(W, K, S=1, P=0, D=1):
    return math.floor((W - (D * (K - 1) + 1) + 2 * P) / S) + 1


def transposed_conv_out(W, K, S=1, P=0, output_padding=0):
    return (W - 1) * S - 2 * P + K + output_padding


def analyse(W, C, layers):
    r, j, total = 1, 1, 0
    print(f"{'layer':22s}{'out (W x W x C)':>18s}{'params':>12s}{'MACs':>16s}{'RF':>6s}{'jump':>6s}")
    for spec in layers:
        kind, *nums = spec.split(":")
        K, S, P = map(int, nums[:3])
        D = int(nums[4]) if kind == "conv" and len(nums) > 4 else 1
        C_out = int(nums[3]) if kind == "conv" else C
        W_out = conv_out(W, K, S, P, D)
        params = (K * K * C + 1) * C_out if kind == "conv" else 0
        macs = W_out * W_out * K * K * C * C_out if kind == "conv" else 0
        r, j = r + (D * (K - 1)) * j, j * S          # receptive-field recurrence (uses K_eff - 1)
        total += params
        print(f"{spec:22s}{f'{W_out} x {W_out} x {C_out}':>18s}{params:>12,d}{macs:>16,d}{r:>6d}{j:>6d}")
        W, C = W_out, C_out
    print(f"Total params: {total:,d}")
    return W, C, total


def torch_check(W, C, layers):
    """Optional: build the same stack in PyTorch and confirm shape + parameter count."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return print("(torch not installed: skipping cross-check)")
    mods, c = [], C
    for spec in layers:
        kind, *n = spec.split(":")
        K, S, P = map(int, n[:3])
        if kind == "conv":
            mods.append(nn.Conv2d(c, int(n[3]), K, S, P, dilation=int(n[4]) if len(n) > 4 else 1))
            c = int(n[3])
        else:
            mods.append(nn.MaxPool2d(K, S, P))
    net = nn.Sequential(*mods)
    out = net(torch.zeros(1, C, W, W))
    print(f"torch check: output {tuple(out.shape[1:])}, params {sum(p.numel() for p in net.parameters()):,d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs=2, type=int, metavar=("W", "C"))
    ap.add_argument("--layers", type=str)
    a = ap.parse_args()
    examples = {"AlexNet conv1": (227, 3, "conv:11:4:0:96"),
                "ResNet stem": (224, 3, "conv:7:2:3:64 pool:3:2:1"),
                "VGG: two 3x3 vs one 5x5": (32, 64, "conv:3:1:1:64 conv:3:1:1:64"),
                "dilated stack": (64, 16, "conv:3:1:1:16 conv:3:1:2:16:2 conv:3:1:4:16:4")}
    if a.input and a.layers:
        examples = {"custom": (a.input[0], a.input[1], a.layers)}
    for name, (W, C, spec) in examples.items():
        print(f"\n=== {name}: input {W} x {W} x {C} ===")
        analyse(W, C, spec.split())
        torch_check(W, C, spec.split())
    print(f"\nTransposed conv 16 -> {transposed_conv_out(16, K=4, S=2, P=1)} (K=4, S=2, P=1 doubles size)")
