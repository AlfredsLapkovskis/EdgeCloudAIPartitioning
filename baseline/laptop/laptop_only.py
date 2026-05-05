"""
Laptop-only baseline. Runs the entire model on the laptop CPU, end to end,
no Pi, no PC, no ZMQ, no chain. Pure single-machine reference.

Supports VGG-16, AlexNet, and MobileNetV2 via the --config file's `model:` key.

Energy (RAPL package delta) and latency (perf_counter) are measured per
inference using the same meter classes as the distributed scripts, so the
numbers are directly comparable to the chain baselines.

Run modes:
    python laptop_only.py --config config_laptop_vgg16.yaml
    python laptop_only.py --config config_laptop_alexnet.yaml
    python laptop_only.py --config config_laptop_mobilenet_v2.yaml
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import yaml
from torchvision.models import (
    vgg16, VGG16_Weights,
    alexnet, AlexNet_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
)

# ---------------------------------------------------------------------------
# Model registry (same as the other scripts)
# ---------------------------------------------------------------------------
def _vgg16():        return vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
def _alexnet():      return alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
def _mobilenet_v2(): return mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)

MODEL_REGISTRY = {
    "vgg16":        _vgg16,
    "alexnet":      _alexnet,
    "mobilenet_v2": _mobilenet_v2,
}

def load_model(name: str) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Available: {list(MODEL_REGISTRY)}")
    m = MODEL_REGISTRY[name]()
    m.eval()
    return m

# ---------------------------------------------------------------------------
# RAPL meter — IDENTICAL to split_infer.py
# ---------------------------------------------------------------------------
class RaplMeter:
    PATH = "/sys/class/powercap/intel-rapl:0/energy_uj"
    def _read(self):
        try:
            with open(self.PATH) as f: return int(f.read().strip()) / 1e6
        except (FileNotFoundError, PermissionError):
            return None
    def __enter__(self):
        self.t0 = time.perf_counter(); self.e0 = self._read(); return self
    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
        e1 = self._read()
        if self.e0 is None or e1 is None:
            self.energy_j = 15.0 * self.dt
        else:
            self.energy_j = (e1 - self.e0) if e1 >= self.e0 \
                else (e1 + 2**32 / 1e6 - self.e0)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    model: str = "vgg16"
    total_runs: int = 500
    warmup: int = 3

    @classmethod
    def load(cls, path):
        if not path or not os.path.exists(path): return cls()
        with open(path) as f: d = yaml.safe_load(f) or {}
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_header():
    print(f"{'run':>4} {'lap_E':>10} {'lat_ms':>10}")

def print_row(run, energy_j, latency_s):
    print(f"{run:>4} {energy_j:>10.4f} {latency_s*1000:>10.2f}")

def final_report(all_stats, model_name):
    if not all_stats:
        print("\n[report] no samples collected"); return

    n = len(all_stats)
    print(f"\n{'='*78}")
    print(f"  LAPTOP-ONLY BASELINE  -  model={model_name}  -  {n} inferences")
    print(f"{'='*78}")

    metric_defs = [
        ("End-to-end latency", "latency",   1000, "ms"),
        ("Laptop energy",      "energy_j",     1, "J"),
    ]
    print(f"\n  PER-INFERENCE STATISTICS ({n} samples)")
    print(f"  {'metric':<22}{'avg':>12}{'stdev':>12}{'min':>12}{'max':>12}  unit")
    print(f"  {'-'*22}{'-'*12}{'-'*12}{'-'*12}{'-'*12}  ----")
    for label, key, scale, unit in metric_defs:
        vals = [s[key] * scale for s in all_stats]
        mu = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        lo, hi = min(vals), max(vals)
        print(f"  {label:<22}{mu:>12.3f}{sd:>12.3f}{lo:>12.3f}{hi:>12.3f}  {unit}")

    cum_e = sum(s["energy_j"] for s in all_stats)
    cum_l = sum(s["latency"]  for s in all_stats)
    print(f"\n  CUMULATIVE TOTALS ({n} inferences)")
    print(f"    Laptop energy total   {cum_e:>10.2f} J")
    print(f"    Wall-clock total      {cum_l:>10.2f} s")
    print(f"    Throughput            {n/cum_l:>10.2f} inf/s")
    print(f"{'='*78}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_laptop.yaml")
    args = p.parse_args()
    cfg = Config.load(args.config)

    print(f"[init] laptop-only baseline  model={cfg.model}")
    print(f"[init] {cfg.total_runs} inferences, dropping first {cfg.warmup} as warmup")
    model = load_model(cfg.model)
    dummy = torch.randn(1, 3, 224, 224)

    # Warmup the model end-to-end before measuring (matches the build_tables
    # warmup in split_infer.py, accounts for first-pass allocator/cache costs).
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy)

    all_stats = []
    print_header()

    for r in range(cfg.total_runs):
        with RaplMeter() as m:
            with torch.no_grad():
                _ = model(dummy)
        s = {"latency": m.dt, "energy_j": m.energy_j}
        if r >= cfg.warmup:
            all_stats.append(s)
        if (r + 1) % 10 == 0:
            print_row(r + 1, s["energy_j"], s["latency"])

    final_report(all_stats, cfg.model)

if __name__ == "__main__":
    main()
