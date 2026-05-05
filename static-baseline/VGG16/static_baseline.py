"""
Static partitioning baseline for VGG-16 over a 3-node chain (Pi -> Laptop -> PC).

This is a stripped-down version of split_infer.py with the adaptive logic
removed. It runs a single fixed split for the entire experiment so its
energy and latency numbers are directly comparable to the adaptive scheduler.

All measurement code (PiPowerMeter / RaplMeter / NvmlMeter), wire format,
and the inference path are byte-for-byte identical to split_infer.py — only
the scheduler is gone. This makes the baseline a fair reference: any
difference in measured energy or latency vs the adaptive run reflects the
adaptive logic, not measurement methodology.

Run modes:
    python static_baseline.py pc      --config config_static.yaml
    python static_baseline.py laptop  --config config_static.yaml
    python static_baseline.py pi      --config config_static.yaml   # driver
"""

from __future__ import annotations

import argparse
import os
import pickle
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
import zmq
from torchvision.models import vgg16, VGG16_Weights

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
N_FEATURES = 31          # VGG-16 model.features has 31 children, indices 0..30
LAST_FEATURE = N_FEATURES - 1

def load_vgg16() -> tuple[nn.Module, nn.Module]:
    m = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    m.eval()
    head = nn.Sequential(m.avgpool, nn.Flatten(1), m.classifier)
    head.eval()
    return m, head

def slice_features(model: nn.Module, start: int, end_inclusive: int) -> nn.Sequential:
    return nn.Sequential(*list(model.features[start:end_inclusive + 1]))

# ---------------------------------------------------------------------------
# Energy meters — IDENTICAL to split_infer.py
# ---------------------------------------------------------------------------
class PiPowerMeter:
    POWER_W = 12.0
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
        self.energy_j = self.POWER_W * self.dt

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

class NvmlMeter:
    def __init__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.pynvml = pynvml; self.ok = True
        except Exception:
            self.ok = False
    def _power_w(self):
        if not self.ok: return 200.0
        return self.pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0
    def __enter__(self):
        self.t0 = time.perf_counter(); self.p0 = self._power_w(); return self
    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
        self.energy_j = 0.5 * (self.p0 + self._power_w()) * self.dt

# ---------------------------------------------------------------------------
# Wire format — IDENTICAL to split_infer.py
# ---------------------------------------------------------------------------
def pack(o):  return pickle.dumps(o, protocol=pickle.HIGHEST_PROTOCOL)
def unpack(b): return pickle.loads(b)

def tensor_to_wire(t):
    return {"shape": tuple(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
            "data": t.detach().cpu().numpy().tobytes()}

def wire_to_tensor(d):
    arr = np.frombuffer(d["data"], dtype=np.dtype(d["dtype"])).reshape(d["shape"])
    return torch.from_numpy(arr.copy())

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    pi_addr: str = "tcp://127.0.0.1:5555"
    laptop_addr: str = "tcp://127.0.0.1:5556"
    static_split: tuple = (10, 30)   # Pi: 0..10, Laptop: 11..30, PC: head only
    total_runs: int = 500
    warmup: int = 3                  # drop first N runs from statistics

    @classmethod
    def load(cls, path):
        if not path or not os.path.exists(path): return cls()
        with open(path) as f: d = yaml.safe_load(f) or {}
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "static_split" in d: d["static_split"] = tuple(d["static_split"])
        return cls(**d)

# ---------------------------------------------------------------------------
# Nodes — IDENTICAL inference path to split_infer.py, no probe/stop commands
# ---------------------------------------------------------------------------
class Node:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model, self.head = load_vgg16()
        self.ctx = zmq.Context.instance()
    def make_segment(self, s, e):
        seg = slice_features(self.model, s, e); seg.eval(); return seg

class PCNode(Node):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head = self.head.to(self.device)
        self.meter = NvmlMeter()
        self.sock = self.ctx.socket(zmq.REP); self.sock.bind(cfg.laptop_addr)
        self._cache = {}
    def _seg(self, s, e):
        if (s, e) not in self._cache:
            self._cache[(s, e)] = self.make_segment(s, e).to(self.device)
        return self._cache[(s, e)]
    def serve(self):
        print(f"[PC ] listening on {self.cfg.laptop_addr} (device={self.device})")
        while True:
            msg = unpack(self.sock.recv())
            if msg.get("cmd") == "stop":
                self.sock.send(pack({"ok": True})); break
            j = msg["j"]; x = wire_to_tensor(msg["x"]).to(self.device)
            with self.meter as m:
                with torch.no_grad():
                    if j < LAST_FEATURE: x = self._seg(j + 1, LAST_FEATURE)(x)
                    out = self.head(x)
                if self.device.type == "cuda": torch.cuda.synchronize()
            self.sock.send(pack({"compute_time": m.dt, "energy_j": m.energy_j,
                                 "logits": tensor_to_wire(out.cpu())}))

class LaptopNode(Node):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.meter = RaplMeter()
        self.front = self.ctx.socket(zmq.REP); self.front.bind(cfg.pi_addr)
        self.back  = self.ctx.socket(zmq.REQ); self.back.connect(cfg.laptop_addr)
        self._cache = {}
    def _seg(self, s, e):
        if (s, e) not in self._cache:
            self._cache[(s, e)] = self.make_segment(s, e)
        return self._cache[(s, e)]
    def serve(self):
        print(f"[LAP] front {self.cfg.pi_addr}  back {self.cfg.laptop_addr}")
        while True:
            msg = unpack(self.front.recv())
            if msg.get("cmd") == "stop":
                self.back.send(pack({"cmd": "stop"})); _ = self.back.recv()
                self.front.send(pack({"ok": True})); break
            i, j = msg["i"], msg["j"]; x = wire_to_tensor(msg["x"])
            with self.meter as m:
                with torch.no_grad():
                    if j > i: x = self._seg(i + 1, j)(x)
            t_send = time.perf_counter()
            self.back.send(pack({"j": j, "x": tensor_to_wire(x)}))
            rep = unpack(self.back.recv())
            transfer_lp = time.perf_counter() - t_send - rep["compute_time"]
            self.front.send(pack({
                "lap_compute_time": m.dt, "lap_energy_j": m.energy_j,
                "pc_compute_time": rep["compute_time"], "pc_energy_j": rep["energy_j"],
                "lp_transfer_time": max(transfer_lp, 0.0),
                "logits": rep["logits"],
            }))

class PiNode(Node):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sock = self.ctx.socket(zmq.REQ); self.sock.connect(cfg.pi_addr)
        self._cache = {}
    def _seg(self, s, e):
        if (s, e) not in self._cache:
            self._cache[(s, e)] = self.make_segment(s, e)
        return self._cache[(s, e)]
    def infer_once(self, x, split):
        i, j = split; t0 = time.perf_counter()
        with PiPowerMeter() as m:
            with torch.no_grad(): x_pi = self._seg(0, i)(x)
        self.sock.send(pack({"i": i, "j": j, "x": tensor_to_wire(x_pi)}))
        rep = unpack(self.sock.recv())
        rtt = time.perf_counter() - t0
        pl = rtt - m.dt - rep["lap_compute_time"] - rep["lp_transfer_time"] - rep["pc_compute_time"]
        return {
            "split": split,
            "pi_compute": m.dt, "pi_energy": m.energy_j,
            "lap_compute": rep["lap_compute_time"], "lap_energy": rep["lap_energy_j"],
            "pc_compute":  rep["pc_compute_time"],  "pc_energy":  rep["pc_energy_j"],
            "pl_transfer": max(pl, 0.0),
            "lp_transfer": rep["lp_transfer_time"],
            "rtt": rtt,
            "total_energy": m.energy_j + rep["lap_energy_j"] + rep["pc_energy_j"],
            "latency": rtt,
        }
    def stop_chain(self):
        self.sock.send(pack({"cmd": "stop"})); _ = self.sock.recv()

# ---------------------------------------------------------------------------
# Output — same shape as split_infer.py final_report so cross-comparison is direct
# ---------------------------------------------------------------------------
def mean(xs): return sum(xs) / len(xs) if xs else 0.0

def print_header():
    print(f"{'run':>4} {'split':>9} {'pi_E':>8} {'lap_E':>8} {'pc_E':>8} "
          f"{'tot_E':>8} {'lat_ms':>8}")

def print_row(run, s):
    i, j = s["split"]
    print(f"{run:>4} {f'({i:>2},{j:>2})':>9} "
          f"{s['pi_energy']:>8.3f} {s['lap_energy']:>8.3f} {s['pc_energy']:>8.3f} "
          f"{s['total_energy']:>8.3f} {s['latency']*1000:>8.1f}")

def final_report(all_stats, split):
    if not all_stats:
        print("\n[report] no samples collected"); return

    n = len(all_stats)
    print(f"\n{'='*78}")
    print(f"  STATIC BASELINE  -  split={split}  -  {n} inferences")
    print(f"{'='*78}")

    metric_defs = [
        ("Pi compute",       "pi_compute",    1000, "ms"),
        ("Laptop compute",   "lap_compute",   1000, "ms"),
        ("PC compute",       "pc_compute",    1000, "ms"),
        ("Pi->Lap transfer", "pl_transfer",   1000, "ms"),
        ("Lap->PC transfer", "lp_transfer",   1000, "ms"),
        ("End-to-end latency","latency",      1000, "ms"),
        ("Pi energy",        "pi_energy",       1, "J"),
        ("Laptop energy",    "lap_energy",      1, "J"),
        ("PC energy",        "pc_energy",       1, "J"),
        ("Total energy",     "total_energy",    1, "J"),
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

    cum_pi  = sum(s["pi_energy"]    for s in all_stats)
    cum_lap = sum(s["lap_energy"]   for s in all_stats)
    cum_pc  = sum(s["pc_energy"]    for s in all_stats)
    cum_tot = sum(s["total_energy"] for s in all_stats)
    cum_lat = sum(s["latency"]      for s in all_stats)
    print(f"\n  CUMULATIVE TOTALS ({n} inferences)")
    print(f"    Pi energy total       {cum_pi:>10.2f} J")
    print(f"    Laptop energy total   {cum_lap:>10.2f} J")
    print(f"    PC energy total       {cum_pc:>10.2f} J")
    print(f"    System energy total   {cum_tot:>10.2f} J")
    print(f"    Wall-clock total      {cum_lat:>10.2f} s")
    print(f"    Throughput            {n/cum_lat:>10.2f} inf/s")
    print(f"{'='*78}")

# ---------------------------------------------------------------------------
# Driver — runs the same fixed split for the entire experiment
# ---------------------------------------------------------------------------
def run_pi_driver(cfg):
    pi = PiNode(cfg)
    i, j = cfg.static_split
    if not (0 <= i < j < N_FEATURES):
        raise ValueError(f"static_split={cfg.static_split} invalid for VGG-16 "
                         f"(need 0 <= i < j < {N_FEATURES})")

    print(f"[init] static baseline VGG-16 split={cfg.static_split}")
    print(f"[init] Pi runs features[0..{i}], Laptop runs features[{i+1}..{j}], "
          f"PC runs features[{j+1}..{LAST_FEATURE}] + head")
    print(f"[init] {cfg.total_runs} inferences, dropping first {cfg.warmup} as warmup")

    dummy = torch.randn(1, 3, 224, 224)
    all_stats = []
    print_header()

    for r in range(cfg.total_runs):
        s = pi.infer_once(dummy, cfg.static_split)
        if r >= cfg.warmup:
            all_stats.append(s)
        if (r + 1) % 10 == 0:
            print_row(r + 1, s)

    pi.stop_chain()
    final_report(all_stats, cfg.static_split)

# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=["pi", "laptop", "pc"])
    p.add_argument("--config", default="config_static.yaml")
    args = p.parse_args()
    cfg = Config.load(args.config)
    if args.role == "pc":       PCNode(cfg).serve()
    elif args.role == "laptop": LaptopNode(cfg).serve()
    else:                       run_pi_driver(cfg)

if __name__ == "__main__":
    main()
