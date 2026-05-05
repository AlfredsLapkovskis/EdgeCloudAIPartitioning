"""
Static partitioning baseline for AlexNet over a 3-node chain.
Layout is hardcoded — no split or model knobs:

    Pi      : features[0..9]      (first 10 feature layers)
    Laptop  : features[10..12]    (remaining 3 feature layers)
              + avgpool
              + flatten
    PC      : classifier only

This is the layout described in the thesis text. Energy/latency reporting,
wire format, and meter classes are byte-identical to split_infer.py and
static_baseline.py so the numbers are directly comparable.

Run modes:
    python static_alexnet.py pc      --config config_static_alexnet_split.yaml
    python static_alexnet.py laptop  --config config_static_alexnet_split.yaml
    python static_alexnet.py pi      --config config_static_alexnet_split.yaml
"""

from __future__ import annotations

import argparse
import os
import pickle
import statistics
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
import zmq
from torchvision.models import alexnet, AlexNet_Weights

# ---------------------------------------------------------------------------
# Hardcoded layout
# ---------------------------------------------------------------------------
PI_LAST_FEATURE  = 9    # Pi runs features[0..9]   (10 layers)
LAP_LAST_FEATURE = 12   # Laptop runs features[10..12] + avgpool + flatten
                        # PC runs the classifier only

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
# Model loading (AlexNet only)
# ---------------------------------------------------------------------------
def load_alexnet():
    m = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    m.eval()
    return m

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    pi_addr: str = "tcp://127.0.0.1:5555"
    laptop_addr: str = "tcp://127.0.0.1:5556"
    total_runs: int = 500
    warmup: int = 3

    @classmethod
    def load(cls, path):
        if not path or not os.path.exists(path): return cls()
        with open(path) as f: d = yaml.safe_load(f) or {}
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
class PCNode:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ctx = zmq.Context.instance()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = load_alexnet()
        self.classifier = m.classifier.to(self.device).eval()
        self.meter = NvmlMeter()
        self.sock = self.ctx.socket(zmq.REP); self.sock.bind(cfg.laptop_addr)

    def serve(self):
        print(f"[PC ] AlexNet classifier-only  listening on {self.cfg.laptop_addr}  "
              f"device={self.device}")
        while True:
            msg = unpack(self.sock.recv())
            if msg.get("cmd") == "stop":
                self.sock.send(pack({"ok": True})); break
            x = wire_to_tensor(msg["x"]).to(self.device)
            with self.meter as m:
                with torch.no_grad():
                    out = self.classifier(x)
                if self.device.type == "cuda": torch.cuda.synchronize()
            self.sock.send(pack({"compute_time": m.dt, "energy_j": m.energy_j,
                                 "logits": tensor_to_wire(out.cpu())}))

class LaptopNode:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ctx = zmq.Context.instance()
        m = load_alexnet()
        # Laptop owns features[10..12] + avgpool + flatten (one Sequential
        # so each inference is a single forward call).
        self.tail = nn.Sequential(
            *list(m.features[PI_LAST_FEATURE + 1 : LAP_LAST_FEATURE + 1]),
            m.avgpool,
            nn.Flatten(1),
        ).eval()
        self.meter = RaplMeter()
        self.front = self.ctx.socket(zmq.REP); self.front.bind(cfg.pi_addr)
        self.back  = self.ctx.socket(zmq.REQ); self.back.connect(cfg.laptop_addr)

    def serve(self):
        print(f"[LAP] AlexNet features[{PI_LAST_FEATURE+1}..{LAP_LAST_FEATURE}] "
              f"+ avgpool + flatten")
        print(f"[LAP] front {self.cfg.pi_addr}  back {self.cfg.laptop_addr}")
        while True:
            msg = unpack(self.front.recv())
            if msg.get("cmd") == "stop":
                self.back.send(pack({"cmd": "stop"})); _ = self.back.recv()
                self.front.send(pack({"ok": True})); break
            x = wire_to_tensor(msg["x"])
            with self.meter as m:
                with torch.no_grad():
                    x = self.tail(x)
            t_send = time.perf_counter()
            self.back.send(pack({"x": tensor_to_wire(x)}))
            rep = unpack(self.back.recv())
            transfer_lp = time.perf_counter() - t_send - rep["compute_time"]
            self.front.send(pack({
                "lap_compute_time": m.dt, "lap_energy_j": m.energy_j,
                "pc_compute_time": rep["compute_time"],
                "pc_energy_j":     rep["energy_j"],
                "lp_transfer_time": max(transfer_lp, 0.0),
                "logits": rep["logits"],
            }))

class PiNode:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ctx = zmq.Context.instance()
        m = load_alexnet()
        # Pi owns features[0..9].
        self.front = nn.Sequential(
            *list(m.features[0 : PI_LAST_FEATURE + 1])
        ).eval()
        self.sock = self.ctx.socket(zmq.REQ); self.sock.connect(cfg.pi_addr)

    def infer_once(self, x):
        t0 = time.perf_counter()
        with PiPowerMeter() as m:
            with torch.no_grad():
                x_pi = self.front(x)
        self.sock.send(pack({"x": tensor_to_wire(x_pi)}))
        rep = unpack(self.sock.recv())
        rtt = time.perf_counter() - t0
        pl = (rtt - m.dt - rep["lap_compute_time"]
                  - rep["lp_transfer_time"] - rep["pc_compute_time"])
        return {
            "pi_compute":  m.dt,    "pi_energy":  m.energy_j,
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
# Output
# ---------------------------------------------------------------------------
def mean(xs): return sum(xs) / len(xs) if xs else 0.0

def print_header():
    print(f"{'run':>4} {'pi_E':>8} {'lap_E':>8} {'pc_E':>8} {'tot_E':>8} {'lat_ms':>8}")

def print_row(run, s):
    print(f"{run:>4} "
          f"{s['pi_energy']:>8.3f} {s['lap_energy']:>8.3f} {s['pc_energy']:>8.3f} "
          f"{s['total_energy']:>8.3f} {s['latency']*1000:>8.1f}")

def final_report(all_stats):
    if not all_stats:
        print("\n[report] no samples collected"); return

    n = len(all_stats)
    print(f"\n{'='*78}")
    print(f"  STATIC BASELINE  -  AlexNet  -  "
          f"Pi[0..{PI_LAST_FEATURE}] | Lap[{PI_LAST_FEATURE+1}..{LAP_LAST_FEATURE}]+avgpool | PC=classifier")
    print(f"  {n} inferences")
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
# Driver
# ---------------------------------------------------------------------------
def run_pi_driver(cfg):
    pi = PiNode(cfg)
    print(f"[init] AlexNet static baseline")
    print(f"[init] Pi runs features[0..{PI_LAST_FEATURE}]")
    print(f"[init] Laptop runs features[{PI_LAST_FEATURE+1}..{LAP_LAST_FEATURE}] + avgpool + flatten")
    print(f"[init] PC runs classifier only")
    print(f"[init] {cfg.total_runs} inferences, dropping first {cfg.warmup} as warmup")

    dummy = torch.randn(1, 3, 224, 224)
    all_stats = []
    print_header()

    for r in range(cfg.total_runs):
        s = pi.infer_once(dummy)
        if r >= cfg.warmup:
            all_stats.append(s)
        if (r + 1) % 10 == 0:
            print_row(r + 1, s)

    pi.stop_chain()
    final_report(all_stats)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=["pi", "laptop", "pc"])
    p.add_argument("--config", default="config_static_alexnet_split.yaml")
    args = p.parse_args()
    cfg = Config.load(args.config)
    if args.role == "pc":       PCNode(cfg).serve()
    elif args.role == "laptop": LaptopNode(cfg).serve()
    else:                       run_pi_driver(cfg)

if __name__ == "__main__":
    main()
