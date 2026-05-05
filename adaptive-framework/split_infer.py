"""
Distributed VGG-16 Split Inference over a 3-node chain (Pi -> Laptop -> PC).

Run modes:
    python split_infer.py pc      --config config.yaml
    python split_infer.py laptop  --config config.yaml
    python split_infer.py pi      --config config.yaml   # driver
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
import zmq
from torchvision.models import (
    vgg16, VGG16_Weights,
    alexnet, AlexNet_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry knows how to build the model and assemble its head. The number
# of feature children (N_FEATURES) is read from model.features after loading.
#
# * VGG-16 / AlexNet: head is avgpool (module) + flatten + classifier. They
#   share _vggish_head.
# * MobileNetV2: head is adaptive_avg_pool2d (functional) + flatten + classifier.
#   torchvision's MobileNetV2.forward calls F.adaptive_avg_pool2d directly,
#   not a module, so we wrap it here in a Sequential via nn.AdaptiveAvgPool2d.
#
# NOTE on MobileNetV2 splitting: model.features has 19 children (indices 0..18).
# features[0] is a ConvBNReLU, features[1..17] are InvertedResidual blocks
# (each containing its OWN internal skip connection), and features[18] is a
# final ConvBNReLU. Splitting between two top-level children is always safe,
# because every child is atomic — the residual connection lives inside the
# InvertedResidual, not across block boundaries. This matches how valid_splits
# already works, so no special handling is required beyond the head.

def _vgg16():       return vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
def _alexnet():     return alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
def _mobilenet_v2(): return mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)

def _vggish_head(model):
    return nn.Sequential(model.avgpool, nn.Flatten(1), model.classifier)

def _mobilenet_head(model):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1),
        model.classifier,
    )

MODEL_REGISTRY = {
    "vgg16":        {"factory": _vgg16,        "head_factory": _vggish_head},
    "alexnet":      {"factory": _alexnet,      "head_factory": _vggish_head},
    "mobilenet_v2": {"factory": _mobilenet_v2, "head_factory": _mobilenet_head},
}

# Populated by load_model() at startup based on the chosen model.
PAYLOAD_BYTES_FP32: dict[int, int] = {}
COMPUTE_WEIGHTS: dict[int, float] = {}
N_FEATURES: int = 0   # number of children in model.features
HEAD_INDEX: int = 0   # virtual index for the head, == N_FEATURES
LAST_FEATURE: int = 0 # index of the last feature module, == N_FEATURES - 1

def build_tables(model: nn.Module, head: nn.Module
                 ) -> tuple[dict[int, int], dict[int, float]]:
    table_bytes = {}
    table_compute = {}

    # Warmup
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(3):
            y = model.features(x)
            _ = head(y)

    # Profile features
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for idx, layer in enumerate(model.features):
            t0 = time.perf_counter()
            x = layer(x)
            dt = time.perf_counter() - t0
            table_bytes[idx] = x.numel() * x.element_size()
            table_compute[idx] = max(dt, 1e-9)

        # Profile head as a single virtual layer
        t0 = time.perf_counter()
        _ = head(x)
        table_compute[len(model.features)] = max(time.perf_counter() - t0, 1e-9)

    total_dt = sum(table_compute.values())
    weights = {k: v / total_dt for k, v in table_compute.items()}
    return table_bytes, weights

def load_model(name: str) -> tuple[nn.Module, nn.Module]:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Available: {list(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[name]
    m = spec["factory"]()
    m.eval()
    head = spec["head_factory"](m)
    head.eval()

    global PAYLOAD_BYTES_FP32, COMPUTE_WEIGHTS, N_FEATURES, HEAD_INDEX, LAST_FEATURE
    if not PAYLOAD_BYTES_FP32:
        PAYLOAD_BYTES_FP32, COMPUTE_WEIGHTS = build_tables(m, head)
        N_FEATURES = len(m.features)
        LAST_FEATURE = N_FEATURES - 1
        HEAD_INDEX = N_FEATURES
    return m, head

def slice_features(model, start, end_inclusive):
    return nn.Sequential(*list(model.features[start:end_inclusive + 1]))

def get_weight(start_idx: int, end_idx: int) -> float:
    """Sum of compute weights for the inclusive layer range [start_idx, end_idx]."""
    return sum(COMPUTE_WEIGHTS.get(k, 0.0) for k in range(start_idx, end_idx + 1))

# ---------------------------------------------------------------------------
# Energy meters
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
# Link model
# ---------------------------------------------------------------------------
@dataclass
class LinkModel:
    overhead_s: float = 0.0
    bw_bps: float = 1e8         # default 100 MB/s (sane fallback)

    def predict(self, n_bytes: int) -> float:
        return self.overhead_s + n_bytes / max(self.bw_bps, 1.0)

    def __repr__(self):
        return (f"LinkModel(overhead={self.overhead_s*1000:.2f}ms, "
                f"bw={self.bw_bps/1e6:.1f}MB/s)")

def fit_link(s1: int, t1: float, s2: int, t2: float, fallback: LinkModel) -> LinkModel:
    """Solve overhead + bytes/bw = time for two points, using fallback if malformed."""
    if t2 <= t1 or s2 <= s1:        
        return fallback
    bw = (s2 - s1) / (t2 - t1)
    overhead = max(0.0, t1 - s1 / bw)
    return LinkModel(overhead_s=overhead, bw_bps=bw)

# ---------------------------------------------------------------------------
# Wire format & iperf
# ---------------------------------------------------------------------------
def pack(o):  return pickle.dumps(o, protocol=pickle.HIGHEST_PROTOCOL)
def unpack(b): return pickle.loads(b)

def tensor_to_wire(t):
    return {"shape": tuple(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
            "data": t.detach().cpu().numpy().tobytes()}

def wire_to_tensor(d):
    arr = np.frombuffer(d["data"], dtype=np.dtype(d["dtype"])).reshape(d["shape"])
    return torch.from_numpy(arr.copy())

def iperf3_probe(target_host, duration=3):
    if shutil.which("iperf3") is None: return None
    try:
        out = subprocess.run(
            ["iperf3", "-c", target_host, "-t", str(duration), "-J"],
            capture_output=True, text=True, timeout=duration + 10,
        )
        if out.returncode != 0: return None
        data = json.loads(out.stdout)
        return data["end"]["sum_received"]["bits_per_second"] / 8.0
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    model: str = "vgg16"
    pi_addr: str = "tcp://127.0.0.1:5555"
    laptop_addr: str = "tcp://127.0.0.1:5556"
    laptop_host: str = "127.0.0.1"
    pc_host: str = "127.0.0.1"
    initial_split: tuple = (10, 30)
    profile_runs: int = 50
    steady_runs: int = 100
    total_runs: int = 500
    warmup: int = 3
    weights: dict = field(default_factory=lambda: {"pi_e": 0.6, "tot_e": 0.3, "lat": 0.1})
    switch_threshold: float = 0.03
    min_pi_layers: int = 1
    max_latency_ms: float = 0.0
    probe_runs_per_split: int = 15   # inferences per reference split in Phase 1
    iperf_on_start: bool = False

    @classmethod
    def load(cls, path):
        if not path or not os.path.exists(path): return cls()
        with open(path) as f: d = yaml.safe_load(f) or {}
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "initial_split" in d: d["initial_split"] = tuple(d["initial_split"])
        return cls(**d)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
class Node:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model, self.head = load_model(cfg.model)
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
            if msg.get("cmd") == "probe":
                self.sock.send(pack({"ack": True})); continue
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
        if self.cfg.iperf_on_start:
            bw = iperf3_probe(self.cfg.pc_host)
            if bw: print(f"[LAP] iperf3 Laptop->PC = {bw/1e6:.1f} MB/s")
        while True:
            msg = unpack(self.front.recv())
            if msg.get("cmd") == "stop":
                self.back.send(pack({"cmd": "stop"})); _ = self.back.recv()
                self.front.send(pack({"ok": True})); break
            if msg.get("cmd") == "probe":
                self.front.send(pack({"ack": True})); continue
            if msg.get("cmd") == "probe_pc":
                payload = b"\x00" * msg["size"]
                t0 = time.perf_counter()
                self.back.send(pack({"cmd": "probe", "data": payload}))
                _ = self.back.recv()
                self.front.send(pack({"elapsed": time.perf_counter() - t0}))
                continue
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
        self.pl_link = LinkModel()
        self.lp_link = LinkModel()
        
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

    def probe_links(self, sizes=(1024, 1_000_000), repeats=2) -> tuple[LinkModel, LinkModel]:
        """Reduced overhead two-point bandwidth probe."""
        pl_t: dict[int, float] = {}
        lp_t: dict[int, float] = {}
        for size in sizes:
            payload = b"\x00" * size
            samples_pl = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                self.sock.send(pack({"cmd": "probe", "data": payload}))
                _ = self.sock.recv()
                samples_pl.append(time.perf_counter() - t0)
            pl_t[size] = mean(samples_pl)
            
            samples_lp = []
            for _ in range(repeats):
                self.sock.send(pack({"cmd": "probe_pc", "size": size}))
                rep = unpack(self.sock.recv())
                samples_lp.append(rep["elapsed"])
            lp_t[size] = mean(samples_lp)
            
        s1, s2 = sizes
        self.pl_link = fit_link(s1, pl_t[s1], s2, pl_t[s2], fallback=self.pl_link)
        self.lp_link = fit_link(s1, lp_t[s1], s2, lp_t[s2], fallback=self.lp_link)
        return self.pl_link, self.lp_link

# ---------------------------------------------------------------------------
# Bandwidth-aware estimator
# ---------------------------------------------------------------------------
def valid_splits(min_pi_layers: int = 1):
    i_min = max(0, min_pi_layers - 1)
    return [(i, j) for i in range(i_min, N_FEATURES) for j in range(i + 1, N_FEATURES)]

def probe_splits_for_model(n_features: int, min_pi_layers: int = 1) -> list[tuple]:
    """Pick reference splits that together give every node grounded measurements.
    Three splits chosen at fifths of the feature range:
      * PC-heavy:  small i, small j (pushes most layers to PC)
      * Balanced:  medium i, medium j
      * Pi-heavy:  large i, medium j (pushes more layers to Pi/Laptop)
    Each split exercises the Laptop with the same fraction of layers, while
    Pi and PC see decreasing/increasing workloads — enough variance to fit
    per-layer compute rates linearly across the whole feature range."""
    n = n_features
    i_min = max(0, min_pi_layers - 1)
    raw = [
        (max(i_min,     n // 5), 2 * n // 5),
        (max(i_min, 2 * n // 5), 3 * n // 5),
        (max(i_min, 3 * n // 5), 4 * n // 5),
    ]
    seen, out = set(), []
    for i, j in raw:
        if i_min <= i < j < n and (i, j) not in seen:
            seen.add((i, j)); out.append((i, j))
    return out

def mean(xs): return sum(xs) / len(xs) if xs else 0.0

def measure_bandwidths(window):
    pl, lp = [], []
    for s in window:
        i, j = s["split"]
        if s["pl_transfer"] > 1e-4: pl.append(PAYLOAD_BYTES_FP32[i] / s["pl_transfer"])
        if s["lp_transfer"] > 1e-4: lp.append(PAYLOAD_BYTES_FP32[j] / s["lp_transfer"])
    return (mean(pl) if pl else 1e8, mean(lp) if lp else 1e8)

def per_layer_rates(window):
    """Calculates compute multipliers utilizing relative profile weights."""
    pi_speeds, lap_speeds, pc_speeds = [], [], []
    lap_w_list, pc_w_list = [], []

    for s in window:
        i, j = s["split"]
        w_pi  = max(get_weight(0, i), 1e-6)
        w_lap = max(get_weight(i + 1, j), 1e-6)
        w_pc  = max(get_weight(j + 1, HEAD_INDEX), 1e-6)  # HEAD_INDEX includes the head

        pi_speeds.append(s["pi_compute"] / w_pi)
        if j > i:                lap_speeds.append(s["lap_compute"] / w_lap)
        if j < LAST_FEATURE:     pc_speeds.append(s["pc_compute"] / w_pc)

        lap_w_list.append(s["lap_energy"])
        pc_w_list.append(s["pc_energy"])

    pi_s  = mean(pi_speeds) if pi_speeds else 0.0
    lap_s = mean(lap_speeds) if lap_speeds else 0.0
    pc_s  = mean(pc_speeds) if pc_speeds else 0.0

    lap_w = mean(lap_w_list) / max(1e-9, mean([s["lap_compute"] for s in window]))
    pc_w  = mean(pc_w_list) / max(1e-9, mean([s["pc_compute"] for s in window]))

    return {"pi_speed": pi_s, "lap_speed": lap_s, "pc_speed": pc_s, "lap_w": lap_w, "pc_w": pc_w}

def estimate_split(cand, rates, pl_link: LinkModel, lp_link: LinkModel):
    i, j = cand
    w_pi  = get_weight(0, i)
    w_lap = get_weight(i + 1, j)
    w_pc  = get_weight(j + 1, HEAD_INDEX)

    pi_t  = rates["pi_speed"]  * w_pi
    lap_t = rates["lap_speed"] * w_lap
    pc_t  = rates["pc_speed"]  * w_pc

    tr_pl = pl_link.predict(PAYLOAD_BYTES_FP32[i])
    tr_lp = lp_link.predict(PAYLOAD_BYTES_FP32[j])
    lat = pi_t + lap_t + pc_t + tr_pl + tr_lp

    pi_e  = PiPowerMeter.POWER_W * pi_t
    lap_e = rates["lap_w"] * lap_t
    pc_e  = rates["pc_w"]  * pc_t
    return {"split": cand, "latency": lat, "pi_energy": pi_e,
            "total_energy": pi_e + lap_e + pc_e}

def score(w, pi_e, tot_e, lat, norms=None):
    if norms is None:
        return w["pi_e"] * pi_e + w["tot_e"] * tot_e + w["lat"] * lat
    return (w["pi_e"]  * (pi_e  / max(norms["pi_e"],  1e-9)) +
            w["tot_e"] * (tot_e / max(norms["tot_e"], 1e-9)) +
            w["lat"]   * (lat   / max(norms["lat"],   1e-9)))

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_header():
    print(f"{'run':>4} {'split':>9} {'pi_E':>8} {'lap_E':>8} {'pc_E':>8} "
          f"{'tot_E':>8} {'lat_ms':>8}")

def print_row(run, s):
    i, j = s["split"]
    print(f"{run:>4} {f'({i:>2},{j:>2})':>9} "
          f"{s['pi_energy']:>8.3f} {s['lap_energy']:>8.3f} {s['pc_energy']:>8.3f} "
          f"{s['total_energy']:>8.3f} {s['latency']*1000:>8.1f}")

def final_report(all_stats, baseline_split, baseline_score, weights, norms):
    if not all_stats:
        print("\n[report] no samples collected"); return

    by_split: dict[tuple, list[dict]] = {}
    for s in all_stats:
        by_split.setdefault(s["split"], []).append(s)

    def stats(vals):
        if not vals: return (0.0, 0.0, 0.0, 0.0)
        mu = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return (mu, sd, min(vals), max(vals))

    n = len(all_stats)
    print(f"\n{'='*78}")
    print(f"  FINAL REPORT  -  {n} inferences across {len(by_split)} split(s)")
    print(f"{'='*78}")

    metric_defs = [
        ("Pi compute",       "pi_compute",   1000, "ms"),
        ("Laptop compute",   "lap_compute",  1000, "ms"),
        ("PC compute",       "pc_compute",   1000, "ms"),
        ("Pi->Lap transfer", "pl_transfer",  1000, "ms"),
        ("Lap->PC transfer", "lp_transfer",  1000, "ms"),
        ("End-to-end latency","latency",     1000, "ms"),
        ("Pi energy",        "pi_energy",       1, "J"),
        ("Laptop energy",    "lap_energy",      1, "J"),
        ("PC energy",        "pc_energy",       1, "J"),
        ("Total energy",     "total_energy",    1, "J"),
    ]
    print(f"\n  OVERALL ({n} samples)")
    print(f"  {'metric':<22}{'avg':>12}{'stdev':>12}{'min':>12}{'max':>12}  unit")
    print(f"  {'-'*22}{'-'*12}{'-'*12}{'-'*12}{'-'*12}  ----")
    for label, key, scale, unit in metric_defs:
        vals = [s[key] * scale for s in all_stats]
        mu, sd, lo, hi = stats(vals)
        print(f"  {label:<22}{mu:>12.3f}{sd:>12.3f}{lo:>12.3f}{hi:>12.3f}  {unit}")

    print(f"\n  PER-SPLIT BREAKDOWN")
    print(f"  {'split':<10}{'n':>6}{'lat_ms':>12}{'pi_E_J':>12}"
          f"{'lap_E_J':>12}{'pc_E_J':>12}{'tot_E_J':>12}{'score':>12}")
    print(f"  {'-'*10}{'-'*6}{'-'*12}{'-'*12}{'-'*12}{'-'*12}{'-'*12}{'-'*12}")
    rows = []
    for split, samples in sorted(by_split.items()):
        lat_ms = mean([s["latency"]*1000 for s in samples])
        pi_e   = mean([s["pi_energy"]    for s in samples])
        lap_e  = mean([s["lap_energy"]   for s in samples])
        pc_e   = mean([s["pc_energy"]    for s in samples])
        tot_e  = mean([s["total_energy"] for s in samples])
        sc     = score(weights, pi_e, tot_e, lat_ms/1000, norms)
        rows.append((split, len(samples), lat_ms, pi_e, lap_e, pc_e, tot_e, sc))
        print(f"  ({split[0]:>2},{split[1]:>2}){'':<2}{len(samples):>6}"
              f"{lat_ms:>12.2f}{pi_e:>12.4f}{lap_e:>12.4f}"
              f"{pc_e:>12.4f}{tot_e:>12.4f}{sc:>12.4f}")

    cum_pi  = sum(s["pi_energy"]    for s in all_stats)
    cum_lap = sum(s["lap_energy"]   for s in all_stats)
    cum_pc  = sum(s["pc_energy"]    for s in all_stats)
    cum_tot = sum(s["total_energy"] for s in all_stats)
    cum_lat = sum(s["latency"]      for s in all_stats)
    print(f"\n  CUMULATIVE ({n} inferences)")
    print(f"    Pi energy total       {cum_pi:>10.2f} J")
    print(f"    Laptop energy total   {cum_lap:>10.2f} J")
    print(f"    PC energy total       {cum_pc:>10.2f} J")
    print(f"    System energy total   {cum_tot:>10.2f} J")
    print(f"    Wall-clock total      {cum_lat:>10.2f} s")
    print(f"    Throughput            {n/cum_lat:>10.2f} inf/s")

    base_samples = by_split.get(baseline_split, [])
    chosen_split = max(by_split, key=lambda k: len(by_split[k]))
    if chosen_split != baseline_split and base_samples:
        chosen_samples = by_split[chosen_split]
        b_pi  = mean([s["pi_energy"]    for s in base_samples])
        b_tot = mean([s["total_energy"] for s in base_samples])
        b_lat = mean([s["latency"]      for s in base_samples]) * 1000
        c_pi  = mean([s["pi_energy"]    for s in chosen_samples])
        c_tot = mean([s["total_energy"] for s in chosen_samples])
        c_lat = mean([s["latency"]      for s in chosen_samples]) * 1000
        def pct(b, c): return (b - c) / b * 100 if b else 0.0
        print(f"\n  ADAPTATION GAIN  (baseline {baseline_split} -> chosen {chosen_split})")
        print(f"    {'metric':<18}{'baseline':>12}{'chosen':>12}{'delta %':>12}")
        print(f"    {'-'*18}{'-'*12}{'-'*12}{'-'*12}")
        print(f"    {'Pi energy (J)':<18}{b_pi:>12.4f}{c_pi:>12.4f}{pct(b_pi,c_pi):>11.1f}%")
        print(f"    {'Total energy (J)':<18}{b_tot:>12.4f}{c_tot:>12.4f}{pct(b_tot,c_tot):>11.1f}%")
        print(f"    {'Latency (ms)':<18}{b_lat:>12.2f}{c_lat:>12.2f}{pct(b_lat,c_lat):>11.1f}%")

    bw_pl, bw_lp = measure_bandwidths(all_stats)
    print(f"\n  LINK BANDWIDTH (derived from telemetry)")
    print(f"    Pi  -> Laptop      {bw_pl/1e6:>8.2f} MB/s")
    print(f"    Laptop -> PC       {bw_lp/1e6:>8.2f} MB/s")
    print(f"{'='*78}")

# ---------------------------------------------------------------------------
# Driver (Pi)
# ---------------------------------------------------------------------------
def find_best(rates, pl_link, lp_link, weights, baseline_score, norms,
              min_pi_layers=1, exclude=None, max_latency_s=0.0):
    best, best_sc = None, float("inf")
    for cand in valid_splits(min_pi_layers):
        if cand == exclude: continue
        est = estimate_split(cand, rates, pl_link, lp_link)
        # Deadline pre-filter: reject candidates whose ESTIMATED latency
        # exceeds the deadline. Self-correcting: if conditions change, the
        # estimate changes too, and the split becomes eligible again.
        if max_latency_s > 0 and est["latency"] > max_latency_s: continue
        sc = score(weights, est["pi_energy"], est["total_energy"],
                   est["latency"], norms)
        if sc > baseline_score: continue
        if sc < best_sc: best, best_sc = cand, sc
    return best, best_sc

def run_pi_driver(cfg):
    pi = PiNode(cfg)
    print(f"[init] model={cfg.model}  features={N_FEATURES}  "
          f"head_index={HEAD_INDEX}")
    # Validate initial split against the loaded model
    i0, j0 = cfg.initial_split
    if not (0 <= i0 < j0 < N_FEATURES):
        raise ValueError(f"initial_split={cfg.initial_split} invalid for "
                         f"model={cfg.model} (need 0 <= i < j < {N_FEATURES})")
    if cfg.iperf_on_start:
        bw = iperf3_probe(cfg.laptop_host)
        if bw: print(f"[PI ] iperf3 Pi->Laptop = {bw/1e6:.1f} MB/s")

    dummy = torch.randn(1, 3, 224, 224)
    all_stats = []
    phase1_samples = []   # baseline + probe samples, used to ground per-layer rates
    current_split = cfg.initial_split
    deadline_s = cfg.max_latency_ms / 1000.0 if cfg.max_latency_ms > 0 else 0.0
    print_header()

    # ---- Phase 1a: baseline split establishes norms + baseline_score ----
    print(f"\n[phase1a] running baseline split={current_split} for "
          f"{cfg.profile_runs} inferences")
    baseline_samples = []
    for r in range(cfg.profile_runs):
        s = pi.infer_once(dummy, current_split)
        if r >= cfg.warmup:
            baseline_samples.append(s)
            phase1_samples.append(s)
            all_stats.append(s)
        if (r + 1) % 10 == 0: print_row(r + 1, s)

    base_pi  = mean([s["pi_energy"]    for s in baseline_samples])
    base_tot = mean([s["total_energy"] for s in baseline_samples])
    base_lat = mean([s["latency"]      for s in baseline_samples])
    print(f"[phase1a] baseline pi_e={base_pi:.3f}J tot_e={base_tot:.3f}J "
          f"lat={base_lat*1000:.1f}ms")
    if deadline_s > 0 and base_lat > deadline_s:
        print(f"[WARN] baseline latency {base_lat*1000:.1f}ms exceeds deadline "
              f"{cfg.max_latency_ms:.0f}ms — deadline may be unrealistic")

    # ---- Phase 1b: probe reference splits to ground per-layer rates ----
    probes = probe_splits_for_model(N_FEATURES, cfg.min_pi_layers)
    probes = [p for p in probes if p != current_split]   # don't double-run baseline
    probe_samples = []   # tracked separately so norms don't depend on initial_split
    if probes:
        print(f"[phase1b] probing {len(probes)} reference splits "
              f"({cfg.probe_runs_per_split} runs each) to ground per-layer rates")
        for probe in probes:
            print(f"[probe] split={probe}")
            for r in range(cfg.probe_runs_per_split):
                s = pi.infer_once(dummy, probe)
                if r >= cfg.warmup:
                    probe_samples.append(s)
                    phase1_samples.append(s)
                    all_stats.append(s)
    else:
        print("[phase1b] no probe splits available for this model")

    # ---- Normalization anchor: derived from PROBE samples, not baseline ----
    # The probes are deterministic for a given model (probe_splits_for_model
    # returns the same set every time), so norms become independent of the
    # user's initial_split choice. This makes weight semantics reproducible:
    # the same weights → the same chosen split, regardless of initial_split.
    if probe_samples:
        norm_pi  = mean([s["pi_energy"]    for s in probe_samples])
        norm_tot = mean([s["total_energy"] for s in probe_samples])
        norm_lat = mean([s["latency"]      for s in probe_samples])
        norm_source = "probes"
    else:
        norm_pi, norm_tot, norm_lat = base_pi, base_tot, base_lat
        norm_source = "baseline (no probes ran)"
    norms = {"pi_e": norm_pi, "tot_e": norm_tot, "lat": norm_lat}
    # Baseline_score still measured from the user's initial split — it's the
    # threshold to beat. Norms are the units, baseline is the bar.
    baseline_score = score(cfg.weights, base_pi, base_tot, base_lat, norms)
    print(f"[norms] anchored to {norm_source}: pi_e={norm_pi:.3f}J "
          f"tot_e={norm_tot:.3f}J lat={norm_lat*1000:.1f}ms")
    print(f"[baseline] split={cfg.initial_split} score={baseline_score:.4f} "
          f"(threshold to beat)")

    # ---- Phase 1c: fit grounded rates and run optimizer ----
    rates = per_layer_rates(phase1_samples)
    print(f"[phase1c] rates fit from {len(phase1_samples)} grounded samples "
          f"across {len({s['split'] for s in phase1_samples})} splits")
    print(f"[probe] re-measuring link models...")
    pl_link, lp_link = pi.probe_links()
    print(f"[link] Pi->Lap  {pl_link}")
    print(f"[link] Lap->PC  {lp_link}")

    best, best_sc = find_best(rates, pl_link, lp_link, cfg.weights, baseline_score,
                              norms, min_pi_layers=cfg.min_pi_layers,
                              max_latency_s=deadline_s)
    if best is not None:
        print(f"[opt] best estimated split={best} score={best_sc:.4f}")
        current_split = best
    else:
        print("[opt] no candidate beats baseline; holding static split")

    # ---- Phase 2: steady state with periodic re-evaluation ----
    # Steady state combines phase1_samples with the recent window when fitting
    # rates, so the rate model never loses its broad grounding even if the
    # current split only exercises one node heavily.
    phase1_runs_used = len(all_stats)
    run = phase1_runs_used
    window = []
    runs_in_window = 0
    while run < cfg.total_runs:
        s = pi.infer_once(dummy, current_split)
        run += 1; runs_in_window += 1
        if runs_in_window > cfg.warmup:
            window.append(s); all_stats.append(s)
        if run % 10 == 0: print_row(run, s)

        if runs_in_window >= cfg.steady_runs and window:
            measured_lat = mean([s["latency"] for s in window])
            cur_rates = per_layer_rates(phase1_samples + window)  # combined
            pl_link, lp_link = pi.probe_links()

            cur_est = estimate_split(current_split, cur_rates, pl_link, lp_link)
            cur_sc = score(cfg.weights, cur_est["pi_energy"],
                           cur_est["total_energy"], cur_est["latency"], norms)
            cand_best, cand_sc = find_best(cur_rates, pl_link, lp_link,
                                           cfg.weights, baseline_score, norms,
                                           min_pi_layers=cfg.min_pi_layers,
                                           exclude=current_split,
                                           max_latency_s=deadline_s)
            if cand_best is None:
                cand_best, cand_sc = current_split, cur_sc
            improvement = (cur_sc - cand_sc) / max(cur_sc, 1e-9)
            print(f"[link] Pi->Lap {pl_link}")
            print(f"[link] Lap->PC {lp_link}")

            # Force a switch if measured latency just violated the deadline,
            # even when the improvement margin is small. The new probe data
            # will already have biased the estimator away from bad splits.
            deadline_violated = (deadline_s > 0 and measured_lat > deadline_s)
            must_switch = deadline_violated and cand_best != current_split
            if must_switch or (cand_best != current_split
                               and improvement >= cfg.switch_threshold):
                tag = "FORCED" if must_switch else "switch"
                print(f"[{tag}] {current_split} -> {cand_best} "
                      f"({improvement*100:.1f}% better, "
                      f"measured {measured_lat*1000:.0f}ms)")
                current_split = cand_best
            elif deadline_violated:
                # Estimator can't find anything better either; revert to
                # the static baseline as the safest known state.
                if current_split != cfg.initial_split:
                    print(f"[FALLBACK] deadline violated and no better "
                          f"alternative; reverting to baseline {cfg.initial_split}")
                    current_split = cfg.initial_split
                else:
                    print(f"[STUCK] baseline itself violates deadline "
                          f"({measured_lat*1000:.0f}ms > "
                          f"{cfg.max_latency_ms:.0f}ms); holding")
            else:
                print(f"[hold] split={current_split} "
                      f"(best alt {cand_best}, improvement {improvement*100:.1f}%)")
            window = []
            runs_in_window = 0

    pi.stop_chain()
    final_report(all_stats, cfg.initial_split, baseline_score, cfg.weights, norms)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=["pi", "laptop", "pc"])
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    cfg = Config.load(args.config)
    if args.role == "pc":       PCNode(cfg).serve()
    elif args.role == "laptop": LaptopNode(cfg).serve()
    else:                       run_pi_driver(cfg)

if __name__ == "__main__":
    main()