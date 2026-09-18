"""
config.py
=========
Central constants shared by every component of the attack (victim server,
attacker query engine, substitute trainer, evaluator).

Keeping the normalization statistics and the image-transport contract in one
place is what guarantees that the victim, the attacker, and the evaluator all
speak *exactly* the same numerical language. A silent normalization mismatch is
the single most common bug in an extraction pipeline, so it lives here and
nowhere else.
"""

from __future__ import annotations

import base64
import numpy as np

# --- Task constants ---------------------------------------------------------
NUM_CLASSES = 10
IMG_SIZE = 32
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)          # H, W, C (uint8, 0..255) on the wire

# CIFAR-10 channel statistics (standard values used everywhere in the repo).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# --- Default filesystem layout ---------------------------------------------
DATA_ROOT = "./data"
CKPT_DIR = "./checkpoints"
CACHE_DIR = "./cache"
RESULTS_DIR = "./results"

VICTIM_CKPT = f"{CKPT_DIR}/victim.pt"

# --- Server config env-var names (server.py reads these) --------------------
ENV_VICTIM_CKPT = "MODELEXT_VICTIM_CKPT"
ENV_DEFENSE = "MODELEXT_DEFENSE"             # none | label_only | round | topk | noise
ENV_ROUND_DECIMALS = "MODELEXT_ROUND_DECIMALS"
ENV_TOPK = "MODELEXT_TOPK"
ENV_NOISE_STD = "MODELEXT_NOISE_STD"
ENV_QUOTA = "MODELEXT_QUOTA"                 # 0 = unlimited

DEFENSES = ("none", "label_only", "round", "topk", "noise")


# --- Image transport contract ----------------------------------------------
# Images travel as raw uint8 bytes (H,W,C or N,H,W,C, C-order) wrapped in
# base64. The server always applies CIFAR normalization itself, so the attacker
# only ever sends *pixels*, never pre-normalized tensors. This mirrors a
# realistic prediction API: clients upload images, the service preprocesses.

def encode_uint8(arr: np.ndarray) -> str:
    """base64-encode a C-contiguous uint8 array's raw bytes."""
    if arr.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {arr.dtype}")
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def decode_uint8(b64: str, shape) -> np.ndarray:
    """Inverse of :func:`encode_uint8`, reshaped to ``shape``."""
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
