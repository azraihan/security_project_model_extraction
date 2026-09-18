"""
evaluate.py
===========
Step 5: score the substitute on a clean CIFAR-10 *test* set with the two metrics
the reference papers distinguish (Jagielski et al.):

  accuracy  -- does f_S do the task well?      P[ argmax f_S(x) == y_true ]
  fidelity  -- does f_S agree with the victim?  P[ argmax f_S(x) == argmax f_V(x) ]
  (also reports victim accuracy for reference)

Fidelity is the more demanding, extraction-specific metric: it rewards the
substitute for reproducing the victim's *mistakes*, not just being correct.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from config import VICTIM_CKPT
from data import cifar10_eval_arrays, normalize_uint8_batch
from models import build_model
from victim_train import load_victim


def _predict_all(model, images_uint8, device, batch_size=512):
    model.eval()
    preds = np.empty(len(images_uint8), dtype=np.int64)
    with torch.no_grad():
        for s in range(0, len(images_uint8), batch_size):
            e = min(s + batch_size, len(images_uint8))
            x = normalize_uint8_batch(images_uint8[s:e]).to(device)
            preds[s:e] = model(x).argmax(1).cpu().numpy()
    return preds


def evaluate_substitute(substitute, victim=None, device=None, download=True,
                        eval_arrays=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    substitute = substitute.to(device)
    if eval_arrays is not None:
        images, y_true = eval_arrays          # (uint8 NHWC, int64 labels)
    else:
        images, y_true = cifar10_eval_arrays(download=download)

    sub_pred = _predict_all(substitute, images, device)
    out = {"substitute_accuracy": float((sub_pred == y_true).mean())}

    if victim is not None:
        victim = victim.to(device)
        vic_pred = _predict_all(victim, images, device)
        out["victim_accuracy"] = float((vic_pred == y_true).mean())
        out["fidelity"] = float((sub_pred == vic_pred).mean())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--substitute", required=True)
    p.add_argument("--victim", default=VICTIM_CKPT)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sub_ckpt = torch.load(a.substitute, map_location=device)
    sub = build_model(sub_ckpt["arch"], sub_ckpt.get("num_classes", 10))
    sub.load_state_dict(sub_ckpt["state_dict"])
    victim, _ = load_victim(a.victim, device=device)

    res = evaluate_substitute(sub, victim, device=device)
    print("[evaluate]")
    for k, v in res.items():
        print(f"  {k:22s} {v:.4f}")


if __name__ == "__main__":
    main()
