"""
server.py
=========
The victim host. Wraps the trained model f_V behind an HTTP prediction API and
is the *only* way any later stage is allowed to touch the victim.

Endpoints
---------
  GET  /health          -> {"status": "ok", ...}
  POST /predict         -> single image     {"input": b64}                (32,32,3)
  POST /predict_batch   -> batch of images  {"inputs": b64, "shape":[N,32,32,3]}

Defenses (Section 6) are selected at launch time via environment variables and
change *only the response granularity*, never the underlying prediction:

  none        full probability vector + label   (richest signal)
  label_only  label only, probabilities withheld
  round       probabilities rounded to K decimals
  topk        only the top-K probabilities kept (rest set to 0)
  noise       Gaussian noise added to probabilities, then renormalized
              (the reported top-1 label is preserved so honest users are
               unaffected -- this is the "useful to honest users" property)

Launch
------
    # standalone
    python server.py --victim checkpoints/victim.pt --defense none --port 8000

    # or via uvicorn, configured through env vars
    MODELEXT_VICTIM_CKPT=checkpoints/victim.pt MODELEXT_DEFENSE=round \
    MODELEXT_ROUND_DECIMALS=1 uvicorn server:app --port 8000
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from config import (ENV_DEFENSE, ENV_NOISE_STD, ENV_QUOTA, ENV_ROUND_DECIMALS,
                    ENV_TOPK, ENV_VICTIM_CKPT, IMG_SHAPE, VICTIM_CKPT,
                    decode_uint8)
from data import normalize_uint8_batch
from victim_train import load_victim

app = FastAPI(title="Model-Extraction Victim Server")

# Filled in at startup. STATE also tracks per-client query counts, which powers
# both the query-quota defense and the "how many queries did this cost" report.
STATE = {
    "model": None,
    "device": "cpu",
    "defense": "none",
    "round_decimals": 1,
    "topk": 1,
    "noise_std": 0.1,
    "quota": 0,                       # 0 = unlimited
    "counts": defaultdict(int),
}


class SingleReq(BaseModel):
    input: str                        # base64 uint8, shape (32,32,3)


class BatchReq(BaseModel):
    inputs: str                       # base64 uint8
    shape: list                       # [N,32,32,3]


@app.on_event("startup")
def _startup():
    ckpt = os.environ.get(ENV_VICTIM_CKPT, VICTIM_CKPT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_victim(ckpt, device=device)
    STATE.update(
        model=model, device=device,
        defense=os.environ.get(ENV_DEFENSE, "none"),
        round_decimals=int(os.environ.get(ENV_ROUND_DECIMALS, 1)),
        topk=int(os.environ.get(ENV_TOPK, 1)),
        noise_std=float(os.environ.get(ENV_NOISE_STD, 0.1)),
        quota=int(os.environ.get(ENV_QUOTA, 0)),
    )
    print(f"[server] loaded {ckpt} (victim acc {meta.get('test_acc')}) "
          f"defense={STATE['defense']} device={device}")


def _infer(nhwc_uint8: np.ndarray):
    """Run the victim. Returns (labels (N,), clean_probs (N,C))."""
    x = normalize_uint8_batch(nhwc_uint8).to(STATE["device"])
    with torch.no_grad():
        probs = F.softmax(STATE["model"](x), dim=1).cpu().numpy()
    labels = probs.argmax(1).astype(int)
    return labels, probs


def _apply_defense(labels: np.ndarray, probs: np.ndarray):
    """Transform clean outputs per the active defense. Returns (labels, probs_or_None)."""
    d = STATE["defense"]
    if d == "label_only":
        return labels, None
    if d == "round":
        return labels, np.round(probs, STATE["round_decimals"])
    if d == "topk":
        k = STATE["topk"]
        out = np.zeros_like(probs)
        idx = np.argsort(-probs, axis=1)[:, :k]
        for i in range(probs.shape[0]):
            out[i, idx[i]] = probs[i, idx[i]]
        return labels, out
    if d == "noise":
        std = STATE["noise_std"]
        noised = probs + np.random.normal(0.0, std, size=probs.shape)
        noised = np.clip(noised, 0.0, None)
        noised = noised / np.clip(noised.sum(1, keepdims=True), 1e-8, None)
        return labels, noised          # label kept clean on purpose
    return labels, probs               # "none"


def _charge_quota(request: Request, n: int):
    if STATE["quota"] <= 0:
        return
    client = request.client.host if request.client else "unknown"
    STATE["counts"][client] += n
    if STATE["counts"][client] > STATE["quota"]:
        raise HTTPException(status_code=429, detail="query quota exceeded")


@app.get("/health")
def health():
    return {"status": "ok" if STATE["model"] is not None else "loading",
            "defense": STATE["defense"]}


@app.post("/predict")
def predict(req: SingleReq, request: Request):
    _charge_quota(request, 1)
    arr = decode_uint8(req.input, IMG_SHAPE)[None]            # (1,32,32,3)
    labels, probs = _apply_defense(*_infer(arr))
    resp = {"label": int(labels[0])}
    if probs is not None:
        resp["probabilities"] = probs[0].round(6).tolist()
    return resp


@app.post("/predict_batch")
def predict_batch(req: BatchReq, request: Request):
    shape = tuple(int(s) for s in req.shape)
    if len(shape) != 4 or shape[1:] != IMG_SHAPE:
        raise HTTPException(400, f"shape must be [N,32,32,3], got {shape}")
    _charge_quota(request, shape[0])
    arr = decode_uint8(req.inputs, shape)
    labels, probs = _apply_defense(*_infer(arr))
    resp = {"labels": labels.astype(int).tolist()}
    if probs is not None:
        resp["probabilities"] = probs.round(6).tolist()
    return resp


def main():
    import argparse
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--victim", default=VICTIM_CKPT)
    p.add_argument("--defense", default="none",
                   choices=("none", "label_only", "round", "topk", "noise"))
    p.add_argument("--round-decimals", type=int, default=1)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--quota", type=int, default=0)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()

    os.environ[ENV_VICTIM_CKPT] = a.victim
    os.environ[ENV_DEFENSE] = a.defense
    os.environ[ENV_ROUND_DECIMALS] = str(a.round_decimals)
    os.environ[ENV_TOPK] = str(a.topk)
    os.environ[ENV_NOISE_STD] = str(a.noise_std)
    os.environ[ENV_QUOTA] = str(a.quota)
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
