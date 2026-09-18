"""
attack.py
=========
The attacker's query engine and response logger (Table 1, steps 2-3).

It selects images from a transfer pool, sends them to the victim's prediction
API in batches until the query budget is spent, and logs every
(image, victim label, victim probs) triple. Nothing here needs the victim's
weights, architecture, or training data -- only the public API.

A small caching layer (`TransferQuerier`) fixes one shuffled ordering of the
pool and remembers what it has already queried, so a budget sweep
5k -> 10k -> 20k only ever pays for the *new* queries at each step. Because the
victim is deterministic under the non-randomized defenses, re-querying the same
image would return the same answer anyway; caching just avoids the waste. (Under
the `noise` defense the cached draw is reused across budgets, which is fine for
the budget curve.)

Usage
-----
    python attack.py --server-url http://127.0.0.1:8000 \
        --transfer cifar100 --budget 20000 --out cache/cifar100_none.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import requests
from tqdm import tqdm

from config import NUM_CLASSES, encode_uint8
from data import transfer_pool


class QueryEngine:
    """Thin HTTP client around the victim's /predict_batch endpoint."""

    def __init__(self, base_url: str, batch_size: int = 256, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def predict_batch(self, images_uint8: np.ndarray):
        """images_uint8: (n,32,32,3) uint8 -> (labels (n,), probs (n,C) or None)."""
        payload = {"inputs": encode_uint8(images_uint8),
                   "shape": list(images_uint8.shape)}
        r = requests.post(f"{self.base_url}/predict_batch", json=payload,
                          timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        labels = np.array(data["labels"], dtype=np.int64)
        probs = (np.array(data["probabilities"], dtype=np.float32)
                 if "probabilities" in data else None)
        return labels, probs

    def query(self, images_uint8: np.ndarray, progress: bool = True):
        """Query an arbitrary set of images, batching under the hood."""
        n = len(images_uint8)
        all_labels = np.empty(n, dtype=np.int64)
        all_probs = None
        it = range(0, n, self.batch_size)
        if progress:
            it = tqdm(it, desc="querying", unit="batch")
        for s in it:
            e = min(s + self.batch_size, n)
            labels, probs = self.predict_batch(images_uint8[s:e])
            all_labels[s:e] = labels
            if probs is not None:
                if all_probs is None:
                    all_probs = np.zeros((n, probs.shape[1]), dtype=np.float32)
                all_probs[s:e] = probs
        return all_labels, all_probs


class TransferQuerier:
    """A transfer pool + a growable cache of victim answers over a fixed order."""

    def __init__(self, engine: QueryEngine, pool_uint8: np.ndarray, seed: int = 0):
        self.engine = engine
        self.pool = pool_uint8
        self.order = np.random.default_rng(seed).permutation(len(pool_uint8))
        self.n_done = 0
        self.images = np.empty((0,) + pool_uint8.shape[1:], dtype=np.uint8)
        self.labels = np.empty((0,), dtype=np.int64)
        self.probs = None
        self._probs_supported = None

    def ensure(self, budget: int):
        """Guarantee the first ``budget`` images in the fixed order are queried."""
        budget = min(budget, len(self.pool))
        if budget > self.n_done:
            idx = self.order[self.n_done:budget]
            new_imgs = self.pool[idx]
            new_labels, new_probs = self.engine.query(new_imgs)

            self.images = np.concatenate([self.images, new_imgs], 0)
            self.labels = np.concatenate([self.labels, new_labels], 0)
            if new_probs is not None:
                self._probs_supported = True
                self.probs = (new_probs if self.probs is None
                              else np.concatenate([self.probs, new_probs], 0))
            else:
                self._probs_supported = False
            self.n_done = budget

        return (self.images[:budget], self.labels[:budget],
                None if not self._probs_supported else self.probs[:budget])

    @property
    def probs_supported(self) -> bool:
        return bool(self._probs_supported)


def save_cache(path, images, labels, probs):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    kw = {"images": images, "labels": labels}
    if probs is not None:
        kw["probs"] = probs
    np.savez_compressed(path, **kw)


def load_cache(path):
    d = np.load(path)
    return d["images"], d["labels"], (d["probs"] if "probs" in d else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server-url", default="http://127.0.0.1:8000")
    p.add_argument("--transfer", default="cifar100", choices=("cifar10", "cifar100"))
    p.add_argument("--budget", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="cache/queried.npz")
    a = p.parse_args()

    engine = QueryEngine(a.server_url, batch_size=a.batch_size)
    print("[attack] server health:", engine.health())
    pool = transfer_pool(a.transfer)
    q = TransferQuerier(engine, pool, seed=a.seed)
    imgs, labels, probs = q.ensure(a.budget)
    save_cache(a.out, imgs, labels, probs)
    kind = "soft (probabilities)" if probs is not None else "hard (labels only)"
    print(f"[attack] logged {len(imgs)} pairs [{kind}] -> {a.out}")


if __name__ == "__main__":
    main()
