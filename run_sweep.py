"""
run_sweep.py
============
The one-command driver that turns the individual stages into the project's two
headline figures.

  --experiment main
      For each transfer pool x loss strategy, sweep the query budget and plot
      accuracy and fidelity vs number of queries. (Server runs with defense=none.)
      This is *the* plot of the project.

  --experiment defense
      Fix a transfer pool and sweep the budget under each defense (none / round /
      noise / topk / label_only), plotting accuracy vs queries per defense to show
      how each raises the attacker's cost.

The victim server is launched automatically as a subprocess (mirroring the
"two hosts, HTTP" topology) unless you pass --server-url to point at one you have
already started.

  python run_sweep.py --experiment main --victim checkpoints/victim.pt
  python run_sweep.py --experiment defense --transfer cifar100
  python run_sweep.py --experiment main --smoke     # end-to-end, CPU, no download
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np
import requests
import torch

from attack import QueryEngine, TransferQuerier
from config import (CACHE_DIR, ENV_DEFENSE, ENV_NOISE_STD, ENV_QUOTA,
                    ENV_ROUND_DECIMALS, ENV_TOPK, ENV_VICTIM_CKPT, RESULTS_DIR,
                    VICTIM_CKPT)
from data import synthetic_labeled, synthetic_pool, transfer_pool
from evaluate import evaluate_substitute
from substitute import train_substitute
from victim_train import load_victim, train_victim


# --------------------------------------------------------------------------- #
# Victim server as a subprocess                                               #
# --------------------------------------------------------------------------- #
class VictimServer:
    """Launch `uvicorn server:app` with a given defense; poll /health; tear down."""

    def __init__(self, victim_ckpt, defense="none", port=8000,
                 round_decimals=1, topk=1, noise_std=0.1, quota=0):
        self.url = f"http://127.0.0.1:{port}"
        self.port = port
        env = dict(os.environ)
        env.update({
            ENV_VICTIM_CKPT: victim_ckpt,
            ENV_DEFENSE: defense,
            ENV_ROUND_DECIMALS: str(round_decimals),
            ENV_TOPK: str(topk),
            ENV_NOISE_STD: str(noise_std),
            ENV_QUOTA: str(quota),
        })
        self._env = env
        self._proc = None

    def __enter__(self):
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server:app",
             "--host", "127.0.0.1", "--port", str(self.port), "--log-level", "warning"],
            env=self._env, cwd=os.path.dirname(os.path.abspath(__file__)))
        self._wait_healthy()
        return self

    def _wait_healthy(self, timeout=120):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._proc.poll() is not None:
                raise RuntimeError("server process exited before becoming healthy")
            try:
                if requests.get(f"{self.url}/health", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                time.sleep(0.5)
        raise TimeoutError("victim server did not become healthy in time")

    def __exit__(self, *exc):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# --------------------------------------------------------------------------- #
# One (transfer, defense, loss, budget-list) sweep against a running server   #
# --------------------------------------------------------------------------- #
def sweep_over_budgets(server_url, pool, budgets, loss, arch, sub_epochs,
                       victim_model, eval_arrays, seed=0, batch_size=256):
    """Query at growing budgets, train + evaluate a substitute at each. Returns rows."""
    engine = QueryEngine(server_url, batch_size=batch_size)
    querier = TransferQuerier(engine, pool, seed=seed)
    rows = []
    for b in budgets:
        images, labels, probs = querier.ensure(b)
        effective_loss = loss
        if loss == "soft" and probs is None:
            print(f"  [budget {b}] no probabilities available -> falling back to hard loss")
            effective_loss = "hard"
        sub = train_substitute(images, labels, probs, arch=arch,
                               loss=effective_loss, epochs=sub_epochs, verbose=False)
        res = evaluate_substitute(sub, victim_model, eval_arrays=eval_arrays,
                                  download=False if eval_arrays is not None else True)
        row = {"queries": len(images), "loss": effective_loss, **res}
        rows.append(row)
        print(f"  [budget {row['queries']:6d}] loss={effective_loss:4s} "
              f"acc={res['substitute_accuracy']:.4f} "
              f"fidelity={res.get('fidelity', float('nan')):.4f}")
    return rows


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #
def _save_csv(path, rows, extra_cols):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(extra_cols) + ["queries", "loss", "substitute_accuracy",
                               "victim_accuracy", "fidelity"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[results] wrote {path}")


def plot_main(series, victim_acc, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_f) = plt.subplots(1, 2, figsize=(12, 5))
    for label, rows in series.items():
        q = [r["queries"] for r in rows]
        ax_a.plot(q, [r["substitute_accuracy"] for r in rows], marker="o", label=label)
        ax_f.plot(q, [r.get("fidelity", np.nan) for r in rows], marker="o", label=label)
    if victim_acc is not None:
        ax_a.axhline(victim_acc, ls="--", c="k", alpha=0.6,
                     label=f"victim acc ({victim_acc:.3f})")
    for ax, title in ((ax_a, "Accuracy of f_S on true task"),
                      (ax_f, "Fidelity: agreement of f_S with f_V")):
        ax.set_xlabel("number of queries")
        ax.set_ylabel(title.split(":")[0].split(" of")[0].strip().lower())
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Model extraction: cost vs quality", fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"[results] wrote {out_png}")


def plot_defense(series, victim_acc, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, rows in series.items():
        q = [r["queries"] for r in rows]
        ax.plot(q, [r["substitute_accuracy"] for r in rows], marker="o", label=label)
    if victim_acc is not None:
        ax.axhline(victim_acc, ls="--", c="k", alpha=0.6,
                   label=f"victim acc ({victim_acc:.3f})")
    ax.set_xlabel("number of queries")
    ax.set_ylabel("substitute accuracy")
    ax.set_title("Effect of defenses on extraction cost", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"[results] wrote {out_png}")


# --------------------------------------------------------------------------- #
# Experiments                                                                  #
# --------------------------------------------------------------------------- #
def _resolve_pool(name, smoke):
    if smoke:
        return synthetic_pool(2000, seed=hash(name) % 1000)
    return transfer_pool(name)


def experiment_main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    victim, meta = load_victim(args.victim, device=device)
    victim_acc = meta.get("test_acc")

    smoke = args.smoke
    eval_arrays = synthetic_labeled(1000, seed=99) if smoke else None
    transfers = ["cifar10", "cifar100"] if not args.transfer else [args.transfer]
    if smoke:
        transfers = ["cifar10", "cifar100"]
    losses = ["soft", "hard"]

    series, all_rows = {}, []
    with VictimServer(args.victim, defense="none", port=args.port) as srv:
        for tname in transfers:
            pool = _resolve_pool(tname, smoke)
            for loss in losses:
                key = f"{tname} / {loss}"
                print(f"[main] {key}")
                rows = sweep_over_budgets(
                    srv.url, pool, args.budgets, loss, args.sub_arch,
                    args.sub_epochs, victim, eval_arrays, seed=args.seed)
                for r in rows:
                    r.update(transfer=tname)
                series[key] = rows
                all_rows.extend(rows)

    _save_csv(os.path.join(RESULTS_DIR, "main_results.csv"), all_rows,
              extra_cols=["transfer"])
    plot_main(series, victim_acc, os.path.join(RESULTS_DIR, "main_curve.png"))


def experiment_defense(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    victim, meta = load_victim(args.victim, device=device)
    victim_acc = meta.get("test_acc")

    smoke = args.smoke
    eval_arrays = synthetic_labeled(1000, seed=99) if smoke else None
    tname = args.transfer or "cifar100"
    pool = _resolve_pool(tname, smoke)

    # (label shown, defense, loss the attacker uses under it, extra kwargs)
    plans = [
        ("none (soft)", "none", "soft", {}),
        ("round-1 (soft)", "round", "soft", {"round_decimals": 1}),
        ("noise (soft)", "noise", "soft", {"noise_std": 0.1}),
        ("top-1 prob (soft)", "topk", "soft", {"topk": 1}),
        ("label_only (hard)", "label_only", "hard", {}),
    ]

    series, all_rows = {}, []
    for label, defense, loss, kw in plans:
        print(f"[defense] {label}")
        with VictimServer(args.victim, defense=defense, port=args.port, **kw) as srv:
            rows = sweep_over_budgets(
                srv.url, pool, args.budgets, loss, args.sub_arch,
                args.sub_epochs, victim, eval_arrays, seed=args.seed)
            for r in rows:
                r.update(defense=label, transfer=tname)
            series[label] = rows
            all_rows.extend(rows)

    _save_csv(os.path.join(RESULTS_DIR, "defense_results.csv"), all_rows,
              extra_cols=["defense", "transfer"])
    plot_defense(series, victim_acc, os.path.join(RESULTS_DIR, "defense_curve.png"))


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=("main", "defense"), default="main")
    p.add_argument("--victim", default=VICTIM_CKPT)
    p.add_argument("--transfer", default=None, choices=(None, "cifar10", "cifar100"))
    p.add_argument("--budgets", type=int, nargs="+", default=[5000, 10000, 20000, 50000])
    p.add_argument("--sub-arch", default="smallcnn")
    p.add_argument("--sub-epochs", type=int, default=40)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        # Tiny everything; make sure a victim checkpoint exists.
        args.budgets = [200, 500]
        args.sub_epochs = 1
        if not os.path.exists(args.victim):
            print("[smoke] no victim checkpoint; training a 1-epoch synthetic victim")
            train_victim(arch="smallcnn", epochs=1, out=args.victim, smoke=True)

    if args.experiment == "main":
        experiment_main(args)
    else:
        experiment_defense(args)


if __name__ == "__main__":
    main()
