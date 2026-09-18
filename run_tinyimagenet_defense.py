"""
run_tinyimagenet_defense.py
===========================
Adds a `tinyimagenet` row-set to an EXISTING defense-results CSV, without
disturbing the cifar10/cifar100 rows already in it.

It reproduces `run_sweep.experiment_defense` -- same five defense plans, same
budgets, same seed, same substitute arch/epochs -- but uses the Tiny ImageNet
out-of-distribution transfer pool, and it is **resumable**:

  * Every completed (defense, budget) row is appended to the output CSV
    immediately, so progress is durable the moment it happens.
  * On start-up it reads the output CSV, notes which (defense, budget) pairs are
    already present for `transfer=tinyimagenet`, and skips them.

That means a re-run continues exactly where a previous run stopped (e.g. after a
Colab disconnect), appending only the missing rows and never duplicating one.
Point `--out` at a durable location (e.g. a mounted Google Drive path) to make
this survive a recycled runtime.

Evaluation is unchanged: the substitute is always scored on the clean CIFAR-10
*test* split, exactly like every other row. Tiny ImageNet is only the query pool.
"""

from __future__ import annotations

import argparse
import csv
import os

import torch

from config import RESULTS_DIR, VICTIM_CKPT
from data import transfer_pool
from evaluate import evaluate_substitute
from substitute import train_substitute
from victim_train import load_victim
from run_sweep import VictimServer
from attack import QueryEngine, TransferQuerier

# Same column order as the existing results/defense_results.csv header.
FIELDS = ["defense", "transfer", "queries", "loss",
          "substitute_accuracy", "victim_accuracy", "fidelity"]

# Identical to run_sweep.experiment_defense: (label, defense, attacker loss, kwargs)
PLANS = [
    ("none (soft)", "none", "soft", {}),
    ("round-1 (soft)", "round", "soft", {"round_decimals": 1}),
    ("noise (soft)", "noise", "soft", {"noise_std": 0.1}),
    ("top-1 prob (soft)", "topk", "soft", {"topk": 1}),
    ("label_only (hard)", "label_only", "hard", {}),
]


def _read_rows(path):
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _append_row(path, row):
    """Append one row, writing the header first if the file is new/empty."""
    new_file = not (os.path.exists(path) and os.path.getsize(path) > 0)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--victim", default=VICTIM_CKPT)
    p.add_argument("--transfer", default="tinyimagenet")
    p.add_argument("--budgets", type=int, nargs="+",
                   default=[5000, 10000, 20000, 50000])
    p.add_argument("--sub-arch", default="smallcnn")
    p.add_argument("--sub-epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=None,
                   help="substitute DataLoader workers (default: 2 on CUDA, else 0)")
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "defense_results.csv"))
    p.add_argument("--max-plans", type=int, default=None,
                   help="only run the first N defense plans (for a quick validation run)")
    a = p.parse_args()

    plans = PLANS if a.max_plans is None else PLANS[:a.max_plans]

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    num_workers = a.num_workers if a.num_workers is not None else (2 if device == "cuda" else 0)
    print(f"[tin-defense] device={device} num_workers={num_workers} transfer={a.transfer} "
          f"budgets={a.budgets} sub_arch={a.sub_arch} sub_epochs={a.sub_epochs}")
    print(f"[tin-defense] output (durable resume file): {a.out}")

    # Victim used for the accuracy/fidelity metrics (same checkpoint the server serves).
    victim, meta = load_victim(a.victim, device=device)
    print(f"[tin-defense] victim loaded (meta test_acc={meta.get('test_acc')})")

    # Attacker query pool: Tiny ImageNet, resized to 32x32 (built + cached in data.py).
    pool = transfer_pool(a.transfer)
    pool_n = len(pool)
    print(f"[tin-defense] transfer pool '{a.transfer}': {pool.shape} {pool.dtype}")

    # Resume: which (defense, queries) rows for this transfer are already recorded?
    existing = _read_rows(a.out)
    completed = {
        (r["defense"], int(r["queries"]))
        for r in existing
        if r.get("transfer") == a.transfer and str(r.get("queries", "")).strip().isdigit()
    }
    total_targets = len(plans) * len(a.budgets)
    if completed:
        print(f"[tin-defense] resume: {len(completed)}/{total_targets} "
              f"{a.transfer} rows already present in {a.out}; they will be skipped")

    n_new = 0
    for label, defense, loss, kw in plans:
        remaining = [b for b in a.budgets if (label, min(b, pool_n)) not in completed]
        if not remaining:
            print(f"\n[tin-defense] === plan: {label} -- all budgets already done, skipping ===")
            continue

        print(f"\n[tin-defense] === plan: {label} (defense={defense}) "
              f"remaining budgets={remaining} ===")
        with VictimServer(a.victim, defense=defense, port=a.port, **kw) as srv:
            engine = QueryEngine(srv.url, batch_size=a.batch_size)
            querier = TransferQuerier(engine, pool, seed=a.seed)
            for b in a.budgets:
                q = min(b, pool_n)
                if (label, q) in completed:
                    continue
                images, labels, probs = querier.ensure(b)
                effective_loss = loss
                if loss == "soft" and probs is None:
                    print(f"  [budget {b}] no probabilities available "
                          f"-> falling back to hard loss")
                    effective_loss = "hard"
                sub = train_substitute(images, labels, probs, arch=a.sub_arch,
                                       loss=effective_loss, epochs=a.sub_epochs,
                                       device=device, verbose=False,
                                       num_workers=num_workers)
                res = evaluate_substitute(sub, victim, device=device,
                                          eval_arrays=None, download=True)
                row = {
                    "defense": label,
                    "transfer": a.transfer,
                    "queries": len(images),
                    "loss": effective_loss,
                    "substitute_accuracy": res["substitute_accuracy"],
                    "victim_accuracy": res.get("victim_accuracy"),
                    "fidelity": res.get("fidelity"),
                }
                _append_row(a.out, row)            # durable: written the instant it's done
                completed.add((label, len(images)))
                n_new += 1
                print(f"  [budget {row['queries']:6d}] loss={effective_loss:4s} "
                      f"acc={res['substitute_accuracy']:.4f} "
                      f"fidelity={res.get('fidelity', float('nan')):.4f}  "
                      f"[{len(completed)}/{total_targets} done]")

    print(f"\n[tin-defense] finished. appended {n_new} new row(s); "
          f"{len(completed)}/{total_targets} {a.transfer} rows now in {a.out}")


if __name__ == "__main__":
    main()
