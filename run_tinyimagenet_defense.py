"""
run_tinyimagenet_defense.py
===========================
One-off driver that adds a `tinyimagenet` row-set to the EXISTING
`results/defense_results.csv`, without touching the cifar10/cifar100 rows that
are already there.

It reproduces `run_sweep.experiment_defense` exactly -- same five defense plans,
same budgets, same seed, same substitute arch/epochs -- but with two differences:

  1. It uses the `tinyimagenet` transfer pool (out-of-distribution, 64->32).
  2. It trains/evaluates on Apple MPS with `num_workers=0` (this machine has no
     CUDA; MPS + a worker-free loader is the fast path here).

Evaluation is unchanged: the substitute is always scored on the clean CIFAR-10
*test* split, exactly like every other row in the CSV. Tiny ImageNet is only the
attacker's query pool.

Rows are written to a partial file as they complete, then appended to the main
CSV in one step at the very end (so a crash can never corrupt the existing rows).
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
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "defense_results.csv"))
    p.add_argument("--partial",
                   default=os.path.join(RESULTS_DIR, "_defense_tinyimagenet_partial.csv"))
    p.add_argument("--max-plans", type=int, default=None,
                   help="only run the first N defense plans (for a quick validation run)")
    a = p.parse_args()

    plans = PLANS if a.max_plans is None else PLANS[:a.max_plans]

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[tin-defense] device={device} transfer={a.transfer} "
          f"budgets={a.budgets} sub_arch={a.sub_arch} sub_epochs={a.sub_epochs}")

    # Victim used for the accuracy/fidelity metrics (same checkpoint the server serves).
    victim, meta = load_victim(a.victim, device=device)
    print(f"[tin-defense] victim loaded (meta test_acc={meta.get('test_acc')})")

    # Attacker query pool: Tiny ImageNet, resized to 32x32 (built + cached in data.py).
    pool = transfer_pool(a.transfer)
    print(f"[tin-defense] transfer pool '{a.transfer}': {pool.shape} {pool.dtype}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Fresh partial file with header (so re-running this script starts clean).
    with open(a.partial, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    all_rows = []
    for label, defense, loss, kw in plans:
        print(f"\n[tin-defense] === plan: {label} (defense={defense}) ===")
        with VictimServer(a.victim, defense=defense, port=a.port, **kw) as srv:
            engine = QueryEngine(srv.url, batch_size=a.batch_size)
            querier = TransferQuerier(engine, pool, seed=a.seed)
            for b in a.budgets:
                images, labels, probs = querier.ensure(b)
                effective_loss = loss
                if loss == "soft" and probs is None:
                    print(f"  [budget {b}] no probabilities available "
                          f"-> falling back to hard loss")
                    effective_loss = "hard"
                sub = train_substitute(images, labels, probs, arch=a.sub_arch,
                                       loss=effective_loss, epochs=a.sub_epochs,
                                       device=device, verbose=False, num_workers=0)
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
                all_rows.append(row)
                # Durable incremental write, so progress survives an interruption.
                with open(a.partial, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
                print(f"  [budget {row['queries']:6d}] loss={effective_loss:4s} "
                      f"acc={res['substitute_accuracy']:.4f} "
                      f"fidelity={res.get('fidelity', float('nan')):.4f}")

    # Append to the existing CSV in one shot (existing rows untouched, no new header).
    file_exists = os.path.exists(a.out) and os.path.getsize(a.out) > 0
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\n[tin-defense] appended {len(all_rows)} tinyimagenet rows -> {a.out}")


if __name__ == "__main__":
    main()
