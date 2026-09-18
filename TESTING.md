# Testing Procedure

This document is the step-by-step procedure to verify the attack works and to
produce the results in the report. It has three layers:

1. **Offline smoke test** — proves the pipeline wires together (no GPU/download).
2. **Component tests** — verify each stage in isolation.
3. **Full experiment** — the real CIFAR run that produces the headline figures.

Run everything from the project root with the virtualenv active.

---

## 0. Preconditions

```bash
pip install -r requirements.txt
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

Expected: imports succeed. `cuda: True` on your GPU box (CPU also works, slower).

---

## 1. Offline smoke test (≈1 minute, CPU, no CIFAR download)

This is the fastest confidence check. It trains a 1-epoch synthetic victim and
runs the entire loop — server, HTTP queries, both losses, evaluation, plotting —
on random data.

```bash
python run_sweep.py --experiment main --smoke
python run_sweep.py --experiment defense --smoke
```

**Pass criteria**

- Both commands exit 0.
- Console shows a `[budget ...] acc=... fidelity=...` line for each configuration.
- `label_only` prints `falling back to hard loss` (probabilities are withheld, so
  soft-label training is impossible — the code must detect this automatically).
- These files appear and are non-empty:
  `results/main_results.csv`, `results/main_curve.png`,
  `results/defense_results.csv`, `results/defense_curve.png`.

> Note: on synthetic data the accuracy/fidelity numbers are meaningless (the toy
> victim is near-constant). This layer tests **plumbing**, not attack quality.

---

## 2. Component tests

### 2.1 Victim trains and saves

```bash
python victim_train.py --arch resnet18 --epochs 60
```

**Pass:** test accuracy climbs each epoch to roughly **0.93–0.95**; a checkpoint
is written to `checkpoints/victim.pt`.

### 2.2 Server serves and defenses change the output

```bash
python server.py --victim checkpoints/victim.pt --defense none --port 8000
# in another terminal:
curl -s localhost:8000/health
```

Then exercise the endpoints and defenses with the provided helper:

```bash
python - <<'PY'
import numpy as np, requests
from config import encode_uint8
img = np.random.randint(0,256,(4,32,32,3),np.uint8)
r = requests.post("http://127.0.0.1:8000/predict_batch",
                  json={"inputs": encode_uint8(img), "shape":[4,32,32,3]}).json()
print("keys:", list(r), "labels:", r["labels"])
PY
```

**Pass, by defense** (restart the server with each `--defense`):

| `--defense` | `probabilities` present? | other checks |
|---|---|---|
| `none` | yes | vector sums to ≈1 |
| `label_only` | **no** | only `labels` returned |
| `round --round-decimals 1` | yes | values are multiples of 0.1 |
| `topk --topk 1` | yes | exactly 1 non-zero entry per row |
| `noise --noise-std 0.1` | yes | vector still sums to ≈1; top-1 label matches the `none` label for the same input |

Rate-limit check:

```bash
python server.py --victim checkpoints/victim.pt --quota 100 --port 8000
# querying more than 100 images from one client should start returning HTTP 429
```

### 2.3 Attacker queries and logs

```bash
python attack.py --server-url http://127.0.0.1:8000 \
    --transfer cifar100 --budget 5000 --out cache/c100_5k.npz
```

**Pass:** progress bar reaches 5000; console reports `logged 5000 pairs
[soft (probabilities)]`; `cache/c100_5k.npz` exists. Re-running with
`--budget 10000` on the same `TransferQuerier` in a sweep only queries the new
5000 (verified by the sweep, see §3).

### 2.4 Substitute trains

```bash
python substitute.py --cache cache/c100_5k.npz --arch smallcnn --loss soft \
    --out checkpoints/sub.pt
```

**Pass:** training loss decreases across epochs; `checkpoints/sub.pt` written.
Repeat with `--loss hard` to test the label-only path.

### 2.5 Evaluation reports both metrics

```bash
python evaluate.py --substitute checkpoints/sub.pt --victim checkpoints/victim.pt
```

**Pass:** prints `substitute_accuracy`, `victim_accuracy`, and `fidelity`, all in
[0,1], with `victim_accuracy ≈ 0.93–0.95`.

---

## 3. Full experiment (produces the report figures)

### 3.1 Main curve — cost vs quality

```bash
python run_sweep.py --experiment main --victim checkpoints/victim.pt
```

**What it does:** for each of {cifar10, cifar100} × {soft, hard}, sweeps budgets
5k→10k→20k→50k, training and scoring a substitute at each.

**Pass / expected shape of results** (`results/main_results.csv`, `main_curve.png`):

- Substitute accuracy **rises with budget and plateaus** a few points below the
  victim (dashed line).
- **Soft** loss beats **hard** at equal budget (dark knowledge in the probabilities).
- **cifar100 (OOD)** works without the victim's data — its curve is somewhat below
  same-domain cifar10 but clearly well above chance (0.10). This is the central
  Knockoff Nets result: no victim data or architecture needed.
- Fidelity tracks accuracy and also plateaus.

Sanity thresholds (ResNet-18 victim, SmallCNN substitute, 40 sub-epochs) — treat
as ballpark, not hard targets:

| transfer / loss | ~queries | substitute acc | fidelity |
|---|---|---|---|
| cifar10 / soft | 20k | ~0.85–0.90 | ~0.88–0.93 |
| cifar100 / soft | 20k | ~0.72–0.82 | ~0.75–0.85 |
| any / hard | 20k | a few points below its soft counterpart | — |

### 3.2 Defense curve

```bash
python run_sweep.py --experiment defense --transfer cifar100 \
    --victim checkpoints/victim.pt
```

**Pass / expected shape** (`results/defense_results.csv`, `defense_curve.png`):

- `none (soft)` is the top curve — the undefended baseline.
- `round-1`, `top-1 prob`, and especially `noise` sit **below** it: same budget,
  lower stolen accuracy — i.e. the attacker now needs *more* queries for the same
  result. This is the intended "raises cost, does not eliminate" effect.
- `label_only (hard)` shows the extraction still succeeds without any
  probabilities, just less efficiently — matching Tramèr et al.'s point that
  withholding confidences is not by itself a sufficient countermeasure.

---

## 4. Reproducibility notes

- Seeds: `--seed` fixes the transfer-pool ordering. Substitute training itself is
  not fully deterministic across GPUs; expect ±1–2% run to run.
- Timing (rough, single modern GPU): victim ~15–30 min; each substitute a few
  minutes; the full main sweep well under an hour.
- The budget sweep queries the victim once at the largest budget per
  (transfer, defense) and subsamples smaller budgets from the same queried set,
  so a sweep is not 4× the cost of its largest point.
