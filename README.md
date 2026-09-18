# Model Extraction Attack — Stealing a Black-Box Image Classifier

Reference implementation for the CSE 406 design report *"Model Extraction (Model
Stealing) Attack — Stealing the Functionality of a Black-Box Image Classifier."*
It reproduces functionality stealing in the style of **Knockoff Nets**
(Orekondy et al., CVPR'19), with the accuracy/fidelity framing of **Jagielski et
al.** (USENIX'20) and the prediction-API threat model of **Tramèr et al.**
(USENIX'16).

The attacker only ever touches the victim through an HTTP prediction API. It
never sees the victim's weights, architecture, or training data.

---

## How the code maps to the report

| Report section | Component | File |
|---|---|---|
| §2 Components — victim `f_V` | trained CIFAR-10 classifier | `victim_train.py`, `models.py` |
| §2 Components — inference server | FastAPI prediction API | `server.py` |
| §3 Packet / message structure | base64 image transport, `/predict` + `/predict_batch` | `server.py`, `config.py` |
| §2 Query engine + response logger | HTTP client, budgeted query loop, logging | `attack.py` |
| §4.4 Substitute trainer `f_S` | soft-label (KL) / hard-label (CE) training | `substitute.py` |
| §5 Accuracy + fidelity | evaluation on clean CIFAR-10 test | `evaluate.py` |
| §4.6 Budget sweep + headline plot | orchestrator, curves, CSV | `run_sweep.py` |
| §6 Defenses | output-granularity, top-k, noise, quota | `server.py`, `run_sweep.py` |

---

## Default design choices (all overridable)

- **Victim:** CIFAR-adapted **ResNet-18** (3×3 stem, no initial max-pool), ~93–95% test accuracy.
- **Substitute:** **SmallCNN** by default — deliberately a *different* architecture from the victim, to demonstrate the Knockoff Nets claim that they need not match. (`--sub-arch resnet18` etc. to change.)
- **Transfer pools:** `cifar10` (same-domain, CIFAR-10 **train** split) and `cifar100` (out-of-distribution). Evaluation is always on the CIFAR-10 **test** split, which is never queried — no train/eval contamination.
- **Loss:** `soft` (distillation-style KL on the victim's probability vector) when probabilities are available; `hard` (cross-entropy on the top-1 label) otherwise.
- **Budgets:** 5k / 10k / 20k / 50k queries.
- **Transport:** images travel as base64 uint8 pixels; the server does all normalization, so the attacker only ever sends pixels.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended for the real runs (victim training + the
sweep). Everything also runs on CPU, just slowly.

---

## Quickstart (real CIFAR runs)

```bash
# 1. Train the victim (downloads CIFAR-10 the first time)
python victim_train.py --arch resnet18 --epochs 60      # -> checkpoints/victim.pt

# 2. Run the headline experiment: transfer x loss x budget sweep.
#    Launches the victim server itself and produces the main curve.
python run_sweep.py --experiment main --victim checkpoints/victim.pt

# 3. Run the defense comparison (accuracy vs queries under each defense)
python run_sweep.py --experiment defense --transfer cifar100

# Outputs: results/main_curve.png, results/main_results.csv,
#          results/defense_curve.png, results/defense_results.csv
```

Or drive the pieces by hand (mirrors the report's step-by-step plan):

```bash
# Terminal A — start the victim server (choose a defense)
python server.py --victim checkpoints/victim.pt --defense none --port 8000

# Terminal B — query, train a substitute, evaluate
python attack.py --server-url http://127.0.0.1:8000 --transfer cifar100 \
    --budget 20000 --out cache/cifar100_20k.npz
python substitute.py --cache cache/cifar100_20k.npz --arch smallcnn \
    --loss soft --out checkpoints/sub.pt
python evaluate.py --substitute checkpoints/sub.pt --victim checkpoints/victim.pt
```

---

## Offline self-test (no GPU, no download)

Every stage has a `--smoke` mode that swaps CIFAR for tiny synthetic data so you
can verify the whole pipeline wires together in under a minute:

```bash
python run_sweep.py --experiment main --smoke
python run_sweep.py --experiment defense --smoke
```

See `TESTING.md` for the full testing procedure and what to check.

---

## Defenses (`--defense` on `server.py`)

| Mode | What the API returns | Purpose |
|---|---|---|
| `none` | label + full probability vector | baseline (richest signal) |
| `label_only` | label only | removes the soft-label signal |
| `round` | probabilities rounded to K decimals | reduces output granularity |
| `topk` | only the top-K probabilities kept | reduces output granularity |
| `noise` | probabilities perturbed + renormalized (top-1 label preserved) | prediction poisoning |
| `--quota N` | HTTP 429 after N queries/client | rate limiting / query quota |

The `noise` defense keeps the reported top-1 label clean, so honest users are
unaffected while the substitute's soft-label training signal is poisoned.
