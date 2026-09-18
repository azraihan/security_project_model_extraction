"""
substitute.py
=============
Step 4: train the stolen model f_S entirely offline on the logged
(image, victim-prediction) pairs. No further queries to the victim.

Two loss strategies, matching the design report:

  soft  -- knowledge-distillation-style loss on the victim's probability vector.
           We only ever receive probabilities from the API (not logits), so the
           loss is the soft cross-entropy  -sum_c p_victim[c] * log_softmax(z)[c],
           i.e. KL(p_victim || softmax(z)) up to a constant. This is the richest
           signal and needs the fewest queries.
  hard  -- ordinary cross-entropy on the victim's top-1 label. This is the only
           option when the victim withholds probabilities (label_only defense).

The substitute architecture is chosen independently of the victim's, to
demonstrate the Knockoff Nets claim that the two need not match.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from attack import load_cache
from config import NUM_CLASSES
from data import substitute_loader
from models import ARCH_CHOICES, build_model


def soft_ce(student_logits: torch.Tensor, victim_probs: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy / KL against the victim's probability vector."""
    log_p = F.log_softmax(student_logits, dim=1)
    return -(victim_probs * log_p).sum(1).mean()


def train_substitute(images, labels, probs=None, arch="smallcnn", loss="soft",
                     epochs=40, batch_size=256, lr=0.1, weight_decay=5e-4,
                     device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    use_soft = (loss == "soft")
    if use_soft and probs is None:
        raise ValueError("soft loss requires probabilities, but none were logged "
                         "(victim likely ran with the label_only defense)")

    loader = substitute_loader(images, labels, probs if use_soft else None,
                               batch_size=batch_size, train=True)
    model = build_model(arch, NUM_CLASSES).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hard_ce = nn.CrossEntropyLoss()

    if verbose:
        print(f"[substitute] arch={arch} loss={loss} n={len(images)} "
              f"epochs={epochs} device={device}")

    for ep in range(epochs):
        model.train()
        running = 0.0
        it = loader
        if verbose:
            it = tqdm(loader, desc=f"epoch {ep + 1}/{epochs}", leave=False)
        for batch in it:
            if use_soft:
                x, _, p = batch
                x, p = x.to(device), p.to(device)
                loss_val = soft_ce(model(x), p)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                loss_val = hard_ce(model(x), y)
            opt.zero_grad()
            loss_val.backward()
            opt.step()
            running += loss_val.item() * x.size(0)
        sched.step()
        if verbose:
            print(f"[substitute] epoch {ep + 1:3d}/{epochs}  "
                  f"loss={running / len(loader.dataset):.4f}")

    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help=".npz from attack.py")
    p.add_argument("--arch", default="smallcnn", choices=ARCH_CHOICES)
    p.add_argument("--loss", default="soft", choices=("soft", "hard"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--out", default="checkpoints/substitute.pt")
    a = p.parse_args()

    images, labels, probs = load_cache(a.cache)
    model = train_substitute(images, labels, probs, arch=a.arch, loss=a.loss,
                             epochs=a.epochs, batch_size=a.batch_size, lr=a.lr)
    torch.save({"arch": a.arch, "num_classes": NUM_CLASSES,
                "state_dict": model.state_dict()}, a.out)
    print(f"[substitute] saved -> {a.out}")


if __name__ == "__main__":
    main()
