"""
victim_train.py
===============
Step 1 of the design report: build the victim f_V.

Trains a classifier on CIFAR-10 and saves a checkpoint. After this script runs
the model is treated purely as a black box -- every later stage reaches it only
through the HTTP prediction API in server.py.

Usage
-----
    python victim_train.py --arch resnet18 --epochs 60
    python victim_train.py --smoke            # 1 epoch on synthetic data, CPU-ok
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import CKPT_DIR, NUM_CLASSES, VICTIM_CKPT
from data import cifar10_loaders, normalize_uint8_batch, synthetic_labeled
from models import ARCH_CHOICES, build_model


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def train_victim(arch="resnet18", epochs=60, batch_size=128, lr=0.1,
                 weight_decay=5e-4, out=VICTIM_CKPT, device=None, smoke=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(f"[victim] arch={arch} epochs={epochs} device={device} smoke={smoke}")

    if smoke:
        imgs, labels = synthetic_labeled(512)
        x = normalize_uint8_batch(imgs)
        y = torch.from_numpy(labels)
        tr = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
        te = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=False)
    else:
        tr, te = cifar10_loaders(batch_size=batch_size)

    model = build_model(arch, NUM_CLASSES).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(epochs):
        model.train()
        running = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
        sched.step()
        acc = evaluate(model, te, device)
        best = max(best, acc)
        print(f"[victim] epoch {ep + 1:3d}/{epochs}  "
              f"loss={running / len(tr.dataset):.4f}  test_acc={acc:.4f}")

    torch.save({"arch": arch, "num_classes": NUM_CLASSES,
                "state_dict": model.state_dict(), "test_acc": best}, out)
    print(f"[victim] saved -> {out}  (best test acc {best:.4f})")
    return out


def load_victim(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["arch"], ckpt.get("num_classes", NUM_CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="resnet18", choices=ARCH_CHOICES)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--out", default=VICTIM_CKPT)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.epochs = min(a.epochs, 1)
    train_victim(a.arch, a.epochs, a.batch_size, a.lr, a.weight_decay,
                 a.out, smoke=a.smoke)


if __name__ == "__main__":
    main()
