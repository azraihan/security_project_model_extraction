"""
data.py
=======
Everything data-related, in one place:

  * victim training/eval loaders for CIFAR-10;
  * the two *transfer pools* the attacker queries with
        - "cifar10"  : same-domain images (CIFAR-10 *train* split);
        - "cifar100" : out-of-distribution images (CIFAR-100 train split);
  * shared normalization + light augmentation used by both the victim and the
    substitute so their numerical preprocessing is identical;
  * a synthetic pool for the offline smoke test (no download, no GPU).

Design choice worth stating explicitly:
  The same-domain transfer pool is CIFAR-10's *train* split, and evaluation
  always happens on CIFAR-10's *test* split. Querying with test images and then
  scoring on the same test images would leak the answer key, so we never do
  that. Using train images as the attacker's same-domain pool is the standard,
  contamination-free setup.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as T

from config import CIFAR10_MEAN, CIFAR10_STD, DATA_ROOT, IMG_SIZE, NUM_CLASSES

_MEAN = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- #
# Normalization helpers (uint8 HWC/NHWC  ->  normalized float NCHW)           #
# --------------------------------------------------------------------------- #
def normalize_uint8_batch(nhwc_uint8: np.ndarray) -> torch.Tensor:
    """(N,32,32,3) uint8 in [0,255]  ->  (N,3,32,32) float, CIFAR-normalized."""
    if nhwc_uint8.ndim == 3:
        nhwc_uint8 = nhwc_uint8[None]
    t = torch.from_numpy(np.ascontiguousarray(nhwc_uint8)).float().div_(255.0)
    t = t.permute(0, 3, 1, 2).contiguous()          # NHWC -> NCHW
    return (t - _MEAN) / _STD


def _augment_uint8(img: np.ndarray) -> np.ndarray:
    """Random 4-px reflect-pad crop + horizontal flip on a (32,32,3) uint8 image."""
    padded = np.pad(img, ((4, 4), (4, 4), (0, 0)), mode="reflect")
    top = np.random.randint(0, 9)
    left = np.random.randint(0, 9)
    out = padded[top:top + IMG_SIZE, left:left + IMG_SIZE, :]
    if np.random.rand() < 0.5:
        out = out[:, ::-1, :]
    return np.ascontiguousarray(out)


# --------------------------------------------------------------------------- #
# Victim CIFAR-10                                                              #
# --------------------------------------------------------------------------- #
def victim_train_transform():
    return T.Compose([
        T.RandomCrop(IMG_SIZE, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def eval_transform():
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


def cifar10_loaders(batch_size: int = 128, num_workers: int = 4,
                    download: bool = True) -> Tuple[DataLoader, DataLoader]:
    train = torchvision.datasets.CIFAR10(
        DATA_ROOT, train=True, download=download, transform=victim_train_transform())
    test = torchvision.datasets.CIFAR10(
        DATA_ROOT, train=False, download=download, transform=eval_transform())
    return (
        DataLoader(train, batch_size, shuffle=True, num_workers=num_workers,
                   pin_memory=True, drop_last=True),
        DataLoader(test, batch_size, shuffle=False, num_workers=num_workers,
                   pin_memory=True),
    )


def cifar10_eval_arrays(download: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """CIFAR-10 test split as (images_uint8 (10000,32,32,3), labels (10000,))."""
    ds = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=download)
    return ds.data.astype(np.uint8), np.array(ds.targets, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Transfer pools (what the attacker queries with)                             #
# --------------------------------------------------------------------------- #
def transfer_pool(name: str, download: bool = True) -> np.ndarray:
    """Return the attacker's query pool as (N,32,32,3) uint8."""
    name = name.lower()
    if name == "cifar10":
        ds = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=download)
        return ds.data.astype(np.uint8)                     # 50k same-domain
    if name == "cifar100":
        ds = torchvision.datasets.CIFAR100(DATA_ROOT, train=True, download=download)
        return ds.data.astype(np.uint8)                     # 50k OOD
    raise ValueError(f"unknown transfer pool '{name}' (use cifar10 | cifar100)")


# --------------------------------------------------------------------------- #
# Substitute training dataset (image, victim-prediction) pairs                #
# --------------------------------------------------------------------------- #
class TransferDataset(Dataset):
    """
    Wraps the logged (image, victim label, victim probs) triples.

    * ``images`` : (N,32,32,3) uint8  -- the queried pixels
    * ``labels`` : (N,) int64         -- victim argmax (always available)
    * ``probs``  : (N,10) float32 or None -- victim soft output (may be withheld)
    """

    def __init__(self, images, labels, probs=None, train: bool = True):
        self.images = np.ascontiguousarray(images, dtype=np.uint8)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.probs = None if probs is None else np.asarray(probs, dtype=np.float32)
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.train:
            img = _augment_uint8(img)
        x = normalize_uint8_batch(img)[0]                   # (3,32,32) float
        y = int(self.labels[i])
        if self.probs is None:
            return x, y
        return x, y, torch.from_numpy(self.probs[i])


def substitute_loader(images, labels, probs=None, batch_size=256,
                      num_workers=4, train=True) -> DataLoader:
    ds = TransferDataset(images, labels, probs, train=train)
    return DataLoader(ds, batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=True, drop_last=False)


# --------------------------------------------------------------------------- #
# Synthetic data for the offline smoke test (no download / no GPU needed)     #
# --------------------------------------------------------------------------- #
def synthetic_pool(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)


def synthetic_labeled(n: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    imgs = rng.integers(0, 256, size=(n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    labels = rng.integers(0, NUM_CLASSES, size=(n,), dtype=np.int64)
    # Bake a faint, learnable signal into the images so training is non-trivial.
    for c in range(NUM_CLASSES):
        imgs[labels == c, 0, 0, 0] = c * 25
    return imgs, labels
