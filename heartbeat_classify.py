#!/usr/bin/env python3
# heartbeat_classify.py
# ---------------------------------------------------------------
# Train TimesNet or Crossformer on the UEA Heartbeat dataset
# Default hyper‑params: d_model=64, e_layers=2, n_heads=4, d_ff=128
# Optimiser: Adam (lr=1e‑3) • 50 epochs • Early‑stopping on val‑loss
# Saves the best‑val checkpoint and evaluates that on the test set.
# ---------------------------------------------------------------

import argparse
import os
import sys
import random
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from scipy.io import arff

# ---------------------------------------------------------------------
# Path so we can `import models.*` directly from the cloned repo
# ---------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
TSL_PATH = os.path.join(ROOT, "Time-Series-Library")
sys.path.insert(0, TSL_PATH)

from models.TimesNet import Model as TimesNetModel  # type: ignore
from models.Crossformer import Model as CrossformerModel  # type: ignore

# ---------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------

def _seq_to_array(seq) -> np.ndarray:
    """Convert one structured ARFF row → (T, C) float32 array."""
    fields = seq.dtype.names
    first = seq[fields[0]]
    if np.isscalar(first):  # scalar per field → univariate series
        return np.asarray([seq[f] for f in fields], dtype=np.float32)[:, None]
    return np.stack([np.asarray(seq[f], dtype=np.float32) for f in fields], axis=1)


class HeartbeatDataset(Dataset):
    def __init__(self, path: str, label_map: dict | None = None):
        data, _ = arff.loadarff(path)
        xs: List[np.ndarray] = [_seq_to_array(row["Heartbeat"]) for row in data]
        shapes = {x.shape for x in xs}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent shapes in dataset: {shapes}")
        self.X = np.stack(xs).astype(np.float32)  # (N, T, C)

        raw = [lbl.decode() if isinstance(lbl, bytes) else str(lbl) for lbl in data["target"]]
        if label_map is None:
            classes = sorted(set(raw))
            self.label_map = {c: i for i, c in enumerate(classes)}
        else:
            self.label_map = label_map
        self.y = np.fromiter((self.label_map[l] for l in raw), dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def collate(batch):
    xs, ys = zip(*batch)
    return torch.from_numpy(np.stack(xs)), torch.tensor(ys, dtype=torch.long)


def stratified_split(ds, val_ratio=0.2, seed=42):
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    split = int(len(idx) * (1 - val_ratio))
    return torch.utils.data.Subset(ds, idx[:split]), torch.utils.data.Subset(ds, idx[split:])


# ---------------------------------------------------------------------
# Training‑and‑evaluation routine
# ---------------------------------------------------------------------

def train_and_eval(args):
    # -------- reproducibility --------
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # -------- data --------
    train_full = HeartbeatDataset(args.train)
    test_ds    = HeartbeatDataset(args.test, label_map=train_full.label_map)

    mean, std = train_full.X.mean(), train_full.X.std() + 1e-6
    train_full.X = (train_full.X - mean) / std
    test_ds.X   = (test_ds.X   - mean) / std

    train_ds, val_ds = stratified_split(train_full, 0.2, seed=args.seed or 42)
    dl_train = DataLoader(train_ds, args.batch_size, True,  collate_fn=collate)
    dl_val   = DataLoader(val_ds,   args.batch_size, False, collate_fn=collate)
    dl_test  = DataLoader(test_ds,  args.batch_size, False, collate_fn=collate)

    T, C = train_full.X.shape[1:3]
    n_class = len(train_full.label_map)

    cfg = SimpleNamespace(task_name="classification", seq_len=T, label_len=0, pred_len=0,
                          enc_in=C, c_out=C, d_model=64, d_ff=128, n_heads=4, e_layers=2,
                          num_class=n_class, dropout=0.2, embed="timeF", freq="h",
                          top_k=5, num_kernels=6, factor=3)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Model = TimesNetModel if args.model == "timesnet" else CrossformerModel
    model = Model(cfg).to(device)
    model.projection = nn.Linear(model.projection.in_features, n_class).to(device)  # type: ignore

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_state, best_val = None, float("inf")
    patience_ctr = 0

    print(f"Training {args.model} | seq_len={T}, channels={C} | device={device}\n")

    mask_fn = lambda b: torch.ones(b.size(0), b.size(1), device=device)

    # -------- epoch loop --------
    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train(); tloss=0; tp, tl = [], []
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb, mask_fn(xb), None, None)  # type: ignore
            loss = criterion(out, yb)
            loss.backward(); optimizer.step()
            tloss += loss.item()*xb.size(0)
            tp.extend(out.argmax(1).cpu().numpy()); tl.extend(yb.cpu().numpy())
        tloss /= len(dl_train.dataset); tacc = accuracy_score(tl, tp)

        # ---- validate ----
        model.eval(); vloss=0; vp, vl = [], []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb, mask_fn(xb), None, None)  # type: ignore
                loss = criterion(out, yb)
                vloss += loss.item()*xb.size(0)
                vp.extend(out.argmax(1).cpu().numpy()); vl.extend(yb.cpu().numpy())
        vloss /= len(dl_val.dataset); vacc = accuracy_score(vl, vp)

        print(f"Epoch {epoch}/{args.epochs} | Train L {tloss:.4f} A {tacc:.4f} | Val L {vloss:.4f} A {vacc:.4f}")

        # ---- early‑stop & save best ----
        if vloss < best_val:
            best_val = vloss; patience_ctr = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print("Early stopping\n"); break

    # restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # -------- test --------
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            out = model(xb.to(device), mask_fn(xb.to(device)), None, None)  # type: ignore
            preds.extend(out.argmax(1).cpu().numpy()); labels.extend(yb.numpy())
    print(f"Final Test Accuracy ({args.model}): {accuracy_score(labels, preds)*100:.2f}%")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['timesnet', 'crossformer'], required=True)
    p.add_argument('--train', default='Heartbeat/Heartbeat_TRAIN.arff')
    p.add_argument('--test',  default='Heartbeat/Heartbeat_TEST.arff')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=None,
                   help='fix seed for deterministic run; omit for random split')
    args = p.parse_args()
    train_and_eval(args)
