#!/usr/bin/env python3
"""
Phase 5f - Brain Network Transformer (BNT) classifier (sixth modeling approach).

A sixth model family, run ALONGSIDE the GCN (phase5), node2vec+RF (phase5b),
TopER+RF (phase5c), GGVec+RF (phase5d), and GEE+RF (phase5e) - none of which are
touched. BNT (Kan, Dai, Cui, Zhang, Guo & Yang, NeurIPS 2022) applies a Transformer
directly to the brain connectivity matrix: each region's own row of the matrix (its
full "connection profile") is used as that region's feature vector - self-attention
over ALL region pairs then learns pairwise relationships, and a learned "Orthonormal
Clustering Readout" (soft clustering, via the DEC algorithm) pools regions into a
smaller number of functional-module-like clusters before a final MLP head classifies
AD vs CN. A fundamentally different mechanism from every other model here (full
dense self-attention, not a sparse/thresholded graph; a supervised, end-to-end
neural net like the GCN, not an unsupervised embedding).

Reference: https://github.com/Wayfear/BrainNetworkTransformer (their official
NeurIPS 2022 code). Per the advisor's explicit instruction, this uses their
UNMODIFIED model classes - BrainNetworkTransformer, TransPoolingEncoder,
InterpretableTransformerEncoder, DEC/ClusterAssignment - cloned read-only into
reference/BrainNetworkTransformer/ (same convention as reference/SGCN/, git-ignored).
Only the data loading, config construction, and training loop below are
project-specific glue (mirroring this project's own existing phase5_gcn.py style).

PyTorch version compatibility note (found, not invented - a real bug in the
intersection of their 2022 code and any currently-installable PyTorch):
    Every PyTorch version installable on this machine (Python 3.11 or 3.12; checked
    both PyPI and PyTorch's own wheel index) is >=2.0, and PyTorch 2.0 changed
    TransformerEncoderLayer.forward() to call _sa_block(..., is_causal=is_causal) -
    an argument that did not exist when this repo's `InterpretableTransformerEncoder
    ._sa_block()` override was written (their documented environment: torch==1.12.1).
    No installable PyTorch version avoids this. Per the advisor's decision, we apply
    a minimal compatibility SHIM (below) rather than modify their file: their file on
    disk is byte-for-byte untouched; at runtime, we capture their original
    `_sa_block` function object and wrap it in a new outer function that accepts and
    discards the extra `is_causal` argument, then calls their original,
    UNCHANGED function body. This is a forward-compatibility bridge for a PyTorch API
    signature change - it does not alter the attention/pooling/architecture logic in
    any way (this encoder was never causal/masked to begin with, so the discarded
    argument was never meaningful here).

Design choice - node_feature is the RAW (untresholded) connectivity matrix, NOT the
top-20%-edge graph the other 5 models share (graph_utils.build_nx_graph): confirmed
from their own dataset/abide.py, `node_feature` is literally the full Pearson
correlation matrix, unmodified - the whole point of "connection profile as node
feature" is the complete profile, and sparsifying it would throw away exactly the
information their method relies on. This is an inherent, deliberate property of the
Transformer's full self-attention design (it needs no sparse edge list at all,
unlike GCN's message-passing or the walk-based/filtration-based embeddings), not a
shortcut - but it does mean BNT alone does not share the "all models see identical
graphs" input with the rest of the comparison. Documented here and in the README.

Design choice - cluster count (config.model.sizes[1]): their own bnt.yaml hardcodes
[<node_sz>, 100], tuned for their 360-ROI atlas (ABIDE/ABCD both use a 360-region
HCP-2016 parcellation) - a fixed value they never published for a 116-ROI atlas. We
interpolate their own compression ratio (100/360 ~= 28%) to our 116 AAL regions,
giving 32 output clusters. Every other hyperparameter (pos_encoding='none',
pooling=[False, True], orthogonal/freeze_center/project_assignment=True, Adam
lr=1e-4 & weight_decay=1e-4, 200 epochs) is their own published default
(source/conf/model/bnt.yaml, source/conf/optimizer/adam.yaml,
source/conf/training/basic_training.yaml) - not re-tuned.

Loss function: confirmed from their actual source/training/training.py that only
plain CrossEntropyLoss on the classification output is used in training - the DEC
clustering module's own auxiliary KL loss (`model.loss()`) exists on the model class
but is never called in their real training loop, so we don't add it either.

Usage:
    .venv/bin/python src/phase5f_brain_transformer.py

Requirements: torch, omegaconf, plus reference/BrainNetworkTransformer/ cloned
(git clone https://github.com/Wayfear/BrainNetworkTransformer.git reference/BrainNetworkTransformer)
"""

import functools
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BNT_SRC = PROJECT_ROOT / "reference" / "BrainNetworkTransformer" / "source"
if not BNT_SRC.exists():
    raise FileNotFoundError(
        f"{BNT_SRC} not found. Clone it first (read-only reference, matching "
        f"reference/SGCN/'s existing convention):\n"
        f"  git clone https://github.com/Wayfear/BrainNetworkTransformer.git "
        f"reference/BrainNetworkTransformer"
    )
sys.path.insert(0, str(BNT_SRC))
from models.BNT.components.transformer_encoder import InterpretableTransformerEncoder  # noqa: E402
from models.BNT.bnt import BrainNetworkTransformer  # noqa: E402

# --- PyTorch >=2.0 compatibility shim (see module docstring) -----------------
_original_sa_block = InterpretableTransformerEncoder._sa_block


@functools.wraps(_original_sa_block)
def _sa_block_compat(self, x, attn_mask, key_padding_mask, is_causal=None):
    return _original_sa_block(self, x, attn_mask, key_padding_mask)


InterpretableTransformerEncoder._sa_block = _sa_block_compat

# --- Paths & config ------------------------------------------------------------
CONN_DIR = PROJECT_ROOT / "data" / "connectivity"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
GROUPS = {"CN": 0, "AD": 1}
METHODS = ["pearson", "spearman", "kendall"]

NODE_SZ = 116
CLUSTER_COUNT = 32     # interpolated from their 100/360 ratio - see module docstring

BNT_CONFIG = OmegaConf.create({
    "dataset": {"node_sz": NODE_SZ},
    "model": {
        "sizes": [NODE_SZ, CLUSTER_COUNT],   # their own bnt.yaml default shape
        "pooling": [False, True],            # their own bnt.yaml default
        "pos_encoding": "none",              # their own bnt.yaml default
        "pos_embed_dim": NODE_SZ,
        "orthogonal": True,                  # their own bnt.yaml default
        "freeze_center": True,               # their own bnt.yaml default
        "project_assignment": True,          # their own bnt.yaml default
    },
})

# Their own published defaults (source/conf/optimizer/adam.yaml, basic_training.yaml).
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 200
SEED = 0


def load_subjects(method: str):
    """Load every subject's RAW connectivity matrix (+ label, id) for one method."""
    mats, y, ids = [], [], []
    for group, label in GROUPS.items():
        for f in sorted((CONN_DIR / method / group).glob("*.npy")):
            mats.append(np.load(f))
            y.append(label)
            ids.append(f"{group}/{f.stem}")
    return np.array(mats), np.array(y), ids


def train_one(model: BrainNetworkTransformer, X: torch.Tensor, y: torch.Tensor) -> BrainNetworkTransformer:
    """Train a fresh model on the given subjects (full-batch, matches phase5_gcn.py's style)."""
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")   # matches their training.py
    time_series_placeholder = torch.zeros(X.shape[0], 1)   # unused by forward(), confirmed from source
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        out = model(time_series_placeholder, X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
    return model


def predict(model: BrainNetworkTransformer, x_one: torch.Tensor) -> int:
    """Predict the class of one held-out subject."""
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1), x_one.unsqueeze(0))
        return int(out.argmax(dim=1).item())


def main() -> None:
    warnings.filterwarnings("ignore")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Phase 5f: Brain Network Transformer (leave-one-out cross-validation)")
    print("labels: 0 = CN, 1 = AD\n")

    summary = {}
    for method in METHODS:
        mats, y, ids = load_subjects(method)
        X = torch.tensor(mats, dtype=torch.float)
        y_t = torch.tensor(y, dtype=torch.long)

        preds = []
        for i in range(len(X)):
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            train_idx = [j for j in range(len(X)) if j != i]
            model = BrainNetworkTransformer(BNT_CONFIG)
            model = train_one(model, X[train_idx], y_t[train_idx])
            preds.append(predict(model, X[i]))

        correct = sum(p == t for p, t in zip(preds, y))
        acc = correct / len(y)
        summary[method] = acc
        print(f"[{method}] leave-one-out accuracy = {acc:.2f}  ({correct}/{len(y)})")
        for sid, p, t in zip(ids, preds, y):
            print(f"    {sid:16s} true={t} pred={p}  {'ok' if p == t else 'MISS'}")

        torch.manual_seed(SEED)
        full_model = BrainNetworkTransformer(BNT_CONFIG)
        full_model = train_one(full_model, X, y_t)
        torch.save(full_model.state_dict(), MODEL_DIR / f"{method}_brain_transformer.pt")
        print()

    print("Summary - leave-one-out accuracy by method:")
    for method in METHODS:
        print(f"  {method:9s}: {summary[method]:.2f}")


if __name__ == "__main__":
    main()
