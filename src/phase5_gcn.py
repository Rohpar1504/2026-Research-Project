#!/usr/bin/env python3
"""
Phase 5 - Graph construction & GCN classifier.

Converts each subject's connectivity matrix into a graph, then trains a Graph
Convolutional Network (GCN) + MLP classifier to predict AD vs CN. Runs separately
for each connectivity method (Pearson / Spearman / Kendall) so their downstream
classification performance can be compared (Phase 6).

Pipeline position:
    data/connectivity/{method}/{AD,CN}/<subject>.npy   (116 x 116 matrix)
        -->  [ graph construction + GCN + MLP ]  -->
    leave-one-out accuracy per method  (+ saved reference models)

Architecture note:
    The GCN + MLP design (graph conv layers -> global readout -> MLP head) is
    INSPIRED BY the SGCN repository (github.com/Houliang-Zhou/SGCN) and the IEEE
    interpretable-GCN paper. It is reimplemented here from scratch in a simpler,
    focused form - not copied.

Honest caveat:
    With only 6 subjects, this is a proof-of-concept. Leave-one-out accuracy will be
    noisy and near chance; the goal is a *fair, identical* comparison across the three
    connectivity methods, not a state-of-the-art classifier.

Usage:
    .venv/bin/python src/phase5_gcn.py
"""

import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# --- Paths & config ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONN_DIR = PROJECT_ROOT / "data" / "connectivity"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
GROUPS = {"CN": 0, "AD": 1}          # diagnosis -> class label
METHODS = ["pearson", "spearman", "kendall"]

# Hyperparameters (kept simple and identical across methods for a fair comparison).
KEEP_FRAC = 0.20      # keep the top 20% strongest edges (sparsify the graph)
HIDDEN = 64           # hidden width of the GCN/MLP
EPOCHS = 100
LR = 0.01
WEIGHT_DECAY = 5e-4
SEED = 0


def build_graph(mat: np.ndarray, label: int, keep_frac: float = KEEP_FRAC) -> Data:
    """
    Turn one (n x n) connectivity matrix into a PyTorch Geometric graph.

    Nodes         = the 116 brain regions.
    Node features = each region's full connectivity profile (its matrix row) - so a
                    region is described by how it connects to every other region.
    Edges         = the strongest connections only. We keep the top `keep_frac` of
                    off-diagonal pairs by |correlation| (sparsification - drops weak,
                    noisy edges), and remove self-loops (the diagonal).
    Edge weights  = |correlation| (magnitude). We use the absolute value here because
                    GCNConv's normalization sums edge weights into node degrees; signed
                    (negative) weights could produce negative degrees and break it. The
                    signed values are still preserved in the node features.
    """
    n = mat.shape[0]
    A = mat.copy()
    np.fill_diagonal(A, 0.0)                          # remove self-loops

    off_diag_mag = np.abs(A[np.triu_indices(n, k=1)])
    threshold = np.quantile(off_diag_mag, 1 - keep_frac)
    mask = np.abs(A) >= threshold
    np.fill_diagonal(mask, False)

    src, dst = np.where(mask)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(np.abs(A[src, dst]), dtype=torch.float)
    x = torch.tensor(mat, dtype=torch.float)          # node features (signed profile)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)


def load_graphs(method: str):
    """Load all subjects' graphs for one connectivity method, with labels + ids."""
    graphs, ids = [], []
    for group, label in GROUPS.items():
        for f in sorted((CONN_DIR / method / group).glob("*.npy")):
            graphs.append(build_graph(np.load(f), label))
            ids.append(f"{group}/{f.stem}")
    return graphs, ids


class GCN(nn.Module):
    """
    GCN + MLP graph classifier (inspired by SGCN; reimplemented from scratch).

    Two graph-conv layers let each region mix in information from its connected
    neighbours; a global mean pool summarizes all 116 regions into one vector; a
    small MLP head turns that summary into an AD/CN decision.
    """
    def __init__(self, in_dim: int, hidden: int = HIDDEN, num_classes: int = 2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin1 = nn.Linear(hidden, hidden // 2)
        self.lin2 = nn.Linear(hidden // 2, num_classes)

    def forward(self, x, edge_index, edge_weight, batch):
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = F.relu(self.conv2(h, edge_index, edge_weight))
        h = global_mean_pool(h, batch)                # graph-level readout
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.lin2(h)


def train_one(model: GCN, graphs) -> GCN:
    """Train a fresh model on the given list of graphs."""
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(graphs, batch_size=len(graphs), shuffle=True)
    model.train()
    for _ in range(EPOCHS):
        for batch in loader:
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            opt.step()
    return model


def predict(model: GCN, graph: Data) -> int:
    """Predict the class of one held-out graph."""
    model.eval()
    with torch.no_grad():
        for batch in DataLoader([graph], batch_size=1):
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            return int(out.argmax(dim=1).item())


def main() -> None:
    warnings.filterwarnings("ignore")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Phase 5: GCN classifier (leave-one-out cross-validation)")
    print("labels: 0 = CN, 1 = AD\n")

    summary = {}
    for method in METHODS:
        graphs, ids = load_graphs(method)
        in_dim = graphs[0].x.shape[1]

        # Leave-one-out CV: train on 5, test on the held-out 1, rotate through all 6.
        preds, trues = [], []
        for i in range(len(graphs)):
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            train_set = [g for j, g in enumerate(graphs) if j != i]
            model = train_one(GCN(in_dim), train_set)
            preds.append(predict(model, graphs[i]))
            trues.append(int(graphs[i].y.item()))

        correct = sum(p == t for p, t in zip(preds, trues))
        acc = correct / len(trues)
        summary[method] = acc
        print(f"[{method}] leave-one-out accuracy = {acc:.2f}  ({correct}/{len(trues)})")
        for sid, p, t in zip(ids, preds, trues):
            print(f"    {sid:16s} true={t} pred={p}  {'ok' if p == t else 'MISS'}")

        # Save one reference model trained on all 6 subjects (artifact for later use).
        torch.manual_seed(SEED)
        full_model = train_one(GCN(in_dim), graphs)
        torch.save(full_model.state_dict(), MODEL_DIR / f"{method}_gcn.pt")
        print()

    print("Summary - leave-one-out accuracy by method:")
    for method in METHODS:
        print(f"  {method:9s}: {summary[method]:.2f}")


if __name__ == "__main__":
    main()
