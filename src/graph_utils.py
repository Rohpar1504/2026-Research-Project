#!/usr/bin/env python3
"""
Shared graph construction, used identically by every model (GCN, node2vec, TopER,
nodevectors) so any performance difference reflects the model or the connectivity
method, not preprocessing.

Kept dependency-free (only numpy + networkx, both version-agnostic here) so it can be
imported from ANY of the project's virtual environments, including the separate one
used for phase5d_nodevectors.py - nodevectors' own dependency (csrgraph) hard-pins
networkx==2.5.1 (it calls the removed nx.adj_matrix()), which conflicts with node2vec's
and toper's networkx>=3 requirement. Importing phase5b_node2vec_rf.py directly would
drag in gensim/node2vec, which isn't installed (and can't be, due to the networkx
conflict) in that separate environment - so this function lives here instead.
"""

import numpy as np
import networkx as nx

KEEP_FRAC = 0.20


def build_nx_graph(mat: np.ndarray, keep_frac: float = KEEP_FRAC) -> nx.Graph:
    """
    Connectivity matrix -> weighted networkx graph, keeping the top `keep_frac` of
    off-diagonal edges by |correlation| (self-loops removed). Edge weight = |corr|.

    A handful of subjects have NaN entries in their connectivity matrix (undefined
    correlation for some region pair, e.g. a near-constant region time-series feeding
    Phase 4). Using plain np.quantile on data containing NaN returns NaN, and since
    every `w >= NaN` comparison is False, that silently produced a completely edgeless
    graph for those subjects - discovered when nodevectors/csrgraph crashed outright on
    an empty edge list, rather than degrading silently like other models did. Fixed
    here (not in any downstream model) since every model shares this function:
    nanquantile ignores NaN when computing the threshold, and any individual NaN-valued
    pair is simply excluded from the graph rather than corrupting the whole threshold.
    """
    n = mat.shape[0]
    A = mat.copy()
    np.fill_diagonal(A, 0.0)

    iu = np.triu_indices(n, k=1)
    threshold = np.nanquantile(np.abs(A[iu]), 1 - keep_frac)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, j in zip(*iu):
        w = abs(A[i, j])
        if not np.isnan(w) and w >= threshold:
            G.add_edge(int(i), int(j), weight=float(w))
    return G
