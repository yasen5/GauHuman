"""Drop-in replacement for the `knn_cuda` package (github.com/unlimblue/KNN_CUDA),
whose release wheel and source repo have both been deleted upstream (404 on
both the release asset and the repo itself as of 2026-07-18).

GauHuman only uses `KNN(k, transpose_mode=True)` called as
`knn(ref_points, query_points) -> (dist, idx)` with ref/query shaped
(B, N, 3) -- see scene/gaussian_model.py. That's exactly torch.cdist +
topk, no custom CUDA kernel required. Point counts here (SMPL/SMPL-X
vertex-scale, ~10k) are small enough that a dense cdist is cheap.

This file lives at the GauHuman repo root (alongside train.py) so plain
`from knn_cuda import KNN` resolves to it via normal Python import search
-- no changes needed to the vendored gaussian_model.py import line itself.
"""
import torch


class KNN:
    def __init__(self, k, transpose_mode=True):
        self.k = k

    def __call__(self, ref, query):
        # ref, query: (B, N, 3) -- matches transpose_mode=True convention,
        # the only mode this codebase ever uses.
        dist = torch.cdist(query, ref)  # (B, Nq, Nr)
        dist, idx = torch.topk(dist, self.k, dim=-1, largest=False, sorted=True)
        return dist, idx
