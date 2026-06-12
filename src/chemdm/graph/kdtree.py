"""On-device KD-tree for radius-neighbour queries.

A balanced, cyclic-axis BSP tree built level-by-level on the input tensor's
device. Designed to replace `scipy.spatial.KDTree.query_pairs` in the
torsional-diffusion sampler so the per-step neighbour computation no longer
forces a CPU sync.

Differences from `scipy.spatial.KDTree`:
- Cyclic split axis (0, 1, 2, 0, ...) rather than the widest-spread axis. Saves
  a per-bucket min/max reduction at every build level; balance is fine for
  chemistry-scale 3-4D point clouds.
- Power-of-two padding: each leaf is exactly `leaf_size` slots, so every level
  is a single batched argsort. Padding entries are copies of `x[0]` and are
  filtered out of both leaf bounding boxes and emitted edges.
- Same as scipy: tight (compact) per-node bounding boxes and balanced
  median splits.
"""
from __future__ import annotations

import math

import torch as pt


class KDTree:
    """Balanced KD-tree built and queried on the input tensor's device.

    Parameters
    ----------
    x : Tensor of shape (N, D)
        Points to index. N >= 1, D >= 1. Any floating dtype the device supports.
    leaf_size : int, default 32
        Max points per leaf; brute force inside.
    """

    def __init__(self, x: pt.Tensor, leaf_size: int = 32):
        if x.dim() != 2:
            raise ValueError(f"x must be 2D (N, D); got shape {tuple(x.shape)}")
        if x.shape[0] == 0:
            raise ValueError("x must contain at least one point")
        if leaf_size < 1:
            raise ValueError(f"leaf_size must be >= 1; got {leaf_size}")

        x = x.detach()
        N, D = x.shape
        device = x.device
        dtype = x.dtype

        if N <= leaf_size:
            depth = 0
            bucket_size = N
        else:
            depth = math.ceil( math.log2(N / leaf_size) )
            bucket_size = leaf_size
        n_leaves = 1 << depth
        N_padded = n_leaves * bucket_size

        if N_padded > N:
            pad = x[:1].expand(N_padded - N, D)
            x_padded = pt.cat( [x, pad], dim=0 )
        else:
            x_padded = x

        perm = pt.arange(N_padded, device=device)
        for level in range(depth):
            axis = level % D
            n_buckets = 1 << level
            cur_bs = N_padded // n_buckets
            perm_b = perm.view(n_buckets, cur_bs)
            vals = x_padded[perm_b.reshape(-1), axis].view(n_buckets, cur_bs)
            sort_idx = pt.argsort(vals, dim=1)
            perm = pt.gather(perm_b, 1, sort_idx).reshape(-1)

        leaf_perm = perm.view(n_leaves, bucket_size)
        leaf_pts = x_padded[leaf_perm]

        is_valid = (leaf_perm < N).unsqueeze(-1)
        pos_inf = leaf_pts.new_full((), float("inf"))
        neg_inf = leaf_pts.new_full((), float("-inf"))
        pts_for_min = pt.where(is_valid, leaf_pts, pos_inf)
        pts_for_max = pt.where(is_valid, leaf_pts, neg_inf)

        bbox_min: list[pt.Tensor] = [pt.empty(0)] * (depth + 1)
        bbox_max: list[pt.Tensor] = [pt.empty(0)] * (depth + 1)
        bbox_min[depth] = pts_for_min.min(dim=1).values
        bbox_max[depth] = pts_for_max.max(dim=1).values
        for level in range(depth - 1, -1, -1):
            n = 1 << level
            cmin = bbox_min[level + 1].view(n, 2, D)
            cmax = bbox_max[level + 1].view(n, 2, D)
            bbox_min[level] = cmin.min(dim=1).values
            bbox_max[level] = cmax.max(dim=1).values

        self._N = N
        self._D = D
        self._depth = depth
        self._bucket_size = bucket_size
        self._n_leaves = n_leaves
        self._device = device
        self._dtype = dtype
        self._x_padded = x_padded
        self._leaf_perm = leaf_perm
        self._bbox_min = bbox_min
        self._bbox_max = bbox_max

    @property
    def data(self) -> pt.Tensor:
        return self._x_padded[: self._N]

    @pt.no_grad()
    def query_pairs(self, r: float) -> pt.Tensor:
        """All (i, j) with i < j and ||x[i] - x[j]|| < r.

        Returns (E, 2) long tensor on the input device. Matches the
        convention of `scipy.spatial.KDTree.query_pairs(r, output_type='ndarray')`.
        """
        return self._query_radius(self.data, r, ordered_self=True)

    @pt.no_grad()
    def query_radius(self, queries: pt.Tensor, r: float) -> pt.Tensor:
        """All (q_idx, t_idx) with ||queries[q_idx] - data[t_idx]|| < r.

        Returns (E, 2) long. Row 0 indexes `queries`, row 1 indexes the tree's
        data. Queries and data are treated as distinct sets — no self/order
        filtering.
        """
        if queries.dim() != 2 or queries.shape[1] != self._D:
            raise ValueError( f"queries must have shape (Q, {self._D}); got {tuple(queries.shape)}" )
        return self._query_radius(queries.detach(), r, ordered_self=False)

    def _query_radius( self, queries: pt.Tensor, r: float, ordered_self: bool ) -> pt.Tensor:
        Q = queries.shape[0]
        device = self._device
        N = self._N
        depth = self._depth
        r2 = float(r) * float(r)

        alive = pt.ones((Q, 1), dtype=pt.bool, device=device)
        for level in range(depth + 1):
            bmin = self._bbox_min[level]
            bmax = self._bbox_max[level]
            q = queries.unsqueeze(1)
            d_lo = (bmin.unsqueeze(0) - q).clamp(min=0)
            d_hi = (q - bmax.unsqueeze(0)).clamp(min=0)
            d = d_lo + d_hi
            dist2 = (d * d).sum(dim=-1)
            alive = alive & (dist2 < r2)
            if level < depth:
                alive = alive.repeat_interleave(2, dim=1)

        q_idx, leaf_idx = alive.nonzero(as_tuple=True)
        if q_idx.numel() == 0:
            return pt.empty((0, 2), dtype=pt.long, device=device)

        leaf_pt_idx = self._leaf_perm[leaf_idx]
        tree_pts = self._x_padded[leaf_pt_idx]
        diff = tree_pts - queries[q_idx].unsqueeze(1)
        leaf_dist2 = (diff * diff).sum(dim=-1)

        valid = (leaf_dist2 < r2) & (leaf_pt_idx < N)
        if ordered_self:
            valid = valid & (leaf_pt_idx > q_idx.unsqueeze(1))

        sel_m, sel_s = valid.nonzero(as_tuple=True)
        src = q_idx[sel_m]
        dst = leaf_pt_idx[sel_m, sel_s]
        return pt.stack([src, dst], dim=1)
