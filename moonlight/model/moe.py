"""Mixture-of-Experts feed-forward (shared + routed SwiGLU experts, aux-loss-free balancing).

Routing (DeepSeek-V3 style):

- ``logits = router(x)`` computed in float32 (autocast disabled), ``scores = sigmoid(logits)``;
- expert *selection* uses ``scores + router_bias`` (top-k); ``router_bias`` (E,) is a persistent
  float32 **buffer**, never a parameter, adjusted outside the optimizer by
  :meth:`MoE.apply_bias_update` (``bias += rate * sign(mean_load - load)``);
- the *combine* weights use the un-biased scores of the selected experts normalised to sum to 1;
- dispatch is dropless: assignments are sorted by expert, each expert with at least one token
  processes its tokens with a dense matmul and results are scattered back with ``index_add_``
  (``capacity_factor > 0`` optionally caps tokens per expert; overflow assignments are dropped in
  token order);
- shared experts (``n_shared > 0``) are one dense SwiGLU of hidden size ``expert_hidden * n_shared``
  applied to every token;
- Switch-style sequence-level balance loss while training:
  ``aux = aux_loss_weight * E * sum_e f_e * P_e`` averaged over sequences, where ``f_e`` is the
  fraction of the sequence's top-k assignments routed to ``e`` and ``P_e`` the mean normalised
  score (``scores / scores.sum(-1)``) of ``e`` over the sequence (= ``aux_loss_weight`` when
  perfectly balanced).

Parameters: ``router`` Linear(d, E), ``w_gate_up`` (E, d, 2*he), ``w_down`` (E, he, d),
``shared`` SwiGLU(d, he*n_shared) if n_shared > 0. Expert e computes
``(silu(x @ W_gu[e][:, :he]) * (x @ W_gu[e][:, he:])) @ W_down[e]``.
"""
from __future__ import annotations

import contextlib
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SwiGLU, is_meta

__all__ = ["MoE"]


def _autocast_off(device: torch.device):
    if device.type in ("cuda", "cpu"):
        return torch.autocast(device_type=device.type, enabled=False)
    return contextlib.nullcontext()


class MoE(nn.Module):
    """Routed + shared expert FFN.

    ``forward(x)``: x (B, T, d) (or (N, d)) -> ``(y, aux_loss)`` with y of the same shape and
    ``x.dtype``; ``aux_loss`` is a float32 scalar tensor while training with
    ``aux_loss_weight > 0``, else ``None``.
    """

    def __init__(
        self,
        dim: int,
        n_experts: int,
        top_k: int,
        expert_hidden: int,
        n_shared: int = 0,
        aux_loss_weight: float = 0.0,
        capacity_factor: float = 0.0,
        init_std: float = 0.02,
        out_std: Optional[float] = None,
    ) -> None:
        super().__init__()
        if not (int(n_experts) >= int(top_k) >= 1):
            raise ValueError(f"MoE needs n_experts >= top_k >= 1, got E={n_experts} k={top_k}")
        if int(expert_hidden) <= 0:
            raise ValueError(f"MoE expert_hidden must be > 0, got {expert_hidden}")
        self.dim = int(dim)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.expert_hidden = int(expert_hidden)
        self.n_shared = int(n_shared or 0)
        self.aux_loss_weight = float(aux_loss_weight or 0.0)
        self.capacity_factor = float(capacity_factor or 0.0)
        self.init_std = float(init_std)
        self.out_std = float(out_std) if out_std is not None else float(init_std)

        E, d, he = self.n_experts, self.dim, self.expert_hidden
        self.router = nn.Linear(d, E, bias=False)
        self.register_buffer("router_bias", torch.zeros(E, dtype=torch.float32), persistent=True)
        self.w_gate_up = nn.Parameter(torch.empty(E, d, 2 * he))
        self.w_down = nn.Parameter(torch.empty(E, he, d))
        self.shared = SwiGLU(d, he * self.n_shared) if self.n_shared > 0 else None
        self._load: Optional[torch.Tensor] = None  # accumulated assignments per expert (E,)
        self.reset_parameters()

    # ------------------------------------------------------------------------------------------
    def reset_parameters(self, init_std: Optional[float] = None, out_std: Optional[float] = None) -> None:
        """Init: router/w_gate_up ~ N(0, init_std); w_down ~ N(0, out_std); router_bias = 0.

        The shared expert keeps whatever its Linear layers were given (the model-level init
        re-initialises it with ``init_std`` / ``out_std`` for ``w_down``).
        """
        if init_std is not None:
            self.init_std = float(init_std)
        if out_std is not None:
            self.out_std = float(out_std)
        for p, std in ((self.router.weight, self.init_std), (self.w_gate_up, self.init_std),
                       (self.w_down, self.out_std)):
            if not is_meta(p):
                nn.init.normal_(p, mean=0.0, std=std)
        if not is_meta(self.router_bias):
            with torch.no_grad():
                self.router_bias.zero_()
        self._load = None

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, n_experts={self.n_experts}, top_k={self.top_k}, "
                f"expert_hidden={self.expert_hidden}, n_shared={self.n_shared}")

    # -- load bookkeeping for aux-loss-free balancing -------------------------------------------
    @property
    def expert_load(self) -> Optional[torch.Tensor]:
        """Assignments per expert (E,) float32, accumulated over training forwards since the last
        bias update (detached). Activation-checkpoint recomputation counts again, which scales a
        layer's counts uniformly and does not change the sign-based update."""
        return self._load

    def consume_load(self) -> Optional[torch.Tensor]:
        """Return and reset the accumulated expert load."""
        load, self._load = self._load, None
        return load

    @torch.no_grad()
    def apply_bias_update(self, load: torch.Tensor, rate: float) -> None:
        """``router_bias += rate * sign(mean(load) - load)`` (under-loaded experts get a boost)."""
        load = load.to(device=self.router_bias.device, dtype=torch.float32)
        delta = float(rate) * torch.sign(load.mean() - load)
        self.router_bias.copy_((self.router_bias.float() + delta).to(self.router_bias.dtype))

    def _record_load(self, counts: torch.Tensor) -> None:
        counts = counts.detach().float()
        self._load = counts if self._load is None else self._load + counts.to(self._load.device)

    # ------------------------------------------------------------------------------------------
    def route(self, x2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Routing for flat tokens x2d (N, d).

        Returns ``(scores, topk_idx, weights)``: scores (N, E) float32 sigmoid scores, topk_idx
        (N, k) int64 selected experts (by biased score), weights (N, k) float32 normalised
        un-biased scores of the selected experts.
        """
        with _autocast_off(x2d.device):
            logits = F.linear(x2d.float(), self.router.weight.float())
        scores = torch.sigmoid(logits)
        biased = scores + self.router_bias.float()
        topk_idx = torch.topk(biased, self.top_k, dim=-1).indices
        weights = scores.gather(-1, topk_idx)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        return scores, topk_idx, weights

    def _experts(self, xf: torch.Tensor, topk_idx: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dropless sorted dispatch. xf (N, d) -> (y (N, d) in xf.dtype, counts (E,) int64)."""
        N = xf.shape[0]
        k = self.top_k
        flat_expert = topk_idx.reshape(-1)  # (N*k,)
        order = torch.argsort(flat_expert, stable=True)
        token_of = order // k  # token index of each sorted assignment
        w_sorted = weights.reshape(-1)[order]  # (N*k,)
        counts = torch.bincount(flat_expert, minlength=self.n_experts)
        cap = None
        if self.capacity_factor > 0:
            cap = max(1, int(math.ceil(self.capacity_factor * N * k / self.n_experts)))
        y = torch.zeros_like(xf)
        he = self.expert_hidden
        start = 0
        for e, c in enumerate(counts.tolist()):
            if c == 0:
                continue
            end = start + c
            use = c if cap is None else min(c, cap)
            idx = token_of[start:start + use]
            h = torch.matmul(xf[idx], self.w_gate_up[e])  # (c, 2he)
            act = F.silu(h[:, :he]) * h[:, he:]
            out = torch.matmul(act, self.w_down[e])  # (c, d)
            out = out.float() * w_sorted[start:start + use, None]
            y.index_add_(0, idx, out.to(y.dtype))
            start = end
        return y, counts

    def _balance_loss(self, scores: torch.Tensor, topk_idx: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """Sequence-level Switch loss: weight * E * mean_b sum_e f_{b,e} P_{b,e} (float32 scalar)."""
        E, k = self.n_experts, self.top_k
        probs = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)  # (N, E)
        P = probs.reshape(B, T, E).mean(dim=1)  # (B, E)
        idx = topk_idx.reshape(B, T * k)
        f = torch.zeros(B, E, dtype=torch.float32, device=scores.device)
        f.scatter_add_(1, idx, torch.ones(idx.shape, dtype=torch.float32, device=scores.device))
        f = f / float(T * k)
        return self.aux_loss_weight * E * (f * P).sum(dim=-1).mean()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        shape = x.shape
        if x.dim() == 2:
            B, T = 1, shape[0]
        else:
            B, T = shape[0], shape[1]
        xf = x.reshape(-1, self.dim)
        scores, topk_idx, weights = self.route(xf)
        y, counts = self._experts(xf, topk_idx, weights)
        if self.shared is not None:
            y = y + self.shared(xf).to(y.dtype)
        aux = None
        if self.training:
            self._record_load(counts)
            if self.aux_loss_weight > 0 and T > 0:
                aux = self._balance_loss(scores, topk_idx, B, T)
        return y.reshape(shape), aux
