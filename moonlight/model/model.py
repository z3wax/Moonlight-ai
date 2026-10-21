"""Moonlight hybrid Attention + Retention causal language model.

Module tree / state-dict names (spec section 3; every Linear has ``bias=False``)::

    embed                       nn.Embedding(V, d)
    layers.{i}                  Block
        attn_norm               RMSNorm(d)
        mixer                   HybridMixer
            qkv                 Linear(d, hd*(H + 2*Hkv + 2*Hr))  -> [q(Ha attn, Hr ret) | k_attn | v_attn | k_ret | v_ret]
            gate                Linear(d, H*hd)            (output_gate)
            o_proj              Linear(H*hd, d)
            shift               Conv1d(d, d, k, groups=d)  (token_shift k > 0)
            decay_logit         Parameter(Hr)              (Hr > 0)
            ret_norm            RMSNorm(hd)                (Hr > 0)
            v_lambda            Parameter(())              (value_residual and i > 0)
        ffn_norm                RMSNorm(d)
        ffn                     SwiGLU(d, ffn_hidden) | MoE
    norm                        RMSNorm(d)
    lm_head                     Linear(d, V)               (only if not tie_embeddings)
    mtp.{j}                     MTPHead: norm_h, norm_e, proj Linear(2d, d)[, block Block(L+j)]
    vision_proj                 VisionProjector            (only if multimodal.enabled)

There are no other parameters; the only buffer is ``router_bias`` in MoE layers (RoPE tables
are computed on the fly, see :class:`moonlight.model.layers.RotaryEmbedding`).

Head bookkeeping: ``Hr = round(H * retention_ratio)``, ``Ha = H - Hr``,
``Hkv = n_kv_heads if Ha > 0 else 0``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import gqa_attention
from .cache import InferenceCache, LayerCache
from .layers import RMSNorm, RotaryEmbedding, SwiGLU, TokenShift, apply_rope, copy_into_, is_meta, rms
from .moe import MoE
from .retention import decay_init_logits, retention
from .vision import VisionProjector

__all__ = [
    "IMAGE_TOKEN_ID",
    "ModelOutput",
    "HybridMixer",
    "Block",
    "MTPHead",
    "MoonlightForCausalLM",
    "head_counts",
    "layer_window",
]

# Fixed by the tokenizer spec (moonlight.tokenizer.IMAGE_ID == 7); duplicated here so the model
# does not depend on the tokenizer module at import time.
IMAGE_TOKEN_ID = 7
IGNORE_INDEX = -100


# --------------------------------------------------------------------------------------------
# config helpers
# --------------------------------------------------------------------------------------------
def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a config dataclass/object or dict, with a default."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    return default if value is None and default is not None else value


def head_counts(cfg: Any) -> Tuple[int, int, int]:
    """Return ``(n_attn, n_ret, n_kv)`` = (Ha, Hr, Hkv) for a model config (spec section 3)."""
    H = int(_get(cfg, "n_heads"))
    ratio = float(_get(cfg, "retention_ratio", 0.0) or 0.0)
    n_ret = int(round(H * ratio))
    n_ret = max(0, min(H, n_ret))
    n_attn = H - n_ret
    n_kv = int(_get(cfg, "n_kv_heads", 1) or 0) if n_attn > 0 else 0
    return n_attn, n_ret, n_kv


def layer_window(cfg: Any, layer_idx: int) -> Optional[int]:
    """Sliding-window size for layer ``layer_idx`` (None = global/full attention).

    With ``attn_window`` set, layer i is global iff ``i % N == N - 1`` (N = ``max(1, global_every)``,
    same as ``ModelConfig.is_global_layer``). Without ``attn_window`` all layers are global.
    """
    window = _get(cfg, "attn_window", None)
    if window is None:
        return None
    every = max(1, int(_get(cfg, "global_every", 1) or 1))
    if layer_idx % every == every - 1:
        return None
    return int(window)


def _moe_cfg(cfg: Any) -> Any:
    return _get(cfg, "moe", None)


def _is_moe_layer(cfg: Any, layer_idx: int) -> bool:
    moe = _moe_cfg(cfg)
    if moe is None or not bool(_get(moe, "enabled", False)):
        return False
    return layer_idx >= int(_get(moe, "first_dense_layers", 0) or 0)


# --------------------------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------------------------
@dataclass
class ModelOutput:
    """Result of :meth:`MoonlightForCausalLM.forward`.

    logits:  (B, T, V) float32, or (B, 1, V) when ``use_cache`` and no targets.
    loss:    total loss (scalar) when targets are given, else None.
    lm_loss: next-token cross-entropy (scalar) when targets are given, else None.
    aux_loss: summed MoE balance loss (float32 scalar; 0 when not training / no MoE).
    cache:   the (updated) InferenceCache when caching, else None.
    hidden:  final hidden states after the final norm (B, T, d) when ``return_hidden``.
    """

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    cache: Optional[InferenceCache] = None
    hidden: Optional[torch.Tensor] = None


# --------------------------------------------------------------------------------------------
# mixer
# --------------------------------------------------------------------------------------------
class HybridMixer(nn.Module):
    """Token mixer with ``Ha`` softmax-attention heads and ``Hr`` retention heads.

    ``forward(x, cos=None, sin=None, v_first=None, cache=None) -> (out, v_first)``

    - x: (B, T, d) already normalised; out: (B, T, d).
    - cos, sin: (T, hd) float32 RoPE tables for the absolute positions of ``x`` (required when
      Ha > 0 and a cache is used; otherwise positions 0..T-1 are assumed if omitted).
    - v_first: (B, Hkv + Hr, T, hd) values of layer 0 (value residual); layer 0 (or any layer
      receiving ``None``) produces it, later layers mix ``v + sigmoid(v_lambda) * (v_first - v)``.
    - cache: LayerCache updated in place (attention k/v, retention state, token-shift buffer).
    """

    def __init__(self, cfg: Any, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        d = int(_get(cfg, "d_model"))
        H = int(_get(cfg, "n_heads"))
        hd = int(_get(cfg, "head_dim"))
        n_attn, n_ret, n_kv = head_counts(cfg)
        if n_attn > 0:
            if n_kv < 1 or n_attn % n_kv != 0:
                raise ValueError(f"n_attn={n_attn} heads must be divisible by n_kv_heads={n_kv} >= 1")
            if hd % 2 != 0:
                raise ValueError(f"attention heads need an even head_dim for RoPE, got {hd}")
        self.d_model, self.n_heads, self.head_dim = d, H, hd
        self.n_attn, self.n_ret, self.n_kv = n_attn, n_ret, n_kv
        self.window = layer_window(cfg, self.layer_idx)
        self.qk_norm = bool(_get(cfg, "qk_norm", True))
        self.value_residual = bool(_get(cfg, "value_residual", False))
        self.ret_scale = hd ** -0.5
        self._split = [H * hd, n_kv * hd, n_kv * hd, n_ret * hd, n_ret * hd]

        self.qkv = nn.Linear(d, hd * (H + 2 * n_kv + 2 * n_ret), bias=False)
        self.gate = nn.Linear(d, H * hd, bias=False) if bool(_get(cfg, "output_gate", True)) else None
        self.o_proj = nn.Linear(H * hd, d, bias=False)
        k = int(_get(cfg, "token_shift", 0) or 0)
        self.shift = TokenShift(d, k) if k > 0 else None
        if n_ret > 0:
            self.decay_logit = nn.Parameter(torch.empty(n_ret))
            self.ret_norm = RMSNorm(hd)
        else:
            self.decay_logit = None
            self.ret_norm = None
        if self.value_residual and self.layer_idx > 0:
            self.v_lambda = nn.Parameter(torch.zeros(()))
        else:
            self.v_lambda = None
        self.rope = RotaryEmbedding(hd, float(_get(cfg, "rope_theta", 10000.0))) if n_attn > 0 else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Init own parameters: decay spread (spec 4.5) and ``v_lambda = 0``."""
        if self.decay_logit is not None and not is_meta(self.decay_logit):
            values = torch.tensor(decay_init_logits(self.n_ret, self.layer_idx), dtype=torch.float32,
                                  device="cpu")
            copy_into_(self.decay_logit, values)
        if self.v_lambda is not None and not is_meta(self.v_lambda):
            nn.init.zeros_(self.v_lambda)

    def extra_repr(self) -> str:
        return (f"attn_heads={self.n_attn}, ret_heads={self.n_ret}, kv_heads={self.n_kv}, "
                f"head_dim={self.head_dim}, window={self.window}")

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        v_first: Optional[torch.Tensor] = None,
        cache: Optional[LayerCache] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, _ = x.shape
        H, hd, Ha, Hr, Hkv = self.n_heads, self.head_dim, self.n_attn, self.n_ret, self.n_kv

        # 1. token shift (causal depthwise conv); the gate reads the unshifted x
        if self.shift is None:
            xs = x
        elif cache is not None:
            xs, cache.conv_buf = self.shift(x, cache.conv_buf, return_buffer=True)
        else:
            xs = self.shift(x)

        # 2. projections -> heads (B, h, T, hd); parameter-free q/k RMS norm
        # (reshape == view here; it also covers the zero-width splits when Hkv == 0 or Hr == 0)
        q, ka, va, kr, vr = torch.split(self.qkv(xs), self._split, dim=-1)
        q = q.reshape(B, T, H, hd).transpose(1, 2)
        ka = ka.reshape(B, T, Hkv, hd).transpose(1, 2)
        va = va.reshape(B, T, Hkv, hd).transpose(1, 2)
        kr = kr.reshape(B, T, Hr, hd).transpose(1, 2)
        vr = vr.reshape(B, T, Hr, hd).transpose(1, 2)
        if self.qk_norm:
            q = rms(q)
            ka = rms(ka)
            kr = rms(kr)

        # 3. value residual over [v_attn (Hkv) | v_ret (Hr)]
        if self.value_residual:
            v = torch.cat([va, vr], dim=1)  # (B, Hkv + Hr, T, hd)
            if v_first is None:
                v_first = v
            elif self.v_lambda is not None:
                lam = torch.sigmoid(self.v_lambda.float()).to(v.dtype)
                v = v + lam * (v_first.to(v.dtype) - v)
                va, vr = v[:, :Hkv], v[:, Hkv:]

        heads: List[torch.Tensor] = []
        # 4. softmax attention heads (GQA, RoPE, causal / sliding window, KV cache)
        if Ha > 0:
            if cos is None or sin is None:
                if cache is not None:
                    raise ValueError("cos/sin (absolute-position RoPE tables) are required with a cache")
                cos, sin = self.rope(torch.arange(T, device=x.device))
            qa = apply_rope(q[:, :Ha], cos, sin)
            ka = apply_rope(ka, cos, sin)
            heads.append(gqa_attention(qa, ka, va, window=self.window, cache=cache))

        # 5. retention heads (decay in float32), then per-head RMSNorm
        if Hr > 0:
            qr = q[:, Ha:] * self.ret_scale
            log_gamma = F.logsigmoid(self.decay_logit.float())
            if cache is not None:
                o_r, cache.state = retention(qr, kr, vr, log_gamma, state=cache.state, return_state=True)
            else:
                o_r = retention(qr, kr, vr, log_gamma)
            heads.append(self.ret_norm(o_r))

        # 6. concat [attn | ret] -> (B, T, H*hd), output gate, projection
        if len(heads) == 1:
            y = heads[0]
        else:
            y = torch.cat([heads[0], heads[1].to(heads[0].dtype)], dim=1)
        y = y.transpose(1, 2).reshape(B, T, H * hd)
        if self.gate is not None:
            y = y * F.silu(self.gate(x))
        return self.o_proj(y), v_first


# --------------------------------------------------------------------------------------------
# block
# --------------------------------------------------------------------------------------------
class Block(nn.Module):
    """Pre-norm residual block: ``x += mixer(attn_norm(x)); x += ffn(ffn_norm(x))``.

    ``forward(x, cos=None, sin=None, v_first=None, cache=None) -> (x, v_first, aux_loss|None)``
    with x (B, T, d). The FFN is a dense SwiGLU(d, ffn_hidden) unless MoE is enabled and
    ``layer_idx >= moe.first_dense_layers``.
    """

    def __init__(self, cfg: Any, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        d = int(_get(cfg, "d_model"))
        n_layers = int(_get(cfg, "n_layers"))
        init_std = float(_get(cfg, "init_std", 0.02))
        self.attn_norm = RMSNorm(d)
        self.mixer = HybridMixer(cfg, layer_idx)
        self.ffn_norm = RMSNorm(d)
        self.is_moe = _is_moe_layer(cfg, layer_idx)
        if self.is_moe:
            moe = _moe_cfg(cfg)
            self.ffn = MoE(
                d,
                n_experts=int(_get(moe, "n_experts")),
                top_k=int(_get(moe, "top_k")),
                expert_hidden=int(_get(moe, "expert_hidden")),
                n_shared=int(_get(moe, "n_shared", 0) or 0),
                aux_loss_weight=float(_get(moe, "aux_loss_weight", 0.0) or 0.0),
                capacity_factor=float(_get(moe, "capacity_factor", 0.0) or 0.0),
                init_std=init_std,
                out_std=init_std / math.sqrt(2 * max(n_layers, 1)),
            )
        else:
            self.ffn = SwiGLU(d, int(_get(cfg, "ffn_hidden")))

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        v_first: Optional[torch.Tensor] = None,
        cache: Optional[LayerCache] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        h, v_first = self.mixer(self.attn_norm(x), cos, sin, v_first, cache)
        x = x + h
        aux = None
        if self.is_moe:
            f, aux = self.ffn(self.ffn_norm(x))
        else:
            f = self.ffn(self.ffn_norm(x))
        x = x + f
        return x, v_first, aux


class MTPHead(nn.Module):
    """Multi-token-prediction head ``j`` (predicts token t+j+2 at position t).

    ``h' = proj(cat(norm_h(h), norm_e(e)))`` followed by ``block`` (a full Block with layer index
    L + j) when ``mtp_block``. Output head and final norm are shared with the main model.
    ``forward(h, e, cos=None, sin=None, v_first=None) -> (h' (B, T, d) in h.dtype, aux|None)``.
    """

    def __init__(self, cfg: Any, depth_idx: int) -> None:
        super().__init__()
        d = int(_get(cfg, "d_model"))
        n_layers = int(_get(cfg, "n_layers"))
        self.depth_idx = int(depth_idx)
        self.norm_h = RMSNorm(d)
        self.norm_e = RMSNorm(d)
        self.proj = nn.Linear(2 * d, d, bias=False)
        self.block = Block(cfg, n_layers + self.depth_idx) if bool(_get(cfg, "mtp_block", False)) else None

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        v_first: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.proj(torch.cat([self.norm_h(h), self.norm_e(e).to(h.dtype)], dim=-1)).to(h.dtype)
        aux = None
        if self.block is not None:
            x, _, aux = self.block(x, cos, sin, v_first, None)
        return x, aux


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _checkpoint(fn, *args):
    """Non-reentrant activation checkpoint of ``fn(*args)``.

    The recomputation during backward re-runs MoE routing with identical inputs and router
    biases, so a checkpointed MoE layer records its expert load twice per step. That scales all
    of the layer's counts by the same factor and leaves ``sign(mean_load - load)`` -- the only use
    of the load -- unchanged.
    """
    from torch.utils.checkpoint import checkpoint

    return checkpoint(fn, *args, use_reentrant=False)


def _add(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _shift_left(t: torch.Tensor, s: int, fill: int) -> torch.Tensor:
    """(B, T) -> (B, T): out[:, i] = t[:, i + s], padded with ``fill`` at the end."""
    kept = t[:, s:]
    pad = t.shape[1] - kept.shape[1]
    return torch.cat([kept, t.new_full((t.shape[0], pad), fill)], dim=1)


def _masked_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over targets != -100 (0 if there are none). logits (B,T,V) float32."""
    V = logits.shape[-1]
    flat = targets.reshape(-1)
    total = F.cross_entropy(logits.reshape(-1, V), flat, ignore_index=IGNORE_INDEX, reduction="sum")
    count = (flat != IGNORE_INDEX).sum().clamp(min=1)
    return total / count


def _z_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """mean(logsumexp(logits)^2) over positions with targets != -100."""
    lse = torch.logsumexp(logits, dim=-1)  # (B, T)
    valid = (targets != IGNORE_INDEX).to(lse.dtype)
    return (lse.pow(2) * valid).sum() / valid.sum().clamp(min=1.0)


# --------------------------------------------------------------------------------------------
# full model
# --------------------------------------------------------------------------------------------
class MoonlightForCausalLM(nn.Module):
    """Hybrid Attention + Retention causal LM (see module docstring for the parameter layout).

    ``MoonlightForCausalLM(cfg: ModelConfig, mm: MultimodalConfig | None = None)``; a full
    ``Config`` is also accepted as ``cfg`` (its ``.model``/``.multimodal`` are used). Construction
    works under ``torch.device("meta")`` (initialisation skips meta tensors); after materialising
    with ``to_empty`` call :meth:`reset_parameters` (alias :meth:`init_weights`).

    ``forward(input_ids, targets=None, image_features=None, cache=None, use_cache=False,
    return_hidden=False) -> ModelOutput``

    - input_ids: (B, T) int64 (a 1-D tensor is treated as B=1).
    - targets: (B, T) labels already shifted by the caller (-100 = ignore).
    - image_features: (B_img, n_image_tokens, vision_dim); projected and scattered (row-major)
      into the positions where ``input_ids == 7`` (<|image|>).
    - cache / use_cache: incremental decoding. ``use_cache=True`` creates a cache if none is
      given. A given cache is always read and updated. Logits are restricted to the last position
      (B, 1, V) iff ``use_cache`` and ``targets is None``.
    """

    _no_split_modules = ["Block"]
    block_class = Block

    def __init__(self, cfg: Any, mm: Any = None) -> None:
        super().__init__()
        if not hasattr(cfg, "d_model") and not (isinstance(cfg, dict) and "d_model" in cfg) \
                and _get(cfg, "model", None) is not None:
            if mm is None:
                mm = _get(cfg, "multimodal", None)
            cfg = _get(cfg, "model")
        self.config = cfg
        self.mm_config = mm if (mm is not None and bool(_get(mm, "enabled", False))) else None

        V = int(_get(cfg, "vocab_size"))
        d = int(_get(cfg, "d_model"))
        L = int(_get(cfg, "n_layers"))
        hd = int(_get(cfg, "head_dim"))
        self.vocab_size, self.d_model, self.n_layers, self.head_dim = V, d, L, hd
        self.n_attn_heads, self.n_ret_heads, self.n_kv_heads = head_counts(cfg)
        self.init_std = float(_get(cfg, "init_std", 0.02))
        self.tie_embeddings = bool(_get(cfg, "tie_embeddings", True))
        self.softcap = float(_get(cfg, "softcap", 0.0) or 0.0)
        self.z_loss_weight = float(_get(cfg, "z_loss", 0.0) or 0.0)
        self.mtp_weight = float(_get(cfg, "mtp_weight", 0.0) or 0.0)
        self.activation_checkpointing = bool(_get(cfg, "activation_checkpointing", False))
        moe = _moe_cfg(cfg)
        self.bias_update_rate = float(_get(moe, "bias_update_rate", 0.0) or 0.0)

        self.embed = nn.Embedding(V, d)
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(L)])
        self.norm = RMSNorm(d)
        self.lm_head = None if self.tie_embeddings else nn.Linear(d, V, bias=False)
        mtp_depth = int(_get(cfg, "mtp_depth", 0) or 0)
        self.mtp = nn.ModuleList([MTPHead(cfg, j) for j in range(mtp_depth)])
        if self.mm_config is not None:
            self.vision_proj = VisionProjector(int(_get(self.mm_config, "vision_dim", 768)), d,
                                               self.init_std)
        else:
            self.vision_proj = None
        self.rope = (RotaryEmbedding(hd, float(_get(cfg, "rope_theta", 10000.0)))
                     if self.n_attn_heads > 0 else None)
        self.reset_parameters()

    # -- construction helpers -------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Any) -> "MoonlightForCausalLM":
        """Build from a full ``Config`` (uses ``config.model`` and ``config.multimodal``) or a
        ``ModelConfig``."""
        return cls(config)

    @staticmethod
    def get_block_class() -> type:
        """The transformer block type (wrap unit for FSDP / activation checkpointing)."""
        return Block

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """(Re-)initialise every parameter and buffer (meta tensors are skipped).

        Linear/Embedding ~ N(0, init_std); ``o_proj`` and FFN down projections (dense, shared and
        routed experts) ~ N(0, init_std / sqrt(2 * n_layers)); RMSNorm weights 1; token shift
        identity; retention decay spread; ``v_lambda`` 0; ``router_bias`` 0.
        """
        std = self.init_std
        out_std = std / math.sqrt(2 * max(self.n_layers, 1))
        for name, module in self.named_modules():
            if isinstance(module, TokenShift):
                module.reset_parameters()
            elif isinstance(module, nn.Linear):
                leaf = name.rsplit(".", 1)[-1]
                if not is_meta(module.weight):
                    nn.init.normal_(module.weight, mean=0.0,
                                    std=out_std if leaf in ("o_proj", "w_down") else std)
            elif isinstance(module, nn.Embedding):
                if not is_meta(module.weight):
                    nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, RMSNorm):
                module.reset_parameters()
            elif isinstance(module, MoE):
                module.reset_parameters(init_std=std, out_std=out_std)
            elif isinstance(module, HybridMixer):
                module.reset_parameters()

    init_weights = reset_parameters

    def new_cache(self) -> InferenceCache:
        return InferenceCache(self.n_layers)

    # -- info -----------------------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.embed.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embed.weight.dtype

    def num_params(self, trainable_only: bool = False, exclude_embedding: bool = False) -> int:
        """Number of (unique) parameters; buffers are not counted."""
        total = 0
        for p in self.parameters():
            if trainable_only and not p.requires_grad:
                continue
            total += p.numel()
        if exclude_embedding:
            total -= self.embed.weight.numel()
        return int(total)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def get_output_embeddings(self) -> nn.Module:
        return self.embed if self.lm_head is None else self.lm_head

    def layer_types(self) -> List[str]:
        """'global' or 'local' (sliding window) for each layer's attention."""
        return ["global" if b.mixer.window is None else "local" for b in self.layers]

    def set_activation_checkpointing(self, enabled: bool = True) -> None:
        self.activation_checkpointing = bool(enabled)

    # -- MoE balancing --------------------------------------------------------------------------
    @torch.no_grad()
    def update_router_biases(self, rate: Optional[float] = None, sync: bool = True) -> None:
        """Aux-loss-free balancing step (call after each optimizer step).

        For every MoE layer: ``router_bias += rate * sign(mean_load - load)`` with the expert load
        accumulated over the training forwards since the previous call (then reset). ``rate``
        defaults to ``moe.bias_update_rate``. With ``sync`` and an initialised process group the
        loads are summed across ranks first (one all-reduce), so every rank stays identical --
        call it on all ranks.
        """
        r = self.bias_update_rate if rate is None else float(rate)
        moes = [m for m in self.modules() if isinstance(m, MoE)]
        pending = [(m, m.consume_load()) for m in moes]
        pending = [(m, load) for m, load in pending if load is not None]
        if not pending:
            return
        import torch.distributed as dist

        if sync and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            flat = torch.cat([load.float().reshape(-1) for _, load in pending])
            dist.all_reduce(flat)
            sizes = [load.numel() for _, load in pending]
            pending = [(m, part) for (m, _), part in zip(pending, torch.split(flat, sizes))]
        if r == 0.0:
            return
        for m, load in pending:
            m.apply_bias_update(load, r)

    # -- forward pieces -------------------------------------------------------------------------
    def _head(self, x: torch.Tensor) -> torch.Tensor:
        """(..., d) -> float32 logits (..., V) with optional soft-cap ``c * tanh(z / c)``."""
        weight = self.embed.weight if self.lm_head is None else self.lm_head.weight
        logits = F.linear(x, weight).float()
        if self.softcap > 0:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits

    def _merge_image_features(self, x: torch.Tensor, input_ids: torch.Tensor,
                              image_features: torch.Tensor) -> torch.Tensor:
        """Project features (B_img, n, vision_dim) and scatter them into <|image|> positions.

        x: (B, T, d) token embeddings; returns the merged (B, T, d) embeddings (``x`` itself when
        there are no image slots).
        """
        if self.vision_proj is None:
            raise ValueError("image_features given but the model was built without multimodal.enabled")
        feats = image_features
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        if feats.dim() != 3:
            raise ValueError(f"image_features must be (B_img, n_tokens, vision_dim), got {tuple(feats.shape)}")
        mask = input_ids == IMAGE_TOKEN_ID  # (B, T)
        n_slots = int(mask.sum().item())
        n_feats = int(feats.shape[0] * feats.shape[1])
        if n_slots != n_feats:
            raise ValueError(f"input_ids contain {n_slots} <|image|> tokens but image_features provide "
                             f"{n_feats} (= {feats.shape[0]} x {feats.shape[1]})")
        if n_slots == 0:
            return x
        proj = self.vision_proj(feats.to(device=x.device, dtype=self.vision_proj.fc1.weight.dtype))
        src = proj.reshape(-1, proj.shape[-1]).to(x.dtype)
        return x.masked_scatter(mask.unsqueeze(-1).expand_as(x), src)

    def _run_layers(self, x: torch.Tensor, cos, sin, cache: Optional[InferenceCache]):
        v_first = None
        aux_total = None
        use_ckpt = (self.activation_checkpointing and self.training and cache is None
                    and torch.is_grad_enabled())
        for i, layer in enumerate(self.layers):
            if use_ckpt:
                x, v_first, aux = _checkpoint(layer, x, cos, sin, v_first, None)
            else:
                x, v_first, aux = layer(x, cos, sin, v_first, cache.layers[i] if cache is not None else None)
            aux_total = _add(aux_total, aux)
        return x, v_first, aux_total

    def _mtp_losses(self, h: torch.Tensor, input_ids: torch.Tensor, targets: torch.Tensor,
                    cos, sin, v_first) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Mean MTP cross-entropy over heads and their summed MoE aux loss.

        Head j sees ``h_t`` (depth j-1 output; depth 0 = last layer output before the final
        norm) and ``embed(input_ids[t+j+1])`` and is trained on ``targets[t+j+1]``.
        """
        losses = []
        aux_total = None
        for j, head in enumerate(self.mtp):
            s = j + 1
            ids_j = _shift_left(input_ids, s, 0)
            tgt_j = _shift_left(targets, s, IGNORE_INDEX)
            h, aux = head(h, self.embed(ids_j), cos, sin, v_first)
            losses.append(_masked_ce(self._head(self.norm(h)), tgt_j))
            aux_total = _add(aux_total, aux)
        return torch.stack(losses).mean(), aux_total

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        cache: Optional[InferenceCache] = None,
        use_cache: bool = False,
        return_hidden: bool = False,
    ) -> ModelOutput:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            if targets is not None and targets.dim() == 1:
                targets = targets.unsqueeze(0)
        B, T = input_ids.shape
        if use_cache and cache is None:
            cache = InferenceCache(self.n_layers)
        offset = cache.seen_tokens if cache is not None else 0

        x = self.embed(input_ids)
        vision_used = False
        if image_features is not None:
            merged = self._merge_image_features(x, input_ids, image_features)
            vision_used = merged is not x
            x = merged
        cos = sin = None
        if self.rope is not None:
            cos, sin = self.rope(torch.arange(offset, offset + T, device=input_ids.device))

        h, v_first, aux_total = self._run_layers(x, cos, sin, cache)
        if cache is not None:
            cache.seen_tokens += T
        hn = self.norm(h)
        if use_cache and targets is None:
            logits = self._head(hn[:, -1:])
        else:
            logits = self._head(hn)

        loss = lm_loss = None
        if targets is not None:
            targets = targets.to(device=logits.device, dtype=torch.long)
            lm_loss = _masked_ce(logits, targets)
            loss = lm_loss
            if len(self.mtp) > 0:
                mtp_loss, mtp_aux = self._mtp_losses(h, input_ids, targets, cos, sin, v_first)
                loss = loss + self.mtp_weight * mtp_loss
                aux_total = _add(aux_total, mtp_aux)
            if aux_total is not None:
                loss = loss + aux_total
            if self.z_loss_weight > 0:
                loss = loss + self.z_loss_weight * _z_loss(logits, targets)
            if self.training and not vision_used and self.vision_proj is not None:
                # keep the projector in the graph (zero contribution) so DDP/FSDP never see
                # parameters without gradients on text-only batches
                loss = loss + 0.0 * sum(p.sum() for p in self.vision_proj.parameters()).float()
        aux_loss = aux_total if aux_total is not None else logits.new_zeros(())
        return ModelOutput(
            logits=logits,
            loss=loss,
            lm_loss=lm_loss,
            aux_loss=aux_loss,
            cache=cache,
            hidden=hn if return_hidden else None,
        )

    # -- generation -----------------------------------------------------------------------------
    def generate(self, input_ids, **kwargs):
        """Shorthand for :func:`moonlight.model.generation.generate` (returns GenerationResult)."""
        from .generation import generate

        return generate(self, input_ids, **kwargs)
