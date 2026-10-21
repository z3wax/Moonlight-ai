"""Exact parameter-count estimator for Moonlight models (SPEC §3). Pure python, torch-free.

The estimator enumerates a *virtual state dict* that mirrors ``MoonlightForCausalLM`` exactly
(names and shapes follow SPEC §3; all Linear layers are bias-free; buffers such as
``router_bias`` and RoPE cos/sin are not parameters and are not counted):

=========================================  ==========================================================
name                                       shape
=========================================  ==========================================================
``embed.weight``                           (V, d)
``layers.{i}.attn_norm.weight``            (d,)
``layers.{i}.mixer.qkv.weight``            (hd*(H + 2*Hkv + 2*Hr), d)
``layers.{i}.mixer.gate.weight``           (H*hd, d)            if output_gate
``layers.{i}.mixer.o_proj.weight``         (d, H*hd)
``layers.{i}.mixer.shift.weight``          (d, 1, k)            if token_shift k > 0 (depthwise Conv1d)
``layers.{i}.mixer.decay_logit``           (Hr,)                if Hr > 0
``layers.{i}.mixer.ret_norm.weight``       (hd,)                if Hr > 0
``layers.{i}.mixer.v_lambda``              ()                   if value_residual and i > 0
``layers.{i}.ffn_norm.weight``             (d,)
``layers.{i}.ffn.w_gate_up.weight``        (2*h, d)             dense SwiGLU (no MoE, or i < first_dense_layers)
``layers.{i}.ffn.w_down.weight``           (d, h)
``layers.{i}.ffn.router.weight``           (E, d)               MoE layers
``layers.{i}.ffn.w_gate_up``               (E, d, 2*he)         routed experts (Parameter)
``layers.{i}.ffn.w_down``                  (E, he, d)
``layers.{i}.ffn.shared.w_gate_up.weight`` (2*he*n_shared, d)   if n_shared > 0
``layers.{i}.ffn.shared.w_down.weight``    (d, he*n_shared)
``norm.weight``                            (d,)
``lm_head.weight``                         (V, d)               only if not tie_embeddings
``mtp.{j}.norm_h.weight`` / ``norm_e``     (d,) each
``mtp.{j}.proj.weight``                    (d, 2*d)
``mtp.{j}.block.*``                        a Block with layer index L+j   if mtp_block
``vision_proj.norm.weight``                (vision_dim,)        if multimodal.enabled
``vision_proj.fc1.weight``                 (d, vision_dim)
``vision_proj.fc2.weight``                 (d, d)
=========================================  ==========================================================

with ``Hr = round(H * retention_ratio)`` (Python round), ``Ha = H - Hr``,
``Hkv = n_kv_heads if Ha > 0 else 0``. MTP blocks follow the same FFN rule as a non-first layer
(MoE when ``moe.enabled``, since ``L + j >= first_dense_layers``) and get ``v_lambda`` because their
index ``L + j`` is > 0.

``active`` counts routed expert weights at ``top_k / n_experts`` (exactly ``3*top_k*d*he`` per MoE
layer) and everything else fully (router, shared experts, embeddings, MTP heads, vision projector).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from moonlight.config import Config, ModelConfig, MultimodalConfig, format_size

__all__ = [
    "ParamReport",
    "estimate_params",
    "param_shapes",
    "block_param_count",
    "flops_per_token",
    "memory_estimate",
    "BREAKDOWN_KEYS",
]

#: Keys of ``ParamReport.breakdown`` (always all present, in this order; sum == total).
BREAKDOWN_KEYS: Tuple[str, ...] = (
    "embed",
    "lm_head",
    "norm",
    "layers.norms",
    "layers.mixer.qkv",
    "layers.mixer.gate",
    "layers.mixer.o_proj",
    "layers.mixer.shift",
    "layers.mixer.retention",
    "layers.mixer.v_lambda",
    "layers.ffn.dense",
    "layers.ffn.router",
    "layers.ffn.experts",
    "layers.ffn.shared",
    "mtp",
    "vision_proj",
)

_GiB = float(2**30)
_DTYPE_BYTES = {"fp32": 4, "float32": 4, "float": 4, "tf32": 4, "bf16": 2, "bfloat16": 2, "fp16": 2,
                "float16": 2, "half": 2, "fp8": 1, "float8": 1, "int8": 1}


class _Entry(NamedTuple):
    name: str                 # state_dict name
    shape: Tuple[int, ...]
    group: str                # BREAKDOWN_KEYS member
    routed: bool              # routed-expert weight (counted top_k/E in ``active``)
    layer: Optional[int]      # main block index, None for non-block / MTP params


def _numel(shape: Tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


# ------------------------------------------------------------------------------------------------
# Virtual state dict
# ------------------------------------------------------------------------------------------------


def _block_entries(m: ModelConfig, i: int, prefix: str, group_prefix: Optional[str],
                   layer: Optional[int]) -> List[_Entry]:
    """Parameters of ``Block`` with layer index ``i`` (``group_prefix=None`` puts all in ``"mtp"``)."""
    d, H, hd, k = m.d_model, m.n_heads, m.head_dim, m.token_shift
    Hr, Hkv = m.n_retention_heads, m.n_attn_kv_heads
    out: List[_Entry] = []

    def add(name: str, shape: Tuple[int, ...], group: str, routed: bool = False) -> None:
        g = "mtp" if group_prefix is None else f"{group_prefix}.{group}"
        out.append(_Entry(prefix + name, tuple(int(s) for s in shape), g, routed, layer))

    add("attn_norm.weight", (d,), "norms")
    add("mixer.qkv.weight", (hd * (H + 2 * Hkv + 2 * Hr), d), "mixer.qkv")
    if m.output_gate:
        add("mixer.gate.weight", (H * hd, d), "mixer.gate")
    add("mixer.o_proj.weight", (d, H * hd), "mixer.o_proj")
    if k > 0:
        add("mixer.shift.weight", (d, 1, k), "mixer.shift")
    if Hr > 0:
        add("mixer.decay_logit", (Hr,), "mixer.retention")
        add("mixer.ret_norm.weight", (hd,), "mixer.retention")
    if m.value_residual and i > 0:
        add("mixer.v_lambda", (), "mixer.v_lambda")
    add("ffn_norm.weight", (d,), "norms")
    if m.is_moe_layer(i):
        E, he, ns = m.moe.n_experts, m.moe.expert_hidden, m.moe.n_shared
        add("ffn.router.weight", (E, d), "ffn.router")
        add("ffn.w_gate_up", (E, d, 2 * he), "ffn.experts", routed=True)
        add("ffn.w_down", (E, he, d), "ffn.experts", routed=True)
        if ns > 0:
            add("ffn.shared.w_gate_up.weight", (2 * he * ns, d), "ffn.shared")
            add("ffn.shared.w_down.weight", (d, he * ns), "ffn.shared")
    else:
        h = m.ffn_hidden
        add("ffn.w_gate_up.weight", (2 * h, d), "ffn.dense")
        add("ffn.w_down.weight", (d, h), "ffn.dense")
    return out


def _entries(m: ModelConfig, mm: Optional[MultimodalConfig]) -> List[_Entry]:
    V, d, L = m.vocab_size, m.d_model, m.n_layers
    out: List[_Entry] = [_Entry("embed.weight", (V, d), "embed", False, None)]
    for i in range(L):
        out.extend(_block_entries(m, i, f"layers.{i}.", "layers", i))
    out.append(_Entry("norm.weight", (d,), "norm", False, None))
    if not m.tie_embeddings:
        out.append(_Entry("lm_head.weight", (V, d), "lm_head", False, None))
    for j in range(m.mtp_depth):
        p = f"mtp.{j}."
        out.append(_Entry(p + "norm_h.weight", (d,), "mtp", False, None))
        out.append(_Entry(p + "norm_e.weight", (d,), "mtp", False, None))
        out.append(_Entry(p + "proj.weight", (d, 2 * d), "mtp", False, None))
        if m.mtp_block:
            out.extend(_block_entries(m, L + j, p + "block.", None, None))
    if mm is not None and mm.enabled:
        vd = mm.vision_dim
        out.append(_Entry("vision_proj.norm.weight", (vd,), "vision_proj", False, None))
        out.append(_Entry("vision_proj.fc1.weight", (d, vd), "vision_proj", False, None))
        out.append(_Entry("vision_proj.fc2.weight", (d, d), "vision_proj", False, None))
    return out


def _resolve(cfg: Any) -> Tuple[ModelConfig, Optional[MultimodalConfig], Optional[Config]]:
    """Accept a Config, a ModelConfig, a mapping (full config or model section) or a registry
    name / path; return ``(model_cfg, multimodal_cfg_or_None, full_config_or_None)``."""
    if isinstance(cfg, Config):
        return cfg.model, cfg.multimodal, cfg
    if isinstance(cfg, ModelConfig):
        return cfg, None, None
    if isinstance(cfg, Mapping):
        top = {"name", "model", "multimodal", "tokenizer", "data", "train", "chat", "tier", "kind", "extends"}
        if any(k in top for k in cfg):
            full = Config.from_dict(cfg)
            return full.model, full.multimodal, full
        return ModelConfig.from_dict(cfg), None, None
    if isinstance(cfg, (str, os.PathLike)):
        from moonlight.registry import load

        full = load(cfg, validate=False)
        return full.model, full.multimodal, full
    raise TypeError(f"expected Config, ModelConfig, mapping or config name, got {type(cfg).__name__}")


def param_shapes(cfg_or_model_cfg: Any) -> Dict[str, Tuple[int, ...]]:
    """Expected ``{state_dict_name: shape}`` of all PARAMETERS (no buffers) of the model.

    Useful to diagnose a mismatch against ``dict(model.named_parameters())``.
    """
    m, mm, _ = _resolve(cfg_or_model_cfg)
    return {e.name: e.shape for e in _entries(m, mm)}


def block_param_count(model_cfg: ModelConfig, i: int) -> int:
    """Parameters of the Block with layer index ``i`` (``i >= n_layers`` = MTP block ``i - n_layers``)."""
    return sum(_numel(e.shape) for e in _block_entries(model_cfg, i, "", "layers", i))


# ------------------------------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------------------------------


@dataclass
class ParamReport:
    """Result of :func:`estimate_params`.

    Attributes:
        total: all parameters (== ``sum(p.numel() for p in model.parameters())``).
        active: parameters used per token (routed experts at ``top_k / n_experts``).
        embedding: vocabulary matrices: ``embed`` (+ ``lm_head`` when untied).
        per_layer: parameters of each main Block ``layers.0 .. layers.{L-1}``.
        breakdown: ``BREAKDOWN_KEYS -> count`` (sums to ``total``; MTP heads incl. their blocks
            are under ``"mtp"``, the vision projector under ``"vision_proj"``).
    """

    total: int
    active: int
    embedding: int
    per_layer: List[int]
    breakdown: Dict[str, int]
    name: str = ""
    nominal: Optional[int] = None
    routed_expert_params: int = 0

    @property
    def non_embedding(self) -> int:
        return self.total - self.embedding

    @property
    def active_fraction(self) -> float:
        return self.active / self.total if self.total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "total": self.total, "active": self.active, "embedding": self.embedding,
                "non_embedding": self.non_embedding, "nominal": self.nominal,
                "routed_expert_params": self.routed_expert_params, "per_layer": list(self.per_layer),
                "breakdown": dict(self.breakdown)}

    def pretty(self) -> str:
        """Multi-line human-readable summary (zero breakdown entries are omitted)."""

        def pct(n: int) -> str:
            return f"{100.0 * n / self.total:5.1f}%" if self.total else "  n/a"

        title = f"Parameters of {self.name}" if self.name else "Parameters"
        lines = [title]
        rows = [("total", self.total, ""), ("active", self.active, f"({pct(self.active).strip()} of total)"),
                ("embedding", self.embedding, f"({pct(self.embedding).strip()} of total)"),
                ("non-embedding", self.non_embedding, "")]
        if self.nominal:
            ratio = self.total / self.nominal
            rows.append(("nominal", self.nominal, f"(estimate is {ratio:.2f}x nominal)"))
        for label, n, note in rows:
            lines.append(f"  {label:<14}{format_size(n, 2):>9}  {n:>19,}  {note}".rstrip())
        if self.per_layer:
            lo, hi = min(self.per_layer), max(self.per_layer)
            spread = f" (min {lo:,} / max {hi:,})" if lo != hi else f" ({hi:,} each)"
            lines.append(f"  {'layers':<14}{len(self.per_layer)} x {format_size(hi, 2)}{spread}")
        lines.append("Breakdown")
        width = max(len(k) for k in self.breakdown) if self.breakdown else 10
        for key, n in self.breakdown.items():
            if n:
                lines.append(f"  {key:<{width}}  {n:>19,}  {pct(n)}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.pretty()


def estimate_params(cfg_or_model_cfg: Any) -> ParamReport:
    """Exact parameter counts for a ``Config`` (incl. the vision projector when
    ``multimodal.enabled``) or a bare ``ModelConfig`` (no vision projector). Also accepts a nested
    mapping or a registry name/path. See the module docstring for the layout."""
    m, mm, full = _resolve(cfg_or_model_cfg)
    entries = _entries(m, mm)
    breakdown: Dict[str, int] = {k: 0 for k in BREAKDOWN_KEYS}
    per_layer = [0] * m.n_layers
    total = active = routed = 0
    E, top_k = max(1, m.moe.n_experts), m.moe.top_k
    for e in entries:
        n = _numel(e.shape)
        total += n
        breakdown[e.group] += n
        if e.layer is not None:
            per_layer[e.layer] += n
        if e.routed:
            routed += n
            active += (n // E) * top_k  # shape[0] == E, so this is exact
        else:
            active += n
    nominal: Optional[int] = None
    if full is not None:
        try:
            nominal = full.nominal_param_count
        except ValueError:
            nominal = None
    return ParamReport(total=total, active=active, embedding=breakdown["embed"] + breakdown["lm_head"],
                       per_layer=per_layer, breakdown=breakdown, name=full.name if full is not None else "",
                       nominal=nominal, routed_expert_params=routed)


# ------------------------------------------------------------------------------------------------
# FLOPs
# ------------------------------------------------------------------------------------------------


def _mean_causal_context(seq_len: int, window: Optional[int]) -> float:
    """Mean number of attended keys per query for causal attention over ``seq_len`` tokens,
    optionally limited to a sliding window (query t sees keys in (t - window, t])."""
    T = max(1, int(seq_len))
    if window is None or window >= T:
        return (T + 1) / 2.0
    w = max(1, int(window))
    return (w * (w + 1) / 2.0 + (T - w) * w) / T


def flops_per_token(cfg: Any, seq_len: Optional[int] = None) -> float:
    """Approximate TRAINING FLOPs per token (forward + backward = 3x forward).

    ``6 * N_matmul + mixer term`` where

    * ``N_matmul = active - embed - lm_head + V*d*(1 + mtp_depth)``: every active weight does one
      multiply-add per token (2 FLOPs fwd, 4 bwd); the input embedding is a free lookup; the output
      head (tied or not) is applied once for the main loss and once per MTP head.
    * attention heads: ``12 * Ha * hd * ctx`` per mixer, where ``ctx`` is the mean causal context
      (``(T+1)/2`` for global layers, window-limited for local ones): QK^T and AV each cost
      ``2*hd`` FLOPs per (query, key) pair forward.
    * retention heads: ``12 * Hr * hd * ctx_chunk`` for the intra-chunk part (``chunk = T`` when
      ``T <= 256`` - parallel form - else 64) plus ``12 * Hr * hd**2`` per token for the
      chunkwise state read/update when ``T > 256``.

    Mixers counted: the ``n_layers`` main blocks plus ``mtp_depth`` MTP blocks if ``mtp_block``.
    ``seq_len`` defaults to ``train.seq_len`` for a full Config, else ``model.max_seq_len``.
    Elementwise ops, norms, softmax and the vision encoder are ignored.
    """
    m, mm, full = _resolve(cfg)
    rep = estimate_params(full if full is not None else m)
    if seq_len is None:
        seq_len = full.train.seq_len if full is not None else m.max_seq_len
    T = max(1, int(seq_len))
    V, d, hd = m.vocab_size, m.d_model, m.head_dim
    Ha, Hr = m.n_attention_heads, m.n_retention_heads
    n_matmul = rep.active - rep.breakdown["embed"] - rep.breakdown["lm_head"] + V * d * (1 + m.mtp_depth)
    flops = 6.0 * n_matmul
    mixer_layers = list(range(m.n_layers))
    if m.mtp_block:
        mixer_layers += [m.n_layers + j for j in range(m.mtp_depth)]
    chunked = T > 256
    chunk = 64 if chunked else T
    for i in mixer_layers:
        if Ha > 0:
            flops += 12.0 * Ha * hd * _mean_causal_context(T, m.layer_window(i))
        if Hr > 0:
            flops += 12.0 * Hr * hd * _mean_causal_context(chunk, None)
            if chunked:
                flops += 12.0 * Hr * hd * hd
    return float(flops)


# ------------------------------------------------------------------------------------------------
# Memory
# ------------------------------------------------------------------------------------------------


def _dtype_bytes(dtype: Any) -> int:
    key = str(dtype).strip().lower()
    if key.startswith("torch."):
        key = key[len("torch."):]
    if key not in _DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of bf16, fp16, fp32 (or torch dtype names)")
    return _DTYPE_BYTES[key]


def _uses_muon(e: _Entry) -> bool:
    """SPEC §7: Muon for ndim >= 2 parameters except embedding / lm_head / router / vision norm."""
    if len(e.shape) < 2:
        return False
    return not (e.name.startswith(("embed.", "lm_head.", "vision_proj.norm")) or ".router." in e.name)


def memory_estimate(cfg: Any, dtype: str = "bf16") -> Dict[str, Any]:
    """Rough per-replica memory (GiB = 2**30 bytes) of model state for training and inference.

    Assumptions (documented, deliberately simple):

    * ``weights_gb``: every parameter stored in ``dtype`` (bf16/fp16 = 2 B, fp32 = 4 B).
    * ``grads_gb``: one gradient per parameter in ``dtype``.
    * ``optimizer_gb``: fp32 state - Muon parameters (matrices except embeddings / lm_head /
      router, SPEC §7) keep 1 momentum buffer, AdamW parameters keep 2 (exp_avg, exp_avg_sq); plus
      an fp32 master copy of all weights when ``dtype`` is not fp32 (``master_weights_gb``,
      included in ``optimizer_gb``). If ``train.optimizer.name == "adamw"`` every parameter uses
      AdamW. With bf16 + AdamW this is the classic 16 bytes/parameter.
    * ``ema_gb``: an fp32 copy of the weights when ``train.ema > 0`` (full Config only).
    * ``total_gb`` = weights + grads + optimizer + ema. Activations, temporary buffers, the
      CUDA context and fragmentation are NOT included; FSDP full sharding divides the total by
      the world size.
    * ``inference_gb``: weights only; ``cache_gb``: inference cache for ONE sequence of
      ``model.max_seq_len`` tokens (attention K/V in ``dtype`` - local layers keep only the window -,
      fp32 retention states ``Hr x hd x hd`` per layer, token-shift buffers).
    """
    m, mm, full = _resolve(cfg)
    nbytes = _dtype_bytes(dtype)
    entries = _entries(m, mm)
    total = sum(_numel(e.shape) for e in entries)
    all_adamw = full is not None and full.train.optimizer.name == "adamw"
    muon = 0 if all_adamw else sum(_numel(e.shape) for e in entries if _uses_muon(e))
    adamw = total - muon
    master = total * 4 if nbytes != 4 else 0
    weights = total * nbytes
    grads = total * nbytes
    optimizer = muon * 4 + adamw * 8 + master
    ema = total * 4 if (full is not None and full.train.ema > 0) else 0

    Ha, Hr, Hkv, hd, d, k = (m.n_attention_heads, m.n_retention_heads, m.n_attn_kv_heads, m.head_dim,
                             m.d_model, m.token_shift)
    cache = 0
    for i in range(m.n_layers):
        if Ha > 0:
            window = m.layer_window(i)
            length = m.max_seq_len if window is None else min(m.max_seq_len, window)
            cache += 2 * Hkv * hd * length * nbytes
        if Hr > 0:
            cache += Hr * hd * hd * 4
        if k > 1:
            cache += d * (k - 1) * nbytes

    return {
        "params": total,
        "dtype": str(dtype),
        "muon_params": muon,
        "adamw_params": adamw,
        "weights_gb": weights / _GiB,
        "grads_gb": grads / _GiB,
        "optimizer_gb": optimizer / _GiB,
        "master_weights_gb": master / _GiB,
        "ema_gb": ema / _GiB,
        "total_gb": (weights + grads + optimizer + ema) / _GiB,
        "inference_gb": weights / _GiB,
        "cache_gb": cache / _GiB,
    }
