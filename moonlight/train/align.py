"""Preference (DPO / IPO) and RL (GRPO) helpers for the ``dpo`` / ``grpo`` phases (SPEC §13.5).

Conventions: ``x`` (B, T) int64 inputs, ``y`` (B, T) int64 targets already shifted
(``y[:, t]`` is the token after ``x[:, t]``), ``-100`` = not a response token. Log-probabilities
are computed in float32 from the model's float32 logits.

* :func:`token_logprobs` / :func:`sequence_logprobs` - per-token / summed response log-probs.
* :func:`dpo_loss` - ``-log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))`` (``loss="dpo"``) or
  IPO ``((pi_c - ref_c) - (pi_r - ref_r) - 1 / (2 beta))^2`` (``loss="ipo"``), with summed
  sequence log-probs for both.
* :func:`group_advantages`, :func:`grpo_loss`, :func:`grpo_step` - GRPO: G samples per prompt,
  verifier rewards, group-normalised advantages ``(r - mean) / (std + 1e-4)``, zero-variance
  groups skipped, clipped policy-gradient objective + ``kl_coef * KL_k3(pi || pi_ref)``.
"""

from __future__ import annotations

import contextlib
import copy
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "IGNORE_INDEX",
    "token_logprobs",
    "sequence_logprobs",
    "dpo_loss",
    "concat_pairs",
    "make_reference_model",
    "compute_reward",
    "group_advantages",
    "grpo_loss",
    "build_completion_batch",
    "grpo_step",
]

IGNORE_INDEX = -100


# ------------------------------------------------------------------------------------------------
# log-probs
# ------------------------------------------------------------------------------------------------


def _logits(model: Any, x: torch.Tensor, image_features: Optional[torch.Tensor] = None) -> torch.Tensor:
    out = model(x, image_features=image_features)
    logits = out.logits if hasattr(out, "logits") else out
    return logits.float()


def token_logprobs(model: Any, x: torch.Tensor, y: torch.Tensor,
                   image_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probs of the targets.

    Args:
        model: causal LM returning ``ModelOutput`` (logits (B, T, V) for every position).
        x, y: (B, T) int64; ``y == -100`` marks positions that are not scored.
    Returns:
        ``(logp, mask)``: float32 (B, T) log-probabilities (0 where masked) and float32 mask.
    """
    logits = _logits(model, x, image_features)  # (B, T, V) float32
    mask = (y != IGNORE_INDEX)
    tgt = y.clamp(min=0).unsqueeze(-1)
    # log p = z_y - logsumexp(z)  (avoids materialising a second (B, T, V) log_softmax tensor)
    logp = logits.gather(-1, tgt).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    maskf = mask.to(logp.dtype)
    return logp * maskf, maskf


def sequence_logprobs(model: Any, x: torch.Tensor, y: torch.Tensor,
                      image_features: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Summed response log-prob per sequence: (B,) float32 (sum over ``y != -100``)."""
    logp, _ = token_logprobs(model, x, y, image_features)
    return logp.sum(dim=-1)


def concat_pairs(c_x: torch.Tensor, c_y: torch.Tensor, r_x: torch.Tensor, r_y: torch.Tensor,
                 pad_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack chosen (B, Tc) and rejected (B, Tr) batches into one (2B, max(Tc, Tr)) batch.

    Right-pads with ``pad_id`` / ``-100``; rows ``[:B]`` are chosen, ``[B:]`` rejected. One forward
    over both halves keeps DDP/FSDP to a single forward per backward.
    """
    T = max(c_x.shape[1], r_x.shape[1])

    def pad(t: torch.Tensor, value: int) -> torch.Tensor:
        return F.pad(t, (0, T - t.shape[1]), value=value) if t.shape[1] < T else t

    x = torch.cat([pad(c_x, pad_id), pad(r_x, pad_id)], dim=0)
    y = torch.cat([pad(c_y, IGNORE_INDEX), pad(r_y, IGNORE_INDEX)], dim=0)
    return x, y


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor, ref_chosen: torch.Tensor,
             ref_rejected: torch.Tensor, beta: float = 0.1, loss_type: str = "dpo"
             ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """DPO / IPO loss from (B,) summed sequence log-probs.

    Returns ``(loss, metrics)`` where metrics hold ``reward_acc`` (fraction with chosen reward >
    rejected reward), ``margin`` (mean reward margin), ``reward_chosen`` / ``reward_rejected``
    (mean ``beta * (pi - ref)``).
    """
    beta = float(beta)
    if beta <= 0:
        raise ValueError(f"DPO beta must be > 0, got {beta}")
    chosen_logratio = policy_chosen.float() - ref_chosen.float()
    rejected_logratio = policy_rejected.float() - ref_rejected.float()
    logits = chosen_logratio - rejected_logratio
    kind = (loss_type or "dpo").lower()
    if kind == "dpo":
        losses = -F.logsigmoid(beta * logits)
    elif kind == "ipo":
        losses = (logits - 1.0 / (2.0 * beta)) ** 2
    else:
        raise ValueError(f"unknown preference loss {loss_type!r}; expected dpo or ipo")
    with torch.no_grad():
        rc = beta * chosen_logratio
        rr = beta * rejected_logratio
        metrics = {
            "reward_acc": float((rc > rr).float().mean().item()),
            "margin": float((rc - rr).mean().item()),
            "reward_chosen": float(rc.mean().item()),
            "reward_rejected": float(rr.mean().item()),
        }
    return losses.mean(), metrics


def make_reference_model(model: torch.nn.Module) -> torch.nn.Module:
    """Frozen deep copy (eval mode, ``requires_grad=False``) used as the DPO/GRPO reference.

    Pass the UNWRAPPED module for DDP / single-process runs. (Under FSDP2 the deep copy keeps the
    sharded DTensor parameters, so the reference forward is collective as well.)
    """
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


# ------------------------------------------------------------------------------------------------
# rewards
# ------------------------------------------------------------------------------------------------

_THINK_END = "<|/think|>"
_STRIP_TOKENS = ("<|end|>", "<|eos|>", "<|pad|>", "<|bos|>")
_NUMBER_TYPES = ("exact_number", "number", "numeric", "math", "arithmetic", "gsm8k")
_CONSTRAINT_TYPES = ("constraints", "constraint", "detail", "details", "ifeval", "format")
_CODE_TYPES = ("code", "code_lite", "python", "unit_tests")


def _final_part(text: str) -> str:
    """Text after the last ``<|/think|>`` (the visible answer)."""
    return text.rsplit(_THINK_END, 1)[-1].strip() if _THINK_END in text else text.strip()


def compute_reward(response: str, reward_type: str, reward_data: Any) -> float:
    """Score one completion with ``moonlight.data.verifiers`` (rewards in [0, 1]).

    ``response`` is the decoded completion (special tokens kept, so ``<|think|>...<|/think|>`` is
    visible to ``extract_final_answer``). Reward types (from ``grpo_prompt_set`` records):

    * ``exact_number`` (aliases number/math/arithmetic/gsm8k) -> ``reward_exact_number(response,
      target)`` where ``reward_data`` is the target or ``{"target"|"answer": ...}``;
    * ``constraints`` (aliases detail/ifeval/format) -> ``reward_constraints(answer, constraints)``
      with ``reward_data`` a constraint list or ``{"constraints": [...]}``;
    * ``code`` (alias code_lite) -> ``reward_code(answer, tests, entry_point)`` with
      ``reward_data = {"tests": ..., "entry_point": ...}``;
    * ``exact`` -> case/whitespace-insensitive string match of the final answer.

    ``answer`` = text after ``<|/think|>``. Unknown types raise ``ValueError``.
    """
    from moonlight.data import verifiers

    text = response or ""
    for tok in _STRIP_TOKENS:
        text = text.replace(tok, "")
    kind = str(reward_type or "").lower()
    data = reward_data
    if kind in _NUMBER_TYPES:
        target = data.get("target", data.get("answer")) if isinstance(data, Mapping) else data
        return float(verifiers.reward_exact_number(text, target))
    if kind in _CONSTRAINT_TYPES:
        cons = data.get("constraints", []) if isinstance(data, Mapping) else data
        return float(verifiers.reward_constraints(_final_part(text), list(cons or [])))
    if kind in _CODE_TYPES:
        if not isinstance(data, Mapping):
            raise ValueError("code reward_data must be a mapping with 'tests' (and 'entry_point')")
        return float(verifiers.reward_code(text, data.get("tests"), data.get("entry_point"),
                                           float(data.get("timeout", 5.0))))
    if kind in ("exact", "exact_match", "string"):
        target = data.get("target", data.get("answer")) if isinstance(data, Mapping) else data
        got = verifiers.extract_final_answer(text) if hasattr(verifiers, "extract_final_answer") else _final_part(text)
        return 1.0 if " ".join(str(got).split()).lower() == " ".join(str(target).split()).lower() else 0.0
    raise ValueError(f"unknown GRPO reward_type {reward_type!r}")


# ------------------------------------------------------------------------------------------------
# GRPO
# ------------------------------------------------------------------------------------------------


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group-normalised advantages.

    Args:
        rewards: (P, G) float rewards (P prompts x G samples).
    Returns:
        ``(adv (P, G), keep (P,) bool)``: ``adv = (r - mean_g) / (std_g + eps)`` (population std),
        ``keep`` False for zero-variance groups (their advantages are 0 and they are skipped).
    """
    r = rewards.float()
    if r.dim() == 1:
        r = r.unsqueeze(0)
    mean = r.mean(dim=-1, keepdim=True)
    std = r.std(dim=-1, unbiased=False, keepdim=True)
    keep = (std.squeeze(-1) > 1e-8)
    adv = (r - mean) / (std + eps)
    adv = torch.where(keep.unsqueeze(-1), adv, torch.zeros_like(adv))
    return adv, keep


def grpo_loss(logp: torch.Tensor, old_logp: torch.Tensor, ref_logp: Optional[torch.Tensor],
              adv: torch.Tensor, mask: torch.Tensor, clip: float = 0.2, kl_coef: float = 0.02
              ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Clipped GRPO objective over completion tokens.

    Args:
        logp: (N, T) policy log-probs WITH grad; old_logp: (N, T) detached (sampling policy);
        ref_logp: (N, T) reference log-probs (no grad) or None (no KL term);
        adv: (N,) advantages; mask: (N, T) float, 1 on completion tokens.
    Returns:
        ``(loss, metrics)``; per sequence the token mean of
        ``-min(ratio * A, clip(ratio, 1 - c, 1 + c) * A) + kl_coef * KL_k3`` then the mean over
        sequences (KL_k3 = ``exp(ref - logp) - (ref - logp) - 1``).
    """
    ratio = torch.exp(logp - old_logp)
    a = adv.float().unsqueeze(-1)
    unclipped = ratio * a
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * a
    per_tok = -torch.minimum(unclipped, clipped)
    kl_mean = 0.0
    if ref_logp is not None and kl_coef > 0:
        d = (ref_logp - logp).clamp(-20.0, 20.0)
        kl = torch.exp(d) - d - 1.0
        per_tok = per_tok + kl_coef * kl
        with torch.no_grad():
            kl_mean = float(((kl * mask).sum() / mask.sum().clamp(min=1.0)).item())
    n_tok = mask.sum(dim=-1).clamp(min=1.0)
    per_seq = (per_tok * mask).sum(dim=-1) / n_tok
    loss = per_seq.mean()
    with torch.no_grad():
        clip_frac = (((ratio - 1.0).abs() > clip).float() * mask).sum() / mask.sum().clamp(min=1.0)
    return loss, {"kl": kl_mean, "clip_frac": float(clip_frac.item())}


def build_completion_batch(prompt_ids: Sequence[int], completions: Sequence[Sequence[int]],
                           finished: Sequence[bool], end_id: int, pad_id: int = 0,
                           device: Any = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """``(x, y)`` (N, T) for scoring completions of one prompt.

    Each row is ``prompt + completion (+ end_id if finished)``; ``x = seq[:-1]``,
    ``y = seq[1:]`` with prompt positions set to -100, right-padded with ``pad_id`` / -100.
    """
    P = len(prompt_ids)
    seqs = []
    for comp, fin in zip(completions, finished):
        seq = list(prompt_ids) + list(comp) + ([int(end_id)] if fin else [])
        seqs.append(seq)
    T = max(len(s) for s in seqs) - 1
    T = max(T, 1)
    x = torch.full((len(seqs), T), int(pad_id), dtype=torch.long)
    y = torch.full((len(seqs), T), IGNORE_INDEX, dtype=torch.long)
    for i, s in enumerate(seqs):
        n = len(s) - 1
        if n <= 0:
            continue
        x[i, :n] = torch.tensor(s[:-1], dtype=torch.long)
        tgt = torch.tensor(s[1:], dtype=torch.long)
        tgt[: max(0, P - 1)] = IGNORE_INDEX
        y[i, :n] = tgt
    return x.to(device), y.to(device)


def _truncate_at_stop(tokens: Sequence[int], stops: Sequence[int]) -> Tuple[List[int], bool]:
    out: List[int] = []
    for t in tokens:
        if t in stops:
            return out, True
        out.append(int(t))
    return out, False


def grpo_step(policy: Any, sampler: Any, ref_model: Optional[Any], tokenizer: Any,
              prompts: Sequence[Mapping[str, Any]], *, group_size: int = 8, max_new_tokens: int = 256,
              temperature: float = 1.0, kl_coef: float = 0.02, clip: float = 0.2, seed: Optional[int] = None,
              max_seq_len: Optional[int] = None, fixed_length: bool = False,
              sync_context: Optional[Callable[[bool], Any]] = None,
              backward: Optional[Callable[[torch.Tensor], None]] = None,
              autocast: Optional[Callable[[], Any]] = None) -> Dict[str, float]:
    """One GRPO update's forward/backward work (the caller steps the optimizer).

    For each prompt record ``{prompt_messages, reward_type, reward_data}``:

    1. sample ``group_size`` completions with :func:`moonlight.model.generation.generate`
       (no grad, eval mode; ``temperature``, no top-k/top-p/repetition shaping);
    2. rewards via :func:`compute_reward`; advantages via :func:`group_advantages`;
    3. recompute token log-probs WITH grad on ``policy`` (train mode), ``old_logp`` = detached
       copy (ratio starts at 1), reference log-probs on ``ref_model`` (no grad) when
       ``kl_coef > 0``; loss = :func:`grpo_loss` / n_prompts; ``backward(loss)``.

    Zero-variance groups contribute ``0 * loss`` (still a forward/backward, so every rank runs the
    same number of collectives under DDP/FSDP). ``fixed_length=True`` (FSDP) samples exactly
    ``max_new_tokens`` tokens without early stopping (identical forward counts on all ranks) and
    truncates at the first stop token afterwards.

    Args:
        policy: the training model (wrapped: DDP/FSDP/compiled) used for the grad pass.
        sampler: model used for generation (unwrapped module for DDP/single process).
        sync_context: ``f(is_last) -> context manager`` (gradient-sync control per prompt).
        backward: ``f(loss)`` (e.g. GradScaler-aware); default ``loss.backward()``.
        autocast: ``f() -> context manager`` for forward passes.
    Returns:
        metrics: ``reward`` (mean), ``solve_rate`` (fraction with reward >= 1), ``loss``,
        ``kl``, ``clip_frac``, ``skipped`` (zero-variance groups), ``tokens`` (completion tokens),
        ``completion_len``.
    """
    from moonlight.model.generation import generate
    from moonlight.tokenizer import END_ID, EOS_ID, PAD_ID

    G = max(1, int(group_size))
    stops = (END_ID, EOS_ID)
    ac = autocast or contextlib.nullcontext
    sc = sync_context or (lambda last: contextlib.nullcontext())
    bw = backward or (lambda loss: loss.backward())
    device = next(policy.parameters()).device
    n_prompts = len(prompts)
    rewards_all: List[float] = []
    losses: List[float] = []
    kls: List[float] = []
    clips: List[float] = []
    skipped = 0
    n_tokens = 0
    for pi, rec in enumerate(prompts):
        msgs = list(rec["prompt_messages"])
        prompt_ids, _ = tokenizer.encode_chat(msgs, add_generation_prompt=True)
        if max_seq_len:
            budget = max(8, int(max_seq_len) - int(max_new_tokens) - 1)
            if len(prompt_ids) > budget:  # keep the end of the prompt (the generation prompt)
                prompt_ids = prompt_ids[-budget:]
        res = generate(sampler, prompt_ids, max_new_tokens=int(max_new_tokens), temperature=float(temperature),
                       top_k=0, top_p=1.0, repetition_penalty=1.0, no_repeat_ngram=0,
                       stop_ids=() if fixed_length else stops, n=G,
                       seed=None if seed is None else int(seed) * 1000 + pi)
        comps: List[List[int]] = []
        fins: List[bool] = []
        for r, toks in enumerate(res.all_tokens):
            if fixed_length:
                c, f = _truncate_at_stop(toks, stops)
            else:
                c, f = list(toks), bool(res.finished[r])
            comps.append(c)
            fins.append(f)
        texts = [tokenizer.decode(c, skip_special=False) for c in comps]
        rewards = [compute_reward(t, rec.get("reward_type"), rec.get("reward_data")) for t in texts]
        rewards_all.extend(rewards)
        adv, keep = group_advantages(torch.tensor([rewards], dtype=torch.float32))
        x, y = build_completion_batch(prompt_ids, comps, fins, END_ID, PAD_ID, device)
        n_tokens += int((y != IGNORE_INDEX).sum().item())
        is_last = pi == n_prompts - 1
        with sc(is_last):
            with ac():
                logp, mask = token_logprobs(policy, x, y)
            ref_logp = None
            if ref_model is not None and kl_coef > 0:
                with torch.no_grad(), ac():
                    ref_logp, _ = token_logprobs(ref_model, x, y)
            loss, m = grpo_loss(logp, logp.detach(), ref_logp, adv[0].to(device), mask, clip, kl_coef)
            if not bool(keep[0]):
                skipped += 1
                loss = loss * 0.0
            bw(loss / max(1, n_prompts))
        losses.append(float(loss.detach().item()))
        kls.append(m["kl"])
        clips.append(m["clip_frac"])
    n = max(1, len(rewards_all))
    return {
        "reward": sum(rewards_all) / n,
        "solve_rate": sum(1.0 for r in rewards_all if r >= 1.0 - 1e-6) / n,
        "loss": sum(losses) / max(1, len(losses)),
        "kl": sum(kls) / max(1, len(kls)),
        "clip_frac": sum(clips) / max(1, len(clips)),
        "skipped": float(skipped),
        "tokens": float(n_tokens),
        "completion_len": n_tokens / max(1, n_prompts * G),
    }
