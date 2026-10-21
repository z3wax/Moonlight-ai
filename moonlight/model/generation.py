"""Sampling / best-of-N generation with the KV + retention cache.

``generate(model, input_ids, ...) -> GenerationResult``

1. The prompt (one sequence) is prefilled once with ``use_cache=True`` (optionally with
   ``image_features`` for its <|image|> slots); for ``n > 1`` the cache is then repeated so the
   ``n`` samples decode as one batch.
2. Each step processes the float32 logits of the last position:
   banned ids -> -inf; repetition penalty (CTRL-style: positive logits divided, negative logits
   multiplied by the penalty) on tokens already *generated* in that row; no-repeat-n-gram ban
   computed on the generated tokens; then greedy (``temperature <= 0``) or temperature + top-k +
   top-p sampling.
3. A row finishes when it emits a token from ``stop_ids``; the stop token is *not* included in the
   returned tokens (``finished`` tells whether it was produced).
4. Score of a sample = mean log-probability of its sampled tokens (including the stop token) under
   the *untempered, unpenalised* model distribution, minus ``UNFINISHED_PENALTY`` if it did not
   finish and ``SHORT_PENALTY`` if it has fewer than ``MIN_TOKENS`` tokens. The best-scoring sample
   is returned in ``tokens``.

``stream(token_id)`` is called for every non-stop token as soon as it is sampled (only when n == 1).
The model's train/eval mode is restored afterwards (generation always runs in eval mode).
Works on CPU in float32 and on CUDA under the caller's ``torch.autocast``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Set, Union

import torch

__all__ = ["GenerationResult", "generate", "sample_next", "PAD_ID", "EOS_ID", "END_ID"]

# Fixed special-token ids (spec section 5; identical to moonlight.tokenizer constants).
PAD_ID = 0
EOS_ID = 2
END_ID = 6

UNFINISHED_PENALTY = 1.0
SHORT_PENALTY = 2.0
MIN_TOKENS = 2
INVALID_LOGPROB = -1e4  # score contribution of a token the model assigned no finite probability


@dataclass
class GenerationResult:
    """tokens: best sample (without stop token); all_tokens/scores/finished: per sample (n)."""

    tokens: List[int] = field(default_factory=list)
    all_tokens: List[List[int]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    finished: List[bool] = field(default_factory=list)


def _model_device(model: torch.nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def _as_prompt(input_ids: Union[Sequence[int], torch.Tensor]) -> List[int]:
    if isinstance(input_ids, torch.Tensor):
        t = input_ids.detach()
        if t.dim() == 2:
            if t.shape[0] != 1:
                raise ValueError("generate() takes a single prompt (got a batch of %d)" % t.shape[0])
            t = t[0]
        if t.dim() != 1:
            raise ValueError(f"input_ids must be 1-D (or (1, T)), got shape {tuple(t.shape)}")
        return [int(i) for i in t.tolist()]
    return [int(i) for i in input_ids]


def _make_generator(device: torch.device, seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    try:
        g = torch.Generator(device=device)
    except (RuntimeError, TypeError):  # device without generator support -> CPU sampling
        g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def sample_next(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Pick one token per row from processed logits (n, V) (float32). Returns (n,) int64.

    ``temperature <= 0`` -> argmax. Rows must contain at least one finite logit.
    """
    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1)
    x = logits / float(temperature)
    V = x.shape[-1]
    if top_k is not None and 0 < int(top_k) < V:
        kth = torch.topk(x, int(top_k), dim=-1).values[..., -1:]
        x = x.masked_fill(x < kth, float("-inf"))
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_x, sorted_idx = torch.sort(x, dim=-1, descending=True)
        probs = torch.softmax(sorted_x, dim=-1)
        cum = probs.cumsum(dim=-1)
        remove = (cum - probs) > float(top_p)  # keeps the smallest prefix reaching top_p (>= 1 token)
        sorted_x = sorted_x.masked_fill(remove, float("-inf"))
        x = torch.full_like(x, float("-inf")).scatter(-1, sorted_idx, sorted_x)
    probs = torch.softmax(x, dim=-1)
    if generator is not None and generator.device != probs.device:
        return torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(-1).to(probs.device)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


def _ngram_bans(tokens: List[int], n: int) -> Set[int]:
    """Tokens that would complete an n-gram already present in ``tokens``."""
    if n <= 0 or len(tokens) < n - 1:
        return set()
    if n == 1:
        return set(tokens)
    prefix = tuple(tokens[len(tokens) - (n - 1):])
    bans = set()
    for i in range(len(tokens) - n + 1):
        if tuple(tokens[i:i + n - 1]) == prefix:
            bans.add(tokens[i + n - 1])
    return bans


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: Union[List[int], torch.Tensor],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.1,
    no_repeat_ngram: int = 0,
    stop_ids: Iterable[int] = (END_ID, EOS_ID),
    n: int = 1,
    image_features: Optional[torch.Tensor] = None,
    banned_ids: Iterable[int] = (),
    stream: Optional[Callable[[int], None]] = None,
    seed: Optional[int] = None,
) -> GenerationResult:
    """Generate up to ``max_new_tokens`` tokens after the prompt (see module docstring).

    ``input_ids``: list of ints or a (T,) / (1, T) tensor. ``image_features``:
    (B_img, n_image_tokens, vision_dim) for the prompt's <|image|> tokens. Returns a
    :class:`GenerationResult`.
    """
    prompt = _as_prompt(input_ids)
    if not prompt:
        raise ValueError("generate() needs a non-empty prompt")
    n = max(1, int(n))
    max_new_tokens = max(0, int(max_new_tokens))
    stops = {int(s) for s in (stop_ids or ())}
    if max_new_tokens == 0:
        return GenerationResult([], [[] for _ in range(n)], [0.0] * n, [False] * n)

    device = _model_device(model)
    was_training = model.training
    model.eval()
    try:
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        feats = None
        if image_features is not None:  # tensor or array-like (e.g. cached fp16 numpy features)
            feats = torch.as_tensor(image_features).to(device)
        out = model(ids, image_features=feats, use_cache=True)
        cache = out.cache
        logits = out.logits[:, -1].float()  # (1, V)
        V = logits.shape[-1]
        if n > 1:
            cache.repeat(n)
            logits = logits.repeat(n, 1)

        banned = sorted({int(b) for b in (banned_ids or ()) if 0 <= int(b) < V})
        banned_t = torch.tensor(banned, dtype=torch.long, device=device) if banned else None
        rp = float(repetition_penalty) if repetition_penalty else 1.0
        presence = torch.zeros(n, V, dtype=torch.bool, device=device) if rp != 1.0 else None
        nr = int(no_repeat_ngram or 0)
        generator = _make_generator(device, seed)
        in_vocab_stops = [s for s in stops if 0 <= s < V]
        fallback_id = min(in_vocab_stops) if in_vocab_stops else 0  # used if a row has no allowed token
        rows = torch.arange(n, device=device)

        gen: List[List[int]] = [[] for _ in range(n)]
        logp_sum = [0.0] * n
        n_scored = [0] * n
        finished = [False] * n

        for step in range(max_new_tokens):
            logits = torch.nan_to_num(logits, nan=float("-inf"), neginf=float("-inf"))
            logprobs = torch.log_softmax(logits, dim=-1)  # untempered, for scoring
            proc = logits.clone()
            if banned_t is not None:
                proc[:, banned_t] = float("-inf")
            if presence is not None:
                penalized = torch.where(proc > 0, proc / rp, proc * rp)
                proc = torch.where(presence, penalized, proc)
            if nr > 0:
                for r in range(n):
                    if not finished[r]:
                        bans = _ngram_bans(gen[r], nr)
                        if bans:
                            proc[r, sorted(bans)] = float("-inf")
            dead = ~torch.isfinite(proc).any(dim=-1)
            if bool(dead.any()):
                proc[dead, fallback_id] = 0.0
            nxt = sample_next(proc, temperature, top_k, top_p, generator)  # (n,)
            nxt_list = nxt.tolist()
            lp_list = logprobs.gather(1, nxt[:, None]).squeeze(1).tolist()
            if presence is not None:
                presence[rows, nxt] = True
            for r in range(n):
                if finished[r]:
                    continue
                tok = int(nxt_list[r])
                lp = lp_list[r]
                logp_sum[r] += lp if math.isfinite(lp) else INVALID_LOGPROB
                n_scored[r] += 1
                if tok in stops:
                    finished[r] = True
                    continue
                gen[r].append(tok)
                if stream is not None and n == 1:
                    stream(tok)
            if all(finished) or step == max_new_tokens - 1:
                break
            out = model(nxt[:, None], cache=cache, use_cache=True)
            logits = out.logits[:, -1].float()
    finally:
        model.train(was_training)

    scores = []
    for r in range(n):
        s = logp_sum[r] / max(1, n_scored[r])
        if not finished[r]:
            s -= UNFINISHED_PENALTY
        if len(gen[r]) < MIN_TOKENS:
            s -= SHORT_PENALTY
        scores.append(float(s))
    best = max(range(n), key=lambda r: scores[r])
    return GenerationResult(tokens=list(gen[best]), all_tokens=gen, scores=scores, finished=finished)
