"""``moonlight`` command line (SPEC §8).

```
moonlight list [--tier T] [--names]
moonlight info NAME [--yaml]
moonlight params NAME [--dtype bf16]
moonlight train NAME [--set k=v ...] [--offline] [--output PATH] [--device D]
moonlight chat PATH_OR_NAME [--tools] [-m TEXT] [--image IMG ...] [--system S]
moonlight bench PATH_OR_NAME --suite cc-001 [--limit N] [--out report.md] [--offline]
moonlight plan PATH_OR_NAME "task"
moonlight build PATH_OR_NAME "task" --workdir DIR [--max-steps 24]
moonlight gen-configs [--check]
moonlight version
```

``PATH_OR_NAME`` is an exported ``.safetensors`` file, a run directory, or a config name / YAML
path (then ``{train.output_dir}/model.safetensors`` is used when it exists, otherwise an untrained
model is built with a warning). ``list`` / ``info`` / ``params`` never import torch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

__all__ = ["main", "build_parser", "format_table"]


# -------------------------------------------------------------------------------------------------
# formatting helpers
# -------------------------------------------------------------------------------------------------

def format_table(rows: Sequence[Sequence[Any]], headers: Sequence[str], align: Optional[str] = None) -> str:
    """Plain-text table. ``align``: one char per column, ``l`` or ``r`` (default: left)."""
    cells = [[str(h) for h in headers]] + [["" if c is None else str(c) for c in r] for r in rows]
    ncol = len(headers)
    widths = [max(len(r[i]) for r in cells if i < len(r)) for i in range(ncol)]
    al = (align or "l" * ncol).ljust(ncol, "l")

    def fmt(r: Sequence[str]) -> str:
        parts = [(r[i].rjust(widths[i]) if al[i] == "r" else r[i].ljust(widths[i])) for i in range(ncol)]
        return "  ".join(parts).rstrip()

    sep = "  ".join("-" * w for w in widths)
    return "\n".join([fmt(cells[0]), sep] + [fmt(r) for r in cells[1:]])


def _size(n: Any) -> str:
    from moonlight.config import format_size

    try:
        return format_size(int(n), 2) if n is not None else "-"
    except (TypeError, ValueError):
        return str(n)


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def _print_flush(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


# -------------------------------------------------------------------------------------------------
# commands: configs
# -------------------------------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> int:
    from moonlight.config import Config
    from moonlight.params import estimate_params
    from moonlight.registry import config_root, find_config, list_configs

    names = list_configs(args.tier)
    if not names:
        where = f" (tier {args.tier})" if args.tier else ""
        print(f"no configs found under {config_root()}{where}; run `moonlight gen-configs`")
        return 0
    if args.names:
        print("\n".join(names))
        return 0
    rows = []
    for name in names:
        try:
            cfg = Config.from_yaml(find_config(name))
            rep = estimate_params(cfg)
            active = _size(rep.active) if rep.active != rep.total else ""
            rows.append([cfg.name, cfg.tier, cfg.kind, cfg.nominal_params, _size(rep.total), active,
                         cfg.benchmark, "yes" if cfg.multimodal.enabled else ""])
        except Exception as e:  # keep listing even if one file is broken
            rows.append([name, "?", "?", "?", "error", "", "", f"{type(e).__name__}: {e}"[:60]])
    print(format_table(rows, ["NAME", "TIER", "KIND", "NOMINAL", "PARAMS", "ACTIVE", "BENCH", "VISION"],
                       "llllrrll"))
    print(f"\n{len(names)} configs")
    return 0


def _phase_text(phase: Any) -> str:
    keys = [k for k in phase if k != "name"]
    inner = ", ".join(f"{k}={phase[k]}" for k in keys)
    return f"{phase.get('name', '?')}({inner})"


def cmd_info(args: argparse.Namespace) -> int:
    from moonlight.params import estimate_params, flops_per_token, memory_estimate
    from moonlight.registry import load

    cfg = load(args.name, overrides=args.set or None)
    if args.yaml:
        print(cfg.to_yaml(), end="")
        return 0
    m = cfg.model
    rep = estimate_params(cfg)
    mem = memory_estimate(cfg)
    n_ret = m.n_retention_heads
    n_attn = m.n_attention_heads
    lines = [
        f"{cfg.name}  —  {cfg.description}".rstrip(" —"),
        "",
        f"  tier / kind        {cfg.tier} / {cfg.kind}",
        f"  benchmark          {cfg.benchmark}",
        f"  params             {_size(rep.total)} total, {_size(rep.active)} active "
        f"(nominal {cfg.nominal_params})",
        f"  shape              d_model {m.d_model}, layers {m.n_layers}, heads {m.n_heads} x {m.head_dim} "
        f"({n_attn} attention + {n_ret} retention, kv {m.n_attn_kv_heads})",
        "  ffn                " + (f"MoE {m.moe.n_experts} experts top-{m.moe.top_k}, shared {m.moe.n_shared}, "
                                    f"expert_hidden {m.moe.expert_hidden}, first_dense {m.moe.first_dense_layers}"
                                    if m.moe.enabled else f"dense SwiGLU {m.ffn_hidden}"),
        f"  attention          window {m.attn_window or 'full'}, global_every {m.global_every}, "
        f"rope_theta {m.rope_theta:g}",
        f"  context            {m.max_seq_len} tokens (train seq_len {cfg.train.seq_len})",
        f"  vocab              {m.vocab_size} (tokenizer path: {cfg.tokenizer.path or 'trained on first run'})",
        f"  mtp                depth {m.mtp_depth}" + (" (block)" if m.mtp_block else ""),
        "  multimodal         " + (f"{cfg.multimodal.vision_encoder} -> {cfg.multimodal.n_image_tokens} tokens"
                                    if cfg.multimodal.enabled else "off"),
        f"  training FLOPs     {flops_per_token(cfg):.3g} per token",
        f"  memory (bf16)      weights {mem['weights_gb']:.2f} GiB, training state ~{mem['total_gb']:.2f} GiB",
        f"  hardware           {cfg.train.hardware} (distributed: {cfg.train.distributed.strategy})",
        f"  output_dir         {cfg.train.output_dir}",
        f"  data               offline={cfg.data.offline}, languages={','.join(cfg.data.languages)}",
    ]
    if cfg.data.pretrain:
        lines.append("  pretrain sources   " + ", ".join(str(e.get("source")) for e in cfg.data.pretrain))
    if cfg.data.sft:
        lines.append("  sft sources        " + ", ".join(str(e.get("source")) for e in cfg.data.sft))
    lines.append("  phases")
    for i, ph in enumerate(cfg.train.phases):
        lines.append(f"    {i + 1}. {_phase_text(ph)}")
    print("\n".join(lines))
    return 0


def cmd_params(args: argparse.Namespace) -> int:
    from moonlight.params import estimate_params, flops_per_token, memory_estimate
    from moonlight.registry import load

    cfg = load(args.name, overrides=args.set or None)
    rep = estimate_params(cfg)
    print(rep.pretty())
    mem = memory_estimate(cfg, dtype=args.dtype)
    print(f"\nMemory per replica ({mem['dtype']}, GiB)")
    rows = [[k[:-3].replace("_", " "), f"{v:.3f}"] for k, v in mem.items() if k.endswith("_gb")]
    print(format_table(rows, ["ITEM", "GiB"], "lr"))
    print(f"\nTraining FLOPs per token: {flops_per_token(cfg):.4g}")
    return 0


def cmd_gen_configs(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_configs.py"
    if not script.is_file():
        _err(f"{script} not found (gen-configs needs a source checkout of moonlight-llm)")
        return 2
    extra = [a for a in (args.args or []) if a != "--"]
    return subprocess.call([sys.executable, str(script)] + extra, cwd=str(script.parents[1]))


# -------------------------------------------------------------------------------------------------
# commands: models
# -------------------------------------------------------------------------------------------------

def _overrides(args: argparse.Namespace) -> List[str]:
    items = list(getattr(args, "set", None) or [])
    if getattr(args, "offline", False):
        items.append("data.offline=true")
    return items


def _load_llm(spec: str, args: argparse.Namespace) -> Any:
    """Exported file / run dir / config name -> LLM (see module docstring)."""
    from moonlight.llm import LLM

    device = getattr(args, "device", None)
    p = Path(spec).expanduser()
    if (p.is_file() and p.suffix.lower() == ".safetensors") or p.is_dir():
        llm = LLM.from_pretrained(p, device=device)
        if _overrides(args):
            llm.config.apply_overrides([o for o in _overrides(args) if not o.startswith(("model.", "multimodal."))])
        return llm
    from moonlight.registry import load

    cfg = load(spec, overrides=_overrides(args) or None)
    trained = Path(cfg.train.output_dir).expanduser() / "model.safetensors"
    if trained.is_file():
        print(f"[moonlight] loading trained model {trained}", file=sys.stderr)
        llm = LLM.from_pretrained(trained, device=device)
        llm.config.data.offline = cfg.data.offline
        return llm
    print(f"[moonlight] warning: no trained model at {trained}; using RANDOM weights of {cfg.name} "
          f"(run `moonlight train {cfg.name}` first)", file=sys.stderr)
    return LLM(cfg, device=device)


def _gen_kwargs(args: argparse.Namespace) -> dict:
    out = {}
    for key in ("max_new_tokens", "temperature", "top_k", "top_p", "best_of", "seed"):
        v = getattr(args, key, None)
        if v is not None:
            out[key] = v
    if getattr(args, "show_thinking", False):
        out["show_thinking"] = True
    return out


def cmd_train(args: argparse.Namespace) -> int:
    from moonlight.llm import train_config

    llm = train_config(args.name, overrides=_overrides(args) or None, output=args.output, device=args.device)
    res = llm.train_result or {}
    rows = []
    for ph in res.get("phases", []) or []:
        rows.append([ph.get("name", "?"), "skipped" if ph.get("skipped") else "done", ph.get("steps", ""),
                     _size(ph.get("tokens")) if ph.get("tokens") is not None else "",
                     f"{ph['val_loss']:.4f}" if isinstance(ph.get("val_loss"), (int, float)) else "",
                     f"{ph['elapsed']:.0f}s" if isinstance(ph.get("elapsed"), (int, float)) else ""])
    if rows:
        print(format_table(rows, ["PHASE", "STATUS", "STEPS", "TOKENS", "VAL_LOSS", "TIME"], "llrrrr"))
    print(f"\nmodel: {res.get('model_path')}")
    if res.get("saved_to"):
        print(f"saved: {res['saved_to']}")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from moonlight.chat import ChatSession

    llm = _load_llm(args.model, args)
    gen = _gen_kwargs(args)
    session = ChatSession(llm, system=args.system, tools=True if args.tools else None,
                          allow_code_exec=args.allow_code_exec, **gen)
    images = list(args.image or [])
    if args.message is not None:
        print(session.send(args.message, images=images or None))
        return 0
    print(f"Moonlight chat — {llm.config.name} ({llm.device}). Commands: /reset, /image PATH, /exit")
    pending: List[str] = images
    while True:
        try:
            line = input("\nsiz> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        text = line.strip()
        if not text:
            continue
        if text in ("/exit", "/quit", "/q"):
            return 0
        if text == "/reset":
            session.reset()
            print("(conversation cleared)")
            continue
        if text.startswith("/image "):
            pending.append(text[len("/image "):].strip())
            print(f"(image queued: {pending[-1]})")
            continue
        _print_flush(f"{llm.config.chat.name}> ")
        streamed = []

        def on_text(delta: str) -> None:
            streamed.append(delta)
            _print_flush(delta)

        try:
            reply = session.send(text, images=pending or None, stream=on_text)
        except (RuntimeError, ValueError) as e:
            print()
            _err(str(e))
            continue
        pending = []
        if not streamed:  # tool round(s) were muted, or nothing was streamed
            _print_flush(reply)
        for call in session.tool_log[-5:] if args.verbose else []:
            print(f"\n  [tool] {call['name']}({json.dumps(call['arguments'], ensure_ascii=False)}) -> "
                  f"{call['result'][:200]}")
        print()


def cmd_bench(args: argparse.Namespace) -> int:
    from moonlight.benchmarks.suites import type as suite_type

    llm = _load_llm(args.model, args)
    suite = suite_type(args.suite or llm.config.benchmark)
    report = suite.run(llm, limit=args.limit, out=args.out, offline=bool(args.offline))
    print(report.to_markdown())
    if args.out:
        print(f"\nreport saved to {args.out}", file=sys.stderr)
    return 1 if (args.strict and not report.passed) else 0


def cmd_plan(args: argparse.Namespace) -> int:
    llm = _load_llm(args.model, args)
    print(llm.plan(args.task, **_gen_kwargs(args)))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    llm = _load_llm(args.model, args)
    res = llm.build_project(args.task, args.workdir, max_steps=args.max_steps, **_gen_kwargs(args))
    print(res.final_answer)
    print(f"\nworkdir: {res.workdir}\nfiles: {', '.join(res.files) or '-'}\nsteps: {res.steps}\n"
          f"tests passed: {res.tests_passed}")
    return 0 if res.tests_passed else 1


def cmd_version(args: argparse.Namespace) -> int:
    from moonlight.version import __version__

    print(f"moonlight {__version__}")
    return 0


# -------------------------------------------------------------------------------------------------
# parser
# -------------------------------------------------------------------------------------------------

def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--set", action="append", metavar="KEY=VALUE", default=[],
                   help="config override (repeatable), e.g. --set chat.temperature=0.5")
    p.add_argument("--offline", action="store_true", help="no downloads (bundled assets + synthetic data)")
    p.add_argument("--device", default=None, help="cuda | cpu | mps | cuda:N (default: auto)")


def _add_gen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--best-of", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--show-thinking", action="store_true", help="also print the <|think|> part")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moonlight", description="Moonlight: hybrid Attention + Retention LLMs.")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("list", help="list configs")
    p.add_argument("--tier", choices=["test", "mini", "si"], default=None)
    p.add_argument("--names", action="store_true", help="names only")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("info", help="describe a config")
    p.add_argument("name")
    p.add_argument("--yaml", action="store_true", help="print the full YAML")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", default=[])
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("params", help="exact parameter count, memory and FLOPs of a config")
    p.add_argument("name")
    p.add_argument("--dtype", default="bf16", help="bf16 | fp16 | fp32")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", default=[])
    p.set_defaults(func=cmd_params)

    p = sub.add_parser("train", help="train a config (all phases) and export the model")
    p.add_argument("name")
    _add_model_args(p)
    p.add_argument("--output", default=None, help="extra export path (.safetensors or directory)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("chat", help="chat with a model")
    p.add_argument("model", metavar="PATH_OR_NAME")
    _add_model_args(p)
    _add_gen_args(p)
    p.add_argument("--system", default=None, help="system prompt")
    p.add_argument("--tools", action="store_true", help="enable built-in tools (calculator, time, units)")
    p.add_argument("--allow-code-exec", action="store_true", help="also enable the python_exec tool")
    p.add_argument("-m", "--message", default=None, help="send one message, print the reply and exit")
    p.add_argument("--image", action="append", default=[], help="image path/URL for the first message")
    p.add_argument("-v", "--verbose", action="store_true", help="show tool calls")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("bench", help="run a benchmark suite")
    p.add_argument("model", metavar="PATH_OR_NAME")
    _add_model_args(p)
    p.add_argument("--suite", default=None, help="cc-001 ... cc-010 (default: the config's benchmark)")
    p.add_argument("--limit", type=int, default=None, help="max examples per task")
    p.add_argument("--out", default=None, help="save the report (.md or .json)")
    p.add_argument("--strict", action="store_true", help="exit code 1 if the suite does not pass")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("plan", help="answer a task in the PLAN FORMAT")
    p.add_argument("model", metavar="PATH_OR_NAME")
    p.add_argument("task")
    _add_model_args(p)
    _add_gen_args(p)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("build", help="let the model build a small project with workspace tools")
    p.add_argument("model", metavar="PATH_OR_NAME")
    p.add_argument("task")
    p.add_argument("--workdir", required=True)
    p.add_argument("--max-steps", type=int, default=24)
    _add_model_args(p)
    _add_gen_args(p)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("gen-configs", help="regenerate configs/ (scripts/generate_configs.py)")
    p.add_argument("args", nargs=argparse.REMAINDER, help="passed to the script, e.g. --check")
    p.set_defaults(func=cmd_gen_configs)

    p = sub.add_parser("version", help="print the version")
    p.set_defaults(func=cmd_version)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    # `gen-configs --check`: argparse would treat --check as unknown before REMAINDER kicks in
    if argv and argv[0] == "gen-configs":
        return cmd_gen_configs(argparse.Namespace(args=argv[1:]))
    args = parser.parse_args(argv)
    if args.version:
        return cmd_version(args)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:  # e.g. `moonlight list | head`
        try:
            sys.stdout = open(os.devnull, "w")
        except OSError:
            pass
        return 0
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, ImportError) as e:
        msg: Any = e.args[0] if isinstance(e, KeyError) and e.args else e
        if isinstance(e, ImportError) and getattr(e, "name", None) in ("torch", "safetensors"):
            msg = f"this command needs PyTorch ({e}); install it, e.g. `pip install torch`"
        _err(str(msg))
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
