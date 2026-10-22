"""Public API (SPEC §1, §8, §13.6): lazy import hygiene, ml.config / LLM / chat facade, CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from conftest import ALL_CONFIG_NAMES, REPO_ROOT, make_tiny_config

TORCH_FREE_MODULES = [
    "moonlight", "moonlight.config", "moonlight.params", "moonlight.registry", "moonlight.utils",
    "moonlight.tokenizer", "moonlight.data.sources", "moonlight.data.cleaning", "moonlight.data.adapters",
    "moonlight.data.synthetic", "moonlight.data.sft", "moonlight.data.verifiers", "moonlight.tools",
    "moonlight.benchmarks.suites",
]


def _run_py(code, **kw):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT), env=env,
                          timeout=180, **kw)


# ================================================================================================
# import hygiene (no torch needed)
# ================================================================================================

def test_torch_free_imports_and_lazy_package():
    code = f"""
import sys, importlib
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is blocked for this test")
sys.meta_path.insert(0, _Block())
for m in {TORCH_FREE_MODULES!r}:
    importlib.import_module(m)
import moonlight as ml
cfg = ml.config("ml-1m-chat")
assert cfg.name == "ml-1m-chat"
assert len(ml.list_configs()) == 100
rep = ml.estimate_params(cfg)
assert rep.total > 0
assert ml.benchmarks.type("cc-001").name == "cc-001"
assert cfg.bnchmrks.type("cc-001").name == "cc-001"
assert "torch" not in sys.modules
print("OK")
"""
    res = _run_py(code)
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-3000:]


def test_exports_and_version():
    import moonlight as ml

    for name in ("config", "list_configs", "LLM", "train", "Config", "estimate_params", "benchmarks",
                 "ChatSession", "Tool", "tool", "__version__"):
        assert name in ml.__all__, name
    assert ml.__version__ == "0.1.0"
    from moonlight.config import Config

    assert ml.Config is Config
    assert callable(ml.config) and callable(ml.train) and callable(ml.tool)
    rep = ml.estimate_params(ml.config("ml-20m-chat"))
    assert rep.active == rep.total and isinstance(rep.pretty(), str)


# ================================================================================================
# CLI
# ================================================================================================

def test_cli_list_info_params(capsys):
    from moonlight.cli import main

    assert main(["list"]) == 0
    out = capsys.readouterr().out
    for name in ("ml-1m-debug", "ml-20m-colab-chat", "ml-1t-a32b-vl"):
        assert name in out
    assert main(["list", "--tier", "mini"]) == 0
    out = capsys.readouterr().out
    assert "ml-300m-uz" in out and "ml-1m-chat" not in out
    assert main(["info", "ml-1m-chat"]) == 0
    assert "ml-1m-chat" in capsys.readouterr().out
    assert main(["params", "ml-30b-a3b-chat"]) == 0
    assert capsys.readouterr().out.strip()
    assert main(["info", "ml-does-not-exist"]) != 0


def test_cli_module_entry_point():
    res = subprocess.run([sys.executable, "-m", "moonlight", "list", "--tier", "test"], capture_output=True,
                         text=True, cwd=str(REPO_ROOT), timeout=120,
                         env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
    assert res.returncode == 0, res.stderr
    assert sum(name in res.stdout for name in ALL_CONFIG_NAMES[:25]) == 25


# ================================================================================================
# LLM facade on a random tiny model (torch)
# ================================================================================================

@pytest.fixture(scope="module")
def llm():
    pytest.importorskip("torch")
    from moonlight.llm import LLM

    return LLM(make_tiny_config(chat={"max_new_tokens": 8, "best_of": 2}), device="cpu")


def test_llm_basics(llm):
    from moonlight.params import estimate_params

    assert llm.model is not None and llm.config.name == "tiny-test"
    assert llm.num_params() == estimate_params(llm.config).total
    out = llm.chat("Salom! Qalaysan?")
    assert isinstance(out, str)
    assert isinstance(llm.generate("Toshkent", max_new_tokens=5), str)
    assert isinstance(llm.chat("2+2?", tools=True, max_new_tokens=6), str)
    history = [{"role": "user", "content": "Salom"}, {"role": "assistant", "content": "Va alaykum assalom"}]
    assert isinstance(llm.chat("Isming nima?", history=history, system="Qisqa javob ber."), str)


def test_chat_session_multiturn_and_tools(llm):
    import moonlight as ml

    @ml.tool
    def double(x: int) -> int:
        """Double a number."""
        return 2 * x

    session = ml.ChatSession(llm, tools=[double], max_tool_rounds=2, max_new_tokens=6)
    r1 = session.send("Salom")
    r2 = session.send("Yana bir savol")
    assert isinstance(r1, str) and isinstance(r2, str)
    users = [m for m in session.history if m["role"] == "user"]
    assert [m["content"] for m in users] == ["Salom", "Yana bir savol"]
    assert session.history[-1]["role"] == "assistant"
    session.reset()
    assert session.history == []


def test_tool_loop_executes_forced_tool_call(llm, monkeypatch):
    """Drive the ChatSession tool loop deterministically: the first generation is a tool call."""
    import moonlight as ml
    from moonlight.tokenizer import TOOL_CALL_ID

    calls = []

    @ml.tool
    def lookup(city: str) -> str:
        """Look up a city."""
        calls.append(city)
        return f"{city}: poytaxt"

    tok = llm.require_tokenizer()
    forced = [TOOL_CALL_ID] + tok.encode(json.dumps({"name": "lookup", "arguments": {"city": "Toshkent"}}))
    from moonlight.model import generation

    real = generation.generate
    state = {"n": 0}

    def fake_generate(model, input_ids, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return generation.GenerationResult(tokens=list(forced), all_tokens=[list(forced)], scores=[0.0],
                                               finished=[True])
        return real(model, input_ids, **kw)

    monkeypatch.setattr(generation, "generate", fake_generate)  # looked up at call time by the LLM
    session = ml.ChatSession(llm, tools=[lookup], max_tool_rounds=2, max_new_tokens=4)
    session.send("Toshkent haqida ayt")
    assert calls == ["Toshkent"]
    roles = [m["role"] for m in session.history]
    assert roles[:3] == ["user", "assistant", "tool"] and roles[-1] == "assistant"
    assert session.history[1]["tool_calls"][0]["name"] == "lookup"
    assert "poytaxt" in session.history[2]["content"]


def test_missing_tokenizer_raises():
    pytest.importorskip("torch")
    from moonlight.llm import LLM

    cfg = make_tiny_config(data={"offline": False})
    llm = LLM(cfg, device="cpu")
    with pytest.raises(RuntimeError, match="train first or load a checkpoint"):
        llm.chat("Salom")


def test_save_and_from_pretrained_roundtrip(llm, tmp_path):
    torch = pytest.importorskip("torch")
    from moonlight.llm import LLM

    path = llm.save(tmp_path / "tiny.safetensors")
    other = LLM.from_pretrained(path, device="cpu")
    ids = torch.tensor([[1, 4, 40, 41, 6, 5]])
    with torch.no_grad():
        torch.testing.assert_close(other.model(ids).logits, llm.model(ids).logits)
    assert other.config.to_dict() == llm.config.to_dict()


def test_plan_and_build_project(llm, tmp_path):
    from moonlight.chat import BUILDER_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT

    assert PLAN_SYSTEM_PROMPT and BUILDER_SYSTEM_PROMPT
    plan = llm.plan("Kichik CLI kalkulyator dasturini rejalashtir", max_new_tokens=8)
    assert isinstance(plan, str)
    work = tmp_path / "proj"
    res = llm.build_project("Write add(a, b) in calc.py with tests", work, max_steps=2, max_new_tokens=8)
    for field in ("workdir", "files", "tests_passed", "steps", "transcript"):
        assert hasattr(res, field), field
    assert res.tests_passed in (True, False) and res.steps <= 2
    assert os.path.realpath(res.workdir) == os.path.realpath(work)
    assert res.transcript and res.transcript[0]["role"] == "system"
    for f in res.files:  # whatever a random model wrote stays inside the workspace
        assert ".." not in f.split("/")
