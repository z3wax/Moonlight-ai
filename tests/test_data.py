"""Data pipeline (SPEC §6 + §13.1, §13.3, §13.4): cleaning, adapters, sources, synthetic generators,
pretraining bins, SFT packing / masks, preference encoding and loaders. Torch-free except where noted."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from moonlight.data import adapters, cleaning, pretrain, sft, sources, synthetic, verifiers
from moonlight.tokenizer import ASSISTANT_ID, END_ID, PAD_ID, TOOL_CALL_ID

ROLES = {"system", "user", "assistant", "tool"}


def _check_conversation(conv):
    assert isinstance(conv, list) and conv, "conversation must be a non-empty list of messages"
    for m in conv:
        assert m["role"] in ROLES, m
        assert isinstance(m.get("content", ""), str)
        for call in m.get("tool_calls") or []:
            assert m["role"] == "assistant" and isinstance(call["name"], str)
            assert isinstance(call.get("arguments", {}), dict)
    assert conv[-1]["role"] == "assistant", "conversations end with an assistant answer"
    assert any(m["role"] == "user" for m in conv)


def _answer(conv):
    return conv[-1]["content"]


# ================================================================================================
# cleaning
# ================================================================================================

def test_norm_apostrophes_and_text():
    assert cleaning.norm_apostrophes("O‘zbekiston gʻalaba Oʼzbek O’zbek") == "O'zbekiston g'alaba O'zbek O'zbek"
    out = cleaning.norm_text("  Salom   dunyo  \r\n")
    assert "Salom" in out and "\r" not in out


def test_clean_doc_removes_urls_page_numbers_and_hyphenation():
    raw = "Bu so'z-\nlarni birlashtiramiz. Batafsil: https://example.com/page?a=1\n\n23\n\nKeyingi gap."
    out = cleaning.clean_doc(raw)
    assert "https://" not in out and "so'zlarni" in out
    assert not re.search(r"^\s*23\s*$", out, re.MULTILINE)
    assert "Keyingi gap." in out


def test_good_doc_filters():
    uz = "Bu o'zbek tilidagi oddiy matn bo'lib, unda turli gaplar mavjud. " * 8
    ru = "Это русский текст, который не должен пройти фильтр для узбекского. " * 8
    assert cleaning.good_doc(uz, min_chars=100, lang="uz")
    assert not cleaning.good_doc(ru, min_chars=100, lang="uz"), "Cyrillic text is rejected for uz"
    assert not cleaning.good_doc("qisqa", min_chars=100, lang="uz")
    assert not cleaning.good_doc("123 456 !!! ### " * 30, min_chars=100, lang="en")


def test_clean_code():
    body = '"""Utility module."""\n\n\ndef add(a, b):\n    """Add."""\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n'
    header = "\n".join(f"# Copyright (c) line {i} of a long license header" for i in range(30)) + "\n"
    out = cleaning.clean_code(header + body)
    assert out is not None and "Copyright" not in out and "def add" in out
    assert cleaning.clean_code("x = '" + "a" * 5000 + "'\nprint(x)\nprint(x)\n") is None, "minified/long lines dropped"
    assert cleaning.clean_code(body) is not None


def test_dedup_exact():
    assert list(cleaning.dedup_exact(["a", "b", "a", " b ", "c"])) == ["a", "b", "c"]


# ================================================================================================
# sources
# ================================================================================================

SPEC_SOURCES = {
    "pretrain": ["uz-news", "uz-telegram", "uz-wiki", "uz-books", "fineweb-edu", "cosmopedia", "stack-python",
                 "codeparrot"],
    "sft": ["alpaca-uz", "uzbek-instruct", "sft-uz-9k", "oasst2-uz", "slimorca-uz", "smoltalk", "smoltalk-everyday",
            "smoltalk-code", "smoltalk-tools", "python-18k", "codefeedback", "glaive-tools",
            "smoltalk-constraints", "smoltalk-math", "smoltalk-magpie", "openthoughts"],
    "vision": ["flickr8k", "flickr30k"],
    "bench": ["humaneval", "mbpp", "mmlu", "arc-easy", "arc-challenge", "hellaswag", "gsm8k", "belebele-uz", "ifeval"],
    "pref": ["ultrafeedback"],
}


def test_sources_registry():
    for kind, names in SPEC_SOURCES.items():
        for name in names:
            spec = sources.get_source(name)
            assert spec.name == name and spec.kind == kind, (name, spec.kind)
            assert spec.fmt in ("parquet", "jsonl", "json", "json.gz"), name
            assert spec.urls and all(u.startswith("https://huggingface.co/datasets/") for u in spec.urls), name
            assert "/resolve/main/" in spec.urls[0]
    assert set(sources.DATASETS) >= {n for names in SPEC_SOURCES.values() for n in names}
    assert len(sources.get_source("uz-news").urls) == 7
    assert len(sources.get_source("uz-books").urls) == 28
    assert len(sources.get_source("smoltalk").urls) == 9
    assert len(sources.get_source("smoltalk-magpie").urls) == 6
    assert len(sources.get_source("openthoughts").urls) == 6
    assert sources.get_source("gsm8k-train").urls[0].endswith("main/train-00000-of-00001.parquet")
    assert sources.get_source("smoltalk-everyday").urls[0].endswith(
        "data/everyday-conversations/train-00000-of-00001.parquet")
    with pytest.raises((KeyError, ValueError)):
        sources.get_source("no-such-dataset")


# ================================================================================================
# adapters
# ================================================================================================

def test_to_text():
    text = "Toshkent — O'zbekistonning poytaxti. " * 20
    assert adapters.to_text("uz-news", {"text": text})
    code = "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n\n\nprint(add(1, 2))\n"
    out = adapters.to_text("stack-python", {"content": code})
    assert out is None or "def add" in out


def test_instruction_adapters():
    conv = adapters.to_conversation("alpaca-uz", {"instruction": "Salom ayt", "input": "", "output": "Salom!"})
    assert [m["role"] for m in conv] == ["user", "assistant"] and conv[-1]["content"] == "Salom!"
    conv = adapters.to_conversation("alpaca-uz", {"instruction": "Tarjima qil", "input": "hello", "output": "salom"})
    assert "hello" in conv[0]["content"]
    conv = adapters.to_conversation("slimorca-uz", {"conversations": [
        {"from": "system", "value": "Siz yordamchisiz."}, {"from": "human", "value": "2+2?"}, {"from": "gpt", "value": "4"}]})
    assert [m["role"] for m in conv] == ["system", "user", "assistant"]
    conv = adapters.to_conversation("smoltalk", {"messages": [{"role": "user", "content": "Hi"},
                                                             {"role": "assistant", "content": "Hello!"}]})
    _check_conversation(conv)
    conv = adapters.to_conversation("codefeedback", {"query": "Reverse a string", "answer": "s[::-1]"})
    assert conv[-1]["content"] == "s[::-1]"
    assert adapters.to_conversation("alpaca-uz", {"instruction": "", "input": "", "output": ""}) is None


def test_glaive_tool_format_conversion():
    system = ('SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n'
              '{\n    "name": "calculate_bmi",\n    "description": "Calculate BMI",\n    "parameters": {\n'
              '        "type": "object",\n        "properties": {"weight": {"type": "number"}, "height": {"type": "number"}},\n'
              '        "required": ["weight", "height"]\n    }\n}\n')
    chat = ('USER: Hi, I weigh 70 kg and I am 1.75 m tall. What is my BMI?\n\n\n'
            'ASSISTANT: <functioncall> {"name": "calculate_bmi", "arguments": \'{"weight": 70, "height": 1.75}\'} '
            '<|endoftext|>\n\n\nFUNCTION RESPONSE: {"bmi": 22.86}\n\n\nASSISTANT: Your BMI is 22.86. <|endoftext|>\n\n\n')
    conv = adapters.to_conversation("glaive-tools", {"system": system, "chat": chat})
    _check_conversation(conv)
    sys_msg = conv[0]
    assert sys_msg["role"] == "system" and sys_msg["tools"][0]["name"] == "calculate_bmi"
    call = next(m for m in conv if m.get("tool_calls"))
    assert call["tool_calls"][0] == {"name": "calculate_bmi", "arguments": {"weight": 70, "height": 1.75}}
    tool = next(m for m in conv if m["role"] == "tool")
    assert json.loads(tool["content"]) == {"bmi": 22.86}
    assert "<|endoftext|>" not in conv[-1]["content"]


def test_openthoughts_and_preference_and_rl_adapters():
    row = {"system": "s", "conversations": [
        {"from": "user", "value": "1+1?"},
        {"from": "assistant", "value": "<|begin_of_thought|>bir qo'sh bir<|end_of_thought|>"
                                       "<|begin_of_solution|>2<|end_of_solution|>"}]}
    conv = adapters.to_conversation("openthoughts", row)
    assert conv[-1]["content"] == "<|think|>bir qo'sh bir<|/think|>2"
    pref = adapters.to_preference({
        "prompt": "hi", "chosen": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}],
        "rejected": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "no"}],
        "score_chosen": 8.0, "score_rejected": 2.0})
    chosen, rejected = pref
    assert chosen[-1]["content"] == "Hello!" and rejected[-1]["content"] == "no"
    assert adapters.gsm8k_final_answer("work...\n#### 1,234") in ("1234", "1,234")
    rl = adapters.to_rl_prompt("gsm8k-train", {"question": "What is 2+3?", "answer": "2+3=5\n#### 5"})
    assert rl["prompt_messages"][-1]["role"] == "user" and float(rl["reward_data"]["target"]) == 5


# ================================================================================================
# synthetic generators
# ================================================================================================

GENERATORS = [
    ("identity_conversations", {"name": "Moonlight", "n": 12}),
    ("greeting_conversations", {"n": 12}),
    ("knowledge_conversations", {"n": 12}),
    ("arithmetic_conversations", {"n": 12}),
    ("tool_call_conversations", {"n": 12}),
    ("code_conversations", {"n": 12}),
    ("think_conversations", {"n": 12}),
    ("plan_conversations", {"n": 12}),
    ("constraint_conversations", {"n": 12}),
    ("detail_conversations", {"n": 12}),
    ("project_conversations", {"n": 4}),
]


@pytest.mark.parametrize("fn_name, kw", GENERATORS, ids=[g[0] for g in GENERATORS])
def test_synthetic_generator_shapes_and_determinism(fn_name, kw, byte_tokenizer):
    fn = getattr(synthetic, fn_name)
    convs = fn(**kw)
    assert len(convs) == kw["n"]
    for conv in convs:
        _check_conversation(conv)
        ids, mask = byte_tokenizer.encode_chat(conv)
        assert len(ids) == len(mask) and sum(mask) > 0
    assert fn(**kw) == convs, "generators are deterministic for a fixed seed"
    if "seed" in fn.__code__.co_varnames:
        other = fn(**dict(kw, seed=1))
        assert other != convs or fn_name == "knowledge_conversations"


def test_identity_uses_name():
    convs = synthetic.identity_conversations(name="Moonlight", n=20)
    assert sum("Moonlight" in _answer(c) for c in convs) >= 15


def test_knowledge_facts_are_accurate():
    convs = synthetic.knowledge_conversations()
    assert len(convs) >= 40
    blob = "\n".join(m["content"] for c in convs for m in c)
    for fact in ("Toshkent", "1991", "Navoiy", "Amir Temur", "Ulug'bek", "Ibn Sino", "Bobur", "Yupiter", "so'm"):
        assert fact in blob, f"missing fact about {fact}"
    assert re.search(r"31.?avgust|1.?sentabr", blob)


def test_arithmetic_answers_are_exact():
    convs = synthetic.arithmetic_conversations(n=60, seed=3)
    checked = 0
    for conv in convs:
        q = conv[[i for i, m in enumerate(conv) if m["role"] == "user"][-1]]["content"]
        m = re.search(r"(-?\d+)\s*([+\-*/×÷])\s*(-?\d+)", q)
        if not m:
            continue
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        expected = {"+": a + b, "-": a - b, "*": a * b, "×": a * b}.get(op)
        if expected is None:
            continue
        nums = [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*(?:\.\d+)?", _answer(conv))]
        assert float(expected) in nums, (q, _answer(conv))
        checked += 1
    assert checked >= 10
    assert any(any(m.get("tool_calls") for m in c) for c in convs), "some arithmetic uses the calculator tool"


def test_tool_call_conversations_full_loop():
    for conv in synthetic.tool_call_conversations(n=20, seed=1):
        roles = [m["role"] for m in conv]
        i = next(k for k, m in enumerate(conv) if m.get("tool_calls"))
        assert roles[i - 1] == "user" or roles[i - 1] == "tool"
        assert roles[i + 1] == "tool" and conv[-1]["role"] == "assistant"
        call = conv[i]["tool_calls"][0]
        assert call["name"] in {"calculator", "current_time", "unit_convert", "weather", "search_wikipedia"}


def test_code_and_train_problems_verified():
    assert synthetic.verify_code_tasks() == []
    assert synthetic.verify_train_problems() == []


@pytest.mark.slow
def test_project_trajectories_pass_their_tests():
    assert synthetic.verify_projects() == []
    conv = synthetic.project_conversations(n=1)[0]
    names = {c["name"] for m in conv for c in (m.get("tool_calls") or [])}
    assert {"write_file", "run_tests"} <= names
    assert names <= {"write_file", "read_file", "list_dir", "run_tests", "run_python"}


def test_think_format():
    for conv in synthetic.think_conversations(n=10):
        a = _answer(conv)
        assert a.startswith("<|think|>") and "<|/think|>" in a and a.split("<|/think|>")[1].strip()


def test_plan_format():
    for conv in synthetic.plan_conversations(n=10, seed=2):
        a = _answer(conv)
        assert a.startswith("<|think|>") and "<|/think|>" in a
        plan = a.split("<|/think|>", 1)[1]
        assert re.search(r"Reja|Plan", plan) and re.search(r"Tekshiruv|Verification", plan)
        assert re.search(r"^\s*1\.", plan, re.MULTILINE) and re.search(r"^\s*2\.", plan, re.MULTILINE)


def test_constraint_and_detail_examples_satisfy_verifiers():
    for ex in synthetic.constraint_examples(n=40, seed=5):
        assert 1 <= len(ex["constraints"]) <= 4
        n_ok, n_total, details = verifiers.check_all(ex["response"], ex["constraints"])
        assert n_ok == n_total, (ex["prompt"], details)
    for ex in synthetic.detail_examples(n=30, seed=5):
        n_ok, n_total, details = verifiers.check_all(ex["response"], ex["constraints"])
        assert n_total >= 1 and n_ok == n_total, (ex["prompt"], details)


def test_grpo_prompt_set():
    kinds = ["arithmetic", "constraints", "detail", "code_lite"]
    prompts = synthetic.grpo_prompt_set(kinds, 24, seed=0)
    assert len(prompts) == 24
    for p in prompts:
        assert set(p) >= {"prompt_messages", "reward_type", "reward_data"}
        assert p["prompt_messages"][-1]["role"] == "user"
    assert synthetic.grpo_prompt_set(kinds, 24, seed=0) == prompts
    assert len({p["reward_type"] for p in prompts}) >= 2


def test_code_lite_train_pool_is_disjoint_from_benchmarks():
    try:
        from moonlight.benchmarks import tasks as bench_tasks
    except ImportError:  # pragma: no cover - benchmarks need torch for some helpers
        pytest.skip("benchmarks.tasks not importable")
    bench = getattr(bench_tasks, "CODE_LITE_PROBLEMS", None)
    if not bench:
        pytest.skip("benchmark problem list not exposed as CODE_LITE_PROBLEMS")

    def key(p):
        return str(p.get("entry_point") or p.get("name") or p.get("prompt")) if isinstance(p, dict) else str(p)

    train = {key(p) for p in synthetic.TRAIN_CODE_PROBLEMS}
    held_out = {key(p) for p in bench}
    assert not (train & held_out), f"benchmark problems leaked into the GRPO train pool: {train & held_out}"


def test_preference_pairs_synthetic():
    pairs = synthetic.preference_pairs_synthetic(n=20, seed=0)
    assert len(pairs) == 20
    for chosen, rejected in pairs:
        _check_conversation(chosen)
        _check_conversation(rejected)
        assert chosen[:-1] == rejected[:-1], "pairs share the prompt"
        assert _answer(chosen) != _answer(rejected)


def test_build_synthetic_and_texts():
    convs = synthetic.build_synthetic({"identity": True, "knowledge": True, "arithmetic": 5, "tool_calls": 5,
                                       "code": 5, "plan": 2, "constraints": 2, "detail": 2, "projects": 1})
    assert len(convs) >= 20
    for c in convs:
        _check_conversation(c)
    texts = synthetic.synthetic_texts(n=10)
    assert len(texts) == 10 and all(isinstance(t, str) and t for t in texts)


# ================================================================================================
# SFT packing / loaders
# ================================================================================================

def _examples(tok, n=12):
    """Short encoded conversations (a 1-word system prompt keeps them well below 256 tokens)."""
    convs = synthetic.greeting_conversations(n) + synthetic.identity_conversations(n=n)
    convs = [[{"role": "system", "content": "S"}] + [m for m in c if m["role"] != "system"] for c in convs]
    return sft.encode_conversations(convs, tok)


def test_encode_conversations_matches_template(byte_tokenizer):
    convs = synthetic.greeting_conversations(3)
    ex = sft.encode_conversations(convs, byte_tokenizer)
    assert len(ex) == 3
    for (ids, mask), conv in zip(ex, convs):
        ref_ids, ref_mask = byte_tokenizer.encode_chat(conv)
        assert list(ids) == list(ref_ids) and list(mask) == list(ref_mask)


def test_pack_shapes_and_padding(byte_tokenizer):
    ex = _examples(byte_tokenizer)
    seq_len = 256
    ids, mask = sft.pack(ex, seq_len)
    assert ids.dtype == np.int32 and mask.dtype == np.uint8
    assert ids.shape == mask.shape and ids.shape[1] == seq_len + 1
    assert mask[ids == PAD_ID].sum() == 0, "padding is never trained"
    # every example is kept; one longer than seq_len + 1 is truncated to its first seq_len + 1 tokens
    expected = sum(int(np.sum(np.asarray(m)[:seq_len + 1])) for _, m in ex)
    assert int(mask.sum()) == expected
    assert np.all(ids[:, 0] == 1), "every packed row starts at a conversation boundary (<|bos|>)"
    assert ids.shape[0] < len(ex), "greedy packing combines short conversations"


def test_split_val(byte_tokenizer):
    ids, mask = sft.pack(_examples(byte_tokenizer, 30), 128)
    tr_ids, tr_mask, va_ids, va_mask = sft.split_val(ids, mask, frac=0.2)
    assert len(tr_ids) + len(va_ids) == len(ids) and len(va_ids) >= 1
    assert tr_ids.shape[1] == va_ids.shape[1] == ids.shape[1]
    assert tr_mask.shape == tr_ids.shape and va_mask.shape == va_ids.shape


def test_sft_loader_targets(byte_tokenizer):
    ids, mask = sft.pack(_examples(byte_tokenizer), 128)
    loader = sft.SFTLoader(ids, mask, batch_size=3, device="cpu", seed=0)
    x, y = loader.next_numpy()
    assert x.shape == y.shape == (3, 128)
    assert np.all((y == -100) | (y >= 0))
    # every target is either ignored or the next input token
    nxt = np.concatenate([x[:, 1:], np.full((3, 1), -1)], axis=1)
    ok = (y == -100) | (y == nxt) | (np.arange(128)[None, :] == 127)
    assert ok.all()
    assert (y != -100).any()


def test_sft_loader_torch_tensors(byte_tokenizer):
    torch = pytest.importorskip("torch")
    ids, mask = sft.pack(_examples(byte_tokenizer), 64)
    x, y = sft.SFTLoader(ids, mask, batch_size=2, device="cpu", seed=1).next()
    assert x.dtype == torch.long and y.dtype == torch.long and x.shape == (2, 64)


def test_encode_preference_and_loader(byte_tokenizer):
    chosen, rejected = synthetic.preference_pairs_synthetic(n=4, seed=1)[0]
    full = sft.encode_preference(chosen, rejected, byte_tokenizer, 4096)
    assert set(full) >= {"chosen_ids", "chosen_mask", "rejected_ids", "rejected_mask"}
    assert len(full["chosen_ids"]) == len(full["chosen_mask"])
    resp_tokens = int(np.sum(full["chosen_mask"]))
    short_len = int(resp_tokens * 1.5)  # prompt must be cut, response (< 3/4 of the budget) must survive
    assert short_len + 1 < len(full["chosen_ids"])
    short = sft.encode_preference(chosen, rejected, byte_tokenizer, short_len)
    assert len(short["chosen_ids"]) <= short_len + 1 and len(short["rejected_ids"]) <= short_len + 1
    assert int(np.sum(short["chosen_mask"])) == resp_tokens, "left-truncation keeps the whole response"
    assert list(short["chosen_ids"][-resp_tokens:]) == list(full["chosen_ids"][-resp_tokens:])

    pairs = [sft.encode_preference(c, r, byte_tokenizer, 512)
             for c, r in synthetic.preference_pairs_synthetic(n=6, seed=2)]
    loader = sft.PreferenceLoader(pairs, batch_size=4, device="cpu", seed=0)
    c_x, c_y, r_x, r_y = loader.next_numpy()
    assert c_x.shape == c_y.shape and r_x.shape == r_y.shape and c_x.shape[0] == 4
    assert (c_y != -100).any() and (r_y != -100).any()
    assert np.all(c_x[c_y == -100][...] >= 0)


# ================================================================================================
# pretraining bins
# ================================================================================================

def test_offline_pretrain_bins_and_loader(tmp_path, byte_tokenizer, tiny_config):
    cfg = tiny_config()
    train_bin, val_bin, stats = pretrain.build_pretrain_tokens(cfg, byte_tokenizer, tmp_path)
    assert train_bin.is_file() and val_bin.is_file()
    meta = pretrain.read_meta(train_bin)
    assert meta["dtype"] == "uint16" and meta["vocab_size"] == byte_tokenizer.vocab_size
    data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    assert data.size > 10_000 and int(data.max()) < byte_tokenizer.vocab_size
    assert int(np.sum(data == 2)) > 10, "documents are separated by EOS"
    loader = pretrain.PretrainLoader(train_bin, seq_len=32, batch_size=4, device="cpu", seed=0)
    x, y = loader.next_numpy()
    assert x.shape == y.shape == (4, 32)
    assert np.array_equal(x[:, 1:], y[:, :-1]), "y is x shifted by one"
    x2, _ = loader.next_numpy(seq_len=16, batch_size=2)
    assert x2.shape == (2, 16)


def test_pretrain_loader_torch(tmp_path, byte_tokenizer, tiny_config):
    torch = pytest.importorskip("torch")
    train_bin, _, _ = pretrain.build_pretrain_tokens(tiny_config(), byte_tokenizer, tmp_path)
    x, y = pretrain.PretrainLoader(train_bin, 16, 2, "cpu", 0).next()
    assert x.dtype == torch.long and y.dtype == torch.long and torch.equal(x[:, 1:], y[:, :-1])


def test_assistant_tool_call_tokens_trained(byte_tokenizer):
    conv = synthetic.tool_call_conversations(n=1, seed=4)[0]
    ((ids, mask),) = sft.encode_conversations([conv], byte_tokenizer)
    ids, mask = list(ids), list(mask)
    i = ids.index(TOOL_CALL_ID)
    assert mask[i] == 1 and ids[i - 1] == ASSISTANT_ID and mask[i - 1] == 0
    assert ids[-1] == END_ID and mask[-1] == 1
