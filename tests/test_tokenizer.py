"""Tokenizer & chat template (SPEC §5). Torch-free."""

from __future__ import annotations

import json

import pytest

from moonlight import tokenizer as T
from moonlight.tokenizer import MoonlightTokenizer, train_tokenizer

SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|image|>",
            "<|tool_call|>", "<|tool_result|>", "<|tools|>", "<|think|>", "<|/think|>", "<|code|>", "<|audio|>"]

CORPUS = [
    "O'zbekiston poytaxti Toshkent shahri. Bugun havo juda yaxshi, biz bog'ga boramiz.",
    "Mustaqillik kuni 1-sentabr kuni nishonlanadi. G'alaba va o'zbek tili haqida maqola.",
    "The quick brown fox jumps over the lazy dog. Language models learn from text.",
    "def add(a, b):\n    \"\"\"Return the sum.\"\"\"\n    return a + b\n\nprint(add(2, 3))\n",
] * 40


@pytest.fixture(scope="module")
def trained():
    return train_tokenizer(CORPUS, vocab_size=420, min_frequency=2)


def test_special_token_ids_fixed(byte_tokenizer):
    assert len(T.SPECIAL_TOKENS) == 32
    assert T.SPECIAL_TOKENS[:15] == SPECIALS
    assert T.SPECIAL_TOKENS[15:] == [f"<|reserved_{i}|>" for i in range(17)]
    consts = (T.PAD_ID, T.BOS_ID, T.EOS_ID, T.SYSTEM_ID, T.USER_ID, T.ASSISTANT_ID, T.END_ID, T.IMAGE_ID,
              T.TOOL_CALL_ID, T.TOOL_RESULT_ID, T.TOOLS_ID, T.THINK_ID, T.THINK_END_ID, T.CODE_ID, T.AUDIO_ID)
    assert consts == tuple(range(15))
    for i, tok in enumerate(T.SPECIAL_TOKENS):
        assert byte_tokenizer.special_ids[tok] == i
    assert byte_tokenizer.vocab_size == 32 + 256


@pytest.mark.parametrize("text", [
    "Salom, dunyo!",
    "O'zbekiston Respublikasi — 2024-yil, 12 ta viloyat.",
    "naïve café 東京 🚀 \t tabs\nnewlines  double  spaces",
    "def f(x):\n    return x ** 2  # comment\n",
    "",
])
def test_byte_level_roundtrip(byte_tokenizer, text):
    ids = byte_tokenizer.encode(text)
    assert all(T.NUM_SPECIAL_TOKENS <= i < 288 for i in ids)
    assert byte_tokenizer.decode(ids) == text


def test_apostrophe_normalization(byte_tokenizer, trained):
    for tok in (byte_tokenizer, trained):
        variants = ["O‘zbekiston", "Oʻzbekiston", "Oʼzbekiston", "O’zbekiston", "O'zbekiston"]
        encoded = {tuple(tok.encode(v)) for v in variants}
        assert len(encoded) == 1
        assert tok.decode(tok.encode("g‘alaba")) == "g'alaba"


def test_bos_and_skip_special(byte_tokenizer):
    ids = byte_tokenizer.encode("hi", add_bos=True)
    assert ids[0] == T.BOS_ID
    assert byte_tokenizer.decode(ids) == "hi"
    assert "<|bos|>" in byte_tokenizer.decode(ids, skip_special=False)
    assert byte_tokenizer.encode_batch(["a", "bc"]) == [byte_tokenizer.encode("a"), byte_tokenizer.encode("bc")]


def test_train_tokenizer(trained, tmp_path):
    assert 288 < trained.vocab_size <= 420
    for i, tok in enumerate(T.SPECIAL_TOKENS):
        assert trained.special_ids[tok] == i
    text = "O'zbekiston poytaxti Toshkent. def add(a, b): return a + b"
    ids = trained.encode(text)
    assert trained.decode(ids) == text
    assert len(ids) < len(text.encode("utf-8")), "BPE merges must compress the training domain"
    # digits are split one by one (split_digits=True)
    digits = trained.encode("1234567")
    assert len(digits) == 7 and trained.decode(digits) == "1234567"
    # persistence
    path = tmp_path / "tok.json"
    trained.save(path)
    loaded = MoonlightTokenizer.from_file(path)
    assert loaded.encode(text) == ids and loaded.vocab_size == trained.vocab_size
    assert MoonlightTokenizer.from_str(trained.to_str()).encode(text) == ids
    json.loads(trained.to_str())  # HF tokenizers JSON


def test_train_tokenizer_with_path_and_grouped_digits(tmp_path):
    path = tmp_path / "t.json"
    tok = train_tokenizer(CORPUS, vocab_size=400, path=path, split_digits=False)
    assert path.is_file()
    assert MoonlightTokenizer.from_file(path).encode("Toshkent 1991") == tok.encode("Toshkent 1991")
    assert tok.decode(tok.encode("1991 2024")) == "1991 2024"


def test_uzbek_apostrophe_words_stay_together(trained):
    pre = trained.backend.pre_tokenizer.pre_tokenize_str("o'zbek tili g'alaba")
    pieces = [p for p, _ in pre]
    assert "o'zbek" in pieces and any(p.endswith("g'alaba") for p in pieces)


# ================================================================================================
# chat template
# ================================================================================================

def _tokens(tok, ids):
    return [tok.decode([i], skip_special=False) for i in ids]


def test_encode_chat_basic_and_mask(byte_tokenizer):
    tok = byte_tokenizer
    msgs = [{"role": "user", "content": "Salom"}, {"role": "assistant", "content": "Va alaykum"}]
    ids, mask = tok.encode_chat(msgs)
    assert len(ids) == len(mask) and set(mask) <= {0, 1}
    default = tok.encode(T.DEFAULT_SYSTEM_PROMPT)
    expected = ([T.BOS_ID, T.SYSTEM_ID] + default + [T.END_ID, T.USER_ID] + tok.encode("Salom") + [T.END_ID]
                + [T.ASSISTANT_ID] + tok.encode("Va alaykum") + [T.END_ID])
    assert ids == expected
    trained_part = [i for i, m in zip(ids, mask) if m]
    assert trained_part == tok.encode("Va alaykum") + [T.END_ID]
    assert tok.render_chat(msgs) == tok.decode(ids, skip_special=False)
    assert "<|assistant|>Va alaykum<|end|>" in tok.render_chat(msgs)


def test_default_system_prompt_text():
    assert T.DEFAULT_SYSTEM_PROMPT.startswith("Sen Moonlight — o'zbek va ingliz tillarida")
    assert T.DEFAULT_SYSTEM_PROMPT.endswith("yordamchisisan.")


def test_explicit_system_and_generation_prompt(byte_tokenizer):
    tok = byte_tokenizer
    ids, mask = tok.encode_chat([{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}],
                                add_generation_prompt=True)
    assert ids[:2] == [T.BOS_ID, T.SYSTEM_ID] and ids[2:5] == tok.encode("SYS")
    assert ids[-1] == T.ASSISTANT_ID and sum(mask) == 0
    ids2, _ = tok.encode_chat([{"role": "user", "content": "hi"}], system="X", add_generation_prompt=True)
    assert ids2[2:3] == tok.encode("X") and ids2[3] == T.END_ID


def test_tools_tool_calls_and_images(byte_tokenizer):
    tok = byte_tokenizer
    schemas = [{"name": "calculator", "description": "Evaluate math",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}]
    msgs = [
        {"role": "user", "content": "Rasmdagi narsa va 2+2?", "images": 2},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "calculator", "arguments": {"expression": "2+2"}}]},
        {"role": "tool", "name": "calculator", "content": "4"},
        {"role": "assistant", "content": "Javob: 4"},
    ]
    ids, mask = tok.encode_chat(msgs, tools=schemas, n_image_tokens=3)
    text = tok.decode(ids, skip_special=False)
    # <|system|>{system}<|tools|>{json schemas}<|end|> (compare parsed JSON: separators are cosmetic)
    t0 = ids.index(T.TOOLS_ID)
    assert ids.index(T.SYSTEM_ID) < t0 < ids.index(T.USER_ID)
    tools_json = tok.decode(ids[t0 + 1:ids.index(T.END_ID, t0)])
    assert json.loads(tools_json) == schemas
    assert ids.count(T.IMAGE_ID) == 6
    assert "<|user|>" + "<|image|>" * 6 + "Rasmdagi" in text
    assert "<|assistant|><|tool_call|>" in text and "<|tool_result|>4<|end|>" in text
    call_start = ids.index(T.TOOL_CALL_ID)
    assert mask[call_start] == 1 and mask[call_start - 1] == 0  # <|assistant|> itself is not trained
    call_end = ids.index(T.END_ID, call_start)
    call_json = json.loads(tok.decode(ids[call_start + 1:call_end]))
    assert call_json == {"name": "calculator", "arguments": {"expression": "2+2"}}
    assert all(m == 1 for m in mask[call_start:call_end + 1])
    res = ids.index(T.TOOL_RESULT_ID)
    assert all(m == 0 for m in mask[res:ids.index(T.END_ID, res) + 1]), "tool results are not trained"
    assert mask[-1] == 1 and ids[-1] == T.END_ID
    assert not any(m and i == T.IMAGE_ID for i, m in zip(ids, mask))
    # no newlines are inserted between template parts
    assert "<|end|>\n" not in text


def test_multiturn_masks_every_assistant_message(byte_tokenizer):
    tok = byte_tokenizer
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]
    ids, mask = tok.encode_chat(msgs)
    trained = [i for i, m in zip(ids, mask) if m]
    assert trained == tok.encode("b") + [T.END_ID] + tok.encode("d") + [T.END_ID]


def test_think_tokens_encoded_as_specials(byte_tokenizer):
    tok = byte_tokenizer
    ids, mask = tok.encode_chat([{"role": "user", "content": "2+3?"},
                                 {"role": "assistant", "content": "<|think|>2+3=5<|/think|>5"}])
    assert T.THINK_ID in ids and T.THINK_END_ID in ids
    assert mask[ids.index(T.THINK_ID)] == 1
