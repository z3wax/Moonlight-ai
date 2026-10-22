"""Deterministic verifiers shared by synthetic data, GRPO rewards and benchmarks (SPEC §13.2). Torch-free."""

from __future__ import annotations

import pytest

from moonlight.data import verifiers as V

SPEC_TYPES = {"word_count_max", "word_count_min", "sentence_count_max", "sentence_count_min", "bullet_count",
              "paragraph_count", "include_keywords", "exclude_keywords", "keyword_frequency", "json_format",
              "lowercase", "uppercase", "no_commas", "ends_with", "starts_with", "title_brackets", "sections",
              "language_uz", "quotation_wrapped", "max_chars"}

UZ = ("Toshkent O'zbekistonning poytaxti va eng katta shahri hisoblanadi. Bu yerda ko'plab tarixiy "
      "obidalar bor, shuning uchun sayyohlar bu shaharni juda yaxshi ko'radi.")


def test_all_spec_types_supported():
    assert SPEC_TYPES <= set(V.CONSTRAINT_TYPES)


@pytest.mark.parametrize("response, constraint, expected", [
    ("one two three", {"type": "word_count_max", "n": 3}, True),
    ("one two three four", {"type": "word_count_max", "n": 3}, False),
    ("one two three", {"type": "word_count_min", "n": 4}, False),
    ("Birinchi gap. Ikkinchi gap! Uchinchi?", {"type": "sentence_count_max", "n": 3}, True),
    ("Birinchi gap. Ikkinchi gap! Uchinchi?", {"type": "sentence_count_min", "n": 4}, False),
    ("Ro'yxat:\n- olma\n- nok\n* uzum", {"type": "bullet_count", "n": 3}, True),
    ("1. birinchi\n2. ikkinchi", {"type": "bullet_count", "n": 2}, True),
    ("- a\n- b", {"type": "bullet_count", "n": 3}, False),
    ("Para one.\n\nPara two.\n\nPara three.", {"type": "paragraph_count", "n": 3}, True),
    ("Python and Moonlight are here", {"type": "include_keywords", "keywords": ["python", "MOONLIGHT"]}, True),
    ("Python only", {"type": "include_keywords", "keywords": ["python", "rust"]}, False),
    ("I like tea", {"type": "exclude_keywords", "keywords": ["coffee"]}, True),
    ("I like coffee", {"type": "exclude_keywords", "keywords": ["coffee"]}, False),
    ("data data and more data", {"type": "keyword_frequency", "word": "data", "at_least": 3}, True),
    ("data and data", {"type": "keyword_frequency", "word": "data", "at_least": 3}, False),
    ('{"a": 1, "b": [1, 2]}', {"type": "json_format"}, True),
    ('```json\n{"a": 1}\n```', {"type": "json_format"}, True),
    ("{a: 1}", {"type": "json_format"}, False),
    ("all lower case here", {"type": "lowercase"}, True),
    ("Not lower", {"type": "lowercase"}, False),
    ("ALL CAPS HERE", {"type": "uppercase"}, True),
    ("no commas at all", {"type": "no_commas"}, True),
    ("one, two", {"type": "no_commas"}, False),
    ("Hammasi bajarildi. Tayyor", {"type": "ends_with", "phrase": "Tayyor"}, True),
    ("Tayyor emas.", {"type": "ends_with", "phrase": "Tayyor"}, False),
    ("Salom! Bugun...", {"type": "starts_with", "phrase": "Salom"}, True),
    ("<<Mening sarlavham>>\nMatn", {"type": "title_brackets"}, True),
    ("No title here", {"type": "title_brackets"}, False),
    ("SECTION 1\nabc\nSECTION 2\ndef", {"type": "sections", "n": 2}, True),
    ("SECTION 1\nabc", {"type": "sections", "n": 2}, False),
    (UZ, {"type": "language_uz"}, True),
    ("This answer is written completely in English and not in Uzbek at all.", {"type": "language_uz"}, False),
    ("Это ответ на русском языке, а не на узбекском.", {"type": "language_uz"}, False),
    ('"Wrapped in quotes"', {"type": "quotation_wrapped"}, True),
    ("Not wrapped", {"type": "quotation_wrapped"}, False),
    ("short", {"type": "max_chars", "n": 10}, True),
    ("much longer than ten", {"type": "max_chars", "n": 10}, False),
])
def test_check(response, constraint, expected):
    assert V.check(response, constraint) is expected


def test_check_ignores_reasoning_block():
    resp = "<|think|>Let me think, carefully, with commas.<|/think|>no commas here"
    assert V.check(resp, {"type": "no_commas"})


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        V.check("x", {"type": "does_not_exist"})


def test_check_all():
    cons = [{"type": "lowercase"}, {"type": "no_commas"}, {"type": "word_count_max", "n": 2}]
    n_ok, n_total, details = V.check_all("hello there friend", cons)
    assert (n_ok, n_total) == (2, 3) and len(details) == 3
    assert [d["passed"] for d in details] == [True, True, False]


def test_ifeval_mapping():
    ids = ["punctuation:no_comma", "length_constraints:number_words", "keywords:existence",
           "detectable_format:number_bullet_lists", "startend:end_checker", "change_case:english_lowercase",
           "keywords:forbidden_words", "detectable_format:title", "detectable_format:multiple_sections",
           "keywords:frequency", "length_constraints:number_sentences", "length_constraints:number_paragraphs",
           "detectable_format:json_format", "change_case:english_capital", "startend:quotation"]
    kwargs = [{}, {"relation": "less than", "num_words": 50}, {"keywords": ["moon"]}, {"num_bullets": 2},
              {"end_phrase": "Any questions?"}, {}, {"forbidden_words": ["sun"]}, {},
              {"section_spliter": "SECTION", "num_sections": 2}, {"keyword": "moon", "frequency": 2,
                                                                     "relation": "at least"},
              {"relation": "at least", "num_sentences": 1}, {"num_paragraphs": 1}, {}, {}, {}]
    cons = V.ifeval_to_constraints(ids, kwargs)
    assert cons is not None and len(cons) == len(ids)
    assert all(c["type"] in SPEC_TYPES for c in cons)
    assert V.ifeval_to_constraints(["language:response_language"], [{"language": "de"}]) is None
    good = "- the moon\n- the moon again. Any questions?"
    small = V.ifeval_to_constraints(ids[:5], kwargs[:5])
    assert V.check_all(good, small)[0] == 5


def test_rewards_and_answer_extraction():
    assert V.extract_final_answer("<|think|>2+2 is 4<|/think|>The answer is 4") == "The answer is 4"
    assert V.extract_final_answer("no reasoning") == "no reasoning"
    assert V.reward_exact_number("<|think|>12*3=36<|/think|>Javob: 36", 36) == 1.0
    assert V.reward_exact_number("Answer: 1,250", 1250) == 1.0
    assert V.reward_exact_number("<|think|>maybe 36<|/think|>Javob: 35", 36) == 0.0
    assert V.reward_exact_number("I do not know", 7) == 0.0
    cons = [{"type": "lowercase"}, {"type": "no_commas"}]
    assert V.reward_constraints("all good here", cons) == 1.0
    partial = V.reward_constraints("Not, good", cons)
    assert 0.0 <= partial < 1.0
    assert V.reward_constraints("ok, fine", cons) < 1.0


def test_reward_code_executes_tests():
    good = "```python\ndef square(x):\n    return x * x\n```"
    bad = "```python\ndef square(x):\n    return x + x\n```"
    tests = ["assert square(3) == 9", "assert square(-2) == 4"]
    assert V.reward_code(good, tests, "square") == 1.0
    assert V.reward_code(bad, tests, "square") < 1.0
    assert V.reward_code("no code at all", tests, "square") == 0.0
