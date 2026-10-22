"""Benchmark task implementations (SPEC §9, §13.7).

Registry: ``TASKS = {name: fn}`` with ``fn(llm, limit, offline, **params) -> dict`` returning a
TaskResult-like mapping ``{"score": float|None, "n": int, "details": dict}`` (``score None`` =
skipped; ``details["skipped"]`` then holds the reason). The runner adds the threshold verdict,
timing and error handling (:mod:`moonlight.benchmarks.runner`).

Conventions
-----------
* Generation is greedy (``temperature=0``, ``best_of=1``) and seeded, so reports are reproducible;
  code tasks additionally disable the repetition penalty and the no-repeat-ngram ban (code repeats
  n-grams legitimately). Chat tasks go through ``llm.chat`` (the real chat template + tool loop).
* ``limit`` caps the number of evaluated examples (bundled lists are not repeated: ``n =
  min(limit, available)``); for ``perplexity`` it is the max number of scored tokens per language
  file and for ``speed`` the number of decoded tokens.
* Held-out data: generated problems use :data:`EVAL_SEED` (12345, training generators default to
  seed 0) and the bundled prompts / code problems / plans / projects below are written for
  evaluation only (they do not appear in :mod:`moonlight.data.synthetic`).
* Tasks needing downloads (``belebele_uz``, ``arc_*``, ``hellaswag``, ``mmlu``, ``gsm8k``,
  ``humaneval``, ``mbpp``, ``ifeval``) are skipped when ``offline`` or when the download fails; the
  datasets come from :mod:`moonlight.data.sources` and are cached in ``config.data.cache_dir``.
  Subsets are deterministic (seeded sample of ``limit`` rows).
* Multiple-choice tasks are scored by log-likelihood (:func:`loglikelihood_batch`): ``acc`` uses
  the summed log-prob of each choice, ``acc_norm`` divides by the choice's UTF-8 byte length.

This module imports torch lazily (inside the functions that run the model); importing it is cheap
and torch-free.
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
import os
import random
import re
import tempfile
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from moonlight.data import verifiers as V
from moonlight.tokenizer import (
    BOS_ID,
    CODE_ID,
    END_ID,
    EOS_ID,
    NUM_SPECIAL_TOKENS,
    PAD_ID,
    THINK_END_ID,
    THINK_ID,
    TOOL_CALL_ID,
)

from .execution import check_program, run_test_functions

__all__ = [
    "TASKS",
    "EVAL_SEED",
    "loglikelihood",
    "loglikelihood_batch",
    "sequence_logprobs",
    "CHAT_FORMAT_PROMPTS",
    "GREETING_PROMPTS",
    "GREETING_KEYWORDS",
    "IDENTITY_PROMPTS",
    "KNOWLEDGE_UZ",
    "CODE_LITE_PROBLEMS",
    "IFEVAL_LITE_TOPICS",
    "DETAIL_CHECK_PROMPTS",
    "PROJECT_SPECS",
    "ifeval_lite_items",
    "plan_bench_problems",
    "check_plan_answer",
    "chat_sanity_ok",
]

#: seed of every generated evaluation set (training generators use other seeds, default 0).
EVAL_SEED = 12345

TaskOutput = Dict[str, Any]

# =================================================================================================
# Bundled evaluation data
# =================================================================================================

#: prompts for ``chat_format`` (Uzbek + English, short everyday requests).
CHAT_FORMAT_PROMPTS: List[str] = [
    "Salom! Bugun qanday kayfiyatdasan?",
    "Hello! Can you help me today?",
    "Menga bitta maslahat ber.",
    "Give me one tip for learning faster.",
    "Bahor haqida bir gap yoz.",
    "Write one sentence about the sea.",
    "Toshkent qayerda joylashgan?",
    "What is your favourite colour?",
    "Ertaga nima qilsam bo'ladi?",
    "Tell me something interesting.",
    "Kitob o'qish nima uchun foydali?",
    "How do I make tea?",
    "Menga bir so'z bilan javob ber: ha yoki yo'q?",
    "Say thank you in Uzbek.",
    "Qisqa she'r yozib ber.",
    "What day comes after Monday?",
    "Python nima?",
    "Explain what a computer is in one sentence.",
    "Rahmat, xayr!",
    "Good night!",
    "Matematikani qanday o'rgansam bo'ladi?",
    "Name three fruits.",
    "Qaysi fasl eng issiq?",
    "Why is the sky blue?",
    "O'zbek tilida 'hello' qanday aytiladi?",
    "Write a short greeting for a friend.",
    "Sport bilan shug'ullanish nima beradi?",
    "What is 2 plus 3?",
    "Bugun havo qanday bo'lishi mumkin?",
    "Recommend a good habit.",
]

#: prompts for ``chat_sanity`` (greetings / courtesy).
GREETING_PROMPTS: List[str] = [
    "Salom!", "Hello!", "Assalomu alaykum!", "Hi there!", "Salom, qalaysiz?", "Good morning!",
    "Xayrli tong!", "Hey, how are you?", "Yaxshimisiz?", "Nice to meet you!", "Rahmat!", "Thank you!",
    "Xayrli kech!", "Good evening!", "Salom, do'stim!", "Greetings!", "Assalomu alaykum, ishlaringiz yaxshimi?",
    "Hello, Moonlight!", "Kechirasiz, yordam bera olasizmi?", "Thanks a lot!",
]

#: greeting / courtesy stems for ``chat_sanity`` (matched as word prefixes, case-insensitive).
GREETING_KEYWORDS: Tuple[str, ...] = (
    "salom", "assalom", "alaykum", "xush", "rahmat", "yaxshi", "xayrli", "hayrli", "qalay", "marhamat", "yordam",
    "arzimaydi", "tanishganimdan", "xursand", "hello", "hi", "hey", "welcome", "thank", "thanks", "good",
    "glad", "greetings", "nice", "pleasure", "help", "morning", "evening", "great", "fine",
)

#: identity questions (Uzbek + English); the reply must contain ``config.chat.name``.
IDENTITY_PROMPTS: List[str] = [
    "Sen kimsan?", "Who are you?", "Isming nima?", "What is your name?", "O'zingni tanishtir.", "Introduce yourself.",
    "Kim bilan gaplashyapman?", "Who am I talking to?", "Sening isming nima?", "What should I call you?",
    "Sen qanday yordamchisan?", "Tell me about yourself.",
]

#: held-out Uzbek knowledge questions: ``(question, accepted regexes)`` -- the (normalized,
#: lower-cased) reply must match at least one regex.
KNOWLEDGE_UZ: List[Tuple[str, List[str]]] = [
    ("O'zbekiston Respublikasining poytaxti qaysi shahar hisoblanadi?", [r"toshkent", r"tashkent"]),
    ("O'zbekiston mustaqilligi qaysi yilda e'lon qilingan?", [r"1991"]),
    ("O'zbekistonda qaysi pul birligi muomalada?", [r"\bso'?m"]),
    ("O'zbekistonda nechta viloyat bor?", [r"\b12\b", r"o'n ikki"]),
    ("Amudaryodan tashqari O'zbekistonning yana qaysi yirik daryosi bor?", [r"sirdaryo"]),
    ("\"Xamsa\" dostonlarini kim yozgan?", [r"navoiy"]),
    ("Alisher Navoiy qaysi yilda tug'ilgan?", [r"1441"]),
    ("Amir Temur davlatining poytaxti qaysi shahar edi?", [r"samarqand"]),
    ("Samarqanddagi mashhur rasadxonani kim qurdirgan?", [r"ulug'?bek"]),
    ("\"Tib qonunlari\" asarining muallifi kim?", [r"ibn sino", r"ibn sina", r"avitsenna", r"avicenna"]),
    ("\"Algoritm\" so'zi qaysi olimning ismidan kelib chiqqan?", [r"xorazmiy", r"khwarizmi"]),
    ("\"Boburnoma\" asarini kim yozgan?", [r"bobur", r"babur"]),
    ("Registon maydoni qaysi shaharda joylashgan?", [r"samarqand"]),
    ("Quyosh tizimidagi eng katta sayyora qanday ataladi?", [r"yupiter"]),
    ("Yer sharidagi eng katta okean qaysi?", [r"tinch okean"]),
    ("Oddiy yil necha kundan iborat?", [r"\b365\b"]),
    ("Dengiz sathida suv necha gradusda qaynaydi?", [r"\b100\b"]),
    ("Toza suv necha gradusda muzlaydi?", [r"\b0\s*(?:°|daraja|gradus)", r"\bnol\b"]),
    ("Suvning kimyoviy formulasi qanday yoziladi?", [r"h2o", r"h₂o"]),
    ("Katta yoshli odam skeletida nechta suyak bor?", [r"\b206\b"]),
    ("Quyoshga eng yaqin joylashgan sayyora qaysi?", [r"merkuriy"]),
    ("Quyosh tizimida nechta sayyora bor?", [r"\b8\b", r"sakkiz"]),
    ("Dunyodagi eng baland cho'qqi qaysi?", [r"everest", r"jomolungma"]),
    ("O'zbekistonning davlat tili qaysi?", [r"o'zbek"]),
    ("Navro'z bayrami qaysi kuni nishonlanadi?", [r"\b21\b"]),
    ("Python dasturlash tilini kim yaratgan?", [r"guido", r"rossum"]),
    ("Bir baytda necha bit bor?", [r"\b8\b", r"sakkiz"]),
    ("Eng yengil kimyoviy element qaysi?", [r"vodorod"]),
    ("Odam yuragi nechta kameradan iborat?", [r"\b4\b", r"to'rt"]),
    ("Toshkent metrosi qaysi yilda ochilgan?", [r"1977"]),
    ("Qizilqum cho'li qaysi ikki daryo oralig'ida joylashgan?", [r"amudaryo[\s\S]*sirdaryo|sirdaryo[\s\S]*amudaryo"]),
    ("Uchburchak ichki burchaklarining yig'indisi necha gradus?", [r"\b180\b"]),
]


def _cl(entry: str, signature: str, en: str, uz: str, tests: Sequence[str], solution: str) -> Dict[str, Any]:
    return {"entry_point": entry, "signature": signature, "en": en, "uz": uz, "tests": list(tests),
            "solution": solution}


#: ``code_lite`` problems (held out: none of these appear in the synthetic training pools).
#: Each: ``entry_point``, ``signature``, ``en`` / ``uz`` task text, ``tests`` (asserts) and a
#: reference ``solution`` (used by the test-suite to validate the asserts).
CODE_LITE_PROBLEMS: List[Dict[str, Any]] = [
    _cl("balanced_brackets", "balanced_brackets(s: str) -> bool",
        "returns True if every bracket in s ('()', '[]', '{}') is closed in the correct order, else False",
        "s satridagi barcha qavslar ('()', '[]', '{}') to'g'ri tartibda yopilgan bo'lsa True, aks holda False qaytarsin",
        ["assert balanced_brackets('([]{})') is True", "assert balanced_brackets('([)]') is False",
         "assert balanced_brackets('') is True", "assert balanced_brackets('((') is False",
         "assert balanced_brackets('a(b)c') is True"],
        "def balanced_brackets(s: str) -> bool:\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n"
        "    for ch in s:\n        if ch in '([{':\n            stack.append(ch)\n        elif ch in pairs:\n"
        "            if not stack or stack.pop() != pairs[ch]:\n                return False\n    return not stack\n"),
    _cl("roman_to_int", "roman_to_int(s: str) -> int",
        "converts a Roman numeral such as 'MCMXCIV' to an integer",
        "'MCMXCIV' kabi rim raqamini butun songa aylantirsin",
        ["assert roman_to_int('III') == 3", "assert roman_to_int('IX') == 9", "assert roman_to_int('LVIII') == 58",
         "assert roman_to_int('MCMXCIV') == 1994"],
        "def roman_to_int(s: str) -> int:\n    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
        "    total = 0\n    for i, ch in enumerate(s):\n        if i + 1 < len(s) and v[ch] < v[s[i + 1]]:\n"
        "            total -= v[ch]\n        else:\n            total += v[ch]\n    return total\n"),
    _cl("is_power_of_two", "is_power_of_two(n: int) -> bool",
        "returns True if n is a positive power of two (1, 2, 4, 8, ...), else False",
        "n ikkining musbat darajasi (1, 2, 4, 8, ...) bo'lsa True, aks holda False qaytarsin",
        ["assert is_power_of_two(1) is True", "assert is_power_of_two(16) is True", "assert is_power_of_two(18) is False",
         "assert is_power_of_two(0) is False", "assert is_power_of_two(-4) is False"],
        "def is_power_of_two(n: int) -> bool:\n    return n > 0 and (n & (n - 1)) == 0\n"),
    _cl("count_digits", "count_digits(n: int) -> int",
        "returns how many decimal digits the integer n has (the sign is ignored; 0 has one digit)",
        "butun n sonida nechta raqam borligini qaytarsin (ishora hisobga olinmaydi; 0 da bitta raqam bor)",
        ["assert count_digits(0) == 1", "assert count_digits(12345) == 5", "assert count_digits(-907) == 3"],
        "def count_digits(n: int) -> int:\n    return len(str(abs(n)))\n"),
    _cl("matrix_trace", "matrix_trace(m: list) -> int",
        "returns the sum of the main diagonal of the square matrix m (a list of rows)",
        "kvadrat m matritsaning (qatorlar ro'yxati) bosh diagonali yig'indisini qaytarsin",
        ["assert matrix_trace([[1, 2], [3, 4]]) == 5", "assert matrix_trace([[5]]) == 5",
         "assert matrix_trace([[1, 0, 0], [0, 2, 0], [0, 0, 3]]) == 6"],
        "def matrix_trace(m: list) -> int:\n    return sum(m[i][i] for i in range(len(m)))\n"),
    _cl("product_of_list", "product_of_list(nums: list) -> int",
        "returns the product of all numbers in nums (1 for an empty list)",
        "nums ro'yxatidagi barcha sonlar ko'paytmasini qaytarsin (bo'sh ro'yxat uchun 1)",
        ["assert product_of_list([2, 3, 4]) == 24", "assert product_of_list([]) == 1",
         "assert product_of_list([5, -1]) == -5"],
        "def product_of_list(nums: list) -> int:\n    result = 1\n    for x in nums:\n        result *= x\n"
        "    return result\n"),
    _cl("collatz_steps", "collatz_steps(n: int) -> int",
        "returns how many Collatz steps (n -> n/2 if even, else 3n+1) it takes to reach 1 from n >= 1",
        "n >= 1 sonidan 1 ga yetish uchun nechta Kollats qadami (juft bo'lsa n/2, aks holda 3n+1) kerakligini "
        "qaytarsin",
        ["assert collatz_steps(1) == 0", "assert collatz_steps(6) == 8", "assert collatz_steps(7) == 16"],
        "def collatz_steps(n: int) -> int:\n    steps = 0\n    while n != 1:\n"
        "        n = n // 2 if n % 2 == 0 else 3 * n + 1\n        steps += 1\n    return steps\n"),
    _cl("hamming_distance", "hamming_distance(a: str, b: str) -> int",
        "returns the number of positions at which the equal-length strings a and b differ",
        "uzunligi teng a va b satrlar nechta pozitsiyada farq qilishini qaytarsin",
        ["assert hamming_distance('karolin', 'kathrin') == 3", "assert hamming_distance('abc', 'abc') == 0",
         "assert hamming_distance('', '') == 0"],
        "def hamming_distance(a: str, b: str) -> int:\n    return sum(1 for x, y in zip(a, b) if x != y)\n"),
    _cl("sum_of_evens", "sum_of_evens(nums: list) -> int",
        "returns the sum of the even numbers in nums",
        "nums ro'yxatidagi juft sonlar yig'indisini qaytarsin",
        ["assert sum_of_evens([1, 2, 3, 4]) == 6", "assert sum_of_evens([]) == 0", "assert sum_of_evens([-2, 3]) == -2"],
        "def sum_of_evens(nums: list) -> int:\n    return sum(x for x in nums if x % 2 == 0)\n"),
    _cl("kth_smallest", "kth_smallest(nums: list, k: int) -> int",
        "returns the k-th smallest element of nums (k is 1-based)",
        "nums ro'yxatidagi k-chi eng kichik elementni qaytarsin (k 1 dan boshlanadi)",
        ["assert kth_smallest([7, 10, 4, 3, 20, 15], 3) == 7", "assert kth_smallest([1], 1) == 1",
         "assert kth_smallest([5, 1, 5, 2], 3) == 5"],
        "def kth_smallest(nums: list, k: int) -> int:\n    return sorted(nums)[k - 1]\n"),
    _cl("compress_spaces", "compress_spaces(text: str) -> str",
        "replaces every run of whitespace in text with a single space and strips both ends",
        "text ichidagi ketma-ket bo'sh joylarni bitta probel bilan almashtirsin va ikki chetini tozalasin",
        ["assert compress_spaces('  a   b  c ') == 'a b c'", "assert compress_spaces('salom') == 'salom'",
         "assert compress_spaces('x\\t\\ny') == 'x y'", "assert compress_spaces('   ') == ''"],
        "def compress_spaces(text: str) -> str:\n    return ' '.join(text.split())\n"),
    _cl("last_word_length", "last_word_length(text: str) -> int",
        "returns the length of the last word in text (0 if there is no word)",
        "text dagi oxirgi so'z uzunligini qaytarsin (so'z bo'lmasa 0)",
        ["assert last_word_length('Salom dunyo  ') == 5", "assert last_word_length('a') == 1",
         "assert last_word_length('   ') == 0"],
        "def last_word_length(text: str) -> int:\n    words = text.split()\n    return len(words[-1]) if words else 0\n"),
    _cl("nth_prime", "nth_prime(n: int) -> int",
        "returns the n-th prime number (nth_prime(1) == 2)",
        "n-chi tub sonni qaytarsin (nth_prime(1) == 2)",
        ["assert nth_prime(1) == 2", "assert nth_prime(5) == 11", "assert nth_prime(10) == 29"],
        "def nth_prime(n: int) -> int:\n    count, candidate = 0, 1\n    while count < n:\n        candidate += 1\n"
        "        if all(candidate % d for d in range(2, int(candidate ** 0.5) + 1)):\n            count += 1\n"
        "    return candidate\n"),
    _cl("to_binary", "to_binary(n: int) -> str",
        "returns the binary representation of the non-negative integer n without the '0b' prefix",
        "manfiy bo'lmagan n sonining ikkilik ko'rinishini '0b' prefiksisiz qaytarsin",
        ["assert to_binary(0) == '0'", "assert to_binary(5) == '101'", "assert to_binary(255) == '11111111'"],
        "def to_binary(n: int) -> str:\n    return bin(n)[2:]\n"),
    _cl("missing_number", "missing_number(nums: list) -> int",
        "nums contains n distinct numbers from 0..n with exactly one missing; return the missing number",
        "nums ro'yxatida 0..n oralig'idagi n ta har xil son bor va bittasi tushib qolgan; o'sha sonni qaytarsin",
        ["assert missing_number([3, 0, 1]) == 2", "assert missing_number([0]) == 1", "assert missing_number([1]) == 0"],
        "def missing_number(nums: list) -> int:\n    n = len(nums)\n    return n * (n + 1) // 2 - sum(nums)\n"),
    _cl("common_elements", "common_elements(a: list, b: list) -> list",
        "returns the sorted list of distinct values that appear in both a and b",
        "a va b ro'yxatlarining ikkalasida ham uchraydigan takrorlanmas qiymatlarni o'sish tartibida qaytarsin",
        ["assert common_elements([1, 2, 2, 3], [2, 3, 4]) == [2, 3]", "assert common_elements([], [1]) == []",
         "assert common_elements([5, 1], [1, 5, 5]) == [1, 5]"],
        "def common_elements(a: list, b: list) -> list:\n    return sorted(set(a) & set(b))\n"),
    _cl("average_word_length", "average_word_length(text: str) -> float",
        "returns the average length of the words in text rounded to 2 decimals (0.0 for no words)",
        "text dagi so'zlar o'rtacha uzunligini 2 xonagacha yaxlitlab qaytarsin (so'z bo'lmasa 0.0)",
        ["assert average_word_length('hi there') == 3.5", "assert average_word_length('') == 0.0",
         "assert average_word_length('a bb ccc') == 2.0"],
        "def average_word_length(text: str) -> float:\n    words = text.split()\n    if not words:\n"
        "        return 0.0\n    return round(sum(len(w) for w in words) / len(words), 2)\n"),
    _cl("pairwise_sums", "pairwise_sums(nums: list) -> list",
        "returns the list of sums of each pair of neighbouring elements of nums",
        "nums ro'yxatidagi har ikki qo'shni element yig'indilari ro'yxatini qaytarsin",
        ["assert pairwise_sums([1, 2, 3, 4]) == [3, 5, 7]", "assert pairwise_sums([5]) == []",
         "assert pairwise_sums([]) == []"],
        "def pairwise_sums(nums: list) -> list:\n    return [nums[i] + nums[i + 1] for i in range(len(nums) - 1)]\n"),
    _cl("is_armstrong", "is_armstrong(n: int) -> bool",
        "returns True if the non-negative n equals the sum of its digits each raised to the number of digits "
        "(e.g. 153 = 1**3 + 5**3 + 3**3)",
        "manfiy bo'lmagan n soni raqamlarining raqamlar soniga teng darajalari yig'indisiga teng bo'lsa True "
        "qaytarsin (masalan 153 = 1**3 + 5**3 + 3**3)",
        ["assert is_armstrong(153) is True", "assert is_armstrong(370) is True", "assert is_armstrong(10) is False",
         "assert is_armstrong(9) is True"],
        "def is_armstrong(n: int) -> bool:\n    digits = str(n)\n    k = len(digits)\n"
        "    return n == sum(int(d) ** k for d in digits)\n"),
    _cl("sort_by_length", "sort_by_length(words: list) -> list",
        "returns the words sorted by length, keeping the original order of words with equal length",
        "so'zlarni uzunligi bo'yicha saralab qaytarsin; uzunligi teng so'zlarning asl tartibi saqlansin",
        ["assert sort_by_length(['ccc', 'a', 'bb']) == ['a', 'bb', 'ccc']",
         "assert sort_by_length(['bb', 'aa', 'c']) == ['c', 'bb', 'aa']", "assert sort_by_length([]) == []"],
        "def sort_by_length(words: list) -> list:\n    return sorted(words, key=len)\n"),
]

#: ``ifeval_lite`` topics (held out from the training topic pool).
IFEVAL_LITE_TOPICS: List[Dict[str, Dict[str, Any]]] = [
    {"uz": {"prompt": "Sport bilan muntazam shug'ullanishning foydasi haqida yozing.", "kw": ["salomatlik", "kuch"],
            "avoid": ["dori", "kasal"], "main": "sport"},
     "en": {"prompt": "Write about the benefits of regular exercise.", "kw": ["health", "energy"],
            "avoid": ["medicine", "illness"], "main": "exercise"}},
    {"uz": {"prompt": "To'g'ri ovqatlanish haqida maslahatlar bering.", "kw": ["sabzavot", "suv"],
            "avoid": ["shakar", "chips"], "main": "ovqat"},
     "en": {"prompt": "Give advice about healthy eating.", "kw": ["vegetables", "water"],
            "avoid": ["sugar", "chips"], "main": "food"}},
    {"uz": {"prompt": "Chet tilini o'rganish yo'llari haqida yozing.", "kw": ["lug'at", "mashq"],
            "avoid": ["qiyin", "imkonsiz"], "main": "til"},
     "en": {"prompt": "Describe good ways to learn a foreign language.", "kw": ["vocabulary", "practice"],
            "avoid": ["difficult", "impossible"], "main": "language"}},
    {"uz": {"prompt": "Shaharda jamoat transportidan foydalanish haqida yozing.", "kw": ["avtobus", "metro"],
            "avoid": ["taksi", "tirbandlik"], "main": "transport"},
     "en": {"prompt": "Write about using public transport in a city.", "kw": ["bus", "metro"],
            "avoid": ["taxi", "traffic"], "main": "transport"}},
    {"uz": {"prompt": "Pul tejash bo'yicha maslahatlar yozing.", "kw": ["byudjet", "jamg'arma"],
            "avoid": ["qarz", "kredit"], "main": "pul"},
     "en": {"prompt": "Write tips on how to save money.", "kw": ["budget", "savings"],
            "avoid": ["debt", "loan"], "main": "money"}},
    {"uz": {"prompt": "Bahor fasli haqida qisqa matn yozing.", "kw": ["gul", "yomg'ir"],
            "avoid": ["qor", "sovuq"], "main": "bahor"},
     "en": {"prompt": "Write a short text about spring.", "kw": ["flowers", "rain"],
            "avoid": ["snow", "cold"], "main": "spring"}},
    {"uz": {"prompt": "Internetning ta'limdagi o'rni haqida yozing.", "kw": ["dars", "bilim"],
            "avoid": ["o'yin", "reklama"], "main": "internet"},
     "en": {"prompt": "Write about the role of the internet in education.", "kw": ["lessons", "knowledge"],
            "avoid": ["games", "advertising"], "main": "internet"}},
    {"uz": {"prompt": "Chiqindilarni qayta ishlash nima uchun muhim?", "kw": ["tabiat", "plastik"],
            "avoid": ["pul", "soliq"], "main": "chiqindi"},
     "en": {"prompt": "Why is recycling important?", "kw": ["nature", "plastic"],
            "avoid": ["money", "tax"], "main": "recycling"}},
    {"uz": {"prompt": "Oilaviy an'analar haqida yozing.", "kw": ["bayram", "mehr"],
            "avoid": ["janjal", "telefon"], "main": "oila"},
     "en": {"prompt": "Write about family traditions.", "kw": ["holidays", "care"],
            "avoid": ["arguments", "phones"], "main": "family"}},
    {"uz": {"prompt": "Muzeylarga borishning foydasi haqida yozing.", "kw": ["tarix", "madaniyat"],
            "avoid": ["zerikarli", "qimmat"], "main": "muzey"},
     "en": {"prompt": "Write about the value of visiting museums.", "kw": ["history", "culture"],
            "avoid": ["boring", "expensive"], "main": "museum"}},
]

# instruction templates of ifeval_lite (deliberately worded differently from verifiers.describe)
_IFL_TEMPLATES: Dict[str, Dict[str, str]] = {
    "word_count_max": {"en": "Your reply must not exceed {n} words.", "uz": "Javobingiz {n} so'zdan uzun bo'lmasin."},
    "word_count_min": {"en": "Write no fewer than {n} words.", "uz": "Kamida {n} so'z ishlating."},
    "sentence_count_max": {"en": "Use {n} sentences or fewer.", "uz": "{n} tadan ortiq gap yozmang."},
    "bullet_count": {"en": "Format the answer as a list of exactly {n} items, each line beginning with \"- \".",
                     "uz": "Javobni aynan {n} bandli ro'yxat qilib yozing, har bir qator \"- \" bilan boshlansin."},
    "paragraph_count": {"en": "Write exactly {n} paragraphs and put a line containing only *** between them.",
                        "uz": "Aynan {n} ta xatboshi yozing va ular orasiga faqat *** dan iborat qator qo'ying."},
    "include_keywords": {"en": "Make sure the words {kws} appear somewhere in your text.",
                         "uz": "Matningizda {kws} so'zlari albatta uchrasin."},
    "exclude_keywords": {"en": "Never mention the words {kws}.", "uz": "{kws} so'zlarini umuman tilga olmang."},
    "keyword_frequency": {"en": "Use the word \"{w}\" at least {k} times.",
                          "uz": "\"{w}\" so'zini kamida {k} marta ishlating."},
    "lowercase": {"en": "Use only lowercase letters, no capitals at all.",
                  "uz": "Faqat kichik harflardan foydalaning, bitta ham bosh harf bo'lmasin."},
    "uppercase": {"en": "Respond entirely in CAPITAL LETTERS.", "uz": "Butun javobni KATTA HARFLARDA yozing."},
    "no_commas": {"en": "Avoid the comma character completely.", "uz": "Vergul belgisini umuman qo'llamang."},
    "ends_with": {"en": "Finish your reply with these exact words: \"{p}\"",
                  "uz": "Javobingizni aynan shu so'zlar bilan yakunlang: \"{p}\""},
    "starts_with": {"en": "Begin your reply with the words \"{p}\".", "uz": "Javobni \"{p}\" so'zlari bilan boshlang."},
    "title_brackets": {"en": "Put a title at the top inside double angle brackets, for example <<My title>>.",
                       "uz": "Eng tepaga qo'sh burchakli qavs ichida sarlavha qo'ying, masalan <<Sarlavha>>."},
    "json_format": {"en": "Return the entire output as valid JSON and nothing else.",
                    "uz": "Butun natijani yaroqli JSON ko'rinishida qaytaring, boshqa hech narsa yozmang."},
    "quotation_wrapped": {"en": "Enclose your whole reply in double quotes.",
                          "uz": "Butun javobingizni qo'shtirnoq ichiga oling."},
    "sections": {"en": "Split the answer into {n} parts labelled SECTION 1, SECTION 2 and so on.",
                 "uz": "Javobni SECTION 1, SECTION 2 va hokazo deb belgilangan {n} qismga bo'ling."},
    "language_uz": {"en": "Answer in Uzbek (Latin script) only.",
                    "uz": "Javobni faqat o'zbek tilida, lotin alifbosida yozing."},
}

_IFL_START = {"uz": "Qisqa qilib aytganda", "en": "To put it briefly"}
_IFL_END = {"uz": "shu bilan tamom.", "en": "that is all."}

# recipes: lists of (constraint type, params builder(topic, lang) -> dict)
_IFL_RECIPES: List[List[Tuple[str, Callable[[Mapping[str, Any], str], Dict[str, Any]]]]] = [
    [("word_count_max", lambda t, l: {"n": 60}), ("include_keywords", lambda t, l: {"keywords": t["kw"][:2]})],
    [("bullet_count", lambda t, l: {"n": 3}), ("exclude_keywords", lambda t, l: {"keywords": t["avoid"]})],
    [("lowercase", lambda t, l: {}), ("ends_with", lambda t, l: {"phrase": _IFL_END[l]})],
    [("title_brackets", lambda t, l: {}), ("sentence_count_max", lambda t, l: {"n": 5})],
    [("json_format", lambda t, l: {}), ("include_keywords", lambda t, l: {"keywords": [t["main"]]})],
    [("no_commas", lambda t, l: {}), ("paragraph_count", lambda t, l: {"n": 2}),
     ("word_count_min", lambda t, l: {"n": 30})],
    [("starts_with", lambda t, l: {"phrase": _IFL_START[l]}),
     ("keyword_frequency", lambda t, l: {"word": t["main"], "at_least": 2})],
    [("sections", lambda t, l: {"n": 2}), ("include_keywords", lambda t, l: {"keywords": t["kw"][:1]})],
    [("uppercase", lambda t, l: {}), ("word_count_max", lambda t, l: {"n": 50})],
    [("quotation_wrapped", lambda t, l: {}), ("sentence_count_max", lambda t, l: {"n": 3}),
     ("language_uz", lambda t, l: {})],
]


def _ifl_instruction(ctype: str, params: Mapping[str, Any], lang: str) -> str:
    tmpl = _IFL_TEMPLATES[ctype][lang]
    kws = ", ".join(f"\"{k}\"" for k in params.get("keywords", []))
    return tmpl.format(n=params.get("n", ""), kws=kws, w=params.get("word", ""), k=params.get("at_least", ""),
                       p=params.get("phrase", ""))


def ifeval_lite_items(n: int = 30, seed: int = EVAL_SEED) -> List[Dict[str, Any]]:
    """The bundled ``ifeval_lite`` set: ``n`` (default 30) dicts ``{"prompt", "constraints", "lang"}``.

    Deterministic: topic ``i % 10``, recipe chosen by a seeded permutation, language alternating so
    every topic appears in both languages. ``language_uz`` is only added to Uzbek prompts, and Uzbek
    keyword constraints match substrings (``"substring": True``, see :func:`_check`).
    """
    rng = random.Random(f"ifeval_lite:{seed}")
    order = list(range(len(_IFL_RECIPES)))
    out: List[Dict[str, Any]] = []
    for i in range(int(n)):
        if i % len(order) == 0:
            rng.shuffle(order)
        lang = "uz" if (i + i // len(IFEVAL_LITE_TOPICS)) % 2 == 0 else "en"
        topic = IFEVAL_LITE_TOPICS[i % len(IFEVAL_LITE_TOPICS)][lang]
        recipe = _IFL_RECIPES[order[i % len(order)]]
        constraints, lines = [], []
        for ctype, build in recipe:
            if ctype == "language_uz" and lang != "uz":
                continue
            params = build(topic, lang)
            con = {"type": ctype, **params}
            if lang == "uz" and ctype in ("include_keywords", "keyword_frequency"):
                con["substring"] = True  # Uzbek words take suffixes ("sportni", "salomatlikka")
            constraints.append(con)
            lines.append(_ifl_instruction(ctype, params, lang))
        head = "Talablar:" if lang == "uz" else "Requirements:"
        prompt = topic["prompt"] + "\n" + head + "\n" + "\n".join(f"* {x}" for x in lines)
        out.append({"prompt": prompt, "constraints": constraints, "lang": lang})
    return out


def _dc(lang: str, prompt: str, *constraints: Dict[str, Any]) -> Dict[str, Any]:
    return {"lang": lang, "prompt": prompt, "constraints": list(constraints)}


def _kw(*words: str) -> Dict[str, Any]:
    return {"type": "include_keywords", "keywords": list(words)}


def _sub(*words: str) -> Dict[str, Any]:
    """Substring keyword requirement (inflected Uzbek words, possessives like "Lola's")."""
    return {"type": "include_keywords", "keywords": list(words), "substring": True}


#: ``detail_check``: 20 prompts hiding small requirements; each constraint is one requirement.
#: ``ends_with`` / ``starts_with`` marked ``loose`` ignore trailing punctuation / leading markup and
#: ``substring`` keywords match inside inflected words (see :func:`_check`).
DETAIL_CHECK_PROMPTS: List[Dict[str, Any]] = [
    _dc("en", "Write a short reminder to Aziz that the library book \"Dune\" must be returned on 2025-03-14. "
              "End the message with the exact word \"Thanks\".",
        _kw("Aziz"), _kw("Dune"), _kw("2025-03-14"), {"type": "ends_with", "phrase": "Thanks", "loose": True}),
    _dc("en", "Write a Python function named `area_of_circle` that takes `radius` and returns the area using `math.pi`.",
        _kw("def area_of_circle"), _kw("radius"), _kw("math.pi")),
    _dc("en", "Give me exactly 4 bullet points (each starting with \"- \") with packing tips for a hiking trip. "
              "One of them must mention water.",
        {"type": "bullet_count", "n": 4}, _kw("water")),
    _dc("en", "Draft a one-paragraph invitation to Lola's birthday party at Cafe Navro'z on Saturday at 18:30. "
              "Do not use the word \"cake\".",
        _sub("Lola"), _sub("Navro'z"), _kw("Saturday"), _kw("18:30"), {"type": "exclude_keywords", "keywords": ["cake"]}),
    _dc("en", "Write a product description for a blue backpack priced at $49 in under 60 words. Mention that it is "
              "waterproof.",
        _kw("backpack"), _kw("$49"), _kw("waterproof"), {"type": "word_count_max", "n": 60}),
    _dc("en", "Reply to the customer Mr. Karimov and apologise for the late delivery of order 5521. Start the reply "
              "with \"Dear Mr. Karimov\".",
        {"type": "starts_with", "phrase": "Dear Mr. Karimov", "loose": True}, _kw("5521")),
    _dc("en", "List the three primary colors in lowercase only, separated by commas, and write nothing else.",
        {"type": "lowercase"}, _kw("red", "blue", "yellow")),
    _dc("en", "Write a SQL query that selects the name and age columns from the students table where age is "
              "greater than 18.",
        _kw("students"), _kw("18"), _kw("name", "age")),
    _dc("en", "Summarize this in one sentence and keep the number 1,250 unchanged: The school library received "
              "1,250 new books this year thanks to donations from parents.",
        {"type": "sentence_count_max", "n": 1}, _kw("1,250")),
    _dc("en", "Write a haiku about autumn. Give it the title <<Autumn Leaves>> and do not use the word \"fall\".",
        _kw("<<Autumn Leaves>>"), {"type": "exclude_keywords", "keywords": ["fall"]}),
    _dc("uz", "Dilnozaga ertangi imtihon 2025-06-02 kuni soat 09:00 da 214-xonada bo'lishi haqida qisqa xabar "
              "yozing. Xabar \"Omad!\" so'zi bilan tugasin.",
        _sub("Dilnoza"), _kw("2025-06-02"), _kw("09:00"), _kw("214"),
        {"type": "ends_with", "phrase": "Omad", "loose": True}),
    _dc("uz", "`juft_sonlar` nomli Python funksiyasini yozing: u `sonlar` ro'yxatidan faqat juft sonlarni qaytarsin.",
        _kw("def juft_sonlar"), _kw("juft_sonlar(sonlar")),
    _dc("uz", "Bahor haqida aynan 3 ta band yozing, har bir band \"- \" bilan boshlansin va bittasida \"lola\" so'zi "
              "bo'lsin.",
        {"type": "bullet_count", "n": 3}, _sub("lola")),
    _dc("uz", "Akmal uchun tug'ilgan kun tabrigini yozing. Tabrikda uning 25 yoshga to'lgani aytilsin, lekin "
              "\"keksa\" so'zi ishlatilmasin.",
        _sub("Akmal"), _kw("25"), {"type": "exclude_keywords", "keywords": ["keksa"]}),
    _dc("uz", "Do'konga xat yozing: 3 kg olma va 2 litr sut buyurtma qiling. Xat \"Assalomu alaykum\" bilan "
              "boshlansin.",
        {"type": "starts_with", "phrase": "Assalomu alaykum", "loose": True}, _kw("3 kg"), _kw("2 litr")),
    _dc("uz", "Toshkentdan Samarqandgacha poyezd chiptasi 120000 so'm turishi haqida 50 so'zdan oshmaydigan e'lon "
              "yozing.",
        _sub("Samarqand"), _kw("120000"), {"type": "word_count_max", "n": 50}),
    _dc("uz", "Faqat kichik harflardan foydalanib, kuz faslini ikki gapda tasvirlang.",
        {"type": "lowercase"}, {"type": "sentence_count_max", "n": 2}),
    _dc("uz", "Jamoa yig'ilishi bayonnomasini yozing: sana 2025-01-20, ishtirokchilar Bekzod va Malika. Bayonnoma "
              "\"Yig'ilish yopildi\" degan so'zlar bilan tugasin.",
        _kw("2025-01-20"), _sub("Bekzod", "Malika"), {"type": "ends_with", "phrase": "Yig'ilish yopildi", "loose": True}),
    _dc("uz", "\"Python\" so'zini kamida 3 marta ishlatib, dasturlashni nima uchun o'rganish kerakligi haqida yozing.",
        {"type": "keyword_frequency", "word": "Python", "at_least": 3, "substring": True}),
    _dc("uz", "Ertangi ob-havo haqida radio uchun qisqa xabar tayyorlang: harorat +18°C, yomg'ir ehtimoli 40%. "
              "Vergul ishlatmang.",
        _kw("+18"), _kw("40%"), {"type": "no_commas"}),
]


#: ``project_build`` mini-projects: ``key``, ``module`` file, ``task`` (the request given to
#: ``llm.build_project``), ``hidden_tests`` (plain-assert test functions, never shown to the model)
#: and a ``reference`` implementation (used by the test-suite to validate the hidden tests).
PROJECT_SPECS: List[Dict[str, Any]] = [
    {
        "key": "bank_account",
        "module": "bank.py",
        "task": (
            "Create a Python module `bank.py` with a class `BankAccount(owner, balance=0)`.\n"
            "Requirements:\n"
            "1. `owner` and `balance` are public attributes; `history` is a list that starts empty.\n"
            "2. `deposit(amount)` adds money; if amount <= 0 raise ValueError. Append ('deposit', amount) to history.\n"
            "3. `withdraw(amount)` removes money; raise ValueError if amount <= 0 or amount > balance. "
            "Append ('withdraw', amount) to history.\n"
            "4. Both methods return the new balance.\n"
            "Write pytest tests in `test_bank.py`, run them and fix the code until they pass."),
        "hidden_tests": (
            "from bank import BankAccount\n\n"
            "def _raises(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return True\n"
            "    return False\n\n"
            "def test_initial_state():\n    a = BankAccount('Ali')\n    assert a.owner == 'Ali'\n"
            "    assert a.balance == 0\n    assert a.history == []\n\n"
            "def test_initial_balance():\n    assert BankAccount('Vali', 50).balance == 50\n\n"
            "def test_deposit_returns_balance():\n    a = BankAccount('Ali')\n    assert a.deposit(100) == 100\n"
            "    assert a.deposit(25) == 125\n    assert a.balance == 125\n\n"
            "def test_withdraw():\n    a = BankAccount('Ali', 100)\n    assert a.withdraw(30) == 70\n"
            "    assert a.balance == 70\n\n"
            "def test_invalid_amounts():\n    a = BankAccount('Ali', 10)\n    assert _raises(a.deposit, 0)\n"
            "    assert _raises(a.deposit, -5)\n    assert _raises(a.withdraw, 0)\n    assert _raises(a.withdraw, 11)\n"
            "    assert a.balance == 10\n\n"
            "def test_history():\n    a = BankAccount('Ali')\n    a.deposit(40)\n    a.withdraw(15)\n"
            "    assert a.history == [('deposit', 40), ('withdraw', 15)]\n"),
        "reference": (
            "class BankAccount:\n    def __init__(self, owner, balance=0):\n        self.owner = owner\n"
            "        self.balance = balance\n        self.history = []\n\n"
            "    def deposit(self, amount):\n        if amount <= 0:\n            raise ValueError('amount must be positive')\n"
            "        self.balance += amount\n        self.history.append(('deposit', amount))\n        return self.balance\n\n"
            "    def withdraw(self, amount):\n        if amount <= 0 or amount > self.balance:\n"
            "            raise ValueError('invalid amount')\n        self.balance -= amount\n"
            "        self.history.append(('withdraw', amount))\n        return self.balance\n"),
    },
    {
        "key": "shopping_cart",
        "module": "cart.py",
        "task": (
            "Create a Python module `cart.py` with a class `Cart` (an empty shopping cart).\n"
            "Requirements:\n"
            "1. `add(name, price, qty=1)`: add an item; adding an existing name increases its quantity. Raise "
            "ValueError if price < 0 or qty < 1.\n"
            "2. `remove(name)`: remove the item completely; raise KeyError if it is not in the cart.\n"
            "3. `total()`: sum of price * qty over all items, rounded to 2 decimals.\n"
            "4. `count()`: total quantity of all items.\n"
            "5. `apply_discount(percent)`: return the total after a percent discount rounded to 2 decimals "
            "(the cart is not changed); raise ValueError unless 0 <= percent <= 100.\n"
            "Write pytest tests in `test_cart.py`, run them and fix the code until they pass."),
        "hidden_tests": (
            "from cart import Cart\n\n"
            "def _raises(exc, fn, *args):\n    try:\n        fn(*args)\n    except exc:\n        return True\n"
            "    return False\n\n"
            "def test_empty():\n    c = Cart()\n    assert c.total() == 0\n    assert c.count() == 0\n\n"
            "def test_add_and_total():\n    c = Cart()\n    c.add('apple', 1.5, 4)\n    c.add('bread', 2.25)\n"
            "    assert c.total() == 8.25\n    assert c.count() == 5\n\n"
            "def test_add_same_item_accumulates():\n    c = Cart()\n    c.add('tea', 3.0)\n    c.add('tea', 3.0, 2)\n"
            "    assert c.count() == 3\n    assert c.total() == 9.0\n\n"
            "def test_remove():\n    c = Cart()\n    c.add('milk', 1.2, 2)\n    c.add('egg', 0.3, 10)\n"
            "    c.remove('milk')\n    assert c.count() == 10\n    assert c.total() == 3.0\n"
            "    assert _raises(KeyError, c.remove, 'milk')\n\n"
            "def test_invalid_add():\n    c = Cart()\n    assert _raises(ValueError, c.add, 'x', -1.0)\n"
            "    assert _raises(ValueError, c.add, 'x', 1.0, 0)\n\n"
            "def test_discount():\n    c = Cart()\n    c.add('book', 20.0, 2)\n    assert c.apply_discount(25) == 30.0\n"
            "    assert c.total() == 40.0\n    assert _raises(ValueError, c.apply_discount, 120)\n"),
        "reference": (
            "class Cart:\n    def __init__(self):\n        self.items = {}\n\n"
            "    def add(self, name, price, qty=1):\n        if price < 0 or qty < 1:\n"
            "            raise ValueError('invalid price or quantity')\n"
            "        old_price, old_qty = self.items.get(name, (price, 0))\n"
            "        self.items[name] = (price, old_qty + qty)\n\n"
            "    def remove(self, name):\n        if name not in self.items:\n            raise KeyError(name)\n"
            "        del self.items[name]\n\n"
            "    def total(self):\n        return round(sum(p * q for p, q in self.items.values()), 2)\n\n"
            "    def count(self):\n        return sum(q for _, q in self.items.values())\n\n"
            "    def apply_discount(self, percent):\n        if not 0 <= percent <= 100:\n"
            "            raise ValueError('percent must be between 0 and 100')\n"
            "        return round(self.total() * (100 - percent) / 100, 2)\n"),
    },
    {
        "key": "gradebook",
        "module": "grades.py",
        "task": (
            "Create a Python module `grades.py` with three functions.\n"
            "Requirements:\n"
            "1. `average(scores)`: arithmetic mean of a list of numbers, 0.0 for an empty list.\n"
            "2. `letter(score)`: 'A' for 90-100, 'B' for 80-89.99, 'C' for 70-79.99, 'D' for 60-69.99, 'F' below 60; "
            "raise ValueError if score < 0 or score > 100.\n"
            "3. `summary(scores)`: return a dict {'average': average(scores), 'letter': letter(average(scores)), "
            "'best': max(scores) or None for an empty list}.\n"
            "Write pytest tests in `test_grades.py`, run them and fix the code until they pass."),
        "hidden_tests": (
            "from grades import average, letter, summary\n\n"
            "def test_average():\n    assert average([80, 90, 100]) == 90\n    assert average([]) == 0.0\n\n"
            "def test_letter_boundaries():\n    assert letter(100) == 'A'\n    assert letter(90) == 'A'\n"
            "    assert letter(89.5) == 'B'\n    assert letter(70) == 'C'\n    assert letter(65) == 'D'\n"
            "    assert letter(59.9) == 'F'\n    assert letter(0) == 'F'\n\n"
            "def test_letter_invalid():\n    for bad in (-1, 101):\n        try:\n            letter(bad)\n"
            "        except ValueError:\n            continue\n        assert False, bad\n\n"
            "def test_summary():\n    assert summary([70, 80, 90]) == {'average': 80, 'letter': 'B', 'best': 90}\n\n"
            "def test_summary_empty():\n    assert summary([]) == {'average': 0.0, 'letter': 'F', 'best': None}\n"),
        "reference": (
            "def average(scores):\n    return sum(scores) / len(scores) if scores else 0.0\n\n"
            "def letter(score):\n    if score < 0 or score > 100:\n        raise ValueError('score out of range')\n"
            "    if score >= 90:\n        return 'A'\n    if score >= 80:\n        return 'B'\n    if score >= 70:\n"
            "        return 'C'\n    if score >= 60:\n        return 'D'\n    return 'F'\n\n"
            "def summary(scores):\n    avg = average(scores)\n"
            "    return {'average': avg, 'letter': letter(avg), 'best': max(scores) if scores else None}\n"),
    },
    {
        "key": "inventory",
        "module": "inventory.py",
        "task": (
            "Create a Python module `inventory.py` with a class `Inventory` (starts empty).\n"
            "Requirements:\n"
            "1. `add(item, qty)`: increase the stock of item; raise ValueError if qty <= 0.\n"
            "2. `remove(item, qty)`: decrease the stock; raise ValueError if qty <= 0 or there is not enough "
            "stock; an item whose stock reaches 0 is removed.\n"
            "3. `count(item)`: current stock (0 for unknown items).\n"
            "4. `items()`: sorted list of item names in stock.\n"
            "5. `low_stock(threshold=5)`: sorted list of items whose stock is below threshold.\n"
            "Write pytest tests in `test_inventory.py`, run them and fix the code until they pass."),
        "hidden_tests": (
            "from inventory import Inventory\n\n"
            "def _raises(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return True\n"
            "    return False\n\n"
            "def test_empty():\n    inv = Inventory()\n    assert inv.items() == []\n    assert inv.count('pen') == 0\n\n"
            "def test_add_and_count():\n    inv = Inventory()\n    inv.add('pen', 10)\n    inv.add('pen', 5)\n"
            "    inv.add('book', 2)\n    assert inv.count('pen') == 15\n    assert inv.items() == ['book', 'pen']\n\n"
            "def test_remove_to_zero():\n    inv = Inventory()\n    inv.add('cup', 3)\n    inv.remove('cup', 3)\n"
            "    assert inv.count('cup') == 0\n    assert inv.items() == []\n\n"
            "def test_remove_too_much():\n    inv = Inventory()\n    inv.add('cup', 2)\n    assert _raises(inv.remove, 'cup', 5)\n"
            "    assert inv.count('cup') == 2\n    assert _raises(inv.remove, 'ghost', 1)\n\n"
            "def test_invalid_quantities():\n    inv = Inventory()\n    assert _raises(inv.add, 'x', 0)\n"
            "    assert _raises(inv.add, 'x', -2)\n    inv.add('x', 1)\n    assert _raises(inv.remove, 'x', 0)\n\n"
            "def test_low_stock():\n    inv = Inventory()\n    inv.add('a', 1)\n    inv.add('b', 10)\n    inv.add('c', 4)\n"
            "    assert inv.low_stock() == ['a', 'c']\n    assert inv.low_stock(2) == ['a']\n"),
        "reference": (
            "class Inventory:\n    def __init__(self):\n        self.stock = {}\n\n"
            "    def add(self, item, qty):\n        if qty <= 0:\n            raise ValueError('qty must be positive')\n"
            "        self.stock[item] = self.stock.get(item, 0) + qty\n\n"
            "    def remove(self, item, qty):\n        if qty <= 0 or self.stock.get(item, 0) < qty:\n"
            "            raise ValueError('not enough stock')\n        self.stock[item] -= qty\n"
            "        if self.stock[item] == 0:\n            del self.stock[item]\n\n"
            "    def count(self, item):\n        return self.stock.get(item, 0)\n\n"
            "    def items(self):\n        return sorted(self.stock)\n\n"
            "    def low_stock(self, threshold=5):\n        return sorted(i for i, q in self.stock.items() if q < threshold)\n"),
    },
]


# ------------------------------------------------------------------------------------------- plan_bench

_PLAN_TASK_LETTERS = "ABCDEF"
_PLAN_MEETINGS = "KLMN"
_PLAN_PEOPLE = ("Aziz", "Bobur", "Dilnoza")


def _plan_topo(rng: random.Random, lang: str) -> Dict[str, Any]:
    n = rng.randint(5, 6)
    names = list(_PLAN_TASK_LETTERS[:n])
    while True:
        order = names[:]
        rng.shuffle(order)
        deps: Dict[str, List[str]] = {t: [] for t in names}
        for i, t in enumerate(order[1:], start=1):
            k = rng.randint(1, min(2, i)) if rng.random() < 0.8 else 0
            deps[t] = sorted(rng.sample(order[:i], k))
        # the alphabetical order must NOT be valid (otherwise the problem is trivial)
        pos = {t: i for i, t in enumerate(names)}
        if any(pos[d] > pos[t] for t in names for d in deps[t]) and sum(len(v) for v in deps.values()) >= n - 1:
            break
    rules = []
    for t in names:
        if deps[t]:
            if lang == "uz":
                rules.append(f"- {t} vazifasidan oldin {' va '.join(deps[t])} bajarilishi kerak")
            else:
                rules.append(f"- {t} requires {' and '.join(deps[t])} to be finished first")
    if lang == "uz":
        prompt = (f"Loyihada {n} ta vazifa bor: {', '.join(names)}. Bog'liqliklar:\n" + "\n".join(rules) +
                  "\nHar bir vazifa o'ziga kerakli vazifalardan keyin keladigan bajarish tartibini toping. "
                  f"Oxirgi qatorda javobni \"Javob: X, Y, ...\" ko'rinishida yozing (barcha {n} ta harf).")
    else:
        prompt = (f"A project has {n} tasks: {', '.join(names)}. Dependencies:\n" + "\n".join(rules) +
                  "\nFind an order in which every task comes after all the tasks it requires. "
                  f"Write the final answer on the last line as \"Answer: X, Y, ...\" (all {n} letters).")
    return {"kind": "order", "lang": lang, "prompt": prompt, "tasks": names, "deps": deps, "example": order}


def _plan_schedule(rng: random.Random, lang: str) -> Dict[str, Any]:
    k = rng.randint(3, 4)
    names = list(_PLAN_MEETINGS[:k])
    seq = names[:]
    rng.shuffle(seq)
    t = 9
    meetings: Dict[str, Dict[str, int]] = {}
    for m in seq:
        t += rng.randint(0, 1)
        dur = rng.randint(1, 2)
        meetings[m] = {"duration": dur, "earliest": max(8, t - rng.randint(0, 2)),
                       "latest_end": min(18, t + dur + rng.randint(0, 2)), "_start": t}
        t += dur
    lines = []
    for m in names:
        d = meetings[m]
        if lang == "uz":
            lines.append(f"- {m}: {d['duration']} soat, soat {d['earliest']}:00 dan oldin boshlanmasin va soat "
                         f"{d['latest_end']}:00 gacha tugasin")
        else:
            lines.append(f"- {m}: {d['duration']} hour(s), must not start before {d['earliest']}:00 and must end by "
                         f"{d['latest_end']}:00")
    example = {m: meetings[m].pop("_start") for m in names}
    if lang == "uz":
        prompt = ("Bitta majlislar xonasi bor va unda bir vaqtda faqat bitta uchrashuv o'tishi mumkin. Quyidagi "
                  "uchrashuvlarni butun soatlarda, bir-biriga to'qnashmaydigan qilib rejalashtiring:\n" + "\n".join(lines) +
                  "\nOxirgi qatorda har bir uchrashuvning boshlanish soatini \"Javob: "
                  + ", ".join(f"{m}=..." for m in names) + "\" ko'rinishida yozing.")
    else:
        prompt = ("There is one meeting room and it can hold only one meeting at a time. Schedule these meetings "
                  "on whole hours without any overlap:\n" + "\n".join(lines) +
                  "\nWrite the start hour of every meeting on the last line as \"Answer: "
                  + ", ".join(f"{m}=..." for m in names) + "\".")
    return {"kind": "schedule", "lang": lang, "prompt": prompt, "meetings": meetings, "example": example}


def _plan_allocate(rng: random.Random, lang: str) -> Dict[str, Any]:
    z = rng.randint(5, 15)
    mult = rng.choice((2, 3))
    y = mult * z
    x = y + rng.randint(3, 25)
    total = x + y + z
    a, b, c = _PLAN_PEOPLE
    item_uz, item_en = rng.choice((("olma", "apples"), ("daftar", "notebooks"), ("kitob", "books")))
    if lang == "uz":
        prompt = (f"{total} ta {item_uz}ni {a}, {b} va {c} o'rtasida shunday taqsimlang:\n"
                  f"- {b} {c}dan aynan {mult} marta ko'p oladi;\n- {a} {b}dan ko'proq oladi;\n"
                  f"- {c} kamida 5 ta oladi;\n- hamma butun son oladi va barcha {total} tasi taqsimlanadi.\n"
                  f"Oxirgi qatorda javobni \"Javob: {a}=..., {b}=..., {c}=...\" ko'rinishida yozing.")
    else:
        prompt = (f"Split {total} {item_en} between {a}, {b} and {c} so that:\n"
                  f"- {b} gets exactly {mult} times as many as {c};\n- {a} gets more than {b};\n"
                  f"- {c} gets at least 5;\n- everyone gets a whole number and all {total} are handed out.\n"
                  f"Write the final answer on the last line as \"Answer: {a}=..., {b}=..., {c}=...\".")
    return {"kind": "allocate", "lang": lang, "prompt": prompt, "total": total, "mult": mult, "min_c": 5,
            "example": {a: x, b: y, c: z}}


_PLAN_CACHE: Dict[int, List[Dict[str, Any]]] = {}


def plan_bench_problems(seed: int = EVAL_SEED) -> List[Dict[str, Any]]:
    """The 20 bundled ``plan_bench`` problems (7 dependency orderings, 7 room schedules, 6 resource
    allocations; Uzbek / English alternating), generated deterministically from ``seed``. Each dict
    has ``kind``, ``lang``, ``prompt``, the data needed by :func:`check_plan_answer` and a valid
    ``example`` answer (always accepted by the checker)."""
    if seed not in _PLAN_CACHE:
        rng = random.Random(f"plan_bench:{seed}")
        makers = [_plan_topo] * 7 + [_plan_schedule] * 7 + [_plan_allocate] * 6
        probs = [make(rng, "uz" if i % 2 == 0 else "en") for i, make in enumerate(makers)]
        order = list(range(len(probs)))
        rng.shuffle(order)
        _PLAN_CACHE[seed] = [probs[i] for i in order]
    return [dict(p) for p in _PLAN_CACHE[seed]]


_ANSWER_LINE_RE = re.compile(r"^\s*(?:\*\*)?(?:final answer|answer|javob|yakuniy javob)\s*(?:\*\*)?\s*[:：]",
                             re.IGNORECASE | re.MULTILINE)


def _answer_text(response: str) -> str:
    """Content of the last ``Answer:`` / ``Javob:`` line (thinking removed); else the last non-empty
    line of the final answer."""
    final = V.strip_think(response or "")
    if _ANSWER_LINE_RE.search(final):
        return V.extract_final_answer(final)
    lines = [ln.strip() for ln in final.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _assignments(text: str, names: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in names:
        m = re.search(r"(?<![A-Za-z])" + re.escape(name) + r"\s*(?:=|:|-|—)\s*(\d{1,4})(?::(\d{2}))?", text)
        if m:
            out[name] = int(m.group(1)) + (int(m.group(2)) / 60.0 if m.group(2) else 0.0)
    return out


def check_plan_answer(problem: Mapping[str, Any], response: Any) -> Tuple[bool, str]:
    """Verify an answer for a :func:`plan_bench_problems` problem. ``response`` is the model reply
    (str; the last ``Answer:``/``Javob:`` line is parsed) or, for testing, the problem's
    ``example``. Returns ``(ok, reason)``."""
    kind = problem["kind"]
    if not isinstance(response, str):
        ex = response
        if kind == "order":
            response = "Answer: " + ", ".join(ex)
        else:
            response = "Answer: " + ", ".join(f"{k}={v}" for k, v in ex.items())
    ans = _answer_text(response)
    if kind == "order":
        tasks = list(problem["tasks"])
        seq = re.findall(r"(?<![A-Za-z])([A-F])(?![A-Za-z])", ans)
        if sorted(seq) != sorted(tasks):
            return False, f"answer must list each of {tasks} once, got {seq}"
        pos = {t: i for i, t in enumerate(seq)}
        for t, ds in problem["deps"].items():
            for d in ds:
                if pos[d] > pos[t]:
                    return False, f"{t} placed before its prerequisite {d}"
        return True, "ok"
    if kind == "schedule":
        meetings = problem["meetings"]
        got = _assignments(ans, meetings)
        if set(got) != set(meetings):
            return False, f"missing start times for {sorted(set(meetings) - set(got))}"
        spans = []
        for m, d in meetings.items():
            s, e = got[m], got[m] + d["duration"]
            if s < d["earliest"] or e > d["latest_end"]:
                return False, f"{m} at {s:g} violates its window {d['earliest']}-{d['latest_end']}"
            spans.append((s, e, m))
        spans.sort()
        for (s1, e1, m1), (s2, _, m2) in zip(spans, spans[1:]):
            if s2 < e1:
                return False, f"{m1} and {m2} overlap"
        return True, "ok"
    if kind == "allocate":
        a, b, c = _PLAN_PEOPLE
        got = _assignments(ans, _PLAN_PEOPLE)
        if set(got) != set(_PLAN_PEOPLE):
            return False, f"missing amounts for {sorted(set(_PLAN_PEOPLE) - set(got))}"
        if any(v != int(v) or v < 0 for v in got.values()):
            return False, "amounts must be non-negative whole numbers"
        if sum(got.values()) != problem["total"]:
            return False, f"amounts sum to {sum(got.values()):g}, not {problem['total']}"
        if got[b] != problem["mult"] * got[c]:
            return False, f"{b} must get {problem['mult']}x {c}"
        if not got[a] > got[b]:
            return False, f"{a} must get more than {b}"
        if got[c] < problem["min_c"]:
            return False, f"{c} must get at least {problem['min_c']}"
        return True, "ok"
    raise ValueError(f"unknown plan_bench kind {kind!r}")


# =================================================================================================
# LLM helpers
# =================================================================================================

def _tokenizer(llm: Any) -> Any:
    req = getattr(llm, "require_tokenizer", None)
    tok = req() if callable(req) else getattr(llm, "tokenizer", None)
    if tok is None:
        raise RuntimeError("train first or load a checkpoint: the model has no tokenizer")
    return tok


def _model(llm: Any) -> Any:
    req = getattr(llm, "require_model", None)
    model = req() if callable(req) else getattr(llm, "model", None)
    if model is None:
        raise RuntimeError("the LLM has no model")
    return model


def _device(model: Any) -> Any:
    dev = getattr(model, "device", None)
    if dev is not None and not callable(dev):
        return dev
    for p in model.parameters():
        return p.device
    import torch

    return torch.device("cpu")


def _autocast(llm: Any) -> Any:
    ac = getattr(llm, "autocast", None)
    return ac() if callable(ac) else contextlib.nullcontext()


def _max_len(llm: Any) -> int:
    return int(llm.config.model.max_seq_len)


def _model_vocab(llm: Any) -> int:
    return int(llm.config.model.vocab_size)


def _gen(max_new_tokens: int, code: bool = False, **over: Any) -> Dict[str, Any]:
    """Greedy, seeded generation settings for ``llm.chat`` / ``llm.generate``."""
    g: Dict[str, Any] = {"temperature": 0.0, "best_of": 1, "seed": EVAL_SEED, "max_new_tokens": int(max_new_tokens)}
    if code:
        g.update(repetition_penalty=1.0, no_repeat_ngram=0)
    g.update({k: v for k, v in over.items() if v is not None})
    return g


def _chat(llm: Any, message: str, max_new_tokens: int = 128, system: Optional[str] = None, tools: Any = None,
          code: bool = False, **over: Any) -> str:
    """One greedy chat reply (``llm.chat``)."""
    return str(llm.chat(message, system=system, tools=tools, **_gen(max_new_tokens, code=code, **over)))


def _prompt_ids(llm: Any, messages: Sequence[Mapping[str, Any]], tools: Any = None, system: Optional[str] = None,
                reserve: int = 64) -> List[int]:
    """Chat-template ids with ``<|assistant|>`` appended, left-truncated to leave ``reserve``
    tokens for the reply."""
    tok = _tokenizer(llm)
    sys_prompt = system if system is not None else str(llm.config.chat.system_prompt)
    ids, _ = tok.encode_chat(list(messages), tools=tools or None, add_generation_prompt=True, system=sys_prompt)
    budget = max(8, _max_len(llm) - max(1, min(int(reserve), _max_len(llm) // 2)))
    if len(ids) > budget:
        ids = [BOS_ID] + ids[-(budget - 1):]
    return list(ids)


def _generate_ids(llm: Any, ids: Sequence[int], max_new_tokens: int, stop_ids: Sequence[int] = (END_ID, EOS_ID),
                  banned_ids: Sequence[int] = ()) -> Any:
    """Greedy ``GenerationResult`` for raw prompt ids (via ``llm.generate_ids`` when available)."""
    max_new = max(1, min(int(max_new_tokens), _max_len(llm) - len(ids)))
    fn = getattr(llm, "generate_ids", None)
    if callable(fn):
        return fn(list(ids), max_new_tokens=max_new, temperature=0.0, n=1, seed=EVAL_SEED,
                  stop_ids=tuple(stop_ids), banned_ids=tuple(banned_ids))
    from moonlight.model.generation import generate

    c = llm.config.chat
    with _autocast(llm):
        return generate(_model(llm), list(ids), max_new_tokens=max_new, temperature=0.0, top_k=c.top_k, top_p=c.top_p,
                        repetition_penalty=c.repetition_penalty, no_repeat_ngram=c.no_repeat_ngram,
                        stop_ids=tuple(stop_ids), n=1, banned_ids=tuple(banned_ids), seed=EVAL_SEED)


def _banned(llm: Any, allow_tool_call: bool) -> List[int]:
    from moonlight.chat import banned_special_ids

    return banned_special_ids(_tokenizer(llm).vocab_size, _model_vocab(llm), allow_tool_call)


def _norm(text: str) -> str:
    """Lower-case and normalize Uzbek apostrophes (ʻʼ‘’`´ -> ')."""
    for ch in "ʻʼ‘’`´":
        text = text.replace(ch, "'")
    return text.lower()


def _clip(text: Any, n: int = 240) -> str:
    s = str(text)
    return s if len(s) <= n else s[:n] + "…"


def _skip(reason: str, **details: Any) -> TaskOutput:
    d = {"skipped": reason}
    d.update(details)
    return {"score": None, "n": 0, "details": d}


def _take(items: Sequence[Any], limit: Optional[int], default: int) -> List[Any]:
    n = int(default if limit is None else limit)
    return list(items[:max(0, n)])


def _subset(n_total: int, limit: Optional[int], seed: int = EVAL_SEED) -> List[int]:
    """Deterministic subset of ``range(n_total)``: everything when ``limit`` is None / large, else a
    seeded sample of ``limit`` indices in ascending order."""
    if limit is None or int(limit) >= n_total:
        return list(range(n_total))
    return sorted(random.Random(f"subset:{seed}:{n_total}").sample(range(n_total), int(limit)))


def _cache_dir(llm: Any) -> str:
    raw = str(getattr(getattr(llm.config, "data", None), "cache_dir", "~/.cache/moonlight") or "~/.cache/moonlight")
    return os.path.expanduser(os.path.expandvars(raw))


def _bench_rows(llm: Any, source: str, offline: bool, columns: Optional[Sequence[str]] = None
                ) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """All rows of a registered bench source (downloaded + cached), or ``(None, reason)``."""
    if offline:
        return None, "offline mode (task needs a dataset download)"
    try:
        from moonlight.data.download import is_offline_env, iter_source_rows

        if is_offline_env():
            return None, "offline mode (HF_HUB_OFFLINE / MOONLIGHT_OFFLINE)"
        rows = list(iter_source_rows(source, _cache_dir(llm), columns=columns))
    except Exception as e:  # network / HTTP / parse errors -> skip, never crash the suite
        return None, f"download of {source!r} failed: {type(e).__name__}: {_clip(e, 300)}"
    if not rows:
        return None, f"dataset {source!r} is empty"
    return rows, ""


_NUM_RE = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:[,  ]\d{3})+|\d+)(?:[.,]\d+)?")


def _numbers_in(text: str) -> List[float]:
    """All numbers in ``text`` (``1,234`` / ``1 234`` thousands separators, ``3.5`` / ``3,5`` decimals)."""
    out = []
    for m in _NUM_RE.finditer(text or ""):
        s = m.group(0)
        if re.fullmatch(r"-?\d{1,3}(?:[,  ]\d{3})+(?:\.\d+)?", s):
            s = re.sub(r"[,  ]", "", s)
        else:
            s = s.replace(",", ".")
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out


def _has_number(text: str, target: float, rel_tol: float = 1e-3, abs_tol: float = 1e-2) -> bool:
    return any(math.isclose(x, target, rel_tol=rel_tol, abs_tol=abs_tol) for x in _numbers_in(text))


def _check(response: str, constraint: Mapping[str, Any]) -> bool:
    """:func:`moonlight.data.verifiers.check` plus two benchmark-local options:

    * ``loose`` (``ends_with`` / ``starts_with``): ignore trailing punctuation / leading markdown;
    * ``substring`` (``include_keywords`` / ``keyword_frequency``): match keywords as substrings of
      the apostrophe-normalized, lower-cased answer instead of whole words -- needed for Uzbek,
      an agglutinative language (``"Samarqandgacha"`` contains ``"Samarqand"``).
    """
    c = dict(constraint)
    loose = bool(c.pop("loose", False))
    substring = bool(c.pop("substring", False))
    ctype = c.get("type")
    if substring and ctype in ("include_keywords", "keyword_frequency"):
        final = _norm(V.strip_think(response or ""))
        if ctype == "include_keywords":
            words = [_norm(str(k)) for k in (c.get("keywords") or []) if str(k).strip()]
            return bool(words) and all(w in final for w in words)
        word = _norm(str(c.get("word") or c.get("keyword") or ""))
        n = int(c.get("at_least", c.get("n", 1)))
        count = final.count(word) if word else 0
        return count < n if str(c.get("relation", "at least")).lower().startswith("less") else count >= n
    if loose and ctype in ("ends_with", "starts_with"):
        final = V.strip_think(response or "").strip()
        if ctype == "ends_with":
            final = final.rstrip(" \t\n.!?…\"'*)»")
        else:
            final = final.lstrip(" \t\n#*_\"'«>")
        return V.check(final, c)
    return V.check(response or "", c)


# =================================================================================================
# Log-likelihood scoring
# =================================================================================================

def sequence_logprobs(llm: Any, sequences: Sequence[Sequence[int]], starts: Sequence[int],
                      max_batch: int = 64) -> List[Tuple[float, int, bool]]:
    """Sum of log-probs of ``seq[start:]`` given ``seq[:start]`` for every sequence.

    Args:
        sequences: token-id lists, each ``len <= max_seq_len``.
        starts: per sequence, index of the first scored token (``>= 1``).
        max_batch: max sequences per forward pass (also bounded so the (B, T, V) float32 logits
            stay around 1 GiB).

    Returns ``[(sum_logprob, n_scored_tokens, greedy_match)]`` in input order; ``greedy_match`` is
    True when every scored token is the argmax prediction. Runs ``llm.model`` without cache in
    eval mode under ``torch.no_grad`` and the LLM's autocast; right padding with PAD (the model is
    causal, so padding never affects earlier positions).
    """
    import torch

    model = _model(llm)
    device = _device(model)
    max_len = _max_len(llm)
    n = len(sequences)
    out: List[Optional[Tuple[float, int, bool]]] = [None] * n
    for i, (seq, s) in enumerate(zip(sequences, starts)):
        if len(seq) > max_len:
            raise ValueError(f"sequence of {len(seq)} tokens exceeds max_seq_len {max_len}")
        if not 1 <= int(s) <= len(seq):
            raise ValueError(f"start must be in [1, len(seq)], got {s} for a sequence of {len(seq)} tokens")
        if int(s) == len(seq):
            out[i] = (0.0, 0, True)
    todo = sorted((i for i in range(n) if out[i] is None), key=lambda i: -len(sequences[i]))
    vocab = max(1, _model_vocab(llm))
    token_budget = max(max_len, (1 << 28) // vocab)  # B*T*V*4 bytes ~ 1 GiB
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad(), _autocast(llm):
            pos = 0
            while pos < len(todo):
                width = len(sequences[todo[pos]])
                bs = max(1, min(int(max_batch), token_budget // max(1, width)))
                idx = todo[pos:pos + bs]
                pos += len(idx)
                x = torch.full((len(idx), width), PAD_ID, dtype=torch.long)
                for r, j in enumerate(idx):
                    x[r, :len(sequences[j])] = torch.tensor(list(sequences[j]), dtype=torch.long)
                logits = model(x.to(device)).logits  # (B, T, V) float32
                for r, j in enumerate(idx):
                    seq = sequences[j]
                    s = int(starts[j])
                    tgt = torch.tensor(list(seq[s:]), dtype=torch.long, device=logits.device)
                    lp = torch.log_softmax(logits[r, s - 1:len(seq) - 1].float(), dim=-1)  # (n_tgt, V)
                    tok_lp = lp.gather(-1, tgt[:, None]).squeeze(-1)
                    greedy = bool((lp.argmax(dim=-1) == tgt).all().item())
                    out[j] = (float(tok_lp.sum().item()), int(tgt.numel()), greedy)
                del logits
    finally:
        model.train(was_training)
    return [o if o is not None else (0.0, 0, True) for o in out]


def _encode_pair(tok: Any, context: str, continuation: str, max_len: int) -> Tuple[List[int], int]:
    """``(ids, n_continuation_tokens)`` for a (context, continuation) pair. Trailing whitespace of the
    context moves to the continuation (the pre-tokenizer attaches spaces to the following word);
    over-long inputs are left-truncated (the context first), keeping >= 1 context token."""
    stripped = context.rstrip(" ")
    continuation = context[len(stripped):] + continuation
    ctx = tok.encode(stripped, add_bos=True)
    cont = tok.encode(continuation) if continuation else []
    cont = cont[:max(0, max_len - 1)]
    ids = ctx + cont
    if len(ids) > max_len:
        ids = ids[-max_len:]
    return ids, len(cont)


def loglikelihood_batch(llm: Any, requests: Sequence[Tuple[str, str]]) -> List[Tuple[float, int, bool]]:
    """Log-likelihood of each ``(context, continuation)``: ``[(sum_logprob, n_tokens, greedy)]``
    (see :func:`sequence_logprobs`). The context is encoded with ``<|bos|>``; no chat template."""
    tok = _tokenizer(llm)
    max_len = _max_len(llm)
    enc = [_encode_pair(tok, c, x, max_len) for c, x in requests]
    return sequence_logprobs(llm, [e[0] for e in enc], [len(e[0]) - e[1] for e in enc])


def loglikelihood(llm: Any, context: str, continuation: str, normalize: Any = False) -> float:
    """Sum of the log-probs of ``continuation``'s tokens given ``context`` (SPEC §9).

    ``normalize``: False (sum), True / ``"tokens"`` (mean per token), ``"chars"`` (per character)
    or ``"bytes"`` (per UTF-8 byte) -- the length-normalized variants.
    """
    s, n_tok, _ = loglikelihood_batch(llm, [(context, continuation)])[0]
    if not normalize:
        return s
    if normalize is True or normalize == "tokens":
        return s / max(1, n_tok)
    if normalize == "chars":
        return s / max(1, len(continuation))
    if normalize == "bytes":
        return s / max(1, len(continuation.encode("utf-8")))
    raise ValueError(f"normalize must be False, True, 'tokens', 'chars' or 'bytes', got {normalize!r}")


def _mc_eval(llm: Any, items: Sequence[Mapping[str, Any]], metric: str = "acc_norm") -> TaskOutput:
    """Multiple choice by log-likelihood. ``items``: ``{"context", "choices" (continuations),
    "gold" (index)}``. Score = ``metric`` (``acc`` or ``acc_norm``); both are reported."""
    reqs, owners = [], []
    for k, it in enumerate(items):
        for ch in it["choices"]:
            reqs.append((it["context"], ch))
            owners.append(k)
    lls = loglikelihood_batch(llm, reqs)
    per: List[List[Tuple[float, int]]] = [[] for _ in items]
    for (s, _, _), k, (_, ch) in zip(lls, owners, reqs):
        per[k].append((s, max(1, len(ch.encode("utf-8")))))
    acc = acc_norm = 0
    samples = []
    for it, scores in zip(items, per):
        pred = max(range(len(scores)), key=lambda c: scores[c][0])
        pred_n = max(range(len(scores)), key=lambda c: scores[c][0] / scores[c][1])
        acc += int(pred == it["gold"])
        acc_norm += int(pred_n == it["gold"])
        if len(samples) < 3:
            samples.append({"context": _clip(it["context"], 160), "gold": it["gold"], "pred": pred, "pred_norm": pred_n})
    n = len(items)
    res = {"acc": acc / max(1, n), "acc_norm": acc_norm / max(1, n)}
    return {"score": res[metric], "n": n, "details": {**res, "metric": metric, "chance": _chance(items),
                                                       "samples": samples}}


def _chance(items: Sequence[Mapping[str, Any]]) -> float:
    return sum(1.0 / max(1, len(it["choices"])) for it in items) / max(1, len(items))


# =================================================================================================
# Chat tasks (every suite)
# =================================================================================================

_SPECIAL_TEXT_RE = re.compile(r"<\|[A-Za-z_/0-9]{1,24}\|>")
_ALLOWED_INLINE_IDS = frozenset({THINK_ID, THINK_END_ID, CODE_ID})


def task_chat_format(llm: Any, limit: Optional[int], offline: bool, max_new_tokens: int = 96, **_: Any) -> TaskOutput:
    """Chat-template sanity: the reply must (1) stop with ``<|end|>`` within ``max_new_tokens``,
    (2) contain no structural special token (e.g. ``<|eos|>``, ``<|user|>``) and no special-token
    text such as a literal ``"<|user|>"``, and (3) have non-empty visible text.

    Generation uses the chat template with only ``<|end|>`` as stop id and the same banned ids as
    ``ChatSession`` (``<|eos|>`` is allowed, so a model that ends with ``<|eos|>`` fails (1)/(2)).
    """
    from moonlight.chat import split_reply

    tok = _tokenizer(llm)
    prompts = _take(CHAT_FORMAT_PROMPTS, limit, 20)
    banned = _banned(llm, allow_tool_call=False)
    ok = finished_n = leak_n = empty_n = 0
    samples = []
    for p in prompts:
        ids = _prompt_ids(llm, [{"role": "user", "content": p}], reserve=max_new_tokens)
        res = _generate_ids(llm, ids, max_new_tokens, stop_ids=(END_ID,), banned_ids=banned)
        tokens = [int(t) for t in res.tokens]
        finished = bool(res.finished[0]) if res.finished else False
        id_leak = [t for t in tokens if t < NUM_SPECIAL_TOKENS and t not in _ALLOWED_INLINE_IDS]
        _, visible = split_reply(tok, [t for t in tokens if t not in id_leak])
        text_leak = bool(_SPECIAL_TEXT_RE.search(tok.decode(tokens, skip_special=True)))
        nonempty = bool(re.search(r"\w", visible))
        good = finished and not id_leak and not text_leak and nonempty
        ok += good
        finished_n += finished
        leak_n += bool(id_leak) or text_leak
        empty_n += not nonempty
        if len(samples) < 5:
            samples.append({"prompt": p, "reply": _clip(visible), "finished": finished,
                            "leak": bool(id_leak) or text_leak, "ok": good})
    n = len(prompts)
    return {"score": ok / max(1, n), "n": n,
            "details": {"finished_rate": finished_n / max(1, n), "leak_rate": leak_n / max(1, n),
                        "empty_rate": empty_n / max(1, n), "samples": samples}}


_GREETING_RE = re.compile(r"(?<![\w'])(?:" + "|".join(re.escape(k) for k in GREETING_KEYWORDS) + r")",
                          re.IGNORECASE)
_HI_EXACT = {"hi", "hey"}


def chat_sanity_ok(reply: str, lenient: bool = False) -> Tuple[bool, str]:
    """The ``chat_sanity`` rule. A reply passes when

    1. it contains a greeting / courtesy keyword from :data:`GREETING_KEYWORDS`, matched as a word
       PREFIX after apostrophe normalization (``"yordam"`` matches ``"yordamchi"``), except the short
       ``hi`` / ``hey`` which must be whole words; or
    2. ``lenient`` (tiny models, cc-001) and the reply is mostly Latin text: >= 80% of its letters
       are Latin and it has >= 2 words of >= 2 letters.

    Returns ``(passed, reason)`` with reason ``"keyword"``, ``"latin"`` or ``"no"``.
    """
    text = _norm(reply or "")
    for m in _GREETING_RE.finditer(text):
        word = re.match(r"[\w']+", text[m.start():])
        stem = m.group(0).lower()
        if stem in _HI_EXACT and (word is None or word.group(0) != stem):
            continue
        return True, "keyword"
    if lenient:
        letters = [ch for ch in text if ch.isalpha()]
        latin = sum(1 for ch in letters if "a" <= ch <= "z")
        words = [w for w in re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)*", text) if len(w) >= 2]
        if letters and latin / len(letters) >= 0.8 and len(words) >= 2:
            return True, "latin"
    return False, "no"


def task_chat_sanity(llm: Any, limit: Optional[int], offline: bool, lenient: bool = False,
                     max_new_tokens: int = 48, **_: Any) -> TaskOutput:
    """Greetings in Uzbek / English; pass rule: :func:`chat_sanity_ok` (``lenient`` for cc-001)."""
    prompts = _take(GREETING_PROMPTS, limit, 20)
    ok = 0
    reasons = {"keyword": 0, "latin": 0, "no": 0}
    samples = []
    for p in prompts:
        reply = _chat(llm, p, max_new_tokens)
        good, why = chat_sanity_ok(reply, lenient=bool(lenient))
        ok += good
        reasons[why] += 1
        if len(samples) < 5:
            samples.append({"prompt": p, "reply": _clip(reply), "ok": good, "rule": why})
    n = len(prompts)
    return {"score": ok / max(1, n), "n": n,
            "details": {"lenient": bool(lenient), "by_rule": reasons, "samples": samples}}


def task_identity(llm: Any, limit: Optional[int], offline: bool, max_new_tokens: int = 64, **_: Any) -> TaskOutput:
    """"Sen kimsan?" / "Who are you?" ...: the reply must contain ``config.chat.name``
    (case-insensitive, default "Moonlight")."""
    name = str(getattr(llm.config.chat, "name", "") or "Moonlight")
    prompts = _take(IDENTITY_PROMPTS, limit, 10)
    ok = 0
    samples = []
    for p in prompts:
        reply = _chat(llm, p, max_new_tokens)
        good = name.lower() in reply.lower()
        ok += good
        if len(samples) < 5:
            samples.append({"prompt": p, "reply": _clip(reply), "ok": good})
    n = len(prompts)
    return {"score": ok / max(1, n), "n": n, "details": {"name": name, "samples": samples}}


# =================================================================================================
# Language modelling
# =================================================================================================

def _asset_text(lang: str) -> str:
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "assets" / f"sample_{lang}.txt"
    return path.read_text(encoding="utf-8")


def task_perplexity(llm: Any, limit: Optional[int], offline: bool, languages: Sequence[str] = ("uz", "en"),
                    stride: Optional[int] = None, **_: Any) -> TaskOutput:
    """Bits-per-byte (lower is better) on the bundled ``moonlight/assets/sample_{uz,en}.txt``.

    Each file is tokenized once (``<|bos|>`` + text, at most ``limit`` tokens, default 8192) and
    scored with sliding windows of ``max_seq_len`` tokens advancing by ``stride`` (default
    ``max_seq_len // 2``); every token is scored exactly once, with up to ``max_seq_len - 1``
    tokens of context. ``bpb = total NLL / ln 2 / UTF-8 bytes of the scored text`` (byte count of
    the decoded scored tokens, i.e. after the tokenizer's apostrophe normalization). The score is
    the byte-weighted mean over ``languages``.
    """
    tok = _tokenizer(llm)
    max_len = _max_len(llm)
    step = int(stride) if stride else max(1, max_len // 2)
    step = max(1, min(step, max_len - 1))
    cap = int(limit) if limit else 8192
    if isinstance(languages, str):
        languages = [languages]
    per_lang: Dict[str, Dict[str, float]] = {}
    total_nll = 0.0
    total_bytes = 0
    total_tokens = 0
    for lang in languages:
        ids = tok.encode(_asset_text(lang))[:cap]
        if not ids:
            continue
        n_bytes = len(tok.decode(ids, skip_special=True).encode("utf-8"))
        seq = [BOS_ID] + ids
        windows: List[List[int]] = []
        starts: List[int] = []
        prev_end = 1  # next target index (into seq) still to be scored
        begin = 0
        while prev_end < len(seq):
            end = min(begin + max_len, len(seq))
            first = max(prev_end, begin + 1)
            if first < end:
                windows.append(seq[begin:end])
                starts.append(first - begin)
                prev_end = end
            begin += step
        res = sequence_logprobs(llm, windows, starts)
        nll = -sum(r[0] for r in res)
        n_tok = sum(r[1] for r in res)
        per_lang[lang] = {"bpb": nll / math.log(2) / max(1, n_bytes), "tokens": n_tok, "bytes": n_bytes,
                          "token_ppl": math.exp(min(50.0, nll / max(1, n_tok))), "windows": len(windows)}
        total_nll += nll
        total_bytes += n_bytes
        total_tokens += n_tok
    if not total_bytes:
        return _skip("no text to score")
    bpb = total_nll / math.log(2) / total_bytes
    return {"score": bpb, "n": total_tokens,
            "details": {"metric": "bits_per_byte", "per_language": per_lang, "stride": step,
                        "summary": ", ".join(f"{k} {v['bpb']:.3f} bpb" for k, v in per_lang.items())}}


# =================================================================================================
# Generated skills: arithmetic, knowledge, tools
# =================================================================================================

def _final_answer_target(text: str) -> str:
    m = re.search(r"(?:Javob|Answer)\s*:\s*(.+)\s*$", text.strip())
    return (m.group(1) if m else text).strip()


def task_arithmetic(llm: Any, limit: Optional[int], offline: bool, max_new_tokens: int = 48,
                    tools: bool = False, **_: Any) -> TaskOutput:
    """Exact +, -, ×, ÷ problems from ``synthetic.arithmetic_conversations(seed=EVAL_SEED)``
    (small operands, Uzbek / English). Correct iff the last number of the final answer equals the
    exact result. ``tools=True`` uses the big-operand calculator variants and enables tools."""
    from moonlight.data.synthetic import arithmetic_conversations

    n = int(limit or 40)
    convs = arithmetic_conversations(n, seed=EVAL_SEED, tool_frac=1.0 if tools else 0.0)
    ok = 0
    samples = []
    for conv in convs:
        question = next(m["content"] for m in conv if m["role"] == "user")
        target = _final_answer_target(conv[-1]["content"])
        reply = _chat(llm, question, max_new_tokens + (128 if tools else 0), tools=True if tools else None)
        good = V.reward_exact_number(reply, target) == 1.0
        ok += good
        if len(samples) < 5:
            samples.append({"question": question, "target": target, "reply": _clip(reply), "ok": good})
    return {"score": ok / max(1, len(convs)), "n": len(convs), "details": {"tools": bool(tools), "samples": samples}}


def task_knowledge_uz(llm: Any, limit: Optional[int], offline: bool, max_new_tokens: int = 64, **_: Any) -> TaskOutput:
    """Held-out Uzbek factual questions (:data:`KNOWLEDGE_UZ`); correct iff the normalized reply
    matches one of the accepted patterns."""
    items = _take(KNOWLEDGE_UZ, limit, len(KNOWLEDGE_UZ))
    ok = 0
    samples = []
    for q, patterns in items:
        reply = _chat(llm, q, max_new_tokens)
        text = _norm(reply)
        good = any(re.search(p, text) for p in patterns)
        ok += good
        if len(samples) < 5:
            samples.append({"question": q, "reply": _clip(reply), "ok": good})
    return {"score": ok / max(1, len(items)), "n": len(items), "details": {"samples": samples}}


def task_tool_format(llm: Any, limit: Optional[int], offline: bool, max_new_tokens: int = 96, **_: Any) -> TaskOutput:
    """First assistant turn of tool-use scenarios from ``synthetic.tool_call_conversations(seed=
    EVAL_SEED)`` (only scenarios whose reference answer starts with a tool call). Correct iff the
    reply starts with the ``<|tool_call|>`` token, the JSON parses (:func:`moonlight.tools.
    parse_tool_call`), names the expected tool and contains every required argument."""
    from moonlight.data.synthetic import tool_call_conversations
    from moonlight.tools import parse_tool_call

    n = int(limit or 30)
    tok = _tokenizer(llm)
    banned = _banned(llm, allow_tool_call=True)
    cases = []
    for conv in tool_call_conversations(max(4 * n, 16), seed=EVAL_SEED):
        sys_msg = conv[0] if conv and conv[0].get("role") == "system" else None
        users = [i for i, m in enumerate(conv) if m["role"] == "user"]
        if sys_msg is None or not sys_msg.get("tools") or not users:
            continue
        nxt = conv[users[0] + 1] if users[0] + 1 < len(conv) else None
        if not nxt or nxt.get("role") != "assistant" or not nxt.get("tool_calls"):
            continue
        cases.append((sys_msg, conv[users[0]], nxt["tool_calls"][0]))
        if len(cases) >= n:
            break
    ok = starts = parsed_n = name_ok = 0
    samples = []
    for sys_msg, user, expected in cases:
        schemas = sys_msg["tools"]
        ids = _prompt_ids(llm, [{"role": "user", "content": user["content"]}], tools=schemas,
                          system=str(sys_msg.get("content") or llm.config.chat.system_prompt), reserve=max_new_tokens)
        res = _generate_ids(llm, ids, max_new_tokens, stop_ids=(END_ID, EOS_ID), banned_ids=banned)
        tokens = [int(t) for t in res.tokens]
        is_call = bool(tokens) and tokens[0] == TOOL_CALL_ID
        call = parse_tool_call(tok.decode(tokens[1:], skip_special=True)) if is_call else None
        schema = next((s for s in schemas if s.get("name") == expected["name"]), {})
        required = list((schema.get("parameters") or {}).get("required") or [])
        good_name = call is not None and call["name"] == expected["name"]
        good = good_name and all(r in call["arguments"] for r in required)
        starts += is_call
        parsed_n += call is not None
        name_ok += good_name
        ok += good
        if len(samples) < 5:
            samples.append({"user": user["content"], "expected": expected["name"],
                            "output": _clip(tok.decode(tokens, skip_special=False)), "ok": good})
    n_cases = len(cases)
    if not n_cases:
        return _skip("no tool-call scenarios generated")
    return {"score": ok / n_cases, "n": n_cases,
            "details": {"starts_with_tool_call": starts / n_cases, "parsed": parsed_n / n_cases,
                        "right_tool": name_ok / n_cases, "samples": samples}}


_E2E_UNITS = [("km", "mi", "kilometr", "mil", "kilometers", "miles"), ("kg", "lb", "kilogramm", "funt", "kilograms", "pounds"),
              ("C", "F", "Selsiy", "Farengeyt", "degrees Celsius", "degrees Fahrenheit"),
              ("m", "ft", "metr", "fut", "meters", "feet"), ("mi", "km", "mil", "kilometr", "miles", "kilometers")]


def _e2e_cases(n: int, multi: bool, seed: int = EVAL_SEED) -> List[Dict[str, Any]]:
    """Held-out tool problems with python-computed answers. Single-step: a calculator product /
    quotient or a unit conversion; multi-step (``multi``): a conversion followed by arithmetic, so
    both ``unit_convert`` and ``calculator`` are needed."""
    from moonlight.data.synthetic import convert_units

    rng = random.Random(f"tool_e2e:{seed}:{int(multi)}")
    cases = []
    for i in range(int(n)):
        uz = i % 2 == 0
        if multi:
            fu, tu, fuz, tuz, fen, ten = rng.choice(_E2E_UNITS)
            v = rng.randint(3, 250)
            k = rng.randint(3, 12)
            target = convert_units(v, fu, tu) * k
            q = (f"{v} {fuz}ni {tuz}ga o'tkazing, so'ng natijani {k} ga ko'paytiring. Vositalardan foydalaning."
                 if uz else f"Convert {v} {fen} to {ten}, then multiply the result by {k}. Use the tools.")
            cases.append({"question": q, "target": target, "needs": ["unit_convert", "calculator"]})
        elif rng.random() < 0.6:
            a, b = rng.randint(1000, 99999), rng.randint(101, 9999)
            if rng.random() < 0.5:
                target, expr = a * b, f"{a} × {b}"
            else:
                a = a * b
                target, expr = a // b, f"{a} ÷ {b}"
            q = (f"{expr} nechaga teng? Kalkulyatordan foydalaning." if uz
                 else f"What is {expr}? Use the calculator.")
            cases.append({"question": q, "target": float(target), "needs": ["calculator"]})
        else:
            fu, tu, fuz, tuz, fen, ten = rng.choice(_E2E_UNITS)
            v = rng.randint(2, 500)
            target = convert_units(v, fu, tu)
            q = (f"{v} {fuz} necha {tuz} bo'ladi?" if uz else f"How many {ten} is {v} {fen}?")
            cases.append({"question": q, "target": target, "needs": ["unit_convert"]})
    return cases


def task_tool_e2e(llm: Any, limit: Optional[int], offline: bool, multi: bool = False, max_new_tokens: int = 160,
                  max_tool_rounds: int = 4, **_: Any) -> TaskOutput:
    """End-to-end tool use through the real chat tool loop (:class:`moonlight.chat.ChatSession`
    with the built-in ``calculator``, ``unit_convert`` and ``current_time`` tools). Correct iff
    every needed tool was actually called AND the final reply contains the right number (rel. tol
    1e-3). ``multi=True`` (agent multi-tool) asks for a conversion followed by arithmetic."""
    from moonlight.chat import ChatSession
    from moonlight.tools import default_registry

    cases = _e2e_cases(int(limit or 20), bool(multi))
    ok = used_n = right_n = 0
    samples = []
    for case in cases:
        session = ChatSession(llm, tools=default_registry(), max_tool_rounds=max_tool_rounds,
                              **_gen(max_new_tokens))
        reply = session.send(case["question"])
        called = [c["name"] for c in session.tool_log]
        used = all(t in called for t in case["needs"])
        right = _has_number(reply, case["target"])
        good = used and right
        ok += good
        used_n += used
        right_n += right
        if len(samples) < 5:
            samples.append({"question": case["question"], "target": round(case["target"], 4), "tools": called,
                            "reply": _clip(reply), "ok": good})
    n = len(cases)
    return {"score": ok / max(1, n), "n": n,
            "details": {"multi": bool(multi), "tool_use_rate": used_n / max(1, n),
                        "answer_rate": right_n / max(1, n), "samples": samples}}


# =================================================================================================
# Code
# =================================================================================================

def _defines(code: str, entry_point: str) -> bool:
    return re.search(r"^\s*(?:async\s+)?def\s+" + re.escape(entry_point) + r"\s*\(", code, re.MULTILINE) is not None


def task_code_lite(llm: Any, limit: Optional[int], offline: bool, max_new_tokens: int = 320, timeout: float = 5.0,
                   **_: Any) -> TaskOutput:
    """pass@1 (greedy) on the bundled :data:`CODE_LITE_PROBLEMS` (Uzbek / English prompts). The
    code is taken from fenced blocks (else the whole reply) and run with the asserts via
    :func:`moonlight.benchmarks.execution.check_program`."""
    probs = _take(CODE_LITE_PROBLEMS, limit, len(CODE_LITE_PROBLEMS))
    ok = 0
    reasons: Dict[str, int] = {}
    samples = []
    for i, p in enumerate(probs):
        if i % 2 == 0:
            prompt = (f"Write a Python function `{p['signature']}` that {p['en']}. "
                      "Return only the code in a ```python block.")
        else:
            prompt = (f"Python'da `{p['signature']}` funksiyasini yozing: u {p['uz']}. "
                      "Faqat kodni ```python bloki ichida qaytaring.")
        reply = _chat(llm, prompt, max_new_tokens, code=True)
        code = V.extract_code(reply)
        if not _defines(code, p["entry_point"]):
            good, info = False, "entry point not defined"
        else:
            good, info = check_program(code, p["tests"], p["entry_point"], timeout=timeout)
        ok += good
        key = "passed" if good else info.split(":")[0][:60]
        reasons[key] = reasons.get(key, 0) + 1
        if len(samples) < 3:
            samples.append({"entry_point": p["entry_point"], "reply": _clip(reply, 400), "ok": good, "info": info})
    n = len(probs)
    return {"score": ok / max(1, n), "n": n, "details": {"outcomes": reasons, "samples": samples}}


_STOP_SEQS = ("\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert ", "\n```")


def _truncate_completion(text: str) -> str:
    cut = len(text)
    for s in _STOP_SEQS:
        i = text.find(s)
        if i >= 0:
            cut = min(cut, i)
    return text[:cut]


def _complete_function(llm: Any, prompt_code: str, entry_point: str, max_new_tokens: int) -> Tuple[str, str]:
    """Ask the model to complete a function given its signature + docstring (HumanEval style).

    Base models (``kind == "base"``) continue ``prompt_code`` as plain text (truncated at the next
    top-level statement); chat models get an instruction and the code in a fence. Returns
    ``(solution_program, raw_reply)``; the solution always starts with ``prompt_code`` so imports and
    helper functions of the prompt are available.
    """
    if str(getattr(llm.config, "kind", "")) == "base":
        raw = str(llm.generate(prompt_code, **_gen(max_new_tokens, code=True)))
        return prompt_code + _truncate_completion(raw), raw
    msg = ("Complete the following Python function. Reply with the complete function in a ```python block.\n\n"
           f"```python\n{prompt_code.rstrip()}\n```")
    raw = _chat(llm, msg, max_new_tokens, code=True)
    code = V.extract_code(raw)
    if _defines(code, entry_point):
        return prompt_code.rstrip() + "\n    pass\n\n\n" + code, raw
    body = code if (code.startswith((" ", "\t")) or not code.strip()) else "\n".join(
        "    " + ln if ln.strip() else ln for ln in code.splitlines())
    return prompt_code + body, raw


def task_humaneval(llm: Any, limit: Optional[int], offline: bool, max_new_tokens: int = 384, timeout: float = 10.0,
                   **_: Any) -> TaskOutput:
    """HumanEval pass@1 (greedy) on a deterministic subset of ``limit`` problems (all 164 if None)."""
    rows, why = _bench_rows(llm, "humaneval", offline)
    if rows is None:
        return _skip(why)
    idx = _subset(len(rows), limit)
    ok = 0
    samples = []
    for i in idx:
        r = rows[i]
        solution, raw = _complete_function(llm, str(r["prompt"]), str(r["entry_point"]), max_new_tokens)
        good, info = check_program(solution, str(r["test"]), str(r["entry_point"]), timeout=timeout)
        ok += good
        if len(samples) < 3:
            samples.append({"task_id": r.get("task_id"), "reply": _clip(raw, 300), "ok": good, "info": info})
    return {"score": ok / max(1, len(idx)), "n": len(idx), "details": {"total_rows": len(rows), "samples": samples}}


_NOT_ENTRY = {"set", "sorted", "list", "tuple", "abs", "round", "math", "isclose", "int", "float", "str", "len",
              "dict", "frozenset", "all", "any", "sum", "max", "min", "type", "isinstance", "bool", "re"}


def _mbpp_entry(row: Mapping[str, Any]) -> Optional[str]:
    tests = list(row.get("test_list") or [])
    ref_defs = re.findall(r"^def\s+(\w+)\s*\(", str(row.get("code") or ""), re.MULTILINE)
    called = re.findall(r"(\w+)\s*\(", tests[0]) if tests else []
    for name in called:
        if name in ref_defs:
            return name
    for name in called:
        if name not in _NOT_ENTRY and name != "assert":
            return name
    return ref_defs[0] if ref_defs else None


def task_mbpp(llm: Any, limit: Optional[int], offline: bool, max_new_tokens: int = 384, timeout: float = 10.0,
              **_: Any) -> TaskOutput:
    """MBPP (sanitized test split) pass@1 (greedy): the prompt shows the task and its asserts; the
    model's function (from a code fence) must pass all asserts."""
    rows, why = _bench_rows(llm, "mbpp", offline)
    if rows is None:
        return _skip(why)
    idx = _subset(len(rows), limit)
    ok = 0
    samples = []
    for i in idx:
        r = rows[i]
        tests = [str(t) for t in (r.get("test_list") or [])]
        entry = _mbpp_entry(r)
        prompt = (f"{r['prompt']}\nYour code should pass these tests:\n" + "\n".join(tests) +
                  "\nWrite the Python function in a ```python block.")
        raw = _chat(llm, prompt, max_new_tokens, code=True)
        code = V.extract_code(raw)
        imports = "\n".join(str(x) for x in (r.get("test_imports") or []))
        if entry and not _defines(code, entry) and not re.search(r"^\s*" + re.escape(entry) + r"\s*=", code, re.M):
            good, info = False, "entry point not defined"
        else:
            good, info = check_program(code, tests, None, prefix=imports, timeout=timeout)
        ok += good
        if len(samples) < 3:
            samples.append({"task_id": r.get("task_id"), "reply": _clip(raw, 300), "ok": good, "info": info})
    return {"score": ok / max(1, len(idx)), "n": len(idx), "details": {"total_rows": len(rows), "samples": samples}}


# =================================================================================================
# Multiple choice (log-likelihood)
# =================================================================================================

def task_belebele_uz(llm: Any, limit: Optional[int], offline: bool, metric: str = "acc_norm", **_: Any) -> TaskOutput:
    """Belebele (``uzn_Latn``) reading comprehension: passage + question, the 4 answers are scored
    by log-likelihood (chance 0.25)."""
    rows, why = _bench_rows(llm, "belebele-uz", offline)
    if rows is None:
        return _skip(why)
    items = []
    for i in _subset(len(rows), limit):
        r = rows[i]
        try:
            gold = int(str(r["correct_answer_num"]).strip()) - 1
            choices = [" " + str(r[f"mc_answer{k}"]).strip() for k in range(1, 5)]
        except (KeyError, ValueError):
            continue
        ctx = f"{str(r['flores_passage']).strip()}\nSavol: {str(r['question']).strip()}\nJavob:"
        items.append({"context": ctx, "choices": choices, "gold": gold})
    if not items:
        return _skip("no usable belebele rows")
    return _mc_eval(llm, items, metric)


def _arc(llm: Any, source: str, limit: Optional[int], offline: bool, metric: str) -> TaskOutput:
    rows, why = _bench_rows(llm, source, offline)
    if rows is None:
        return _skip(why)
    items = []
    for i in _subset(len(rows), limit):
        r = rows[i]
        ch = r.get("choices") or {}
        texts, labels = list(ch.get("text") or []), [str(x) for x in (ch.get("label") or [])]
        key = str(r.get("answerKey", "")).strip()
        if not texts or key not in labels:
            continue
        items.append({"context": f"Question: {str(r['question']).strip()}\nAnswer:",
                      "choices": [" " + str(t).strip() for t in texts], "gold": labels.index(key)})
    if not items:
        return _skip(f"no usable {source} rows")
    return _mc_eval(llm, items, metric)


def task_arc_easy(llm: Any, limit: Optional[int], offline: bool, metric: str = "acc_norm", **_: Any) -> TaskOutput:
    """ARC-Easy (test) zero-shot multiple choice by log-likelihood."""
    return _arc(llm, "arc-easy", limit, offline, metric)


def task_arc_challenge(llm: Any, limit: Optional[int], offline: bool, metric: str = "acc_norm", **_: Any) -> TaskOutput:
    """ARC-Challenge (test) zero-shot multiple choice by log-likelihood."""
    return _arc(llm, "arc-challenge", limit, offline, metric)


def _hs_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def task_hellaswag(llm: Any, limit: Optional[int], offline: bool, metric: str = "acc_norm", **_: Any) -> TaskOutput:
    """HellaSwag (validation) sentence completion by log-likelihood (lm-eval-harness preprocessing)."""
    rows, why = _bench_rows(llm, "hellaswag", offline)
    if rows is None:
        return _skip(why)
    items = []
    for i in _subset(len(rows), limit):
        r = rows[i]
        label = str(r.get("label", "")).strip()
        endings = list(r.get("endings") or [])
        if not label.isdigit() or not endings:
            continue
        ctx = str(r.get("ctx_a", "")) + " " + str(r.get("ctx_b", "")).capitalize()
        query = _hs_clean(str(r.get("activity_label", "")) + ": " + ctx)
        items.append({"context": query, "choices": [" " + _hs_clean(str(e)) for e in endings], "gold": int(label)})
    if not items:
        return _skip("no usable hellaswag rows")
    return _mc_eval(llm, items, metric)


def task_mmlu(llm: Any, limit: Optional[int], offline: bool, metric: str = "acc", **_: Any) -> TaskOutput:
    """MMLU (all, test) zero-shot: the question with lettered options; the answer letters
    ``" A".." D"`` are scored by log-likelihood (score ``acc``)."""
    rows, why = _bench_rows(llm, "mmlu", offline)
    if rows is None:
        return _skip(why)
    letters = ["A", "B", "C", "D"]
    items = []
    subjects: Dict[str, int] = {}
    for i in _subset(len(rows), limit):
        r = rows[i]
        choices = list(r.get("choices") or [])
        if len(choices) != 4:
            continue
        subject = str(r.get("subject", "")).replace("_", " ")
        subjects[subject] = subjects.get(subject, 0) + 1
        ctx = (f"The following is a multiple choice question about {subject}.\n\n{str(r['question']).strip()}\n"
               + "\n".join(f"{letters[k]}. {str(c).strip()}" for k, c in enumerate(choices)) + "\nAnswer:")
        items.append({"context": ctx, "choices": [f" {x}" for x in letters], "gold": int(r["answer"])})
    if not items:
        return _skip("no usable mmlu rows")
    out = _mc_eval(llm, items, metric)
    out["details"]["n_subjects"] = len(subjects)
    return out


# =================================================================================================
# Reasoning / long context / vision
# =================================================================================================

def task_gsm8k(llm: Any, limit: Optional[int], offline: bool, max_new_tokens: int = 384, **_: Any) -> TaskOutput:
    """GSM8K (test) zero-shot: generate a step-by-step solution; the last number of the final answer
    (after ``<|/think|>`` / on the ``Answer:`` line when present) must equal ``#### N``."""
    rows, why = _bench_rows(llm, "gsm8k", offline)
    if rows is None:
        return _skip(why)
    idx = _subset(len(rows), limit)
    ok = 0
    samples = []
    for i in idx:
        r = rows[i]
        target = str(r["answer"]).split("####")[-1].strip().replace(",", "")
        prompt = ("Solve the following math problem step by step. Write the final answer on the last line as "
                  f"\"Answer: <number>\".\n\n{str(r['question']).strip()}")
        reply = _chat(llm, prompt, max_new_tokens)
        good = V.reward_exact_number(reply, target) == 1.0
        ok += good
        if len(samples) < 3:
            samples.append({"question": _clip(r["question"], 160), "target": target, "reply": _clip(reply, 300),
                            "ok": good})
    return {"score": ok / max(1, len(idx)), "n": len(idx), "details": {"samples": samples}}


_NEEDLE_FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
                  "Birds sing in the morning and rivers flow to the sea. ")


def task_needle(llm: Any, limit: Optional[int], offline: bool, fractions: Sequence[float] = (0.25, 0.5, 0.9),
                depths: Sequence[float] = (0.1, 0.5, 0.9), max_new_tokens: int = 24, **_: Any) -> TaskOutput:
    """Synthetic long-context passkey retrieval: a 5-digit pass key is hidden at relative ``depths``
    of filler text whose chat prompt fills ``fractions`` of ``max_seq_len``; the reply must contain
    the key. ``limit`` caps the number of (fraction, depth) trials."""
    tok = _tokenizer(llm)
    max_len = _max_len(llm)
    rng = random.Random(f"needle:{EVAL_SEED}")
    trials = [(f, d) for f in fractions for d in depths]
    if limit is not None:
        trials = trials[:int(limit)]
    unit = tok.encode(_NEEDLE_FILLER)
    head = "There is an important pass key hidden inside a lot of irrelevant text. Find it and remember it.\n\n"
    ok = 0
    per: Dict[str, List[int]] = {}
    samples = []
    for frac, depth in trials:
        key = str(rng.randint(10000, 99999))
        needle = f" The pass key is {key}. Remember it. {key} is the pass key. "
        tail = "\n\nWhat is the pass key? Answer with the number only."
        target = int(max_len * float(frac)) - max_new_tokens
        overhead = len(_prompt_ids(llm, [{"role": "user", "content": head + needle + tail}], reserve=0))
        n_units = max(1, (target - overhead) // max(1, len(unit)))
        while True:
            pos = int(round(float(depth) * n_units))
            text = _NEEDLE_FILLER * pos + needle + _NEEDLE_FILLER * (n_units - pos)
            msg = head + text + tail
            length = len(_prompt_ids(llm, [{"role": "user", "content": msg}], reserve=0))
            if length <= max(16, max_len - max_new_tokens - 1) or n_units <= 1:
                break
            n_units = max(1, n_units - max(1, (length - (max_len - max_new_tokens - 1)) // max(1, len(unit)) - 1))
        reply = _chat(llm, msg, max_new_tokens)
        good = key in re.sub(r"[\s,]", "", reply)
        ok += good
        per.setdefault(f"{frac:g}", []).append(int(good))
        if len(samples) < 3:
            samples.append({"fraction": frac, "depth": depth, "prompt_tokens": length, "key": key,
                            "reply": _clip(reply, 80), "ok": good})
    n = len(trials)
    return {"score": ok / max(1, n), "n": n,
            "details": {"by_fraction": {k: sum(v) / len(v) for k, v in per.items()}, "max_seq_len": max_len,
                        "samples": samples}}


_COLORS = [((220, 30, 30), ["red", "qizil"]), ((30, 160, 50), ["green", "yashil"]), ((30, 60, 220), ["blue", "ko'k"]),
           ((240, 220, 30), ["yellow", "sariq"]), ((10, 10, 10), ["black", "qora"]), ((250, 250, 250), ["white", "oq"]),
           ((245, 140, 20), ["orange", "to'q sariq", "apelsin"]), ((140, 40, 180), ["purple", "binafsha", "violet"])]


def task_vision_caption(llm: Any, limit: Optional[int], offline: bool, max_new_tokens: int = 48, **_: Any) -> TaskOutput:
    """Caption synthetic images (solid colour with a contrasting square, drawn with PIL) and check
    that the reply names the dominant colour (English or Uzbek). Skipped unless the model is
    multimodal and ``transformers`` + ``PIL`` are installed and the vision encoder loads."""
    if not bool(getattr(llm.config.multimodal, "enabled", False)):
        return _skip("model is not multimodal")
    if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("PIL") is None:
        return _skip("needs transformers and pillow")
    from PIL import Image, ImageDraw

    def make(rgb: Tuple[int, int, int]) -> Any:
        img = Image.new("RGB", (224, 224), rgb)
        inv = tuple(255 - c for c in rgb)
        ImageDraw.Draw(img).rectangle([88, 88, 136, 136], fill=inv)
        return img

    try:
        llm.encode_images([make(_COLORS[0][0])])
    except Exception as e:  # encoder download / load failure -> skip
        return _skip(f"vision encoder unavailable: {type(e).__name__}: {_clip(e, 200)}")
    cases = _take(_COLORS, limit, len(_COLORS))
    ok = 0
    samples = []
    for i, (rgb, words) in enumerate(cases):
        q = ("Bu rasmda asosan qaysi rang bor?" if i % 2 == 0 else "Describe this image. What is its main colour?")
        reply = str(llm.chat(q, images=[make(rgb)], **_gen(max_new_tokens)))
        text = _norm(reply)
        good = any(re.search(r"(?<![\w'])" + re.escape(w) + r"(?![\w'])", text) for w in words)
        ok += good
        if len(samples) < 4:
            samples.append({"colour": words[0], "reply": _clip(reply), "ok": good})
    n = len(cases)
    return {"score": ok / max(1, n), "n": n, "details": {"samples": samples}}


# =================================================================================================
# Speed
# =================================================================================================

def task_speed(llm: Any, limit: Optional[int], offline: bool, prompt_len: Optional[int] = None, **_: Any) -> TaskOutput:
    """Throughput: prefill tokens/s (one forward of a random ``prompt_len``-token prompt with cache)
    and greedy decode tokens/s (``limit`` steps, default 64, through the KV/retention cache), plus
    peak CUDA memory. Score = decode tokens/s; the task passes whenever it runs (no threshold)."""
    import torch

    model = _model(llm)
    device = _device(model)
    max_len = _max_len(llm)
    vocab = _model_vocab(llm)
    p_len = int(prompt_len) if prompt_len else min(512, max(8, max_len // 2))
    p_len = max(2, min(p_len, max_len - 2))
    steps = max(1, min(int(limit or 64), max_len - p_len - 1))
    gen = torch.Generator().manual_seed(EVAL_SEED)
    low = NUM_SPECIAL_TOKENS if vocab > NUM_SPECIAL_TOKENS + 1 else 0
    prompt = torch.randint(low, vocab, (1, p_len), generator=gen).to(device)
    cuda = getattr(device, "type", str(device)) == "cuda"

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize(device)

    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad(), _autocast(llm):
            model(prompt[:, :min(8, p_len)], use_cache=True)  # warm-up (kernels, allocator)
            sync()
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            out = model(prompt, use_cache=True)
            sync()
            prefill_s = time.perf_counter() - t0
            cache = out.cache
            nxt = out.logits[:, -1].argmax(dim=-1)
            t0 = time.perf_counter()
            for _step in range(steps):
                out = model(nxt[:, None], cache=cache, use_cache=True)
                nxt = out.logits[:, -1].argmax(dim=-1)
            sync()
            decode_s = time.perf_counter() - t0
    finally:
        model.train(was_training)
    details: Dict[str, Any] = {
        "prefill_tokens_per_s": p_len / max(prefill_s, 1e-9), "decode_tokens_per_s": steps / max(decode_s, 1e-9),
        "prompt_tokens": p_len, "decode_tokens": steps, "device": str(device),
        "dtype": str(getattr(model, "dtype", "")).replace("torch.", ""),
    }
    if cuda:
        details["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
    else:
        try:
            import resource
            import sys

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            details["peak_rss_gb"] = rss / (2**30 if sys.platform == "darwin" else 2**20)
        except (ImportError, OSError):
            pass
    details["summary"] = (f"prefill {details['prefill_tokens_per_s']:.0f} tok/s, decode "
                          f"{details['decode_tokens_per_s']:.1f} tok/s on {device}")
    return {"score": details["decode_tokens_per_s"], "n": steps, "details": details}


# =================================================================================================
# Instruction following, details, planning, projects (SPEC §13.7)
# =================================================================================================

def _constraint_eval(llm: Any, items: Sequence[Mapping[str, Any]], max_new_tokens: int) -> Dict[str, Any]:
    prompt_ok = 0
    inst_ok = inst_total = 0
    fractions: List[float] = []
    by_type: Dict[str, List[int]] = {}
    samples = []
    for it in items:
        reply = _chat(llm, str(it["prompt"]), max_new_tokens)
        results = [bool(_check(reply, c)) for c in it["constraints"]]
        for c, r in zip(it["constraints"], results):
            by_type.setdefault(str(c["type"]), []).append(int(r))
        all_ok = bool(results) and all(results)
        prompt_ok += all_ok
        inst_ok += sum(results)
        inst_total += len(results)
        fractions.append(sum(results) / max(1, len(results)))
        if len(samples) < 4:
            samples.append({"prompt": _clip(it["prompt"], 200), "reply": _clip(reply, 300), "passed": results})
    n = len(items)
    return {"prompt_level": prompt_ok / max(1, n), "instruction_level": inst_ok / max(1, inst_total),
            "mean_fraction": sum(fractions) / max(1, n), "n": n,
            "by_type": {k: sum(v) / len(v) for k, v in sorted(by_type.items())}, "samples": samples}


def task_ifeval_lite(llm: Any, limit: Optional[int], offline: bool, max_new_tokens: int = 256, **_: Any) -> TaskOutput:
    """Bundled 30-prompt instruction-following set (:func:`ifeval_lite_items`; held-out topics and
    instruction wording). Score = prompt-level strict accuracy (all constraints satisfied);
    instruction-level accuracy in the details."""
    items = _take(ifeval_lite_items(), limit, 30)
    r = _constraint_eval(llm, items, max_new_tokens)
    details = {k: v for k, v in r.items() if k != "n"}
    details["summary"] = f"prompt-level {r['prompt_level']:.3f}, instruction-level {r['instruction_level']:.3f}"
    return {"score": r["prompt_level"], "n": r["n"], "details": details}


def task_detail_check(llm: Any, limit: Optional[int], offline: bool, max_new_tokens: int = 256, **_: Any) -> TaskOutput:
    """Bundled 20 prompts with small hidden requirements (:data:`DETAIL_CHECK_PROMPTS`).
    Score = mean fraction of requirements satisfied per prompt."""
    items = _take(DETAIL_CHECK_PROMPTS, limit, len(DETAIL_CHECK_PROMPTS))
    r = _constraint_eval(llm, items, max_new_tokens)
    return {"score": r["mean_fraction"], "n": r["n"],
            "details": {"all_satisfied_rate": r["prompt_level"], "requirement_level": r["instruction_level"],
                        "by_type": r["by_type"], "samples": r["samples"]}}


def task_ifeval(llm: Any, limit: Optional[int], offline: bool, max_new_tokens: int = 512, **_: Any) -> TaskOutput:
    """Google IFEval: only prompts whose every instruction id is supported by
    :func:`moonlight.data.verifiers.ifeval_to_constraints` are used (deterministic subset of
    ``limit``). Score = prompt-level strict accuracy; instruction-level accuracy in details."""
    rows, why = _bench_rows(llm, "ifeval", offline)
    if rows is None:
        return _skip(why)
    usable = []
    for r in rows:
        cons = V.ifeval_to_constraints(list(r.get("instruction_id_list") or []), list(r.get("kwargs") or []))
        if cons:
            usable.append({"prompt": str(r.get("prompt", "")), "constraints": cons})
    if not usable:
        return _skip("no IFEval prompts with only supported instructions")
    items = [usable[i] for i in _subset(len(usable), limit)]
    r = _constraint_eval(llm, items, max_new_tokens)
    return {"score": r["prompt_level"], "n": r["n"],
            "details": {"instruction_level": r["instruction_level"], "supported_rows": len(usable),
                        "total_rows": len(rows), "by_type": r["by_type"], "samples": r["samples"],
                        "summary": f"prompt-level {r['prompt_level']:.3f}, instruction-level "
                                   f"{r['instruction_level']:.3f} on {r['n']} of {len(usable)} supported prompts"}}


def task_plan_bench(llm: Any, limit: Optional[int], offline: bool, max_new_tokens: int = 384,
                    use_plan_format: bool = True, **_: Any) -> TaskOutput:
    """Bundled 20 verifiable planning problems (:func:`plan_bench_problems`): dependency ordering,
    room scheduling with time windows and resource allocation. The model answers with
    ``llm.plan`` (Moonlight PLAN FORMAT) unless ``use_plan_format`` is false; the final
    ``Javob:`` / ``Answer:`` line is checked by :func:`check_plan_answer`."""
    probs = _take(plan_bench_problems(), limit, 20)
    ok = 0
    by_kind: Dict[str, List[int]] = {}
    samples = []
    plan = getattr(llm, "plan", None)
    for p in probs:
        if use_plan_format and callable(plan):
            reply = str(plan(p["prompt"], **_gen(max_new_tokens)))
        else:
            reply = _chat(llm, p["prompt"], max_new_tokens)
        good, info = check_plan_answer(p, reply)
        ok += good
        by_kind.setdefault(p["kind"], []).append(int(good))
        if len(samples) < 4:
            samples.append({"kind": p["kind"], "answer": _clip(_answer_text(reply), 120), "ok": good, "info": info})
    n = len(probs)
    return {"score": ok / max(1, n), "n": n,
            "details": {"by_kind": {k: sum(v) / len(v) for k, v in by_kind.items()}, "samples": samples}}


def task_project_build(llm: Any, limit: Optional[int], offline: bool, max_steps: int = 16,
                       max_new_tokens: int = 768, timeout: float = 30.0, **_: Any) -> TaskOutput:
    """Agentic project building: for each of the 4 bundled :data:`PROJECT_SPECS` the model runs
    ``llm.build_project(task, tmpdir, max_steps)`` (workspace tools: write/read files, run tests),
    then the HIDDEN tests (never shown to the model) are run against the produced module.
    Score = fraction of hidden tests passed over all projects; details hold per-project results.
    (Suites include this task from cc-005 on.)"""
    build = getattr(llm, "build_project", None)
    if not callable(build):
        return _skip("LLM has no build_project()")
    specs = _take(PROJECT_SPECS, limit, len(PROJECT_SPECS))
    passed = total = full = 0
    per: Dict[str, Any] = {}
    for spec in specs:
        with tempfile.TemporaryDirectory(prefix=f"moonlight_project_{spec['key']}_") as tmp:
            work = os.path.join(tmp, "project")
            os.makedirs(work, exist_ok=True)
            info: Dict[str, Any] = {}
            try:
                res = build(spec["task"], work, max_steps=int(max_steps), **_gen(max_new_tokens, code=True))
                info.update(steps=int(getattr(res, "steps", 0)), own_tests_passed=bool(getattr(res, "tests_passed", False)),
                            files=list(getattr(res, "files", []))[:20])
            except Exception as e:  # an agent crash only fails this project
                info["build_error"] = f"{type(e).__name__}: {_clip(e, 200)}"
            r = run_test_functions(spec["hidden_tests"], work, timeout=timeout)
        info.update(hidden_passed=r["passed"], hidden_total=r["total"], error=r["error"])
        passed += r["passed"]
        total += r["total"]
        full += int(r["total"] > 0 and r["passed"] == r["total"])
        per[spec["key"]] = info
    return {"score": passed / max(1, total), "n": len(specs),
            "details": {"tests_passed": passed, "tests_total": total, "projects_fully_passed": full,
                        "projects": per, "summary": f"{passed}/{total} hidden tests, {full}/{len(specs)} projects green"}}


# =================================================================================================
# Registry
# =================================================================================================

TaskFn = Callable[..., TaskOutput]

#: task name -> ``fn(llm, limit, offline, **params) -> {"score", "n", "details"}`` (SPEC §9).
TASKS: Dict[str, TaskFn] = {
    "chat_format": task_chat_format,
    "chat_sanity": task_chat_sanity,
    "identity": task_identity,
    "perplexity": task_perplexity,
    "arithmetic": task_arithmetic,
    "knowledge_uz": task_knowledge_uz,
    "tool_format": task_tool_format,
    "tool_e2e": task_tool_e2e,
    "code_lite": task_code_lite,
    "belebele_uz": task_belebele_uz,
    "arc_easy": task_arc_easy,
    "arc_challenge": task_arc_challenge,
    "hellaswag": task_hellaswag,
    "mmlu": task_mmlu,
    "gsm8k": task_gsm8k,
    "humaneval": task_humaneval,
    "mbpp": task_mbpp,
    "needle": task_needle,
    "vision_caption": task_vision_caption,
    "speed": task_speed,
    "ifeval": task_ifeval,
    "ifeval_lite": task_ifeval_lite,
    "detail_check": task_detail_check,
    "plan_bench": task_plan_bench,
    "project_build": task_project_build,
}
