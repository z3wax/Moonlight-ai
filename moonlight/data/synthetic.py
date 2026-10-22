"""Deterministic synthetic conversations (SPEC §6 ``synthetic.py`` and §13.3).

Every generator takes ``n`` (number of conversations) and a ``seed`` and uses its own
``random.Random`` seeded from ``"<generator>:<seed>"``, so the output is identical across runs and
machines. All generators return a list of conversations (list of SPEC §5 message dicts).

Conventions:

* Tool schemas of a conversation live in the leading system message under ``"tools"`` (see
  :mod:`moonlight.data.adapters`); assistant tool calls are ``{"role": "assistant", "content": "",
  "tool_calls": [{"name": ..., "arguments": {...}}]}`` and results are ``{"role": "tool", "name":
  ..., "content": <json string>}``. Tool names match the built-ins of :mod:`moonlight.tools`
  (``calculator``, ``current_time``, ``unit_convert``) plus ``weather`` (a clearly FAKE demo tool
  whose results say so), a ``search_wikipedia`` stub answering from the verified fact table below
  and the workspace tools (``write_file``, ``read_file``, ``list_dir``, ``run_tests``).
  Tool results are ``json.dumps`` of the value the tool returns. The schemas are defined here
  (static, so data is reproducible) following the SPEC §8 signatures.
* Reasoning uses ``<|think|>...<|/think|>`` inside assistant content (the tokenizer turns these into
  the real special tokens). Plans follow the Moonlight PLAN FORMAT (§13.3).
* Facts in :data:`FACTS` are double-checked; arithmetic is computed in python; code solutions,
  training problems and project trajectories are verified by :func:`verify_code_tasks`,
  :func:`verify_train_problems` and :func:`verify_projects` (used by the test-suite).

Pure python (no torch, no numpy).
"""

from __future__ import annotations

import json
import math
import random
import re
from fractions import Fraction
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import verifiers as V

__all__ = [
    "Conversation", "TOOL_SCHEMAS", "WORKSPACE_TOOL_SCHEMAS", "FACTS", "CODE_TASKS", "TRAIN_CODE_PROBLEMS",
    "PROJECTS", "identity_conversations", "greeting_conversations", "knowledge_conversations",
    "arithmetic_conversations", "tool_call_conversations", "code_conversations", "think_conversations",
    "plan_conversations", "constraint_conversations", "detail_conversations", "project_conversations",
    "grpo_prompt_set", "preference_pairs_synthetic", "build_synthetic", "synthetic_texts",
    "verify_code_tasks", "verify_train_problems", "verify_projects", "constraint_examples", "detail_examples",
]

Conversation = List[Dict[str, Any]]


def _default_system() -> str:
    from ..tokenizer import DEFAULT_SYSTEM_PROMPT  # torch-free
    return DEFAULT_SYSTEM_PROMPT


def _rng(name: str, seed: int) -> random.Random:
    return random.Random(f"{name}:{int(seed)}")


def _system(tools: Optional[Sequence[Mapping[str, Any]]] = None, content: Optional[str] = None) -> Dict[str, Any]:
    m: Dict[str, Any] = {"role": "system", "content": content or _default_system()}
    if tools:
        m["tools"] = [dict(t) for t in tools]
    return m


def _user(text: str, **extra: Any) -> Dict[str, Any]:
    m = {"role": "user", "content": text}
    m.update(extra)
    return m


def _asst(text: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": text}


def _call(name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": [{"name": name, "arguments": dict(arguments)}]}


def _tool(name: str, value: Any) -> Dict[str, Any]:
    """Tool-result message serialized exactly like ``moonlight.tools.ToolRegistry.call`` does at
    inference time: strings verbatim, everything else ``json.dumps(..., ensure_ascii=False)``."""
    content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return {"role": "tool", "name": name, "content": content}


def _calc_result(expression: str, fallback: Any) -> Any:
    """What the real ``calculator`` tool returns for ``expression`` (e.g. ``41`` for ``123 / 3``),
    so training data matches inference-time tool results; ``fallback`` if it cannot evaluate."""
    from moonlight.tools import safe_eval  # torch-free; lazy to keep this module's import light

    try:
        return safe_eval(expression)
    except Exception:
        return fallback


def _unit_result(value: float, from_unit: str, to_unit: str, result: float) -> Any:
    """What the real ``unit_convert`` tool returns (a dict with ``result``), falling back to the
    same shape built from the locally computed ``result``."""
    from moonlight.tools import unit_convert

    try:
        return unit_convert(value, from_unit, to_unit)
    except Exception:
        return {"value": value, "from_unit": from_unit, "to_unit": to_unit, "result": round(result, 10)}


def _num(x: Any, places: int = 6) -> str:
    """Pretty number: ints without decimals, floats rounded to ``places`` with trailing zeros removed."""
    if isinstance(x, Fraction):
        x = int(x) if x.denominator == 1 else float(x)
    if isinstance(x, float):
        if math.isfinite(x) and abs(x - round(x)) < 1e-9 and abs(x) < 1e15:
            return str(int(round(x)))
        s = f"{x:.{places}f}".rstrip("0").rstrip(".")
        return "0" if s in ("-0", "") else s
    return str(x)


def _pick(rng: random.Random, seq: Sequence[Any]) -> Any:
    return seq[rng.randrange(len(seq))]


def _distribute(n: int, k: int) -> List[int]:
    """Split ``n`` into ``k`` near-equal non-negative parts."""
    base, rem = divmod(max(0, int(n)), max(1, k))
    return [base + (1 if i < rem else 0) for i in range(k)]


# =============================================================================================== tools

def _schema(name: str, description: str, props: Dict[str, Any], required: Sequence[str]) -> Dict[str, Any]:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}


TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "calculator": _schema(
        "calculator", "Evaluate an arithmetic expression safely (+ - * / // % **, parentheses, sqrt, sin, "
        "cos, log, exp, pi, e).", {"expression": {"type": "string", "description": "e.g. '12 * (3 + 4)'"}},
        ["expression"]),
    "current_time": _schema(
        "current_time", "Current date and time in an IANA timezone.",
        {"timezone": {"type": "string", "description": "IANA timezone, e.g. 'Asia/Tashkent'", "default": "UTC"}},
        []),
    "unit_convert": _schema(
        "unit_convert", "Convert a value between units of length, mass or temperature.",
        {"value": {"type": "number"}, "from_unit": {"type": "string", "description": "e.g. km, m, mi, kg, lb, C, F"},
         "to_unit": {"type": "string"}}, ["value", "from_unit", "to_unit"]),
    "weather": _schema(
        "weather", "FAKE demo weather tool: returns made-up weather data for testing tool use (not real).",
        {"city": {"type": "string"}}, ["city"]),
    "search_wikipedia": _schema(
        "search_wikipedia", "Stub encyclopedia lookup: returns a short summary for a query (offline demo).",
        {"query": {"type": "string"}}, ["query"]),
}

WORKSPACE_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "write_file": _schema("write_file", "Write text content to a file inside the workspace (creates or overwrites).",
                          {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    "read_file": _schema("read_file", "Read a text file from the workspace.", {"path": {"type": "string"}}, ["path"]),
    "list_dir": _schema("list_dir", "List files in a workspace directory.",
                        {"path": {"type": "string", "default": "."}}, []),
    "run_tests": _schema("run_tests", "Run the test-suite (pytest -q) in the workspace and report the result.",
                         {"target": {"type": "string", "default": "."}}, []),
}


def _use_runtime_schemas() -> None:
    """Replace the static schemas of tools that exist at inference time with the exact schemas
    ``moonlight.tools`` renders into the ``<|tools|>`` block (built-ins and workspace tools), so the
    training prompts match what ChatSession / build_project show the model. The static copies
    above remain as a fallback if ``moonlight.tools`` cannot be imported."""
    try:
        from types import SimpleNamespace

        from moonlight.tools import builtin_tools, workspace_tools
    except Exception:  # pragma: no cover - moonlight.tools is torch-free and always importable
        return
    for t in builtin_tools():
        if t.name in TOOL_SCHEMAS:
            TOOL_SCHEMAS[t.name] = t.schema()
    # workspace_tools only reads ``allow_exec`` at construction; a stand-in avoids creating a directory
    for t in workspace_tools(SimpleNamespace(allow_exec=True)):  # type: ignore[arg-type]
        if t.name in WORKSPACE_TOOL_SCHEMAS:
            WORKSPACE_TOOL_SCHEMAS[t.name] = t.schema()


_use_runtime_schemas()


# =============================================================================================== identity

_ID_UZ: List[Tuple[str, str]] = [
    ("Sen kimsan?", "Men {name} — o'zbek va ingliz tillarida suhbatlasha oladigan sun'iy intellekt "
     "yordamchisiman. Savollarga javob beraman, matn yozaman, kod yozishda yordam beraman va kerak bo'lsa "
     "vositalardan (masalan, kalkulyatordan) foydalanaman."),
    ("O'zingni tanishtir.", "Albatta! Mening ismim {name}. Men Moonlight oilasiga mansub til modeliman: "
     "arxitekturam attention va retention qatlamlarini birlashtiradi. O'zbek va ingliz tillarida yozishim, "
     "kod yozishim va vositalarni chaqirishim mumkin."),
    ("Isming nima?", "Mening ismim {name}. Sizga qanday yordam bera olaman?"),
    ("Seni kim yaratgan?", "Meni Moonlight ochiq manbali loyihasi ishtirokchilari Moonlight kutubxonasi "
     "yordamida o'qitgan. Men {name} nomli sun'iy intellekt yordamchisiman."),
    ("Sen ChatGPTmisan?", "Yo'q, men ChatGPT emasman. Men {name} — Moonlight oilasidagi alohida til modeliman."),
    ("Sen insonmisan?", "Yo'q, men inson emasman. Men {name} nomli sun'iy intellekt yordamchisiman, "
     "his-tuyg'ularim yo'q, lekin sizga foydali bo'lishga harakat qilaman."),
    ("Qaysi tillarni bilasan?", "Men asosan o'zbek (lotin yozuvida) va ingliz tillarida yozaman, shuningdek "
     "Python kabi dasturlash tillarida kod yoza olaman."),
    ("Nimalar qila olasan?", "Men {name} sifatida savollarga javob beraman, matnlarni tushuntiraman va "
     "qisqartiraman, reja tuzaman, Python kodini yozaman va tekshiraman, hisob-kitoblar uchun esa "
     "kalkulyator kabi vositalardan foydalanaman."),
    ("Internetga ulanganmisan?", "Yo'q, menda o'zimdan internetga kirish imkoni yo'q. Agar suhbatga vositalar "
     "ulangan bo'lsa, ulardan foydalanaman; aks holda faqat o'qitilgan bilimlarim asosida javob beraman. "
     "Bilimlarim eskirgan yoki xato bo'lishi mumkin."),
    ("Sen qanday ishlaysan?", "Men {name} — matndagi navbatdagi tokenni bashorat qiladigan neyron tarmoqman. "
     "Arxitekturamda attention qatlamlari bilan birga retention qatlamlari ham bor; retention uzun matnlarda "
     "xotirani tejashga yordam beradi."),
    ("Xato qilishing mumkinmi?", "Ha, men xato qilishim mumkin. Muhim ma'lumotlarni, ayniqsa sana, raqam va "
     "tibbiy yoki huquqiy masalalarni ishonchli manbalardan tekshirib ko'ring."),
]

_ID_EN: List[Tuple[str, str]] = [
    ("Who are you?", "I'm {name}, an AI assistant that can chat in Uzbek and English, answer questions, "
     "help with writing and code, and use tools such as a calculator when needed."),
    ("Introduce yourself.", "Sure! My name is {name}. I'm a language model from the Moonlight family: my "
     "architecture combines attention and retention layers. I can write in Uzbek and English, write code "
     "and call tools."),
    ("What is your name?", "My name is {name}. How can I help you?"),
    ("Who created you?", "I was trained by contributors to the open-source Moonlight project using the "
     "Moonlight library. I'm an AI assistant called {name}."),
    ("Are you ChatGPT?", "No, I'm not ChatGPT. I'm {name}, a separate language model from the Moonlight family."),
    ("Are you a human?", "No, I'm not a human. I'm {name}, an AI assistant. I don't have feelings, but I try "
     "to be genuinely helpful."),
    ("Which languages do you speak?", "I mainly write in Uzbek (Latin script) and English, and I can also "
     "write code in programming languages like Python."),
    ("What can you do?", "As {name}, I can answer questions, explain and summarize text, make plans, write "
     "and review Python code, and use tools like a calculator for exact calculations."),
    ("Can you browse the internet?", "No, I can't access the internet on my own. If tools are connected to "
     "this conversation I can use them; otherwise I answer from what I learned during training, which may "
     "be outdated or wrong."),
    ("How do you work?", "I'm {name}, a neural network that predicts the next token of text. My "
     "architecture mixes attention layers with retention layers, which keep a compact state and save "
     "memory on long inputs."),
    ("Can you make mistakes?", "Yes, I can make mistakes. Please double-check important information, "
     "especially dates, numbers and medical or legal topics, with reliable sources."),
]


def identity_conversations(name: str = "Moonlight", n: int = 200, seed: int = 0) -> List[Conversation]:
    """Who-am-I conversations (Uzbek and English) mentioning ``name``; some are preceded by a greeting."""
    rng = _rng("identity", seed)
    out: List[Conversation] = []
    for i in range(int(n)):
        uz = i % 2 == 0
        q, a = _pick(rng, _ID_UZ if uz else _ID_EN)
        msgs: Conversation = []
        if rng.random() < 0.25:
            gq, ga = _pick(rng, _GREET_UZ if uz else _GREET_EN)
            msgs += [_user(gq), _asst(ga.format(name=name))]
        msgs += [_user(q), _asst(a.format(name=name))]
        out.append(msgs)
    return out


# =============================================================================================== greetings

_GREET_UZ: List[Tuple[str, str]] = [
    ("Salom!", "Assalomu alaykum! Sizga qanday yordam bera olaman?"),
    ("Assalomu alaykum", "Va alaykum assalom! Bugun sizga nimada yordam beray?"),
    ("Salom, qalaysan?", "Salom! Rahmat, yaxshiman. O'zingiz qalaysiz? Biror savolingiz bo'lsa, bemalol so'rang."),
    ("Hayrli tong!", "Hayrli tong! Kuningiz xayrli o'tsin. Qanday yordam kerak?"),
    ("Xayrli kech!", "Xayrli kech! Sizga qanday yordam bera olaman?"),
    ("Rahmat!", "Arzimaydi! Yana savollaringiz bo'lsa, bemalol murojaat qiling."),
    ("Katta rahmat, juda yordam berding.", "Xursandman! Yana kerak bo'lsa, men shu yerdaman."),
    ("Xayr!", "Xayr! Sizga omad tilayman, yana kutib qolaman."),
    ("Ishlaring qalay?", "Rahmat, hammasi joyida! Siz bilan suhbatlashishga tayyorman. Nima haqida gaplashamiz?"),
    ("Menga yordam bera olasanmi?", "Albatta! Savolingizni yoki vazifani yozing, birgalikda hal qilamiz."),
    ("Zerikdim.", "Unda qiziqarli narsa qilaylik! Sizga topishmoq aytishim, qiziqarli fakt aytib berishim "
     "yoki kichik dastur yozishga yordam berishim mumkin. Qaysi biri yoqadi?"),
]
_GREET_EN: List[Tuple[str, str]] = [
    ("Hi!", "Hello! How can I help you today?"),
    ("Hello there", "Hi! What can I do for you?"),
    ("Hey, how are you?", "I'm doing well, thanks for asking! How about you? Let me know if there's "
     "anything I can help with."),
    ("Good morning!", "Good morning! I hope you have a great day. What can I help you with?"),
    ("Good evening!", "Good evening! How can I help?"),
    ("Thanks!", "You're welcome! Feel free to ask if you have more questions."),
    ("Thank you so much, that helped a lot.", "I'm glad it helped! I'm here if you need anything else."),
    ("Bye!", "Goodbye! Have a great day."),
    ("Can you help me?", "Of course! Tell me what you need and we'll work through it together."),
    ("I'm bored.", "Let's do something fun! I could tell you a riddle, share an interesting fact, or help "
     "you write a small program. Which sounds good?"),
]


def greeting_conversations(n: int = 200, seed: int = 0) -> List[Conversation]:
    """Greetings, thanks and goodbyes (Uzbek / English); about a third are two-turn exchanges."""
    rng = _rng("greetings", seed)
    out: List[Conversation] = []
    for i in range(int(n)):
        pool = _GREET_UZ if i % 2 == 0 else _GREET_EN
        turns = 2 if rng.random() < 0.33 else 1
        msgs: Conversation = []
        for q, a in rng.sample(pool, turns):
            msgs += [_user(q), _asst(a)]
        out.append(msgs)
    return out


# =============================================================================================== knowledge

def _fact(key: str, uz_q: Sequence[str], uz_a: str, en_q: Sequence[str], en_a: str) -> Dict[str, Any]:
    return {"key": key, "uz_q": list(uz_q), "uz_a": uz_a, "en_q": list(en_q), "en_a": en_a}


#: Verified facts (each checked against standard references). Keys are used by the wiki stub.
FACTS: List[Dict[str, Any]] = [
    _fact("toshkent", ["O'zbekistonning poytaxti qaysi shahar?", "O'zbekiston poytaxti qayer?"],
          "O'zbekistonning poytaxti — Toshkent shahri. U mamlakatdagi eng yirik shahar hamdir.",
          ["What is the capital of Uzbekistan?", "Which city is Uzbekistan's capital?"],
          "The capital of Uzbekistan is Tashkent, which is also the country's largest city."),
    _fact("mustaqillik", ["O'zbekiston qachon mustaqil bo'lgan?", "Mustaqillik kuni qachon nishonlanadi?"],
          "O'zbekiston mustaqilligi 1991-yil 31-avgustda e'lon qilingan. Mustaqillik kuni har yili "
          "1-sentabrda nishonlanadi.",
          ["When did Uzbekistan become independent?", "When is Uzbekistan's Independence Day?"],
          "Uzbekistan declared independence on 31 August 1991. Independence Day is celebrated every year "
          "on 1 September."),
    _fact("som", ["O'zbekistonning milliy valyutasi nima?", "O'zbekistonda qaysi pul birligi ishlatiladi?"],
          "O'zbekistonning milliy valyutasi — so'm.",
          ["What is the currency of Uzbekistan?", "Which money is used in Uzbekistan?"],
          "The national currency of Uzbekistan is the Uzbek som (so'm)."),
    _fact("viloyatlar", ["O'zbekiston qanday ma'muriy birliklardan iborat?", "O'zbekistonda nechta viloyat bor?"],
          "O'zbekiston 12 ta viloyat, Qoraqalpog'iston Respublikasi va Toshkent shahridan iborat.",
          ["How many regions does Uzbekistan have?", "What are the administrative divisions of Uzbekistan?"],
          "Uzbekistan consists of 12 regions (viloyats), the Republic of Karakalpakstan and the city of Tashkent."),
    _fact("daryolar", ["O'zbekistonning eng katta daryolari qaysilar?", "Amudaryo va Sirdaryo haqida ayt."],
          "O'zbekistonning eng yirik daryolari — Amudaryo va Sirdaryo. Ikkalasi ham tarixan Orol dengiziga quyilgan.",
          ["What are the main rivers of Uzbekistan?", "Tell me about the Amu Darya and Syr Darya."],
          "The main rivers of Uzbekistan are the Amu Darya and the Syr Darya; historically both flowed into the Aral Sea."),
    _fact("navoiy", ["Alisher Navoiy kim bo'lgan?", "Alisher Navoiy qachon yashagan?"],
          "Alisher Navoiy (1441–1501) — buyuk o'zbek shoiri va mutafakkiri. U besh dostondan iborat "
          "\"Xamsa\" asarining muallifi.",
          ["Who was Alisher Navoi?", "When did Alisher Navoi live?"],
          "Alisher Navoi (1441–1501) was a great Uzbek poet and thinker, the author of the \"Khamsa\", a "
          "collection of five epic poems."),
    _fact("temur", ["Amir Temur qachon yashagan?", "Amir Temur kim?"],
          "Amir Temur (1336–1405) — markazi Samarqand bo'lgan ulkan davlat asoschisi, mashhur sarkarda va davlat arbobi.",
          ["When did Amir Timur live?", "Who was Amir Timur?"],
          "Amir Timur (1336–1405) was a famous military leader and statesman who founded a vast empire with "
          "Samarkand as its capital."),
    _fact("ulugbek", ["Mirzo Ulug'bek kim bo'lgan?", "Ulug'bek rasadxonasi qayerda?"],
          "Mirzo Ulug'bek (1394–1449) — Amir Temurning nabirasi, buyuk astronom va hukmdor. U Samarqandda "
          "mashhur rasadxona qurdirgan.",
          ["Who was Ulugh Beg?", "Where is Ulugh Beg's observatory?"],
          "Ulugh Beg (1394–1449) was Amir Timur's grandson, a great astronomer and ruler. He built a famous "
          "observatory in Samarkand."),
    _fact("ibn_sino", ["Ibn Sino kim bo'lgan?", "\"Tib qonunlari\" asarini kim yozgan?"],
          "Abu Ali ibn Sino (980–1037) — buxorolik buyuk olim va tabib. Uning \"Tib qonunlari\" asari asrlar "
          "davomida tibbiyotning asosiy darsligi bo'lgan.",
          ["Who was Avicenna?", "Who wrote The Canon of Medicine?"],
          "Avicenna (Ibn Sina, 980–1037) was a great scholar and physician born near Bukhara. His \"Canon of "
          "Medicine\" was a standard medical textbook for centuries."),
    _fact("xorazmiy", ["Al-Xorazmiy nima bilan mashhur?", "\"Algoritm\" so'zi qayerdan kelgan?"],
          "Muhammad ibn Muso al-Xorazmiy — IX asrda yashagan xorazmlik buyuk matematik. \"Algebra\" so'zi uning "
          "asari nomidan, \"algoritm\" so'zi esa uning ismidan kelib chiqqan.",
          ["What is al-Khwarizmi famous for?", "Where does the word \"algorithm\" come from?"],
          "Muhammad ibn Musa al-Khwarizmi was a 9th-century mathematician from Khwarazm. The word \"algebra\" "
          "comes from the title of his book, and \"algorithm\" comes from his name."),
    _fact("bobur", ["Zahiriddin Muhammad Bobur kim?", "\"Boburnoma\"ni kim yozgan?"],
          "Zahiriddin Muhammad Bobur (1483–1530) — shoir, tarixchi va Hindistonda Boburiylar sulolasiga asos "
          "solgan hukmdor. U \"Boburnoma\" asarining muallifi.",
          ["Who was Babur?", "Who wrote the Baburnama?"],
          "Zahir-ud-din Muhammad Babur (1483–1530) was a poet, historian and the ruler who founded the Mughal "
          "dynasty in India. He wrote the \"Baburnama\"."),
    _fact("registon", ["Registon maydoni qayerda joylashgan?", "Registon nima?"],
          "Registon — Samarqanddagi mashhur maydon. U yerda uchta madrasa bor: Ulug'bek, Sherdor va Tillakori.",
          ["Where is the Registan?", "What is the Registan?"],
          "The Registan is a famous square in Samarkand framed by three madrasahs: Ulugh Beg, Sher-Dor and Tilya-Kori."),
    _fact("yupiter", ["Quyosh tizimidagi eng katta sayyora qaysi?", "Eng katta sayyora qaysi?"],
          "Quyosh tizimidagi eng katta sayyora — Yupiter.",
          ["What is the largest planet in the Solar System?", "Which planet is the biggest?"],
          "The largest planet in the Solar System is Jupiter."),
    _fact("tinch_okeani", ["Dunyodagi eng katta okean qaysi?", "Eng katta okean qaysi?"],
          "Dunyodagi eng katta okean — Tinch okeani.",
          ["What is the largest ocean on Earth?", "Which ocean is the biggest?"],
          "The largest ocean on Earth is the Pacific Ocean."),
    _fact("yil_kunlari", ["Bir yilda necha kun bor?", "Kabisa yilida necha kun bo'ladi?"],
          "Oddiy yilda 365 kun, kabisa yilida esa 366 kun bo'ladi.",
          ["How many days are in a year?", "How many days does a leap year have?"],
          "A common year has 365 days and a leap year has 366 days."),
    _fact("suv_qaynash", ["Suv necha darajada qaynaydi?", "Suvning qaynash harorati qancha?"],
          "Dengiz sathida, normal atmosfera bosimida suv 100 °C da qaynaydi.",
          ["At what temperature does water boil?", "What is the boiling point of water?"],
          "At sea level, under standard atmospheric pressure, water boils at 100 °C (212 °F)."),
    _fact("suv_muzlash", ["Suv necha darajada muzlaydi?", "Suvning muzlash harorati qancha?"],
          "Normal atmosfera bosimida toza suv 0 °C da muzlaydi.",
          ["At what temperature does water freeze?", "What is the freezing point of water?"],
          "Under standard atmospheric pressure, pure water freezes at 0 °C (32 °F)."),
    _fact("h2o", ["Suvning kimyoviy formulasi qanday?", "H2O nima?"],
          "Suvning kimyoviy formulasi — H2O: ikkita vodorod atomi va bitta kislorod atomi.",
          ["What is the chemical formula of water?", "What does H2O mean?"],
          "The chemical formula of water is H2O: two hydrogen atoms and one oxygen atom."),
    _fact("yorug_lik", ["Yorug'lik tezligi qancha?", "Yorug'lik bir soniyada qancha masofani bosib o'tadi?"],
          "Vakuumda yorug'lik tezligi taxminan 299 792 km/s (aniq qiymati 299 792 458 m/s).",
          ["What is the speed of light?", "How far does light travel in one second?"],
          "The speed of light in a vacuum is about 299,792 km/s (exactly 299,792,458 m/s)."),
    _fact("suyaklar", ["Katta yoshli odamda nechta suyak bor?", "Inson skeleti nechta suyakdan iborat?"],
          "Katta yoshli odam skeletida odatda 206 ta suyak bo'ladi.",
          ["How many bones does an adult human have?", "How many bones are in the human skeleton?"],
          "An adult human skeleton usually has 206 bones."),
    _fact("merkuriy", ["Quyoshga eng yaqin sayyora qaysi?", "Birinchi sayyora qaysi?"],
          "Quyoshga eng yaqin sayyora — Merkuriy.",
          ["Which planet is closest to the Sun?", "What is the first planet from the Sun?"],
          "The planet closest to the Sun is Mercury."),
    _fact("sayyoralar", ["Quyosh tizimida nechta sayyora bor?", "Sayyoralar sonini ayt."],
          "Quyosh tizimida 8 ta sayyora bor: Merkuriy, Venera, Yer, Mars, Yupiter, Saturn, Uran va Neptun.",
          ["How many planets are in the Solar System?", "Name the planets of the Solar System."],
          "There are 8 planets in the Solar System: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus and Neptune."),
    _fact("everest", ["Dunyodagi eng baland tog' qaysi?", "Everest balandligi qancha?"],
          "Dunyodagi eng baland tog' — Everest (Jomolungma). 2020-yilgi o'lchovga ko'ra, uning balandligi "
          "8 848,86 metr.",
          ["What is the highest mountain in the world?", "How tall is Mount Everest?"],
          "The highest mountain in the world is Mount Everest. According to the 2020 survey it is 8,848.86 metres tall."),
    _fact("fotosintez", ["Fotosintez nima?", "O'simliklar qanday oziqlanadi?"],
          "Fotosintez — o'simliklar quyosh nuri yordamida karbonat angidrid va suvdan glyukoza hosil qilib, "
          "kislorod ajratib chiqaradigan jarayon.",
          ["What is photosynthesis?", "How do plants make food?"],
          "Photosynthesis is the process in which plants use sunlight to turn carbon dioxide and water into "
          "glucose, releasing oxygen."),
    _fact("davlat_tili", ["O'zbekistonning davlat tili qaysi?", "O'zbek lotin alifbosi qachon qabul qilingan?"],
          "O'zbekistonning davlat tili — o'zbek tili. Lotin yozuviga asoslangan o'zbek alifbosi 1993-yilda qabul qilingan.",
          ["What is the official language of Uzbekistan?", "When was the Uzbek Latin alphabet adopted?"],
          "The official language of Uzbekistan is Uzbek. The Latin-based Uzbek alphabet was adopted in 1993."),
    _fact("quruqlik", ["O'zbekistonning geografik xususiyati nimada?", "Ikki karra quruqlik bilan o'ralgan davlatlar qaysilar?"],
          "O'zbekiston ikki karra quruqlik bilan o'ralgan davlat: uning qo'shnilari ham dengizga chiqmaydi. "
          "Dunyoda bunday davlat faqat ikkita — O'zbekiston va Lixtenshteyn.",
          ["What is special about Uzbekistan's geography?", "Which countries are doubly landlocked?"],
          "Uzbekistan is doubly landlocked: none of its neighbours has access to the sea either. Only two "
          "countries in the world are doubly landlocked — Uzbekistan and Liechtenstein."),
    _fact("qoshnilar", ["O'zbekiston qaysi davlatlar bilan chegaradosh?", "O'zbekistonning qo'shnilari kimlar?"],
          "O'zbekiston beshta davlat bilan chegaradosh: Qozog'iston, Qirg'iziston, Tojikiston, Afg'oniston va Turkmaniston.",
          ["Which countries border Uzbekistan?", "Who are Uzbekistan's neighbours?"],
          "Uzbekistan borders five countries: Kazakhstan, Kyrgyzstan, Tajikistan, Afghanistan and Turkmenistan."),
    _fact("konstitutsiya", ["O'zbekistonning birinchi Konstitutsiyasi qachon qabul qilingan?",
                            "Konstitutsiya kuni qachon?"],
          "Mustaqil O'zbekistonning birinchi Konstitutsiyasi 1992-yil 8-dekabrda qabul qilingan.",
          ["When was Uzbekistan's first constitution adopted?", "When is Constitution Day in Uzbekistan?"],
          "The first Constitution of independent Uzbekistan was adopted on 8 December 1992."),
    _fact("navroz", ["Navro'z qachon nishonlanadi?", "Navro'z bayrami qaysi kuni?"],
          "Navro'z har yili 21-martda, bahorgi kun-tun tengligi davrida nishonlanadi.",
          ["When is Navruz celebrated?", "What date is Navruz?"],
          "Navruz is celebrated every year on 21 March, around the spring equinox."),
    _fact("orol", ["Orol dengizi bilan nima bo'lgan?", "Orol fojiasi nima?"],
          "Orol dengizi XX asrning ikkinchi yarmidan boshlab keskin qurib bormoqda, chunki Amudaryo va "
          "Sirdaryo suvi asosan sug'orishga sarflangan. Bu katta ekologik fojiaga aylangan.",
          ["What happened to the Aral Sea?", "Why did the Aral Sea shrink?"],
          "Since the second half of the 20th century the Aral Sea has shrunk dramatically because the water of "
          "the Amu Darya and Syr Darya was diverted for irrigation, causing a major ecological disaster."),
    _fact("oy", ["Oy Yer atrofini necha kunda aylanib chiqadi?", "Yerning tabiiy yo'ldoshi nima?"],
          "Yerning tabiiy yo'ldoshi — Oy. U Yer atrofini taxminan 27,3 kunda bir marta aylanib chiqadi.",
          ["How long does the Moon take to orbit the Earth?", "What is Earth's natural satellite?"],
          "Earth's natural satellite is the Moon. It orbits the Earth about once every 27.3 days."),
    _fact("gravitatsiya", ["Erkin tushish tezlanishi qancha?", "g qiymati nechaga teng?"],
          "Yer yuzasida erkin tushish tezlanishi taxminan 9,8 m/s² ga teng.",
          ["What is the acceleration due to gravity on Earth?", "What is the value of g?"],
          "Near the Earth's surface the acceleration due to gravity is about 9.8 m/s²."),
    _fact("absolyut_nol", ["Absolyut nol necha daraja?", "Kelvin shkalasida 0 nimani bildiradi?"],
          "Absolyut nol — 0 K, ya'ni −273,15 °C. Bu nazariy jihatdan eng past harorat.",
          ["What is absolute zero?", "What does 0 kelvin mean?"],
          "Absolute zero is 0 K, which equals −273.15 °C; it is the lowest possible temperature."),
    _fact("atom", ["Atom nimalardan tashkil topgan?", "Elektronning zaryadi qanday?"],
          "Atom yadrosi proton va neytronlardan iborat, uning atrofida esa manfiy zaryadli elektronlar harakatlanadi.",
          ["What is an atom made of?", "What is the charge of an electron?"],
          "An atom has a nucleus made of protons and neutrons, surrounded by negatively charged electrons."),
    _fact("python", ["Python dasturlash tilini kim yaratgan?", "Python qachon chiqqan?"],
          "Python tilini Guido van Rossum yaratgan; uning birinchi versiyasi 1991-yilda chiqqan.",
          ["Who created the Python programming language?", "When was Python first released?"],
          "Python was created by Guido van Rossum and first released in 1991."),
    _fact("bayt", ["Bir baytda necha bit bor?", "Bayt nima?"],
          "Bir baytda 8 bit bor; bitta bayt 0 dan 255 gacha bo'lgan 256 xil qiymatni saqlay oladi.",
          ["How many bits are in a byte?", "What is a byte?"],
          "A byte has 8 bits, so it can store 256 different values, from 0 to 255."),
    _fact("ikkilik", ["Ikkilik sanoq sistemasi nima?", "Kompyuterlar qaysi raqamlardan foydalanadi?"],
          "Ikkilik sanoq sistemasida faqat ikkita raqam bor: 0 va 1. Kompyuterlar ma'lumotlarni aynan shu tarzda saqlaydi.",
          ["What is the binary number system?", "Which digits do computers use?"],
          "The binary system uses only two digits, 0 and 1; computers store data this way."),
    _fact("pi", ["Pi soni nimaga teng?", "π ning qiymati qancha?"],
          "π (pi) — aylana uzunligining diametriga nisbati, taxminan 3,14159 ga teng.",
          ["What is pi?", "What is the value of π?"],
          "π (pi) is the ratio of a circle's circumference to its diameter, approximately 3.14159."),
    _fact("uchburchak", ["Uchburchak burchaklari yig'indisi nechaga teng?", "Uchburchakning ichki burchaklari yig'indisi qancha?"],
          "Tekislikdagi har qanday uchburchakning ichki burchaklari yig'indisi 180° ga teng.",
          ["What is the sum of the angles of a triangle?", "How many degrees are in a triangle?"],
          "The interior angles of any triangle in a plane add up to 180°."),
    _fact("qitalar", ["Yer yuzida nechta qit'a bor?", "Qit'alarni sanab ber."],
          "Ko'p qo'llaniladigan tasnif bo'yicha 7 ta qit'a bor: Osiyo, Afrika, Shimoliy Amerika, Janubiy "
          "Amerika, Antarktida, Yevropa va Avstraliya.",
          ["How many continents are there?", "Name the continents."],
          "In the most common model there are 7 continents: Asia, Africa, North America, South America, "
          "Antarctica, Europe and Australia."),
    _fact("palov", ["O'zbek milliy taomi nima?", "Osh qanday tayyorlanadi?"],
          "O'zbeklarning eng mashhur milliy taomi — palov (osh). U guruch, sabzi, piyoz, go'sht va yog'dan "
          "qozonda tayyorlanadi.",
          ["What is the Uzbek national dish?", "What is plov made of?"],
          "The most famous Uzbek national dish is plov (osh), cooked in a cauldron from rice, carrots, onions, "
          "meat and oil."),
    _fact("vodorod", ["Eng yengil kimyoviy element qaysi?", "Vodorodning atom raqami nechchi?"],
          "Eng yengil kimyoviy element — vodorod; uning atom raqami 1.",
          ["What is the lightest chemical element?", "What is the atomic number of hydrogen?"],
          "The lightest chemical element is hydrogen; its atomic number is 1."),
    _fact("xromosomalar", ["Odamda nechta xromosoma bor?", "Inson hujayrasida necha juft xromosoma bor?"],
          "Odam hujayralarida odatda 46 ta, ya'ni 23 juft xromosoma bo'ladi.",
          ["How many chromosomes do humans have?", "How many pairs of chromosomes are in a human cell?"],
          "Human cells normally have 46 chromosomes, arranged in 23 pairs."),
    _fact("yurak", ["Odam yuragi nechta kameradan iborat?", "Yurak qanday tuzilgan?"],
          "Odam yuragi to'rtta kameradan iborat: ikkita bo'lmacha va ikkita qorincha.",
          ["How many chambers does the human heart have?", "How is the heart structured?"],
          "The human heart has four chambers: two atria and two ventricles."),
    _fact("metro", ["Toshkent metrosi qachon ochilgan?", "Markaziy Osiyodagi birinchi metro qayerda?"],
          "Toshkent metrosi 1977-yilda ochilgan va u Markaziy Osiyodagi birinchi metro hisoblanadi.",
          ["When did the Tashkent Metro open?", "Where is the first metro in Central Asia?"],
          "The Tashkent Metro opened in 1977 and was the first metro system in Central Asia."),
    _fact("bayroq", ["O'zbekiston bayrog'i qanday?", "Bayroqdagi yulduzlar nechta?"],
          "O'zbekiston bayrog'i ko'k, oq va yashil rangli uchta gorizontal yo'ldan iborat, ular orasida ingichka "
          "qizil chiziqlar bor. Ko'k qismida oq yarim oy va 12 ta yulduz tasvirlangan.",
          ["What does the flag of Uzbekistan look like?", "How many stars are on the Uzbek flag?"],
          "The flag of Uzbekistan has three horizontal stripes — blue, white and green — separated by thin "
          "red lines. The blue stripe shows a white crescent and 12 white stars."),
    _fact("qizilqum", ["O'zbekistondagi eng katta cho'l qaysi?", "Qizilqum qayerda?"],
          "Qizilqum — Amudaryo bilan Sirdaryo oralig'ida joylashgan katta cho'l. Uning asosiy qismi O'zbekiston va "
          "Qozog'iston hududida, kichik qismi Turkmanistonda.",
          ["What is the largest desert in Uzbekistan?", "Where is the Kyzylkum desert?"],
          "The Kyzylkum is a large desert between the Amu Darya and Syr Darya rivers, mostly in Uzbekistan and "
          "Kazakhstan, with a small part in Turkmenistan."),
    _fact("vaqt_mintaqasi", ["O'zbekiston qaysi vaqt mintaqasida?", "Toshkent vaqti UTC dan qancha farq qiladi?"],
          "O'zbekiston UTC+5 vaqt mintaqasida joylashgan va yozgi vaqtga o'tmaydi.",
          ["What time zone is Uzbekistan in?", "How far is Tashkent time from UTC?"],
          "Uzbekistan uses UTC+5 and does not observe daylight saving time."),
    _fact("tovush", ["Havoda tovush tezligi qancha?", "Tovush qanchalik tez tarqaladi?"],
          "20 °C haroratli havoda tovush tezligi taxminan 343 m/s.",
          ["What is the speed of sound in air?", "How fast does sound travel?"],
          "In air at 20 °C, sound travels at about 343 m/s."),
    _fact("vaqt_birliklari", ["Bir soatda necha soniya bor?", "Bir sutkada necha soat bor?"],
          "Bir soatda 3600 soniya, bir sutkada esa 24 soat bor.",
          ["How many seconds are in an hour?", "How many hours are in a day?"],
          "There are 3,600 seconds in an hour and 24 hours in a day."),
    _fact("dnk", ["DNK qanday tuzilishga ega?", "DNK nima?"],
          "DNK — irsiy axborotni saqlaydigan molekula. U qo'sh spiral shaklida tuzilgan.",
          ["What is the structure of DNA?", "What is DNA?"],
          "DNA is the molecule that stores genetic information; it has a double-helix structure."),
]

_FACT_INDEX: Dict[str, Dict[str, Any]] = {f["key"]: f for f in FACTS}


def knowledge_conversations(n: Optional[int] = None, seed: int = 0) -> List[Conversation]:
    """Question-answer pairs over :data:`FACTS` (Uzbek and English phrasings).

    ``n=None`` -> every fact x every phrasing in both languages (~4 per fact); otherwise ``n``
    conversations sampled round-robin over facts so each fact appears.
    """
    rng = _rng("knowledge", seed)
    items: List[Tuple[str, str]] = []
    for f in FACTS:
        items += [(q, f["uz_a"]) for q in f["uz_q"]] + [(q, f["en_a"]) for q in f["en_q"]]
    if n is None:
        return [[_user(q), _asst(a)] for q, a in items]
    order = list(range(len(items)))
    out: List[Conversation] = []
    while len(out) < int(n):
        rng.shuffle(order)
        for i in order[: int(n) - len(out)]:
            q, a = items[i]
            out.append([_user(q), _asst(a)])
    return out


# =============================================================================================== arithmetic

_OPS = {"+": ("+", lambda a, b: a + b), "-": ("-", lambda a, b: a - b), "×": ("*", lambda a, b: a * b),
        "÷": ("/", lambda a, b: a / b)}
_ARITH_UZ = ["{a} {op} {b} nechaga teng?", "{a} {op} {b} ni hisoblang.", "Hisoblang: {a} {op} {b}",
             "{a} {op} {b} = ?", "Iltimos, {a} {op} {b} ni hisoblab ber."]
_ARITH_EN = ["What is {a} {op} {b}?", "Calculate {a} {op} {b}.", "Compute {a} {op} {b}.", "{a} {op} {b} = ?",
             "Please work out {a} {op} {b}."]


def _arith_operands(rng: random.Random, op: str, big: bool) -> Tuple[int, int]:
    hi = 99999 if big else 999
    if op == "+":
        return rng.randint(1, hi), rng.randint(1, hi)
    if op == "-":
        a, b = rng.randint(1, hi), rng.randint(1, hi)
        return (a, b) if rng.random() < 0.8 else (min(a, b), max(a, b))  # some negative results
    if op == "×":
        return (rng.randint(100, 9999), rng.randint(12, 999)) if big else (rng.randint(2, 99), rng.randint(2, 30))
    b = rng.randint(2, 99 if big else 12)
    q = rng.randint(2, 9999 if big else 99)
    return b * q, b  # exact division


def arithmetic_conversations(n: int = 1000, seed: int = 0, tool_frac: float = 0.25) -> List[Conversation]:
    """Exact +, -, ×, ÷ problems (Uzbek / English). ``tool_frac`` of them use big operands and solve the
    problem through the ``calculator`` tool (full loop: call -> result -> answer)."""
    rng = _rng("arithmetic", seed)
    out: List[Conversation] = []
    for i in range(int(n)):
        uz = i % 2 == 0
        op = _pick(rng, list(_OPS))
        use_tool = rng.random() < tool_frac
        a, b = _arith_operands(rng, op, use_tool)
        py_op, fn = _OPS[op]
        exact = Fraction(a) / Fraction(b) if op == "÷" else fn(a, b)
        res = _num(exact)
        q = _pick(rng, _ARITH_UZ if uz else _ARITH_EN).format(a=a, op=op, b=b)
        if use_tool:
            tools = [TOOL_SCHEMAS["calculator"]] + ([TOOL_SCHEMAS["unit_convert"]] if rng.random() < 0.5 else [])
            value = fn(a, b)  # what python (and the calculator tool) returns, e.g. 41.0 for 123 / 3
            ans = (f"{a} {op} {b} = {res}.\nJavob: {res}" if uz else f"{a} {op} {b} = {res}.\nAnswer: {res}")
            expr = f"{a} {py_op} {b}"
            out.append([_system(tools), _user(q), _call("calculator", {"expression": expr}),
                        _tool("calculator", _calc_result(expr, value)), _asst(ans)])
        else:
            ans = (f"{a} {op} {b} = {res}.\nJavob: {res}" if uz else f"{a} {op} {b} = {res}.\nAnswer: {res}")
            out.append([_user(q), _asst(ans)])
    return out


# =============================================================================================== tool calls

_TZ = [  # (tz, utc offset minutes, uz city, en city) -- zones without daylight saving time
    ("Asia/Tashkent", 300, "Toshkent", "Tashkent"), ("Asia/Samarkand", 300, "Samarqand", "Samarkand"),
    ("Asia/Tokyo", 540, "Tokio", "Tokyo"), ("Asia/Dubai", 240, "Dubay", "Dubai"),
    ("Asia/Shanghai", 480, "Shanxay", "Shanghai"), ("Asia/Kolkata", 330, "Kalkutta", "Kolkata"),
    ("Europe/Moscow", 180, "Moskva", "Moscow"), ("Europe/Istanbul", 180, "Istanbul", "Istanbul"),
    ("Asia/Seoul", 540, "Seul", "Seoul"), ("UTC", 0, "UTC bo'yicha", "UTC"),
]
_UZ_MONTHS = ["yanvar", "fevral", "mart", "aprel", "may", "iyun", "iyul", "avgust", "sentabr", "oktabr",
              "noyabr", "dekabr"]
_EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
              "October", "November", "December"]
_UNITS = {  # unit -> (kind, factor to base, uz name, en name)
    "km": ("length", 1000.0, "kilometr", "kilometres"), "m": ("length", 1.0, "metr", "metres"),
    "cm": ("length", 0.01, "santimetr", "centimetres"), "mi": ("length", 1609.344, "mil", "miles"),
    "ft": ("length", 0.3048, "fut", "feet"), "in": ("length", 0.0254, "dyuym", "inches"),
    "kg": ("mass", 1.0, "kilogramm", "kilograms"), "g": ("mass", 0.001, "gramm", "grams"),
    "lb": ("mass", 0.45359237, "funt", "pounds"), "oz": ("mass", 0.028349523125, "unsiya", "ounces"),
    "C": ("temperature", None, "Selsiy darajasi", "degrees Celsius"),
    "F": ("temperature", None, "Farengeyt darajasi", "degrees Fahrenheit"),
    "K": ("temperature", None, "Kelvin", "kelvins"),
}
_CONVERSIONS = [("km", "mi"), ("mi", "km"), ("m", "ft"), ("ft", "m"), ("cm", "in"), ("in", "cm"), ("km", "m"),
                ("kg", "lb"), ("lb", "kg"), ("g", "oz"), ("oz", "g"), ("kg", "g"), ("C", "F"), ("F", "C"),
                ("C", "K"), ("K", "C")]
_WEATHER_CITIES = [("Toshkent", "Tashkent"), ("Samarqand", "Samarkand"), ("Buxoro", "Bukhara"),
                   ("Xiva", "Khiva"), ("Namangan", "Namangan"), ("Nukus", "Nukus"), ("London", "London"),
                   ("Parij", "Paris")]
_CONDITIONS = [("quyoshli", "sunny"), ("bulutli", "cloudy"), ("yomg'irli", "rainy"), ("qorli", "snowy"),
               ("shamolli", "windy"), ("qisman bulutli", "partly cloudy")]
_WIKI_TOPICS = [("navoiy", "Alisher Navoiy", "Alisher Navoi"), ("temur", "Amir Temur", "Amir Timur"),
                ("ulugbek", "Mirzo Ulug'bek", "Ulugh Beg"), ("ibn_sino", "Ibn Sino", "Avicenna"),
                ("xorazmiy", "Al-Xorazmiy", "Al-Khwarizmi"), ("bobur", "Bobur", "Babur"),
                ("registon", "Registon", "Registan"), ("everest", "Everest", "Mount Everest"),
                ("python", "Python (dasturlash tili)", "Python (programming language)"),
                ("metro", "Toshkent metrosi", "Tashkent Metro"), ("orol", "Orol dengizi", "Aral Sea"),
                ("yupiter", "Yupiter", "Jupiter")]


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Reference conversion used for synthetic tool results (length / mass / temperature)."""
    k1, f1 = _UNITS[from_unit][0], _UNITS[from_unit][1]
    k2, f2 = _UNITS[to_unit][0], _UNITS[to_unit][1]
    if k1 != k2:
        raise ValueError(f"cannot convert {from_unit} to {to_unit}")
    if k1 != "temperature":
        return value * f1 / f2
    c = {"C": value, "F": (value - 32.0) * 5.0 / 9.0, "K": value - 273.15}[from_unit]
    return {"C": c, "F": c * 9.0 / 5.0 + 32.0, "K": c + 273.15}[to_unit]


def _tool_set(rng: random.Random, needed: Sequence[str]) -> List[Dict[str, Any]]:
    names = list(dict.fromkeys(list(needed) + [t for t in ("calculator", "current_time", "unit_convert", "weather",
                                                           "search_wikipedia") if rng.random() < 0.4]))
    rng.shuffle(names)
    return [TOOL_SCHEMAS[t] for t in names]


def _random_datetime(rng: random.Random) -> Tuple[int, int, int, int, int, int]:
    month = rng.randint(1, 12)
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    return 2025, month, rng.randint(1, days), rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)


def _iso(dt: Tuple[int, int, int, int, int, int], offset_min: int) -> str:
    sign = "+" if offset_min >= 0 else "-"
    hh, mm = divmod(abs(offset_min), 60)
    y, mo, d, h, mi, s = dt
    return f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}{sign}{hh:02d}:{mm:02d}"


_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _time_result(tz: str, dt: Tuple[int, int, int, int, int, int], offset_min: int) -> Dict[str, str]:
    """Same keys as the real ``current_time`` tool: timezone, datetime, date, time, weekday, utc_offset."""
    import datetime as _datetime

    y, mo, d, h, mi, sec = dt
    iso = _iso(dt, offset_min)
    return {"timezone": tz, "datetime": iso, "date": f"{y:04d}-{mo:02d}-{d:02d}",
            "time": f"{h:02d}:{mi:02d}:{sec:02d}", "weekday": _WEEKDAYS[_datetime.date(y, mo, d).weekday()],
            "utc_offset": iso[-6:]}


def _calc_scenario(rng: random.Random, uz: bool) -> Conversation:
    kind = rng.randrange(4)
    if kind == 0:
        x, p = rng.randint(20, 5000), _pick(rng, [5, 10, 12, 15, 18, 20, 25, 30, 40, 75])
        expr, value = f"{x} * {p} / 100", x * p / 100
        q = (f"{x} ning {p} foizi qancha?" if uz else f"What is {p}% of {x}?")
        a = (f"{x} ning {p} foizi {_num(value)} ga teng." if uz else f"{p}% of {x} is {_num(value)}.")
    elif kind == 1:
        a1, b1, c1 = rng.randint(10, 999), rng.randint(10, 999), rng.randint(2, 99)
        expr, value = f"({a1} + {b1}) * {c1}", (a1 + b1) * c1
        q = (f"({a1} + {b1}) × {c1} ni hisoblab ber." if uz else f"Please calculate ({a1} + {b1}) × {c1}.")
        a = (f"({a1} + {b1}) × {c1} = {value}." if uz else f"({a1} + {b1}) × {c1} = {value}.")
    elif kind == 2:
        root = rng.randint(11, 300)
        sq = root * root if rng.random() < 0.6 else root * root + rng.randint(1, 2 * root)
        expr, value = f"sqrt({sq})", math.sqrt(sq)
        exact = value == int(value)
        r = _num(round(value, 4))
        q = (f"{sq} ning kvadrat ildizi nechaga teng?" if uz else f"What is the square root of {sq}?")
        a = ((f"{sq} ning kvadrat ildizi {r} ga teng." if exact else f"{sq} ning kvadrat ildizi taxminan {r}.") if uz
             else (f"The square root of {sq} is {r}." if exact else f"The square root of {sq} is approximately {r}."))
    else:
        base, exp = rng.randint(2, 12), rng.randint(3, 12)
        expr, value = f"{base} ** {exp}", base ** exp
        q = (f"{base} ning {exp}-darajasi nechchi?" if uz else f"What is {base} to the power of {exp}?")
        a = (f"{base}^{exp} = {value}." if uz else f"{base}^{exp} = {value}.")
    return [_system(_tool_set(rng, ["calculator"])), _user(q), _call("calculator", {"expression": expr}),
            _tool("calculator", _calc_result(expr, value)), _asst(a)]


def _time_scenario(rng: random.Random, uz: bool) -> Conversation:
    tz, off, cuz, cen = _pick(rng, _TZ)
    dt = _random_datetime(rng)
    y, mo, d, h, mi, _ = dt
    result = _time_result(tz, dt, off)
    if tz == "UTC":
        q = "Hozir UTC bo'yicha soat necha?" if uz else "What time is it in UTC right now?"
        a = (f"Hozir UTC bo'yicha soat {h:02d}:{mi:02d} ({y}-yil {d}-{_UZ_MONTHS[mo - 1]})." if uz
             else f"It is currently {h:02d}:{mi:02d} UTC on {d} {_EN_MONTHS[mo - 1]} {y}.")
    elif rng.random() < 0.3 and tz == "Asia/Tashkent":
        q = "Bugun sana nechanchi?" if uz else "What's today's date in Tashkent?"
        a = (f"Bugun {y}-yil {d}-{_UZ_MONTHS[mo - 1]}, Toshkent vaqti bilan soat {h:02d}:{mi:02d}." if uz
             else f"Today is {d} {_EN_MONTHS[mo - 1]} {y}; the local time in Tashkent is {h:02d}:{mi:02d}.")
    else:
        q = f"{cuz}da hozir soat necha?" if uz else f"What time is it in {cen} now?"
        a = (f"Hozir {cuz}da soat {h:02d}:{mi:02d} ({y}-yil {d}-{_UZ_MONTHS[mo - 1]})." if uz
             else f"It's {h:02d}:{mi:02d} in {cen} right now ({d} {_EN_MONTHS[mo - 1]} {y}).")
    return [_system(_tool_set(rng, ["current_time"])), _user(q), _call("current_time", {"timezone": tz}),
            _tool("current_time", result), _asst(a)]


def _two_time_scenario(rng: random.Random, uz: bool) -> Conversation:
    (tz1, off1, cuz1, cen1), (tz2, off2, cuz2, cen2) = rng.sample([t for t in _TZ if t[0] != "UTC"], 2)
    dt = _random_datetime(rng)
    # same instant expressed in both zones: shift dt by the offset difference via minutes arithmetic
    y, mo, d, h, mi, s = dt
    total = h * 60 + mi + (off2 - off1)
    day_shift, rem = divmod(total, 24 * 60)
    h2, mi2 = divmod(rem, 60)
    diff = (off2 - off1) / 60
    r1 = _time_result(tz1, dt, off1)
    msgs = [_system(_tool_set(rng, ["current_time"])),
            _user(f"{cuz1} va {cuz2}da hozir soat necha? Farqi qancha?" if uz
                  else f"What time is it in {cen1} and {cen2}? What's the difference?"),
            _call("current_time", {"timezone": tz1}), _tool("current_time", r1)]
    # second call at the same instant (seconds identical); compute the calendar date in zone 2
    days_in = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    d2, mo2, y2 = d + day_shift, mo, y
    if d2 < 1:
        mo2 -= 1
        if mo2 < 1:
            mo2, y2 = 12, y2 - 1
        d2 = days_in[mo2 - 1]
    elif d2 > days_in[mo - 1]:
        d2, mo2 = 1, mo2 + 1
        if mo2 > 12:
            mo2, y2 = 1, y2 + 1
    dt2 = (y2, mo2, d2, h2, mi2, s)
    msgs += [_call("current_time", {"timezone": tz2}), _tool("current_time", _time_result(tz2, dt2, off2))]
    dtxt = _num(abs(diff))
    if uz:
        rel = ("bir xil" if diff == 0 else f"{cuz2} {cuz1}dan {dtxt} soat " + ("oldinda" if diff > 0 else "orqada"))
        a = f"{cuz1}da soat {h:02d}:{mi:02d}, {cuz2}da esa {h2:02d}:{mi2:02d}. Vaqt farqi: {rel}."
    else:
        rel = ("none" if diff == 0 else f"{cen2} is {dtxt} hour{'s' if abs(diff) != 1 else ''} "
               + ("ahead of" if diff > 0 else "behind") + f" {cen1}")
        a = f"It's {h:02d}:{mi:02d} in {cen1} and {h2:02d}:{mi2:02d} in {cen2}. Time difference: {rel}."
    msgs.append(_asst(a))
    return msgs


def _unit_scenario(rng: random.Random, uz: bool) -> Conversation:
    fu, tu = _pick(rng, _CONVERSIONS)
    kind = _UNITS[fu][0]
    if kind == "temperature":
        value = float(rng.randint(-30, 45)) if fu == "C" else float(rng.randint(-20, 110)) if fu == "F" else float(
            rng.randint(250, 320))
    else:
        value = float(_pick(rng, [1, 2, 3, 5, 10, 12, 15, 20, 25, 42, 50, 100, 250, 500]))
    res = convert_units(value, fu, tu)
    shown = _num(round(res, 2))
    v = _num(value)
    uzf, uzt, enf, ent = _UNITS[fu][2], _UNITS[tu][2], _UNITS[fu][3], _UNITS[tu][3]
    q = (f"{v} {uzf} necha {uzt} bo'ladi?" if uz else f"How many {ent} is {v} {enf}?")
    approx = abs(res - round(res, 2)) > 1e-9
    if uz:
        a = f"{v} {uzf} {'taxminan ' if approx else ''}{shown} {uzt} ga teng."
    else:
        a = f"{v} {enf} is {'about ' if approx else ''}{shown} {ent}."
    return [_system(_tool_set(rng, ["unit_convert"])), _user(q),
            _call("unit_convert", {"value": value, "from_unit": fu, "to_unit": tu}),
            _tool("unit_convert", _unit_result(value, fu, tu, res)), _asst(a)]


def _weather_scenario(rng: random.Random, uz: bool) -> Conversation:
    cuz, cen = _pick(rng, _WEATHER_CITIES)
    cond_uz, cond_en = _pick(rng, _CONDITIONS)
    temp, hum = rng.randint(-10, 40), rng.randint(15, 90)
    city = cuz if uz else cen
    result = {"city": city, "temperature_c": temp, "condition": cond_en, "humidity": hum,
              "note": "FAKE demo data, not a real forecast"}
    q = (f"{cuz}da ob-havo qanday?" if uz else f"What's the weather like in {cen}?")
    a = (f"Vosita qaytargan ma'lumotga ko'ra {cuz}da harorat {temp} °C, havo {cond_uz}, namlik {hum}%. "
         "Diqqat: bu sinov uchun namunaviy (haqiqiy bo'lmagan) ma'lumot." if uz else
         f"According to the tool, it's {temp} °C and {cond_en} in {cen}, with {hum}% humidity. Note: this is "
         "fake demo data, not a real forecast.")
    return [_system(_tool_set(rng, ["weather"])), _user(q), _call("weather", {"city": city}),
            _tool("weather", result), _asst(a)]


def _wiki_scenario(rng: random.Random, uz: bool) -> Conversation:
    key, tuz, ten = _pick(rng, _WIKI_TOPICS)
    fact = _FACT_INDEX[key]
    title = tuz if uz else ten
    summary = fact["uz_a"] if uz else fact["en_a"]
    q = (f"{tuz} haqida qisqacha ma'lumot topib ber." if uz else f"Look up some information about {ten}.")
    a = (f"Topilgan ma'lumot: {summary}" if uz else f"Here's what I found: {summary}")
    return [_system(_tool_set(rng, ["search_wikipedia"])), _user(q), _call("search_wikipedia", {"query": title}),
            _tool("search_wikipedia", {"title": title, "summary": summary, "source": "stub"}), _asst(a)]


def _no_tool_scenario(rng: random.Random, uz: bool) -> Conversation:
    if rng.random() < 0.5:
        q, a = _pick(rng, _GREET_UZ if uz else _GREET_EN)
    else:
        f = _pick(rng, FACTS)
        q, a = (_pick(rng, f["uz_q"]), f["uz_a"]) if uz else (_pick(rng, f["en_q"]), f["en_a"])
    return [_system(_tool_set(rng, [])), _user(q), _asst(a)]


_TOOL_SCENARIOS: List[Tuple[float, Callable[[random.Random, bool], Conversation]]] = [
    (0.28, _calc_scenario), (0.14, _time_scenario), (0.06, _two_time_scenario), (0.2, _unit_scenario),
    (0.12, _weather_scenario), (0.1, _wiki_scenario), (0.1, _no_tool_scenario),
]


def tool_call_conversations(n: int = 1000, seed: int = 0) -> List[Conversation]:
    """Full tool-use loops (user -> assistant tool_call -> tool result -> assistant answer) with the
    ``calculator``, ``current_time``, ``unit_convert``, fake ``weather`` and ``search_wikipedia`` stub
    tools; ~10% have tools available but need none (the model must answer directly)."""
    rng = _rng("tool_calls", seed)
    weights = [w for w, _ in _TOOL_SCENARIOS]
    out = []
    for i in range(int(n)):
        fn = rng.choices([f for _, f in _TOOL_SCENARIOS], weights=weights)[0]
        out.append(fn(rng, i % 2 == 0))
    return out


# =============================================================================================== code

def _task(key: str, uz: str, en: str, code: str, tests: str, ex_uz: str, ex_en: str) -> Dict[str, Any]:
    return {"key": key, "uz": uz, "en": en, "code": code.strip("\n"), "tests": tests.strip("\n"),
            "explain_uz": ex_uz, "explain_en": ex_en}


#: Small verified Python tasks (``verify_code_tasks`` executes ``code`` + ``tests``).
CODE_TASKS: List[Dict[str, Any]] = [
    _task("reverse_string", "Satrni teskari aylantiradigan Python funksiyasini yozing.",
          "Write a Python function that reverses a string.",
          '''
def reverse_string(s: str) -> str:
    """Return ``s`` reversed."""
    return s[::-1]
''', "assert reverse_string('salom') == 'molas'\nassert reverse_string('') == ''",
          "`s[::-1]` kesmasi satrni oxiridan boshiga qarab o'qiydi.",
          "The slice `s[::-1]` walks the string from the end to the start."),
    _task("fizzbuzz", "1 dan n gacha FizzBuzz ro'yxatini qaytaradigan funksiya yozing.",
          "Write a function that returns the FizzBuzz sequence from 1 to n as a list of strings.",
          '''
def fizzbuzz(n: int) -> list:
    """FizzBuzz for 1..n: multiples of 3 -> "Fizz", of 5 -> "Buzz", of both -> "FizzBuzz"."""
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result
''', "assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']\nassert fizzbuzz(15)[-1] == 'FizzBuzz'",
          "Avval 15 ga bo'linishni tekshiramiz, aks holda 3 yoki 5 ga bo'linish \"FizzBuzz\" holatini yashirib qo'yadi.",
          "We test divisibility by 15 first; otherwise the 3 or 5 branch would hide the \"FizzBuzz\" case."),
    _task("factorial", "Sonning faktorialini hisoblaydigan funksiya yozing.",
          "Write a function that computes the factorial of a non-negative integer.",
          '''
def factorial(n: int) -> int:
    """Return n! for n >= 0."""
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
''', "assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800",
          "Takrorlash orqali 2 dan n gacha bo'lgan sonlarni ko'paytiramiz; 0! = 1.",
          "We multiply the numbers from 2 to n in a loop; 0! is 1."),
    _task("is_prime", "Son tub ekanligini tekshiradigan funksiya yozing.",
          "Write a function that checks whether a number is prime.",
          '''
def is_prime(n: int) -> bool:
    """True if n is a prime number."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True
''', "assert [x for x in range(20) if is_prime(x)] == [2, 3, 5, 7, 11, 13, 17, 19]\nassert not is_prime(1)",
          "Bo'luvchilarni faqat √n gacha, toq sonlar bo'yicha tekshirish kifoya.",
          "It is enough to test odd divisors up to √n."),
    _task("fibonacci", "n-Fibonachchi sonini qaytaradigan funksiya yozing (F(0)=0, F(1)=1).",
          "Write a function that returns the n-th Fibonacci number (F(0)=0, F(1)=1).",
          '''
def fibonacci(n: int) -> int:
    """Return F(n) with F(0) = 0 and F(1) = 1 (iterative, O(n))."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
''', "assert [fibonacci(i) for i in range(8)] == [0, 1, 1, 2, 3, 5, 8, 13]\nassert fibonacci(30) == 832040",
          "Ikki oxirgi qiymatni saqlab, iterativ hisoblaymiz — rekursiyadan ancha tez.",
          "We keep the last two values and iterate, which is much faster than naive recursion."),
    _task("is_palindrome", "Satr palindrom ekanligini (bo'sh joy va katta-kichik harfga e'tibor bermasdan) tekshiruvchi funksiya yozing.",
          "Write a function that checks if a string is a palindrome, ignoring case and non-alphanumeric characters.",
          '''
def is_palindrome(text: str) -> bool:
    """True if ``text`` reads the same backwards (letters and digits only, case-insensitive)."""
    cleaned = [ch.lower() for ch in text if ch.isalnum()]
    return cleaned == cleaned[::-1]
''', "assert is_palindrome('A man, a plan, a canal: Panama')\nassert not is_palindrome('salom')\nassert is_palindrome('')",
          "Avval faqat harf va raqamlarni kichik harfda qoldiramiz, keyin teskarisi bilan solishtiramiz.",
          "We keep only lowercase letters and digits, then compare with the reversed list."),
    _task("count_vowels", "Matndagi unli harflar sonini hisoblaydigan funksiya yozing.",
          "Write a function that counts the vowels in a text.",
          '''
def count_vowels(text: str) -> int:
    """Number of vowels (a, e, i, o, u) in ``text``, case-insensitive."""
    return sum(1 for ch in text.lower() if ch in "aeiou")
''', "assert count_vowels('Moonlight') == 3\nassert count_vowels('xyz') == 0",
          "Har bir harfni kichik harfga o'tkazib, unlilar to'plamida bor-yo'qligini tekshiramiz.",
          "Each character is lowercased and checked against the vowel set."),
    _task("list_sum", "Ro'yxatdagi sonlar yig'indisini sum() ishlatmasdan hisoblang.",
          "Compute the sum of a list of numbers without using sum().",
          '''
def list_sum(numbers: list) -> float:
    """Sum of ``numbers`` using an explicit loop."""
    total = 0
    for x in numbers:
        total += x
    return total
''', "assert list_sum([1, 2, 3, 4]) == 10\nassert list_sum([]) == 0",
          "`total` o'zgaruvchisida har bir elementni qo'shib boramiz.",
          "We accumulate each element into the `total` variable."),
    _task("find_max", "Ro'yxatdagi eng katta elementni max() ishlatmasdan topadigan funksiya yozing.",
          "Write a function that finds the largest element of a list without using max().",
          '''
def find_max(items: list):
    """Largest element of a non-empty list."""
    if not items:
        raise ValueError("empty list")
    best = items[0]
    for x in items[1:]:
        if x > best:
            best = x
    return best
''', "assert find_max([3, 9, 2]) == 9\nassert find_max([-5, -2, -9]) == -2",
          "Birinchi elementni eng katta deb olamiz va qolganlari bilan solishtirib boramiz.",
          "We start with the first element and compare it with the rest."),
    _task("gcd", "Ikki sonning EKUBini (eng katta umumiy bo'luvchi) Evklid algoritmi bilan toping.",
          "Find the greatest common divisor of two numbers with Euclid's algorithm.",
          '''
def gcd(a: int, b: int) -> int:
    """Greatest common divisor (Euclid's algorithm)."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a
''', "assert gcd(48, 18) == 6\nassert gcd(17, 5) == 1\nassert gcd(0, 7) == 7",
          "Har qadamda (a, b) juftligini (b, a % b) bilan almashtiramiz, b nolga teng bo'lganda a javob bo'ladi.",
          "Each step replaces (a, b) with (b, a % b); when b reaches 0, a is the answer."),
    _task("c_to_f", "Selsiy darajasini Farengeytga o'tkazadigan funksiya yozing.",
          "Write a function that converts Celsius to Fahrenheit.",
          '''
def celsius_to_fahrenheit(c: float) -> float:
    """Convert degrees Celsius to Fahrenheit."""
    return c * 9 / 5 + 32
''', "assert celsius_to_fahrenheit(0) == 32\nassert celsius_to_fahrenheit(100) == 212\nassert celsius_to_fahrenheit(-40) == -40",
          "Formula: F = C × 9/5 + 32.", "The formula is F = C × 9/5 + 32."),
    _task("word_count", "Matndagi har bir so'z necha marta uchrashini sanaydigan funksiya yozing.",
          "Write a function that counts how many times each word occurs in a text.",
          '''
def word_count(text: str) -> dict:
    """Map each lowercase word to its number of occurrences."""
    counts = {}
    for word in text.lower().split():
        counts[word] = counts.get(word, 0) + 1
    return counts
''', "assert word_count('olma nok olma') == {'olma': 2, 'nok': 1}\nassert word_count('') == {}",
          "Lug'atda har bir so'z uchun hisoblagichni `get(word, 0) + 1` bilan oshiramiz.",
          "A dictionary counter is incremented with `get(word, 0) + 1`."),
    _task("dedupe", "Ro'yxatdagi takrorlarni tartibni saqlagan holda olib tashlang.",
          "Remove duplicates from a list while preserving the original order.",
          '''
def dedupe(items: list) -> list:
    """Items without duplicates, keeping the first occurrence order."""
    seen = set()
    result = []
    for x in items:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result
''', "assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]\nassert dedupe([]) == []",
          "`seen` to'plami ko'rilgan elementlarni eslab qoladi, natijaga faqat yangilari qo'shiladi.",
          "The `seen` set remembers elements; only new ones are appended."),
    _task("flatten", "Ichma-ich ro'yxatni (har qanday chuqurlikda) tekis ro'yxatga aylantiring.",
          "Flatten an arbitrarily nested list into a flat list.",
          '''
def flatten(nested: list) -> list:
    """Flatten nested lists of any depth."""
    flat = []
    for x in nested:
        if isinstance(x, list):
            flat.extend(flatten(x))
        else:
            flat.append(x)
    return flat
''', "assert flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]\nassert flatten([]) == []",
          "Element ro'yxat bo'lsa, funksiyani rekursiv chaqiramiz.",
          "When an element is a list we recurse into it."),
    _task("binary_search", "Saralangan ro'yxatda ikkilik qidiruvni amalga oshiring (topilmasa -1).",
          "Implement binary search on a sorted list (return -1 if not found).",
          '''
def binary_search(items: list, target) -> int:
    """Index of ``target`` in sorted ``items`` or -1."""
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''', "assert binary_search([1, 3, 5, 7, 9], 7) == 3\nassert binary_search([1, 3, 5], 4) == -1\nassert binary_search([], 1) == -1",
          "Har qadamda qidiruv oralig'ini ikki baravar qisqartiramiz: O(log n).",
          "Each step halves the search interval, giving O(log n)."),
    _task("bubble_sort", "Pufakcha saralash (bubble sort) algoritmini yozing.",
          "Write the bubble sort algorithm.",
          '''
def bubble_sort(items: list) -> list:
    """Return a new list sorted in ascending order (bubble sort)."""
    arr = list(items)
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr
''', "assert bubble_sort([5, 2, 9, 1]) == [1, 2, 5, 9]\nassert bubble_sort([]) == []",
          "Qo'shni elementlarni almashtirib boramiz; almashtirish bo'lmasa, ro'yxat allaqachon saralangan.",
          "Adjacent pairs are swapped; if a pass makes no swap the list is already sorted."),
    _task("merge_sorted", "Ikki saralangan ro'yxatni bitta saralangan ro'yxatga birlashtiring.",
          "Merge two sorted lists into one sorted list.",
          '''
def merge_sorted(a: list, b: list) -> list:
    """Merge two ascending lists in O(len(a) + len(b))."""
    i = j = 0
    merged = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1
    merged.extend(a[i:])
    merged.extend(b[j:])
    return merged
''', "assert merge_sorted([1, 4, 6], [2, 3, 7]) == [1, 2, 3, 4, 6, 7]\nassert merge_sorted([], [1]) == [1]",
          "Ikkita ko'rsatkich bilan kichigini olib boramiz, oxirida qolganlarini qo'shamiz.",
          "Two pointers pick the smaller head each time; leftovers are appended at the end."),
    _task("anagram", "Ikki so'z anagramma ekanligini tekshiring.",
          "Check whether two words are anagrams.",
          '''
def is_anagram(a: str, b: str) -> bool:
    """True if ``a`` and ``b`` contain the same letters (case-insensitive, spaces ignored)."""
    norm = lambda s: sorted(s.replace(" ", "").lower())
    return norm(a) == norm(b)
''', "assert is_anagram('listen', 'silent')\nassert not is_anagram('olma', 'nok')",
          "Harflarni saralab, ikki natijani solishtiramiz.", "We sort the letters and compare the results."),
    _task("second_largest", "Ro'yxatdagi ikkinchi eng katta (takrorlanmas) sonni toping.",
          "Find the second largest distinct number in a list.",
          '''
def second_largest(numbers: list):
    """Second largest distinct value, or None if it does not exist."""
    distinct = sorted(set(numbers), reverse=True)
    return distinct[1] if len(distinct) >= 2 else None
''', "assert second_largest([4, 1, 4, 3]) == 3\nassert second_largest([7]) is None",
          "Takrorlarni to'plam bilan olib tashlab, kamayish tartibida saralaymiz.",
          "Duplicates are removed with a set, then values are sorted in descending order."),
    _task("digit_sum", "Butun sonning raqamlari yig'indisini hisoblang.",
          "Compute the sum of the digits of an integer.",
          '''
def digit_sum(n: int) -> int:
    """Sum of the decimal digits of ``n`` (sign ignored)."""
    return sum(int(d) for d in str(abs(n)))
''', "assert digit_sum(12345) == 15\nassert digit_sum(-907) == 16\nassert digit_sum(0) == 0",
          "Sonni satrga aylantirib, har bir raqamni qo'shamiz.", "We convert the number to a string and add up each digit."),
    _task("leap_year", "Yil kabisa yili ekanligini aniqlaydigan funksiya yozing.",
          "Write a function that tells whether a year is a leap year.",
          '''
def is_leap_year(year: int) -> bool:
    """Gregorian leap-year rule."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
''', "assert is_leap_year(2024)\nassert not is_leap_year(1900)\nassert is_leap_year(2000)\nassert not is_leap_year(2023)",
          "4 ga bo'linadigan yillar kabisa, lekin 100 ga bo'linadiganlari 400 ga ham bo'linishi shart.",
          "Years divisible by 4 are leap years, except centuries not divisible by 400."),
    _task("capitalize_words", "Gapdagi har bir so'zning birinchi harfini katta qiling.",
          "Capitalize the first letter of every word in a sentence.",
          '''
def capitalize_words(sentence: str) -> str:
    """Capitalize the first letter of each space-separated word."""
    return " ".join(w[:1].upper() + w[1:] for w in sentence.split(" "))
''', "assert capitalize_words('salom dunyo') == 'Salom Dunyo'\nassert capitalize_words('') == ''",
          "Har bir so'zning birinchi harfini `upper()` bilan katta qilamiz, qolganini o'zgartirmaymiz.",
          "Each word's first letter is upper-cased with `upper()`; the rest is unchanged."),
    _task("transpose", "Matritsani (ro'yxatlar ro'yxati) transponirlang.",
          "Transpose a matrix given as a list of lists.",
          '''
def transpose(matrix: list) -> list:
    """Transpose a rectangular matrix."""
    return [list(row) for row in zip(*matrix)]
''', "assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]\nassert transpose([]) == []",
          "`zip(*matrix)` ustunlarni qatorlarga aylantiradi.", "`zip(*matrix)` turns columns into rows."),
    _task("running_sum", "Ro'yxatning jamlanma yig'indilarini (prefix sum) qaytaring.",
          "Return the running (prefix) sums of a list.",
          '''
def running_sum(numbers: list) -> list:
    """[a, b, c] -> [a, a + b, a + b + c]."""
    sums = []
    total = 0
    for x in numbers:
        total += x
        sums.append(total)
    return sums
''', "assert running_sum([1, 2, 3, 4]) == [1, 3, 6, 10]\nassert running_sum([]) == []",
          "Joriy yig'indini saqlab, har qadamda natijaga qo'shamiz.",
          "We keep a running total and append it at every step."),
    _task("two_sum", "Yig'indisi target ga teng bo'lgan ikki elementning indekslarini toping.",
          "Find the indices of two numbers that add up to a target.",
          '''
def two_sum(numbers: list, target: int):
    """Indices (i, j), i < j, with numbers[i] + numbers[j] == target, or None."""
    index = {}
    for j, x in enumerate(numbers):
        need = target - x
        if need in index:
            return index[need], j
        index[x] = j
    return None
''', "assert two_sum([2, 7, 11, 15], 9) == (0, 1)\nassert two_sum([1, 2], 7) is None",
          "Lug'atda avval ko'rilgan qiymatlarning indeksini saqlaymiz — bitta o'tishda O(n).",
          "A dictionary remembers indices of seen values, giving a single O(n) pass."),
    _task("char_frequency", "Satrdagi eng ko'p uchraydigan belgini toping.",
          "Find the most frequent character in a string.",
          '''
from collections import Counter


def most_common_char(text: str):
    """Most frequent character (first one on ties), or None for an empty string."""
    if not text:
        return None
    counts = Counter(text)
    return max(text, key=lambda ch: (counts[ch], -text.index(ch)))
''', "assert most_common_char('banana') == 'a'\nassert most_common_char('') is None\nassert most_common_char('abab') == 'a'",
          "`Counter` belgilarni sanaydi; tenglikda birinchi uchragan belgi tanlanadi.",
          "`Counter` counts characters; ties go to the character that appears first."),
]


def code_conversations(n: int = 500, seed: int = 0) -> List[Conversation]:
    """Coding requests (Uzbek / English) answered with a verified solution in a ```python block, a short
    explanation and an example call."""
    rng = _rng("code", seed)
    out: List[Conversation] = []
    for i in range(int(n)):
        t = CODE_TASKS[(i // 2) % len(CODE_TASKS)] if i < 2 * len(CODE_TASKS) else _pick(rng, CODE_TASKS)
        uz = i % 2 == 0
        first_test = t["tests"].split("\n")[0].replace("assert ", "", 1)
        if uz:
            intro = _pick(rng, ["Mana yechim:", "Albatta, quyidagi funksiya buni bajaradi:", "Mana sodda va tushunarli variant:"])
            ans = f"{intro}\n\n```python\n{t['code']}\n```\n\n{t['explain_uz']}\n\nTekshiruv: `{first_test}`"
            q = t["uz"]
        else:
            intro = _pick(rng, ["Here's a solution:", "Sure, this function does it:", "Here's a simple, readable version:"])
            ans = f"{intro}\n\n```python\n{t['code']}\n```\n\n{t['explain_en']}\n\nCheck: `{first_test}`"
            q = t["en"]
        out.append([_user(q), _asst(ans)])
    return out


def _exec_ok(source: str) -> Tuple[bool, str]:
    """Run ``source`` in a fresh namespace (trusted, bundled code only); returns (ok, error)."""
    try:
        exec(compile(source, "<synthetic>", "exec"), {"__name__": "__synthetic__"})
    except Exception as e:  # noqa: BLE001 - report any failure
        return False, f"{type(e).__name__}: {e}"
    return True, ""


def verify_code_tasks() -> List[str]:
    """Execute every :data:`CODE_TASKS` solution against its asserts; returns failing task keys."""
    return [t["key"] for t in CODE_TASKS if not _exec_ok(t["code"] + "\n\n" + t["tests"])[0]]


# =============================================================================================== think

def _wp_apples(rng: random.Random, uz: bool) -> Tuple[str, List[str], int, str]:
    name = _pick(rng, ["Ali", "Vali", "Malika", "Dilnoza", "Sardor", "Aziz"])
    x, y, z = rng.randint(10, 60), rng.randint(1, 9), rng.randint(2, 30)
    r = x - y + z
    if uz:
        q = f"{name}da {x} ta olma bor edi. U do'stiga {y} ta olma berdi, keyin bozordan yana {z} ta olma sotib oldi. {name}da nechta olma bo'ldi?"
        steps = [f"Boshida {x} ta olma bor edi.", f"{y} ta berdi: {x} - {y} = {x - y}.", f"{z} ta sotib oldi: {x - y} + {z} = {r}."]
        final = f"{name}da {r} ta olma bo'ldi."
    else:
        q = f"{name} had {x} apples, gave {y} to a friend and then bought {z} more at the market. How many apples does {name} have now?"
        steps = [f"Start with {x} apples.", f"After giving {y}: {x} - {y} = {x - y}.", f"After buying {z}: {x - y} + {z} = {r}."]
        final = f"{name} now has {r} apples."
    return q, steps, r, final


def _wp_change(rng: random.Random, uz: bool) -> Tuple[str, List[str], int, str]:
    qty, price = rng.randint(2, 9), rng.choice([1500, 2000, 2500, 3000, 4500, 5000, 7000, 12000])
    cost = qty * price
    bill = next(b for b in (5000, 10000, 20000, 50000, 100000, 200000) if b >= cost)
    r = bill - cost
    if uz:
        q = f"Bitta daftar {price} so'm turadi. Aziza {qty} ta daftar sotib olib, sotuvchiga {bill} so'm berdi. U qancha qaytim oladi?"
        steps = [f"Daftarlar narxi: {qty} × {price} = {cost} so'm.", f"Qaytim: {bill} - {cost} = {r} so'm."]
        final = f"Aziza {r} so'm qaytim oladi."
    else:
        q = f"A notebook costs {price} som. Aziza buys {qty} notebooks and pays with {bill} som. How much change does she get?"
        steps = [f"Total cost: {qty} × {price} = {cost} som.", f"Change: {bill} - {cost} = {r} som."]
        final = f"She gets {r} som in change."
    return q, steps, r, final


def _wp_distance(rng: random.Random, uz: bool) -> Tuple[str, List[str], int, str]:
    v, t = rng.choice([40, 50, 60, 70, 80, 90, 100]), rng.randint(2, 7)
    r = v * t
    if uz:
        q = f"Avtomobil soatiga {v} km tezlik bilan {t} soat yurdi. U qancha masofani bosib o'tdi?"
        steps = ["Masofa = tezlik × vaqt.", f"{v} × {t} = {r} km."]
        final = f"Avtomobil {r} km yo'l bosdi."
    else:
        q = f"A car travels at {v} km/h for {t} hours. How far does it go?"
        steps = ["Distance = speed × time.", f"{v} × {t} = {r} km."]
        final = f"The car travels {r} km."
    return q, steps, r, final


def _wp_share(rng: random.Random, uz: bool) -> Tuple[str, List[str], int, str]:
    k, each = rng.randint(2, 8), rng.randint(3, 25)
    total = k * each
    if uz:
        q = f"{total} ta konfetni {k} nafar bola o'rtasida teng bo'lishdi. Har biriga nechtadan konfet tegdi?"
        steps = [f"Teng bo'lish uchun bo'lamiz: {total} ÷ {k} = {each}."]
        final = f"Har bir bolaga {each} tadan konfet tegdi."
    else:
        q = f"{total} candies are shared equally among {k} children. How many does each child get?"
        steps = [f"Divide equally: {total} ÷ {k} = {each}."]
        final = f"Each child gets {each} candies."
    return q, steps, each, final


def _wp_discount(rng: random.Random, uz: bool) -> Tuple[str, List[str], int, str]:
    d = rng.choice([10, 15, 20, 25, 30, 40, 50])
    price = rng.randint(2, 40) * 10000
    off = price * d // 100
    r = price - off
    if uz:
        q = f"Kurtka {price} so'm turadi. Unga {d}% chegirma berildi. Chegirmadan keyingi narx qancha?"
        steps = [f"Chegirma miqdori: {price} × {d} / 100 = {off} so'm.", f"Yangi narx: {price} - {off} = {r} so'm."]
        final = f"Chegirmadan keyin kurtka {r} so'm turadi."
    else:
        q = f"A jacket costs {price} som and is {d}% off. What is the price after the discount?"
        steps = [f"Discount: {price} × {d} / 100 = {off} som.", f"New price: {price} - {off} = {r} som."]
        final = f"After the discount the jacket costs {r} som."
    return q, steps, r, final


def _wp_rect(rng: random.Random, uz: bool) -> Tuple[str, List[str], int, str]:
    l, w = rng.randint(3, 40), rng.randint(2, 30)
    if rng.random() < 0.5:
        r = l * w
        if uz:
            q, steps, final = (f"To'g'ri to'rtburchakning uzunligi {l} m, eni {w} m. Uning yuzi qancha?",
                               ["Yuz = uzunlik × en.", f"{l} × {w} = {r} m²."], f"Yuzi {r} m².")
        else:
            q, steps, final = (f"A rectangle is {l} m long and {w} m wide. What is its area?",
                               ["Area = length × width.", f"{l} × {w} = {r} m²."], f"The area is {r} m².")
    else:
        r = 2 * (l + w)
        if uz:
            q, steps, final = (f"To'g'ri to'rtburchakning uzunligi {l} m, eni {w} m. Uning perimetri qancha?",
                               ["Perimetr = 2 × (uzunlik + en).", f"2 × ({l} + {w}) = {r} m."], f"Perimetri {r} m.")
        else:
            q, steps, final = (f"A rectangle is {l} m long and {w} m wide. What is its perimeter?",
                               ["Perimeter = 2 × (length + width).", f"2 × ({l} + {w}) = {r} m."],
                               f"The perimeter is {r} m.")
    return q, steps, r, final


def _wp_pages(rng: random.Random, uz: bool) -> Tuple[str, List[str], int, str]:
    pages, per_day = rng.randint(80, 400), rng.choice([15, 20, 25, 30, 40, 50])
    days = -(-pages // per_day)
    full, rest = divmod(pages, per_day)
    if uz:
        q = f"Kitob {pages} betdan iborat. Sevara har kuni {per_day} bet o'qiydi. U kitobni necha kunda tugatadi?"
        steps = [f"{pages} ÷ {per_day} = {full}, qoldiq {rest}."]
        steps.append(f"Qoldiq bor, demak yana bir kun kerak: {full} + 1 = {days}." if rest else f"Qoldiq yo'q, demak {days} kun.")
        final = f"Sevara kitobni {days} kunda tugatadi."
    else:
        q = f"A book has {pages} pages. Sevara reads {per_day} pages a day. How many days does she need to finish it?"
        steps = [f"{pages} ÷ {per_day} = {full} remainder {rest}."]
        steps.append(f"There is a remainder, so one more day is needed: {full} + 1 = {days}." if rest
                     else f"No remainder, so {days} days.")
        final = f"She needs {days} days."
    return q, steps, days, final


def _wp_age(rng: random.Random, uz: bool) -> Tuple[str, List[str], int, str]:
    x, k, y = rng.randint(5, 30), rng.randint(2, 15), rng.randint(2, 10)
    r = x + (x + k) + 2 * y
    if uz:
        q = f"Jasur {x} yoshda, akasi undan {k} yosh katta. {y} yildan keyin ularning yoshlari yig'indisi nechaga teng bo'ladi?"
        steps = [f"Akasi hozir {x} + {k} = {x + k} yoshda.", f"Hozirgi yig'indi: {x} + {x + k} = {2 * x + k}.",
                 f"{y} yildan keyin har biri {y} yoshga katta bo'ladi: {2 * x + k} + 2 × {y} = {r}."]
        final = f"{y} yildan keyin yoshlari yig'indisi {r} bo'ladi."
    else:
        q = f"Jasur is {x} years old and his brother is {k} years older. What will the sum of their ages be in {y} years?"
        steps = [f"The brother is {x} + {k} = {x + k} years old.", f"Current sum: {x} + {x + k} = {2 * x + k}.",
                 f"In {y} years each is {y} years older: {2 * x + k} + 2 × {y} = {r}."]
        final = f"In {y} years the sum of their ages will be {r}."
    return q, steps, r, final


_WORD_PROBLEMS = [_wp_apples, _wp_change, _wp_distance, _wp_share, _wp_discount, _wp_rect, _wp_pages, _wp_age]


def word_problem(rng: random.Random, uz: bool) -> Tuple[str, List[str], int, str]:
    """One random word problem: ``(question, reasoning_steps, answer, final_sentence)``."""
    return _pick(rng, _WORD_PROBLEMS)(rng, uz)


def think_conversations(n: int = 500, seed: int = 0) -> List[Conversation]:
    """Arithmetic word problems answered as ``<|think|>steps<|/think|>final sentence\\nJavob/Answer: N``."""
    rng = _rng("think", seed)
    out: List[Conversation] = []
    for i in range(int(n)):
        uz = i % 2 == 0
        q, steps, r, final = word_problem(rng, uz)
        label = "Javob" if uz else "Answer"
        out.append([_user(q), _asst("<|think|>" + "\n".join(steps) + f"<|/think|>{final}\n{label}: {r}")])
    return out


# =============================================================================================== plans

def _S(uz: Tuple[str, str, str], en: Tuple[str, str, str]) -> Dict[str, Tuple[str, str, str]]:
    """Plan step: (action, expected output, check) in both languages."""
    return {"uz": uz, "en": en}


def _R(uz: str, en: str, step: Dict[str, Tuple[str, str, str]], pos: int) -> Dict[str, Any]:
    """Optional user requirement satisfied by ``step`` inserted before base step index ``pos``."""
    return {"uz": uz, "en": en, "step": step, "pos": pos}


def _C(uz: str, en: str, base_step: int) -> Dict[str, Any]:
    """User constraint honoured mainly by base step ``base_step``."""
    return {"uz": uz, "en": en, "base": base_step}


_PLAN_DOMAINS: List[Dict[str, Any]] = [
    {
        "task": ("Python'da vazifalar ro'yxatini boshqaradigan buyruq qatori (CLI) dasturini yaratish rejasini tuzib bering.",
                 "Make a plan for building a command-line to-do list app in Python."),
        "goal": ("vazifalarni qo'shish, ko'rish va bajarilgan deb belgilash mumkin bo'lgan ishonchli CLI dastur",
                 "a reliable CLI app to add, list and complete tasks"),
        "unknowns": [("Vazifalar qayerda saqlanishi kerakligi (fayl yoki ma'lumotlar bazasi) aniq aytilmagan.",
                      "Where tasks should be stored (file or database) is not specified."),
                     ("Bir nechta foydalanuvchi bo'lishi kerakmi — noma'lum.", "Whether multiple users are needed is unknown.")],
        "risks": [("Fayl buzilsa, ma'lumotlar yo'qolishi mumkin.", "A corrupted file could lose data."),
                  ("Buyruqlar nomi chalkash bo'lsa, foydalanish qiyinlashadi.", "Confusing command names hurt usability.")],
        "steps": [
            _S(("Buyruqlarni aniqlash: add, list, done", "buyruqlar ro'yxati va misollar", "har bir buyruq uchun kutilgan natija yozilgan"),
               ("Define the commands: add, list, done", "a command list with examples", "every command has an expected result")),
            _S(("argparse yordamida dastur skeletini yozish", "todo.py fayli", "`python todo.py --help` ishlaydi"),
               ("Write the program skeleton with argparse", "todo.py", "`python todo.py --help` works")),
            _S(("Vazifalarni JSON faylda saqlash va o'qish funksiyalarini yozish", "load/save funksiyalari", "saqlangan vazifa qayta o'qilganda o'zgarmaydi"),
               ("Implement loading and saving tasks in a JSON file", "load/save functions", "a saved task reads back unchanged")),
            _S(("pytest bilan testlar yozish", "test_todo.py", "barcha testlar o'tadi"),
               ("Write tests with pytest", "test_todo.py", "all tests pass")),
            _S(("README yozish va foydalanish misollarini qo'shish", "README.md", "yangi foydalanuvchi misollar bo'yicha dasturni ishga tushira oladi"),
               ("Write a README with usage examples", "README.md", "a new user can run the app from the examples")),
        ],
        "reqs": [
            _R("Vazifalarga muddat (deadline) qo'shish mumkin bo'lsin.", "Tasks can have a due date.",
               _S(("add buyrug'iga --due parametrini qo'shish va sanani tekshirish", "muddatli vazifalar", "noto'g'ri sana xato xabarini beradi"),
                  ("Add a --due option to add and validate the date", "tasks with due dates", "an invalid date gives an error message")), 3),
            _R("Vazifani o'chirish buyrug'i bo'lsin.", "There must be a command to delete a task.",
               _S(("delete buyrug'ini qo'shish", "vazifani raqami bo'yicha o'chirish", "o'chirilgan vazifa ro'yxatda ko'rinmaydi"),
                  ("Add a delete command", "deleting a task by its number", "a deleted task no longer appears in the list")), 3),
            _R("Ro'yxatni faqat bajarilmagan vazifalar bilan ko'rsatish imkoni bo'lsin.", "It should be possible to list only unfinished tasks.",
               _S(("list buyrug'iga --pending filtrini qo'shish", "filtrlangan ro'yxat", "bajarilgan vazifalar filtrda chiqmaydi"),
                  ("Add a --pending filter to list", "a filtered list", "completed tasks are hidden by the filter")), 3),
            _R("Dastur rangli chiqish bermasin, oddiy matn bo'lsin.", "Output must be plain text without colors.",
               _S(("Chiqishni oddiy matnda formatlash funksiyasini yozish", "format_task funksiyasi", "chiqishda ANSI kodlari yo'q"),
                  ("Write a plain-text formatting function", "a format_task function", "the output has no ANSI codes")), 2),
        ],
        "constraints": [_C("Faqat Python standart kutubxonasidan foydalanish kerak.", "Only the Python standard library may be used.", 1),
                        _C("Ish 3 kun ichida tugashi kerak.", "The work must be finished within 3 days.", 0)],
    },
    {
        "task": ("Imtihonga 4 hafta qoldi. Matematika va ingliz tilidan o'qish jadvalini rejalashtirishga yordam bering.",
                 "My exam is in 4 weeks. Help me plan a study schedule for math and English."),
        "goal": ("4 hafta ichida matematika va ingliz tilidan imtihonga tizimli tayyorlanish",
                 "a structured 4-week preparation for math and English"),
        "unknowns": [("Kuniga qancha vaqt ajratish mumkinligi aniq emas.", "How much time per day is available is unclear."),
                     ("Qaysi mavzular eng qiyinligi hali aniqlanmagan.", "Which topics are hardest is not yet known.")],
        "risks": [("Charchoq tufayli jadvaldan chetga chiqish mumkin.", "Fatigue may break the schedule."),
                  ("Bitta fanga ko'p vaqt ketib, ikkinchisi orqada qolishi mumkin.", "One subject may crowd out the other.")],
        "steps": [
            _S(("Diagnostik test yechib, kuchsiz mavzularni aniqlash", "kuchsiz mavzular ro'yxati", "har bir fan bo'yicha kamida 3 ta mavzu belgilangan"),
               ("Take a diagnostic test to find weak topics", "a list of weak topics", "at least 3 topics per subject are marked")),
            _S(("Haftalik jadval tuzish: har kuni ikkala fanga vaqt ajratish", "4 haftalik jadval", "har kuni ikkala fan jadvalda bor"),
               ("Build a weekly timetable with time for both subjects every day", "a 4-week timetable", "both subjects appear every day")),
            _S(("Har hafta oxirida sinov testi yechish", "haftalik natijalar", "ballar jadvalga yozib boriladi"),
               ("Take a practice test at the end of each week", "weekly scores", "scores are recorded")),
            _S(("Natijalarga qarab jadvalni moslashtirish", "yangilangan jadval", "eng past ballli mavzuga ko'proq vaqt ajratilgan"),
               ("Adjust the timetable based on results", "an updated timetable", "the lowest-scoring topic gets more time")),
            _S(("Oxirgi haftada umumiy takrorlash va to'liq imtihon simulyatsiyasi", "to'liq sinov natijasi", "simulyatsiya imtihon vaqtida o'tkazilgan"),
               ("Final week: full review and a mock exam", "a full mock-exam score", "the mock exam is done under exam timing")),
        ],
        "reqs": [
            _R("Har kuni ingliz tilidan 20 ta yangi so'z yodlash bo'lsin.", "Learn 20 new English words every day.",
               _S(("Kunlik 20 so'zlik lug'at kartochkalarini tayyorlash", "so'z kartochkalari", "har kuni 20 ta so'z takrorlanadi"),
                  ("Prepare daily flashcards with 20 words", "flashcards", "20 words are reviewed every day")), 2),
            _R("Yakshanba dam olish kuni bo'lsin.", "Sunday must be a rest day.",
               _S(("Yakshanbani jadvaldan bo'shatish", "dam olish kuni", "yakshanbaga hech qanday mashg'ulot qo'yilmagan"),
                  ("Keep Sunday free in the timetable", "a rest day", "no sessions are scheduled on Sunday")), 2),
            _R("Matematikadan har kuni kamida 10 ta masala yechilsin.", "Solve at least 10 math problems every day.",
               _S(("Kunlik 10 ta masaladan iborat to'plam tanlash", "masalalar to'plami", "har kuni yechilgan masalalar soni yozib boriladi"),
                  ("Pick a daily set of 10 math problems", "problem sets", "the number of solved problems is logged daily")), 2),
        ],
        "constraints": [_C("Kuniga 2 soatdan ko'p vaqt yo'q.", "No more than 2 hours a day are available.", 1),
                        _C("Darsliklar faqat bepul manbalardan bo'lishi kerak.", "Study materials must be free.", 0)],
    },
    {
        "task": ("Kitoblar do'koni uchun REST API loyihalash rejasini tuzing.", "Plan the design of a REST API for a bookstore."),
        "goal": ("kitoblar va buyurtmalarni boshqaradigan aniq va xavfsiz REST API",
                 "a clear and secure REST API for books and orders"),
        "unknowns": [("Kutilayotgan yuklama (so'rovlar soni) noma'lum.", "The expected load is unknown."),
                     ("To'lov tizimi bilan integratsiya kerakmi — aniq emas.", "Whether payment integration is needed is unclear.")],
        "risks": [("Noto'g'ri autentifikatsiya ma'lumotlar sizib chiqishiga olib keladi.", "Weak authentication could leak data."),
                  ("API versiyalanmasa, mijozlar buziladi.", "Without versioning, clients may break.")],
        "steps": [
            _S(("Resurslarni aniqlash: books, authors, orders", "resurslar va maydonlar ro'yxati", "har bir resursning majburiy maydonlari belgilangan"),
               ("Identify resources: books, authors, orders", "resources with their fields", "required fields are defined for each resource")),
            _S(("Endpointlarni loyihalash (GET/POST/PUT/DELETE, /v1 prefiksi bilan)", "endpointlar jadvali", "har bir endpoint uchun status kodlari yozilgan"),
               ("Design endpoints (GET/POST/PUT/DELETE under /v1)", "an endpoint table", "status codes are listed for every endpoint")),
            _S(("OpenAPI spetsifikatsiyasini yozish", "openapi.yaml", "spetsifikatsiya validatordan o'tadi"),
               ("Write the OpenAPI specification", "openapi.yaml", "the spec passes a validator")),
            _S(("Xatolar formatini yagona qilish", "xato javobi namunasi", "barcha xatolar bir xil JSON tuzilmada qaytadi"),
               ("Standardize the error format", "an error response example", "all errors share one JSON structure")),
            _S(("Integratsion testlar rejasini tuzish", "testlar ro'yxati", "har bir endpoint kamida bitta test bilan qoplangan"),
               ("Plan integration tests", "a test list", "every endpoint is covered by at least one test")),
        ],
        "reqs": [
            _R("Kitoblar ro'yxatida sahifalash (pagination) bo'lsin.", "The book list must support pagination.",
               _S(("GET /v1/books uchun page va limit parametrlarini qo'shish", "sahifalangan javob", "limit 100 dan oshmaydi va keyingi sahifa havolasi bor"),
                  ("Add page and limit parameters to GET /v1/books", "a paginated response", "limit is capped at 100 and a next link exists")), 2),
            _R("Buyurtma berish faqat avtorizatsiyadan o'tgan foydalanuvchilarga ruxsat etilsin.", "Only authenticated users may place orders.",
               _S(("POST /v1/orders uchun token asosidagi autentifikatsiyani belgilash", "himoyalangan endpoint", "tokensiz so'rov 401 qaytaradi"),
                  ("Require token authentication for POST /v1/orders", "a protected endpoint", "a request without a token returns 401")), 2),
            _R("Kitoblarni muallif bo'yicha qidirish mumkin bo'lsin.", "Books can be searched by author.",
               _S(("GET /v1/books ga author filtrini qo'shish", "filtrlangan qidiruv", "filtr faqat mos kitoblarni qaytaradi"),
                  ("Add an author filter to GET /v1/books", "filtered search", "the filter returns only matching books")), 2),
        ],
        "constraints": [_C("Javoblar faqat JSON formatida bo'lsin.", "Responses must be JSON only.", 3),
                        _C("API ochiq hujjatlashtirilgan bo'lishi shart.", "The API must be publicly documented.", 2)],
    },
    {
        "task": ("Maktabimizda 100 kishilik ilmiy konferensiya tashkil qilish rejasini tuzib bering.",
                 "Help me plan a science conference for 100 people at our school."),
        "goal": ("100 ishtirokchi uchun muammosiz o'tadigan ilmiy konferensiya", "a smooth science conference for 100 participants"),
        "unknowns": [("Aniq byudjet hali tasdiqlanmagan.", "The exact budget is not yet approved."),
                     ("Ma'ruzachilar soni noma'lum.", "The number of speakers is unknown.")],
        "risks": [("Ma'ruzachi oxirgi daqiqada kelmasligi mumkin.", "A speaker might cancel at the last minute."),
                  ("Texnik jihozlar ishlamay qolishi mumkin.", "Equipment could fail.")],
        "steps": [
            _S(("Sana, joy va byudjetni tasdiqlash", "tasdiqlangan sana va zal", "zal 100 kishiga mo'ljallangan"),
               ("Confirm date, venue and budget", "a confirmed date and hall", "the hall fits 100 people")),
            _S(("Ma'ruzachilarni taklif qilish va dastur tuzish", "konferensiya dasturi", "har bir ma'ruzaga vaqt ajratilgan"),
               ("Invite speakers and draft the programme", "a conference programme", "every talk has a time slot")),
            _S(("Ro'yxatdan o'tish shaklini ochish", "ro'yxatdan o'tish havolasi", "ishtirokchilar soni kunlik kuzatiladi"),
               ("Open a registration form", "a registration link", "the participant count is tracked daily")),
            _S(("Texnik jihozlarni tekshirish (proyektor, mikrofon)", "jihozlar ro'yxati", "tadbirdan bir kun oldin sinov o'tkazilgan"),
               ("Check equipment (projector, microphones)", "an equipment checklist", "a test run is done the day before")),
            _S(("Tadbirdan keyin fikr-mulohaza so'rovnomasini yuborish", "so'rovnoma natijalari", "kamida 50% ishtirokchi javob bergan"),
               ("Send a feedback survey after the event", "survey results", "at least 50% of participants respond")),
        ],
        "reqs": [
            _R("Tushlik uchun tanaffus bo'lsin.", "There must be a lunch break.",
               _S(("Dasturga 1 soatlik tushlik tanaffusini qo'shish va ovqat buyurtma qilish", "tushlik rejasi", "100 kishilik ovqat tasdiqlangan"),
                  ("Add a 1-hour lunch break and order food", "a lunch plan", "food for 100 is confirmed")), 2),
            _R("Ishtirokchilarga sertifikat berilsin.", "Participants should receive certificates.",
               _S(("Sertifikat shablonini tayyorlash va chop etish", "sertifikatlar", "har bir ro'yxatdan o'tgan ishtirokchiga sertifikat bor"),
                  ("Prepare and print certificates", "certificates", "every registered participant has a certificate")), 4),
            _R("Konferensiya onlayn ham translyatsiya qilinsin.", "The conference should also be streamed online.",
               _S(("Onlayn translyatsiya platformasini sozlash", "translyatsiya havolasi", "sinov efiri muvaffaqiyatli o'tgan"),
                  ("Set up a live-streaming platform", "a streaming link", "a test stream succeeded")), 4),
        ],
        "constraints": [_C("Byudjet 5 million so'mdan oshmasin.", "The budget must not exceed 5 million som.", 0),
                        _C("Tadbir bir kunda o'tishi kerak.", "The event must fit into one day.", 1)],
    },
    {
        "task": ("Ilovamiz ma'lumotlar bazasini SQLite'dan PostgreSQL'ga ko'chirish rejasini tuzing.",
                 "Plan a migration of our application's database from SQLite to PostgreSQL."),
        "goal": ("ma'lumotlarni yo'qotmasdan va minimal to'xtash bilan PostgreSQL'ga o'tish",
                 "moving to PostgreSQL without data loss and with minimal downtime"),
        "unknowns": [("Ma'lumotlar hajmi va o'sish sur'ati aniq emas.", "Data size and growth rate are unclear."),
                     ("Ilova SQLite'ga xos funksiyalardan foydalanadimi — tekshirilmagan.", "Whether the app relies on SQLite-specific features is unchecked.")],
        "risks": [("Turlar mos kelmasligi (masalan, sana va boolean) xatolarga olib keladi.", "Type mismatches (dates, booleans) may cause bugs."),
                  ("Ko'chirish paytida yangi yozuvlar yo'qolishi mumkin.", "New writes during the migration could be lost.")],
        "steps": [
            _S(("Joriy sxema va SQL so'rovlarini inventarizatsiya qilish", "jadvallar va so'rovlar ro'yxati", "barcha jadvallar hisobga olingan"),
               ("Inventory the current schema and SQL queries", "a list of tables and queries", "every table is accounted for")),
            _S(("PostgreSQL sxemasini yaratish va turlarni moslashtirish", "migratsiya skripti", "skript bo'sh bazada xatosiz ishlaydi"),
               ("Create the PostgreSQL schema and map types", "a migration script", "the script runs cleanly on an empty database")),
            _S(("Ma'lumotlarni nusxalash skriptini yozish", "ko'chirish skripti", "har bir jadvalda qatorlar soni mos keladi"),
               ("Write the data copy script", "a copy script", "row counts match for every table")),
            _S(("Ilovani staging muhitida PostgreSQL bilan sinash", "test hisobotlari", "barcha avtomatik testlar o'tadi"),
               ("Test the app against PostgreSQL in staging", "test reports", "all automated tests pass")),
            _S(("Ishlab chiqarishga o'tish va kuzatish", "yangi baza ishlamoqda", "birinchi 24 soatda xatolar darajasi oshmagan"),
               ("Cut over to production and monitor", "the new database in service", "error rates do not rise in the first 24 hours")),
        ],
        "reqs": [
            _R("Orqaga qaytish (rollback) rejasi bo'lsin.", "There must be a rollback plan.",
               _S(("SQLite zaxira nusxasi va qaytish tartibini yozish", "rollback qo'llanmasi", "qaytish staging'da sinab ko'rilgan"),
                  ("Keep a SQLite backup and document the rollback procedure", "a rollback runbook", "the rollback is rehearsed in staging")), 4),
            _R("To'xtash vaqti 30 daqiqadan oshmasin.", "Downtime must stay under 30 minutes.",
               _S(("Ko'chirish vaqtini o'lchash uchun sinov o'tkazish", "o'lchangan davomiylik", "sinovda ko'chirish 30 daqiqadan kam vaqt olgan"),
                  ("Do a timed dry run of the migration", "a measured duration", "the dry run takes under 30 minutes")), 3),
            _R("Ma'lumotlar yaxlitligi tekshirilsin.", "Data integrity must be verified.",
               _S(("Har bir jadval uchun nazorat yig'indilarini solishtirish", "yaxlitlik hisoboti", "barcha nazorat yig'indilari mos"),
                  ("Compare checksums for every table", "an integrity report", "all checksums match")), 3),
        ],
        "constraints": [_C("Ish faqat dam olish kunida bajarilishi mumkin.", "The cut-over can only happen on a weekend.", 4),
                        _C("Ilova kodi minimal o'zgarishi kerak.", "Application code changes must be minimal.", 1)],
    },
    {
        "task": ("Onlayn do'konning to'lov sahifasi uchun test rejasini tuzing.", "Write a test plan for an online shop's checkout page."),
        "goal": ("to'lov sahifasidagi xatolarni chiqarishdan oldin topish", "catching checkout bugs before release"),
        "unknowns": [("Qaysi brauzerlar qo'llab-quvvatlanishi kerakligi aniq emas.", "Which browsers must be supported is unclear."),
                     ("Test to'lov tizimi mavjudmi — noma'lum.", "Whether a payment sandbox exists is unknown.")],
        "risks": [("Haqiqiy kartadan pul yechilib qolishi mumkin.", "A real card could be charged by mistake."),
                  ("Kam uchraydigan holatlar e'tibordan chetda qolishi mumkin.", "Edge cases may be missed.")],
        "steps": [
            _S(("Test qamrovini aniqlash: savat, manzil, to'lov, tasdiq", "qamrov ro'yxati", "har bir bosqich uchun mas'ul belgilangan"),
               ("Define scope: cart, address, payment, confirmation", "a scope list", "each stage has an owner")),
            _S(("Ijobiy test holatlarini yozish", "test holatlari jadvali", "har bir holatda kutilgan natija bor"),
               ("Write positive test cases", "a test case table", "every case has an expected result")),
            _S(("Salbiy va chegara holatlarini yozish (bo'sh maydon, noto'g'ri karta)", "salbiy testlar", "har bir maydon uchun kamida bitta salbiy test"),
               ("Write negative and boundary cases (empty fields, invalid card)", "negative tests", "each field has at least one negative test")),
            _S(("Testlarni test to'lov muhitida bajarish", "natijalar hisoboti", "barcha muhim testlar o'tgan"),
               ("Run the tests in the payment sandbox", "a results report", "all critical tests pass")),
            _S(("Topilgan xatolarni qayd etish va qayta tekshirish", "xatolar ro'yxati", "tuzatilgan xatolar qayta sinovdan o'tgan"),
               ("Log defects and retest fixes", "a defect list", "fixed defects are retested")),
        ],
        "reqs": [
            _R("Mobil qurilmalarda ham test qilinsin.", "Testing must include mobile devices.",
               _S(("Android va iOS qurilmalarida asosiy holatlarni bajarish", "mobil test natijalari", "ikkala platformada ham to'lov yakunlangan"),
                  ("Run the main cases on Android and iOS", "mobile test results", "checkout completes on both platforms")), 4),
            _R("Promo-kod kiritish ham tekshirilsin.", "Promo codes must be tested too.",
               _S(("Promo-kod uchun to'g'ri, eskirgan va noto'g'ri kod holatlarini yozish", "promo-kod testlari", "chegirma summasi to'g'ri hisoblanadi"),
                  ("Write valid, expired and invalid promo-code cases", "promo-code tests", "the discount amount is computed correctly")), 3),
            _R("Sahifa 3 soniyadan tez yuklanishi tekshirilsin.", "Verify that the page loads in under 3 seconds.",
               _S(("Yuklanish tezligini o'lchash", "tezlik hisoboti", "o'rtacha yuklanish 3 soniyadan kam"),
                  ("Measure page load time", "a performance report", "average load time is under 3 seconds")), 4),
        ],
        "constraints": [_C("Haqiqiy kartalardan foydalanish taqiqlanadi.", "Real cards must never be used.", 3),
                        _C("Test rejasi 2 kunda tayyor bo'lsin.", "The test plan must be ready in 2 days.", 0)],
    },
    {
        "task": ("Savdo ma'lumotlari (CSV) asosida oylik hisobot tayyorlash rejasini tuzing.",
                 "Plan how to produce a monthly report from sales data in a CSV file."),
        "goal": ("CSV savdo ma'lumotlaridan aniq va takrorlanadigan oylik hisobot", "an accurate, repeatable monthly report from CSV sales data"),
        "unknowns": [("CSV ustunlari nomi va formati to'liq ma'lum emas.", "The CSV column names and formats are not fully known."),
                     ("Hisobotni kim o'qishi aniq emas.", "Who will read the report is unclear.")],
        "risks": [("Takroriy yoki bo'sh qatorlar natijani buzishi mumkin.", "Duplicate or empty rows can skew results."),
                  ("Valyuta yoki sana formati har xil bo'lishi mumkin.", "Currency or date formats may vary.")],
        "steps": [
            _S(("CSV faylni o'qib, ustunlarni tekshirish", "ustunlar tavsifi", "majburiy ustunlar mavjud"),
               ("Load the CSV and check the columns", "a column description", "required columns are present")),
            _S(("Ma'lumotlarni tozalash: takrorlar va bo'sh qiymatlar", "tozalangan jadval", "takroriy qatorlar soni 0"),
               ("Clean the data: duplicates and missing values", "a clean table", "zero duplicate rows remain")),
            _S(("Asosiy ko'rsatkichlarni hisoblash: jami savdo, o'rtacha chek", "ko'rsatkichlar jadvali", "jami summa qo'lda tekshirilgan namunaga mos"),
               ("Compute key metrics: total sales, average order", "a metrics table", "the total matches a hand-checked sample")),
            _S(("Grafiklar tayyorlash", "kunlik savdo grafigi", "o'qlar va sarlavhalar to'g'ri belgilangan"),
               ("Create charts", "a daily sales chart", "axes and titles are labelled")),
            _S(("Hisobotni yozish va yuborish", "PDF hisobot", "hisobotda xulosa va tavsiyalar bor"),
               ("Write and send the report", "a PDF report", "the report has a summary and recommendations")),
        ],
        "reqs": [
            _R("Eng ko'p sotilgan 5 ta mahsulot ko'rsatilsin.", "Show the top 5 best-selling products.",
               _S(("Mahsulotlar bo'yicha guruhlab, top-5 ni aniqlash", "top-5 jadvali", "jadvalda aynan 5 ta mahsulot"),
                  ("Group by product and find the top 5", "a top-5 table", "the table lists exactly 5 products")), 3),
            _R("O'tgan oy bilan taqqoslash bo'lsin.", "Include a comparison with the previous month.",
               _S(("O'tgan oy ko'rsatkichlarini hisoblab, o'zgarish foizini topish", "taqqoslash jadvali", "o'zgarish foizlari to'g'ri hisoblangan"),
                  ("Compute last month's metrics and the percent change", "a comparison table", "percent changes are correct")), 3),
            _R("Hisobot har oy avtomatik tayyorlansin.", "The report must be generated automatically every month.",
               _S(("Skriptni rejalashtiruvchi (cron) orqali oylik ishga tushirish", "avtomatik vazifa", "sinov ishga tushirish muvaffaqiyatli"),
                  ("Schedule the script monthly with cron", "an automated job", "a test run succeeds")), 5),
        ],
        "constraints": [_C("Python va pandas'dan foydalanish kerak.", "Use Python and pandas.", 0),
                        _C("Hisobot 2 sahifadan oshmasin.", "The report must not exceed 2 pages.", 4)],
    },
    {
        "task": ("Shaxsiy portfolio veb-saytini yaratish rejasini tuzib bering.", "Make a plan for building a personal portfolio website."),
        "goal": ("ishlarimni chiroyli va tez yuklanadigan sahifada ko'rsatish", "showing my work on a clean, fast-loading site"),
        "unknowns": [("Hosting qayerda bo'lishi tanlanmagan.", "The hosting provider is not chosen."),
                     ("Nechta loyiha ko'rsatilishi aniq emas.", "How many projects to show is unclear.")],
        "risks": [("Dizaynga juda ko'p vaqt ketib, kontent kechikishi mumkin.", "Design work may delay the content."),
                  ("Sayt mobil qurilmalarda yomon ko'rinishi mumkin.", "The site might look bad on phones.")],
        "steps": [
            _S(("Sahifalar tuzilmasini rejalash: bosh sahifa, loyihalar, aloqa", "sayt xaritasi", "har bir sahifaning maqsadi yozilgan"),
               ("Plan the structure: home, projects, contact", "a site map", "each page has a stated purpose")),
            _S(("Kontentni tayyorlash: tarjimai hol va loyiha tavsiflari", "matnlar va rasmlar", "har bir loyiha uchun tavsif va rasm bor"),
               ("Prepare content: bio and project descriptions", "texts and images", "every project has a description and image")),
            _S(("HTML/CSS bilan sahifalarni yozish", "sayt fayllari", "sahifalar brauzerda xatosiz ochiladi"),
               ("Build the pages with HTML/CSS", "site files", "pages open without errors in a browser")),
            _S(("Saytni hostingga joylash", "ishlayotgan havola", "sayt boshqa qurilmadan ochiladi"),
               ("Deploy the site to hosting", "a live URL", "the site opens from another device")),
        ],
        "reqs": [
            _R("Sayt mobil telefonlarda ham yaxshi ko'rinsin.", "The site must look good on phones.",
               _S(("Moslashuvchan (responsive) dizaynni qo'shish va tekshirish", "mobil versiya", "375 px kenglikda gorizontal aylantirish yo'q"),
                  ("Add and test a responsive layout", "a mobile version", "no horizontal scrolling at 375 px width")), 3),
            _R("Aloqa formasi bo'lsin.", "There must be a contact form.",
               _S(("Aloqa formasini qo'shish va xabar yuborishni sinash", "ishlaydigan forma", "sinov xabari pochtaga yetib kelgan"),
                  ("Add a contact form and test sending a message", "a working form", "a test message arrives by e-mail")), 3),
            _R("Sayt ikki tilda (o'zbek va ingliz) bo'lsin.", "The site must be bilingual (Uzbek and English).",
               _S(("Matnlarni ikki tilda tayyorlash va til almashtirgich qo'shish", "ikki tilli sahifalar", "har bir sahifa ikkala tilda ochiladi"),
                  ("Prepare texts in both languages and add a language switch", "bilingual pages", "every page opens in both languages")), 2),
        ],
        "constraints": [_C("Hosting bepul bo'lishi kerak.", "Hosting must be free.", 3),
                        _C("Sayt 1 haftada tayyor bo'lishi kerak.", "The site must be ready in 1 week.", 0)],
    },
]


def _plan_example(rng: random.Random, uz: bool, drop_step: bool = False) -> Tuple[str, str, Dict[str, Any]]:
    """Compose one planning prompt + PLAN FORMAT answer. Returns ``(prompt, answer, meta)``."""
    lang = "uz" if uz else "en"
    li = 0 if uz else 1
    dom = _pick(rng, _PLAN_DOMAINS)
    reqs = rng.sample(dom["reqs"], rng.randint(1, min(3, len(dom["reqs"]))))
    cons = rng.sample(dom["constraints"], rng.randint(0, len(dom["constraints"])))
    # steps: base steps with requirement steps inserted before their position (stable)
    entries: List[Tuple[float, Dict[str, Any], Optional[int]]] = [(float(i), s, None) for i, s in enumerate(dom["steps"])]
    for k, r in enumerate(sorted(reqs, key=lambda r: r["pos"])):
        entries.append((r["pos"] - 0.5 + k * 0.01, r["step"], k))
    entries.sort(key=lambda e: e[0])
    req_sorted = sorted(reqs, key=lambda r: r["pos"])
    base_to_num: Dict[int, int] = {}
    req_to_num: Dict[int, int] = {}
    lines = []
    for num, (key, step, rk) in enumerate(entries, 1):
        action, output, chk = step[lang]
        if rk is None:
            base_to_num[int(key)] = num
        else:
            req_to_num[rk] = num
        lines.append(f"{num}. {action}. " + (f"Natija: {output}. Tekshiruv: {chk}." if uz else
                                              f"Output: {output}. Check: {chk}."))
    removed = None
    if drop_step and req_to_num:  # flawed variant for preference pairs: silently omit a required step
        removed = sorted(req_to_num.values())[0]
        lines = [ln for ln in lines if not ln.startswith(f"{removed}. ")]
    # prompt
    if uz:
        prompt = dom["task"][0] + "\nTalablar:\n" + "\n".join(f"- {r['uz']}" for r in req_sorted)
        if cons:
            prompt += "\nCheklovlar: " + " ".join(c["uz"] for c in cons)
    else:
        prompt = dom["task"][1] + "\nRequirements:\n" + "\n".join(f"- {r['en']}" for r in req_sorted)
        if cons:
            prompt += "\nConstraints: " + " ".join(c["en"] for c in cons)
    unknown = _pick(rng, dom["unknowns"])[li]
    risk = _pick(rng, dom["risks"])[li]
    cons_txt = " ".join(c[lang] for c in cons) or ("Aniq cheklov aytilmagan." if uz else "No explicit constraints given.")
    if uz:
        think = (f"Maqsad: {dom['goal'][0]}.\nCheklovlar: {cons_txt}\nNoma'lumlar: {unknown} Buni birinchi qadamda "
                 f"aniqlashtiraman.\nXavflar: {risk}")
        verif = [f"- {r['uz']} → {req_to_num[k]}-qadam" for k, r in enumerate(req_sorted)]
        verif += [f"- {c['uz']} → {base_to_num[c['base']]}-qadam" for c in cons]
        answer = f"<|think|>{think}<|/think|>**Reja**\n" + "\n".join(lines) + "\n\n**Tekshiruv**\n" + "\n".join(verif)
    else:
        think = (f"Goal: {dom['goal'][1]}.\nConstraints: {cons_txt}\nUnknowns: {unknown} I will clarify this in the "
                 f"first step.\nRisks: {risk}")
        verif = [f"- {r['en']} → step {req_to_num[k]}" for k, r in enumerate(req_sorted)]
        verif += [f"- {c['en']} → step {base_to_num[c['base']]}" for c in cons]
        answer = f"<|think|>{think}<|/think|>**Plan**\n" + "\n".join(lines) + "\n\n**Verification**\n" + "\n".join(verif)
    return prompt, answer, {"n_steps": len(entries), "removed": removed, "domain": dom["task"][1]}


def plan_conversations(n: int = 300, seed: int = 0) -> List[Conversation]:
    """Planning prompts (Uzbek / English) answered in the Moonlight PLAN FORMAT:
    ``<|think|>`` goal / constraints / unknowns / risks ``<|/think|>``, **Reja / Plan** numbered steps
    (action + expected output + check) and a **Tekshiruv / Verification** checklist mapping every user
    requirement and constraint to the step that satisfies it."""
    rng = _rng("plan", seed)
    # half of the examples carry the PLAN system prompt that ``LLM.plan`` uses at inference time,
    # the other half the default system prompt (plans requested in ordinary chat)
    plan_system = _chat_prompt("PLAN_SYSTEM_PROMPT", "")
    out = []
    for i in range(int(n)):
        prompt, answer, _ = _plan_example(rng, i % 2 == 0)
        conv = [_user(prompt), _asst(answer)]
        if plan_system and (i // 2) % 2 == 1:
            conv.insert(0, _system(None, plan_system))
        out.append(conv)
    return out


# =============================================================================================== constraints

_TOPICS: List[Dict[str, Any]] = [
    {"uz": {"prompt": "Kitob o'qishning foydalari haqida yozing.", "title": "Kitob o'qishning foydasi", "main": "kitob",
            "keywords": ["bilim", "tasavvur", "lug'at", "diqqat"], "avoid": ["televizor", "o'yin", "telefon"],
            "sentences": ["Kitob o'qish bilimni oshiradi va dunyoqarashni kengaytiradi.",
                          "Har kuni bir necha bet kitob o'qigan odamning lug'at boyligi tez o'sadi.",
                          "Badiiy kitob tasavvurni rivojlantiradi va boshqa odamlarni tushunishga yordam beradi.",
                          "Kitob o'qish diqqatni bir joyga jamlashni o'rgatadi.",
                          "Yaxshi kitob kun davomida to'plangan charchoqni kamaytiradi.",
                          "Bolalikdan kitob o'qishga o'rgangan inson o'qishda ham, ishda ham muvaffaqiyatli bo'ladi.",
                          "Kitob muhokamasi do'stlar bilan qiziqarli suhbatga sabab bo'ladi."]},
     "en": {"prompt": "Write about the benefits of reading books.", "title": "Why Reading Matters", "main": "reading",
            "keywords": ["knowledge", "imagination", "vocabulary", "focus"], "avoid": ["television", "games", "phone"],
            "sentences": ["Reading builds knowledge and widens your view of the world.",
                          "People who spend a few minutes reading every day grow their vocabulary quickly.",
                          "Fiction trains the imagination and helps us understand other people.",
                          "Reading teaches the mind to keep its focus on one thing.",
                          "A good book before bed is a calm way to end the day.",
                          "Children who enjoy reading often do better at school.",
                          "Talking about a book with friends turns reading into a shared experience."]}},
    {"uz": {"prompt": "Sog'lom uyquning ahamiyati haqida yozing.", "title": "Sog'lom uyqu", "main": "uyqu",
            "keywords": ["salomatlik", "dam", "kayfiyat", "xotira"], "avoid": ["kofe", "kompyuter", "kechki ovqat"],
            "sentences": ["Sog'lom uyqu salomatlik uchun ovqat va harakat kabi muhim.",
                          "Kattalar uchun har kecha yetti-sakkiz soatlik uyqu tavsiya etiladi.",
                          "Yaxshi uyqudan keyin kayfiyat ko'tariladi va ish unumdorligi oshadi.",
                          "Uyqu paytida miya kun davomida olingan ma'lumotlarni xotirada mustahkamlaydi.",
                          "Har kuni bir xil vaqtda yotish organizmga to'liq dam olishga yordam beradi.",
                          "Yotishdan oldin ekranlardan uzoqroq turish uyquni yaxshilaydi.",
                          "Qorong'i va salqin xona tez uxlab qolishga yordam beradi."]},
     "en": {"prompt": "Write about why healthy sleep matters.", "title": "The Value of Sleep", "main": "sleep",
            "keywords": ["health", "rest", "mood", "memory"], "avoid": ["coffee", "computer", "dinner"],
            "sentences": ["Healthy sleep matters for health as much as food and exercise.",
                          "Most adults need seven to eight hours of sleep each night.",
                          "After a good night of sleep your mood improves and work feels easier.",
                          "During sleep the brain stores the day's lessons in long-term memory.",
                          "Going to bed at the same time every day helps the body rest fully.",
                          "Staying away from screens before bed makes sleep deeper.",
                          "A dark and cool room helps you fall asleep faster."]}},
    {"uz": {"prompt": "Dasturlashni o'rganishni boshlayotganlarga maslahatlar yozing.", "title": "Dasturlashni o'rganish",
            "main": "dasturlash", "keywords": ["Python", "mashq", "loyiha", "sabr"], "avoid": ["pul", "tez", "oson"],
            "sentences": ["Dasturlashni o'rganishni Python kabi sodda tildan boshlash qulay.",
                          "Har kuni oz bo'lsa ham mashq qilish katta natija beradi.",
                          "Kichik loyiha yaratish nazariyani amalda mustahkamlaydi.",
                          "Xatolardan qo'rqmaslik kerak, chunki dasturlash xatolarni tuzatishdan iborat.",
                          "Boshqalarning kodini o'qish yangi usullarni o'rganishga yordam beradi.",
                          "Dasturlashda sabr va izchillik iste'doddan ham muhimroq.",
                          "Savollaringizni hamjamiyatda so'rash o'rganishni tezlashtiradi."]},
     "en": {"prompt": "Write tips for people who are starting to learn programming.", "title": "Learning to Code",
            "main": "programming", "keywords": ["Python", "practice", "project", "patience"],
            "avoid": ["money", "fast", "easy"],
            "sentences": ["Starting programming with a simple language like Python makes the first steps smoother.",
                          "A little practice every day beats one long session per week.",
                          "Building a small project turns theory into real skill.",
                          "Do not fear errors, because programming is mostly about fixing them.",
                          "Reading other people's code shows you new ways to solve problems.",
                          "In programming, patience and consistency matter more than talent.",
                          "Asking questions in a community speeds up learning."]}},
    {"uz": {"prompt": "Suvni tejash nima uchun muhimligini tushuntiring.", "title": "Suvni asraylik", "main": "suv",
            "keywords": ["tabiat", "kelajak", "jo'mrak", "sug'orish"], "avoid": ["dengiz", "yomg'ir", "muz"],
            "sentences": ["Toza ichimlik suvi cheksiz emas, shuning uchun uni tejash kerak.",
                          "Tishni yuvayotganda jo'mrakni yopib qo'yish har kuni ko'p suvni saqlab qoladi.",
                          "Tomchilatib sug'orish dalalarda suv sarfini sezilarli kamaytiradi.",
                          "Suvni tejash tabiatni va kelajak avlodlarni asrashdir.",
                          "Oqib turgan quvurni o'z vaqtida tuzatish suv isrofini to'xtatadi.",
                          "Har bir oila suvdan oqilona foydalansa, butun shaharga foyda bo'ladi.",
                          "Bolalarga suvni qadrlashni yoshlikdan o'rgatish muhim."]},
     "en": {"prompt": "Explain why saving water is important.", "title": "Every Drop Counts", "main": "water",
            "keywords": ["nature", "future", "tap", "irrigation"], "avoid": ["ocean", "rain", "ice"],
            "sentences": ["Clean drinking water is limited, so we need to save it.",
                          "Turning off the tap while brushing your teeth saves a lot of water every day.",
                          "Drip irrigation greatly reduces water use on farms.",
                          "Saving water protects nature and the future of the next generations.",
                          "Fixing a leaking pipe quickly stops water from being wasted.",
                          "When every family uses water wisely, the whole city benefits.",
                          "Teaching children to value water early makes a lasting difference."]}},
    {"uz": {"prompt": "O'zbekistonga sayohat qilish haqida qisqa tavsiya yozing.", "title": "O'zbekistonga sayohat",
            "main": "sayohat", "keywords": ["Samarqand", "Buxoro", "Xiva", "palov"], "avoid": ["dengiz", "plyaj", "qor"],
            "sentences": ["O'zbekistonga sayohat qadimiy shaharlar bilan tanishishdan boshlanadi.",
                          "Samarqanddagi Registon maydoni har bir mehmonni hayratga soladi.",
                          "Buxoroning tor ko'chalari va qadimiy madrasalari tarix bilan nafas oladi.",
                          "Xivadagi Ichan qal'a ochiq osmon ostidagi muzeyga o'xshaydi.",
                          "Sayohat davomida albatta o'zbek palovini tatib ko'rish kerak.",
                          "Bahor va kuz sayohat uchun eng qulay fasllar hisoblanadi.",
                          "Shaharlar o'rtasida tezyurar poyezdda qulay sayohat qilish mumkin."]},
     "en": {"prompt": "Write a short recommendation about travelling to Uzbekistan.", "title": "Travelling in Uzbekistan",
            "main": "travel", "keywords": ["Samarkand", "Bukhara", "Khiva", "plov"], "avoid": ["beach", "ocean", "snow"],
            "sentences": ["Travel in Uzbekistan starts with its ancient cities.",
                          "The Registan square in Samarkand amazes every visitor.",
                          "The narrow streets and old madrasahs of Bukhara are full of history.",
                          "The Itchan Kala of Khiva feels like an open-air museum.",
                          "Do not leave without tasting Uzbek plov.",
                          "Spring and autumn are the most comfortable seasons for travel there.",
                          "Fast trains make travel between the cities easy."]}},
    {"uz": {"prompt": "Jamoada ishlashning afzalliklari haqida yozing.", "title": "Jamoaviy ish", "main": "jamoa",
            "keywords": ["hamkorlik", "maqsad", "muloqot", "ishonch"], "avoid": ["raqobat", "janjal", "yolg'iz"],
            "sentences": ["Jamoada ishlash murakkab vazifalarni tezroq bajarishga imkon beradi.",
                          "Aniq maqsad jamoa a'zolarini bir yo'nalishda harakatlantiradi.",
                          "Ochiq muloqot tushunmovchiliklarning oldini oladi.",
                          "Hamkorlik har bir insonning kuchli tomonini yuzaga chiqaradi.",
                          "O'zaro ishonch bo'lgan jamoada yangi g'oyalar tez tug'iladi.",
                          "Yaxshi jamoa muvaffaqiyatni ham, xatolarni ham birga baham ko'radi.",
                          "Muntazam qisqa uchrashuvlar jamoa ishini tartibga soladi."]},
     "en": {"prompt": "Write about the advantages of working in a team.", "title": "The Power of Teamwork", "main": "team",
            "keywords": ["collaboration", "goal", "communication", "trust"], "avoid": ["competition", "conflict", "alone"],
            "sentences": ["Working in a team lets people finish complex tasks faster.",
                          "A clear goal keeps every team member moving in the same direction.",
                          "Open communication prevents misunderstandings.",
                          "Collaboration brings out the strengths of each person.",
                          "In a team built on trust, new ideas appear quickly.",
                          "A good team shares both its successes and its mistakes.",
                          "Short regular meetings keep the team's work organized."]}},
]

_START_PHRASES = {"uz": ["Qisqacha aytganda", "Mening fikrimcha", "Avvalo"], "en": ["In short", "In my view", "First of all"]}
_END_PHRASES = {"uz": ["Tayyor.", "E'tiboringiz uchun rahmat.", "Shu bilan tugatdim."],
                "en": ["Is there anything else I can help with?", "That's all.", "Thank you for reading."]}
_KW_TEMPLATES = {"uz": ["Bu borada {kw} ham muhim o'rin tutadi.", "Shuningdek {kw} haqida ham o'ylash kerak."],
                 "en": ["It is also worth thinking about {kw}.", "Here {kw} plays an important role as well."]}
_MAIN_TEMPLATES = {"uz": ["Xulosa shuki {main} haqida doim o'ylash kerak.", "{Main} mavzusi har birimizga tegishli.",
                          "Demak {main} kundalik hayotimizning muhim qismidir."],
                   "en": ["In the end {main} deserves our attention.", "The topic of {main} concerns all of us.",
                          "So {main} is an important part of everyday life."]}

_STRUCTURES = ("plain", "bullets", "paragraphs", "sections", "json")
_OPTIONAL = ("include_keywords", "exclude_keywords", "keyword_frequency", "case", "no_commas", "ends_with",
             "starts_with", "title_brackets", "quotation_wrapped", "word_count", "sentence_count", "max_chars",
             "language_uz")
_JSON_OK = {"include_keywords", "exclude_keywords", "keyword_frequency", "case", "word_count", "max_chars", "language_uz"}
_EXCLUSIVE = [{"starts_with", "title_brackets"}, {"quotation_wrapped", "starts_with"}, {"quotation_wrapped", "ends_with"}]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _build_constraint_example(rng: random.Random, uz: bool) -> Optional[Dict[str, Any]]:
    lang = "uz" if uz else "en"
    topic = _pick(rng, _TOPICS)[lang]
    structure = rng.choices(_STRUCTURES, weights=[3, 3, 2, 2, 1])[0]
    n_total = rng.randint(1, 4)
    n_opt = n_total - (0 if structure == "plain" else 1)
    pool = [o for o in _OPTIONAL if (structure != "json" or o in _JSON_OK) and (o != "language_uz" or uz)]
    if structure == "json":
        pool = [o for o in pool if o != "case"] + ["case_lower"]
    chosen: List[str] = []
    for o in rng.sample(pool, len(pool)):
        if len(chosen) >= max(0, n_opt):
            break
        if any(o in ex and (ex - {o}) & set(chosen) for ex in _EXCLUSIVE):
            continue
        chosen.append(o)
    sents = rng.sample(topic["sentences"], len(topic["sentences"]))
    used: List[str] = []

    def take(k: int) -> List[str]:
        out = [sents.pop() for _ in range(min(k, len(sents)))]
        used.extend(out)
        return out

    constraints: List[V.Constraint] = []
    # ---- body
    if structure == "plain":
        blocks = [" ".join(take(rng.randint(2, 4)))]
    elif structure == "bullets":
        nb = rng.randint(2, 5)
        blocks = [f"- {s}" for s in take(nb)]
        constraints.append({"type": "bullet_count", "n": nb})
    elif structure == "paragraphs":
        npar = rng.randint(2, 3)
        blocks = [" ".join(take(rng.randint(1, 2))) for _ in range(npar)]
        constraints.append({"type": "paragraph_count", "n": npar})
    elif structure == "sections":
        nsec = rng.randint(2, 3)
        blocks = [" ".join(take(rng.randint(1, 2))) for _ in range(nsec)]
        constraints.append({"type": "sections", "n": nsec, "marker": "SECTION"})
    else:
        blocks = take(rng.randint(2, 4))
        constraints.append({"type": "json_format"})

    def add_sentence(s: str) -> None:
        if structure == "json" or structure == "plain":
            if structure == "json":
                blocks.append(s)
            else:
                blocks[-1] = blocks[-1] + " " + s
        else:
            blocks[-1] = blocks[-1] + " " + s
        used.append(s)

    if "include_keywords" in chosen:
        kws = rng.sample(topic["keywords"], rng.randint(1, 2))
        kw_templates = rng.sample(_KW_TEMPLATES[lang], len(_KW_TEMPLATES[lang]))
        for kw in kws:
            if not any(V._kw_count(s, kw) for s in used):
                add_sentence(kw_templates.pop().format(kw=kw))
        constraints.append({"type": "include_keywords", "keywords": kws})
    if "keyword_frequency" in chosen:
        main = topic["main"]
        need = rng.randint(2, 3)
        templates = rng.sample(_MAIN_TEMPLATES[lang], len(_MAIN_TEMPLATES[lang]))  # never repeat a filler
        while sum(V._kw_count(s, main) for s in used) < need and templates:
            add_sentence(templates.pop().format(main=main, Main=_cap(main)))
        have = sum(V._kw_count(s, main) for s in used)
        constraints.append({"type": "keyword_frequency", "word": main, "at_least": min(need, have)})

    # ---- render
    if structure == "json":
        key_t, key_p = ("mavzu", "fikrlar") if uz else ("topic", "points")
        text = json.dumps({key_t: topic["title"], key_p: blocks}, ensure_ascii=False, indent=2)
    elif structure == "bullets":
        text = "\n".join(blocks)
    elif structure == "paragraphs":
        text = "\n\n***\n\n".join(blocks)
    elif structure == "sections":
        text = "\n\n".join(f"SECTION {i + 1}\n{b}" for i, b in enumerate(blocks))
    else:
        text = blocks[0]
    if "no_commas" in chosen:
        text = text.replace(", ", " ").replace(",", "")
        constraints.append({"type": "no_commas"})
    if "title_brackets" in chosen:
        text = f"<<{topic['title']}>>\n\n{text}"
        constraints.append({"type": "title_brackets"})
    if "starts_with" in chosen:
        phrase = _pick(rng, _START_PHRASES[lang])
        text = f"{phrase}:\n{text}" if structure in ("bullets", "sections") else f"{phrase}: {text}"
        constraints.append({"type": "starts_with", "phrase": phrase})
    if "ends_with" in chosen:
        phrase = _pick(rng, _END_PHRASES[lang])
        text = f"{text}\n\n{phrase}"
        constraints.append({"type": "ends_with", "phrase": phrase})
    if "case" in chosen or "case_lower" in chosen:
        if "case_lower" in chosen or rng.random() < 0.5:
            text = text.lower()
            constraints.append({"type": "lowercase"})
        else:
            text = text.upper()
            constraints.append({"type": "uppercase"})
    if "quotation_wrapped" in chosen:
        text = f"\"{text}\""
        constraints.append({"type": "quotation_wrapped"})
    if "exclude_keywords" in chosen:
        bad = [w for w in topic["avoid"] if V._kw_count(text, w) == 0]
        if bad:
            constraints.append({"type": "exclude_keywords", "keywords": rng.sample(bad, min(len(bad), rng.randint(1, 2)))})
    if "language_uz" in chosen:
        constraints.append({"type": "language_uz"})
    # ---- numeric limits derived from the finished text (with slack, so they are natural)
    if "word_count" in chosen:
        w = V.count_words(text)
        if rng.random() < 0.5:
            constraints.append({"type": "word_count_max", "n": int(math.ceil((w + rng.randint(3, 30)) / 5.0) * 5)})
        else:
            constraints.append({"type": "word_count_min", "n": max(1, (w - rng.randint(0, 15)) // 5 * 5) or 1})
    if "sentence_count" in chosen:
        s = V.count_sentences(text)
        if rng.random() < 0.5:
            constraints.append({"type": "sentence_count_max", "n": s + rng.randint(0, 3)})
        else:
            constraints.append({"type": "sentence_count_min", "n": max(1, s - rng.randint(0, 2))})
    if "max_chars" in chosen:
        constraints.append({"type": "max_chars", "n": int(math.ceil((len(text.strip()) + rng.randint(10, 120)) / 50.0) * 50)})
    if not constraints:
        return None
    passed, total, _ = V.check_all(text, constraints)
    if passed != total:
        return None
    rng.shuffle(constraints)
    instr = [V.describe(c, lang) for c in constraints]
    if rng.random() < 0.5:
        prompt = topic["prompt"] + " " + " ".join(instr)
    else:
        head = "Talablar:" if uz else "Requirements:"
        prompt = topic["prompt"] + f"\n{head}\n" + "\n".join(f"- {x}" for x in instr)
    return {"prompt": prompt, "response": text, "constraints": constraints, "lang": lang}


def constraint_examples(n: int = 300, seed: int = 0) -> List[Dict[str, Any]]:
    """``n`` dicts ``{"prompt", "response", "constraints", "lang"}`` whose response provably satisfies
    every constraint (checked with :func:`moonlight.data.verifiers.check_all`)."""
    rng = _rng("constraints", seed)
    out: List[Dict[str, Any]] = []
    attempts = 0
    while len(out) < int(n):
        attempts += 1
        if attempts > 50 * max(1, int(n)) + 100:  # pragma: no cover - defensive
            raise RuntimeError("constraint generator failed to produce valid examples")
        ex = _build_constraint_example(rng, len(out) % 2 == 0)
        if ex is not None:
            out.append(ex)
    return out


def constraint_conversations(n: int = 300, seed: int = 0) -> List[Conversation]:
    """Prompts with 1-4 verifiable constraints and responses that satisfy all of them."""
    return [[_user(ex["prompt"]), _asst(ex["response"])] for ex in constraint_examples(n, seed)]


# =============================================================================================== details

_DETAIL_CODE = [
    {"uz": "Sonlar ro'yxatining yig'indisini qaytaradigan funksiya yozing", "en": "Write a function that returns the sum of a list of numbers",
     "fn": ["sum_values", "total_of", "add_all"], "var": ["total", "acc", "running"],
     "body": "def {fn}(numbers):\n    {var} = 0\n    for x in numbers:\n        {var} += x\n    return {var}",
     "test": "assert {fn}([1, 2, 3]) == 6\nassert {fn}([]) == 0"},
    {"uz": "Ro'yxatdagi juft sonlar sonini hisoblaydigan funksiya yozing", "en": "Write a function that counts the even numbers in a list",
     "fn": ["count_even", "even_count", "how_many_even"], "var": ["count", "evens", "n_even"],
     "body": "def {fn}(numbers):\n    {var} = 0\n    for x in numbers:\n        if x % 2 == 0:\n            {var} += 1\n    return {var}",
     "test": "assert {fn}([1, 2, 4, 7]) == 2\nassert {fn}([]) == 0"},
    {"uz": "Ro'yxatdagi sonlarning o'rtacha qiymatini qaytaradigan funksiya yozing", "en": "Write a function that returns the average of a list of numbers",
     "fn": ["average", "mean_value", "avg_of"], "var": ["avg", "result", "mean"],
     "body": "def {fn}(numbers):\n    if not numbers:\n        return 0.0\n    {var} = sum(numbers) / len(numbers)\n    return {var}",
     "test": "assert {fn}([2, 4, 6]) == 4\nassert {fn}([]) == 0.0"},
    {"uz": "So'zlar ro'yxatini bitta gapga birlashtiradigan funksiya yozing", "en": "Write a function that joins a list of words into one sentence",
     "fn": ["join_words", "make_sentence", "glue"], "var": ["sentence", "result", "text"],
     "body": "def {fn}(words):\n    {var} = \" \".join(words)\n    return {var}",
     "test": "assert {fn}(['salom', 'dunyo']) == 'salom dunyo'\nassert {fn}([]) == ''"},
]
_NAMES = ["Dilshod", "Madina", "Bekzod", "Nigora", "Jahongir", "Shahnoza", "Otabek", "Kamola"]
_FRUITS = {"uz": ["olma", "nok", "uzum", "anor", "o'rik", "shaftoli", "gilos", "behi"],
           "en": ["apple", "pear", "grape", "pomegranate", "apricot", "peach", "cherry", "quince"]}


def _detail_code(rng: random.Random, uz: bool) -> Dict[str, Any]:
    t = _pick(rng, _DETAIL_CODE)
    fn, var = _pick(rng, t["fn"]), _pick(rng, t["var"])
    code = t["body"].format(fn=fn, var=var)
    if uz:
        prompt = f"{t['uz']}. Funksiya nomi `{fn}` bo'lsin va natija uchun `{var}` o'zgaruvchisidan foydalaning."
        resp = f"Mana `{fn}` funksiyasi:\n\n```python\n{code}\n```\n\nNatija `{var}` o'zgaruvchisida hisoblanadi."
    else:
        prompt = f"{t['en']}. Name the function `{fn}` and use a variable called `{var}` for the result."
        resp = f"Here is `{fn}`:\n\n```python\n{code}\n```\n\nThe result is computed in the `{var}` variable."
    cons = [{"type": "include_keywords", "keywords": [f"def {fn}", var]}]
    return {"prompt": prompt, "response": resp, "constraints": cons, "code": code + "\n\n" + t["test"].format(fn=fn)}


def _detail_note(rng: random.Random, uz: bool) -> Dict[str, Any]:
    name = _pick(rng, _NAMES)
    month, day = rng.randint(1, 12), rng.randint(1, 28)
    date = f"2024-{month:02d}-{day:02d}"
    hour = rng.randint(9, 17)
    time = f"{hour:02d}:{_pick(rng, ['00', '30'])}"
    room = rng.randint(101, 512)
    end = "Tayyor" if uz else _pick(rng, ["Thanks", "Done"])
    if uz:
        prompt = (f"Jamoaga uchrashuv haqida qisqa xabar yozing. Uchrashuv {date} kuni soat {time} da {room}-xonada "
                  f"bo'ladi, uni {name} olib boradi. Xabar \"{end}\" so'zi bilan tugasin.")
        resp = (f"Hurmatli hamkasblar! {date} kuni soat {time} da {room}-xonada jamoa uchrashuvi bo'ladi. "
                f"Uchrashuvni {name} olib boradi. Iltimos, kechikmang. {end}")
    else:
        prompt = (f"Write a short message to the team about a meeting on {date} at {time} in room {room}, led by "
                  f"{name}. End the message with the word \"{end}\".")
        resp = (f"Hi team! We have a meeting on {date} at {time} in room {room}. {name} will lead it. "
                f"Please be on time. {end}")
    cons = [{"type": "include_keywords", "keywords": [date, time, str(room), name]}, {"type": "ends_with", "phrase": end}]
    return {"prompt": prompt, "response": resp, "constraints": cons}


def _detail_list(rng: random.Random, uz: bool) -> Dict[str, Any]:
    lang = "uz" if uz else "en"
    k = rng.randint(3, 5)
    fruits = rng.sample(_FRUITS[lang], k)
    first = fruits[0]
    if uz:
        prompt = (f"{k} ta meva nomini yozing. Har birini yangi qatorda \"- \" bilan boshlang va ro'yxat {first} "
                  "bilan boshlansin. Boshqa hech narsa yozmang.")
    else:
        prompt = (f"List {k} fruits, one per line starting with \"- \", and put {first} first. Write nothing else.")
    resp = "\n".join(f"- {f}" for f in fruits)
    cons = [{"type": "bullet_count", "n": k}, {"type": "starts_with", "phrase": f"- {first}"}]
    return {"prompt": prompt, "response": resp, "constraints": cons}


def _detail_email(rng: random.Random, uz: bool) -> Dict[str, Any]:
    order = rng.randint(1000, 9999)
    amount = rng.randint(5, 95) * 10000
    name = _pick(rng, _NAMES)
    if uz:
        prompt = (f"Mijoz {name}ga buyurtmasi yetkazilgani haqida xat yozing. Buyurtma raqami #{order}, summasi "
                  f"{amount} so'm. Xat \"Mavzu:\" qatoridan boshlansin.")
        resp = (f"Mavzu: #{order} buyurtmangiz yetkazildi\n\nHurmatli {name}!\n\n#{order} raqamli buyurtmangiz "
                f"muvaffaqiyatli yetkazib berildi. To'langan summa: {amount} so'm. Xaridingiz uchun rahmat!")
        cons = [{"type": "starts_with", "phrase": "Mavzu:"}, {"type": "include_keywords", "keywords": [f"#{order}", str(amount), name]}]
    else:
        prompt = (f"Write an e-mail to the customer {name} saying their order #{order} for {amount} som was delivered. "
                  "Start with a line beginning \"Subject:\".")
        resp = (f"Subject: Your order #{order} has been delivered\n\nDear {name},\n\nYour order #{order} has been "
                f"delivered successfully. Amount paid: {amount} som. Thank you for shopping with us!")
        cons = [{"type": "starts_with", "phrase": "Subject:"}, {"type": "include_keywords", "keywords": [f"#{order}", str(amount), name]}]
    return {"prompt": prompt, "response": resp, "constraints": cons}


_DETAIL_BUILDERS = [_detail_code, _detail_note, _detail_list, _detail_email]


def detail_examples(n: int = 300, seed: int = 0) -> List[Dict[str, Any]]:
    """``n`` dicts ``{"prompt", "response", "constraints"}``: prompts hiding small requirements (exact
    names, dates, ending word, first item...) with responses satisfying every one (asserted)."""
    rng = _rng("detail", seed)
    out = []
    for i in range(int(n)):
        ex = _pick(rng, _DETAIL_BUILDERS)(rng, i % 2 == 0)
        passed, total, details = V.check_all(ex["response"], ex["constraints"])
        if passed != total:  # pragma: no cover - templates are verified by tests
            raise AssertionError(f"detail example violates its constraints: {details}")
        out.append(ex)
    return out


def detail_conversations(n: int = 300, seed: int = 0) -> List[Conversation]:
    """Conversations whose prompts hide small requirements; responses meet each one."""
    return [[_user(ex["prompt"]), _asst(ex["response"])] for ex in detail_examples(n, seed)]


# =============================================================================================== projects

#: system prompt of synthetic agentic trajectories (Uzbek + English sentence, like the chat.py constants).
def _chat_prompt(name: str, fallback: str) -> str:
    """A system-prompt constant from ``moonlight.chat`` (SPEC §13.6 places PLAN/BUILDER prompts there),
    so training trajectories use exactly the prompt ``LLM.plan`` / ``LLM.build_project`` use."""
    try:
        from moonlight import chat as _chat  # torch-free (tokenizer + tools)

        return str(getattr(_chat, name))
    except Exception:  # pragma: no cover - fallback if the chat module is unavailable
        return fallback


#: Builder system prompt of the agent trajectories (= ``moonlight.chat.BUILDER_SYSTEM_PROMPT``).
BUILDER_SYSTEM = _chat_prompt("BUILDER_SYSTEM_PROMPT", (
    "Sen Moonlight — dasturlash agentisan: avval reja tuz, keyin kod va testlarni yoz, testlarni ishga tushir, "
    "xatolarni tuzat va yakunda har bir talab qayerda bajarilganini sanab chiq. You are Moonlight, a coding agent: "
    "plan first, write the code and tests, run the tests, fix failures, and finish with a summary that maps every "
    "requirement to where it is satisfied."))


def _project(key: str, uz: str, en: str, module: str, code: str, tests: str, reqs: Sequence[Tuple[str, str, str]],
             bug: Tuple[str, str, str, int, Sequence[str]]) -> Dict[str, Any]:
    """``reqs``: (uz requirement, en requirement, where satisfied). ``bug``: (original snippet, buggy snippet,
    failing test name, line number of the failing assert in the test file, pytest ``E`` lines)."""
    return {"key": key, "uz": uz, "en": en, "module": module, "test_file": f"test_{module}",
            "code": code.strip("\n") + "\n", "tests": tests.strip("\n") + "\n", "reqs": list(reqs),
            "bug": {"orig": bug[0], "buggy": bug[1], "test": bug[2], "line": bug[3], "e_lines": list(bug[4])}}


#: Mini projects for agentic trajectories; ``verify_projects`` runs pytest on the correct code (must pass)
#: and on the bug-injected code (must fail exactly the declared test).
PROJECTS: List[Dict[str, Any]] = [
    _project(
        "calculator", "Oddiy kalkulyator modulini yarating.", "Create a simple calculator module.", "calc.py",
        '''
"""Basic arithmetic helpers."""


def add(a: float, b: float) -> float:
    """Return a + b."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Return a - b."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Return a * b."""
    return a * b


def divide(a: float, b: float) -> float:
    """Return a / b; raise ValueError when b is zero."""
    if b == 0:
        raise ValueError("division by zero")
    return a / b
''',
        '''
import pytest

from calc import add, divide, multiply, subtract


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(10, 4) == 6


def test_multiply():
    assert multiply(6, 7) == 42


def test_divide():
    assert divide(7, 2) == 3.5


def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(1, 0)
''',
        [("Qo'shish, ayirish, ko'paytirish va bo'lish funksiyalari bo'lsin.", "Provide add, subtract, multiply and divide.",
          "calc.py: add, subtract, multiply, divide"),
         ("Nolga bo'lishda ValueError chiqsin.", "Division by zero must raise ValueError.",
          "calc.py: divide(); test_calc.py: test_divide_by_zero"),
         ("Har bir funksiya uchun test yozilsin.", "Every function must have a test.", "test_calc.py: 5 tests")],
        ("    return a / b", "    return a // b", "test_divide", 19, ["assert 3 == 3.5", " +  where 3 = divide(7, 2)"])),
    _project(
        "string_utils", "Satrlar bilan ishlash uchun yordamchi modul yozing.", "Write a helper module for working with strings.",
        "strutils.py",
        '''
"""Small string utilities."""


def reverse(text: str) -> str:
    """Return ``text`` reversed."""
    return text[::-1]


def is_palindrome(text: str) -> bool:
    """True if ``text`` reads the same backwards (letters and digits only, case-insensitive)."""
    cleaned = [ch.lower() for ch in text if ch.isalnum()]
    return cleaned == cleaned[::-1]


def count_words(text: str) -> int:
    """Number of whitespace-separated words."""
    return len(text.split())


def title_case(text: str) -> str:
    """Capitalize the first letter of every word."""
    return " ".join(word[:1].upper() + word[1:] for word in text.split())
''',
        '''
from strutils import count_words, is_palindrome, reverse, title_case


def test_reverse():
    assert reverse("salom") == "molas"


def test_is_palindrome():
    assert is_palindrome("Kiyik")
    assert not is_palindrome("salom")


def test_count_words():
    assert count_words("salom  dunyo") == 2
    assert count_words("") == 0


def test_title_case():
    assert title_case("toshkent shahri") == "Toshkent Shahri"
''',
        [("Satrni teskari aylantirish funksiyasi bo'lsin.", "Provide a function to reverse a string.", "strutils.py: reverse"),
         ("Palindromni tekshirish katta-kichik harfga bog'liq bo'lmasin.", "Palindrome checks must ignore case.",
          "strutils.py: is_palindrome; test_strutils.py: test_is_palindrome"),
         ("So'zlarni sanash bir nechta bo'sh joyda ham to'g'ri ishlasin.", "Word counting must handle repeated spaces.",
          "strutils.py: count_words; test_strutils.py: test_count_words")],
        ("    return len(text.split())", "    return len(text.split(\" \"))", "test_count_words", 14,
         ["AssertionError: assert 3 == 2", " +  where 3 = count_words('salom  dunyo')"])),
    _project(
        "todo", "Vazifalar ro'yxati (TodoList) klassini yarating.", "Create a TodoList class.", "todo.py",
        '''
"""In-memory to-do list."""


class TodoList:
    """Tasks with integer ids; ids start at 1 and are never reused."""

    def __init__(self) -> None:
        self._tasks = {}
        self._next_id = 1

    def add(self, title: str) -> int:
        """Add a task and return its id."""
        if not title.strip():
            raise ValueError("title must not be empty")
        task_id = self._next_id
        self._tasks[task_id] = {"title": title, "done": False}
        self._next_id += 1
        return task_id

    def complete(self, task_id: int) -> None:
        """Mark a task as done (KeyError for unknown ids)."""
        self._tasks[task_id]["done"] = True

    def remove(self, task_id: int) -> None:
        """Delete a task (KeyError for unknown ids)."""
        del self._tasks[task_id]

    def pending(self) -> list:
        """Titles of unfinished tasks in insertion order."""
        return [t["title"] for t in self._tasks.values() if not t["done"]]
''',
        '''
import pytest

from todo import TodoList


def test_add_returns_ids():
    todo = TodoList()
    assert todo.add("non olish") == 1
    assert todo.add("kitob o'qish") == 2


def test_pending_excludes_completed():
    todo = TodoList()
    first = todo.add("non olish")
    todo.add("kitob o'qish")
    todo.complete(first)
    assert todo.pending() == ["kitob o'qish"]


def test_remove_unknown_raises():
    todo = TodoList()
    with pytest.raises(KeyError):
        todo.remove(42)


def test_empty_title_rejected():
    with pytest.raises(ValueError):
        TodoList().add("   ")
''',
        [("Vazifa qo'shilganda uning raqami qaytarilsin.", "Adding a task returns its id.",
          "todo.py: TodoList.add; test_todo.py: test_add_returns_ids"),
         ("Bajarilmagan vazifalarni ko'rish mumkin bo'lsin.", "It must be possible to list unfinished tasks.",
          "todo.py: TodoList.pending; test_todo.py: test_pending_excludes_completed"),
         ("Mavjud bo'lmagan vazifani o'chirishda xato chiqsin.", "Removing an unknown task must raise an error.",
          "todo.py: TodoList.remove; test_todo.py: test_remove_unknown_raises"),
         ("Bo'sh nomli vazifa qabul qilinmasin.", "Empty titles must be rejected.",
          "todo.py: TodoList.add; test_todo.py: test_empty_title_rejected")],
        ("        return [t[\"title\"] for t in self._tasks.values() if not t[\"done\"]]",
         "        return [t[\"title\"] for t in self._tasks.values()]", "test_pending_excludes_completed", 17,
         ["assert ['non olish', \"kitob o'qish\"] == [\"kitob o'qish\"]", "  ",
          "  At index 0 diff: 'non olish' != \"kitob o'qish\"", "  Left contains one more item: \"kitob o'qish\"",
          "  Use -v to get more diff"])),
    _project(
        "temperature", "Harorat birliklarini o'giruvchi modul yozing.", "Write a temperature conversion module.",
        "temperature.py",
        '''
"""Temperature conversions between Celsius, Fahrenheit and Kelvin."""

ABSOLUTE_ZERO_C = -273.15


def c_to_f(c: float) -> float:
    """Celsius -> Fahrenheit."""
    _check(c)
    return c * 9 / 5 + 32


def f_to_c(f: float) -> float:
    """Fahrenheit -> Celsius."""
    c = (f - 32) * 5 / 9
    _check(c)
    return c


def c_to_k(c: float) -> float:
    """Celsius -> Kelvin."""
    _check(c)
    return c + 273.15


def _check(c: float) -> None:
    if c < ABSOLUTE_ZERO_C:
        raise ValueError("temperature below absolute zero")
''',
        '''
import pytest

from temperature import c_to_f, c_to_k, f_to_c


def test_c_to_f():
    assert c_to_f(100) == 212
    assert c_to_f(-40) == -40


def test_f_to_c():
    assert f_to_c(32) == 0


def test_c_to_k():
    assert c_to_k(0) == 273.15


def test_below_absolute_zero():
    with pytest.raises(ValueError):
        c_to_k(-300)
''',
        [("Selsiy, Farengeyt va Kelvin o'rtasida o'girish bo'lsin.", "Convert between Celsius, Fahrenheit and Kelvin.",
          "temperature.py: c_to_f, f_to_c, c_to_k"),
         ("Absolyut noldan past harorat uchun ValueError chiqsin.", "Temperatures below absolute zero must raise ValueError.",
          "temperature.py: _check; test_temperature.py: test_below_absolute_zero")],
        ("    return c * 9 / 5 + 32", "    return c * 5 / 9 + 32", "test_c_to_f", 7,
         ["assert 87.55555555555556 == 212", " +  where 87.55555555555556 = c_to_f(100)"])),
    _project(
        "word_counter", "Matndagi so'zlarni sanaydigan modul yarating.", "Create a module that counts words in a text.",
        "wordcount.py",
        '''
"""Word frequency helpers."""

import re
from collections import Counter

_WORD = re.compile(r"[\\w']+")


def word_frequencies(text: str) -> dict:
    """Lowercased word -> count (punctuation ignored)."""
    return dict(Counter(w.lower() for w in _WORD.findall(text)))


def top_words(text: str, n: int = 3) -> list:
    """The ``n`` most frequent words as (word, count), ties broken alphabetically."""
    freq = word_frequencies(text)
    return sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
''',
        '''
from wordcount import top_words, word_frequencies


def test_frequencies_ignore_case_and_punctuation():
    assert word_frequencies("Olma, olma! Nok.") == {"olma": 2, "nok": 1}


def test_top_words():
    assert top_words("a b b c c c", 2) == [("c", 3), ("b", 2)]


def test_empty_text():
    assert word_frequencies("") == {}
''',
        [("So'zlar katta-kichik harfdan qat'i nazar sanalsin.", "Counting must be case-insensitive.",
          "wordcount.py: word_frequencies; test_wordcount.py: test_frequencies_ignore_case_and_punctuation"),
         ("Tinish belgilari hisobga olinmasin.", "Punctuation must be ignored.", "wordcount.py: _WORD regex"),
         ("Eng ko'p uchragan so'zlarni qaytaruvchi funksiya bo'lsin.", "Provide a function returning the most frequent words.",
          "wordcount.py: top_words; test_wordcount.py: test_top_words")],
        ("    return dict(Counter(w.lower() for w in _WORD.findall(text)))",
         "    return dict(Counter(_WORD.findall(text)))", "test_frequencies_ignore_case_and_punctuation", 5,
         ["AssertionError: assert {'Olma': 1, '...: 1, 'Nok': 1} == {'olma': 2, 'nok': 1}", "  ", "  Differing items:",
          "  {'olma': 1} != {'olma': 2}", "  Left contains 2 more items:", "  {'Nok': 1, 'Olma': 1}",
          "  Right contains 1 more item:", "  {'nok': 1}", "  Use -v to get more diff"])),
    _project(
        "stack", "Stek (Stack) ma'lumotlar tuzilmasini yozing.", "Implement a Stack data structure.", "stack.py",
        '''
"""A simple LIFO stack."""


class Stack:
    """Last-in, first-out container."""

    def __init__(self) -> None:
        self._items = []

    def push(self, item) -> None:
        """Put ``item`` on top."""
        self._items.append(item)

    def pop(self):
        """Remove and return the top item (IndexError when empty)."""
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        """Return the top item without removing it (IndexError when empty)."""
        if not self._items:
            raise IndexError("peek at empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        return not self._items

    def size(self) -> int:
        return len(self._items)
''',
        '''
import pytest

from stack import Stack


def test_push_pop_order():
    s = Stack()
    s.push(1)
    s.push(2)
    assert s.pop() == 2
    assert s.pop() == 1


def test_peek_does_not_remove():
    s = Stack()
    s.push("a")
    assert s.peek() == "a"
    assert s.size() == 1


def test_pop_empty_raises():
    with pytest.raises(IndexError):
        Stack().pop()
''',
        [("push, pop va peek amallari bo'lsin.", "Provide push, pop and peek.", "stack.py: Stack.push/pop/peek"),
         ("peek elementni o'chirmasin.", "peek must not remove the item.", "stack.py: Stack.peek; test_stack.py: test_peek_does_not_remove"),
         ("Bo'sh stekdan olishda IndexError chiqsin.", "Popping an empty stack must raise IndexError.",
          "stack.py: Stack.pop; test_stack.py: test_pop_empty_raises")],
        ("        return self._items[-1]", "        return self._items.pop()", "test_peek_does_not_remove", 18,
         ["assert 0 == 1", " +  where 0 = size()", " +    where size = <stack.Stack object at 0x7f3a2c1b9d50>.size"])),
]


def _pytest_progress(tests: Sequence[str], failed: Optional[str]) -> str:
    marks = "".join("F" if t == failed else "." for t in tests)
    return marks + " " * max(1, 80 - len(marks) - 6) + "[100%]"


def _test_names(test_src: str) -> List[str]:
    return re.findall(r"^def (test_\w+)\(", test_src, re.MULTILINE)


def _fail_block(p: Dict[str, Any]) -> str:
    """pytest -q style failure report for the project's injected bug."""
    bug = p["bug"]
    lines = p["tests"].split("\n")
    name = bug["test"]
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"def {name}("))
    fail_idx = bug["line"] - 1
    body = []
    for i in range(start, fail_idx + 1):
        prefix = ">   " if i == fail_idx else "    "
        body.append((prefix + lines[i]).rstrip())
    title = f" {name} "
    bar = "_" * ((80 - len(title)) // 2)
    out = [f"{bar}{title}{bar}".ljust(80, "_"), ""] + body
    out += [f"E       {e}" for e in bug["e_lines"]]
    out += ["", f"{p['test_file']}:{bug['line']}: AssertionError"]
    return "\n".join(out)


def _run_tests_result(p: Dict[str, Any], buggy: bool, seconds: float) -> Dict[str, Any]:
    tests = _test_names(p["tests"])
    if not buggy:
        out = _pytest_progress(tests, None) + f"\n{len(tests)} passed in {seconds:.2f}s"
        return {"passed": True, "returncode": 0, "runner": "pytest", "output": out}
    name = p["bug"]["test"]
    first_e = p["bug"]["e_lines"][0]
    summary = f"FAILED {p['test_file']}::{name} - {first_e}"
    if len(summary) > 80:  # pytest shortens the summary line to the terminal width
        summary = summary[:77] + "..."
    n_ok = len(tests) - 1
    out = "\n".join([_pytest_progress(tests, name), "=" * 35 + " FAILURES " + "=" * 35, _fail_block(p),
                     "=" * 27 + " short test summary info " + "=" * 28, summary,
                     f"1 failed, {n_ok} passed in {seconds:.2f}s"])
    return {"passed": False, "returncode": 1, "runner": "pytest", "output": out}


def _project_plan(p: Dict[str, Any], uz: bool) -> str:
    mod, tf = p["module"], p["test_file"]
    if uz:
        steps = [f"1. {mod} modulini yozish — talablardagi funksiyalar bilan.",
                 f"2. {tf} faylida har bir talab uchun test yozish.",
                 "3. Testlarni ishga tushirish va xatolarni tuzatish.",
                 "4. Yakuniy hisobot: har bir talab qayerda bajarilgani."]
        reqs = "\n".join(f"- {r[0]} → {r[2]}" for r in p["reqs"])
        return "# Reja\n\n" + "\n".join(steps) + f"\n\n## Talablar\n{reqs}\n"
    steps = [f"1. Write the {mod} module with the required functions.",
             f"2. Write tests in {tf} covering every requirement.",
             "3. Run the tests and fix any failures.",
             "4. Final report: where each requirement is satisfied."]
    reqs = "\n".join(f"- {r[1]} → {r[2]}" for r in p["reqs"])
    return "# Plan\n\n" + "\n".join(steps) + f"\n\n## Requirements\n{reqs}\n"


def _write_result(path: str, content: str) -> str:
    """The text ``moonlight.tools.Workspace.write_file`` returns for this write."""
    n_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return f"wrote {path} ({len(content)} characters, {n_lines} lines)"


def project_conversations(n: int = 100, seed: int = 0, bug_frac: float = 0.5) -> List[Conversation]:
    """Agentic project-building trajectories with workspace tools.

    Each trajectory: user task + requirements -> ``write_file PLAN.md`` -> ``write_file`` module ->
    ``write_file`` tests -> ``run_tests``; for ``bug_frac`` of them the first module version carries an
    injected bug, so the run fails (realistic pytest output), the agent reads the file, writes the fix
    and re-runs the tests (pass). Final assistant message summarizes every requirement and where it is
    satisfied. All code is verified by :func:`verify_projects`.
    """
    rng = _rng("projects", seed)
    tools = [WORKSPACE_TOOL_SCHEMAS[k] for k in ("write_file", "read_file", "list_dir", "run_tests")]
    out: List[Conversation] = []
    for i in range(int(n)):
        uz = i % 2 == 0
        p = PROJECTS[(i // 2) % len(PROJECTS)] if i < 2 * len(PROJECTS) else _pick(rng, PROJECTS)
        with_bug = rng.random() < bug_frac
        mod, tf = p["module"], p["test_file"]
        req_lines = "\n".join(f"- {r[0] if uz else r[1]}" for r in p["reqs"])
        task = (f"{p['uz']}\nTalablar:\n{req_lines}\nKod va testlarni yozib, testlar o'tishini tekshiring." if uz
                else f"{p['en']}\nRequirements:\n{req_lines}\nWrite the code and tests and make sure the tests pass.")
        plan = _project_plan(p, uz)
        code = p["code"]
        buggy = code.replace(p["bug"]["orig"], p["bug"]["buggy"], 1)
        first = buggy if with_bug else code
        msgs: Conversation = [_system(tools, BUILDER_SYSTEM), _user(task),
                              _call("write_file", {"path": "PLAN.md", "content": plan}),
                              _tool("write_file", _write_result("PLAN.md", plan)),
                              _call("write_file", {"path": mod, "content": first}),
                              _tool("write_file", _write_result(mod, first)),
                              _call("write_file", {"path": tf, "content": p["tests"]}),
                              _tool("write_file", _write_result(tf, p["tests"])),
                              _call("run_tests", {"target": "."}),
                              _tool("run_tests", _run_tests_result(p, with_bug, rng.uniform(0.01, 0.09)))]
        if with_bug:
            msgs += [_call("read_file", {"path": mod}), _tool("read_file", first),
                     _call("write_file", {"path": mod, "content": code}), _tool("write_file", _write_result(mod, code)),
                     _call("run_tests", {"target": "."}),
                     _tool("run_tests", _run_tests_result(p, False, rng.uniform(0.01, 0.09)))]
        if rng.random() < 0.3:
            msgs += [_call("list_dir", {"path": "."}), _tool("list_dir", sorted(["PLAN.md", mod, tf]))]
        n_tests = len(_test_names(p["tests"]))
        if uz:
            fix = (f"Birinchi ishga tushirishda {p['bug']['test']} testi xato berdi; {mod} dagi xatoni tuzatdim "
                   "va testlarni qayta ishga tushirdim.\n" if with_bug else "")
            summary = "\n".join(f"- {r[0]} → {r[2]}" for r in p["reqs"])
            final = (f"Tayyor! {mod} moduli va {tf} testlari yozildi.\n{fix}Natija: barcha {n_tests} ta test "
                     f"o'tdi.\n\nTalablar qayerda bajarilgan:\n{summary}")
        else:
            fix = (f"The first run failed in {p['bug']['test']}; I fixed the bug in {mod} and re-ran the tests.\n"
                   if with_bug else "")
            summary = "\n".join(f"- {r[1]} → {r[2]}" for r in p["reqs"])
            final = (f"Done! I wrote the {mod} module and the {tf} tests.\n{fix}Result: all {n_tests} tests "
                     f"pass.\n\nWhere each requirement is satisfied:\n{summary}")
        msgs.append(_asst(final))
        out.append(msgs)
    return out


def verify_projects(timeout: float = 60.0) -> List[str]:
    """Run pytest (fallback: a minimal in-process runner) on every :data:`PROJECTS` entry: the correct
    code must pass all tests and the bug-injected code must fail exactly the declared test with the
    declared first ``E`` line. Returns a list of problems (empty == all good)."""
    import os
    import subprocess
    import sys
    import tempfile

    problems: List[str] = []
    for p in PROJECTS:
        buggy = p["code"].replace(p["bug"]["orig"], p["bug"]["buggy"], 1)
        if buggy == p["code"]:
            problems.append(f"{p['key']}: bug snippet not found in code")
            continue
        lines = p["tests"].split("\n")
        if not lines[p["bug"]["line"] - 1].lstrip().startswith("assert"):
            problems.append(f"{p['key']}: line {p['bug']['line']} of {p['test_file']} is not an assert")
        for label, code, want_ok in (("correct", p["code"], True), ("buggy", buggy, False)):
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, p["module"]), "w", encoding="utf-8") as fh:
                    fh.write(code)
                with open(os.path.join(d, p["test_file"]), "w", encoding="utf-8") as fh:
                    fh.write(p["tests"])
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
                try:
                    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "."],
                                          cwd=d, capture_output=True, text=True, timeout=timeout, env=env)
                except subprocess.TimeoutExpired:
                    problems.append(f"{p['key']} ({label}): pytest timed out")
                    continue
                out = proc.stdout + proc.stderr
                if "No module named pytest" in out:
                    problems.append("pytest is not installed")
                    return problems
                ok = proc.returncode == 0
                if ok != want_ok:
                    problems.append(f"{p['key']} ({label}): expected {'pass' if want_ok else 'fail'}\n{out[-1500:]}")
                elif not want_ok:
                    if f"::{p['bug']['test']}" not in out or "1 failed" not in out:
                        problems.append(f"{p['key']}: wrong failing test\n{out[-1500:]}")
                    elif f"E       {p['bug']['e_lines'][0]}" not in out:
                        problems.append(f"{p['key']}: E line differs\n{out[-1500:]}")
                    elif f"{p['test_file']}:{p['bug']['line']}: AssertionError" not in out:
                        problems.append(f"{p['key']}: failure location differs\n{out[-1500:]}")
    return problems


# =============================================================================================== GRPO

def _train(entry: str, uz: str, en: str, solution: str, tests: Sequence[str]) -> Dict[str, Any]:
    return {"entry_point": entry, "uz": uz, "en": en, "solution": solution.strip("\n"), "tests": list(tests)}


#: TRAINING code problems for GRPO (kept separate from benchmark problems, which stay held-out).
TRAIN_CODE_PROBLEMS: List[Dict[str, Any]] = [
    _train("clamp", "`clamp(x, lo, hi)` funksiyasini yozing: x ni [lo, hi] oralig'iga cheklaydi.",
           "Write `clamp(x, lo, hi)` that limits x to the range [lo, hi].",
           "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))",
           ["assert clamp(5, 0, 10) == 5", "assert clamp(-3, 0, 10) == 0", "assert clamp(42, 0, 10) == 10"]),
    _train("rotate_right", "`rotate_right(items, k)` ro'yxatni k qadam o'ngga aylantirsin.",
           "Write `rotate_right(items, k)` that rotates a list k steps to the right.",
           "def rotate_right(items, k):\n    if not items:\n        return []\n    k %= len(items)\n    return items[-k:] + items[:-k] if k else list(items)",
           ["assert rotate_right([1, 2, 3, 4], 1) == [4, 1, 2, 3]", "assert rotate_right([1, 2, 3], 5) == [2, 3, 1]",
            "assert rotate_right([], 3) == []", "assert rotate_right([1, 2], 0) == [1, 2]"]),
    _train("chunk", "`chunk(items, size)` ro'yxatni size uzunlikdagi bo'laklarga ajratsin.",
           "Write `chunk(items, size)` that splits a list into pieces of length size.",
           "def chunk(items, size):\n    return [items[i:i + size] for i in range(0, len(items), size)]",
           ["assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]", "assert chunk([], 3) == []"]),
    _train("run_length_encode", "`run_length_encode(s)`: 'aaabb' -> 'a3b2' ko'rinishida siqsin.",
           "Write `run_length_encode(s)` so that 'aaabb' becomes 'a3b2'.",
           "def run_length_encode(s):\n    out = []\n    i = 0\n    while i < len(s):\n        j = i\n        while j < len(s) and s[j] == s[i]:\n            j += 1\n        out.append(f\"{s[i]}{j - i}\")\n        i = j\n    return \"\".join(out)",
           ["assert run_length_encode('aaabb') == 'a3b2'", "assert run_length_encode('') == ''",
            "assert run_length_encode('abc') == 'a1b1c1'"]),
    _train("is_sorted", "`is_sorted(items)` ro'yxat o'sish tartibida ekanini tekshirsin.",
           "Write `is_sorted(items)` that checks whether a list is in non-decreasing order.",
           "def is_sorted(items):\n    return all(a <= b for a, b in zip(items, items[1:]))",
           ["assert is_sorted([1, 2, 2, 5])", "assert not is_sorted([3, 1])", "assert is_sorted([])"]),
    _train("digital_root", "`digital_root(n)`: raqamlar yig'indisini bitta raqam qolguncha hisoblasin.",
           "Write `digital_root(n)`: repeatedly sum the digits until one digit remains.",
           "def digital_root(n):\n    n = abs(n)\n    while n >= 10:\n        n = sum(int(d) for d in str(n))\n    return n",
           ["assert digital_root(942) == 6", "assert digital_root(7) == 7", "assert digital_root(0) == 0"]),
    _train("camel_to_snake", "`camel_to_snake(name)`: 'myVariableName' -> 'my_variable_name'.",
           "Write `camel_to_snake(name)` converting 'myVariableName' to 'my_variable_name'.",
           "def camel_to_snake(name):\n    out = []\n    for ch in name:\n        if ch.isupper():\n            if out:\n                out.append('_')\n            out.append(ch.lower())\n        else:\n            out.append(ch)\n    return ''.join(out)",
           ["assert camel_to_snake('myVariableName') == 'my_variable_name'", "assert camel_to_snake('Simple') == 'simple'",
            "assert camel_to_snake('x') == 'x'"]),
    _train("count_uppercase", "`count_uppercase(s)` satrdagi katta harflar sonini qaytarsin.",
           "Write `count_uppercase(s)` returning the number of uppercase letters.",
           "def count_uppercase(s):\n    return sum(1 for ch in s if ch.isupper())",
           ["assert count_uppercase('Salom Dunyo') == 2", "assert count_uppercase('abc') == 0"]),
    _train("interleave", "`interleave(a, b)` ikki ro'yxat elementlarini navbatma-navbat birlashtirsin.",
           "Write `interleave(a, b)` that alternates elements of two lists (leftovers at the end).",
           "def interleave(a, b):\n    out = []\n    for i in range(max(len(a), len(b))):\n        if i < len(a):\n            out.append(a[i])\n        if i < len(b):\n            out.append(b[i])\n    return out",
           ["assert interleave([1, 3], [2, 4, 6]) == [1, 2, 3, 4, 6]", "assert interleave([], []) == []"]),
    _train("triangular", "`triangular(n)` n-uchburchak sonni qaytarsin (1+2+...+n).",
           "Write `triangular(n)` returning the n-th triangular number (1+2+...+n).",
           "def triangular(n):\n    return n * (n + 1) // 2",
           ["assert triangular(1) == 1", "assert triangular(4) == 10", "assert triangular(0) == 0"]),
    _train("parse_kv", "`parse_kv(text)`: 'a=1;b=2' -> {'a': '1', 'b': '2'} lug'atini qaytarsin.",
           "Write `parse_kv(text)` turning 'a=1;b=2' into {'a': '1', 'b': '2'}.",
           "def parse_kv(text):\n    result = {}\n    for part in text.split(';'):\n        if '=' in part:\n            key, value = part.split('=', 1)\n            result[key.strip()] = value.strip()\n    return result",
           ["assert parse_kv('a=1;b=2') == {'a': '1', 'b': '2'}", "assert parse_kv('') == {}",
            "assert parse_kv('x = 5') == {'x': '5'}"]),
    _train("max_subarray_sum", "`max_subarray_sum(nums)` uzluksiz qism-ro'yxatning eng katta yig'indisini topsin.",
           "Write `max_subarray_sum(nums)` returning the largest sum of a contiguous sublist (non-empty input).",
           "def max_subarray_sum(nums):\n    best = cur = nums[0]\n    for x in nums[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best",
           ["assert max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6", "assert max_subarray_sum([-3, -1]) == -1"]),
    _train("remove_vowels", "`remove_vowels(s)` satrdan unli harflarni (a, e, i, o, u) olib tashlasin.",
           "Write `remove_vowels(s)` that removes the vowels a, e, i, o, u (any case).",
           "def remove_vowels(s):\n    return ''.join(ch for ch in s if ch.lower() not in 'aeiou')",
           ["assert remove_vowels('Moonlight') == 'Mnlght'", "assert remove_vowels('') == ''"]),
    _train("sum_of_squares", "`sum_of_squares(n)` 1² + 2² + ... + n² ni hisoblasin.",
           "Write `sum_of_squares(n)` computing 1² + 2² + ... + n².",
           "def sum_of_squares(n):\n    return sum(i * i for i in range(1, n + 1))",
           ["assert sum_of_squares(3) == 14", "assert sum_of_squares(0) == 0"]),
    _train("differences", "`differences(nums)` qo'shni elementlar farqlari ro'yxatini qaytarsin.",
           "Write `differences(nums)` returning the differences between neighbouring elements.",
           "def differences(nums):\n    return [b - a for a, b in zip(nums, nums[1:])]",
           ["assert differences([1, 4, 9]) == [3, 5]", "assert differences([5]) == []"]),
    _train("initials", "`initials(name)`: 'alisher navoiy' -> 'AN'.", "Write `initials(name)`: 'alisher navoiy' -> 'AN'.",
           "def initials(name):\n    return ''.join(w[0].upper() for w in name.split())",
           ["assert initials('alisher navoiy') == 'AN'", "assert initials('') == ''"]),
    _train("unique_sorted", "`unique_sorted(items)` takrorlarsiz, saralangan ro'yxat qaytarsin.",
           "Write `unique_sorted(items)` returning the distinct items in sorted order.",
           "def unique_sorted(items):\n    return sorted(set(items))",
           ["assert unique_sorted([3, 1, 3, 2]) == [1, 2, 3]", "assert unique_sorted([]) == []"]),
    _train("safe_divide", "`safe_divide(a, b, default=None)`: b nol bo'lsa default qaytarsin.",
           "Write `safe_divide(a, b, default=None)` that returns default when b is zero.",
           "def safe_divide(a, b, default=None):\n    return default if b == 0 else a / b",
           ["assert safe_divide(6, 3) == 2", "assert safe_divide(1, 0) is None", "assert safe_divide(1, 0, 0) == 0"]),
    _train("long_words", "`long_words(text, k)` k harfdan uzun so'zlar sonini qaytarsin.",
           "Write `long_words(text, k)` returning how many words are longer than k letters.",
           "def long_words(text, k):\n    return sum(1 for w in text.split() if len(w) > k)",
           ["assert long_words('men kitob o\\'qiyman', 3) == 2", "assert long_words('', 2) == 0"]),
    _train("zip_dict", "`zip_dict(keys, values)` ikki ro'yxatdan lug'at tuzsin.",
           "Write `zip_dict(keys, values)` building a dict from two lists (extra items ignored).",
           "def zip_dict(keys, values):\n    return dict(zip(keys, values))",
           ["assert zip_dict(['a', 'b'], [1, 2, 3]) == {'a': 1, 'b': 2}", "assert zip_dict([], []) == {}"]),
    _train("caesar", "`caesar(text, k)` kichik lotin harflarini k ga suradigan Sezar shifrini yozing.",
           "Write `caesar(text, k)`: a Caesar cipher shifting lowercase latin letters by k.",
           "def caesar(text, k):\n    out = []\n    for ch in text:\n        if 'a' <= ch <= 'z':\n            out.append(chr((ord(ch) - 97 + k) % 26 + 97))\n        else:\n            out.append(ch)\n    return ''.join(out)",
           ["assert caesar('abc', 1) == 'bcd'", "assert caesar('xyz', 3) == 'abc'", "assert caesar('a b!', 1) == 'b c!'"]),
    _train("median", "`median(nums)` sonlar medianasini qaytarsin.", "Write `median(nums)` returning the median of a non-empty list.",
           "def median(nums):\n    s = sorted(nums)\n    m = len(s) // 2\n    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2",
           ["assert median([3, 1, 2]) == 2", "assert median([4, 1, 3, 2]) == 2.5"]),
]


def verify_train_problems() -> List[str]:
    """Entry points of :data:`TRAIN_CODE_PROBLEMS` whose reference solution fails its tests."""
    return [p["entry_point"] for p in TRAIN_CODE_PROBLEMS
            if not _exec_ok(p["solution"] + "\n\n" + "\n".join(p["tests"]))[0]]


def _arith_prompt(rng: random.Random, uz: bool) -> Dict[str, Any]:
    op = _pick(rng, list(_OPS))
    a, b = _arith_operands(rng, op, rng.random() < 0.5)
    exact = Fraction(a) / Fraction(b) if op == "÷" else _OPS[op][1](a, b)
    q = (f"{a} {op} {b} ni hisoblang. Oxirgi qatorda \"Javob: <son>\" deb yozing." if uz
         else f"Compute {a} {op} {b}. End with a line \"Answer: <number>\".")
    return {"prompt_messages": [_user(q)], "reward_type": "exact_number", "reward_data": {"target": float(exact)}}


def _gsm8k_prompts(n: int, seed: int, cache_dir: Any, offline: Optional[bool]) -> List[Dict[str, Any]]:
    from .download import is_offline_env

    out: List[Dict[str, Any]] = []
    if offline is None:
        offline = is_offline_env()
    if not offline:
        try:
            from pathlib import Path

            from .adapters import to_rl_prompt
            from .download import iter_source_rows

            root = Path(str(cache_dir or "~/.cache/moonlight")).expanduser() / "raw"
            rows = list(iter_source_rows("gsm8k-train", root))
            random.Random(f"gsm8k:{seed}").shuffle(rows)
            for row in rows:
                item = to_rl_prompt("gsm8k-train", row)
                if item is not None:
                    item["prompt_messages"][0]["content"] += "\nEnd with a line \"Answer: <number>\"."
                    out.append(item)
                if len(out) >= n:
                    break
        except Exception as e:  # noqa: BLE001 - network / parsing failure -> synthetic fallback
            from ..utils import get_logger
            get_logger("moonlight.data").warning("gsm8k-train unavailable (%s); using synthetic word problems", e)
    rng = _rng("grpo_wp", seed)
    while len(out) < n:
        uz = len(out) % 2 == 0
        q, _, r, _ = word_problem(rng, uz)
        q += "\nOxirgi qatorda \"Javob: <son>\" deb yozing." if uz else "\nEnd with a line \"Answer: <number>\"."
        out.append({"prompt_messages": [_user(q)], "reward_type": "exact_number", "reward_data": {"target": float(r)}})
    return out


GRPO_KINDS = ("arithmetic", "constraints", "detail", "code_lite", "gsm8k")


def grpo_prompt_set(kinds: Sequence[str], n: int, seed: int = 0, cache_dir: Any = None,
                    offline: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Prompts with verifiable rewards for GRPO.

    Returns ``n`` dicts ``{"prompt_messages": [user msg], "reward_type": str, "reward_data": dict}``
    spread evenly over ``kinds`` (any of :data:`GRPO_KINDS`) and shuffled deterministically.
    Reward types: ``exact_number`` (``{"target": float}``), ``constraints`` (``{"constraints": [...]}``),
    ``code`` (``{"tests": [asserts], "entry_point": name}``); score with
    :func:`moonlight.data.verifiers.compute_reward`. ``gsm8k`` reads the ``gsm8k-train`` source (cached
    under ``cache_dir``, default ``~/.cache/moonlight``) unless ``offline`` is True (None -> the
    ``HF_HUB_OFFLINE`` environment decides); if it cannot be read, synthetic word problems are used.
    ``code_lite`` uses :data:`TRAIN_CODE_PROBLEMS` only (benchmark problems stay held-out).
    """
    kinds = [k for k in kinds]
    unknown = [k for k in kinds if k not in GRPO_KINDS]
    if unknown:
        raise ValueError(f"unknown GRPO prompt kinds {unknown}; supported: {GRPO_KINDS}")
    if not kinds or n <= 0:
        return []
    out: List[Dict[str, Any]] = []
    for kind, cnt in zip(kinds, _distribute(n, len(kinds))):
        if cnt == 0:
            continue
        if kind == "arithmetic":
            rng = _rng("grpo_arith", seed)
            out += [_arith_prompt(rng, i % 2 == 0) for i in range(cnt)]
        elif kind == "constraints":
            out += [{"prompt_messages": [_user(ex["prompt"])], "reward_type": "constraints",
                     "reward_data": {"constraints": ex["constraints"]}}
                    for ex in constraint_examples(cnt, seed=10_000 + seed)]
        elif kind == "detail":
            out += [{"prompt_messages": [_user(ex["prompt"])], "reward_type": "constraints",
                     "reward_data": {"constraints": ex["constraints"]}}
                    for ex in detail_examples(cnt, seed=10_000 + seed)]
        elif kind == "code_lite":
            rng = _rng("grpo_code", seed)
            for i in range(cnt):
                p = TRAIN_CODE_PROBLEMS[i % len(TRAIN_CODE_PROBLEMS)] if i < len(TRAIN_CODE_PROBLEMS) else _pick(
                    rng, TRAIN_CODE_PROBLEMS)
                uz = rng.random() < 0.5
                q = (p["uz"] + " Kodni ```python bloki ichida bering." if uz
                     else p["en"] + " Put the code in a ```python block.")
                out.append({"prompt_messages": [_user(q)], "reward_type": "code",
                            "reward_data": {"tests": list(p["tests"]), "entry_point": p["entry_point"]}})
        elif kind == "gsm8k":
            out += _gsm8k_prompts(cnt, seed, cache_dir, offline)
    random.Random(f"grpo_shuffle:{seed}").shuffle(out)
    return out


# =============================================================================================== preferences

def _violate(ex: Dict[str, Any], rng: random.Random) -> Optional[str]:
    """Realistic flawed variant of a constraint-following response that breaks >= 1 constraint."""
    text: str = ex["response"]
    lang = ex.get("lang", "uz")
    extra = "Bu ham muhim, albatta." if lang == "uz" else "This also matters, of course."
    cands = list(ex["constraints"])
    rng.shuffle(cands)
    for c in cands:
        t = c["type"]
        bad = None
        if t == "bullet_count":
            bad = text.rstrip("\"") + f"\n- {extra}"
        elif t == "paragraph_count":
            bad = text + f"\n\n***\n\n{extra}"
        elif t == "sections":
            bad = re.sub(r"SECTION \d+\n?", "", text, count=1)
        elif t == "include_keywords":
            bad = text
            for kw in c["keywords"]:
                bad = V._kw_pattern(kw).sub("", bad)
        elif t == "exclude_keywords":
            bad = text + f" {c['keywords'][0]}."
        elif t == "keyword_frequency":
            bad = V._kw_pattern(c["word"]).sub("", text)
        elif t == "json_format":
            bad = re.sub(r"[{}\[\]\"]", "", text)
        elif t == "lowercase":
            bad = "\n".join(_cap(line) for line in text.split("\n"))
        elif t == "uppercase":
            bad = text.lower()
        elif t == "no_commas":
            bad = text.replace(" ", ", ", 1)
        elif t == "ends_with":
            bad = text + " " + extra
        elif t == "starts_with":
            bad = (("Albatta! " if lang == "uz" else "Sure! ") + text)
        elif t == "title_brackets":
            bad = text.replace("<<", "").replace(">>", "")
        elif t == "quotation_wrapped":
            bad = text.strip("\"")
        elif t in ("word_count_max", "sentence_count_max", "max_chars"):
            bad = text + ("\n\n" + text)
            while V.check(bad, c):
                bad += "\n\n" + text
        elif t in ("word_count_min", "sentence_count_min"):
            bad = text.split(".")[0] + "."
        elif t == "language_uz":
            bad = "Sure! Here is my answer in English: reading and learning are important for everyone."
        if bad is not None and bad != text:
            passed, total, _ = V.check_all(bad, ex["constraints"])
            if passed < total:
                return bad
    return None


def preference_pairs_synthetic(n: int = 500, seed: int = 0) -> List[Tuple[Conversation, Conversation]]:
    """``n`` ``(chosen_msgs, rejected_msgs)`` pairs sharing the prompt. Chosen answers are correct,
    complete and format-following; rejected ones are realistic flawed variants: a missed constraint,
    a wrong number, a plan missing a required step, or an answer in the wrong language."""
    rng = _rng("preferences", seed)
    kinds = ["constraint", "number", "plan", "language", "detail"]
    out: List[Tuple[Conversation, Conversation]] = []
    c_pool = constraint_examples(max(8, n // 3 + 8), seed=20_000 + seed)
    d_pool = detail_examples(max(8, n // 5 + 8), seed=20_000 + seed)
    ci = di = 0
    guard = 0
    while len(out) < int(n) and guard < 20 * int(n) + 50:
        guard += 1
        kind = kinds[len(out) % len(kinds)] if guard <= int(n) else _pick(rng, kinds)
        uz = rng.random() < 0.5
        if kind in ("constraint", "detail"):
            pool = c_pool if kind == "constraint" else d_pool
            idx = ci if kind == "constraint" else di
            if kind == "constraint":
                ci += 1
            else:
                di += 1
            ex = pool[idx % len(pool)]
            bad = _violate(ex, rng)
            if bad is None:
                continue
            prompt = _user(ex["prompt"])
            out.append(([prompt, _asst(ex["response"])], [dict(prompt), _asst(bad)]))
        elif kind == "number":
            q, steps, r, final = word_problem(rng, uz)
            label = "Javob" if uz else "Answer"
            wrong = r + _pick(rng, [-10, -2, -1, 1, 2, 10])
            good = "<|think|>" + "\n".join(steps) + f"<|/think|>{final}\n{label}: {r}"
            bad_steps = steps[:-1] + [re.sub(r"= -?\d+", f"= {wrong}", steps[-1], count=1)]
            bad_final = final.replace(str(r), str(wrong))
            bad = "<|think|>" + "\n".join(bad_steps) + f"<|/think|>{bad_final}\n{label}: {wrong}"
            out.append(([_user(q), _asst(good)], [_user(q), _asst(bad)]))
        elif kind == "plan":
            state = rng.getstate()
            prompt, good, _ = _plan_example(rng, uz)
            rng.setstate(state)
            prompt2, bad, meta = _plan_example(rng, uz, drop_step=True)
            if prompt2 != prompt or bad == good or meta["removed"] is None:
                continue
            out.append(([_user(prompt), _asst(good)], [_user(prompt), _asst(bad)]))
        else:  # language switch: Uzbek question, English answer
            f = _pick(rng, FACTS)
            q = _pick(rng, f["uz_q"])
            out.append(([_user(q), _asst(f["uz_a"])], [_user(q), _asst(f["en_a"])]))
    return out


# =============================================================================================== config entry

_DEFAULT_COUNTS = {"identity": 300, "greetings": 200, "arithmetic": 1000, "tool_calls": 1000, "code": 500,
                   "think": 500, "plan": 300, "constraints": 300, "detail": 300, "projects": 100}
_ALIASES = {"greeting": "greetings", "tools": "tool_calls", "tool_call": "tool_calls", "math": "arithmetic",
            "reasoning": "think", "plans": "plan", "constraint": "constraints", "details": "detail",
            "project": "projects"}


def _count(value: Any, default: Optional[int]) -> Optional[int]:
    """Config value -> count: True -> default, False/None/0 -> 0, number -> int."""
    if value is True:
        return default
    if value is False or value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def build_synthetic(spec: Mapping[str, Any], name: str = "Moonlight", seed: int = 0) -> List[Conversation]:
    """Conversations for ``cfg.data.synthetic`` (mapping generator -> bool/count).

    Keys: ``identity`` (also adds greetings), ``greetings``, ``knowledge`` (True -> every fact x every
    phrasing), ``arithmetic``, ``tool_calls``, ``code``, ``think``, ``plan``, ``constraints``,
    ``detail``, ``projects``. Unknown keys are ignored with a warning. Deterministic for a given seed.
    """
    out: List[Conversation] = []
    seen = set()
    for raw_key, value in dict(spec or {}).items():
        key = _ALIASES.get(str(raw_key), str(raw_key))
        if key in seen:
            continue
        seen.add(key)
        if key == "knowledge":
            if value is True:
                out += knowledge_conversations(None, seed)
            elif _count(value, None):
                out += knowledge_conversations(_count(value, None), seed)
            continue
        if key not in _DEFAULT_COUNTS:
            from ..utils import get_logger
            get_logger("moonlight.data").warning("data.synthetic: unknown generator %r ignored", raw_key)
            continue
        cnt = _count(value, _DEFAULT_COUNTS[key]) or 0
        if cnt <= 0:
            continue
        if key == "identity":
            out += identity_conversations(name, cnt, seed)
            if "greetings" not in {_ALIASES.get(str(k), str(k)) for k in spec}:
                out += greeting_conversations(_DEFAULT_COUNTS["greetings"], seed)
        elif key == "greetings":
            out += greeting_conversations(cnt, seed)
        elif key == "arithmetic":
            out += arithmetic_conversations(cnt, seed)
        elif key == "tool_calls":
            out += tool_call_conversations(cnt, seed)
        elif key == "code":
            out += code_conversations(cnt, seed)
        elif key == "think":
            out += think_conversations(cnt, seed)
        elif key == "plan":
            out += plan_conversations(cnt, seed)
        elif key == "constraints":
            out += constraint_conversations(cnt, seed)
        elif key == "detail":
            out += detail_conversations(cnt, seed)
        elif key == "projects":
            out += project_conversations(cnt, seed)
    return out


def synthetic_texts(n: int = 2000, seed: int = 0, name: str = "Moonlight") -> List[str]:
    """Plain-text renderings (``User: ... / Assistant: ...``) of a mix of synthetic conversations, used
    for offline pretraining and tokenizer training."""
    from .adapters import conversation_to_text

    parts = _distribute(n, 9)
    convs: List[Conversation] = []
    convs += identity_conversations(name, parts[0], seed)
    convs += greeting_conversations(parts[1], seed)
    convs += knowledge_conversations(parts[2], seed)
    convs += arithmetic_conversations(parts[3], seed)
    convs += tool_call_conversations(parts[4], seed)
    convs += code_conversations(parts[5], seed)
    convs += think_conversations(parts[6], seed)
    convs += plan_conversations(parts[7], seed)
    convs += constraint_conversations(parts[8], seed)
    texts = [conversation_to_text(c) for c in convs]
    random.Random(f"synthetic_texts:{seed}").shuffle(texts)
    return [t for t in texts if t]
