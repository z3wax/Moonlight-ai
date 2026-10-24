# Moonlight

**Gibrid Attention + Retention til modellari oilasi (1M → 1T parametr) — o'zbek va ingliz tillari uchun.**
**A family of hybrid Attention + Retention language models (1M → 1T parameters) for Uzbek and English.**

[O'zbekcha](#ozbekcha) · [English](#english) · [Arxitektura](docs/ARCHITECTURE.md) ·
[Training](docs/TRAINING.md) · [Benchmarks](docs/BENCHMARKS.md) · [Tools](docs/TOOLS.md) ·
[100 configs](docs/CONFIGS.md)

---------------------------------------------------------------------------------------------------

<a id="ozbekcha"></a>
## O'zbekcha

### Moonlight nima?

Moonlight — bitta kutubxonada jamlangan to'liq til modeli tizimi:

* **model**: har bir qatlamda softmax-attention (GQA, RoPE, sirpanuvchi oyna) va **retention**
  (o'rganiladigan eksponensial so'nish bilan chiziqli rekurrensiya) boshlari birga ishlaydi; zich
  (dense) va Mixture-of-Experts variantlari, multi-token prediction, rasm kiritish (SigLIP);
* **ma'lumotlar**: o'zbek va ingliz matnlari, kod, suhbat (SFT), vosita chaqiruvlari, sintetik
  ma'lumotlar (tekshirilgan arifmetika, kod, rejalashtirish, cheklovlarga rioya qilish);
* **tokenizer**: o'zbekcha apostrofni (o'zbek, g'alaba) to'g'ri tushunadigan byte-level BPE;
* **o'qitish**: bitta GPU'dan ko'p tugunli klastergacha (DDP / FSDP), Muon optimizatori, WSD jadvali,
  bosqichlar: pretrain → (long) → (align) → SFT → DPO → GRPO;
* **suhbat**: ko'p bosqichli chat, vositalar (tools) bilan ishlash, rasmlar, reja tuzish va kichik
  loyihalarni testlari bilan qurish;
* **benchmarklar**: `cc-001` … `cc-010` darajali to'plamlar.

Barcha chat modellari **suhbat + kod yozish + vosita chaqirish** uchun o'qitiladi.

### Halol pozitsiya

Moonlight Transformer'lardan **o'lchab isbotlangan ustunlikni da'vo qilmaydi**. Bular — loyiha
maqsadlari (design advantages):

* retention boshlarida dekodlash holati **ketma-ketlik uzunligiga bog'liq emas** (O(1) xotira/token);
* sirpanuvchi oyna + kam sonli global qatlam tufayli **kichikroq KV-kesh**;
* **Muon** optimizatori va **multi-token prediction** orqali o'qitish samaradorligi;
* tekshiriladigan (verifier) mukofotlar bilan DPO/GRPO moslashtirish.

Bu maqsadlar haqiqatda qanchalik ishlashini **o'lchash** uchun `cc` benchmark to'plamlari bor —
natijalarni o'zingiz ishga tushirib, hisobotni e'lon qiling.

### Tez boshlash

Repozitoriy **yopiq (private)** — GitHub tokeni kerak (`repo` ruxsati bilan):

```bash
export GITHUB_TOKEN=ghp_...            # shaxsiy token
pip install "moonlight-llm @ git+https://${GITHUB_TOKEN}@github.com/z3wax/moonlight-ai.git"
# rasm (vision) qo'llab-quvvatlashi bilan:
pip install "moonlight-llm[vision] @ git+https://${GITHUB_TOKEN}@github.com/z3wax/moonlight-ai.git"
```

**Colab / Kaggle (bitta katak)**: [`notebooks/colab_quickstart.py`](notebooks/colab_quickstart.py)
faylini bitta katakka joylang va ishga tushiring — u repozitoriyni o'rnatadi, `ml-20m-colab-chat`
modelini o'qitadi, u bilan suhbatlashadi va `cc-002` benchmarkini ishga tushiradi.

### API

```python
import moonlight as ml

cnf = ml.config("ml-20m-colab-chat")       # registr nomi yoki .yaml yo'li
llm = ml.LLM(cnf)                          # model + tokenizer
llm.train()                                # cnf.train.phases bosqichlarini bajaradi (resume bilan)
print(llm.chat("Salom! O'zing haqingda gapirib ber."))

report = cnf.bnchmrks.type("cc-001").run(llm)
print(report.to_markdown())

llm.save("out/ml-20m-colab-chat.safetensors")
llm2 = ml.LLM.from_pretrained("out/ml-20m-colab-chat.safetensors")
```

Qo'shimcha: `ml.config(name, overrides={"train.batch_size": 16})`, `ml.list_configs(tier="test")`,
`ml.train("ml-20m-chat")` (bir qatorda: config → o'qitish → saqlash), `llm.benchmark("cc-002")`,
`ml.benchmarks.type("cc-001")`, `ml.estimate_params(cnf)`, `ml.ChatSession(llm, tools=True)`,
`llm.plan("...")`, `llm.build_project("...", "work/dir")`.

### Buyruq qatori (CLI)

```bash
moonlight list --tier test               # konfiguratsiyalar ro'yxati
moonlight info ml-20m-chat               # tavsif
moonlight params ml-70b-chat             # aniq parametrlar, xotira, FLOPs
moonlight train ml-20m-colab-chat --set train.batch_size=16
moonlight train ml-1m-debug --offline    # CPU'da ~1 daqiqalik to'liq sinov
moonlight chat runs/ml-20m-colab-chat --tools
moonlight bench runs/ml-20m-colab-chat --suite cc-002 --out report.md
moonlight plan  runs/ml-300m-chat "Talabalar uchun o'qish jadvalini rejalashtir"
moonlight build runs/ml-1b-agent "Testlari bilan so'zlarni sanovchi modul yoz" --workdir work/wc
moonlight gen-configs --check
```

Ko'p GPU: `torchrun --nproc_per_node 8 -m moonlight train ml-1b-chat`.

### 100 ta konfiguratsiya

To'liq jadval: [docs/CONFIGS.md](docs/CONFIGS.md).

| daraja | o'lchamlar | variantlar | soni | benchmark | apparat |
|---|---|---|---|---|---|
| `test` | 1M, 2M, 5M, 10M, 20M, 50M | base, chat, code, vl, tools, think, colab, kaggle, debug | 25 | cc-001 … cc-003 | CPU (debug), Colab T4 / Kaggle 2×T4 |
| `mini` | 100M, 150M, 250M, 300M | base, chat, code, vl, tools, think, long, uz, kaggle | 18 | cc-003 / cc-004 | Kaggle 2×T4, 1×A100 (DDP) |
| `si` (zich) | 1B, 2B, 3B, 7B, 13B, 30B, 70B, 175B, 405B, 1T | base, chat, code, agent, think, long, vl, vl-chat | 41 | cc-005 … cc-010 | 8 → 2048 GPU (FSDP) |
| `si` (MoE) | 16B-A2B, 30B-A3B, 100B-A10B, 235B-A22B, 671B-A37B, 1T-A32B | base, chat, code, vl | 16 | cc-007 … cc-010 | 512 → 2048 GPU |

### Benchmark darajalari

| to'plam | parametrlar | qo'shiladi |
|---|---|---|
| cc-001 | 1M–2M | chat formati, salomlashish, identitet, perplexity, tezlik |
| cc-002 | 5M–20M | + arifmetika, o'zbekcha bilim, tool formati, ifeval_lite, detail_check |
| cc-003 | 50M–100M | + code_lite (bajarish bilan), belebele-uz, plan_bench |
| cc-004 | 150M–300M | + arc-easy, hellaswag, humaneval (qism), tool e2e, ifeval |
| cc-005 | 1B–3B | + mmlu, arc-challenge, gsm8k, mbpp, humaneval, project_build |
| cc-006 | 7B–13B | to'liq mmlu/arc/hellaswag/gsm8k/humaneval/mbpp + uzun kontekst (needle) |
| cc-007 | 16B–70B | + rasm tavsifi (multimodal), ko'p vositali agent |
| cc-008 / 009 / 010 | 100B–1T | qat'iyroq chegaralar |

Batafsil: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### Vositalar (tool calling)

```python
import moonlight as ml

@ml.tool
def valyuta(miqdor: float, kurs: float = 12650.0) -> float:
    """Dollarni so'mga aylantiradi.

    Args:
        miqdor: dollar miqdori
        kurs: 1 dollar necha so'm
    """
    return miqdor * kurs

print(llm.chat("150 dollar necha so'm bo'ladi?", tools=[valyuta]))
print(llm.chat("12 * 37 nechchi?", tools=True))      # o'rnatilgan: calculator, current_time, unit_convert
```

Model `<|tool_call|>{"name": ..., "arguments": {...}}` yozadi, kutubxona vositani chaqiradi va
natijani `<|tool_result|>` sifatida qaytaradi. Batafsil: [docs/TOOLS.md](docs/TOOLS.md).

### Multimodal (rasm)

`vl` / `vl-chat` konfiguratsiyalari rasmni qabul qiladi (muzlatilgan SigLIP + o'qitiladigan proyektor):

```python
llm = ml.LLM.from_pretrained("runs/ml-20m-vl")
print(llm.chat("Rasmda nima tasvirlangan?", images=["mushuk.jpg"]))
```

**30B va undan katta** modellarda tanlov sizda: matnga ixtisoslashgan `ml-30b-chat` /
`ml-70b-chat`, yoki rasmni ham tushunadigan `ml-30b-vl`, `ml-30b-vl-chat`, `ml-70b-vl`,
`ml-175b-vl`, … `ml-1t-a32b-vl`.

### Apparat talablari (halol)

| daraja | minimal apparat |
|---|---|
| 1M–50M (`test`) | CPU (debug) yoki bitta T4 (Colab bepul) — daqiqalar/soatlar |
| 100M–300M (`mini`) | Kaggle 2×T4 yoki 1×A100 — bir necha soatdan bir necha kungacha |
| 1B–3B | 8×A100 (bitta tugun), FSDP |
| 7B–13B | 8–16×H100 |
| 30B–70B | 64–256×H100, ko'p tugunli klaster |
| 100B–1T, MoE | 512–2048×H100 |

**≥ 1B konfiguratsiyalar — katta klasterlar uchun mo'ljallangan namunaviy retseptlar.** FSDP
amalga oshirilgan; tensor- va expert-parallel (`tp_size`, `ep_size`) hozircha faqat hujjatlashtirilgan
joy egallovchilar (amalga oshirilmagan).

### Yo'l xaritasi

* **ovozdan-ovozga (speech-to-speech) yordamchi** — zaxiralangan `<|audio|>` tokeni orqali;
* tensor-parallel va expert-parallel o'qitish;
* v2 model: YaRN RoPE masshtablash, attention sinks, int8 KV-kesh, bo'laklab prefill, ixtiyoriy MLA,
  MTP orqali o'z-o'zini spekulyativ dekodlash, 2M tokengacha kontekst.

### Litsenziya

Apache-2.0 — [LICENSE](LICENSE).

---------------------------------------------------------------------------------------------------

<a id="english"></a>
## English

### What is Moonlight?

Moonlight is a complete language-model stack in one library:

* **model**: every layer mixes softmax-attention heads (GQA, RoPE, sliding window) with **retention**
  heads (a linear recurrence with a learned exponential decay); dense and Mixture-of-Experts
  variants, multi-token prediction, image input (SigLIP);
* **data**: Uzbek and English text, code, conversations (SFT), tool calls, and synthetic data (verified
  arithmetic, code, planning, constraint following);
* **tokenizer**: byte-level BPE that understands the Uzbek apostrophe (o'zbek, g'alaba);
* **training**: from one GPU to multi-node clusters (DDP / FSDP), Muon optimizer, WSD schedule,
  with phases pretrain → (long) → (align) → SFT → DPO → GRPO;
* **chat**: multi-turn chat, tool calling, images, planning, and building small projects with tests;
* **benchmarks**: tiered suites `cc-001` … `cc-010`.

All chat models are trained for **conversation + coding + tool calling**.

### Honest positioning

Moonlight makes **no claim of measured superiority** over Transformers. These are its design advantages:

* retention heads keep a decoding state whose size is **independent of sequence length** (O(1) memory per token);
* a **smaller KV cache** thanks to sliding-window layers and few global layers;
* training efficiency from the **Muon** optimizer and **multi-token prediction**;
* DPO/GRPO alignment with deterministic verifier rewards.

The `cc` benchmark suites exist to **measure** how far these goals hold in practice. Run them and
publish your reports.

### Quickstart

The repository is **private**, so you need a GitHub token with `repo` scope:

```bash
export GITHUB_TOKEN=ghp_...
pip install "moonlight-llm @ git+https://${GITHUB_TOKEN}@github.com/z3wax/moonlight-ai.git"
# with image support:
pip install "moonlight-llm[vision] @ git+https://${GITHUB_TOKEN}@github.com/z3wax/moonlight-ai.git"
```

**Colab / Kaggle (one cell)**: paste [`notebooks/colab_quickstart.py`](notebooks/colab_quickstart.py)
into a single cell and run it. It installs the repo, trains `ml-20m-colab-chat`, chats with the
model and runs the `cc-002` benchmark.

### API

```python
import moonlight as ml

cnf = ml.config("ml-20m-colab-chat")       # registry name or path to a .yaml
llm = ml.LLM(cnf)                          # model + tokenizer
llm.train()                                # runs cnf.train.phases (auto-resume)
print(llm.chat("Hi! Tell me about yourself."))

report = cnf.bnchmrks.type("cc-001").run(llm)
print(report.to_markdown())

llm.save("out/ml-20m-colab-chat.safetensors")
llm2 = ml.LLM.from_pretrained("out/ml-20m-colab-chat.safetensors")
```

Also available: `ml.config(name, overrides={"train.batch_size": 16})`, `ml.list_configs(tier="test")`,
`ml.train("ml-20m-chat")` (config → train → save in one line), `llm.benchmark("cc-002")`,
`ml.benchmarks.type("cc-001")`, `ml.estimate_params(cnf)`, `ml.ChatSession(llm, tools=True)`,
`llm.plan("...")`, `llm.build_project("...", "work/dir")`.

### Command line

```bash
moonlight list --tier test
moonlight info ml-20m-chat
moonlight params ml-70b-chat             # exact parameters, memory, FLOPs
moonlight train ml-20m-colab-chat --set train.batch_size=16
moonlight train ml-1m-debug --offline    # ~1 minute end-to-end check on CPU
moonlight chat runs/ml-20m-colab-chat --tools
moonlight bench runs/ml-20m-colab-chat --suite cc-002 --out report.md
moonlight plan  runs/ml-300m-chat "Plan a study schedule for students"
moonlight build runs/ml-1b-agent "Write a word-counter module with tests" --workdir work/wc
moonlight gen-configs --check
```

Multi-GPU: `torchrun --nproc_per_node 8 -m moonlight train ml-1b-chat`.

### The 100 configs

Full table: [docs/CONFIGS.md](docs/CONFIGS.md).

| tier | sizes | variants | count | benchmark | hardware |
|---|---|---|---|---|---|
| `test` | 1M, 2M, 5M, 10M, 20M, 50M | base, chat, code, vl, tools, think, colab, kaggle, debug | 25 | cc-001 … cc-003 | CPU (debug), Colab T4 / Kaggle 2×T4 |
| `mini` | 100M, 150M, 250M, 300M | base, chat, code, vl, tools, think, long, uz, kaggle | 18 | cc-003 / cc-004 | Kaggle 2×T4, 1×A100 (DDP) |
| `si` (dense) | 1B, 2B, 3B, 7B, 13B, 30B, 70B, 175B, 405B, 1T | base, chat, code, agent, think, long, vl, vl-chat | 41 | cc-005 … cc-010 | 8 → 2048 GPUs (FSDP) |
| `si` (MoE) | 16B-A2B, 30B-A3B, 100B-A10B, 235B-A22B, 671B-A37B, 1T-A32B | base, chat, code, vl | 16 | cc-007 … cc-010 | 512 → 2048 GPUs |

### Benchmark tiers

| suite | params | adds |
|---|---|---|
| cc-001 | 1M–2M | chat format, greetings, identity, perplexity, speed |
| cc-002 | 5M–20M | + arithmetic, Uzbek knowledge, tool format, ifeval_lite, detail_check |
| cc-003 | 50M–100M | + code_lite (executed), belebele-uz, plan_bench |
| cc-004 | 150M–300M | + arc-easy, hellaswag, humaneval (subset), tool e2e, ifeval |
| cc-005 | 1B–3B | + mmlu, arc-challenge, gsm8k, mbpp, humaneval, project_build |
| cc-006 | 7B–13B | full mmlu/arc/hellaswag/gsm8k/humaneval/mbpp + long-context needle |
| cc-007 | 16B–70B | + image captioning (multimodal), multi-tool agent |
| cc-008 / 009 / 010 | 100B–1T | stricter thresholds |

Details: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### Tool calling

```python
import moonlight as ml

@ml.tool
def currency(amount: float, rate: float = 12650.0) -> float:
    """Convert US dollars to Uzbek so'm.

    Args:
        amount: amount in USD
        rate: so'm per dollar
    """
    return amount * rate

print(llm.chat("How many so'm is 150 dollars?", tools=[currency]))
print(llm.chat("What is 12 * 37?", tools=True))     # built-ins: calculator, current_time, unit_convert
```

The model writes `<|tool_call|>{"name": ..., "arguments": {...}}`; the library runs the tool and
feeds the result back as `<|tool_result|>`. See [docs/TOOLS.md](docs/TOOLS.md).

### Multimodal (images)

`vl` / `vl-chat` configs accept images (a frozen SigLIP encoder plus a trainable projector):

```python
llm = ml.LLM.from_pretrained("runs/ml-20m-vl")
print(llm.chat("What is in this picture?", images=["cat.jpg"]))
```

**At 30B and above** you choose: text-focused `ml-30b-chat` / `ml-70b-chat`, or image-capable
`ml-30b-vl`, `ml-30b-vl-chat`, `ml-70b-vl`, `ml-175b-vl`, … `ml-1t-a32b-vl`.

### Hardware requirements (honest)

| tier | minimum hardware |
|---|---|
| 1M–50M (`test`) | CPU (debug) or one T4 (free Colab); minutes to hours |
| 100M–300M (`mini`) | Kaggle 2×T4 or 1×A100; hours to days |
| 1B–3B | 8×A100 (one node), FSDP |
| 7B–13B | 8–16×H100 |
| 30B–70B | 64–256×H100, multi-node cluster |
| 100B–1T, MoE | 512–2048×H100 |

**Configs ≥ 1B are reference recipes for large clusters.** FSDP is implemented; tensor and expert
parallelism (`tp_size`, `ep_size`) are documented placeholders and not implemented yet.

### Roadmap

* **speech-to-speech voice assistant** using the reserved `<|audio|>` token;
* tensor-parallel and expert-parallel training;
* v2 model: YaRN RoPE scaling, attention sinks, int8 KV cache, chunked prefill, optional MLA,
  MTP self-speculative decoding, up to 2M-token context.

### Development

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q                                  # CPU-only, a few minutes
python scripts/generate_configs.py --check
```

### License

Apache-2.0, see [LICENSE](LICENSE).


## Repozitoriy bo'yicha qo'llanma

[Arxitektura](docs/ARCHITECTURE.md) · [Konfiguratsiyalar](docs/CONFIGS.md) · [O'qitish](docs/TRAINING.md) · [Benchmarklar](docs/BENCHMARKS.md)

**Repozitoriy egasi:** [z3wax](https://github.com/z3wax). Mavjud uchinchi tomon muallifligi va litsenziya bildirishnomalari saqlanadi.

O'zgarish yuborish tartibi: [CONTRIBUTING.md](CONTRIBUTING.md).

### Muammoni yuborishda

- Branch va commit SHA, operatsion tizim hamda runtime versiyasini ko'rsating.
- Minimal qayta chiqarish qadamlari, kutilgan natija va amaldagi javobni yozing.
- Token, cookie, parol, sessiya va shaxsiy ma'lumotlarni loglardan olib tashlang.
- Benchmark natijasida qurilma, model/checkpoint, konfiguratsiya va o'lchov usulini kiriting.
