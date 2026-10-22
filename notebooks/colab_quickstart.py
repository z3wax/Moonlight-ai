# Moonlight - Colab / Kaggle quickstart (paste this whole file into ONE notebook cell and run it).
#
# What it does:
#   1. installs Moonlight from the PRIVATE GitHub repo (token from the GITHUB_TOKEN environment
#      variable, Colab "Secrets" (userdata) or an interactive prompt - never printed or saved),
#   2. trains `ml-20m-colab-chat` (time-budgeted for a free T4 GPU, ~1.5-2 h; resumes if the
#      runtime restarts and the cell is run again),
#   3. chats with the model (plain chat + tool calling),
#   4. runs the `cc-002` benchmark suite and saves the report.
#
# Settings below can be changed before running. Set QUICK_TEST = True for a ~5 minute smoke run.

import getpass
import importlib
import os
import subprocess
import sys

REPO = "github.com/z3wax/moonlight-ai.git"
BRANCH = os.environ.get("MOONLIGHT_BRANCH", "main")
CONFIG = "ml-20m-colab-chat"
SUITE = "cc-002"
QUICK_TEST = False                       # True -> tiny budgets (pipeline check only)
OUTPUT_DIR = "/content/runs/" + CONFIG if os.path.isdir("/content") else os.path.abspath("runs/" + CONFIG)
if os.path.isdir("/kaggle/working"):
    OUTPUT_DIR = "/kaggle/working/runs/" + CONFIG


def _github_token():
    """GITHUB_TOKEN env -> Colab/Kaggle secrets -> interactive prompt."""
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        return token
    try:  # Google Colab secrets
        from google.colab import userdata  # type: ignore

        token = (userdata.get("GITHUB_TOKEN") or "").strip()
    except Exception:
        token = ""
    if not token:
        try:  # Kaggle secrets
            from kaggle_secrets import UserSecretsClient  # type: ignore

            token = (UserSecretsClient().get_secret("GITHUB_TOKEN") or "").strip()
        except Exception:
            token = ""
    if not token:
        token = getpass.getpass("GitHub token (repo scope) for the private Moonlight repo: ").strip()
    if not token:
        raise SystemExit("A GitHub token is required to install the private repository.")
    return token


def _install():
    try:
        import moonlight  # noqa: F401

        print("moonlight already installed:", moonlight.__version__)
        return
    except ImportError:
        pass
    token = _github_token()
    url = f"git+https://{token}@{REPO}@{BRANCH}"
    print(f"Installing moonlight-llm from https://{REPO} ({BRANCH}) ...")
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"moonlight-llm[vision] @ {url}"],
                          capture_output=True, text=True)
    if proc.returncode != 0:  # never echo the token
        raise SystemExit("pip install failed:\n" + (proc.stdout + proc.stderr).replace(token, "***")[-4000:])
    importlib.invalidate_caches()


_install()

import torch  # noqa: E402

import moonlight as ml  # noqa: E402

print(f"moonlight {ml.__version__} | torch {torch.__version__} | "
      f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU - training will be slow)'}")

# ---------------------------------------------------------------------------------------------
# 1) config (+ optional overrides) and parameter report
# ---------------------------------------------------------------------------------------------
overrides = {"train.output_dir": OUTPUT_DIR}
if QUICK_TEST:
    overrides.update({
        "train.phases": [{"name": "pretrain", "minutes": 3, "lr_scale": 1.0},
                         {"name": "sft", "minutes": 2, "lr_scale": 0.5}],
        "tokenizer.train_chars": 2_000_000,
    })
cnf = ml.config(CONFIG, overrides=overrides)
print(ml.estimate_params(cnf).pretty())

# ---------------------------------------------------------------------------------------------
# 2) train (resumes automatically from OUTPUT_DIR/checkpoint.safetensors)
# ---------------------------------------------------------------------------------------------
llm = ml.LLM(cnf)
result = llm.train()
print("trained:", {k: result.get(k) for k in ("steps", "tokens", "final_val_loss", "model_path")})

# ---------------------------------------------------------------------------------------------
# 3) chat
# ---------------------------------------------------------------------------------------------
for prompt in ("Salom! Sen kimsan?", "O'zbekistonning poytaxti qaysi shahar?",
               "Write a Python function that reverses a string."):
    print(f"\n>>> {prompt}\n{llm.chat(prompt)}")

print("\n>>> 12 * 37 nechchi? (tools)")
print(llm.chat("12 * 37 nechchi?", tools=True))

chat = ml.ChatSession(llm, tools=True)
print("\n[multi-turn]", chat.send("Salom!"), "|", chat.send("Hozir soat nechchi (Toshkent vaqti)?"))

# ---------------------------------------------------------------------------------------------
# 4) benchmark
# ---------------------------------------------------------------------------------------------
report = cnf.bnchmrks.type(SUITE).run(llm, limit=20 if QUICK_TEST else None)
print(report.to_markdown())
report_path = os.path.join(OUTPUT_DIR, f"report-{SUITE}.md")
report.save(report_path)
print("report saved to", report_path, "| model:", result.get("model_path"))
