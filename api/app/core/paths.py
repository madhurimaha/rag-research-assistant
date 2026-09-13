"""Repo-root data locations. Resolved from this file so process cwd does not matter."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
EVAL_DIR = DATA_DIR / "eval"
GOLD_DIR = EVAL_DIR / "gold"
GOLD_PATH = GOLD_DIR / "uda_gold.json"
RESULTS_DIR = EVAL_DIR / "results"
