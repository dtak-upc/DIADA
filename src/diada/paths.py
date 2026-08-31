"""
Central, cwd-independent path resolution for the whole package.

Every path here is resolved from this file's own location rather than the
current working directory, so scripts under `diada.arda` / `diada.experiments`
/ `diada.pipeline` behave the same whether they're run as `python -m ...`,
imported from a notebook, or invoked through the CLI from an arbitrary
directory. This replaces the old convention (in arda_improved/ and
benchmarks_generation/) of hardcoded `../data/...`-style relative paths that
only worked if you `cd`'d into the right folder first.
"""

from __future__ import annotations

from pathlib import Path

# src/diada/paths.py -> parents[0]=diada, [1]=src, [2]=repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
BENCHMARKS_YAML = REPO_ROOT / "benchmarks.yaml"
JAR_PATH = REPO_ROOT / "src" / "diada" / "core" / "DIADA-0.8.jar"

BASE_DATASETS_DIR = DATA_DIR / "base_datasets"
OUTPUT_DIR = DATA_DIR / "output"
