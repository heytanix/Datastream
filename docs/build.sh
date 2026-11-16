#!/usr/bin/env bash
# Rebuild docs/architecture.pdf from the latest benchmark numbers.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f results/bench.json ] || .venv/bin/python run.py bench
python3 docs/make_results_tex.py
cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error architecture.tex >/dev/null && echo "built docs/architecture.pdf"
