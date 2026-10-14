#!/usr/bin/env bash
# Assemble the MkDocs source tree in docs/ from the repo, so the repo itself keeps its
# plain layout (README.md, notes/, practice/, code/) for browsing on GitHub.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf docs
mkdir docs
cp -R 00_formula_sheet.md 00_comparison_tables.md notes practice code quiz.html docs/
cp README.md docs/index.md                 # the README becomes the site's home page
cp -R scripts/site_assets/. docs/          # MathJax config
rm -rf docs/code/__pycache__
