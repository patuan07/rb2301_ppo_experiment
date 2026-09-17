#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
PYTHONPATH=src/rb2301_ca1 python3 -m unittest discover -s tests -v
