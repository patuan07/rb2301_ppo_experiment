#!/usr/bin/env bash
set -eo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
source .venv/bin/activate
source install/setup.bash
set -u

python -m rb2301_ca1.train_imitation "$@"
