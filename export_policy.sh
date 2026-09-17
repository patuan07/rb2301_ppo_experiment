#!/usr/bin/env bash
set -eo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
#source .venv/bin/activate
source install/setup.bash
set -u

# Development machine only: this loads PyTorch and Stable-Baselines3 to read the
# checkpoint.  The robot never runs it -- it consumes the .npz files it writes.
python -m rb2301_ca1.export_policy "$@"
