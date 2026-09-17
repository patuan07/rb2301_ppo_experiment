#!/usr/bin/env bash
set -eo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
source install/setup.bash
set -u
ros2 run rb2301_ca1 obstacle_avoidance "$@"
