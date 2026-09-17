#!/usr/bin/env bash
set -eo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if [[ -z "${ROS_DISTRO:-}" ]]; then
  if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    source /opt/ros/jazzy/setup.bash
  else
    echo "ROS_DISTRO is not set and /opt/ros/jazzy/setup.bash was not found." >&2
    echo "Source your ROS 2 installation, then run this script again." >&2
    exit 1
  fi
fi

system_python="${RB2301_SYSTEM_PYTHON:-/usr/bin/python3.12}"
if [[ ! -x "$system_python" ]]; then
  echo "Ubuntu Python 3.12 was not found at $system_python." >&2
  echo "Install python3.12-venv, or set RB2301_SYSTEM_PYTHON explicitly." >&2
  exit 1
fi

if [[ -x .venv/bin/python ]]; then
  existing_version="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$existing_version" != "3.12" ]]; then
    echo ".venv uses Python $existing_version, but ROS 2 Jazzy requires Python 3.12." >&2
    echo "Move or remove .venv, then run this script again." >&2
    exit 1
  fi
fi

"$system_python" -m venv --system-site-packages .venv
source .venv/bin/activate
set -u
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
python -c 'import rclpy; print("ROS 2 Python import: OK")'
python -m pip install --upgrade pip
python -m pip install 'setuptools>=68,<80'
python -m pip install -r requirements-rl.txt
python -m pip check

if command -v rosdep >/dev/null 2>&1; then
  rosdep install --from-paths src --ignore-src --rosdistro "$ROS_DISTRO" -r -y
fi

colcon build --symlink-install
echo "Setup complete. Run ./run_unit_tests.sh, then follow README.md for training."
