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
  echo "Set RB2301_SYSTEM_PYTHON to a Python 3.12 interpreter." >&2
  exit 1
fi

echo "Deployment installs no Python packages."
echo "The exported policy runs on NumPy, which ROS 2 Jazzy already provides;"
echo "requirements-rl.txt is a training-time dependency list.  See"
echo "requirements-deploy.txt for the details."

set -u
"$system_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
"$system_python" -c 'import rclpy; print("ROS 2 Python import: OK")'

if ! "$system_python" -c 'import numpy' 2>/dev/null; then
  echo "NumPy is missing.  Install it from apt, which is what ROS 2 expects:" >&2
  echo "    sudo apt install python3-numpy" >&2
  echo "Do not use pip: it would shadow the system package rclpy was built against." >&2
  exit 1
fi
"$system_python" -c 'import numpy; print("NumPy import: OK (" + numpy.__version__ + ")")'

# The actor must be exported on the training machine before this will build, so
# say so here rather than failing later inside colcon.
if ! compgen -G "src/rb2301_ca1/policy/*.npz" >/dev/null; then
  echo "No exported policy found in src/rb2301_ca1/policy/*.npz." >&2
  echo "On the training machine, run:" >&2
  echo "    python -m rb2301_ca1.export_policy policy/best_model_ppo.zip" >&2
  echo "and commit the resulting actor.npz and actor_reference.npz." >&2
  exit 1
fi

if [[ -d .venv ]]; then
  echo "NOTE: a .venv is present.  The robot does not need one -- this script"
  echo "      installs nothing -- so it can be removed if storage is tight."
fi

if command -v rosdep >/dev/null 2>&1; then
  rosdep install --from-paths src --ignore-src --rosdistro "$ROS_DISTRO" -r -y
fi

# --symlink-install makes the installed actor a symlink back into this tree, so
# do not move or delete src/rb2301_ca1/policy/actor.npz after building.
colcon build --symlink-install

echo
echo "Setup complete.  Nothing was pip-installed."
echo "Prove NumPy inference matches the training machine, with no PyTorch present:"
echo "    ./deploy_policy.sh --self-test"
echo "Then, with base_control_ros2 and rplidar_ros already running:"
echo "    ./deploy_policy.sh --dry-run"
