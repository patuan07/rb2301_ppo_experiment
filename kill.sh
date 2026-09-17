#!/usr/bin/env bash
set -euo pipefail

find_pids() {
  pgrep -f "$1" || true
}

mapfile -t launch_pids < <(
  find_pids '/opt/ros/jazzy/bin/[r]os2 launch rb2301_gz ca1_gazebo.launch.py'
)
if (( ${#launch_pids[@]} > 0 )); then
  echo "Stopping RB2301 launch processes: ${launch_pids[*]}"
  kill -INT "${launch_pids[@]}" 2>/dev/null || true
fi

sleep 3

mapfile -t simulator_pids < <(
  find_pids '[g]z sim .*obstacle_world_ca1.sdf'
)
if (( ${#simulator_pids[@]} > 0 )); then
  echo "Stopping orphaned RB2301 Gazebo processes: ${simulator_pids[*]}"
  kill -TERM "${simulator_pids[@]}" 2>/dev/null || true
fi

echo "Remaining matching processes:"
pgrep -af '[r]os2 launch rb2301_gz ca1_gazebo.launch.py|[g]z sim .*obstacle_world_ca1.sdf' || true
