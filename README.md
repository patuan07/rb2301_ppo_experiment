# RB2301 Continuous LiDAR Navigation v1.1.1

This ROS 2 Jazzy and Gazebo Harmonic project trains a holonomic robot to reach
the end of an obstacle corridor using continuous forward, lateral, and yaw
commands. Version 1.1.1 adds exact expert execution and four-worker imitation
collection to the connected-maze, DAgger, and fault-tolerant training system.

See `TECHNICAL_OVERVIEW.md` for the standalone methodology, theory, parameter
justification, software architecture, and evaluation protocol.

## v1.1.1 corrective changes

- Expert and safety-intervention actions use `step_direct`, exactly matching the
  immediate Twist commands issued by `ca1.sh`.
- Learned actions use a brake-fast, accelerate-smoothly filter. Releasing an
  axis is immediate; increasing its magnitude is filtered with `alpha=0.15`.
- Demonstration collection supports `--num-envs`; the automatic pipeline uses
  four isolated collectors by default.
- Workers write recoverable NPZ shards every 25 retained episodes.
- Every collection attempt records layout seed, outcome, minimum clearance,
  final pose, requested/applied actions, action mode, and intervention counts.
- Infrastructure-restart episodes are diagnosed but never added to DAgger data.

## Major changes from v1.0.2

- Obstacles are sampled throughout the corridor; no route is carved first.
- Inflated-grid A* rejects disconnected and trivial straight-centre layouts.
- `easy`, `medium`, and `hard` maze distributions provide an explicit curriculum.
- The assignment expert uses four 90-degree cones at a 20 Hz sensor/control rate.
- Demonstration collection, behavior cloning, and three-round DAgger are included.
- PPO/SAC training can resume from a saved model.
- PPO fine-tuning can anchor actor updates to a frozen DAgger teacher so that
  the policy does not catastrophically forget a competent imitation policy.
- PPO exposes conservative fine-tuning controls (`--ppo-clip-range`,
  `--ppo-n-epochs`, and `--ppo-target-kl`).
- Managed evaluation supports several isolated workers and writes a JSON
  summary with Wilson confidence intervals.
- Gazebo workers restart after recoverable reset or sensor failures.
- Workers recycle periodically, after 200 episodes by default.
- Uncaught training failures save `emergency_model.zip` and a traceback.
- Collision and out-of-bounds penalties are now large enough that a late failure
  cannot appear successful merely because the robot travelled far forward.

The observation and action dimensions are unchanged, so an old PPO checkpoint
can be loaded. Its previous reward, control rate, and maze distribution were
different, however, so resumed results must be treated as transfer learning—not
as a continuation of an identical experiment.

## Task

| Component | v1.1.1 definition |
| --- | --- |
| Observation | 36 normalized LiDAR rays, body-frame goal direction, normalized goal distance, previous applied action: 42 floats |
| Policy action | Continuous requested `[x, y, yaw]` in `[-1, 1]^3` |
| Physical limits | ±0.4 m/s forward/lateral and ±0.8 rad/s yaw |
| Sensor/control rate | 20 Hz LiDAR, one fresh scan per policy action |
| Learned-action filter | Immediate braking/sign-change stop; filtered acceleration with `alpha=0.15` |
| Expert action | Applied directly with no filter, matching `ca1.sh` |
| Success | Robot centre reaches `x >= 7.2 m` without collision |
| Collision | Minimum sampled LiDAR range is at most 0.12 m |
| Out of bounds | `abs(y) >= 2.15 m` |
| Time limit | 800 actions, nominally 40 simulated seconds |

The default reward is:

```text
5.0 * forward_progress
- 0.0025 per action
- 0.10 * squared applied-action change
- 0.002 * absolute yaw rate
- up to 0.05 inside the 0.45 m safety distance
+ 20 on success
- 40 on collision or leaving the corridor
```

At 20 Hz, the step cost preserves the earlier cost of approximately 0.05 reward
per simulated second. The stronger terminal penalties prevent a late collision
from retaining a positive episode return.

## Maze generation

The generator first samples cans across the full `x=0.8..6.8`, `y=-2.0..2.0`
grid. It then inflates every can by a planning clearance and runs eight-connected
A*. A layout is retained only when:

1. A collision-inflated start-to-goal route exists.
2. Several cans obstruct the straight centre route.
3. The shortest route has a minimum lateral excursion.
4. Its path length exceeds the direct distance by a configured ratio.

This verifies solvability without disclosing the solution through an empty
corridor. The A* path is used only to accept or reject the simulated map; it is
not included in the policy observation.

| Difficulty | Active cans | Minimum centre blockers | Planning clearance | Required lateral excursion |
| --- | ---: | ---: | ---: | ---: |
| `easy` | 32–40 | 1 | 0.28 m | 0.30 m |
| `medium` | 40–48 | 3 | 0.30 m | 0.50 m |
| `hard` | 48–58 | 4 | 0.32 m | 0.70 m |

The fixed pool still contains 64 Gazebo entities. Unused cans are parked below
the world, and one `set_pose_vector` request randomizes a complete episode.

## Requirements and setup

- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Harmonic and `ros_gz`
- Ubuntu Python 3.12

Install the ROS dependencies if needed:

```bash
sudo apt update
sudo apt install \
  python3-colcon-common-extensions python3-rosdep python3-venv \
  ros-jazzy-ros-gz ros-jazzy-robot-state-publisher ros-jazzy-xacro
```

Then run:

```bash
chmod +x *.sh
./setup_rl.sh
./run_unit_tests.sh
```

The virtual environment deliberately uses `/usr/bin/python3.12` with
`--system-site-packages` so it can import Jazzy's binary `rclpy` extension.
`setuptools` remains below version 80 for compatibility with `colcon-core`.

## Train PPO or SAC

Training now owns its simulator even with one environment. Do not start
`gz_rl.sh` first unless using `--external-sim`.

```bash
./train_rl.sh \
  --algorithm ppo \
  --num-envs 4 \
  --maze-difficulty medium \
  --timesteps 250000 \
  --device cpu \
  --run-dir runs/ppo_medium_seed2301
```

For conservative PPO fine-tuning from DAgger 3, use a frozen teacher anchor.
The anchor is applied after each PPO update to the actor mean only; the value
network and action standard deviation are left free to learn. The following settings are a safe starting point for
this maze:

```bash
TEACHER=runs/imitation_medium/models/dagger_3_model.zip

./train_rl.sh \
  --algorithm ppo \
  --resume "$TEACHER" \
  --teacher-model "$TEACHER" \
  --teacher-anchor-strength 0.02 \
  --learning-rate 0.000025 \
  --ent-coef 0.0 \
  --action-std 0.18 \
  --ppo-clip-range 0.10 \
  --ppo-n-epochs 5 \
  --ppo-target-kl 0.01 \
  --num-envs 4 \
  --maze-difficulty medium \
  --timesteps 100000 \
  --checkpoint-frequency 25000 \
  --device cpu \
  --run-dir runs/ppo_medium_anchored
```

For a curriculum, run separate experiments or resume successively:

```bash
./train_rl.sh --algorithm ppo --num-envs 4 --maze-difficulty easy \
  --timesteps 100000 --run-dir runs/curriculum_easy

./train_rl.sh --algorithm ppo --num-envs 4 --maze-difficulty medium \
  --resume runs/curriculum_easy/final_model.zip \
  --learning-rate 0.0001 --timesteps 200000 \
  --run-dir runs/curriculum_medium
```

`--timesteps` means **additional transitions** when `--resume` is supplied.
The saved timestep counter is retained, so checkpoint names continue from the
stored count.

Resume the surviving 75k PPO checkpoint from the earlier run with:

```bash
./train_rl.sh \
  --algorithm ppo \
  --resume runs/OLD_RUN/checkpoints/rb2301_ppo_75000_steps.zip \
  --num-envs 4 \
  --maze-difficulty medium \
  --learning-rate 0.0001 \
  --timesteps 250000 \
  --run-dir runs/ppo_v111_from_75k
```

For SAC, pair a checkpoint with its replay buffer when possible:

```bash
./train_rl.sh \
  --algorithm sac \
  --resume MODEL.zip \
  --resume-replay-buffer REPLAY_BUFFER.pkl \
  --timesteps 250000
```

Checkpoint-style replay-buffer names are detected automatically when the files
are beside one another.

## Imitation learning and DAgger

The expert shares the same 36-ray preprocessing and uses forward, left, right,
and backward decisions with 90-degree cones. Pure expert actions and DAgger
safety interventions are applied directly, exactly as `ca1.sh` publishes them.
Student and PPO actions use the safety-aware filter.

### Complete automatic pipeline

The default pipeline uses four isolated simulators to collect 500 successful
expert episodes, trains a PPO actor by behavior cloning, performs three
parallel DAgger rounds, and prints the PPO fine-tuning command:

```bash
./run_imitation_pipeline.sh runs/imitation_medium
```

For a smaller pipeline test:

```bash
EXPERT_EPISODES=10 \
DAGGER_EPISODES=5 \
BC_EPOCHS=3 \
DAGGER_EPOCHS=2 \
MAZE_DIFFICULTY=easy \
COLLECTION_ENVS=4 \
./run_imitation_pipeline.sh runs/imitation_smoke
```

The final teacher-guided policy is:

```text
runs/imitation_medium/models/dagger_3_model.zip
```

Fine-tune it on the actual continuous-control reward:

```bash
./train_rl.sh \
  --algorithm ppo \
  --resume runs/imitation_medium/models/dagger_3_model.zip \
  --learning-rate 0.0001 \
  --num-envs 4 \
  --maze-difficulty medium \
  --timesteps 250000 \
  --run-dir runs/imitation_medium/ppo_finetune
```

Behavior cloning teaches the solver's safe decisions. DAgger labels states
visited by the student, including recovery states. PPO then removes the fixed
imitation objective and can discover simultaneous continuous `x/y/yaw` commands
that improve time, clearance, and command smoothness beyond the cardinal teacher.

### Manual demonstration and DAgger commands

Collect successful expert trajectories:

```bash
./collect_demonstrations.sh \
  --output datasets/expert_medium.npz \
  --episodes 500 \
  --num-envs 4 \
  --maze-difficulty medium \
  --run-dir runs/expert_medium_collection
```

Summarize successes and inspect the first failures:

```bash
./summarize_collection.sh runs/expert_medium_collection --show-failures 10
```

Each worker checkpoints a shard under `RUN/shards/`. If collection is
interrupted, these contain all complete episodes saved so far. The combined
`--output` file is written only after every worker reaches its assigned total.

Train the initial behavior-cloned policy:

```bash
./train_imitation.sh \
  --dataset datasets/expert_medium.npz \
  --output runs/bc_medium/model.zip \
  --epochs 50 \
  --action-std 0.30
```

Collect a DAgger round. `--no-successful-only` deliberately retains student
failure and recovery states:

```bash
./collect_demonstrations.sh \
  --append datasets/expert_medium.npz \
  --output datasets/dagger_1.npz \
  --episodes 100 \
  --num-envs 4 \
  --student-model runs/bc_medium/model.zip \
  --student-control-probability 0.5 \
  --no-successful-only
```

Retrain on the aggregated data:

```bash
./train_imitation.sh \
  --dataset datasets/dagger_1.npz \
  --initial-model runs/bc_medium/model.zip \
  --output runs/dagger_1/model.zip \
  --epochs 15
```

Dataset train/validation splits are performed by whole episode, preventing
neighbouring transitions from the same maze rollout from leaking across the
split. The cloned PPO action standard deviation is set to 0.30 instead of the
default 1.0 so initial stochastic fine-tuning does not immediately destroy the
expert behaviour.

## Run the expert directly

Launch the GUI simulator, then run the comparison controller:

```bash
# Terminal 1
./gz_ca1.sh maze_difficulty:=medium

# Terminal 2
source .venv/bin/activate
source install/setup.bash
ros2 run rb2301_ca1 obstacle_avoidance
```

The ROS timer and LiDAR both run at 20 Hz. The collector imports the exact same
`ConeExpertPolicy` class and applies its action without filtering. Do not run
this node while an RL policy controls the same `/cmd_vel` topic.

The RL `collision` outcome remains a conservative LiDAR proxy
(`minimum_scan <= 0.20 m`), not a Gazebo contact event. The collection log makes
the threshold, final pose, layout seed, and requested/applied action visible.

## Worker recovery and emergency saves

Each managed worker has a private ROS domain, Gazebo transport partition, world
file, launch process group, and log directory. On a recoverable reset failure:

1. Only that worker's launch group is stopped.
2. A fresh partition, world file, and simulator are started.
3. The failed reset is retried with the same Gym seed.

A sensor timeout during `step()` becomes a `worker_restart` truncation, after
which the vector worker resets normally. This rare infrastructure boundary has
zero reward so it is visible in `monitor.csv` without being treated as a robot
collision.

Workers are recycled every 200 episode starts by default, before the previously
observed long-run timeout range. Configure this with:

```bash
--worker-recycle-episodes 200 --worker-recovery-attempts 2
```

Set recycling to zero only for diagnosis. Restart events are recorded in:

```text
RUN/gazebo_logs/worker_XX/restart_events.jsonl
```

Normal completion writes `final_model.zip`. Ctrl+C writes
`interrupted_model.zip`. Any other exception writes:

```text
emergency_model.zip
emergency_error.txt
```

SAC additionally saves its replay buffer.

## Evaluate

Evaluation launches one isolated simulator automatically. For faster evaluation,
use several independent workers; each episode keeps its original seed and the
results are merged in episode order:

```bash
./evaluate_rl.sh MODEL.zip \
  --algorithm ppo \
  --episodes 100 \
  --num-envs 4 \
  --maze-difficulty medium
```

Report success, collision, out-of-bounds, time-limit, and `worker_restart`
outcomes separately. Return alone is not a navigation success metric. Managed
evaluation also writes `evaluation_summary.json`, including mean return,
episode length, per-episode results, and an approximate Wilson 95% interval for
success rate.

For visual evaluation:

```bash
# Terminal 1
./gz_ca1.sh maze_difficulty:=medium world_seed:=92301

# Terminal 2
./evaluate_rl.sh MODEL.zip --episodes 20 --external-sim
```

## Run artifacts

| Path | Contents |
| --- | --- |
| `run_config.json` | Release, maze, resume, worker, environment, and algorithm configuration |
| `monitor.csv` | Episode return, length, reason, clearance, and final pose |
| `tensorboard/` | PPO/SAC optimization and rollout metrics |
| `checkpoints/` | Periodic model and optional SAC replay snapshots |
| `evaluation_summary.json` | Per-episode deterministic results and success-rate confidence interval |
| `gazebo_logs/worker_XX/` | Per-restart worlds, launch logs, active PID, and restart history |
| `emergency_error.txt` | Full traceback for an uncaught training failure |
| `collection_config.json` | Parallel collector seeds, action modes, and environment definition |
| `collection_logs/worker_XX.jsonl` | One diagnostic record for every attempted episode |
| `shards/worker_XX.npz` | Periodically checkpointed per-worker demonstration data |
| `collection_summary.json` | Final per-worker counts and combined output metadata |

View training curves with:

```bash
source .venv/bin/activate
tensorboard --logdir runs
```

## Validation limits

The unit suite exercises the generator, A* validation, expert decisions,
dataset integrity, reward, restart/recycling state machine, service requests,
syntax, and package structure without requiring ROS. A real Gazebo smoke run is
still required on the target machine because this delivery environment does not
provide ROS 2 Jazzy, Gazebo Harmonic, or an NVIDIA render device.

Run the real integration checks after rebuilding:

```bash
./setup_rl.sh
./run_unit_tests.sh
./train_rl.sh --num-envs 4 --maze-difficulty easy \
  --timesteps 4096 --run-dir runs/v111_pipeline_check
./collect_demonstrations.sh --episodes 8 --num-envs 4 \
  --no-successful-only --maze-difficulty easy \
  --output runs/v111_teacher_probe/data.npz \
  --run-dir runs/v111_teacher_probe
./summarize_collection.sh runs/v111_teacher_probe
./evaluate_rl.sh runs/v111_pipeline_check/final_model.zip \
  --episodes 10 --maze-difficulty easy
```
