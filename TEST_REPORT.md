# RB2301 v1.1.1 test report

Date: 2026-09-12

## Passed in the delivery environment

- 65 ROS-independent unit and static tests pass.
- A separate stress pass generated 100 accepted layouts at each difficulty
  (300 total). Maximum rejection-sampling attempts were 3 for easy, 3 for
  medium, and 8 for hard.
- Seeded maze generation is deterministic and maintains exactly 64 pooled cans.
- Every accepted maze contains the configured number of centre-route blockers.
- Inflated-grid A* verifies connectivity, minimum detour length, and minimum
  lateral excursion without carving a route before obstacle placement.
- `easy`, `medium`, and `hard` configurations all generate accepted layouts.
- The 90-degree-cone expert selects forward and lateral actions correctly,
  covers diagonal rays, and maintains its lateral commitment state.
- The `ca1.sh` controller and parallel collector instantiate the same canonical
  expert-policy class.
- Demonstration NPZ files validate shapes, finite values, and normalized actions.
- A mocked collection rollout verifies that pure expert episodes use only the
  direct, unfiltered step path and persist identical labelled/executed actions.
- Parallel episode allocation preserves the requested total and balances worker
  counts; static checks cover spawned collection, distinct seed streams, and
  four-worker automatic DAgger.
- Collection diagnostic loading combines JSONL records from all workers.
- Dataset aggregation reindexes episodes, and validation splitting keeps complete
  episodes together.
- Behavior-cloned and ROS policies use one canonical observation-space definition,
  preventing SB3 space mismatch on fine-tuning.
- Mocked worker tests verify initial-launch retries, transparent reset recovery,
  periodic recycling, and conversion of a step timeout into a
  `worker_restart` truncation.
- Gazebo service tests verify command construction, false responses, retries,
  worker-partition discovery, robot reset, and one-call movement of all 64 cans.
- Reward tests verify brake-fast/accelerate-smoothly action filtering, stronger
  terminal failures, proximity/smoothness costs, success, collision priority,
  and time-limit flags.
- Static checks verify continuous three-axis actions, 20 Hz LiDAR control,
  resume support, emergency saves, managed evaluation, isolated workers, and the
  installed imitation-learning entry points.
- All Python sources parse and byte-compile.
- All XML, URDF, and SDF files parse.
- All shell scripts pass `bash -n`.
- Both ROS packages report version 1.1.1.

The committed fixed world was regenerated with the medium configuration and
seed 2301. It contains 46 active cans, five centre-route blockers, and a
9.37 m inflated-grid path, plus parked entities up to the 64-model pool.

## Design checks represented by the tests

The previous generator guaranteed solvability by excluding obstacles around a
slowly wandering path. That made a central-forward shortcut common. Version
1.1.x tests the opposite order of operations: sample obstacles over the full
corridor, then accept only fields that satisfy independent path validation.

The lifecycle tests model the timeout that stopped the real 76,800-step run.
They verify that one worker can close its failed runtime, launch a replacement,
retry reset, and continue without propagating the original recoverable error to
the vector environment.

## Not runnable in the delivery environment

ROS 2 Jazzy, Gazebo Harmonic, Gymnasium, Stable-Baselines3, and PyTorch are not
installed in this container. Consequently, these checks require the target
Ubuntu robot-development machine:

- actual ROS topic rates and QoS compatibility;
- Gazebo rendering and service response under four workers;
- real process-group restart after an induced Gazebo failure;
- serialization and numerical training of the behavior-cloned PPO policy;
- expert success rate on the new maze distributions;
- end-to-end DAgger and PPO fine-tuning performance.

## Target-machine integration gate

Rebuild first:

```bash
./setup_rl.sh
./run_unit_tests.sh
```

Run a short automatic four-worker training check:

```bash
./train_rl.sh \
  --algorithm ppo \
  --num-envs 4 \
  --maze-difficulty easy \
  --timesteps 4096 \
  --checkpoint-frequency 2048 \
  --run-dir runs/v111_pipeline_check
```

Confirm that it completes and writes `final_model.zip`. A 4,096-transition run
is generally too short for ten complete episodes per worker, so it is not a
valid recycling test. Exercise recycling separately with a one-episode interval:

```bash
./train_rl.sh \
  --algorithm ppo \
  --num-envs 4 \
  --maze-difficulty easy \
  --timesteps 4096 \
  --worker-recycle-episodes 1 \
  --run-dir runs/v111_recycle_check
```

Confirm that `restart_events.jsonl` is written under at least one worker log,
training continues after the event, and `final_model.zip` is saved.

Run a small imitation gate before the full dataset:

```bash
EXPERT_EPISODES=10 \
DAGGER_EPISODES=5 \
BC_EPOCHS=3 \
DAGGER_EPOCHS=2 \
MAZE_DIFFICULTY=easy \
COLLECTION_ENVS=4 \
./run_imitation_pipeline.sh runs/imitation_smoke
```

Finally evaluate the resulting DAgger model and the PPO-refined model on fixed,
held-out maze seeds. Success, collision, out-of-bounds, time-limit, and
`worker_restart` outcomes must be reported separately.
