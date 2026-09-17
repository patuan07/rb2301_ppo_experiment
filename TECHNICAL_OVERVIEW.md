# Technical overview: continuous holonomic maze learning

## Objective

This project learns a local navigation policy for the RB2301 holonomic robot.
At every control decision, the policy observes a 360-degree LiDAR scan, the
relative goal direction, the goal distance, and its preceding command. It emits
three simultaneous continuous commands: longitudinal velocity, lateral
velocity, and yaw rate. The learned controller therefore is not restricted to
the expert's forward/left/right/back actions.

The intended training sequence is:

1. Generate connected obstacle fields without reserving an empty route.
2. Collect safe demonstrations from the assignment's cone-based controller.
3. Pretrain a PPO actor by behavior cloning.
4. Use DAgger to label states visited by the imperfect student.
5. Fine-tune the policy with PPO on navigation return and smoothness.
6. Evaluate on held-out layout seeds, including harder obstacle distributions.

The expert supplies competence and safe recovery examples. It is not the final
action vocabulary or the performance ceiling.

## Software stack

| Layer | Software | Responsibility |
| --- | --- | --- |
| Operating system | Ubuntu 24.04 | Target runtime |
| Robotics | ROS 2 Jazzy / `rclpy` | Topics, messages, nodes, launch |
| Simulation | Gazebo Harmonic / `ros_gz` | Dynamics, LiDAR, odometry, entity poses |
| Environment API | Gymnasium | `reset`, `step`, spaces, episode semantics |
| Learning | Stable-Baselines3 | PPO, SAC, vector environments, checkpoints |
| Numerics | NumPy and PyTorch | Observation processing and optimization |
| Build | colcon / ament Python and CMake | ROS package installation |

The virtual environment is created with Ubuntu's Python 3.12 and
`--system-site-packages`. This is required because Jazzy's `rclpy` extension is
compiled for Python 3.12; a Conda Python 3.14 interpreter cannot import it. The
project pins `setuptools<80` to remain compatible with the installed colcon
version.

## ROS and simulation interface

The Gazebo model includes velocity control, odometry, and a 20 Hz GPU LiDAR.
`ros_gz_bridge` maps these interfaces:

- Gazebo LiDAR to ROS `/scan` as `sensor_msgs/LaserScan`;
- Gazebo odometry to ROS `/odom` as `nav_msgs/Odometry`;
- ROS `/cmd_vel` to Gazebo as a `geometry_msgs/Twist` command;
- Gazebo simulation time to ROS `/clock`.

One environment action is held until one new LiDAR scan and one newer odometry
message arrive. Thus the policy operates at a nominal 20 Hz and never consumes
the same scan twice as two different transitions. Time advances according to
sensor messages instead of a wall-clock sleep, which is important when Gazebo
runs slower than real time.

## Markov decision process

### Observation

The 42-element `float32` observation is

\[
o_t = [\ell_0,\ldots,\ell_{35},\;g_x^b,g_y^b,\;d_g,\;a_{t-1}^{applied}],
\]

where the 36 evenly sampled LiDAR ranges are divided by the 10 m maximum range,
the two-dimensional unit goal vector is rotated into the robot body frame, the
goal distance is divided by 8 m and clipped, and the preceding applied action
contains three values. All features are bounded.

The body-frame goal vector makes yaw controllable without ambiguity. The
previous applied command lets the policy account for the actuator filter and
makes the observation closer to Markovian.

### Action

The action is a continuous vector

\[
a_t=[a_x,a_y,a_\omega]\in[-1,1]^3.
\]

It is independently scaled to \(\pm0.4\) m/s in body-frame x and y and
\(\pm0.8\) rad/s about z. Learned actions use a safety-aware asymmetric
filter. Increasing magnitude follows:

\[
a_t^{applied}=a_{t-1}^{applied}+0.15(a_t-a_{t-1}^{applied}),
\]

while a magnitude decrease is immediate and a requested sign reversal first
sets that axis to zero. An avoidance decision can therefore brake forward
motion on the first tick instead of drifting toward an obstacle, while
acceleration into a new direction remains smooth. Arbitrary diagonal and curved
commands remain available.

Pure expert actions and DAgger safety interventions bypass this filter. They are
applied directly, matching the immediate Twist path used by `ca1.sh`.
Demonstration files retain both the raw expert label and the action actually
executed.

### Reward and termination

For a non-terminal transition, the reward is approximately

\[
r_t = 5\Delta x - 0.0025
-0.10\lVert\Delta a^{applied}\rVert_2^2
-0.002|\omega|
-0.05\,p(d_{min}),
\]

where \(p\) increases linearly as clearance falls below 0.45 m. Success adds 20.
Collision or leaving the corridor subtracts 40. The large failure penalties are
deliberate: maximum progress through the corridor must not make a late collision
look like a good episode. Collision also suppresses positive progress reward on
the terminal step.

The step charge is 0.0025 because 20 decisions per second produces the same
nominal 0.05 reward-per-second cost as the earlier slower interface. Episodes
end on success, collision, or boundary violation, and truncate after 800 actions
(about 40 seconds of simulation).

## Maze generation

Obstacle candidates span the full corridor rather than excluding a precomputed
route. Each episode samples a requested number of cans, inflates their occupancy
by a safety clearance, and runs eight-connected A*. Diagonal moves are rejected
when either adjacent cardinal cell is blocked, preventing corner cutting.

A candidate field is accepted only if:

- the inflated grid contains a start-to-goal path;
- enough cans intersect the straight centre band;
- the shortest path reaches a required lateral displacement; and
- the path is longer than the direct route by a minimum factor.

The validation path is discarded after acceptance and is never observed by the
policy. Solvability therefore does not create a visible empty channel. A fixed
pool of 64 Gazebo can entities is spawned once; unused models are parked under
the world, while episode resets move the entire pool in one
`set_pose_vector` service call.

| Level | Active cans | Clearance | Centre blockers | Lateral detour |
| --- | ---: | ---: | ---: | ---: |
| Easy | 32–40 | 0.28 m | at least 1 | at least 0.30 m |
| Medium | 40–48 | 0.30 m | at least 3 | at least 0.50 m |
| Hard | 48–58 | 0.32 m | at least 4 | at least 0.70 m |

These levels support curriculum learning. Easy layouts are suitable for the
first expert/behavior-cloning gate; medium is the default training distribution;
hard is best introduced after competence has been established.

## Imitation learning

### Expert

The expert uses the supplied four-heading strategy with 90-degree cones. It
tries forward first, preserves its previous lateral preference, uses the other
side next, and reverses only if necessary. Its actual ROS timer is 0.05 seconds,
matching the 20 Hz sensor. The expert returns normalized requested actions so it
uses the same interface as the neural policy. Collection calls the environment's
direct expert step, so demonstrated actuator behavior matches the comparison
node instead of passing cardinal switches through a low-pass filter.

### Behavior cloning

Given expert pairs \((o_i,a_i^E)\), behavior cloning minimizes

\[
L_{BC}(\theta)=\frac{1}{N}\sum_i
\lVert\mu_\theta(o_i)-a_i^E\rVert_2^2,
\]

where \(\mu_\theta\) is the mean of the PPO Gaussian actor. The train/validation
split is performed by episode, avoiding leakage of neighbouring transitions
from one trajectory. The saved actor's standard deviation is initialized to
0.30 rather than PPO's broad default, reducing destructive exploration at the
start of reinforcement learning.

Pure behavior cloning suffers from covariate shift: a small mistake can move the
student into a state that never appeared in expert demonstrations. DAgger
addresses this by allowing the student to control the robot while the expert
labels every visited state. Three rounds raise the student-control probability
from 0.35 to 0.70 to 1.00. A 0.24 m safety threshold can give control back to the
expert near obstacles. DAgger rounds retain task failures because difficult
recovery states are precisely the missing data; simulator-restart episodes are
excluded because they are infrastructure artifacts.

Collection is sharded across independent spawned processes. Each worker owns a
ROS domain, Gazebo partition, deterministic seed stream, diagnostic JSONL, and
periodically saved NPZ shard. The parent combines shards only after all workers
finish, avoiding large trajectory arrays over multiprocessing pipes.

The cardinal labels do not constrain the final policy to four actions. A neural
actor is continuous, the environment filters its commands, and PPO subsequently
optimizes without the cloning loss. It can interpolate between demonstrated
actions and discover simultaneous x/y/yaw commands that improve the reward.

## Reinforcement learning

PPO is the recommended first fine-tuner. It clips the probability ratio between
new and old policies, limiting destructive updates to a competent cloned actor:

\[
L^{clip}=\mathbb{E}\left[\min(r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\right],
\]

with \(\epsilon=0.2\), discount \(\gamma=0.99\), GAE
\(\lambda=0.95\), 512 steps per worker per rollout, batch size 256, and a
`[128,128]` multilayer perceptron. A learning rate of \(10^{-4}\) is recommended
after cloning; \(3\times10^{-4}\) is the default from scratch.

SAC is also implemented for continuous control and can be more sample-efficient,
but replay-buffer state must accompany a resumed model. PPO is operationally
simpler for the first end-to-end experiment and matches the behavior-cloning
actor format directly. DQN is intentionally not used because its action space is
discrete and would discard the holonomic controller's continuous degrees of
freedom.

The RTX 3050 can run either network, but these small MLPs rarely saturate a GPU.
Gazebo physics, sensor generation, ROS messaging, and service resets dominate
wall time. Multiple CPU-hosted headless workers usually improve collection more
than moving this policy update to CUDA. GPU becomes more valuable for larger
networks, image observations, or large off-policy batches.

## Fault tolerance and reproducibility

Each managed worker owns a separate ROS domain, Gazebo transport partition,
world file, launch process group, and log directory. Initial launch, reset, and
sensor timeouts trigger replacement attempts. A step-timeout becomes an explicit
zero-reward `worker_restart` boundary instead of killing the multiprocessing
pipe. Workers recycle every 200 episode starts by default to limit long-lived
Gazebo degradation.

PPO/SAC models can resume with their stored timestep counter. SAC can also load
its replay buffer. Checkpoints are written periodically; Ctrl+C writes an
`interrupted_model.zip`; an unexpected exception writes `emergency_model.zip`
and `emergency_error.txt`. Restart events and per-launch logs remain under each
worker directory.

Seeds reproduce the Python-side layout distribution and evaluation sequence.
Exact floating-point trajectories may still vary with simulator timing and
hardware, so results should be summarized over many held-out seeds rather than
from one visually inspected run.

## Evaluation protocol

Training return alone is insufficient. Evaluate deterministic policies on a
fixed held-out seed range and report:

- success rate with a confidence interval;
- collision, boundary, time-limit, and infrastructure-restart rates separately;
- return and episode-length distributions;
- minimum clearance and action-change statistics; and
- performance on easy, medium, and hard layouts.

Compare the expert, behavior-cloned model, final DAgger model, PPO checkpoints,
and a from-scratch PPO baseline under the same seeds. This ablation identifies
whether improvement comes from imitation, reinforcement learning, or an easier
maze distribution.

## Practical starting experiment

Use a small easy-maze pipeline as an integration gate, then collect the full
medium dataset. Four collectors and four PPO workers fit the validated target
laptop configuration. For PPO fine-tuning, start from DAgger 3 with a frozen
teacher anchor, `learning_rate=0.000025`, `ent_coef=0`, action standard
deviation `0.18`, `clip_range=0.10`, five epochs per rollout, and
`target_kl=0.01`. Use checkpoints every 25,000 transitions and evaluate them
with several isolated workers on fixed seeds. Select the best checkpoint by
held-out success rate, not by the latest training return.
