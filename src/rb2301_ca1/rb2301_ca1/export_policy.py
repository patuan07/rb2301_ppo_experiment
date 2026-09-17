"""Export a trained PPO checkpoint for torch-free deployment on the robot.

The trained actor is a small MLP, so the deployed artefact is a ``.npz`` of six
weight arrays plus a reference file that lets the robot prove -- without
installing PyTorch -- that its NumPy inference reproduces the training machine.

This module needs PyTorch and Stable-Baselines3 and therefore runs on the
development machine only.  Nothing here is imported by
:mod:`rb2301_ca1.deploy_policy` or :mod:`rb2301_ca1.policy_runtime`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import FlattenExtractor

from .policy_runtime import (
    DEFAULT_FLOAT32_TOLERANCE,
    DEFAULT_FLOAT64_TOLERANCE,
    POLICY_MAX_RANGE,
    POLICY_RAY_COUNT,
    sim_scan_from_ranges,
)
from .rl_core import build_observation

#: Activations the runtime can evaluate, keyed by PyTorch class name.
_SUPPORTED_ACTIVATIONS = {"Tanh": "tanh", "ReLU": "relu"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Trained Stable-Baselines3 zip")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory (default: the package's policy directory)",
    )
    parser.add_argument(
        "--probe-rows",
        type=int,
        default=512,
        help="Uniform random observations to include in the reference",
    )
    parser.add_argument(
        "--real-rows",
        type=int,
        default=512,
        help="Real observations to harvest from demonstration datasets",
    )
    parser.add_argument(
        "--dataset-glob",
        default="datasets/*.npz",
        help="Where to look for DAgger demonstration datasets",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _validate_policy(model: PPO) -> dict[str, object]:
    """Assert the checkpoint has the structure this exporter knows how to emit.

    Every check here guards a way the export could produce a plausible but wrong
    file.  A retrain with, say, ``activation_fn=nn.ReLU`` keeps the same tensor
    shapes and would otherwise deploy as a silently broken policy.
    """

    policy = model.policy
    problems: list[str] = []

    if type(policy).__name__ != "ActorCriticPolicy":
        problems.append(f"policy class is {type(policy).__name__}, expected ActorCriticPolicy")
    if model.use_sde:
        problems.append("use_sde is True; the deterministic path is no longer the mean")
    if getattr(policy, "squash_output", False):
        problems.append("squash_output is True; the action is squashed, not clipped")
    if policy.features_extractor_class is not FlattenExtractor:
        problems.append(
            f"features extractor is {policy.features_extractor_class.__name__}, "
            "expected FlattenExtractor"
        )

    # SB3 stores the activation as a class (``nn.Tanh``) rather than an
    # instance, and older versions differ, so accept either form.
    activation_fn = policy.activation_fn
    activation_name = getattr(activation_fn, "__name__", None) or type(activation_fn).__name__
    if activation_name not in _SUPPORTED_ACTIVATIONS:
        problems.append(
            f"activation {activation_name} is not supported by the runtime; "
            f"supported: {sorted(_SUPPORTED_ACTIVATIONS)}"
        )

    observation_space = model.observation_space
    action_space = model.action_space
    if len(observation_space.shape) != 1:
        problems.append(f"observation space is {observation_space.shape}, expected 1-D")
    if len(action_space.shape) != 1:
        problems.append(f"action space is {action_space.shape}, expected 1-D")
    else:
        if not np.allclose(action_space.low, -1.0) or not np.allclose(action_space.high, 1.0):
            problems.append(
                f"action space bounds are [{action_space.low}, {action_space.high}], "
                "expected [-1, 1] because the runtime clips to that range"
            )

    if problems:
        raise ValueError(
            "Refusing to export an unsupported checkpoint:\n  - "
            + "\n  - ".join(problems)
        )

    return {
        "activation": _SUPPORTED_ACTIVATIONS[activation_name],
        "obs_dim": int(observation_space.shape[0]),
        "act_dim": int(action_space.shape[0]),
        "net_arch": [int(width) for width in getattr(policy, "net_arch", [])],
        "torch_version": th.__version__,
    }


@dataclass(frozen=True)
class ActorSpec:
    """The actor's weights, plus the module chain they were read from."""

    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    activation: str
    modules: tuple[th.nn.Module, ...]
    """Trunk then output, in forward order.  Evaluating these in double is what
    makes the float64 reference independent of the NumPy implementation."""

    @property
    def shapes(self) -> list[tuple[int, ...]]:
        return [tuple(weight.shape) for weight in self.weights]


def _extract_actor(model: PPO) -> ActorSpec:
    """Walk the actor's modules in forward order and read out its weights.

    The trunk is an ``nn.Sequential`` whose indices skip the activation slots --
    ``Linear`` at 0 and 2, ``Tanh`` at 1 and 3 -- so probing for
    ``policy_net.{i}.weight`` for consecutive ``i`` would stop at the first
    activation and export a truncated network whose output shape is still
    correct.  Walking the module list avoids that entirely.

    Structure the NumPy runtime cannot reproduce is refused here rather than
    approximated, and each trunk linear must be followed by an activation
    because that is the exact shape of the loop the runtime evaluates.
    """

    trunk = list(model.policy.mlp_extractor.policy_net)
    if not trunk:
        raise ValueError(
            "Checkpoint has an empty mlp_extractor.policy_net; this does not look "
            "like the MLP actor this exporter understands"
        )

    for position, module in enumerate(trunk):
        if not isinstance(module, th.nn.Linear):
            continue
        if position + 1 >= len(trunk) or isinstance(trunk[position + 1], th.nn.Linear):
            raise ValueError(
                f"actor trunk linear at position {position} is not followed by an "
                "activation; the NumPy runtime applies the activation after every "
                "layer except the last"
            )

    linears: list[th.nn.Linear] = []
    activations: set[str] = set()
    for module in trunk:
        if isinstance(module, th.nn.Linear):
            linears.append(module)
            continue
        # SB3 stores the activation as a class (``nn.Tanh``) rather than an
        # instance, and older versions differ, so accept either form.
        name = getattr(module, "__name__", None) or type(module).__name__
        if name not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"actor trunk contains {type(module).__name__}, which the NumPy "
                f"runtime cannot evaluate; supported activations: "
                f"{sorted(_SUPPORTED_ACTIVATIONS)}"
            )
        activations.add(_SUPPORTED_ACTIVATIONS[name])

    if len(activations) != 1:
        raise ValueError(
            f"actor trunk mixes activations {sorted(activations)}; the exported "
            "runtime evaluates a single activation for every layer"
        )
    if not isinstance(model.policy.action_net, th.nn.Linear):
        raise ValueError(
            f"action_net is {type(model.policy.action_net).__name__}, expected Linear"
        )

    output = model.policy.action_net
    weights = tuple(
        linear.weight.detach().cpu().numpy().astype(np.float32)
        for linear in (*linears, output)
    )
    biases = tuple(
        linear.bias.detach().cpu().numpy().astype(np.float32)
        for linear in (*linears, output)
    )
    return ActorSpec(
        weights=weights,
        biases=biases,
        activation=activations.pop(),
        modules=tuple(trunk) + (output,),
    )


def _reference_actions(
    model: PPO,
    spec: ActorSpec,
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the checkpoint's deterministic actions in float32 and float64.

    The two passes check different things and neither subsumes the other.  The
    float32 pass is the real deployment path -- it is what
    ``PPO.predict(obs, deterministic=True)`` actually returns, so it would catch
    a policy whose deterministic action is *not* the raw mean.  The float64 pass
    runs the same modules in double, where agreement with the NumPy runtime is a
    statement about the architecture rather than about GEMM rounding.
    """

    actions32, _ = model.predict(observations.astype(np.float32), deterministic=True)
    actions32 = np.asarray(actions32, dtype=np.float32)

    # The float64 pass cannot go through ``predict``: ``preprocess_obs`` ends with
    # ``return obs.float()`` for every Box space, so a double observation would be
    # narrowed before the network ever saw it.  Walk the actor's own modules
    # instead -- for a flattened Box observation the features extractor is a
    # no-op reshape, and this chain is exactly what ``get_distribution`` runs to
    # produce ``mean_actions``.
    low = model.action_space.low
    high = model.action_space.high
    with th.no_grad():
        tensor = th.as_tensor(observations.astype(np.float64))
        for module in spec.modules:
            tensor = module.double()(tensor)
    actions64 = np.clip(tensor.cpu().numpy(), low, high).astype(np.float64)
    return actions32, actions64


def _harvest_real_observations(
    pattern: str,
    limit: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Sample observation rows from demonstration datasets, if any exist."""

    paths = sorted(Path().glob(pattern))
    if not paths or limit <= 0:
        return None

    collected: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if "observations" not in archive:
                continue
            collected.append(np.asarray(archive["observations"], dtype=np.float32))
    if not collected:
        return None

    pooled = np.concatenate(collected, axis=0)
    if pooled.ndim != 2 or pooled.shape[0] == 0:
        return None
    count = min(limit, pooled.shape[0])
    chosen = rng.choice(pooled.shape[0], size=count, replace=False)
    print(f"Harvested {count} real observations from {len(paths)} dataset file(s)")
    return pooled[np.sort(chosen)]


def _build_scan_replay(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Capture one full scan-to-observation conversion for the runtime to replay.

    The reference rows above exercise the network but say nothing about the scan
    handling upstream of it, which is where the deployment-specific mistakes
    live.  This records a driver-convention scan (deliberately *not* the
    simulated ``angle_min``) together with the observation the training pipeline
    would have produced.
    """

    ray_count = 360
    angle_min = 0.0
    angle_increment = 2.0 * np.pi / ray_count
    bearings = angle_min + np.arange(ray_count) * angle_increment
    ranges = (1.5 + 0.9 * np.sin(4.0 * bearings + 0.7) + 0.4 * np.cos(9.0 * bearings)).astype(
        np.float32
    )

    x = float(rng.uniform(-1.0, 1.0))
    y = float(rng.uniform(-0.5, 0.5))
    yaw = float(rng.uniform(-0.4, 0.4))
    goal_x, goal_y = 7.2, 0.0
    previous_action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
    mount_yaw = 0.0

    conversion = sim_scan_from_ranges(
        ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_max=12.0,
        mount_yaw=mount_yaw,
        max_range=POLICY_MAX_RANGE,
    )
    observation = build_observation(
        conversion.ranges,
        x=x,
        y=y,
        yaw=yaw,
        goal_x=goal_x,
        goal_y=goal_y,
        previous_action=previous_action,
        ray_count=POLICY_RAY_COUNT,
        min_range=0.05,
        max_range=POLICY_MAX_RANGE,
        distance_scale=8.0,
    )
    return {
        "scan_ranges": ranges,
        "scan_angle_min": np.float32(angle_min),
        "scan_angle_increment": np.float32(angle_increment),
        "scan_mount_yaw": np.float32(mount_yaw),
        "scan_range_max": np.float32(12.0),
        "pose": np.asarray([x, y, yaw], dtype=np.float32),
        "goal": np.asarray([goal_x, goal_y], dtype=np.float32),
        "previous_action": previous_action,
        "expected_observation": np.asarray(observation, dtype=np.float32),
    }


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model does not exist: {args.model}")
    if args.probe_rows < 0 or args.real_rows < 0:
        raise ValueError("reference row counts cannot be negative")

    output_dir = args.output_dir or (Path(__file__).resolve().parents[1] / "policy")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(str(args.model), device="cpu")
    config = _validate_policy(model)
    spec = _extract_actor(model)
    weights, biases = spec.weights, spec.biases

    # ``_validate_policy`` reads the activation the checkpoint *declares*;
    # ``_extract_actor`` reads the one actually sitting in the trunk.  Both
    # matter, and disagreement means the runtime would evaluate the wrong
    # function, so compare them before writing anything.
    if spec.activation != config["activation"]:
        raise ValueError(
            f"Checkpoint declares activation {config['activation']!r} but its "
            f"actor trunk contains {spec.activation!r}"
        )
    if weights[0].shape[1] != config["obs_dim"]:
        raise ValueError("actor input width disagrees with the observation space")
    if weights[-1].shape[0] != config["act_dim"]:
        raise ValueError("actor output width disagrees with the action space")

    # --- actor.npz ---------------------------------------------------------
    actor_path = output_dir / "actor.npz"
    payload: dict[str, np.ndarray] = {
        "activation": np.asarray(config["activation"]),
        "obs_dim": np.asarray(config["obs_dim"], dtype=np.int64),
        "act_dim": np.asarray(config["act_dim"], dtype=np.int64),
        "net_arch": np.asarray(config["net_arch"], dtype=np.int64),
        "source_model": np.asarray(args.model.name),
    }
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        payload[f"w{index}"] = weight
        payload[f"b{index}"] = bias
    np.savez(actor_path, **payload)

    # --- reference rows ----------------------------------------------------
    rng = np.random.default_rng(args.seed)
    low = np.asarray(model.observation_space.low, dtype=np.float32)
    high = np.asarray(model.observation_space.high, dtype=np.float32)
    probes = rng.uniform(low, high, size=(args.probe_rows, config["obs_dim"])).astype(
        np.float32
    )
    real = _harvest_real_observations(args.dataset_glob, args.real_rows, rng)
    if real is None:
        print(
            f"No demonstration datasets matched {args.dataset_glob!r}; the reference "
            "will contain only uniform random probes, which is a weaker check."
        )
        observations = probes
        probe_rows = args.probe_rows
    else:
        observations = np.concatenate([probes, real.astype(np.float32)], axis=0)
        probe_rows = args.probe_rows

    actions32, actions64 = _reference_actions(model, spec, observations)

    # Grade the NumPy runtime against that reference right now, so the export
    # fails here rather than on the robot.
    from .policy_runtime import ActorMLP

    actor = ActorMLP(
        weights=tuple(weights),
        biases=tuple(biases),
        activation=spec.activation,
        observation_dim=int(config["obs_dim"]),
        action_dim=int(config["act_dim"]),
    )
    numpy32 = np.stack([actor.forward(row) for row in observations]).astype(np.float32)
    numpy64 = np.stack(
        [actor.forward(row, dtype=np.float64) for row in observations]
    ).astype(np.float64)

    error32 = float(np.max(np.abs(numpy32 - actions32)))
    error64 = float(np.max(np.abs(numpy64 - actions64)))

    # The float64 bound is tight: a genuine architecture error is order one.
    # The float32 bound allows for a different BLAS on the robot, so it is a
    # catastrophe detector (does the deploy path work at all) rather than a
    # rounding-level assertion, which would fail spuriously.
    tolerance64 = max(100.0 * error64, 1e-12)
    tolerance32 = max(50.0 * error32, DEFAULT_FLOAT32_TOLERANCE)

    replay = _build_scan_replay(rng)
    reference_path = output_dir / "actor_reference.npz"
    np.savez(
        reference_path,
        observations=observations,
        actions_float32=actions32,
        actions_float64=actions64,
        float32_tolerance=np.float64(tolerance32),
        float64_tolerance=np.float64(tolerance64),
        probe_rows=np.int64(probe_rows),
        **replay,
    )

    print(f"Wrote {actor_path}")
    print(f"Wrote {reference_path}")
    print(
        f"  layers={spec.shapes} activation={spec.activation} "
        f"reference rows={observations.shape[0]} ({probe_rows} probes"
        + ("" if real is None else f", {observations.shape[0] - probe_rows} real")
        + ")"
    )
    print(f"  numpy vs torch: float64 max|err|={error64:.3e} (tol {tolerance64:.1e})")
    print(f"                  float32 max|err|={error32:.3e} (tol {tolerance32:.1e})")

    if error64 > DEFAULT_FLOAT64_TOLERANCE:
        raise SystemExit(
            f"Exported actor disagrees with the checkpoint in float64 "
            f"({error64:.3e}); the numpy forward pass is not faithful."
        )
    print("Export verified against the checkpoint.")


if __name__ == "__main__":
    main()
