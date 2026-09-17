"""Conservative PPO fine-tuning anchored to a frozen imitation policy.

Stable-Baselines3's PPO implementation is intentionally left intact. This
subclass applies a small proximal pull to the actor after each PPO update,
    while leaving the value function and exploration scale free to learn. That is important for this
project: the DAgger actor is already competent, but its critic is not trained
by behavior cloning and early unconstrained PPO updates can catastrophically
forget the cloned policy.
"""

from __future__ import annotations

from typing import Any

import torch as th
from stable_baselines3 import PPO


class TeacherAnchoredPPO(PPO):
    """PPO with a conservative parameter-space anchor on the policy actor.

    ``anchor_strength`` is the fraction of the distance to the frozen teacher
    removed after every PPO update. Only the actor mean network is anchored;
    the Gaussian standard deviation is deliberately controlled independently
    by ``--action-std`` and the value network is left free to learn.
    """

    _ACTOR_PARAMETER_PREFIXES = (
        "mlp_extractor.policy_net.",
        "action_net.",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._teacher_policy: Any | None = None
        self.teacher_anchor_strength = 0.0
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        # The teacher is loaded separately for every run and must not be
        # serialized into every checkpoint.
        return [*super()._excluded_save_params(), "_teacher_policy"]

    def set_teacher(self, teacher_model: PPO, anchor_strength: float) -> None:
        """Attach a frozen teacher actor after construction or checkpoint load."""

        if not 0.0 <= anchor_strength <= 1.0:
            raise ValueError("anchor_strength must lie in [0, 1]")
        if self.observation_space != teacher_model.observation_space:
            raise ValueError("teacher and student observation spaces differ")
        if self.action_space != teacher_model.action_space:
            raise ValueError("teacher and student action spaces differ")

        teacher_policy = teacher_model.policy
        teacher_policy.set_training_mode(False)
        for parameter in teacher_policy.parameters():
            parameter.requires_grad_(False)
        self._teacher_policy = teacher_policy
        self.teacher_anchor_strength = float(anchor_strength)

        student_names = dict(self.policy.named_parameters())
        teacher_names = dict(teacher_policy.named_parameters())
        anchored_names = self._anchored_parameter_names(student_names, teacher_names)
        if not anchored_names:
            raise ValueError("no compatible actor parameters were found to anchor")

    @classmethod
    def _anchored_parameter_names(
        cls,
        student_names: dict[str, th.Tensor],
        teacher_names: dict[str, th.Tensor],
    ) -> list[str]:
        names: list[str] = []
        for name, student_parameter in student_names.items():
            is_actor = any(
                name.startswith(prefix) for prefix in cls._ACTOR_PARAMETER_PREFIXES
            )
            if not is_actor or name not in teacher_names:
                continue
            if student_parameter.shape != teacher_names[name].shape:
                raise ValueError(f"teacher parameter shape differs for {name}")
            names.append(name)
        return names

    def _apply_teacher_anchor(self) -> None:
        if self._teacher_policy is None or self.teacher_anchor_strength <= 0.0:
            return

        student_names = dict(self.policy.named_parameters())
        teacher_names = dict(self._teacher_policy.named_parameters())
        anchored_names = self._anchored_parameter_names(student_names, teacher_names)
        strength = self.teacher_anchor_strength
        squared_distance = 0.0
        with th.no_grad():
            for name in anchored_names:
                student_parameter = student_names[name]
                teacher_parameter = teacher_names[name].to(student_parameter.device)
                squared_distance += float(
                    th.mean(th.square(student_parameter - teacher_parameter)).cpu()
                )
                student_parameter.lerp_(teacher_parameter, strength)

        # PPO's logger is dumped by BaseAlgorithm after train() returns.
        self.logger.record("train/teacher_anchor_l2", squared_distance)
        self.logger.record("train/teacher_anchor_strength", strength)

    def train(self) -> None:
        super().train()
        self._apply_teacher_anchor()
