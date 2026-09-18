"""One native W&B row per completed optimizer update, including its media."""

from __future__ import annotations

from typing import Any


class NativeUpdateHistory:
    def __init__(self, run: Any) -> None:
        self.run = run
        self._rollout: tuple[int, dict[str, Any]] | None = None

    def stage_rollout(self, policy_version: int, payload: dict[str, Any]) -> None:
        if self._rollout is not None:
            raise RuntimeError("A rollout is already waiting for its optimizer update")
        self.require_next_step(policy_version + 1)
        self._rollout = (policy_version, dict(payload))

    def training_payload(self, step: int, payload: dict[str, Any]) -> dict[str, Any]:
        if self._rollout is None or self._rollout[0] != step - 1:
            raise RuntimeError("Completed update has no matching pre-update rollout")
        version, rollout = self._rollout
        return {
            **rollout,
            **payload,
            "experiment/update": step,
            "train_details/policy_update": version,
        }

    def require_next_step(self, step: int) -> None:
        current = self.run.step
        if type(step) is not int or step < 0 or current != step:
            raise RuntimeError(
                "W&B native step mismatch before writing: "
                f"next={current}, requested={step}; refusing a gap or duplicate"
            )

    def commit(self, step: int, payload: dict[str, Any]) -> None:
        self.require_next_step(step)
        if payload.get("experiment/update") != step:
            raise RuntimeError("W&B native step and experiment/update must agree")
        self.run.log(payload, step=step, commit=True)
        if self.run.step != step + 1:
            raise RuntimeError("W&B did not acknowledge exactly one native history row")
        self._rollout = None

    def discard_unfinished(self) -> None:
        # Closing or resuming must never manufacture a completed update.
        self._rollout = None
