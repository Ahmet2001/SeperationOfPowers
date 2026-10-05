"""Mercan Separation of Powers agent pipeline.

This module owns orchestration only. Model-specific inference stays inside
YasamaRuntime, YurutmeRuntime and YargiRuntime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from yasama.yasama import YasamaRuntime
from yargi.yargi import YargiRuntime
from yurutme.yurutme import YurutmeRuntime


ExecutorFn = Callable[[str, dict[str, Any]], Mapping[str, Any]]
ControlHandler = Callable[[str, Sequence[Mapping[str, Any]], Mapping[str, Any]], str]


class AgentPipeline:
    """Coordinates Yasama -> Yurutme -> Executor -> Yasama -> Yargi."""

    def __init__(
        self,
        *,
        yasama: YasamaRuntime,
        yurutme: YurutmeRuntime,
        yargi: YargiRuntime,
        tool_registry: Mapping[str, Mapping[str, Any]],
        executor: ExecutorFn,
        max_steps: int = 12,
        respond_handler: ControlHandler | None = None,
        clarification_handler: ControlHandler | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        self.yasama = yasama
        self.yurutme = yurutme
        self.yargi = yargi
        self.tool_registry = tool_registry
        self.executor = executor
        self.max_steps = max_steps
        self.respond_handler = respond_handler
        self.clarification_handler = clarification_handler

    def run(
        self,
        user_prompt: str,
        conversation_history: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        history = list(conversation_history or [])
        observations: list[dict[str, Any]] = []
        state: dict[str, Any] = {
            "completed_actions": [],
            "last_observation": None,
            "recent_observations": [],
            "step_index": 0,
            "turn_index": 0,
        }

        for step_index in range(self.max_steps):
            state["step_index"] = step_index

            action = self.yasama.run(
                user_prompt=user_prompt,
                conversation_history=history,
                state=state,
            )

            if action == "FINISH":
                return self.yargi.run(
                    user_prompt=user_prompt,
                    conversation_history=history,
                    observations=observations,
                )

            if action == "RESPOND":
                if self.respond_handler is not None:
                    return self.respond_handler(user_prompt, history, state)

                # Default: keep response generation out of Yasama and delegate
                # natural-language generation to Yargi without executing a tool.
                return self.yargi.run(
                    user_prompt=user_prompt,
                    conversation_history=history,
                    observations=observations,
                )

            if action == "ASK_CLARIFICATION":
                if self.clarification_handler is None:
                    raise RuntimeError(
                        "ASK_CLARIFICATION was selected, but the runtime "
                        "clarification policy is not configured yet."
                    )
                return self.clarification_handler(user_prompt, history, state)

            tool = self.tool_registry.get(action)
            if tool is None:
                raise KeyError(f"No tool registry entry for action {action!r}.")

            placeholder = tool.get("placeholder")
            if placeholder is not None and not isinstance(placeholder, Mapping):
                raise TypeError(f"Invalid placeholder for action {action!r}.")

            arguments = self.yurutme.run(
                user_prompt=user_prompt,
                conversation_history=history,
                state=state,
                action=action,
                tool_schema=tool,
                placeholder=placeholder,
            )

            observation = self._validate_observation(
                action=action,
                observation=self.executor(action, arguments),
            )
            observations.append(observation)

            state["completed_actions"].append(action)
            state["last_observation"] = observation
            state["recent_observations"] = observations.copy()

        raise RuntimeError(
            f"Agent reached max_steps={self.max_steps} without FINISH/RESPOND."
        )

    @staticmethod
    def _validate_observation(
        *,
        action: str,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        required = {"action", "data", "status", "error"}
        missing = required.difference(observation)
        if missing:
            raise ValueError(
                "Executor observation is missing required fields: "
                + ", ".join(sorted(missing))
            )

        normalized = dict(observation)
        if normalized["action"] != action:
            raise ValueError(
                f"Observation action mismatch: expected {action!r}, "
                f"got {normalized['action']!r}."
            )
        if normalized["status"] not in {"success", "error"}:
            raise ValueError("Observation status must be 'success' or 'error'.")

        return normalized
