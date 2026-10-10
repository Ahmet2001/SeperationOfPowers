"""Mercan Separation of Powers agent pipeline.

This module owns orchestration only. Model-specific inference stays inside
YasamaRuntime, YurutmeRuntime and YargiRuntime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from runtime.local_tools import BUILTIN_TOOL_REGISTRY, execute_builtin_local
from runtime.tool_validation import validate_tool_arguments
from runtime.mail_clarification import (
    clarification_question,
    is_mail_cancel_request,
    is_mail_send_request,
    missing_mail_details,
)
from yasama.yasama import YasamaRuntime
from yargi.yargi import YargiRuntime
from yurutme.yurutme import YurutmeRuntime


ExecutorFn = Callable[[str, dict[str, Any]], Mapping[str, Any]]
ControlHandler = Callable[[str, Sequence[Mapping[str, Any]], Mapping[str, Any]], str]
TraceHandler = Callable[[str, Mapping[str, Any]], None]


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
        trace_handler: TraceHandler | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        self.yasama = yasama
        self.yurutme = yurutme
        self.yargi = yargi
        self.tool_registry = {
            **BUILTIN_TOOL_REGISTRY,
            **{key: dict(value) for key, value in tool_registry.items()},
        }
        self.executor = executor
        self.max_steps = max_steps
        self.respond_handler = respond_handler
        self.clarification_handler = clarification_handler
        self.trace_handler = trace_handler
        self.last_action: str | None = None

    def _trace(self, event: str, **payload: Any) -> None:
        if self.trace_handler is not None:
            self.trace_handler(event, payload)

    def run(
        self,
        user_prompt: str,
        conversation_history: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        self.last_action = None
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

            model_action = self.yasama.run(
                user_prompt=user_prompt,
                conversation_history=history,
                state=state,
            )
            action = model_action

            # Validate the CURRENT user's intent independently of chat history.
            # Generic small models sometimes carry the previous SEND_MAIL
            # action into unrelated turns such as "Nasılsın?". Never execute
            # or request mail details unless the current prompt explicitly
            # authorizes a send.
            if is_mail_cancel_request(user_prompt):
                action = "RESPOND"
            elif is_mail_send_request(user_prompt):
                if any(
                    item["action"] == "SEND_MAIL" and item["status"] == "success"
                    for item in observations
                ):
                    # Prevent duplicate delivery after a successful send.
                    action = "FINISH"
                else:
                    missing = missing_mail_details(user_prompt)
                    if missing:
                        action = "ASK_CLARIFICATION"
                        state["missing_mail_details"] = missing
                    elif action in {"RESPOND", "ASK_CLARIFICATION"}:
                        action = "SEND_MAIL"
            elif action == "SEND_MAIL":
                # Stale action from history is never a valid authorization.
                # If work already ran this turn, synthesize its observations.
                action = "FINISH" if observations else "RESPOND"

            self.last_action = action
            trace_fields = {"step": step_index, "action": action}
            if action != model_action:
                trace_fields["model_action"] = model_action
            self._trace("yasama", **trace_fields)

            if action == "FINISH":
                response = self.yargi.run(
                    user_prompt=user_prompt,
                    conversation_history=history,
                    observations=observations,
                )
                self._trace("yargi", response=response)
                return response

            if action == "RESPOND":
                if is_mail_cancel_request(user_prompt):
                    response = "Tamam, mail göndermeyeceğim."
                elif self.respond_handler is not None:
                    response = self.respond_handler(user_prompt, history, state)
                else:
                    response = self.yargi.run(
                        user_prompt=user_prompt,
                        conversation_history=history,
                        observations=observations,
                    )
                self._trace("yargi", response=response)
                return response

            if action == "ASK_CLARIFICATION":
                if self.clarification_handler is None:
                    response = clarification_question(
                        tuple(state.get("missing_mail_details", ()))
                    )
                else:
                    response = self.clarification_handler(user_prompt, history, state)
                self._trace("clarification", response=response)
                return response

            tool = self.tool_registry.get(action)
            if tool is None:
                raise KeyError(f"No tool registry entry for action {action!r}.")

            placeholder = tool.get("placeholder")
            if placeholder is not None and not isinstance(placeholder, Mapping):
                raise TypeError(f"Invalid placeholder for action {action!r}.")

            public_tool_schema = {
                key: value for key, value in tool.items() if not str(key).startswith("_")
            }
            arguments = self.yurutme.run(
                user_prompt=user_prompt,
                conversation_history=history,
                state=state,
                action=action,
                tool_schema=public_tool_schema,
                placeholder=placeholder,
            )
            arguments = validate_tool_arguments(
                action, arguments, public_tool_schema
            )
            self._trace("yurutme", step=step_index, action=action, arguments=arguments)

            if action == "SEND_MAIL":
                required = ("to", "subject", "body")
                invalid = [
                    field for field in required
                    if not isinstance(arguments.get(field), str)
                    or not arguments[field].strip()
                ]
                if invalid or set(arguments) != set(required):
                    raise ValueError(
                        "Yurutme must output only non-empty to/subject/body "
                        "fields for SEND_MAIL; got " + repr(arguments)
                    )
                if arguments["to"].strip().casefold() not in user_prompt.casefold():
                    raise ValueError(
                        "SEND_MAIL recipient was not explicitly provided "
                        "in the user's request."
                    )

            if tool.get("_builtin_executor") == "local_readonly":
                raw_observation = execute_builtin_local(action, arguments)
            else:
                raw_observation = self.executor(action, arguments)

            observation = self._validate_observation(
                action=action,
                observation=raw_observation,
            )
            self._trace("executor", step=step_index, observation=observation)
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
