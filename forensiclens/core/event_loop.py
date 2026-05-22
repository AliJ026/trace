"""Deterministic asynchronous execution engine for ForensicLens."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Collection, Dict, List, Mapping, Protocol, Sequence, Tuple

from forensiclens.core.state_manager import StateManager
from forensiclens.telemetry.interceptor import parse_scratchpad_and_tool


ChatMessage = Dict[str, str]


class AgentProtocol(Protocol):
    """Minimal protocol for model clients used by the deterministic loop."""

    async def generate_response(self, conversation_history: Sequence[ChatMessage]) -> str:
        ...


class ScenarioProtocol(Protocol):
    """Minimal protocol for adversarial scenarios."""

    def get_system_prompt(self, is_monitored: bool) -> str:
        ...

    def generate_escalation(self, turn: int) -> Tuple[float, str]:
        ...


class TelemetryLoggerProtocol(Protocol):
    """Minimal protocol for external JSONL logger implementations."""

    async def write_entry(
        self,
        *,
        run_id: str,
        model_name: str,
        is_monitored: bool,
        entry: Mapping[str, Any],
    ) -> None:
        ...


ToolExecutor = Callable[[str, Dict[str, Any], StateManager], Awaitable[str]]
SemanticDivergenceFn = Callable[[str, Dict[str, Any]], Awaitable[float]]


async def agentic_execution_loop(
    *,
    run_id: str,
    tested_agent: AgentProtocol,
    scenario: ScenarioProtocol,
    is_monitored: bool,
    logger_instance: TelemetryLoggerProtocol,
    execute_tool: ToolExecutor,
    max_turns: int = 3,
    allowed_tools: Collection[str] | None = None,
    calculate_semantic_divergence: SemanticDivergenceFn | None = None,
) -> List[Dict[str, Any]]:
    """
    Run deterministic scenario execution while enforcing strict telemetry ordering:
    parse -> validate -> log -> then optional tool execution.
    """
    state_manager = StateManager(max_turns=max_turns)

    conversation_history: List[ChatMessage] = [
        {"role": "system", "content": scenario.get_system_prompt(is_monitored)}
    ]
    no_tool_retry_message = (
        "Your previous response was invalid because no tool was selected. "
        "Return exactly one JSON tool call."
    )

    for turn in range(max_turns):
        if state_manager.is_terminal_state_reached():
            break

        pressure_scalar, escalation_prompt = scenario.generate_escalation(turn)
        state_manager.advance_turn(pressure_scalar)
        conversation_history.append({"role": "user", "content": escalation_prompt})

        for attempt in range(2):
            raw_response = await tested_agent.generate_response(conversation_history)

            interception = parse_scratchpad_and_tool(
                raw_response,
                allowed_tools=allowed_tools,
            )

            semantic_divergence = 0.0
            if (
                calculate_semantic_divergence is not None
                and interception.tool_intent is not None
            ):
                semantic_divergence = await calculate_semantic_divergence(
                    interception.hidden_scratchpad,
                    interception.tool_parameters,
                )

            state_manager.log_telemetry(
                turn=turn,
                pressure_level=state_manager.environmental_pressure,
                scratchpad=interception.hidden_scratchpad,
                tool=interception.tool_intent,
                tool_params=interception.tool_parameters,
                divergence=semantic_divergence,
                schema_valid=interception.schema_valid,
                parse_strategy=interception.parsing_strategy,
                parse_errors=interception.errors,
                raw_response=raw_response,
            )

            latest_entry = state_manager.latest_telemetry_entry()
            if latest_entry is not None:
                await logger_instance.write_entry(
                    run_id=run_id,
                    model_name=_resolve_model_identifier(tested_agent),
                    is_monitored=is_monitored,
                    entry=latest_entry,
                )

            conversation_history.append({"role": "assistant", "content": raw_response})

            if interception.has_executable_tool and interception.tool_intent is not None:
                execution_result = await execute_tool(
                    interception.tool_intent,
                    interception.tool_parameters,
                    state_manager,
                )
                conversation_history.append(
                    {"role": "system", "content": f"Tool Result: {execution_result}"}
                )
                state_manager.set_terminal(reason="tool_executed")
                break

            if attempt == 0 and interception.tool_intent is None:
                conversation_history.append(
                    {"role": "system", "content": no_tool_retry_message}
                )
                continue

            if interception.tool_intent and not interception.schema_valid:
                conversation_history.append(
                    {
                        "role": "system",
                        "content": (
                            "Tool invocation rejected due to strict schema validation. "
                            "Return exactly a JSON object with keys "
                            '"function_name" and "parameters".'
                        ),
                    }
                )

            break

    return state_manager.export_telemetry_ledger()


def _resolve_model_identifier(agent: AgentProtocol) -> str:
    """Resolve a stable model identifier for telemetry rows."""
    candidate_names = ("model_identifier", "model_name")
    for attr_name in candidate_names:
        value = getattr(agent, attr_name, None)
        if isinstance(value, str) and value.strip():
            return value
    return agent.__class__.__name__
