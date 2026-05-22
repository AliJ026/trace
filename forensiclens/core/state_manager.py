"""Execution-state and telemetry ledger primitives for deterministic runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class TelemetryEntry:
    """Immutable snapshot of one execution turn."""

    turn_index: int
    pressure_level: float
    hidden_thought: str
    tool_intent: str
    tool_parameters: Dict[str, Any]
    semantic_divergence: float
    schema_valid: bool
    parse_strategy: str
    parse_errors: List[str]
    raw_response: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep-copied dictionary for safe downstream serialization."""
        return {
            "turn_index": self.turn_index,
            "pressure_level": self.pressure_level,
            "hidden_thought": self.hidden_thought,
            "tool_intent": self.tool_intent,
            "tool_parameters": deepcopy(self.tool_parameters),
            "semantic_divergence": self.semantic_divergence,
            "schema_valid": self.schema_valid,
            "parse_strategy": self.parse_strategy,
            "parse_errors": list(self.parse_errors),
            "raw_response": self.raw_response,
        }


class StateManager:
    """
    Immutable-feeling state ledger for deterministic scenario execution.

    The manager owns:
    - Temporal progression (turn counters)
    - Environmental pressure scalars
    - Terminal-state signaling
    - Structured telemetry snapshots
    """

    def __init__(self, max_turns: int = 3) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than 0.")

        self.current_turn: int = 0
        self.max_turns: int = max_turns
        self.environmental_pressure: float = 0.0
        self.is_terminal: bool = False
        self.terminal_reason: Optional[str] = None
        self._telemetry_ledger: List[TelemetryEntry] = []

    def advance_turn(self, pressure_scalar: float) -> None:
        """Advance turn index and update pressure; guard against post-terminal mutation."""
        if self.is_terminal:
            raise RuntimeError(
                "Cannot advance turn after terminal state is reached."
            )

        self.current_turn += 1
        self.environmental_pressure = float(pressure_scalar)
        if self.current_turn >= self.max_turns:
            self.is_terminal = True
            self.terminal_reason = self.terminal_reason or "max_turns_reached"

    def set_terminal(self, reason: str = "terminal_condition_met") -> None:
        """Hard-stop the scenario state machine."""
        self.is_terminal = True
        self.terminal_reason = reason

    def is_terminal_state_reached(self) -> bool:
        """Expose terminal-state status to the execution loop."""
        return self.is_terminal

    def log_telemetry(
        self,
        *,
        turn: int,
        pressure_level: float,
        scratchpad: str,
        tool: Optional[str],
        tool_params: Optional[Dict[str, Any]],
        divergence: float,
        schema_valid: bool,
        parse_strategy: str,
        parse_errors: Optional[Sequence[str]],
        raw_response: str,
    ) -> None:
        """Append a structured, immutable telemetry entry."""
        entry = TelemetryEntry(
            turn_index=int(turn),
            pressure_level=float(pressure_level),
            hidden_thought=(scratchpad or "").strip(),
            tool_intent=tool or "None",
            tool_parameters=deepcopy(tool_params or {}),
            semantic_divergence=float(divergence),
            schema_valid=bool(schema_valid),
            parse_strategy=parse_strategy.strip() or "unknown",
            parse_errors=list(parse_errors or []),
            raw_response=raw_response,
        )
        self._telemetry_ledger.append(entry)

    def latest_telemetry_entry(self) -> Optional[Dict[str, Any]]:
        """Return latest telemetry row as a copy-safe dictionary."""
        if not self._telemetry_ledger:
            return None
        return self._telemetry_ledger[-1].to_dict()

    def export_telemetry_ledger(self) -> List[Dict[str, Any]]:
        """Export full ledger as detached dictionaries to prevent external mutation."""
        return [entry.to_dict() for entry in self._telemetry_ledger]

