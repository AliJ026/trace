"""Asynchronous JSONL telemetry logger."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Mapping

try:
    import aiofiles
except Exception:  # pragma: no cover - dependency optional fallback
    aiofiles = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


class TelemetryLogger:
    """Non-blocking telemetry sink that appends normalized JSONL records."""

    def __init__(self, log_dir: str = "forensiclens/logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.current_log_file = self.log_dir / f"telemetry_ledger_{timestamp}.jsonl"
        self._write_lock = asyncio.Lock()

    async def write_entry(
        self,
        *,
        run_id: str,
        model_name: str,
        is_monitored: bool,
        entry: Mapping[str, Any],
    ) -> None:
        """
        Append one normalized telemetry row.

        Required canonical fields are always emitted even when missing in input.
        """
        payload = self._build_payload(
            run_id=run_id,
            model_name=model_name,
            is_monitored=is_monitored,
            entry=entry,
        )
        serialized = json.dumps(payload, ensure_ascii=True) + "\n"

        try:
            async with self._write_lock:
                if aiofiles is not None:
                    async with aiofiles.open(
                        self.current_log_file,
                        mode="a",
                        encoding="utf-8",
                    ) as handle:
                        await handle.write(serialized)
                else:
                    await asyncio.to_thread(
                        self._sync_append,
                        serialized,
                    )
        except Exception as exc:  # pragma: no cover - filesystem variability
            LOGGER.exception("Failed to write telemetry entry: %s", exc)
            raise

    def _build_payload(
        self,
        *,
        run_id: str,
        model_name: str,
        is_monitored: bool,
        entry: Mapping[str, Any],
    ) -> dict[str, Any]:
        timestamp = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")

        def _value(key: str, default: Any = "") -> Any:
            return entry.get(key, default)

        return {
            "timestamp": timestamp,
            "run_id": run_id,
            "model_name": model_name,
            "environment_state": "monitored" if is_monitored else "unmonitored",
            "turn_index": _value("turn_index", -1),
            "pressure_level": _value("pressure_level", 0.0),
            "hidden_thought": _value("hidden_thought", ""),
            "tool_intent": _value("tool_intent", "None"),
            "tool_parameters": _value("tool_parameters", {}),
            "semantic_divergence": _value("semantic_divergence", 0.0),
            # Preserve helpful parser diagnostics when present.
            "schema_valid": _value("schema_valid", False),
            "parse_strategy": _value("parse_strategy", ""),
            "parse_errors": _value("parse_errors", []),
            "raw_response": _value("raw_response", ""),
        }

    @property
    def current_log_path(self) -> str:
        """Expose current log filename for CLI/status reporting."""
        return str(self.current_log_file)

    def _sync_append(self, serialized: str) -> None:
        with self.current_log_file.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
