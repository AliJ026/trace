"""Hybrid interception middleware for hidden-thought and tool-call parsing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class InterceptionResult:
    """Parsed view of model output after hybrid extraction and schema checks."""

    hidden_scratchpad: str
    tool_intent: Optional[str]
    tool_parameters: Dict[str, Any]
    schema_valid: bool
    parsing_strategy: str
    errors: List[str]
    raw_payload: Optional[Dict[str, Any]]

    @property
    def has_executable_tool(self) -> bool:
        return bool(self.schema_valid and self.tool_intent)


HIDDEN_THOUGHT_PATTERN = re.compile(
    r"<\s*hidden_thought\s*>(.*?)<\s*/\s*hidden_thought\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
HIDDEN_THOUGHT_OPEN_PATTERN = re.compile(
    r"<\s*hidden_thought\s*>",
    flags=re.IGNORECASE,
)

JSON_CODEBLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)

TOOL_CALL_REQUIRED_KEYS = frozenset({"function_name", "parameters"})


def validate_tool_payload(
    payload: Any,
    *,
    allowed_tools: Optional[Iterable[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate tool invocation object against a strict schema:
    {
      "function_name": <non-empty string>,
      "parameters": <object>
    }
    No additional keys are permitted.
    """
    errors: List[str] = []

    if not isinstance(payload, dict):
        return False, ["Tool payload must be a JSON object."]

    payload_keys = set(payload.keys())
    missing_keys = TOOL_CALL_REQUIRED_KEYS - payload_keys
    extra_keys = payload_keys - TOOL_CALL_REQUIRED_KEYS

    if missing_keys:
        errors.append(
            f"Missing required key(s): {', '.join(sorted(missing_keys))}."
        )
    if extra_keys:
        errors.append(
            f"Unexpected key(s): {', '.join(sorted(extra_keys))}. "
            "Strict schema allows only function_name and parameters."
        )

    function_name = payload.get("function_name")
    parameters = payload.get("parameters")

    if not isinstance(function_name, str) or not function_name.strip():
        errors.append("function_name must be a non-empty string.")

    if not isinstance(parameters, dict):
        errors.append("parameters must be a JSON object.")
    else:
        try:
            json.dumps(parameters)
        except (TypeError, ValueError) as exc:
            errors.append(f"parameters must be JSON-serializable: {exc}")

    if allowed_tools and isinstance(function_name, str):
        allowed_set = set(allowed_tools)
        if function_name not in allowed_set:
            errors.append(
                f"function_name '{function_name}' is not registered in allowed tools."
            )

    return len(errors) == 0, errors


def parse_scratchpad_and_tool(
    raw_llm_output: str,
    *,
    allowed_tools: Optional[Iterable[str]] = None,
) -> InterceptionResult:
    """
    Hybrid parsing strategy:
    1) Aggressively extract <hidden_thought>...</hidden_thought>.
    2) Parse strict JSON tool call from fenced blocks or naked JSON.
    3) Fall back to relaxed field extraction for resilience diagnostics.

    Execution should still require strict schema validity.
    """
    text = raw_llm_output if isinstance(raw_llm_output, str) else str(raw_llm_output)
    scratchpad, residual_text = _extract_hidden_thought(text)

    candidates = _collect_json_candidates(residual_text)
    parse_errors: List[str] = []

    for candidate, strategy in candidates:
        payload, payload_error = _safe_json_loads(candidate)
        if payload_error:
            parse_errors.append(payload_error)
            continue

        valid, schema_errors = validate_tool_payload(payload, allowed_tools=allowed_tools)
        if valid:
            return InterceptionResult(
                hidden_scratchpad=scratchpad,
                tool_intent=payload["function_name"],
                tool_parameters=dict(payload["parameters"]),
                schema_valid=True,
                parsing_strategy=strategy,
                errors=[],
                raw_payload=payload,
            )

        parse_errors.extend(schema_errors)
        relaxed_name, relaxed_parameters = _relaxed_payload_projection(payload)
        if relaxed_name is not None:
            return InterceptionResult(
                hidden_scratchpad=scratchpad,
                tool_intent=relaxed_name,
                tool_parameters=relaxed_parameters,
                schema_valid=False,
                parsing_strategy=f"{strategy}:schema_rejected",
                errors=_dedupe_errors(parse_errors),
                raw_payload=payload if isinstance(payload, dict) else None,
            )

    fallback_name, fallback_params = _fallback_text_extraction(residual_text)
    if fallback_name is not None:
        parse_errors.append(
            "Recovered tool-like output via fallback text extraction; schema invalid."
        )
        return InterceptionResult(
            hidden_scratchpad=scratchpad,
            tool_intent=fallback_name,
            tool_parameters=fallback_params,
            schema_valid=False,
            parsing_strategy="fallback_text_extraction",
            errors=_dedupe_errors(parse_errors),
            raw_payload=None,
        )

    return InterceptionResult(
        hidden_scratchpad=scratchpad,
        tool_intent=None,
        tool_parameters={},
        schema_valid=False,
        parsing_strategy="no_tool_detected",
        errors=_dedupe_errors(parse_errors),
        raw_payload=None,
    )


def _extract_hidden_thought(text: str) -> Tuple[str, str]:
    matches = list(HIDDEN_THOUGHT_PATTERN.finditer(text))
    if matches:
        parts = [match.group(1).strip() for match in matches if match.group(1).strip()]
        scratchpad = "\n\n".join(parts).strip()

        residual_text = HIDDEN_THOUGHT_PATTERN.sub("", text).strip()
        return scratchpad, residual_text

    # Fallback for malformed outputs where `<hidden_thought>` is opened but never closed.
    open_match = HIDDEN_THOUGHT_OPEN_PATTERN.search(text)
    if open_match is None:
        return "", text

    content_start = open_match.end()
    cutoff = _find_unclosed_hidden_thought_cutoff(text, content_start)

    scratchpad = text[content_start:cutoff].strip()
    residual_text = (text[: open_match.start()] + text[cutoff:]).strip()
    return scratchpad, residual_text


def _find_unclosed_hidden_thought_cutoff(text: str, content_start: int) -> int:
    """
    Determine where unclosed hidden-thought content should stop.
    Priority:
    1) First markdown code fence (```), which usually introduces the JSON action.
    2) First likely JSON-object start when no code fence exists.
    3) End of text.
    """
    fence_index = text.find("```", content_start)
    if fence_index != -1:
        return fence_index

    trailing_text = text[content_start:]
    likely_json_start = re.search(
        r"\{\s*\"(?:function_name|tool_name|name|parameters|arguments)\"",
        trailing_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if likely_json_start is not None:
        return content_start + likely_json_start.start()

    return len(text)


def _collect_json_candidates(text: str) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    seen: set[str] = set()

    for match in JSON_CODEBLOCK_PATTERN.finditer(text):
        candidate = match.group(1).strip()
        if candidate and candidate not in seen:
            candidates.append((candidate, "fenced_json"))
            seen.add(candidate)

    for candidate in _extract_balanced_json_objects(text):
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            candidates.append((candidate, "balanced_json"))
            seen.add(candidate)

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}") and stripped not in seen:
        candidates.append((stripped, "whole_text_json"))

    return candidates


def _extract_balanced_json_objects(text: str) -> Sequence[str]:
    """Extract balanced JSON objects from arbitrary text using brace tracking."""
    objects: List[str] = []
    depth = 0
    start_idx: Optional[int] = None
    in_string = False
    escape = False

    for idx, char in enumerate(text):
        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start_idx is not None:
                objects.append(text[start_idx : idx + 1])
                start_idx = None

    return objects


def _safe_json_loads(candidate: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        payload = json.loads(candidate)
        return payload, None
    except json.JSONDecodeError as exc:
        return None, f"JSON parse failed: {exc}"


def _relaxed_payload_projection(payload: Any) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Best-effort extraction from near-miss payloads for diagnostics only.
    Result still counts as schema-invalid unless validated separately.
    """
    if not isinstance(payload, dict):
        return None, {}

    name = payload.get("function_name")
    if not isinstance(name, str) or not name.strip():
        alt_name = payload.get("tool_name") or payload.get("name")
        name = alt_name if isinstance(alt_name, str) else None
    if not isinstance(name, str) or not name.strip():
        return None, {}

    params = payload.get("parameters")
    if isinstance(params, dict):
        return name, params

    args = payload.get("arguments")
    if isinstance(args, dict):
        return name, args

    if isinstance(params, str):
        loaded, error = _safe_json_loads(params)
        if error is None and isinstance(loaded, dict):
            return name, loaded

    return name, {}


def _fallback_text_extraction(text: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Last-resort parser for malformed smaller-model outputs.
    Supports patterns like:
    - submit_true_cms_metrics({"psi90_score": 3.2})
    """
    call_pattern = re.compile(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(\{.*\})\s*\)",
        flags=re.DOTALL,
    )
    call_match = call_pattern.search(text)
    if not call_match:
        return None, {}

    tool_name = call_match.group(1)
    params_blob = call_match.group(2)
    payload, error = _safe_json_loads(params_blob)
    if error is None and isinstance(payload, dict):
        return tool_name, payload
    return tool_name, {}


def _dedupe_errors(errors: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for item in errors:
        normalized = item.strip()
        if normalized and normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
    return deduped
