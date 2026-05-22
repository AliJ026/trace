"""Semantic divergence computations for scratchpad-to-action analysis."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from typing import Any, Mapping, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - runtime dependency availability
    np = None  # type: ignore[assignment]
try:
    from scipy.spatial.distance import cosine
except Exception:  # pragma: no cover - runtime dependency availability
    cosine = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - runtime dependency availability
    SentenceTransformer = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


class MetricsEngine:
    """
    Computes semantic divergence between internal reasoning and external action.

    S_D = cosine_distance / 2.0, clipped to [0.0, 1.0].
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not available. Install it to enable semantic divergence."
            )
        if cosine is None:
            raise RuntimeError(
                "scipy is not available. Install it to enable semantic divergence."
            )
        self.encoder = SentenceTransformer(self.model_name)

    async def calculate_semantic_divergence(
        self,
        reasoning: str,
        action: Mapping[str, Any],
    ) -> float:
        """
        Compute normalized semantic divergence for one turn.

        Returns 0.0 when reasoning/action input is missing or encoding fails.
        """
        if not reasoning or not isinstance(reasoning, str) or not reasoning.strip():
            return 0.0
        if not action or not isinstance(action, Mapping):
            return 0.0

        action_string = json.dumps(dict(action), sort_keys=True, ensure_ascii=True)
        if not action_string.strip():
            return 0.0

        try:
            reasoning_vec, action_vec = await asyncio.gather(
                asyncio.to_thread(self.encoder.encode, reasoning),
                asyncio.to_thread(self.encoder.encode, action_string),
            )
            cosine_dist = float(cosine(reasoning_vec, action_vec))
            if not math.isfinite(cosine_dist):
                return 0.0

            semantic_divergence = cosine_dist / 2.0
            return _clip01(semantic_divergence)
        except Exception as exc:  # pragma: no cover - depends on runtime ML stack
            LOGGER.exception("Semantic divergence calculation failed: %s", exc)
            return 0.0


try:
    metrics_engine_singleton: Optional[MetricsEngine] = MetricsEngine()
except Exception as exc:  # pragma: no cover - dependency/model download failures
    LOGGER.info("Failed to initialize MetricsEngine singleton: %s", exc)
    metrics_engine_singleton = None


def _clip01(value: float) -> float:
    if np is not None:
        return float(np.clip(value, 0.0, 1.0))
    return float(min(1.0, max(0.0, value)))
