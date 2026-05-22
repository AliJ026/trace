"""Abstract scenario contracts for adversarial ForensicLens environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple


class BaseScenario(ABC):
    """Base contract for any scenario that drives deterministic turn escalation."""

    @abstractmethod
    def get_system_prompt(self, is_monitored: bool) -> str:
        """Return the system prompt injected at the start of each run."""
        raise NotImplementedError

    @abstractmethod
    def generate_escalation(self, turn: int) -> Tuple[float, str]:
        """
        Return `(pressure_scalar, user_prompt)` for the provided turn index.
        Pressure scalar should be normalized to [0.0, 1.0].
        """
        raise NotImplementedError
