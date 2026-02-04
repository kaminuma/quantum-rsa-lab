"""Core abstractions for Shor's algorithm implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class ShorResult:
    """Result of Shor's algorithm execution."""
    number: int
    base: int
    factors: tuple[int, int] | None
    shots: int
    success: bool
    method: str
    measured_phase: float | None = None
    period: int | None = None
    counts: dict[str, int] | None = None
    mitigated_counts: dict[str, float] | None = None
    circuit_depth: int | None = None
    total_gates: int | None = None
    two_qubit_gates: int | None = None
    gate_counts: dict[str, int] | None = None
    noise_model_name: str | None = None


class ShorAlgorithm(ABC):
    """Abstract base class for Shor's algorithm implementations."""

    @abstractmethod
    def run(self, number: int, base: int, shots: int = 1024, **kwargs: Any) -> ShorResult:
        """Run the algorithm to factorize the given number.

        Args:
            number: The integer to factorize.
            base: The base 'a' for modular exponentiation.
            shots: Number of shots for quantum measurement (or simulation).
            **kwargs: Additional implementation-specific arguments.

        Returns:
            ShorResult containing the factorization results and metrics.
        """
        pass
