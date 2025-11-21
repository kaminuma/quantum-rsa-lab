"""Runner module for Shor's algorithm."""

from __future__ import annotations

from typing import Optional, Any

from .core import ShorResult
from .algorithms import ClassicalShor, QuantumShor

DEFAULT_N_COUNT = 8
DEFAULT_SHOTS = 1024


def run_shor(
    number: int = 15,
    base: Optional[int] = None,
    shots: int = DEFAULT_SHOTS,
    method: str = "quantum",
    n_count: int = DEFAULT_N_COUNT,
    **kwargs: Any,
) -> ShorResult:
    """Run Shor's algorithm.
    
    Args:
        number: The integer to factorize.
        base: The base 'a'.
        shots: Number of shots.
        method: "quantum" or "classical".
        n_count: Number of counting qubits (quantum only).
        **kwargs: Additional arguments.
        
    Returns:
        ShorResult object.
    """
    if method == "classical":
        algo = ClassicalShor()
        return algo.run(number, base, shots=shots, **kwargs)
    elif method == "quantum":
        algo = QuantumShor(n_count=n_count)
        return algo.run(number, base, shots=shots, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
