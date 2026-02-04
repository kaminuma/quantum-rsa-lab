"""Quantum Crypto Lab - Quantum cryptography research toolkit.

This package provides tools for:
- Shor's algorithm experiments (factoring RSA)
- Grover's algorithm experiments (symmetric key search)
- Post-Quantum Cryptography (ML-KEM, ML-DSA)

Subpackages:
- quantum_crypto.algorithms: Shor's algorithm implementations
- quantum_crypto.grover: Grover's algorithm for symmetric cipher attacks
- quantum_crypto.pqc: Post-quantum cryptography (NIST standards)
- quantum_crypto.backends: Quantum hardware backends
- quantum_crypto.modexp: Modular exponentiation circuits
"""

__all__ = [
    # Shor's algorithm
    "run_shor",
    "QuantumRunSetting",
    "sweep_shot_counts",
    "summarize_success",
]

from .runner import run_shor
from .experiment_logging import QuantumRunSetting, summarize_success, sweep_shot_counts
