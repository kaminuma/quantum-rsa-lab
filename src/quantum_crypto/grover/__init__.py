"""Grover's algorithm implementation for quantum cryptanalysis.

This module provides Grover's algorithm implementations for:
- Generic unstructured search
- Symmetric cipher key search (toy ciphers)
- Educational demonstrations

Grover's algorithm provides quadratic speedup for unstructured search:
- Classical: O(N) queries
- Quantum: O(sqrt(N)) queries

For cryptographic applications:
- AES-128: Effective security reduced to 64 bits (still secure)
- AES-256: Effective security reduced to 128 bits (still secure)
- Toy ciphers (4-bit key): 16 -> 4 queries (demonstrable on NISQ)

Example usage:
    >>> from quantum_crypto.grover import GroverSearch, create_marking_oracle
    >>> # Search for |101> in 3-qubit space
    >>> oracle = create_marking_oracle(3, [5])  # 5 = 0b101
    >>> grover = GroverSearch(n_qubits=3, oracle=oracle)
    >>> circuit = grover.build_circuit()
"""

__all__ = [
    # Core algorithm
    "GroverSearch",
    "GroverResult",
    # Oracle construction
    "create_marking_oracle",
    "create_phase_oracle",
    # Toy cipher attack
    "ToyCipherOracle",
    "GroverCipherAttack",
    "toy_encrypt",
    "toy_decrypt",
    # Utilities
    "optimal_iterations",
]

from .core import GroverSearch, GroverResult, optimal_iterations
from .oracles import create_marking_oracle, create_phase_oracle
from .toy_cipher import ToyCipherOracle, GroverCipherAttack, toy_encrypt, toy_decrypt
