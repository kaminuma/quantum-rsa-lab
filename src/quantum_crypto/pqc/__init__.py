"""Post-Quantum Cryptography (PQC) module.

This module provides implementations of NIST-standardized post-quantum
cryptographic algorithms using liboqs-python:

- ML-KEM (FIPS 203): Module-Lattice-Based Key-Encapsulation Mechanism
- ML-DSA (FIPS 204): Module-Lattice-Based Digital Signature Algorithm
- Hybrid schemes: X25519 + ML-KEM, ECDSA + ML-DSA

These algorithms are designed to be secure against attacks by both
classical and quantum computers, complementing the Shor's algorithm
experiments in this repository.

Example usage:
    >>> from quantum_crypto.pqc import ml_kem_keygen, ml_kem_encapsulate, ml_kem_decapsulate
    >>> pub, sec = ml_kem_keygen(768)
    >>> ct, shared_bob = ml_kem_encapsulate(pub, 768)
    >>> shared_alice = ml_kem_decapsulate(sec, ct, 768)
    >>> assert shared_alice == shared_bob

    >>> from quantum_crypto.pqc import PQCBenchmark
    >>> bench = PQCBenchmark(iterations=100)
    >>> results = bench.run_full_benchmark()
"""

__all__ = [
    # Core KEM functions
    "ml_kem_keygen",
    "ml_kem_encapsulate",
    "ml_kem_decapsulate",
    "ml_kem_full_exchange",
    # Core signature functions
    "ml_dsa_keygen",
    "ml_dsa_sign",
    "ml_dsa_verify",
    "ml_dsa_full_demo",
    # Hybrid cryptography
    "HybridKEM",
    "HybridSignature",
    # Benchmarking
    "PQCBenchmark",
    # Utilities
    "list_available_kems",
    "list_available_sigs",
    "check_pqc_available",
]

from .kem import (
    ml_kem_keygen,
    ml_kem_encapsulate,
    ml_kem_decapsulate,
    ml_kem_full_exchange,
    list_available_kems,
)
from .signatures import (
    ml_dsa_keygen,
    ml_dsa_sign,
    ml_dsa_verify,
    ml_dsa_full_demo,
    list_available_sigs,
)
from .hybrid import HybridKEM, HybridSignature
from .benchmark import PQCBenchmark
from .utils import check_pqc_available
