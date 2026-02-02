#!/usr/bin/env python3
"""
Post-Quantum Cryptography Demo: Shor's Algorithm vs PQC Defense

This script demonstrates both sides of the quantum cryptography story:
1. ATTACK: Shor's algorithm threatens RSA/ECDSA
2. DEFENSE: PQC algorithms (ML-KEM, ML-DSA) provide quantum-safe alternatives

Usage:
    python scripts/run_pqc_demo.py
    python scripts/run_pqc_demo.py --benchmark  # Run performance benchmark
    python scripts/run_pqc_demo.py --hybrid     # Demo hybrid cryptography
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []

    try:
        import oqs
    except ImportError:
        missing.append("liboqs-python")

    try:
        from cryptography.hazmat.primitives.asymmetric import x25519
    except ImportError:
        missing.append("cryptography")

    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print("  pip install liboqs-python cryptography")
        return False
    return True


def print_security_comparison():
    """Print security comparison table."""
    print("=" * 70)
    print("Security Comparison: Classical vs Post-Quantum")
    print("=" * 70)

    comparisons = [
        ("RSA-2048", 2048, 112, 0, True, "Broken by ~4000 logical qubits"),
        ("RSA-3072", 3072, 128, 0, True, "Broken by ~6000 logical qubits"),
        ("ECDSA-P256", 256, 128, 0, True, "Broken by ~2500 logical qubits"),
        ("ML-KEM-768", 1184, 192, 128, False, "NIST FIPS 203, lattice-based KEM"),
        ("ML-DSA-65", 1952, 192, 128, False, "NIST FIPS 204, lattice-based signature"),
    ]

    print(f"{'Algorithm':<15} {'Classical':<12} {'Quantum':<10} {'Shor-Safe':<10}")
    print("-" * 60)

    for alg, key_size, classical, quantum, vulnerable, notes in comparisons:
        shor_safe = "Yes" if not vulnerable else "NO"
        q_bits = str(quantum) if quantum > 0 else "BROKEN"
        print(f"{alg:<15} {classical:<12} {q_bits:<10} {shor_safe:<10}")

    print("\nNotes:")
    for alg, key_size, classical, quantum, vulnerable, notes in comparisons:
        print(f"  {alg}: {notes}")


def demo_ml_kem():
    """Demonstrate ML-KEM key encapsulation."""
    import oqs
    import time

    print("\n" + "=" * 70)
    print("ML-KEM Key Encapsulation (NIST FIPS 203)")
    print("=" * 70)

    for level in [512, 768, 1024]:
        alg = f"ML-KEM-{level}"
        print(f"\n[{alg}]")

        start = time.perf_counter()

        with oqs.KeyEncapsulation(alg) as alice:
            alice_public = alice.generate_keypair()

            with oqs.KeyEncapsulation(alg) as bob:
                ciphertext, bob_secret = bob.encap_secret(alice_public)

            alice_secret = alice.decap_secret(ciphertext)

        elapsed = (time.perf_counter() - start) * 1000

        print(f"  Public key size:  {len(alice_public):,} bytes")
        print(f"  Ciphertext size:  {len(ciphertext):,} bytes")
        print(f"  Shared secret:    {alice_secret.hex()[:32]}...")
        print(f"  Key exchange:     {elapsed:.2f} ms")
        print(f"  Secrets match:    {alice_secret == bob_secret}")


def demo_ml_dsa():
    """Demonstrate ML-DSA digital signatures."""
    import oqs
    import time

    print("\n" + "=" * 70)
    print("ML-DSA Digital Signatures (NIST FIPS 204)")
    print("=" * 70)

    message = b"This document is signed with post-quantum cryptography!"

    for level in [44, 65, 87]:
        alg = f"ML-DSA-{level}"
        print(f"\n[{alg}]")

        start = time.perf_counter()

        with oqs.Signature(alg) as signer:
            public_key = signer.generate_keypair()
            signature = signer.sign(message)

            with oqs.Signature(alg) as verifier:
                is_valid = verifier.verify(message, signature, public_key)

        elapsed = (time.perf_counter() - start) * 1000

        print(f"  Public key size:  {len(public_key):,} bytes")
        print(f"  Signature size:   {len(signature):,} bytes")
        print(f"  Signature valid:  {is_valid}")
        print(f"  Sign + verify:    {elapsed:.2f} ms")


def demo_hybrid_kem():
    """Demonstrate hybrid key encapsulation."""
    print("\n" + "=" * 70)
    print("Hybrid KEM: X25519 + ML-KEM-768")
    print("=" * 70)

    from quantum_crypto.pqc import HybridKEM

    hybrid = HybridKEM(ml_kem_level=768)

    # Alice generates keypair
    alice_keys = hybrid.generate_keypair()
    print(f"\nAlice's keys:")
    print(f"  X25519 public:   {alice_keys.x25519_public.hex()[:32]}...")
    print(f"  ML-KEM public:   {len(alice_keys.ml_kem_public):,} bytes")

    # Bob encapsulates
    encap = hybrid.encapsulate(alice_keys)
    print(f"\nBob's encapsulation:")
    print(f"  X25519 ephemeral: {encap.x25519_public.hex()[:32]}...")
    print(f"  ML-KEM ciphertext: {len(encap.ml_kem_ciphertext):,} bytes")
    print(f"  Bob's secret:     {encap.combined_secret.hex()}")

    # Alice decapsulates
    alice_secret = hybrid.decapsulate(alice_keys, encap)
    print(f"\nAlice's secret:     {alice_secret.hex()}")
    print(f"Secrets match:      {alice_secret == encap.combined_secret}")


def demo_hybrid_signature():
    """Demonstrate hybrid signatures."""
    print("\n" + "=" * 70)
    print("Hybrid Signature: ECDSA-P256 + ML-DSA-65")
    print("=" * 70)

    from quantum_crypto.pqc import HybridSignature

    hybrid = HybridSignature(ml_dsa_level=65)

    # Generate keypair
    keys = hybrid.generate_keypair()
    print(f"\nKeypair generated:")
    print(f"  ECDSA public key: (P-256 curve)")
    print(f"  ML-DSA public key: {len(keys['ml_dsa_public']):,} bytes")

    # Sign message
    message = b"Critical document requiring maximum security"
    signatures = hybrid.sign(keys, message)
    print(f"\nSignatures:")
    print(f"  ECDSA signature:  {len(signatures['ecdsa_signature'])} bytes")
    print(f"  ML-DSA signature: {len(signatures['ml_dsa_signature']):,} bytes")

    # Verify
    is_valid = hybrid.verify(keys, message, signatures)
    print(f"\nBoth signatures valid: {is_valid}")

    # Test with tampered message
    tampered = b"Tampered document"
    is_tampered_valid = hybrid.verify(keys, tampered, signatures)
    print(f"Tampered verification: {is_tampered_valid}")


def run_benchmark():
    """Run PQC vs classical benchmark."""
    print("\n" + "=" * 70)
    print("Performance Benchmark: PQC vs Classical")
    print("=" * 70)

    from quantum_crypto.pqc import PQCBenchmark

    bench = PQCBenchmark(iterations=50, warmup=5)
    bench.run_full_benchmark(verbose=True)
    bench.print_summary()

    # Save results
    output_dir = project_root / "logs"
    output_dir.mkdir(exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = output_dir / f"pqc-benchmark-{timestamp}.json"

    bench.save_results(str(output_file))


def print_connection_to_shor():
    """Print connection to Shor's algorithm research."""
    print("\n" + "=" * 70)
    print("Connection to Shor's Algorithm Research")
    print("=" * 70)
    print("""
This repository demonstrates both sides of the quantum cryptography story:

ATTACK (Shor's Algorithm):
  - Factoring N=15, 21, 33, 35 using quantum period finding
  - Demonstrates theoretical vulnerability of RSA
  - Current quantum computers: ~1000 noisy qubits
  - Required for RSA-2048: ~4000 logical (error-corrected) qubits

DEFENSE (Post-Quantum Cryptography):
  - ML-KEM: Lattice-based key encapsulation (replaces RSA/ECDH)
  - ML-DSA: Lattice-based signatures (replaces RSA/ECDSA)
  - Based on hard problems believed quantum-resistant
  - NIST standardized in 2024 (FIPS 203, 204)

TIMELINE:
  - 2024: NIST PQC standards published
  - 2025-2030: Migration period ("crypto-agility")
  - 2030+: Expected quantum computers threatening RSA-2048

RECOMMENDATION:
  Use hybrid cryptography (classical + PQC) during transition period
  to protect against both classical and potential quantum attacks.
""")


def main():
    parser = argparse.ArgumentParser(
        description="Post-Quantum Cryptography Demo"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run performance benchmark"
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Demo hybrid cryptography only"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick demo (skip benchmark)"
    )

    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    if args.benchmark:
        run_benchmark()
        return

    if args.hybrid:
        demo_hybrid_kem()
        demo_hybrid_signature()
        return

    # Full demo
    print_security_comparison()
    demo_ml_kem()
    demo_ml_dsa()

    if not args.quick:
        demo_hybrid_kem()
        demo_hybrid_signature()

    print_connection_to_shor()

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
