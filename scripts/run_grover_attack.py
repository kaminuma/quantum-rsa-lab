#!/usr/bin/env python3
"""
Grover's Algorithm: Toy Cipher Key Search Attack

This script demonstrates Grover's algorithm attacking symmetric ciphers,
complementing the Shor's algorithm experiments for asymmetric cryptography.

Grover provides quadratic speedup for unstructured search:
- Classical: O(N) queries to find key in space of N keys
- Quantum: O(sqrt(N)) queries

For real ciphers:
- AES-128: 2^128 -> 2^64 (still secure, just use AES-256)
- AES-256: 2^256 -> 2^128 (effectively AES-128 strength)

This demo uses toy 4-bit ciphers that can run on NISQ devices.

Usage:
    python scripts/run_grover_attack.py
    python scripts/run_grover_attack.py --key-bits 4 --shots 1000
    python scripts/run_grover_attack.py --braket  # Run on AWS Braket simulator
"""

from __future__ import annotations
import argparse
import sys
import random
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def demo_grover_basics():
    """Demonstrate basic Grover search."""
    from quantum_crypto.grover import GroverSearch, create_marking_oracle, optimal_iterations

    print("=" * 70)
    print("Grover's Algorithm: Basic Search Demo")
    print("=" * 70)

    for n_qubits in [3, 4, 5]:
        target = random.randint(0, 2**n_qubits - 1)
        print(f"\n[{n_qubits}-qubit search]")
        print(f"  Search space: 2^{n_qubits} = {2**n_qubits}")
        print(f"  Target state: |{target:0{n_qubits}b}> = {target}")
        print(f"  Optimal iterations: {optimal_iterations(n_qubits, 1)}")

        # Create oracle and run
        oracle = create_marking_oracle(n_qubits, [target])
        grover = GroverSearch(n_qubits, oracle, n_solutions=1)

        result = grover.run_simulation(shots=1000)

        print(f"  Found state: |{result.measured_state:0{n_qubits}b}> = {result.measured_state}")
        print(f"  Success prob: {result.success_probability:.1%}")
        print(f"  Circuit depth: {result.circuit_depth}")


def demo_cipher_attack(
    key_bits: int = 4,
    use_sbox: bool = False,
    shots: int = 1000,
    verbose: bool = True
):
    """Demonstrate Grover attack on toy cipher."""
    from quantum_crypto.grover import GroverCipherAttack, toy_encrypt

    if verbose:
        print("\n" + "=" * 70)
        print(f"Grover's Algorithm: {key_bits}-bit Cipher Attack")
        print("=" * 70)

    # Generate random key and plaintext
    key = random.randint(0, 2**key_bits - 1)
    plaintext = random.randint(0, 2**key_bits - 1)
    ciphertext = toy_encrypt(plaintext, key, use_sbox)

    if verbose:
        print(f"\nSetup:")
        print(f"  Key bits: {key_bits}")
        print(f"  Search space: 2^{key_bits} = {2**key_bits}")
        print(f"  Secret key: {key} (0b{key:0{key_bits}b})")
        print(f"  Plaintext: {plaintext} (0b{plaintext:0{key_bits}b})")
        print(f"  Ciphertext: {ciphertext} (0b{ciphertext:0{key_bits}b})")
        print(f"  S-box: {'Yes' if use_sbox else 'No (XOR only)'}")

    # Run attack
    attack = GroverCipherAttack(
        plaintext=plaintext,
        ciphertext=ciphertext,
        key_bits=key_bits,
        use_sbox=use_sbox
    )

    if verbose:
        print(f"\nRunning Grover attack...")

    result = attack.run_simulation(shots=shots)

    if verbose:
        print(f"\nResults:")
        print(f"  Grover iterations: {result.iterations}")
        print(f"  Circuit depth: {result.circuit_depth}")
        print(f"  Found key: {result.found_key} (0b{result.found_key:0{key_bits}b})")
        print(f"  Correct key: {result.correct_key}")
        print(f"  Attack success: {result.success}")

        # Show measurement distribution
        print(f"\nTop measurement outcomes:")
        sorted_counts = sorted(
            result.counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for state, count in sorted_counts[:5]:
            key_val = int(state[::-1], 2)
            pct = 100 * count / shots
            print(f"  Key {key_val:2d} (0b{key_val:0{key_bits}b}): {count:4d} ({pct:.1f}%)")

    return result


def compare_classical_vs_quantum(key_bits: int = 4, trials: int = 10):
    """Compare classical and quantum search complexity."""
    from quantum_crypto.grover import optimal_iterations

    print("\n" + "=" * 70)
    print("Classical vs Quantum Search Complexity")
    print("=" * 70)

    print(f"\n{'Key Bits':<10} {'Space':<12} {'Classical':<15} {'Quantum':<15} {'Speedup':<10}")
    print("-" * 62)

    for bits in range(2, 9):
        space = 2 ** bits
        classical = space  # O(N)
        quantum = optimal_iterations(bits, 1)  # O(sqrt(N))
        speedup = classical / quantum if quantum > 0 else float('inf')

        print(f"{bits:<10} {space:<12} {classical:<15} {quantum:<15} {speedup:.1f}x")

    print(f"\nNote: Quantum advantage grows with sqrt(N).")
    print(f"For AES-128: 2^128 classical queries -> 2^64 quantum queries")


def run_multiple_attacks(
    key_bits: int = 4,
    n_trials: int = 10,
    shots: int = 1000
):
    """Run multiple attack trials and collect statistics."""
    print("\n" + "=" * 70)
    print(f"Statistical Analysis: {n_trials} Attacks on {key_bits}-bit Cipher")
    print("=" * 70)

    successes = 0
    depths = []
    iterations = []

    for i in range(n_trials):
        result = demo_cipher_attack(
            key_bits=key_bits,
            shots=shots,
            verbose=False
        )

        if result.success:
            successes += 1
        depths.append(result.circuit_depth)
        iterations.append(result.iterations)

        print(f"  Trial {i+1}: {'Success' if result.success else 'Failed'} "
              f"(found {result.found_key}, correct {result.correct_key})")

    success_rate = successes / n_trials * 100
    avg_depth = sum(depths) / len(depths)

    print(f"\nSummary:")
    print(f"  Success rate: {successes}/{n_trials} ({success_rate:.1f}%)")
    print(f"  Avg circuit depth: {avg_depth:.1f}")
    print(f"  Grover iterations: {iterations[0]}")


def save_experiment_results(results: dict, output_dir: Path):
    """Save experiment results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = output_dir / f"grover-attack-{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def print_cryptographic_context():
    """Print context about Grover and symmetric cryptography."""
    print("\n" + "=" * 70)
    print("Grover's Algorithm and Symmetric Cryptography")
    print("=" * 70)
    print("""
GROVER'S ALGORITHM:
  - Provides quadratic speedup for unstructured search
  - Finding key in space of N keys: O(N) -> O(sqrt(N))
  - Unlike Shor, does NOT break symmetric crypto - just weakens it

IMPACT ON SYMMETRIC CIPHERS:
  - AES-128: Effective security 64 bits (still impractical to break)
  - AES-256: Effective security 128 bits (recommended for PQ)
  - Mitigation: Simply double key sizes

COMPARISON WITH SHOR:
  - Shor: Exponential speedup, BREAKS RSA/ECDSA completely
  - Grover: Quadratic speedup, WEAKENS symmetric crypto by half

CURRENT NISQ LIMITATIONS:
  - Toy cipher (4-bit): Feasible with ~10 qubits
  - AES-128: Would need thousands of qubits + error correction
  - Real threat: 10-20+ years away

THIS DEMO:
  - Uses 4-bit toy cipher (XOR-based)
  - Demonstrates the attack concept
  - Can run on simulators or small quantum hardware
""")


def main():
    parser = argparse.ArgumentParser(
        description="Grover's Algorithm Toy Cipher Attack Demo"
    )
    parser.add_argument(
        "--key-bits", type=int, default=4,
        help="Number of key bits (default: 4)"
    )
    parser.add_argument(
        "--shots", type=int, default=1000,
        help="Number of measurement shots (default: 1000)"
    )
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Number of attack trials (default: 1)"
    )
    parser.add_argument(
        "--sbox", action="store_true",
        help="Use S-box in cipher (more complex)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Show classical vs quantum comparison"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Run multiple trials for statistics"
    )
    parser.add_argument(
        "--context", action="store_true",
        help="Print cryptographic context"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save results to logs/"
    )

    args = parser.parse_args()

    if args.context:
        print_cryptographic_context()
        return

    if args.compare:
        compare_classical_vs_quantum(args.key_bits)
        return

    if args.stats:
        run_multiple_attacks(
            key_bits=args.key_bits,
            n_trials=10,
            shots=args.shots
        )
        return

    # Default: run demos
    demo_grover_basics()

    if args.trials == 1:
        result = demo_cipher_attack(
            key_bits=args.key_bits,
            use_sbox=args.sbox,
            shots=args.shots
        )
    else:
        run_multiple_attacks(
            key_bits=args.key_bits,
            n_trials=args.trials,
            shots=args.shots
        )

    compare_classical_vs_quantum()
    print_cryptographic_context()

    if args.save:
        # Save results
        results = {
            "key_bits": args.key_bits,
            "shots": args.shots,
            "timestamp": datetime.now().isoformat(),
        }
        save_experiment_results(results, project_root / "logs")


if __name__ == "__main__":
    main()
