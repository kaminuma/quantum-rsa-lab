"""Performance benchmarking: Post-Quantum vs Classical Cryptography.

Compares:
- Key Encapsulation: ML-KEM vs RSA, ECDH
- Digital Signatures: ML-DSA vs RSA, ECDSA
"""

from __future__ import annotations
import time
import statistics
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional
import json
from pathlib import Path

from .utils import require_oqs


@dataclass
class BenchmarkResult:
    """Result of a single benchmark operation."""
    algorithm: str
    operation: str
    iterations: int
    times_ms: List[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.times_ms) if self.times_ms else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "operation": self.operation,
            "iterations": self.iterations,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


def _benchmark_operation(func: Callable, iterations: int = 100) -> List[float]:
    """Benchmark a single operation."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    return times


class PQCBenchmark:
    """Benchmark suite for PQC vs classical algorithms.

    Example:
        >>> bench = PQCBenchmark(iterations=100, warmup=5)
        >>> bench.run_full_benchmark()
        >>> bench.print_summary()
        >>> bench.save_results("logs/pqc_benchmark.json")
    """

    def __init__(self, iterations: int = 100, warmup: int = 5):
        """Initialize benchmark suite.

        Args:
            iterations: Number of iterations per operation
            warmup: Number of warmup iterations (not counted)
        """
        require_oqs()
        self.iterations = iterations
        self.warmup = warmup
        self.results: List[BenchmarkResult] = []

    def _warmup_run(self, func: Callable) -> None:
        """Run warmup iterations."""
        for _ in range(self.warmup):
            func()

    # ==================== ML-KEM Benchmarks ====================

    def benchmark_ml_kem(self, security_level: int = 768) -> Dict[str, BenchmarkResult]:
        """Benchmark ML-KEM operations."""
        import oqs

        alg = f"ML-KEM-{security_level}"
        results = {}

        # Key generation
        def keygen():
            with oqs.KeyEncapsulation(alg) as kem:
                kem.generate_keypair()

        self._warmup_run(keygen)
        result = BenchmarkResult(alg, "keygen", self.iterations)
        result.times_ms = _benchmark_operation(keygen, self.iterations)
        results["keygen"] = result
        self.results.append(result)

        # Encapsulation
        with oqs.KeyEncapsulation(alg) as kem:
            public_key = kem.generate_keypair()

            def encap():
                with oqs.KeyEncapsulation(alg) as k:
                    k.encap_secret(public_key)

            self._warmup_run(encap)
            result = BenchmarkResult(alg, "encap", self.iterations)
            result.times_ms = _benchmark_operation(encap, self.iterations)
            results["encap"] = result
            self.results.append(result)

            # Decapsulation
            ciphertext, _ = kem.encap_secret(public_key)

            def decap():
                kem.decap_secret(ciphertext)

            self._warmup_run(decap)
            result = BenchmarkResult(alg, "decap", self.iterations)
            result.times_ms = _benchmark_operation(decap, self.iterations)
            results["decap"] = result
            self.results.append(result)

        return results

    # ==================== ML-DSA Benchmarks ====================

    def benchmark_ml_dsa(self, security_level: int = 65) -> Dict[str, BenchmarkResult]:
        """Benchmark ML-DSA operations."""
        import oqs

        alg = f"ML-DSA-{security_level}"
        message = b"Benchmark message for digital signatures"
        results = {}

        # Key generation
        def keygen():
            with oqs.Signature(alg) as sig:
                sig.generate_keypair()

        self._warmup_run(keygen)
        result = BenchmarkResult(alg, "keygen", self.iterations)
        result.times_ms = _benchmark_operation(keygen, self.iterations)
        results["keygen"] = result
        self.results.append(result)

        # Signing
        with oqs.Signature(alg) as signer:
            public_key = signer.generate_keypair()

            def sign():
                signer.sign(message)

            self._warmup_run(sign)
            result = BenchmarkResult(alg, "sign", self.iterations)
            result.times_ms = _benchmark_operation(sign, self.iterations)
            results["sign"] = result
            self.results.append(result)

            # Verification
            signature = signer.sign(message)

            def verify():
                with oqs.Signature(alg) as v:
                    v.verify(message, signature, public_key)

            self._warmup_run(verify)
            result = BenchmarkResult(alg, "verify", self.iterations)
            result.times_ms = _benchmark_operation(verify, self.iterations)
            results["verify"] = result
            self.results.append(result)

        return results

    # ==================== RSA Benchmarks ====================

    def benchmark_rsa(self, key_size: int = 2048) -> Dict[str, BenchmarkResult]:
        """Benchmark RSA operations."""
        from cryptography.hazmat.primitives.asymmetric import rsa, padding
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend

        alg = f"RSA-{key_size}"
        message = b"Benchmark message for RSA signatures"
        results = {}

        # Key generation
        def keygen():
            rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )

        self._warmup_run(keygen)
        result = BenchmarkResult(alg, "keygen", self.iterations)
        result.times_ms = _benchmark_operation(keygen, self.iterations)
        results["keygen"] = result
        self.results.append(result)

        # Signing
        private_key = rsa.generate_private_key(65537, key_size, default_backend())
        public_key = private_key.public_key()

        def sign():
            private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

        self._warmup_run(sign)
        result = BenchmarkResult(alg, "sign", self.iterations)
        result.times_ms = _benchmark_operation(sign, self.iterations)
        results["sign"] = result
        self.results.append(result)

        # Verification
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        def verify():
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

        self._warmup_run(verify)
        result = BenchmarkResult(alg, "verify", self.iterations)
        result.times_ms = _benchmark_operation(verify, self.iterations)
        results["verify"] = result
        self.results.append(result)

        return results

    # ==================== ECDSA Benchmarks ====================

    def benchmark_ecdsa(self, curve_name: str = "SECP256R1") -> Dict[str, BenchmarkResult]:
        """Benchmark ECDSA operations."""
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend

        alg = f"ECDSA-{curve_name}"
        message = b"Benchmark message for ECDSA signatures"

        curve_map = {
            "SECP256R1": ec.SECP256R1(),
            "SECP384R1": ec.SECP384R1(),
            "SECP521R1": ec.SECP521R1(),
        }
        curve = curve_map[curve_name]
        results = {}

        # Key generation
        def keygen():
            ec.generate_private_key(curve, default_backend())

        self._warmup_run(keygen)
        result = BenchmarkResult(alg, "keygen", self.iterations)
        result.times_ms = _benchmark_operation(keygen, self.iterations)
        results["keygen"] = result
        self.results.append(result)

        # Signing
        private_key = ec.generate_private_key(curve, default_backend())
        public_key = private_key.public_key()

        def sign():
            private_key.sign(message, ec.ECDSA(hashes.SHA256()))

        self._warmup_run(sign)
        result = BenchmarkResult(alg, "sign", self.iterations)
        result.times_ms = _benchmark_operation(sign, self.iterations)
        results["sign"] = result
        self.results.append(result)

        # Verification
        signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))

        def verify():
            public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))

        self._warmup_run(verify)
        result = BenchmarkResult(alg, "verify", self.iterations)
        result.times_ms = _benchmark_operation(verify, self.iterations)
        results["verify"] = result
        self.results.append(result)

        return results

    # ==================== ECDH Benchmarks ====================

    def benchmark_ecdh(self, curve_name: str = "X25519") -> Dict[str, BenchmarkResult]:
        """Benchmark ECDH key exchange operations."""
        from cryptography.hazmat.primitives.asymmetric import ec, x25519
        from cryptography.hazmat.backends import default_backend

        alg = f"ECDH-{curve_name}"
        results = {}

        if curve_name == "X25519":
            def keygen():
                x25519.X25519PrivateKey.generate()

            self._warmup_run(keygen)
            result = BenchmarkResult(alg, "keygen", self.iterations)
            result.times_ms = _benchmark_operation(keygen, self.iterations)
            results["keygen"] = result
            self.results.append(result)

            private_key = x25519.X25519PrivateKey.generate()
            peer_key = x25519.X25519PrivateKey.generate().public_key()

            def exchange():
                private_key.exchange(peer_key)

            self._warmup_run(exchange)
            result = BenchmarkResult(alg, "exchange", self.iterations)
            result.times_ms = _benchmark_operation(exchange, self.iterations)
            results["exchange"] = result
            self.results.append(result)
        else:
            curve_map = {
                "SECP256R1": ec.SECP256R1(),
                "SECP384R1": ec.SECP384R1(),
            }
            curve = curve_map[curve_name]

            def keygen():
                ec.generate_private_key(curve, default_backend())

            self._warmup_run(keygen)
            result = BenchmarkResult(alg, "keygen", self.iterations)
            result.times_ms = _benchmark_operation(keygen, self.iterations)
            results["keygen"] = result
            self.results.append(result)

            private_key = ec.generate_private_key(curve, default_backend())
            peer_key = ec.generate_private_key(curve, default_backend()).public_key()

            def exchange():
                private_key.exchange(ec.ECDH(), peer_key)

            self._warmup_run(exchange)
            result = BenchmarkResult(alg, "exchange", self.iterations)
            result.times_ms = _benchmark_operation(exchange, self.iterations)
            results["exchange"] = result
            self.results.append(result)

        return results

    # ==================== Full Benchmark Suite ====================

    def run_full_benchmark(self, verbose: bool = True) -> Dict[str, Any]:
        """Run complete benchmark suite.

        Args:
            verbose: Print progress messages

        Returns:
            Dict with all benchmark results
        """
        all_results = {}

        if verbose:
            print("=" * 70)
            print("Post-Quantum vs Classical Cryptography Benchmark")
            print("=" * 70)

        # ML-KEM (PQC KEM)
        if verbose:
            print("\n[ML-KEM Key Encapsulation]")
        for level in [512, 768, 1024]:
            if verbose:
                print(f"  Benchmarking ML-KEM-{level}...")
            all_results[f"ML-KEM-{level}"] = {
                k: v.to_dict() for k, v in self.benchmark_ml_kem(level).items()
            }

        # ML-DSA (PQC Signatures)
        if verbose:
            print("\n[ML-DSA Digital Signatures]")
        for level in [44, 65, 87]:
            if verbose:
                print(f"  Benchmarking ML-DSA-{level}...")
            all_results[f"ML-DSA-{level}"] = {
                k: v.to_dict() for k, v in self.benchmark_ml_dsa(level).items()
            }

        # RSA (Classical)
        if verbose:
            print("\n[RSA Signatures]")
        for size in [2048, 3072]:
            if verbose:
                print(f"  Benchmarking RSA-{size}...")
            all_results[f"RSA-{size}"] = {
                k: v.to_dict() for k, v in self.benchmark_rsa(size).items()
            }

        # ECDSA (Classical)
        if verbose:
            print("\n[ECDSA Signatures]")
        for curve in ["SECP256R1", "SECP384R1"]:
            if verbose:
                print(f"  Benchmarking ECDSA-{curve}...")
            all_results[f"ECDSA-{curve}"] = {
                k: v.to_dict() for k, v in self.benchmark_ecdsa(curve).items()
            }

        # ECDH (Classical KEM)
        if verbose:
            print("\n[ECDH Key Exchange]")
        for curve in ["X25519", "SECP256R1"]:
            if verbose:
                print(f"  Benchmarking ECDH-{curve}...")
            all_results[f"ECDH-{curve}"] = {
                k: v.to_dict() for k, v in self.benchmark_ecdh(curve).items()
            }

        return all_results

    def print_summary(self) -> None:
        """Print formatted summary table."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        # Group by algorithm type
        kem_results = [r for r in self.results if "KEM" in r.algorithm or "ECDH" in r.algorithm]
        sig_results = [r for r in self.results if "DSA" in r.algorithm or "RSA" in r.algorithm]

        print("\n[Key Encapsulation / Exchange]")
        print(f"{'Algorithm':<20} {'Operation':<10} {'Mean (ms)':<12} {'Std (ms)':<12}")
        print("-" * 54)
        for r in kem_results:
            print(f"{r.algorithm:<20} {r.operation:<10} {r.mean_ms:<12.4f} {r.std_ms:<12.4f}")

        print("\n[Digital Signatures]")
        print(f"{'Algorithm':<20} {'Operation':<10} {'Mean (ms)':<12} {'Std (ms)':<12}")
        print("-" * 54)
        for r in sig_results:
            print(f"{r.algorithm:<20} {r.operation:<10} {r.mean_ms:<12.4f} {r.std_ms:<12.4f}")

    def save_results(self, filepath: str) -> None:
        """Save benchmark results to JSON file."""
        import oqs

        results_dict = self.run_full_benchmark(verbose=False)

        output = {
            "metadata": {
                "iterations": self.iterations,
                "warmup": self.warmup,
                "liboqs_version": oqs.oqs_version(),
            },
            "results": results_dict,
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {filepath}")
