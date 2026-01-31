"""Quantum implementation of Shor's algorithm."""

from __future__ import annotations

import math
from typing import Any, Optional

from ..core import ShorAlgorithm, ShorResult
from ..utils import gcd, factor_from_period, find_period_from_phase

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import QFT
    from qiskit.primitives import Sampler
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    from ..modexp import get_config
    MODEXP_AVAILABLE = True
except ImportError:
    MODEXP_AVAILABLE = False


class QuantumShor(ShorAlgorithm):
    """Quantum implementation of Shor's algorithm using QPE."""

    def __init__(self, n_count: int = 8):
        """Initialize QuantumShor.

        Args:
            n_count: Number of counting qubits for QPE.
        """
        self.n_count = n_count
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit is required for QuantumShor. Please install qiskit.")

    def run(self, number: int, base: Optional[int] = None, shots: int = 1024, **kwargs: Any) -> ShorResult:
        """Run the quantum algorithm.

        Args:
            number: The integer to factorize.
            base: The base 'a'.
            shots: Number of shots.
            **kwargs: Additional arguments (e.g., sampler options).

        Returns:
            ShorResult containing the factorization results.
        """
        if not MODEXP_AVAILABLE:
            raise RuntimeError("modexp module is required.")

        # 1. Validate inputs and check for trivial factors
        try:
            config = get_config(number)
        except NotImplementedError as e:
            raise NotImplementedError(f"N={number} is not supported yet.") from e

        valid_bases = config["valid_bases"]
        
        if base is None:
            base = valid_bases[len(valid_bases) // 2]

        if base not in valid_bases:
            # If base is not in valid_bases, it might still be valid if gcd(base, N) == 1
            # But our modexp circuits are pre-compiled for specific bases.
            # We'll check if it's a supported base for the circuit.
             raise ValueError(f"a={base} is not supported for N={number}. Supported bases: {valid_bases}")

        g = gcd(base, number)
        if g > 1:
            return ShorResult(
                number=number,
                base=base,
                factors=(g, number // g),
                shots=shots,
                success=True,
                method="quantum_gcd",
            )

        # 2. Construct Quantum Circuit
        qc = self._construct_circuit(base, number, config)

        # Transpile circuit and collect metrics
        from qiskit_aer import AerSimulator
        simulator = AerSimulator()
        qc_transpiled = transpile(qc, simulator)
        circuit_depth, total_gates, two_qubit_gates, gate_counts = self._collect_metrics(qc_transpiled)

        # 3. Execute using Sampler Primitive
        sampler = Sampler()
        # Note: Sampler V1 run takes circuits
        job = sampler.run(circuits=[qc], shots=shots)
        result = job.result()
        
        # 4. Process Results
        quasi_dists = result.quasi_dists[0]
        
        # Convert quasi-dists to counts (approximate) for compatibility
        counts = {
            format(k, f"0{self.n_count}b"): int(v * shots) 
            for k, v in quasi_dists.items() 
            if v > 0
        }
        
        # Sort by probability
        sorted_dist = sorted(quasi_dists.items(), key=lambda x: x[1], reverse=True)

        best_phase = 0.0
        
        for measured_int, prob in sorted_dist[:10]:
            if measured_int == 0:
                continue
                
            phase = measured_int / (2 ** self.n_count)
            r = find_period_from_phase(phase, number, config.get("max_denominator"))
            
            if r is None or r % 2 != 0:
                continue
                
            factors = factor_from_period(base, number, r)
            
            if factors:
                return ShorResult(
                    number=number,
                    base=base,
                    factors=factors,
                    shots=shots,
                    success=True,
                    method=f"quantum_qpe_r={r}",
                    measured_phase=phase,
                    period=r,
                    counts=counts,
                    circuit_depth=circuit_depth,
                    total_gates=total_gates,
                    two_qubit_gates=two_qubit_gates,
                    gate_counts=gate_counts,
                )
            
            # Keep track of the most likely non-zero phase for reporting failure
            if best_phase == 0:
                best_phase = phase

        return ShorResult(
            number=number,
            base=base,
            factors=None,
            shots=shots,
            success=False,
            method="quantum_qpe_failed",
            measured_phase=best_phase,
            counts=counts,
            circuit_depth=circuit_depth,
            total_gates=total_gates,
            two_qubit_gates=two_qubit_gates,
            gate_counts=gate_counts,
        )

    def _construct_circuit(self, a: int, N: int, config: dict) -> QuantumCircuit:
        """Construct the QPE circuit for a^x mod N."""
        c_amod_func = config["func"]
        n_work = config["n_work_qubits"]
        
        qc = QuantumCircuit(self.n_count + n_work, self.n_count)

        # Initialize counting qubits to |+>
        for q in range(self.n_count):
            qc.h(q)

        # Initialize work register to |1>
        qc.x(self.n_count)

        # Apply controlled-U operations
        for q in range(self.n_count):
            # 2^q power
            c_U = c_amod_func(a, q)
            # Append c_U to the circuit
            # The control qubit is q, target qubits are n_count to n_count + n_work - 1
            # But c_amod_func returns a gate that expects control as the first qubit(s) usually?
            # Let's check the original implementation.
            # Original: qc.append(c_U, [q] + list(range(n_count, n_count + n_work)))
            # c_amod15 returns a controlled gate.
            qc.append(c_U, [q] + list(range(self.n_count, self.n_count + n_work)))

        # Inverse QFT
        qc.append(QFT(self.n_count, inverse=True), range(self.n_count))

        # Measure counting qubits
        qc.measure(range(self.n_count), range(self.n_count))

        return qc

    def _collect_metrics(self, qc: QuantumCircuit) -> tuple[int, int, int, dict]:
        """Collect circuit metrics like depth and gate counts."""
        TWO_QUBIT_GATES = {
            "cx", "cy", "cz", "cp", "cu", "cu1", "cu3",
            "ecr", "iswap", "swap", "cswap",
            "rxx", "ryy", "rzz", "xx_plus_yy", "xx_minus_yy",
        }
        
        gate_counts = dict(qc.count_ops())
        total_gates = int(sum(gate_counts.values()))
        two_qubit_gates = int(sum(count for gate, count in gate_counts.items() if gate in TWO_QUBIT_GATES))
        depth = qc.depth()
        
        return depth, total_gates, two_qubit_gates, gate_counts

