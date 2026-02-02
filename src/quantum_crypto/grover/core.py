"""Grover's algorithm core implementation.

This module provides the core Grover search algorithm that can be used
with various backends (Qiskit, Braket, etc.).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import ZGate


@dataclass
class GroverResult:
    """Result of Grover's algorithm execution."""
    n_qubits: int
    n_iterations: int
    target_states: List[int]
    measured_state: int | None
    counts: Dict[str, int]
    success: bool
    success_probability: float
    circuit_depth: int
    total_gates: int


def optimal_iterations(n_qubits: int, n_solutions: int = 1) -> int:
    """Calculate optimal number of Grover iterations.

    The optimal number is approximately (pi/4) * sqrt(N/M) where
    N = 2^n is the search space size and M is the number of solutions.

    Args:
        n_qubits: Number of qubits (search space = 2^n_qubits)
        n_solutions: Number of marked solutions

    Returns:
        Optimal number of iterations
    """
    N = 2 ** n_qubits
    if n_solutions >= N:
        return 0
    return max(1, int(math.floor(math.pi / 4 * math.sqrt(N / n_solutions))))


class GroverSearch:
    """Grover's quantum search algorithm.

    This class implements Grover's algorithm with customizable oracles.
    It can be used for:
    - Unstructured database search
    - SAT solving
    - Cryptographic key search

    Example:
        >>> from quantum_crypto.grover import GroverSearch, create_marking_oracle
        >>> oracle = create_marking_oracle(3, [5])  # Mark |101>
        >>> grover = GroverSearch(3, oracle)
        >>> circuit = grover.build_circuit()
        >>> # Run on simulator or hardware
    """

    def __init__(
        self,
        n_qubits: int,
        oracle: Callable[[QuantumCircuit, List[int]], None],
        n_solutions: int = 1,
        n_iterations: Optional[int] = None,
    ):
        """Initialize Grover search.

        Args:
            n_qubits: Number of search qubits
            oracle: Function that applies the oracle to a circuit.
                    Signature: oracle(circuit, qubit_list) -> None
            n_solutions: Number of solutions (for iteration calculation)
            n_iterations: Override automatic iteration calculation
        """
        self.n_qubits = n_qubits
        self.oracle = oracle
        self.n_solutions = n_solutions

        if n_iterations is not None:
            self.n_iterations = n_iterations
        else:
            self.n_iterations = optimal_iterations(n_qubits, n_solutions)

    def build_circuit(self, measure: bool = True) -> QuantumCircuit:
        """Build the complete Grover circuit.

        Args:
            measure: Whether to add measurement gates

        Returns:
            Qiskit QuantumCircuit
        """
        qr = QuantumRegister(self.n_qubits, 'q')
        circuit = QuantumCircuit(qr)

        if measure:
            cr = ClassicalRegister(self.n_qubits, 'c')
            circuit.add_register(cr)

        # Step 1: Initialize to uniform superposition
        circuit.h(range(self.n_qubits))

        # Step 2: Apply Grover iterations
        for _ in range(self.n_iterations):
            # Apply oracle (marks solutions with phase -1)
            self.oracle(circuit, list(range(self.n_qubits)))

            # Apply diffusion operator
            self._apply_diffusion(circuit)

        # Step 3: Measure
        if measure:
            circuit.measure(range(self.n_qubits), range(self.n_qubits))

        return circuit

    def _apply_diffusion(self, circuit: QuantumCircuit) -> None:
        """Apply Grover diffusion operator: 2|s><s| - I.

        This amplifies the amplitude of marked states.
        """
        n = self.n_qubits
        qubits = list(range(n))

        # H on all qubits
        circuit.h(qubits)

        # X on all qubits
        circuit.x(qubits)

        # Multi-controlled Z (phase flip on |11...1>)
        self._apply_mcz(circuit, qubits)

        # X on all qubits
        circuit.x(qubits)

        # H on all qubits
        circuit.h(qubits)

    def _apply_mcz(self, circuit: QuantumCircuit, qubits: List[int]) -> None:
        """Apply multi-controlled Z gate.

        Flips phase of |11...1> state.
        """
        n = len(qubits)

        if n == 1:
            circuit.z(qubits[0])
        elif n == 2:
            circuit.cz(qubits[0], qubits[1])
        elif n == 3:
            # CCZ = H-Toffoli-H decomposition
            circuit.h(qubits[2])
            circuit.ccx(qubits[0], qubits[1], qubits[2])
            circuit.h(qubits[2])
        else:
            # For n > 3, use Qiskit's controlled operation
            mcz = ZGate().control(n - 1)
            circuit.append(mcz, qubits)

    def run_simulation(self, shots: int = 1024) -> GroverResult:
        """Run Grover's algorithm on a local simulator.

        Args:
            shots: Number of measurement shots

        Returns:
            GroverResult with execution results
        """
        from qiskit_aer import AerSimulator
        from qiskit import transpile

        circuit = self.build_circuit(measure=True)
        simulator = AerSimulator()

        # Transpile to decompose high-level gates (MCZ etc.)
        transpiled = transpile(circuit, simulator)

        # Run simulation
        job = simulator.run(transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Find most frequent measurement
        max_count = 0
        measured_state = None
        for state_str, count in counts.items():
            if count > max_count:
                max_count = count
                # Qiskit measurement string is already in correct order (c[n-1]...c[0])
                measured_state = int(state_str, 2)

        # Calculate success probability (sum of solution probabilities)
        total_shots = sum(counts.values())
        success_prob = max_count / total_shots

        return GroverResult(
            n_qubits=self.n_qubits,
            n_iterations=self.n_iterations,
            target_states=[],  # Not tracked in generic search
            measured_state=measured_state,
            counts=counts,
            success=True,  # Generic search always "succeeds"
            success_probability=success_prob,
            circuit_depth=circuit.depth(),
            total_gates=circuit.size(),
        )

    def to_braket_circuit(self):
        """Convert to Amazon Braket circuit.

        Returns:
            braket.circuits.Circuit

        Raises:
            ImportError: If amazon-braket-sdk is not installed
        """
        try:
            from braket.circuits import Circuit as BraketCircuit
        except ImportError:
            raise ImportError(
                "amazon-braket-sdk is required. "
                "Install with: pip install amazon-braket-sdk"
            )

        # Build Qiskit circuit without measurement
        qiskit_circuit = self.build_circuit(measure=False)

        # Convert to Braket using qiskit-braket-provider
        try:
            from qiskit_braket_provider import BraketProvider
            from qiskit_braket_provider.providers import BraketLocalBackend

            # Use the local backend for conversion
            backend = BraketLocalBackend()
            # Transpile to Braket-compatible gates
            from qiskit import transpile
            transpiled = transpile(qiskit_circuit, backend)

            # Manual conversion for basic gates
            braket_circuit = BraketCircuit()

            for instruction in transpiled.data:
                gate = instruction.operation
                qubits = [q._index for q in instruction.qubits]

                if gate.name == 'h':
                    braket_circuit.h(qubits[0])
                elif gate.name == 'x':
                    braket_circuit.x(qubits[0])
                elif gate.name == 'z':
                    braket_circuit.z(qubits[0])
                elif gate.name == 'cx':
                    braket_circuit.cnot(qubits[0], qubits[1])
                elif gate.name == 'cz':
                    braket_circuit.cz(qubits[0], qubits[1])
                elif gate.name == 'ccx':
                    braket_circuit.ccnot(qubits[0], qubits[1], qubits[2])
                # Add more gates as needed

            return braket_circuit

        except Exception:
            # Fallback: return Qiskit circuit and let caller handle conversion
            raise NotImplementedError(
                "Direct Braket conversion failed. "
                "Use qiskit-braket-provider for full compatibility."
            )
