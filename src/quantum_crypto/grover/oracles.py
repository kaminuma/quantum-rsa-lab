"""Oracle construction for Grover's algorithm.

Oracles mark the target states by flipping their phase.
"""

from __future__ import annotations
from typing import List, Callable

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate


def create_marking_oracle(
    n_qubits: int,
    marked_states: List[int]
) -> Callable[[QuantumCircuit, List[int]], None]:
    """Create a phase oracle that marks specified states.

    The oracle applies a phase flip (-1) to the marked states.

    Args:
        n_qubits: Number of qubits
        marked_states: List of integers representing states to mark
                      (e.g., [5] marks |101> in 3-qubit system)

    Returns:
        Oracle function with signature (circuit, qubits) -> None

    Example:
        >>> oracle = create_marking_oracle(3, [5])  # Mark |101>
        >>> circuit = QuantumCircuit(3)
        >>> oracle(circuit, [0, 1, 2])
    """
    def oracle(circuit: QuantumCircuit, qubits: List[int]) -> None:
        for state in marked_states:
            _mark_state(circuit, qubits, state, n_qubits)

    return oracle


def _mark_state(
    circuit: QuantumCircuit,
    qubits: List[int],
    state: int,
    n_qubits: int
) -> None:
    """Mark a single state with phase -1.

    Uses X gates to convert the target to |11...1>, applies MCZ,
    then unconverts.
    """
    # Convert state to binary
    binary = format(state, f'0{n_qubits}b')

    # Apply X to qubits that should be |0> in target state
    for i, bit in enumerate(reversed(binary)):
        if bit == '0':
            circuit.x(qubits[i])

    # Apply multi-controlled Z
    _apply_mcz(circuit, qubits)

    # Undo X gates
    for i, bit in enumerate(reversed(binary)):
        if bit == '0':
            circuit.x(qubits[i])


def _apply_mcz(circuit: QuantumCircuit, qubits: List[int]) -> None:
    """Apply multi-controlled Z gate."""
    n = len(qubits)

    if n == 1:
        circuit.z(qubits[0])
    elif n == 2:
        circuit.cz(qubits[0], qubits[1])
    elif n == 3:
        # CCZ using H-Toffoli-H
        circuit.h(qubits[2])
        circuit.ccx(qubits[0], qubits[1], qubits[2])
        circuit.h(qubits[2])
    else:
        # General case using Qiskit's controlled Z
        mcz = ZGate().control(n - 1)
        circuit.append(mcz, qubits)


def create_phase_oracle(
    n_qubits: int,
    phase_function: Callable[[int], bool]
) -> Callable[[QuantumCircuit, List[int]], None]:
    """Create an oracle from a boolean function.

    The oracle marks states where phase_function returns True.

    Args:
        n_qubits: Number of qubits
        phase_function: Function that takes an integer state and
                       returns True if it should be marked

    Returns:
        Oracle function

    Example:
        >>> # Mark all even numbers
        >>> oracle = create_phase_oracle(3, lambda x: x % 2 == 0)
    """
    # Pre-compute marked states
    marked_states = [
        state for state in range(2 ** n_qubits)
        if phase_function(state)
    ]

    return create_marking_oracle(n_qubits, marked_states)


def create_comparison_oracle(
    n_qubits: int,
    target_value: int
) -> Callable[[QuantumCircuit, List[int]], None]:
    """Create an oracle that marks states equal to target_value.

    This is useful for key search where we want to find a specific key.

    Args:
        n_qubits: Number of qubits
        target_value: The value to search for

    Returns:
        Oracle function

    Example:
        >>> oracle = create_comparison_oracle(4, 7)  # Find |0111>
    """
    return create_marking_oracle(n_qubits, [target_value])
