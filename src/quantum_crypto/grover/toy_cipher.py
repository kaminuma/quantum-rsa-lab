"""Toy cipher oracle for Grover's algorithm key search demonstration.

This module implements a simplified block cipher that can be attacked
using Grover's algorithm on NISQ devices or simulators.

The toy cipher is designed to be:
1. Simple enough to implement as a quantum circuit
2. Complex enough to demonstrate the attack concept
3. Small enough to run on current hardware (4-8 qubits)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import ZGate


# 4-bit S-box (from mini-AES)
SBOX_4BIT = [0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8,
             0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7]

SBOX_4BIT_INV = [SBOX_4BIT.index(i) for i in range(16)]


@dataclass
class CipherAttackResult:
    """Result of Grover cipher attack."""
    plaintext: int
    ciphertext: int
    found_key: int
    correct_key: int
    success: bool
    counts: Dict[str, int]
    iterations: int
    circuit_depth: int


def toy_encrypt(plaintext: int, key: int, use_sbox: bool = False) -> int:
    """Encrypt using toy cipher (classical).

    Simple XOR-based cipher with optional S-box.

    Args:
        plaintext: 4-bit plaintext (0-15)
        key: 4-bit key (0-15)
        use_sbox: Whether to apply S-box substitution

    Returns:
        4-bit ciphertext
    """
    if use_sbox:
        # AddKey -> SubBytes
        state = plaintext ^ key
        return SBOX_4BIT[state]
    else:
        # Simple XOR cipher
        return plaintext ^ key


def toy_decrypt(ciphertext: int, key: int, use_sbox: bool = False) -> int:
    """Decrypt using toy cipher (classical)."""
    if use_sbox:
        # Inverse SubBytes -> Inverse AddKey
        state = SBOX_4BIT_INV[ciphertext]
        return state ^ key
    else:
        return ciphertext ^ key


class ToyCipherOracle:
    """Quantum oracle for toy cipher key search.

    This oracle marks key states where encryption of known plaintext
    produces known ciphertext.

    For a 4-bit key search:
    - 4 qubits for key
    - 4 qubits for state (work register)
    - Total: 8 qubits (feasible on current hardware)

    The attack uses a known-plaintext scenario:
    Given (plaintext, ciphertext) pair, find key.
    """

    def __init__(
        self,
        plaintext: int,
        ciphertext: int,
        key_bits: int = 4,
        use_sbox: bool = False
    ):
        """Initialize cipher oracle.

        Args:
            plaintext: Known plaintext (0 to 2^key_bits - 1)
            ciphertext: Known ciphertext
            key_bits: Number of key bits (default 4)
            use_sbox: Whether cipher uses S-box
        """
        self.plaintext = plaintext
        self.ciphertext = ciphertext
        self.key_bits = key_bits
        self.use_sbox = use_sbox

        # Pre-compute correct key for verification
        self.correct_key = self._find_key_classical()

    def _find_key_classical(self) -> int:
        """Find the key using classical exhaustive search (for verification)."""
        for key in range(2 ** self.key_bits):
            if toy_encrypt(self.plaintext, key, self.use_sbox) == self.ciphertext:
                return key
        return -1

    def build_oracle_circuit(
        self,
        circuit: QuantumCircuit,
        key_qubits: List[int],
        work_qubits: List[int],
        ancilla_qubit: int
    ) -> None:
        """Build the Grover oracle for key search.

        The oracle:
        1. Computes Enc(plaintext, key) into work register
        2. Compares with ciphertext
        3. Flips phase if match
        4. Uncomputes to restore state

        Args:
            circuit: Quantum circuit to add oracle to
            key_qubits: Qubits holding the key (search space)
            work_qubits: Qubits for encryption computation
            ancilla_qubit: Ancilla for phase flip
        """
        # 1. Initialize work qubits with plaintext
        self._load_value(circuit, work_qubits, self.plaintext)

        # 2. XOR key into work register (AddRoundKey)
        for i in range(self.key_bits):
            circuit.cx(key_qubits[i], work_qubits[i])

        # 3. Apply S-box if used (simplified for demo)
        if self.use_sbox:
            self._apply_sbox_approximate(circuit, work_qubits)

        # 4. Compare with ciphertext and flip phase
        self._compare_and_flip(circuit, work_qubits, ancilla_qubit)

        # 5. Uncompute: reverse steps 2-3
        if self.use_sbox:
            self._apply_sbox_inverse_approximate(circuit, work_qubits)

        for i in range(self.key_bits):
            circuit.cx(key_qubits[i], work_qubits[i])

        # 6. Unload plaintext
        self._load_value(circuit, work_qubits, self.plaintext)

    def _load_value(
        self,
        circuit: QuantumCircuit,
        qubits: List[int],
        value: int
    ) -> None:
        """Load a classical value into qubits using X gates."""
        binary = format(value, f'0{len(qubits)}b')
        for i, bit in enumerate(reversed(binary)):
            if bit == '1':
                circuit.x(qubits[i])

    def _compare_and_flip(
        self,
        circuit: QuantumCircuit,
        work_qubits: List[int],
        ancilla: int
    ) -> None:
        """Compare work register with ciphertext, flip phase if match."""
        n = len(work_qubits)

        # XOR with ciphertext - result is |0...0> if match
        self._load_value(circuit, work_qubits, self.ciphertext)

        # Multi-controlled Z on |0...0> state
        # Convert to |1...1> check
        circuit.x(work_qubits)

        # Apply MCZ
        if n <= 3:
            if n == 2:
                circuit.cz(work_qubits[0], work_qubits[1])
            elif n == 3:
                circuit.h(work_qubits[2])
                circuit.ccx(work_qubits[0], work_qubits[1], work_qubits[2])
                circuit.h(work_qubits[2])
        else:
            # For n=4, use ancilla
            circuit.ccx(work_qubits[0], work_qubits[1], ancilla)
            circuit.h(work_qubits[3])
            circuit.ccx(work_qubits[2], ancilla, work_qubits[3])
            circuit.h(work_qubits[3])
            circuit.ccx(work_qubits[0], work_qubits[1], ancilla)

        circuit.x(work_qubits)

        # Undo ciphertext XOR
        self._load_value(circuit, work_qubits, self.ciphertext)

    def _apply_sbox_approximate(
        self,
        circuit: QuantumCircuit,
        qubits: List[int]
    ) -> None:
        """Apply approximate S-box (simplified for demonstration).

        A full S-box requires many Toffoli gates.
        For demonstration, we use a simpler transformation.
        """
        # Simplified: just apply some reversible operations
        # In reality, this should be the full S-box circuit
        if len(qubits) >= 4:
            circuit.ccx(qubits[0], qubits[1], qubits[2])
            circuit.cx(qubits[2], qubits[3])

    def _apply_sbox_inverse_approximate(
        self,
        circuit: QuantumCircuit,
        qubits: List[int]
    ) -> None:
        """Apply inverse of approximate S-box."""
        if len(qubits) >= 4:
            circuit.cx(qubits[2], qubits[3])
            circuit.ccx(qubits[0], qubits[1], qubits[2])


class GroverCipherAttack:
    """Grover's algorithm attack on toy cipher.

    This class orchestrates the full attack:
    1. Build oracle from known plaintext-ciphertext pair
    2. Apply Grover iterations
    3. Measure to find the key

    Example:
        >>> attack = GroverCipherAttack(plaintext=5, ciphertext=10, key_bits=4)
        >>> result = attack.run_simulation(shots=1000)
        >>> print(f"Found key: {result.found_key}, Correct: {result.correct_key}")
    """

    def __init__(
        self,
        plaintext: int,
        ciphertext: int,
        key_bits: int = 4,
        use_sbox: bool = False,
        n_iterations: Optional[int] = None
    ):
        """Initialize Grover cipher attack.

        Args:
            plaintext: Known plaintext
            ciphertext: Known ciphertext
            key_bits: Number of key bits (search space = 2^key_bits)
            use_sbox: Whether cipher uses S-box
            n_iterations: Override automatic iteration count
        """
        self.plaintext = plaintext
        self.ciphertext = ciphertext
        self.key_bits = key_bits
        self.use_sbox = use_sbox

        self.oracle = ToyCipherOracle(
            plaintext, ciphertext, key_bits, use_sbox
        )

        # Calculate optimal iterations
        import math
        if n_iterations is None:
            N = 2 ** key_bits
            self.n_iterations = max(1, int(math.floor(math.pi / 4 * math.sqrt(N))))
        else:
            self.n_iterations = n_iterations

    def build_circuit(self) -> QuantumCircuit:
        """Build the complete Grover attack circuit.

        Returns:
            QuantumCircuit ready for execution
        """
        n = self.key_bits

        # Allocate qubits
        key_reg = QuantumRegister(n, 'key')
        work_reg = QuantumRegister(n, 'work')
        ancilla_reg = QuantumRegister(1, 'anc')
        classical_reg = ClassicalRegister(n, 'result')

        circuit = QuantumCircuit(key_reg, work_reg, ancilla_reg, classical_reg)

        # Initialize key register to superposition
        circuit.h(key_reg)

        # Grover iterations
        for _ in range(self.n_iterations):
            # Apply oracle
            self.oracle.build_oracle_circuit(
                circuit,
                list(range(n)),  # key qubits
                list(range(n, 2*n)),  # work qubits
                2*n  # ancilla
            )

            # Apply diffusion on key register
            self._apply_diffusion(circuit, list(range(n)))

        # Measure key register only
        circuit.measure(key_reg, classical_reg)

        return circuit

    def _apply_diffusion(
        self,
        circuit: QuantumCircuit,
        qubits: List[int]
    ) -> None:
        """Apply Grover diffusion operator."""
        n = len(qubits)

        circuit.h(qubits)
        circuit.x(qubits)

        # MCZ
        if n == 2:
            circuit.cz(qubits[0], qubits[1])
        elif n == 3:
            circuit.h(qubits[2])
            circuit.ccx(qubits[0], qubits[1], qubits[2])
            circuit.h(qubits[2])
        else:
            mcz = ZGate().control(n - 1)
            circuit.append(mcz, qubits)

        circuit.x(qubits)
        circuit.h(qubits)

    def run_simulation(self, shots: int = 1024) -> CipherAttackResult:
        """Run the attack on a simulator.

        Args:
            shots: Number of measurement shots

        Returns:
            CipherAttackResult with attack results
        """
        from qiskit_aer import AerSimulator
        from qiskit import transpile

        circuit = self.build_circuit()
        simulator = AerSimulator()

        # Transpile to decompose high-level gates
        transpiled = transpile(circuit, simulator)

        job = simulator.run(transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Find most frequent measurement (the key)
        max_count = 0
        found_key = None
        for state_str, count in counts.items():
            if count > max_count:
                max_count = count
                # Qiskit measurement string is already in correct order
                found_key = int(state_str, 2)

        success = (found_key == self.oracle.correct_key)

        return CipherAttackResult(
            plaintext=self.plaintext,
            ciphertext=self.ciphertext,
            found_key=found_key,
            correct_key=self.oracle.correct_key,
            success=success,
            counts=counts,
            iterations=self.n_iterations,
            circuit_depth=circuit.depth(),
        )


def demo_grover_attack():
    """Demonstrate Grover's algorithm attacking a toy cipher.

    This function shows:
    1. Classical encryption with a random key
    2. Grover's algorithm finding the key
    3. Verification of the found key
    """
    import random

    print("=" * 60)
    print("Grover's Algorithm: Toy Cipher Key Search Demo")
    print("=" * 60)

    # Parameters
    key_bits = 4
    key = random.randint(0, 15)
    plaintext = random.randint(0, 15)
    ciphertext = toy_encrypt(plaintext, key, use_sbox=False)

    print(f"\nSetup:")
    print(f"  Key bits: {key_bits}")
    print(f"  Search space: 2^{key_bits} = {2**key_bits}")
    print(f"  Secret key: {key} (0b{key:04b})")
    print(f"  Plaintext: {plaintext} (0b{plaintext:04b})")
    print(f"  Ciphertext: {ciphertext} (0b{ciphertext:04b})")

    # Run attack
    print(f"\nRunning Grover attack...")
    attack = GroverCipherAttack(
        plaintext=plaintext,
        ciphertext=ciphertext,
        key_bits=key_bits,
        use_sbox=False
    )

    result = attack.run_simulation(shots=1000)

    print(f"\nResults:")
    print(f"  Grover iterations: {result.iterations}")
    print(f"  Circuit depth: {result.circuit_depth}")
    print(f"  Found key: {result.found_key} (0b{result.found_key:04b})")
    print(f"  Correct key: {result.correct_key}")
    print(f"  Attack success: {result.success}")

    # Show measurement distribution
    print(f"\nTop measurement outcomes:")
    sorted_counts = sorted(result.counts.items(), key=lambda x: x[1], reverse=True)
    for state, count in sorted_counts[:5]:
        key_val = int(state[::-1], 2)
        print(f"  Key {key_val:2d} (0b{key_val:04b}): {count:4d} shots ({100*count/1000:.1f}%)")

    return result


if __name__ == "__main__":
    demo_grover_attack()
