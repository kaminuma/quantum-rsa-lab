"""AWS Braket backend for running on real quantum hardware."""

from typing import Any, Optional
from qiskit import QuantumCircuit

from .base import QuantumBackend

try:
    from braket.aws import AwsDevice, AwsQuantumTask
    from braket.circuits import Circuit as BraketCircuit
    BRAKET_AVAILABLE = True
except ImportError:
    BRAKET_AVAILABLE = False


# Device ARNs
DEVICE_ARNS = {
    # IonQ devices (trapped-ion)
    "ionq_aria": "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
    "ionq_forte": "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",

    # Rigetti devices (superconducting)
    "rigetti_ankaa2": "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-2",

    # IQM devices (superconducting)
    "iqm_garnet": "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet",

    # Simulators
    "sv1": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    "dm1": "arn:aws:braket:::device/quantum-simulator/amazon/dm1",
    "tn1": "arn:aws:braket:::device/quantum-simulator/amazon/tn1",
}


class AWSBraketBackend(QuantumBackend):
    """AWS Braket backend for real quantum hardware."""

    def __init__(
        self,
        device: str = "ionq_aria",
        s3_folder: Optional[tuple[str, str]] = None,
        poll_timeout_seconds: int = 3600,
    ):
        """Initialize AWS Braket backend.

        Parameters
        ----------
        device : str
            Device name. Options:
            - "ionq_aria": IonQ Aria (25 qubits, high fidelity) - recommended
            - "ionq_forte": IonQ Forte (32 qubits)
            - "rigetti_ankaa2": Rigetti Ankaa-2 (84 qubits)
            - "iqm_garnet": IQM Garnet (20 qubits)
            - "sv1": State vector simulator
            - "dm1": Density matrix simulator
            - "tn1": Tensor network simulator
        s3_folder : tuple[str, str], optional
            S3 bucket and prefix for results. If None, uses default.
        poll_timeout_seconds : int
            Timeout for waiting for results (default: 1 hour)
        """
        if not BRAKET_AVAILABLE:
            raise ImportError(
                "amazon-braket-sdk is required. Install with:\n"
                "  pip install amazon-braket-sdk\n"
                "Also ensure AWS credentials are configured."
            )

        if device not in DEVICE_ARNS:
            available = list(DEVICE_ARNS.keys())
            raise ValueError(f"Unknown device: {device}. Available: {available}")

        self._device_name = device
        self._device_arn = DEVICE_ARNS[device]
        self._device = AwsDevice(self._device_arn)
        self._s3_folder = s3_folder
        self._poll_timeout = poll_timeout_seconds

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> dict[str, int]:
        """Run circuit on AWS Braket device.

        Parameters
        ----------
        circuit : QuantumCircuit
            Qiskit QuantumCircuit to run
        shots : int
            Number of shots

        Returns
        -------
        dict[str, int]
            Measurement counts
        """
        # Convert Qiskit circuit to Braket circuit
        braket_circuit = self._qiskit_to_braket(circuit)

        # Submit task
        task = self._device.run(
            braket_circuit,
            s3_destination_folder=self._s3_folder,
            shots=shots,
        )

        print(f"Task submitted: {task.id}")
        print(f"Device: {self._device_name}")
        print(f"Status: {task.state()}")

        # Wait for result
        result = task.result()

        # Convert to Qiskit-style counts
        counts = {}
        for bitstring, count in result.measurement_counts.items():
            # Braket returns bitstrings in reverse order
            counts[bitstring[::-1]] = count

        return counts

    def _qiskit_to_braket(self, qc: QuantumCircuit) -> "BraketCircuit":
        """Convert Qiskit circuit to Braket circuit.

        Note: This is a simplified converter. For complex circuits,
        consider using qiskit-braket-provider.
        """
        braket_circuit = BraketCircuit()

        # Gate mapping
        gate_map = {
            "h": lambda q: braket_circuit.h(q),
            "x": lambda q: braket_circuit.x(q),
            "y": lambda q: braket_circuit.y(q),
            "z": lambda q: braket_circuit.z(q),
            "s": lambda q: braket_circuit.s(q),
            "sdg": lambda q: braket_circuit.si(q),
            "t": lambda q: braket_circuit.t(q),
            "tdg": lambda q: braket_circuit.ti(q),
            "cx": lambda q1, q2: braket_circuit.cnot(q1, q2),
            "cz": lambda q1, q2: braket_circuit.cz(q1, q2),
            "swap": lambda q1, q2: braket_circuit.swap(q1, q2),
        }

        for instruction, qargs, cargs in qc.data:
            gate_name = instruction.name.lower()
            qubit_indices = [q._index for q in qargs]

            if gate_name == "measure":
                continue  # Braket measures all qubits at the end
            elif gate_name == "barrier":
                continue
            elif gate_name in gate_map:
                gate_map[gate_name](*qubit_indices)
            elif gate_name == "rx":
                braket_circuit.rx(qubit_indices[0], instruction.params[0])
            elif gate_name == "ry":
                braket_circuit.ry(qubit_indices[0], instruction.params[0])
            elif gate_name == "rz":
                braket_circuit.rz(qubit_indices[0], instruction.params[0])
            elif gate_name == "p" or gate_name == "u1":
                braket_circuit.phaseshift(qubit_indices[0], instruction.params[0])
            elif gate_name == "cp" or gate_name == "cu1":
                braket_circuit.cphaseshift(
                    qubit_indices[0], qubit_indices[1], instruction.params[0]
                )
            elif gate_name == "unitary":
                # Custom unitary - need to decompose first
                raise NotImplementedError(
                    "Custom unitary gates must be decomposed before running on real hardware. "
                    "Use: circuit = transpile(circuit, basis_gates=['h','cx','rz','rx','ry'])"
                )
            else:
                raise NotImplementedError(
                    f"Gate '{gate_name}' not supported. "
                    "Transpile the circuit first with basis gates."
                )

        return braket_circuit

    def name(self) -> str:
        return f"aws_braket_{self._device_name}"

    @property
    def is_simulator(self) -> bool:
        return self._device_name in ["sv1", "dm1", "tn1"]

    def get_info(self) -> dict[str, Any]:
        """Get device information."""
        props = self._device.properties
        return {
            "name": self._device_name,
            "arn": self._device_arn,
            "is_simulator": self.is_simulator,
            "status": self._device.status,
            "provider": props.provider.name if hasattr(props, "provider") else "unknown",
            "qubit_count": getattr(props, "qubitCount", None),
        }


def list_available_devices() -> dict[str, dict]:
    """List all available AWS Braket devices with their status."""
    if not BRAKET_AVAILABLE:
        raise ImportError("amazon-braket-sdk is required")

    devices = {}
    for name, arn in DEVICE_ARNS.items():
        try:
            device = AwsDevice(arn)
            devices[name] = {
                "arn": arn,
                "status": device.status,
                "is_available": device.is_available,
            }
        except Exception as e:
            devices[name] = {
                "arn": arn,
                "status": "error",
                "error": str(e),
            }

    return devices
