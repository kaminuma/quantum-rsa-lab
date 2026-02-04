"""IBM Quantum backend for running on real quantum hardware."""

from typing import Any, Optional
from qiskit import QuantumCircuit, transpile

from .base import QuantumBackend

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False


class IBMQuantumBackend(QuantumBackend):
    """IBM Quantum backend for real quantum hardware."""

    def __init__(
        self,
        backend_name: Optional[str] = None,
        optimization_level: int = 3,
    ):
        """Initialize IBM Quantum backend.

        Parameters
        ----------
        backend_name : str, optional
            Specific backend name (e.g., "ibm_kyoto").
            If None, auto-selects least busy available backend.
        optimization_level : int
            Transpiler optimization level (0-3). Higher = more optimization.
        """
        if not IBM_AVAILABLE:
            raise ImportError(
                "qiskit-ibm-runtime is required. Install with:\n"
                "  pip install qiskit-ibm-runtime\n"
                "Then save your API token:\n"
                "  IBMQuantumBackend.save_account('YOUR_TOKEN')"
            )

        self._service = QiskitRuntimeService()
        self._optimization_level = optimization_level

        if backend_name:
            self._backend = self._service.backend(backend_name)
        else:
            # Auto-select least busy backend
            self._backend = self._service.least_busy(
                operational=True,
                simulator=False,
            )

        self._backend_name = self._backend.name

    @staticmethod
    def save_account(token: str, overwrite: bool = True):
        """Save IBM Quantum API token.

        Parameters
        ----------
        token : str
            Your IBM Quantum API token from https://quantum.ibm.com/
        overwrite : bool
            Whether to overwrite existing credentials
        """
        if not IBM_AVAILABLE:
            raise ImportError("qiskit-ibm-runtime is required")

        QiskitRuntimeService.save_account(
            channel="ibm_quantum",
            token=token,
            overwrite=overwrite,
        )
        print("Account saved successfully!")

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> dict[str, int]:
        """Run circuit on IBM Quantum hardware.

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
        # Transpile for the target backend
        qc_transpiled = transpile(
            circuit,
            self._backend,
            optimization_level=self._optimization_level,
        )

        print(f"Original circuit: {circuit.depth()} depth, {sum(circuit.count_ops().values())} gates")
        print(f"Transpiled circuit: {qc_transpiled.depth()} depth, {sum(qc_transpiled.count_ops().values())} gates")
        print(f"Running on: {self._backend_name}")

        # Run using SamplerV2
        sampler = SamplerV2(self._backend)
        job = sampler.run([qc_transpiled], shots=shots)

        print(f"Job ID: {job.job_id()}")
        print("Waiting for results...")

        result = job.result()

        # Extract counts from SamplerV2 result
        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()

        return counts

    def name(self) -> str:
        return f"ibm_quantum_{self._backend_name}"

    @property
    def is_simulator(self) -> bool:
        return False

    def get_info(self) -> dict[str, Any]:
        """Get backend information."""
        return {
            "name": self._backend_name,
            "num_qubits": self._backend.num_qubits,
            "basis_gates": self._backend.basis_gates,
            "is_simulator": False,
            "status": self._backend.status().status_msg,
        }

    @staticmethod
    def list_backends() -> list[dict]:
        """List all available IBM Quantum backends."""
        if not IBM_AVAILABLE:
            raise ImportError("qiskit-ibm-runtime is required")

        service = QiskitRuntimeService()
        backends = []

        for backend in service.backends():
            backends.append({
                "name": backend.name,
                "num_qubits": backend.num_qubits,
                "status": backend.status().status_msg,
                "pending_jobs": backend.status().pending_jobs,
            })

        return backends
