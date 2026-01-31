"""Simulator backend using Qiskit Aer."""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from .base import QuantumBackend


class SimulatorBackend(QuantumBackend):
    """Local simulator backend using Qiskit Aer."""

    def __init__(self, noise_model=None):
        """Initialize the simulator.

        Parameters
        ----------
        noise_model : NoiseModel, optional
            Noise model to simulate realistic hardware
        """
        self._simulator = AerSimulator(noise_model=noise_model)
        self._noise_model = noise_model

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> dict[str, int]:
        """Run circuit on the simulator."""
        qc_transpiled = transpile(circuit, self._simulator)
        job = self._simulator.run(qc_transpiled, shots=shots)
        result = job.result()
        return result.get_counts()

    def name(self) -> str:
        return "aer_simulator" + ("_noisy" if self._noise_model else "")

    @property
    def is_simulator(self) -> bool:
        return True
