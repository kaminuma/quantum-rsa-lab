"""Base class for quantum backends."""

from abc import ABC, abstractmethod
from typing import Any
from qiskit import QuantumCircuit


class QuantumBackend(ABC):
    """Abstract base class for quantum backends."""

    @abstractmethod
    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> dict[str, int]:
        """Run a quantum circuit and return measurement counts.

        Parameters
        ----------
        circuit : QuantumCircuit
            The quantum circuit to run
        shots : int
            Number of shots (measurements)

        Returns
        -------
        dict[str, int]
            Measurement counts (e.g., {"00": 512, "11": 512})
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the backend name."""
        pass

    @property
    @abstractmethod
    def is_simulator(self) -> bool:
        """Return True if this is a simulator backend."""
        pass

    def get_info(self) -> dict[str, Any]:
        """Return backend information."""
        return {
            "name": self.name(),
            "is_simulator": self.is_simulator,
        }
