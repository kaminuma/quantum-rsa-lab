"""Backend implementations for running Shor's algorithm on real quantum hardware."""

from .base import QuantumBackend
from .simulator import SimulatorBackend

__all__ = ["QuantumBackend", "SimulatorBackend"]

# Optional backends
try:
    from .aws_braket import AWSBraketBackend
    __all__.append("AWSBraketBackend")
except ImportError:
    pass

try:
    from .ibm_quantum import IBMQuantumBackend
    __all__.append("IBMQuantumBackend")
except ImportError:
    pass
