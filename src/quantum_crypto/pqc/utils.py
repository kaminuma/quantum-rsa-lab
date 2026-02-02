"""Utility functions for PQC module."""

from __future__ import annotations


def check_pqc_available() -> bool:
    """Check if liboqs-python is available.

    Returns:
        True if liboqs is installed and functional, False otherwise.
    """
    try:
        import oqs
        # Try to get version to ensure library is properly loaded
        _ = oqs.oqs_version()
        return True
    except ImportError:
        return False
    except Exception:
        return False


def require_oqs():
    """Decorator/helper to raise informative error if oqs is not available."""
    if not check_pqc_available():
        raise ImportError(
            "liboqs-python is required for PQC functionality. "
            "Install it with: pip install liboqs-python"
        )
