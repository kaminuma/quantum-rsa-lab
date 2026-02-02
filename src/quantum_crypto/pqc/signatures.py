"""ML-DSA (FIPS 204) Digital Signature implementation.

ML-DSA (Module-Lattice-Based Digital Signature Algorithm) is the NIST
standardized post-quantum digital signature scheme, formerly known as
CRYSTALS-Dilithium.

Security Levels:
- ML-DSA-44: NIST Level 2 (roughly equivalent to SHA-256/AES-128)
- ML-DSA-65: NIST Level 3 (roughly equivalent to AES-192)
- ML-DSA-87: NIST Level 5 (roughly equivalent to AES-256)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

from .utils import require_oqs


@dataclass
class SignatureResult:
    """Result of signature operation."""
    algorithm: str
    public_key: bytes
    secret_key: bytes
    signature: bytes
    message: bytes
    public_key_size: int
    signature_size: int
    is_valid: bool


def list_available_sigs() -> list[str]:
    """List all available signature algorithms from liboqs.

    Returns:
        List of algorithm names.
    """
    require_oqs()
    import oqs
    return oqs.get_enabled_sig_mechanisms()


def ml_dsa_keygen(security_level: int = 65) -> Tuple[bytes, bytes]:
    """Generate ML-DSA keypair.

    Args:
        security_level: 44, 65, or 87 (default: 65 for NIST Level 3)

    Returns:
        Tuple of (public_key, secret_key)

    Raises:
        ValueError: If security_level is not 44, 65, or 87
    """
    require_oqs()
    import oqs

    if security_level not in (44, 65, 87):
        raise ValueError(f"security_level must be 44, 65, or 87, got {security_level}")

    alg = f"ML-DSA-{security_level}"
    with oqs.Signature(alg) as signer:
        public_key = signer.generate_keypair()
        secret_key = signer.export_secret_key()
        return public_key, secret_key


def ml_dsa_sign(secret_key: bytes, message: bytes, security_level: int = 65) -> bytes:
    """Sign a message using ML-DSA.

    Args:
        secret_key: Signer's secret key
        message: Message to sign (bytes)
        security_level: Must match the key's security level (44, 65, or 87)

    Returns:
        Signature bytes
    """
    require_oqs()
    import oqs

    if security_level not in (44, 65, 87):
        raise ValueError(f"security_level must be 44, 65, or 87, got {security_level}")

    alg = f"ML-DSA-{security_level}"
    with oqs.Signature(alg, secret_key=secret_key) as signer:
        signature = signer.sign(message)
        return signature


def ml_dsa_verify(
    public_key: bytes,
    message: bytes,
    signature: bytes,
    security_level: int = 65
) -> bool:
    """Verify an ML-DSA signature.

    Args:
        public_key: Signer's public key
        message: Original message
        signature: Signature to verify
        security_level: Must match the key's security level (44, 65, or 87)

    Returns:
        True if valid, False otherwise
    """
    require_oqs()
    import oqs

    if security_level not in (44, 65, 87):
        raise ValueError(f"security_level must be 44, 65, or 87, got {security_level}")

    alg = f"ML-DSA-{security_level}"
    with oqs.Signature(alg) as verifier:
        return verifier.verify(message, signature, public_key)


def ml_dsa_full_demo(message: bytes, security_level: int = 65) -> SignatureResult:
    """Perform complete signature generation and verification.

    This demonstrates the full signature protocol:
    1. Generate keypair
    2. Sign message
    3. Verify signature

    Args:
        message: Message to sign
        security_level: 44, 65, or 87

    Returns:
        SignatureResult with all components

    Example:
        >>> result = ml_dsa_full_demo(b"Hello, quantum-safe world!", 65)
        >>> print(f"Signature valid: {result.is_valid}")
        >>> print(f"Signature size: {result.signature_size} bytes")
    """
    require_oqs()
    import oqs

    if security_level not in (44, 65, 87):
        raise ValueError(f"security_level must be 44, 65, or 87, got {security_level}")

    alg = f"ML-DSA-{security_level}"

    with oqs.Signature(alg) as signer:
        # Generate keypair
        public_key = signer.generate_keypair()
        secret_key = signer.export_secret_key()

        # Sign message
        signature = signer.sign(message)

        # Verify with separate verifier instance
        with oqs.Signature(alg) as verifier:
            is_valid = verifier.verify(message, signature, public_key)

        return SignatureResult(
            algorithm=alg,
            public_key=public_key,
            secret_key=secret_key,
            signature=signature,
            message=message,
            public_key_size=len(public_key),
            signature_size=len(signature),
            is_valid=is_valid,
        )
