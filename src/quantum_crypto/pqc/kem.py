"""ML-KEM (FIPS 203) Key Encapsulation Mechanism implementation.

ML-KEM (Module-Lattice-Based Key-Encapsulation Mechanism) is the NIST
standardized post-quantum key encapsulation scheme, formerly known as
CRYSTALS-Kyber.

Security Levels:
- ML-KEM-512:  NIST Level 1 (equivalent to AES-128)
- ML-KEM-768:  NIST Level 3 (equivalent to AES-192)
- ML-KEM-1024: NIST Level 5 (equivalent to AES-256)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

from .utils import require_oqs


@dataclass
class KEMResult:
    """Result of KEM operation."""
    algorithm: str
    public_key: bytes
    secret_key: bytes
    ciphertext: bytes
    shared_secret: bytes
    key_size: int
    ciphertext_size: int


def list_available_kems() -> list[str]:
    """List all available KEM algorithms from liboqs.

    Returns:
        List of algorithm names.
    """
    require_oqs()
    import oqs
    return oqs.get_enabled_kem_mechanisms()


def ml_kem_keygen(security_level: int = 768) -> Tuple[bytes, bytes]:
    """Generate ML-KEM keypair.

    Args:
        security_level: 512, 768, or 1024 (default: 768 for NIST Level 3)

    Returns:
        Tuple of (public_key, secret_key)

    Raises:
        ValueError: If security_level is not 512, 768, or 1024
    """
    require_oqs()
    import oqs

    if security_level not in (512, 768, 1024):
        raise ValueError(f"security_level must be 512, 768, or 1024, got {security_level}")

    alg = f"ML-KEM-{security_level}"
    with oqs.KeyEncapsulation(alg) as kem:
        public_key = kem.generate_keypair()
        secret_key = kem.export_secret_key()
        return public_key, secret_key


def ml_kem_encapsulate(public_key: bytes, security_level: int = 768) -> Tuple[bytes, bytes]:
    """Encapsulate a shared secret using recipient's public key.

    Args:
        public_key: Recipient's public key
        security_level: Must match the key's security level (512, 768, or 1024)

    Returns:
        Tuple of (ciphertext, shared_secret)
    """
    require_oqs()
    import oqs

    if security_level not in (512, 768, 1024):
        raise ValueError(f"security_level must be 512, 768, or 1024, got {security_level}")

    alg = f"ML-KEM-{security_level}"
    with oqs.KeyEncapsulation(alg) as kem:
        ciphertext, shared_secret = kem.encap_secret(public_key)
        return ciphertext, shared_secret


def ml_kem_decapsulate(secret_key: bytes, ciphertext: bytes, security_level: int = 768) -> bytes:
    """Decapsulate to recover the shared secret.

    Args:
        secret_key: Recipient's secret key
        ciphertext: Encapsulated ciphertext
        security_level: Must match the key's security level (512, 768, or 1024)

    Returns:
        Shared secret bytes
    """
    require_oqs()
    import oqs

    if security_level not in (512, 768, 1024):
        raise ValueError(f"security_level must be 512, 768, or 1024, got {security_level}")

    alg = f"ML-KEM-{security_level}"
    with oqs.KeyEncapsulation(alg, secret_key=secret_key) as kem:
        shared_secret = kem.decap_secret(ciphertext)
        return shared_secret


def ml_kem_full_exchange(security_level: int = 768) -> KEMResult:
    """Perform complete KEM key exchange demonstration.

    This demonstrates the full key exchange protocol:
    1. Alice generates keypair
    2. Bob encapsulates secret using Alice's public key
    3. Alice decapsulates to get the same secret

    Args:
        security_level: 512, 768, or 1024

    Returns:
        KEMResult with all components

    Example:
        >>> result = ml_kem_full_exchange(768)
        >>> print(f"Public key size: {result.key_size} bytes")
        >>> print(f"Shared secret: {result.shared_secret.hex()[:32]}...")
    """
    require_oqs()
    import oqs

    if security_level not in (512, 768, 1024):
        raise ValueError(f"security_level must be 512, 768, or 1024, got {security_level}")

    alg = f"ML-KEM-{security_level}"

    # Alice (receiver) generates keypair
    with oqs.KeyEncapsulation(alg) as alice:
        alice_public = alice.generate_keypair()
        alice_secret = alice.export_secret_key()

        # Bob (sender) encapsulates
        with oqs.KeyEncapsulation(alg) as bob:
            ciphertext, shared_secret_bob = bob.encap_secret(alice_public)

        # Alice decapsulates
        shared_secret_alice = alice.decap_secret(ciphertext)

        # Verify secrets match
        assert shared_secret_alice == shared_secret_bob, "Key exchange failed!"

        return KEMResult(
            algorithm=alg,
            public_key=alice_public,
            secret_key=alice_secret,
            ciphertext=ciphertext,
            shared_secret=shared_secret_alice,
            key_size=len(alice_public),
            ciphertext_size=len(ciphertext),
        )
