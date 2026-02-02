"""Hybrid cryptography: Classical + Post-Quantum combined schemes.

Hybrid cryptography provides defense-in-depth by combining:
- Classical algorithms (X25519, ECDSA) - battle-tested, well-understood
- Post-quantum algorithms (ML-KEM, ML-DSA) - quantum-resistant

This approach is recommended during the transition period (2024-2035)
and is used by major implementations:
- AWS KMS (X25519 + ML-KEM)
- Cloudflare (X25519 + Kyber)
- Google Chrome (X25519Kyber768)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from .utils import require_oqs


@dataclass
class HybridKEMKeys:
    """Hybrid keypair containing both classical and PQC keys."""
    x25519_private: "x25519.X25519PrivateKey"
    x25519_public: bytes
    ml_kem_public: bytes
    ml_kem_secret: bytes
    ml_kem_level: int


@dataclass
class HybridEncapsulation:
    """Result of hybrid encapsulation."""
    x25519_public: bytes  # Ephemeral X25519 public key
    ml_kem_ciphertext: bytes
    combined_secret: bytes


class HybridKEM:
    """Hybrid Key Encapsulation: X25519 + ML-KEM.

    Security: The combined shared secret uses HKDF to derive the final key
    from both X25519 and ML-KEM shared secrets, ensuring security as long
    as at least one algorithm remains secure.

    Example:
        >>> hybrid = HybridKEM(ml_kem_level=768)
        >>> alice_keys = hybrid.generate_keypair()
        >>> encap = hybrid.encapsulate(alice_keys)
        >>> alice_secret = hybrid.decapsulate(alice_keys, encap)
        >>> assert alice_secret == encap.combined_secret
    """

    def __init__(self, ml_kem_level: int = 768):
        """Initialize hybrid KEM.

        Args:
            ml_kem_level: ML-KEM security level (512, 768, or 1024)
        """
        require_oqs()
        if ml_kem_level not in (512, 768, 1024):
            raise ValueError(f"ml_kem_level must be 512, 768, or 1024, got {ml_kem_level}")

        self.ml_kem_level = ml_kem_level
        self.ml_kem_alg = f"ML-KEM-{ml_kem_level}"

    def generate_keypair(self) -> HybridKEMKeys:
        """Generate hybrid keypair (X25519 + ML-KEM)."""
        import oqs
        from cryptography.hazmat.primitives.asymmetric import x25519

        # X25519 keypair
        x25519_private = x25519.X25519PrivateKey.generate()
        x25519_public = x25519_private.public_key().public_bytes_raw()

        # ML-KEM keypair
        with oqs.KeyEncapsulation(self.ml_kem_alg) as kem:
            ml_kem_public = kem.generate_keypair()
            ml_kem_secret = kem.export_secret_key()

        return HybridKEMKeys(
            x25519_private=x25519_private,
            x25519_public=x25519_public,
            ml_kem_public=ml_kem_public,
            ml_kem_secret=ml_kem_secret,
            ml_kem_level=self.ml_kem_level,
        )

    def encapsulate(self, recipient_keys: HybridKEMKeys) -> HybridEncapsulation:
        """Encapsulate a shared secret using recipient's hybrid public keys.

        Args:
            recipient_keys: Recipient's hybrid public keys

        Returns:
            HybridEncapsulation containing ciphertexts and combined secret
        """
        import oqs
        from cryptography.hazmat.primitives.asymmetric import x25519

        # X25519: Generate ephemeral keypair and perform ECDH
        ephemeral_x25519 = x25519.X25519PrivateKey.generate()
        ephemeral_public = ephemeral_x25519.public_key().public_bytes_raw()

        recipient_x25519_public = x25519.X25519PublicKey.from_public_bytes(
            recipient_keys.x25519_public
        )
        x25519_shared = ephemeral_x25519.exchange(recipient_x25519_public)

        # ML-KEM: Encapsulate
        with oqs.KeyEncapsulation(self.ml_kem_alg) as kem:
            ml_kem_ct, ml_kem_shared = kem.encap_secret(recipient_keys.ml_kem_public)

        # Combine shared secrets using HKDF
        combined_secret = self._combine_secrets(x25519_shared, ml_kem_shared)

        return HybridEncapsulation(
            x25519_public=ephemeral_public,
            ml_kem_ciphertext=ml_kem_ct,
            combined_secret=combined_secret,
        )

    def decapsulate(
        self,
        keys: HybridKEMKeys,
        encapsulation: HybridEncapsulation
    ) -> bytes:
        """Decapsulate to recover the shared secret.

        Args:
            keys: Recipient's hybrid keypair (including private keys)
            encapsulation: The encapsulation from sender

        Returns:
            Combined shared secret
        """
        import oqs
        from cryptography.hazmat.primitives.asymmetric import x25519

        # X25519: Perform ECDH with sender's ephemeral public key
        sender_x25519_public = x25519.X25519PublicKey.from_public_bytes(
            encapsulation.x25519_public
        )
        x25519_shared = keys.x25519_private.exchange(sender_x25519_public)

        # ML-KEM: Decapsulate
        with oqs.KeyEncapsulation(self.ml_kem_alg,
                                   secret_key=keys.ml_kem_secret) as kem:
            ml_kem_shared = kem.decap_secret(encapsulation.ml_kem_ciphertext)

        # Combine shared secrets
        combined_secret = self._combine_secrets(x25519_shared, ml_kem_shared)

        return combined_secret

    def _combine_secrets(self, x25519_shared: bytes, ml_kem_shared: bytes) -> bytes:
        """Combine two shared secrets using HKDF.

        Uses concatenation followed by HKDF as recommended by NIST.
        """
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend

        # Concatenate both shared secrets
        combined_input = x25519_shared + ml_kem_shared

        # Derive final key using HKDF
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"hybrid-kem-v1",
            info=b"X25519-ML-KEM combined secret",
            backend=default_backend()
        )

        return hkdf.derive(combined_input)


class HybridSignature:
    """Hybrid Digital Signature: ECDSA + ML-DSA.

    Both signatures must be valid for verification to succeed,
    providing defense-in-depth.

    Example:
        >>> hybrid = HybridSignature(ml_dsa_level=65)
        >>> keys = hybrid.generate_keypair()
        >>> sigs = hybrid.sign(keys, b"Important message")
        >>> is_valid = hybrid.verify(keys, b"Important message", sigs)
    """

    def __init__(self, ml_dsa_level: int = 65):
        """Initialize hybrid signature scheme.

        Args:
            ml_dsa_level: ML-DSA security level (44, 65, or 87)
        """
        require_oqs()
        if ml_dsa_level not in (44, 65, 87):
            raise ValueError(f"ml_dsa_level must be 44, 65, or 87, got {ml_dsa_level}")

        self.ml_dsa_level = ml_dsa_level
        self.ml_dsa_alg = f"ML-DSA-{ml_dsa_level}"

    def generate_keypair(self) -> dict:
        """Generate hybrid signature keypair (ECDSA + ML-DSA)."""
        import oqs
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.backends import default_backend

        # ECDSA keypair (P-256)
        ecdsa_private = ec.generate_private_key(ec.SECP256R1(), default_backend())
        ecdsa_public = ecdsa_private.public_key()

        # ML-DSA keypair
        with oqs.Signature(self.ml_dsa_alg) as sig:
            ml_dsa_public = sig.generate_keypair()
            ml_dsa_secret = sig.export_secret_key()

        return {
            "ecdsa_private": ecdsa_private,
            "ecdsa_public": ecdsa_public,
            "ml_dsa_public": ml_dsa_public,
            "ml_dsa_secret": ml_dsa_secret,
        }

    def sign(self, keys: dict, message: bytes) -> dict:
        """Create hybrid signature (both ECDSA and ML-DSA).

        Args:
            keys: Hybrid keypair dict
            message: Message to sign

        Returns:
            Dict containing both signatures
        """
        import oqs
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import hashes

        # ECDSA signature
        ecdsa_sig = keys["ecdsa_private"].sign(
            message,
            ec.ECDSA(hashes.SHA256())
        )

        # ML-DSA signature
        with oqs.Signature(self.ml_dsa_alg,
                          secret_key=keys["ml_dsa_secret"]) as signer:
            ml_dsa_sig = signer.sign(message)

        return {
            "ecdsa_signature": ecdsa_sig,
            "ml_dsa_signature": ml_dsa_sig,
        }

    def verify(self, keys: dict, message: bytes, signatures: dict) -> bool:
        """Verify hybrid signature (both must be valid).

        Args:
            keys: Hybrid keypair dict (public keys)
            message: Original message
            signatures: Dict containing both signatures

        Returns:
            True only if BOTH signatures are valid
        """
        import oqs
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import hashes
        from cryptography.exceptions import InvalidSignature

        # Verify ECDSA
        try:
            keys["ecdsa_public"].verify(
                signatures["ecdsa_signature"],
                message,
                ec.ECDSA(hashes.SHA256())
            )
            ecdsa_valid = True
        except InvalidSignature:
            ecdsa_valid = False

        # Verify ML-DSA
        with oqs.Signature(self.ml_dsa_alg) as verifier:
            ml_dsa_valid = verifier.verify(
                message,
                signatures["ml_dsa_signature"],
                keys["ml_dsa_public"]
            )

        # Both must be valid
        return ecdsa_valid and ml_dsa_valid
