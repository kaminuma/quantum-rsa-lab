"""Classical implementation of Shor's algorithm logic."""

import random
from typing import Any, Optional

from ..core import ShorAlgorithm, ShorResult
from ..utils import gcd, factor_from_period


class ClassicalShor(ShorAlgorithm):
    """Classical implementation of period finding for factoring."""

    def run(self, number: int, base: Optional[int] = None, shots: int = 1, **kwargs: Any) -> ShorResult:
        """Run the classical period finding algorithm.
        
        Args:
            number: The integer to factorize.
            base: The base 'a'. If None, a random base is chosen.
            shots: Unused in classical method, kept for interface consistency.
            **kwargs: Unused.
            
        Returns:
            ShorResult containing the factorization results.
        """
        if number % 2 == 0:
            return ShorResult(
                number=number,
                base=2,
                factors=(2, number // 2),
                shots=shots,
                success=True,
                method="classical_trivial_even",
            )

        if base is None:
            # Try to find a valid base
            for _ in range(100): # Avoid infinite loop
                candidate = random.randint(2, number - 1)
                if gcd(candidate, number) == 1:
                    base = candidate
                    break
            else:
                 # Fallback if no coprime found (unlikely for composite N)
                 base = random.randint(2, number - 1)

        g = gcd(base, number)
        if g > 1:
            return ShorResult(
                number=number,
                base=base,
                factors=(g, number // g),
                shots=shots,
                success=True,
                method="classical_gcd",
            )

        r = self._find_period(base, number)

        if r is None:
            return ShorResult(
                number=number,
                base=base,
                factors=None,
                shots=shots,
                success=False,
                method="classical_period_not_found",
            )

        factors = factor_from_period(base, number, r)

        if factors:
            return ShorResult(
                number=number,
                base=base,
                factors=factors,
                shots=shots,
                success=True,
                method=f"classical_period_r={r}",
                period=r,
            )
        else:
            return ShorResult(
                number=number,
                base=base,
                factors=None,
                shots=shots,
                success=False,
                method=f"classical_period_r={r}_but_failed",
                period=r,
            )

    def _find_period(self, a: int, N: int) -> Optional[int]:
        """Find period r such that a^r = 1 (mod N) using classical iteration."""
        if gcd(a, N) != 1:
            return None

        r = 1
        value = a
        # Safety limit to prevent infinite loops for very large N or bugs
        limit = N * 2 
        
        while value != 1:
            value = (value * a) % N
            r += 1
            if r > limit:
                return None
        return r
