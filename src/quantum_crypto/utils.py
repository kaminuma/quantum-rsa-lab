"""Utility functions for Shor's algorithm."""

import math
from fractions import Fraction
from typing import Optional


def gcd(a: int, b: int) -> int:
    """Calculate the Greatest Common Divisor of a and b."""
    return math.gcd(a, b)


def factor_from_period(a: int, N: int, r: int) -> Optional[tuple[int, int]]:
    """Try to find factors of N using the period r.

    Args:
        a: The base.
        N: The number to factorize.
        r: The period of f(x) = a^x mod N.

    Returns:
        A tuple of factors (p, q) if successful, else None.
    """
    if r % 2 != 0:
        return None

    x = pow(a, r // 2, N)
    if x == N - 1:
        return None

    factor1 = math.gcd(x - 1, N)
    factor2 = math.gcd(x + 1, N)

    if factor1 > 1 and factor1 < N:
        return (factor1, N // factor1)
    if factor2 > 1 and factor2 < N:
        return (factor2, N // factor2)

    return None


def find_period_from_phase(phase: float, N: int, max_denominator: int | None = None) -> int | None:
    """Estimate the period from the measured phase using continued fractions.

    Args:
        phase: The measured phase value (0 <= phase < 1).
        N: The number to factorize.
        max_denominator: The maximum denominator for fraction approximation.
                         Defaults to N if None.

    Returns:
        The estimated period r, or None if it cannot be determined.
    """
    if phase == 0:
        return None

    if max_denominator is None:
        max_denominator = N

    try:
        frac = Fraction(phase).limit_denominator(max_denominator)
        r = frac.denominator
        if r > 0 and r <= N:
            return r
    except (ValueError, ZeroDivisionError):
        pass
        
    return None
