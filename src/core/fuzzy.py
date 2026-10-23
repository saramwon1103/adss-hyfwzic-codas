"""Reusable Hyperbolic Fuzzy Set (HyFS) operations.

The scalar operation intentionally uses the reproduction form documented by
this project because it reproduces the paper's Table 5 values.  Paper-specific
CSV loading and orchestration live outside this module.
"""
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class HyFS:
    """Hyperbolic fuzzy value ``H = (mu, nu)``."""

    mu: float
    nu: float

    def __post_init__(self):
        if not (0.0 <= self.mu <= 1.0):
            raise ValueError(f"mu must be in [0, 1], got {self.mu}")
        if not (0.0 <= self.nu <= 1.0):
            raise ValueError(f"nu must be in [0, 1], got {self.nu}")

        value = self.mu + self.nu - self.mu * self.nu
        if not (0.0 <= value <= 1.0 + 1e-12):
            raise ValueError(
                "Invalid HyFS: condition 0 <= mu + nu - mu*nu <= 1 "
                "is not satisfied."
            )


def score(h: HyFS) -> float:
    """Eq. (2) score/defuzzification function."""
    numerator = 2 * h.mu - h.mu * h.nu + 1
    denominator = 2 * h.nu + 1
    return numerator / denominator


def scalar_multiply(lam: float, h: HyFS) -> HyFS:
    """Scale one HyFS by ``lambda`` using the project reproduction form.

    ``lambda • H = (1 - (1-mu)^lambda, nu^lambda)``
    """
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError(f"lambda must be finite and > 0, got {lam}")

    new_mu = 1 - (1 - h.mu) ** lam
    new_nu = h.nu ** lam
    return HyFS(mu=float(new_mu), nu=float(new_nu))


def scale(values, weights):
    """Vectorized HyFS scalar multiplication used by hyperbolic CODAS.

    Parameters
    ----------
    values:
        Array-like of shape ``(alternatives, criteria, 2)`` where the last
        axis stores ``(mu, nu)``.
    weights:
        One positive scalar per criterion.

    Returns
    -------
    numpy.ndarray
        Weighted HyFS array with the same shape as ``values``.

    Notes
    -----
    This is the array counterpart of :func:`scalar_multiply`; keeping both
    APIs lets the Hy-FWZIC code work with ``HyFS`` objects while CODAS works
    efficiently with matrices.
    """
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError("values must have shape (alternatives, criteria, 2)")
    if w.ndim != 1 or len(w) != x.shape[1]:
        raise ValueError("weights must contain one scalar per criterion")
    if not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("values and weights must be finite")
    if np.any((x < 0) | (x > 1)):
        raise ValueError("HyFS mu/nu values must be in [0, 1]")
    if np.any(w <= 0):
        raise ValueError("weights must be positive")

    mu = x[..., 0]
    nu = x[..., 1]
    weighted_mu = 1.0 - np.power(1.0 - mu, w[None, :])
    weighted_nu = np.power(nu, w[None, :])
    return np.stack([weighted_mu, weighted_nu], axis=-1)


def hyfwa(values: Sequence[HyFS], weights: Sequence[float]) -> HyFS:
    """Eq. (6) Hyperbolic Fuzzy Weighted Averaging operator."""
    if len(values) == 0:
        raise ValueError("values must not be empty")
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")

    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1, got {weight_sum}")

    mu_product = 1.0
    nu_product = 1.0
    for h, w in zip(values, weights):
        mu_product *= (1 - h.mu) ** w
        nu_product *= h.nu ** w

    return HyFS(mu=1 - mu_product, nu=nu_product)


def defuzzify(h: HyFS) -> float:
    """Defuzzify an aggregated HyFS using Eq. (2)."""
    return score(h)
