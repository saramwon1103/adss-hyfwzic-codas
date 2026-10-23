"""
Basic Hyperbolic Fuzzy Set (HyFS) operations.

This module contains only reusable mathematical operations used by
Hy-FWZIC. It does not load datasets or contain paper-specific logic.
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class HyFS:
    """
    Hyperbolic Fuzzy Set represented by:

        H = (mu, nu)

    where:
        mu : membership degree
        nu : non-membership degree
    """

    mu: float
    nu: float

    def __post_init__(self):
        # Basic range validation
        if not (0.0 <= self.mu <= 1.0):
            raise ValueError(f"mu must be in [0, 1], got {self.mu}")

        if not (0.0 <= self.nu <= 1.0):
            raise ValueError(f"nu must be in [0, 1], got {self.nu}")

        # HyFS condition from Eq. (1)
        value = self.mu + self.nu - self.mu * self.nu

        if not (0.0 <= value <= 1.0 + 1e-12):
            raise ValueError(
                "Invalid HyFS: condition "
                "0 <= mu + nu - mu*nu <= 1 is not satisfied."
            )


def score(h: HyFS) -> float:
    """
    Compute the HyFS score function.

    Eq. (2):

        S(H) = (2*mu - mu*nu + 1) / (2*nu + 1)

    This function is used:
    1. to convert each expert HyFS judgment to a crisp score;
    2. later to defuzzify the aggregated criterion HyFS.
    """

    numerator = 2 * h.mu - h.mu * h.nu + 1
    denominator = 2 * h.nu + 1

    return numerator / denominator


def scalar_multiply(lam: float, h: HyFS) -> HyFS:
    """
    Scale one HyFS by scalar lambda.

    Reproduction form used by this project:

        lambda • H =
        (
            1 - (1 - mu)^lambda,
            nu^lambda
        )

    Important:
    The typesetting of Eq. (3) in the PDF is ambiguous.
    This expression is the form used in our reproduction because
    it reproduces the values reported in Table 5.

    lambda must be > 0.
    """

    if lam <= 0:
        raise ValueError(f"lambda must be > 0, got {lam}")

    new_mu = 1 - (1 - h.mu) ** lam
    new_nu = h.nu ** lam

    return HyFS(mu=new_mu, nu=new_nu)


def hyfwa(values: Sequence[HyFS], weights: Sequence[float]) -> HyFS:
    """
    Hyperbolic Fuzzy Weighted Averaging operator.

    Eq. (6):

        HyFWA =
        (
            1 - product((1 - mu_i)^w_i),
            product(nu_i^w_i)
        )

    Parameters
    ----------
    values:
        HyFS values to aggregate.

    weights:
        Weight of each expert.

    Returns
    -------
    HyFS
        Aggregated fuzzy value.
    """

    if len(values) == 0:
        raise ValueError("values must not be empty")

    if len(values) != len(weights):
        raise ValueError(
            "values and weights must have the same length"
        )

    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")

    weight_sum = sum(weights)

    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(
            f"weights must sum to 1, got {weight_sum}"
        )

    mu_product = 1.0
    nu_product = 1.0

    for h, w in zip(values, weights):
        mu_product *= (1 - h.mu) ** w
        nu_product *= h.nu ** w

    aggregated_mu = 1 - mu_product
    aggregated_nu = nu_product

    return HyFS(
        mu=aggregated_mu,
        nu=aggregated_nu
    )


def defuzzify(h: HyFS) -> float:
    """
    Convert an aggregated HyFS into a crisp value.

    The paper applies Eq. (2) again after HyFWA aggregation,
    therefore defuzzification is equivalent to score(H).
    """

    return score(h)