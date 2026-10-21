"""Reference HyFS adapter (Quyên can replace this module with the same API)."""
import numpy as np


def validate_pairs(pairs):
    p = np.asarray(pairs, dtype=float)
    if p.shape[-1] != 2 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('HyFS pairs must be finite [..., 2] values within [0,1].')
    return p


def score(pairs):
    p = validate_pairs(pairs)
    mu, nu = p[..., 0], p[..., 1]
    return (2 * mu - mu * nu + 1) / (2 * nu + 1)


def scale(pairs, factor):
    """Eq.3: scalar multiplication is (1-(1-mu)**lambda, nu**lambda)."""
    p = validate_pairs(pairs)
    f = np.asarray(factor, dtype=float)
    if not np.isfinite(f).all() or np.any(f <= 0):
        raise ValueError('Scale factors must be positive and finite.')
    return np.stack([1 - (1 - p[..., 0]) ** f, p[..., 1] ** f], axis=-1)


def aggregate(pairs, expert_weights=None):
    p = validate_pairs(pairs)
    if p.ndim != 3 or p.shape[0] < 1:
        raise ValueError('Expected [experts, criteria, 2].')
    w = np.ones(p.shape[0]) / p.shape[0] if expert_weights is None else np.asarray(expert_weights, dtype=float)
    if w.shape != (p.shape[0],) or not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1):
        raise ValueError('Expert weights must be nonnegative and sum to 1.')
    return np.stack([1 - np.prod((1 - p[..., 0]) ** w[:, None], axis=0),
                     np.prod(p[..., 1] ** w[:, None], axis=0)], axis=-1)
