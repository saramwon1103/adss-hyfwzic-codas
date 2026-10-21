"""Hyperbolic CODAS for paper reproduction; crisp extension for operational KPIs.

Paper mode deliberately follows Eqs.8-17 including the *close* threshold and
sum of per-criterion pair distances in Eq.12, which differ from some CODAS code.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .fuzzy import scale


@dataclass
class CodasResult:
    ranking: pd.DataFrame
    normalized: np.ndarray
    weighted: np.ndarray
    nis: np.ndarray
    relative: pd.DataFrame


def validate_inputs(matrix, weights, directions):
    if matrix.empty or matrix.columns.has_duplicates or matrix.index.has_duplicates:
        raise ValueError('Matrix must have unique labels and at least one row/column.')
    x = matrix.to_numpy(dtype=float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Performance values must be finite and nonnegative.')
    if set(weights.index) != set(matrix.columns) or weights.index.has_duplicates:
        raise ValueError('Weight criterion labels must exactly match matrix columns.')
    if set(directions.index) != set(matrix.columns) or directions.index.has_duplicates:
        raise ValueError('Direction labels must exactly match matrix columns.')
    w = weights.reindex(matrix.columns).to_numpy(dtype=float)
    direct = directions.reindex(matrix.columns).to_numpy()
    if not np.isfinite(w).all() or np.any(w <= 0) or not np.isclose(w.sum(), 1, atol=1e-8, rtol=0):
        raise ValueError('Criterion weights must be positive, finite and sum to 1.')
    if not np.isin(direct, ['benefit', 'cost']).all():
        raise ValueError('Each direction must be benefit or cost.')
    return x, w, direct


def assess_distances(E, T, labels, tau=.02, threshold_rule='close'):
    if not np.isfinite(tau) or tau < 0:
        raise ValueError('tau must be finite and nonnegative.')
    de = E[:, None] - E[None, :]
    dt = T[:, None] - T[None, :]
    if threshold_rule == 'close':
        gate = np.abs(de) <= tau  # exact Eq.16 from the supplied paper
    elif threshold_rule == 'far':
        gate = np.abs(de) >= tau  # alternate conventional implementation
    else:
        raise ValueError('threshold_rule must be close or far.')
    rel = de + gate * dt
    H = rel.sum(axis=1)
    ranks = pd.Series(np.round(H, 10)).rank(ascending=False, method='min').astype(int).to_numpy()
    table = pd.DataFrame({'alternative': list(labels), 'E': E, 'T': T, 'H': H, 'rank': ranks})
    # Stable sort preserves original order among mathematically equal alternatives.
    return table.sort_values('H', ascending=False, kind='stable').reset_index(drop=True), pd.DataFrame(rel, index=labels, columns=labels)


def hyperbolic_codas(matrix, weights, directions, mapping, tau=.02, threshold_rule='close'):
    x, w, direct = validate_inputs(matrix, weights, directions)
    if mapping['rating'].duplicated().any():
        raise ValueError('Duplicate performance mapping keys.')
    lookup = mapping.set_index('rating')[['mu', 'nu']]
    if not np.isin(x, lookup.index).all():
        raise ValueError('Paper mode accepts only declared ratings 1/2/3.')
    p = lookup.loc[x.ravel()].to_numpy().reshape(*x.shape, 2)
    normalized = p.copy()
    normalized[:, direct == 'cost', :] = p[:, direct == 'cost', ::-1]
    weighted = scale(normalized, w)
    nis = np.stack([weighted[..., 0].min(0), weighted[..., 1].max(0)], axis=-1)
    delta = weighted - nis
    E = np.sqrt((delta ** 2).sum(axis=-1) / 2).sum(axis=1)
    T = np.abs(delta).sum(axis=-1).sum(axis=1) / 2
    ranking, relative = assess_distances(E, T, matrix.index, tau, threshold_rule)
    return CodasResult(ranking, normalized, weighted, nis, relative)


def crisp_codas(matrix, weights, directions, tau=.02, threshold_rule='close'):
    """Mode2 extension: zero-safe min-max; a constant column contributes no distance."""
    x, w, direct = validate_inputs(matrix, weights, directions)
    lo, hi = x.min(axis=0), x.max(axis=0)
    span = hi - lo
    normalized = np.divide(x - lo, span, out=np.zeros_like(x), where=span > 0)
    normalized[:, direct == 'cost'] = 1 - normalized[:, direct == 'cost']
    normalized[:, span == 0] = 1  # common value, no discrimination
    weighted = normalized * w
    nis = weighted.min(axis=0)
    delta = weighted - nis
    E = np.linalg.norm(delta, axis=1)
    T = np.abs(delta).sum(axis=1)
    ranking, relative = assess_distances(E, T, matrix.index, tau, threshold_rule)
    return CodasResult(ranking, normalized, weighted, nis, relative)
