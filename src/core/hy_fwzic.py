"""Minimal formula-based reference implementation for the shared integration."""
import numpy as np
import pandas as pd
from .fuzzy import score, scale, aggregate


def compute_weights(judgments: pd.DataFrame, mapping: pd.DataFrame, expert_weights=None):
    """Input numeric Likert expert×criterion DataFrame; output criterion, weight.

    Mapping is keyed by numerical_score. Labels and column order are preserved.
    Scaling uses the sum of HyFS scores per expert, not the sum of Likert integers.
    """
    if judgments.empty or judgments.columns.has_duplicates or judgments.index.has_duplicates:
        raise ValueError('Judgments must be nonempty with unique expert and criterion labels.')
    if mapping['numerical_score'].duplicated().any():
        raise ValueError('Duplicate linguistic mapping keys.')
    lookup = mapping.set_index('numerical_score')[['mu', 'nu']]
    if not np.isin(judgments.to_numpy(), lookup.index).all():
        raise ValueError('Judgments must use the declared numerical Likert scale.')
    pairs = lookup.loc[judgments.to_numpy().ravel()].to_numpy().reshape(*judgments.shape, 2)
    scaled = scale(pairs, 1 / score(pairs).sum(axis=1)[:, None])
    avg = aggregate(scaled, expert_weights)
    crisp = score(avg)
    return pd.DataFrame({'criterion': judgments.columns, 'mu': avg[:, 0], 'nu': avg[:, 1],
                         'crisp': crisp, 'weight': crisp / crisp.sum()})
