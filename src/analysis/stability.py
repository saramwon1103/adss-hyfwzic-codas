"""Tie-aware rank correlation, labels are aligned before comparison."""
import numpy as np
import pandas as pd


def rank_correlation(first, second, already_ranked=False):
    a,b=pd.Series(first),pd.Series(second)
    if not a.index.is_unique or not b.index.is_unique or set(a.index)!=set(b.index):
        raise ValueError('Rank stability requires unique matching alternative labels.')
    b=b.reindex(a.index)
    if not already_ranked:
        a=a.round(10).rank(ascending=False,method='average')
        b=b.round(10).rank(ascending=False,method='average')
    x,y=a.to_numpy(dtype=float),b.to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Rank stability inputs must be finite.')
    if np.std(x)==0 or np.std(y)==0:
        return 1.0 if np.allclose(x,y) else None
    return float(np.corrcoef(x,y)[0,1])
