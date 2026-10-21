"""One-criterion emphasis, rank movement and tie-aware Spearman correlation."""
import numpy as np
import pandas as pd
from .stability import rank_correlation


def weight_scenarios(weights, high=.3):
    m = len(weights)
    if m < 2 or not (0 < high < 1):
        raise ValueError('Need >=2 criteria and 0<high<1.')
    low = (1 - high) / (m - 1)
    return pd.DataFrame([[high if j == i else low for j in range(m)] for i in range(m)],
                        index=[f'S{i+1}' for i in range(m)], columns=weights.index)


def analyze(rank_function, weights, scenarios):
    if not scenarios.index.is_unique or set(scenarios.columns) != set(weights.index):
        raise ValueError('Sensitivity scenarios require unique IDs and exact criterion labels.')
    base = rank_function(weights).ranking.set_index('alternative')
    tables, summary = [], []
    base_ranks = base['H'].round(10).rank(ascending=False, method='average')
    for sid, row in scenarios.iterrows():
        ranked = rank_function(row).ranking.copy()
        ranked['scenario'] = sid
        ranked['rank_change'] = ranked['alternative'].map(base['rank']) - ranked['rank']
        ranks = ranked.set_index('alternative')['H'].round(10).rank(ascending=False, method='average').reindex(base.index)
        rho = rank_correlation(base_ranks,ranks,already_ranked=True)
        winners = ranked.loc[ranked['rank'] == 1, 'alternative'].tolist()
        summary.append({'scenario': sid, 'winner': ', '.join(winners), 'spearman': rho,
                        'max_rank_movement': int(ranked['rank_change'].abs().max())})
        tables.append(ranked)
    if not tables:
        raise ValueError('At least one sensitivity scenario is required.')
    return pd.concat(tables, ignore_index=True), pd.DataFrame(summary)
