"""Comparison of calculated scores with immutable published reference values."""
import pandas as pd
from src.config import PAPER_EXPECTED


def compare_results(result, sensitivity, computed_weights, criteria):
    expected=pd.read_csv(PAPER_EXPECTED/'expected_ranking.csv')
    comparison=result.ranking.merge(expected,on='alternative',validate='one_to_one')
    comparison['score_error']=comparison.H-comparison.expected_score
    comparison['score_within_tolerance']=comparison.score_error.abs()<=.0011
    comparison['rank_match']=comparison['rank']==comparison.expected_rank_ties_min
    sx=pd.read_csv(PAPER_EXPECTED/'expected_sensitivity.csv')
    sc=sensitivity.merge(sx,on=['scenario','alternative'],validate='one_to_one')
    sc['score_error']=sc.H-sc.expected_score
    sc['score_within_tolerance']=sc.score_error.abs()<=.0011
    sc['rank_match']=sc['rank']==sc.expected_rank
    wc=computed_weights.merge(criteria[['weight']].rename(columns={'weight':'expected_weight'}),
                              left_on='criterion',right_index=True,validate='one_to_one')
    wc['abs_error']=(wc.weight-wc.expected_weight).abs()
    return comparison,sc,wc
