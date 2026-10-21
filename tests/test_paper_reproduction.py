from src.paper.reproduce_paper import run_mode1
from src.paper.load_paper_data import load_mode1
from src.analysis.stability import rank_correlation


def test_expected_output_directory_matches_scaffold():
    from src.config import PAPER_EXPECTED
    assert (PAPER_EXPECTED/'expected_ranking.csv').is_file()
    assert (PAPER_EXPECTED/'expected_sensitivity.csv').is_file()
    assert (PAPER_EXPECTED/'expected_weights.csv').is_file()


def test_both_modes_paper_validation():
    for source in ['published','computed']:
        run=run_mode1(source=source)
        assert run['comparison'].score_within_tolerance.sum()==13
        assert run['sensitivity_comparison'].score_within_tolerance.sum()==104


def test_stability_aligns_labels():
    import pandas as pd
    x=pd.Series({'a':3.,'b':1.,'c':2.})
    assert rank_correlation(x,x.iloc[::-1])==1.
