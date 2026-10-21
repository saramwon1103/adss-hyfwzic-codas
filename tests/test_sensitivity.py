from src.pipelines import run_mode1


def test_all_table10_values_and_ranks():
    run=run_mode1();compare=run['sensitivity_comparison']
    assert len(compare)==104
    assert compare.score_within_tolerance.all()
    assert compare.rank_match.all()


def test_winners_s1_a9_s6_a5():
    summary=run_mode1()['sensitivity_summary'].set_index('scenario')
    assert summary.loc['S1','winner']=='A9'
    assert summary.loc['S6','winner']=='A5'
    assert summary.spearman.between(-1,1).all()


def test_equal_performance_columns_c2_c3():
    d=run_mode1()['sensitivity']
    s2=d[d.scenario=='S2'].set_index('alternative').H
    s3=d[d.scenario=='S3'].set_index('alternative').H
    assert (s2-s3).abs().max()<1e-12
