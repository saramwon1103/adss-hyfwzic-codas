import numpy as np
import pandas as pd
import pytest
from src.core.hy_fwzic import compute_weights
from src.core.fuzzy import score,scale,aggregate
from src.pipelines import run_mode1,load_mode1


def test_table5_all_weights():
    w=run_mode1()['weight_comparison']
    assert (w.abs_error<1e-8).all()
    assert w.loc[w.criterion=='C5','weight'].iloc[0]==pytest.approx(.142287487,abs=1e-9)
    assert w.weight.sum()==pytest.approx(1)


def test_scalar_operation_by_hand():
    np.testing.assert_allclose(scale([[.75,.25]],.5),[[.5,.5]])
    assert score([[.5,.5]])[0]==pytest.approx(.875)


def test_single_expert_aggregate_identity():
    x=np.array([[[.75,.35],[.55,.55]]])
    np.testing.assert_allclose(aggregate(x),x[0])


def test_computed_pipeline_matches_published():
    a=run_mode1(source='published')['result'].ranking
    b=run_mode1(source='computed')['result'].ranking
    np.testing.assert_allclose(a.H,b.H,atol=1e-7)
    assert a['rank'].tolist()==b['rank'].tolist()


def test_invalid_judgment():
    d=load_mode1();d['judgments'].iloc[0,0]=8
    with pytest.raises(ValueError):compute_weights(d['judgments'],d['linguistic_mapping'])
