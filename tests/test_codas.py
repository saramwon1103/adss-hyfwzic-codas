import numpy as np
import pandas as pd
import pytest
from src.core.codas import hyperbolic_codas, crisp_codas, assess_distances
from src.pipelines import load_mode1, run_mode1, ROOT


@pytest.fixture
def paper():
    return load_mode1()


def test_distances_against_table7(paper):
    run=run_mode1()
    actual=run['result'].ranking.set_index('alternative')
    expected=pd.read_csv(ROOT/'data/paper/expected/expected_distances.csv',index_col='alternative')
    assert np.allclose(actual.E.reindex(expected.index),expected.expected_E,atol=5.1e-7,rtol=0)
    assert np.allclose(actual['T'].reindex(expected.index),expected.expected_T,atol=5.1e-7,rtol=0)


def test_table8_all_scores_and_ties():
    run=run_mode1()
    assert run['comparison'].score_within_tolerance.all()
    assert run['comparison'].rank_match.all()
    r=run['result'].ranking.set_index('alternative')
    assert r.loc['A8','H']==pytest.approx(3.843744,abs=1e-6)
    assert r.loc['A11','H']==pytest.approx(-3.519666,abs=1e-6)
    assert r.loc['A1','H']==r.loc['A2','H']
    assert r.loc['A6','H']==r.loc['A10','H']


def test_relative_antisymmetry_and_zero_sum():
    run=run_mode1()
    r=run['result'].relative.to_numpy()
    assert np.allclose(r,-r.T)
    assert np.allclose(np.diag(r),0)
    assert abs(run['result'].ranking.H.sum())<1e-12


def test_paper_close_threshold_boundary():
    # Hand-computable independent fixture: de=0.01 included; de=0.1 excluded.
    E=np.array([0.,.01,.1]);T=np.array([0.,.02,.2])
    _,r=assess_distances(E,T,['x','y','z'],tau=.01)
    assert r.loc['y','x']==pytest.approx(.03)
    assert r.loc['z','x']==pytest.approx(.1)


def test_weights_align_by_name(paper):
    c=paper['criteria']
    a=hyperbolic_codas(paper['matrix'],c.weight,c.direction,paper['mapping'])
    b=hyperbolic_codas(paper['matrix'],c.weight.iloc[::-1],c.direction.iloc[::-1],paper['mapping'])
    np.testing.assert_allclose(a.ranking.H,b.ranking.H)


@pytest.mark.parametrize('fault',['nan','negative','weight_labels','weight_sum','bad_direction','rating','duplicate'])
def test_bad_inputs_rejected(paper,fault):
    m=paper['matrix'].copy();w=paper['criteria'].weight.copy();di=paper['criteria'].direction.copy()
    if fault=='nan':m.iloc[0,0]=np.nan
    if fault=='negative':m.iloc[0,0]=-1
    if fault=='weight_labels':w=w.rename(index={'C1':'X'})
    if fault=='weight_sum':w*=2
    if fault=='bad_direction':di.iloc[0]='other'
    if fault=='rating':m.iloc[0,0]=4
    if fault=='duplicate':m.index=['same']*len(m)
    with pytest.raises(ValueError):hyperbolic_codas(m,w,di,paper['mapping'])


def test_crisp_zero_cost_and_constant_column():
    matrix=pd.DataFrame({'cost':[0.,10.], 'constant':[2.,2.]},index=['good','bad'])
    w=pd.Series({'cost':.5,'constant':.5});di=pd.Series({'cost':'cost','constant':'benefit'})
    r=crisp_codas(matrix,w,di)
    assert r.ranking.iloc[0].alternative=='good'
    np.testing.assert_allclose(r.normalized,[[1,1],[0,1]])
    assert np.isfinite(r.ranking.H).all()


def test_crisp_unit_scale_invariance():
    m=pd.DataFrame({'cost':[0.,4.,10.],'benefit':[5.,1.,4.]},index=['a','b','c'])
    w=pd.Series({'cost':.6,'benefit':.4});di=pd.Series({'cost':'cost','benefit':'benefit'})
    a=crisp_codas(m,w,di).ranking
    m['cost']*=1000
    b=crisp_codas(m,w,di).ranking
    pd.testing.assert_frame_equal(a,b)


def test_one_alternative_is_valid():
    m=pd.DataFrame({'x':[0.]},index=['a'])
    r=crisp_codas(m,pd.Series({'x':1.}),pd.Series({'x':'cost'}))
    assert r.ranking.iloc[0].H==0
    assert r.ranking.iloc[0]['rank']==1
