from dataclasses import fields
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from src.irrigation.preprocess import (load_daily_sector,prepare_real_run,fit_response_model,
                           SOURCE_DIR,WEATHER,FORBIDDEN_FEATURES)
from src.irrigation.run_irrigation import (run_real_mode2,real_scenario,simulate_real,REAL_STRATEGIES,
                                 RealStrategyInput,RealStrategyOutput,real_kpis)
from src.pipelines import run_mode2,run_mode1


@pytest.fixture(scope='module')
def real():return prepare_real_run(1)


@pytest.mark.parametrize('sector',[1,2])
def test_source_provenance_and_litres(sector):
    daily,q=load_daily_sector(sector)
    assert q['source_rows']==33259
    assert not daily.is_synthetic.any()
    assert q['flow_consistency_max_error_liters']<1e-6
    assert daily.sector_id.nunique()==1
    assert daily.sector_id.iloc[0]==f'ARNESANO_TOMATO_{sector}'
    assert daily.timestamp.diff().dropna().eq(pd.Timedelta(days=1)).all()
    valid=daily.sensor_age_hours.dropna()
    assert valid.between(0,12).all()
    assert (daily.loc[daily.soil_moisture_index.notna(),'sensor_observed_at']<=daily.loc[daily.soil_moisture_index.notna(),'timestamp']).all()


def test_uploaded_bytes_not_rewritten():
    meta=json.loads((SOURCE_DIR/'provenance.json').read_text())
    for item in meta['files']:
        assert hashlib.sha256((SOURCE_DIR/item['local_name']).read_bytes()).hexdigest()==item['sha256']


def test_source_missing_is_not_replaced_by_mock(tmp_path):
    with pytest.raises(FileNotFoundError,match='No mock fallback'):
        load_daily_sector(1,tmp_path)


def test_hand_computed_10minute_aggregation_and_no_future_fill(tmp_path):
    ts=pd.date_range('2025-05-01',periods=144*3,freq='10min')
    raw=pd.DataFrame({'ts':ts,'soil_moisture':50.,'liters_total':2.2,
                      'irrigation_duration_minutes':2.,'weather_temp':20.,'weather_humidity':60.,
                      'weather_rain':1.,'weather_radiation':100.})
    raw.loc[0,'soil_moisture']=np.nan
    raw.to_csv(tmp_path/'sector_1_10min.csv',index=False)
    d,_=load_daily_sector(1,tmp_path)
    assert d.loc[0,'rain_mm']==pytest.approx(24.) # hourly rate held at 1 mm/h for 24 hours
    assert d.loc[0,'radiation_mj_m2']==pytest.approx(8.64)
    assert d.loc[0,'historical_irrigation_liters']==pytest.approx(316.8)
    assert pd.isna(d.loc[0,'soil_moisture_index']) # 00:10 observation cannot fill midnight
    assert d.loc[1,'temperature_c_forecast']==20.


def test_training_strictly_precedes_holdout(real):
    _,model,_,q,_,_=real
    assert model.training_report['training_rows']==87
    assert pd.Timestamp(model.training_report['training_last_target_day'])<pd.Timestamp(q['evaluation_start'])
    assert model.irrigation_gain>=0
    assert 0<=model.coefficients_raw[0]<=1


def test_future_targets_and_weather_do_not_change_training(real):
    _,model,_,_,_,daily=real
    altered=daily.copy()
    future=altered.timestamp>='2025-06-01'
    altered.loc[future,'next_soil_moisture_index']=99.
    altered.loc[future,'historical_irrigation_liters']=99999.
    altered.loc[future,WEATHER]=999.
    new=fit_response_model(altered,'2025-06-01')
    np.testing.assert_allclose(new.feature_mean,model.feature_mean)
    np.testing.assert_allclose(new.coefficients_standardized,model.coefficients_standardized)
    assert new.training_report['water_budget_liters']==model.training_report['water_budget_liters']


def test_default_is_real_no_random_weather():
    r=run_mode2()
    assert not r['is_synthetic']
    assert r['outcomes_modeled']
    assert len(r['dataset'])==43
    assert 'et0_mm' not in r['dataset']
    assert all(col not in r['dataset'] for col in FORBIDDEN_FEATURES)
    assert r['model'].feature_names==['soil_moisture_index','irrigation_liters',*WEATHER]
    assert 'water_use_liters' in r['matrix']
    assert 'water_use_mm' not in r['matrix']


@pytest.mark.parametrize('sector',[1,2])
def test_holdout_validation_and_persistence_baseline(sector):
    _,model,cfg,q,v,_=prepare_real_run(sector)
    mae=float((v.predicted_next_index-v.observed_next_index).abs().mean())
    assert q['holdout_mae_points']==pytest.approx(mae)
    assert q['holdout_pairs']==len(v)
    assert q['model_beats_persistence_mae']==(q['holdout_mae_points']<q['persistence_mae_points'])
    if sector==2:assert not q['model_beats_persistence_mae'] # visible negative finding, not hidden


def test_no_privileged_eval_fields_in_policy_interface():
    names={x.name for x in fields(RealStrategyInput)}
    assert names.isdisjoint({'historical_irrigation_liters','soil_moisture_index',
                             'next_soil_moisture_index','actual_weather','demand_event'})
    assert 'weather_forecast' in names
    assert 'water_budget_liters' in names


def test_persistence_forecast_and_drought_no_lookahead(real):
    base,_,_,_,_,_=real
    d=real_scenario(base,'Drought')
    assert d.loc[10,'temperature_c']==pytest.approx(base.loc[10,'temperature_c']+4)
    assert d.loc[10,'temperature_c_forecast']==pytest.approx(base.loc[9,'temperature_c'])
    assert d.loc[11,'temperature_c_forecast']==pytest.approx(d.loc[10,'temperature_c'])
    assert d.loc[10,'water_budget_liters']==pytest.approx(base.loc[10,'water_budget_liters']*.6)
    pd.testing.assert_frame_equal(base,real[0])


@pytest.mark.parametrize('scenario',['Normal','Drought','Missing Sensor'])
def test_real_runs_deterministic_finite_with_budgets(scenario):
    a=run_mode2(scenario);b=run_mode2(scenario)
    pd.testing.assert_frame_equal(a['matrix'],b['matrix'])
    assert a['matrix'].shape==(3,5)
    assert np.isfinite(a['matrix'].to_numpy()).all()
    assert a['matrix'].missing_data_robustness.between(0,1).all()
    assert len(a['sensitivity'])==15
    for name,t in a['traces'].items():
        assert t.moisture_end_index.between(0,100).all()
        assert t.irrigation_liters.between(0,t.water_budget_liters+1e-9).all()
        assert t.used_fallback.tolist()==t.sensor_missing.tolist()
    if scenario=='Missing Sensor':assert a['dataset'].scenario_sensor_outage.sum()==9


def test_new_policy_does_not_replay_observed_test_soil(real):
    base,model,cfg,_,_,_=real
    frame=real_scenario(base,'Normal')
    factory=REAL_STRATEGIES['FDSS-like']
    a=simulate_real(frame,model,cfg,factory)
    frame['soil_moisture_index']=99.
    frame['next_soil_moisture_index']=1.
    frame['historical_irrigation_liters']=9999.
    frame['irrigation_liters']=9999.
    b=simulate_real(frame,model,cfg,factory)
    pd.testing.assert_frame_equal(a,b)


def test_real_action_caps_enforced(real):
    base,model,cfg,_,_,_=real
    class Bad:
        name='bad'
        def decide(self,item):return RealStrategyOutput(item.water_budget_liters+1,False,'bad')
    with pytest.raises(ValueError):simulate_real(base,model,cfg,Bad)


@pytest.mark.parametrize('case',['unsupported_sector','early_training','short_window','inverted_window'])
def test_invalid_real_selection_rejected(case):
    with pytest.raises(ValueError):
        if case=='unsupported_sector':prepare_real_run(3)
        if case=='early_training':prepare_real_run(1,'2025-02-17','2025-03-17')
        if case=='short_window':prepare_real_run(1,'2025-06-01','2025-06-07')
        if case=='inverted_window':prepare_real_run(1,'2025-07-01','2025-06-01')


def test_legacy_schema_needs_explicit_mock_selection():
    with pytest.raises(ValueError,match='Legacy'):
        run_mode2(data=pd.DataFrame({'et0_mm':[1.]}))


def test_mode1_is_unchanged():
    r=run_mode1()
    assert r['comparison'].score_within_tolerance.all()
    assert r['sensitivity_comparison'].score_within_tolerance.all()
    assert r['sensitivity_comparison'].rank_match.all()


def test_scenario_csv_controls_transformation(tmp_path,monkeypatch):
    from src.irrigation import scenario_generator as engine
    base,_,_,_,_,_=prepare_real_run(1)
    params=pd.read_csv(engine.SCENARIO_FILE)
    params.loc[params.scenario=='Drought','temp_add_c']=2.
    file=tmp_path/'scenarios.csv';params.to_csv(file,index=False)
    monkeypatch.setattr(engine,'SCENARIO_FILE',file)
    scenario=engine.real_scenario(base,'Drought')
    assert scenario.loc[10,'temperature_c']==pytest.approx(base.loc[10,'temperature_c']+2.)
