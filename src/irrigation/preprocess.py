"""Offline adapter: uploaded 10-minute measurements to daily real-data input."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from src.config import IRRIGATION_RAW,IRRIGATION_PROCESSED
from .interfaces import WEATHER,MODEL_FEATURES,FORBIDDEN_FEATURES
from .soil_response import SoilResponseModel,fit_response_model
SOURCE_DIR = IRRIGATION_RAW
DEFAULT_START='2025-06-01'
DEFAULT_END='2025-07-13'

def load_daily_sector(sector=1, source_dir=None):
    if sector not in (1,2):
        raise ValueError('Real Mode 2 supports open-field tomato sectors 1 and 2 only.')
    source_dir = Path(source_dir) if source_dir is not None else SOURCE_DIR
    path = source_dir / f'sector_{sector}_10min.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing real source CSV: {path}. No mock fallback is used.')
    # Explicit allowlist: never load precomputed 24h targets into model/policy features.
    columns = ['ts','soil_moisture','liters_total','irrigation_duration_minutes',
               'weather_temp','weather_humidity','weather_rain','weather_radiation']
    raw = pd.read_csv(path, usecols=columns)
    raw['ts'] = pd.to_datetime(raw.ts, errors='raise')
    raw = raw.sort_values('ts').reset_index(drop=True)
    if raw.ts.isna().any() or raw.ts.duplicated().any():
        raise ValueError('Source timestamps must be valid and unique.')
    if not raw.ts.diff().dropna().eq(pd.Timedelta(minutes=10)).all():
        raise ValueError('Expected the supplied complete 10-minute grid; gaps need explicit handling.')
    for col in columns[1:]:
        raw[col] = pd.to_numeric(raw[col], errors='raise')
    weather_cols = ['weather_temp','weather_humidity','weather_rain','weather_radiation']
    if not np.isfinite(raw[weather_cols+['liters_total','irrigation_duration_minutes']]).all().all():
        raise ValueError('Weather/actuator forcing cannot contain missing or infinite values.')
    if not raw.weather_humidity.between(0,100).all() or (raw[['weather_rain','weather_radiation','liters_total']] < 0).any().any():
        raise ValueError('Invalid weather or water values.')
    if not raw.irrigation_duration_minutes.between(0,10).all():
        raise ValueError('Valve duration must be within its 10-minute bin.')
    flow_error = (raw.liters_total - 1.1 * raw.irrigation_duration_minutes).abs().max()
    if flow_error > 1e-6:
        raise ValueError('Sector 1/2 water volume does not match documented 1.1 L/min nominal flow.')
    valid = raw.soil_moisture.between(0,100)
    raw['valid_soil'] = raw.soil_moisture.where(valid)
    daily = raw.set_index('ts').resample('D').agg(
        temperature_c=('weather_temp','mean'),
        humidity_percent=('weather_humidity','mean'),
        rain_mm=('weather_rain',lambda x:float(x.sum()/6)),
        radiation_mj_m2=('weather_radiation',lambda x:float(x.sum()*600/1e6)),
        historical_irrigation_liters=('liters_total','sum'),
        irrigation_duration_minutes=('irrigation_duration_minutes','sum'),
        sensor_valid_count=('valid_soil','count'),
        interval_count=('weather_temp','size'),
        historical_daily_median_index=('valid_soil','median')).reset_index().rename(columns={'ts':'timestamp'})
    # Boundaries extend one day for y(d+1); incomplete final weather day excluded below.
    boundary = pd.DataFrame({'timestamp':pd.date_range(raw.ts.min().normalize(),raw.ts.max().normalize()+pd.Timedelta(days=1),freq='D')})
    observed = raw.loc[valid,['ts','valid_soil']].rename(columns={'ts':'sensor_observed_at','valid_soil':'soil_moisture_index'})
    boundary = pd.merge_asof(boundary,observed,left_on='timestamp',right_on='sensor_observed_at',
                             direction='backward',tolerance=pd.Timedelta(hours=12))
    boundary['sensor_age_hours'] = (boundary.timestamp-boundary.sensor_observed_at).dt.total_seconds()/3600
    boundary['next_soil_moisture_index'] = boundary.soil_moisture_index.shift(-1)
    daily = daily.merge(boundary,on='timestamp',how='left',validate='one_to_one')
    daily['sector_id'] = f'ARNESANO_TOMATO_{sector}'
    daily['sensor_missing'] = daily.soil_moisture_index.isna()
    # Persistence forecasts are available at the beginning of d and only use day d-1.
    for col in WEATHER:
        daily[f'{col}_forecast'] = daily[col].shift(1)
    daily['is_synthetic'] = False
    daily['sensor_valid_fraction'] = daily.sensor_valid_count / daily.interval_count
    daily['source_kind'] = 'uploaded_field_sensor_actuator_and_reanalysis_weather'
    # No filling across missing soil boundaries; retain missingness and actual coverage.
    daily = daily.loc[daily.interval_count == 144].reset_index(drop=True)
    report = {'sector':sector,'crop':'open-field tomato','source_file':path.name,
              'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'source_rows':len(raw),'source_start':str(raw.ts.min()),'source_end':str(raw.ts.max()),
              'daily_rows':len(daily),'sensor_raw_missing_rows':int(raw.soil_moisture.isna().sum()),
              'sensor_out_of_range_rows':int((raw.soil_moisture.notna() & ~valid).sum()),
              'valid_sensor_rows':int(valid.sum()),'daily_missing_boundary_count':int(daily.sensor_missing.sum()),
              'nominal_flow_liters_per_minute':1.1,'flow_consistency_max_error_liters':float(flow_error),
              'soil_unit':'relative probe index, 0..100; NOT calibrated volumetric water content',
              'water_unit':'litres reconstructed from nominal flow × valve duration, NOT flow-meter readings',
              'rain_method':'sum(interpolated hourly rain values × 10/60); approximate rate integration, not raw sum',
              'radiation_method':'sum(direct radiation W/m2 × 600 s) / 1e6; direct-radiation proxy, not total solar',
              'timezone':'source naive Europe/Rome wall-clock dates preserved; no fabricated UTC precision',
              'soil_boundary_method':'latest valid 0..100 observation at/before local midnight, maximum age 12h; no backward fill',
              'excluded_features':sorted(FORBIDDEN_FEATURES),
              'weather_origin':'Open-Meteo historical ERA5 reanalysis already in uploaded merged CSVs',
              'forecasts':'one-day persistence, not archived operational forecasts',
              'quality_note':'Range check only; in-range sensor spikes/flatlines may remain. No centred smoothing with future observations.'}
    return daily, report

def prepare_real_run(sector=1,start=DEFAULT_START,end=DEFAULT_END,source_dir=None):
    daily, quality = load_daily_sector(sector,source_dir)
    start,end = pd.Timestamp(start).normalize(),pd.Timestamp(end).normalize()
    evaluation = daily.loc[daily.timestamp.between(start,end)].copy().reset_index(drop=True)
    if len(evaluation)<21 or evaluation.timestamp.iloc[0]!=start or evaluation.timestamp.iloc[-1]!=end:
        raise ValueError('Choose a complete evaluation period of >=21 days inside the source coverage.')
    if not evaluation.timestamp.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError('Evaluation dates contain incomplete weather days; select another window.')
    if pd.isna(evaluation.soil_moisture_index.iloc[0]):
        raise ValueError('First evaluation day has no valid prior sensor reading within 12h; choose another start date.')
    if not np.isfinite(evaluation[[f'{c}_forecast' for c in WEATHER]].to_numpy()).all():
        raise ValueError('Persistence forecasts need a complete previous day.')
    model = fit_response_model(daily,start)
    evaluation['water_budget_liters'] = model.training_report['water_budget_liters']
    evaluation['demand_event'] = False
    evaluation['irrigation_liters'] = evaluation.historical_irrigation_liters
    cfg = {'target_index':50.,'lower_index':40.,'upper_index':60.,'initial_index':float(evaluation.soil_moisture_index.iloc[0]),
           'response_window_days':7,'robustness_tolerance_points':10.}
    # Teacher-forced one-step replay uses actual historical actions solely for holdout validation.
    validation = []
    for _,row in evaluation.dropna(subset=['soil_moisture_index','next_soil_moisture_index']).iterrows():
        predicted = model.predict(row.soil_moisture_index,row.historical_irrigation_liters,row)
        validation.append({'timestamp':row.timestamp,'observed_start_index':row.soil_moisture_index,
                           'historical_irrigation_liters':row.historical_irrigation_liters,
                           'observed_next_index':row.next_soil_moisture_index,'predicted_next_index':predicted,
                           'persistence_next_index':row.soil_moisture_index})
    validation = pd.DataFrame(validation)
    if validation.empty:raise ValueError('No valid holdout transitions for model validation.')
    error = validation.predicted_next_index-validation.observed_next_index
    persistence_error = validation.persistence_next_index-validation.observed_next_index
    quality.update({'evaluation_start':str(start.date()),'evaluation_end':str(end.date()),'evaluation_days':len(evaluation),
                    'evaluation_missing_sensor_days':int(evaluation.sensor_missing.sum()),
                    'holdout_pairs':len(validation),'holdout_mae_points':float(error.abs().mean()),
                    'holdout_rmse_points':float(np.sqrt(np.square(error).mean())),
                    'persistence_mae_points':float(persistence_error.abs().mean()),
                    'model_beats_persistence_mae':bool(error.abs().mean()<persistence_error.abs().mean()),
                    'evaluation_historical_water_liters':float(evaluation.historical_irrigation_liters.sum()),
                    'counterfactual_results':'modelled outcomes under proposed actions, not observed trial outcomes for these policies',
                    'expert_weights':'illustrative preferences retained; dataset contains no expert judgments'})
    return evaluation,model,cfg,quality,validation,daily

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sector',type=int,choices=[1,2],default=1)
    p.add_argument('--output',type=Path,default=IRRIGATION_PROCESSED/'processed_irrigation.csv')
    args=p.parse_args()
    data,quality=load_daily_sector(args.sector)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    data.to_csv(args.output,index=False)
    args.output.with_suffix('.quality.json').write_text(json.dumps(quality,indent=2),encoding='utf-8')
    print(f'Prepared {len(data)} daily rows for sector {args.sector}: {args.output}')

if __name__=='__main__':main()
