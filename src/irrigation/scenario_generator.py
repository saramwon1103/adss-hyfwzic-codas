import pandas as pd
from src.config import IRRIGATION_DATA
from .interfaces import WEATHER
SCENARIO_FILE=IRRIGATION_DATA/'scenarios.csv'
SCENARIOS=('Normal','Drought','Missing Sensor')

def real_scenario(base,name,start=None,duration=None):
    if name not in ['Normal','Drought','Missing Sensor']:raise ValueError('Unknown scenario.')
    settings=pd.read_csv(SCENARIO_FILE)
    if settings.scenario.duplicated().any() or set(settings.scenario)!=set(SCENARIOS):
        raise ValueError('scenarios.csv must contain each supported scenario exactly once.')
    row=settings.set_index('scenario').loc[name]
    start=int(row.start_step) if start is None else start
    duration=int(row.duration_days) if duration is None else duration
    if name=='Drought' and not 1<=start<len(base):raise ValueError('Drought must start inside evaluation period.')
    if name=='Missing Sensor' and (start<1 or duration<1 or start+duration>len(base)):
        raise ValueError('Stress interval must fit in evaluation period.')
    if row.rain_factor<0 or row.radiation_factor<0 or not 0<=row.water_budget_factor<=1:
        raise ValueError('Invalid scenario factors.')
    frame=base.copy(deep=True)
    frame['demand_event']=False
    frame['scenario_sensor_outage']=False
    if name=='Drought':
        frame.loc[start:,'temperature_c']+=float(row.temp_add_c)
        frame.loc[start:,'humidity_percent']=(frame.loc[start:,'humidity_percent']-float(row.rh_subtract_points)).clip(lower=0)
        frame.loc[start:,'radiation_mj_m2']*=float(row.radiation_factor)
        frame.loc[start:,'rain_mm']*=float(row.rain_factor)
        frame.loc[start:,'water_budget_liters']*=float(row.water_budget_factor)
        frame.loc[start,'demand_event']=True
    # Persistence forecast for d is ONLY scenario weather at d-1.
    for col in WEATHER:
        first=frame.loc[0,f'{col}_forecast']
        frame[f'{col}_forecast']=frame[col].shift(1)
        frame.loc[0,f'{col}_forecast']=first
    if name=='Missing Sensor':
        frame.loc[start:start+duration-1,'sensor_missing']=True
        frame.loc[start:start+duration-1,'scenario_sensor_outage']=True
    return frame
