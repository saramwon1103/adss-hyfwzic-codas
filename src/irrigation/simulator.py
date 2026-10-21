import numpy as np
import pandas as pd
from .interfaces import WEATHER,RealStrategyInput

def simulate_real(frame,model,cfg,factory):
    if frame.empty:raise ValueError('Simulation frame cannot be empty.')
    strategy=factory();theta=cfg['initial_index'];rows=[]
    for step,row in frame.reset_index(drop=True).iterrows():
        observation=None if bool(row.sensor_missing) else float(theta)
        forecast={c:float(row[f'{c}_forecast']) for c in WEATHER}
        item=RealStrategyInput(step,observation,forecast,float(row.water_budget_liters),dict(cfg),model)
        action=strategy.decide(item);action.validate(item.water_budget_liters)
        actual_weather={c:float(row[c]) for c in WEATHER}
        predicted_without=model.predict(theta,0.,actual_weather)
        end=model.predict(theta,action.irrigation_liters,actual_weather)
        gain=model.irrigation_gain
        safe_water=max(0.,(cfg['upper_index']-predicted_without)/gain) if gain>1e-8 else float('inf')
        excess=max(0.,action.irrigation_liters-safe_water)
        demand=model.dose_to_target(theta,cfg['target_index'],actual_weather,item.water_budget_liters)
        # Audit extrapolation; actual historical soil index/next targets never enter policy.
        actual_inputs={'soil_moisture_index':theta,'irrigation_liters':action.irrigation_liters,**actual_weather}
        lo=model.training_report['training_feature_min'];hi=model.training_report['training_feature_max']
        out_of_range=any(actual_inputs[c]<lo[c]-1e-9 or actual_inputs[c]>hi[c]+1e-9 for c in actual_inputs)
        rows.append({'timestamp':row.timestamp,'sector_id':row.sector_id,'alternative':strategy.name,
                     'moisture_start_index':theta,'moisture_observation_index':observation,
                     'moisture_end_index':end,'irrigation_liters':action.irrigation_liters,
                     'water_budget_liters':item.water_budget_liters,'modeled_excess_irrigation_liters':excess,
                     'modeled_feasible_demand_liters':demand,'used_fallback':action.used_fallback,
                     'sensor_missing':bool(row.sensor_missing),'demand_event':bool(row.demand_event),
                     'outside_training_range':out_of_range,'reason':action.reason})
        theta=end
    return pd.DataFrame(rows)
