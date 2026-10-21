import numpy as np

def real_kpis(trace,base_trace,outage_trace,cfg):
    for other in [base_trace,outage_trace]:
        if not trace.timestamp.equals(other.timestamp) or not trace.alternative.equals(other.alternative):
            raise ValueError('Robustness traces require the same timeline and strategy.')
    mae=lambda t:float((t.moisture_end_index-cfg['target_index']).abs().mean())
    degradation=max(0.,mae(outage_trace)-mae(base_trace))
    delays=[]
    for pos in np.flatnonzero(trace.demand_event.to_numpy()):
        window=trace.iloc[pos:pos+cfg['response_window_days']]
        tol=np.maximum(.1,.05*window.water_budget_liters.to_numpy())
        hits=np.flatnonzero(window.irrigation_liters.to_numpy()+tol>=window.modeled_feasible_demand_liters.to_numpy())
        delays.append(float(hits[0]) if len(hits) else float(cfg['response_window_days']))
    return {'water_use_liters':float(trace.irrigation_liters.sum()),'soil_moisture_mae_points':mae(trace),
            'over_irrigation_liters':float(trace.modeled_excess_irrigation_liters.sum()),
            'replanning_response_days':float(np.mean(delays)) if delays else 0.,
            'missing_data_robustness':float(np.clip(1-degradation/cfg['robustness_tolerance_points'],0,1))}
