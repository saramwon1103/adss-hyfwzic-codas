import numpy as np
import pandas as pd
REAL_KPIS=['water_use_liters','soil_moisture_mae_points','over_irrigation_liters',
           'replanning_response_days','missing_data_robustness']


def build_performance_matrix(records):
    frame=pd.DataFrame(records)
    if frame.empty or 'alternative' not in frame or frame.alternative.duplicated().any():
        raise ValueError('Expected one complete KPI record per unique alternative.')
    matrix=frame.set_index('alternative')[REAL_KPIS].astype(float)
    if not np.isfinite(matrix.to_numpy()).all() or (matrix<0).any().any():
        raise ValueError('Invalid real KPI matrix.')
    if (matrix.missing_data_robustness>1).any():raise ValueError('Robustness must lie in [0,1].')
    return matrix
