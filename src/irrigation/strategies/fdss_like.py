import numpy as np
from ..interfaces import RealStrategyOutput

class FDSSReal:
    name = 'FDSS-like'
    def __init__(self):self.last=None
    def decide(self,item):
        missing = item.moisture_observation_index is None
        if not missing:self.last=item.moisture_observation_index
        theta = item.config['initial_index'] if self.last is None else self.last
        dry = float(np.clip((item.config['upper_index']-theta)/(item.config['upper_index']-item.config['lower_index']),0,1))
        hot = float(np.clip((item.weather_forecast['temperature_c']-20)/15,0,1))
        rainy = float(np.clip(item.weather_forecast['rain_mm']/10,0,1))
        needed = item.model.dose_to_target(theta,item.config['target_index'],item.weather_forecast,item.water_budget_liters)
        dose = np.clip(needed*(.65*dry+.35*hot)*(1-.5*rainy),0,item.water_budget_liters)
        return RealStrategyOutput(float(dose),missing,'heuristic fuzzy dryness/heat/rain rules; sample-and-hold fallback')
