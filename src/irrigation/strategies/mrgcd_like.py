import numpy as np
from ..interfaces import RealStrategyOutput

class MRGCDReal:
    name = 'MRGCD-like'
    def __init__(self):self.backlog=0.;self.last=None
    def decide(self,item):
        missing = item.moisture_observation_index is None
        if not missing:self.last=item.moisture_observation_index
        theta = item.config['initial_index'] if self.last is None else self.last
        needed=item.model.dose_to_target(theta,item.config['target_index'],item.weather_forecast,item.water_budget_liters)
        self.backlog+=needed
        dose=min(item.water_budget_liters,self.backlog) if item.step%3==0 else 0.
        self.backlog=max(0.,self.backlog-dose)
        return RealStrategyOutput(float(dose),missing,'heuristic 3-day rotation and demand backlog under allocation limit')
