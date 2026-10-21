import numpy as np
from ..interfaces import RealStrategyOutput

class SIDSSRealStub:
    name = 'SIDSS-like (stub)'
    def __init__(self):self.estimate=None
    def decide(self,item):
        missing = item.moisture_observation_index is None
        if not missing:self.estimate=item.moisture_observation_index
        if self.estimate is None:self.estimate=item.config['initial_index']
        dose = item.model.dose_to_target(self.estimate,item.config['target_index'],item.weather_forecast,item.water_budget_liters)
        self.estimate=item.model.predict(self.estimate,dose,item.weather_forecast)
        return RealStrategyOutput(dose,missing,'STUB: fitted-response target feedback; observer uses persistence weather forecast')
