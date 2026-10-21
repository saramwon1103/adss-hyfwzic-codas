"""Current real-data contract: litres and relative probe-index points."""
from dataclasses import dataclass
import math
WEATHER = ['temperature_c','rain_mm','radiation_mj_m2','humidity_percent']
MODEL_FEATURES = ['soil_moisture_index','irrigation_liters',*WEATHER]
FORBIDDEN_FEATURES = {'target_point_24h','target_mean_24h','water_vol_to_24h','real_moisture_delta'}

@dataclass(frozen=True)
class RealStrategyInput:
    step: int
    moisture_observation_index: float | None
    weather_forecast: dict
    water_budget_liters: float
    config: dict
    model: object  # fitted strictly before evaluation; no test observations/targets

@dataclass(frozen=True)
class RealStrategyOutput:
    irrigation_liters: float
    used_fallback: bool
    reason: str

    def validate(self,budget):
        if not math.isfinite(self.irrigation_liters) or not 0<=self.irrigation_liters<=budget+1e-9:
            raise ValueError('Real policy action must be finite litres within the daily allocation.')
        if not isinstance(self.used_fallback,bool) or not isinstance(self.reason,str):
            raise ValueError('Real policy output requires fallback flag and reason.')
