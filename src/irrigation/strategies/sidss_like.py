import numpy as np

from ..interfaces import RealStrategyOutput


class SIDSSReal:
    """
    Predictive sensor-feedback proxy inspired by SIDSS.

    This is NOT the original SIDSS implementation.

    The policy uses:
    - current soil-moisture observation,
    - persistence weather forecast,
    - water budget,
    - training-only soil-response surrogate.

    It produces a daily irrigation recommendation.
    """

    name = "SIDSS-like"

    def __init__(self):
        self.estimate = None

    def decide(self, item):

        # 1. Determine current moisture state
        missing = (
            item.moisture_observation_index
            is None
        )

        if not missing:
            self.estimate = float(
                item.moisture_observation_index
            )

        elif self.estimate is None:
            self.estimate = float(
                item.config["initial_index"]
            )

        theta = float(self.estimate)

        target = float(
            item.config["target_index"]
        )

        lower = float(
            item.config["lower_index"]
        )

        # 2. Predict moisture if no irrigation is applied
        predicted_no_water = (
            item.model.predict(
                theta,
                0.0,
                item.weather_forecast,
            )
        )

        # 3. Estimate water required to move toward target
        needed = (
            item.model.dose_to_target(
                theta,
                target,
                item.weather_forecast,
                item.water_budget_liters,
            )
        )

        # 4. Determine irrigation urgency
        if predicted_no_water >= target:
            urgency = 0.0

        elif target <= lower:
            urgency = 1.0

        else:
            urgency = float(
                np.clip(
                    (
                        target
                        - predicted_no_water
                    )
                    /
                    (
                        target
                        - lower
                    ),
                    0.0,
                    1.0,
                )
            )

        # 5. Calculate irrigation amount
        dose = float(
            np.clip(
                needed * urgency,
                0.0,
                item.water_budget_liters,
            )
        )

        # 6. Update internal moisture estimate
        self.estimate = (
            item.model.predict(
                theta,
                dose,
                item.weather_forecast,
            )
        )

        # 7. Explain decision
        if missing:
            mode = "model-propagated fallback"
        else:
            mode = "sensor feedback"

        reason = (
            "predictive moisture/weather control; "
            f"{mode}; "
            f"forecast_no_water="
            f"{predicted_no_water:.3f}; "
            f"urgency={urgency:.3f}"
        )

        return RealStrategyOutput(
            irrigation_liters=dose,
            used_fallback=missing,
            reason=reason,
        )