import math

from src.irrigation.interfaces import RealStrategyInput
from src.irrigation.strategies.sidss_like import SIDSSReal


class FakeSoilModel:
    """
    Mô hình giả chỉ dùng cho unit test.

    Quy ước:
    - nếu không tưới, moisture giảm 2 điểm/ngày
    - mỗi 1 lít nước làm tăng 0.1 điểm
    """

    irrigation_gain = 0.1

    def predict(self, moisture, liters, weather):
        return float(
            moisture
            - 2.0
            + liters * self.irrigation_gain
        )

    def dose_to_target(
        self,
        moisture,
        target,
        weather,
        budget,
    ):
        predicted_no_water = self.predict(
            moisture,
            0.0,
            weather,
        )

        needed = (
            target - predicted_no_water
        ) / self.irrigation_gain

        return float(
            max(
                0.0,
                min(needed, budget),
            )
        )


def make_input(
    moisture=45.0,
    budget=500.0,
    target=50.0,
    lower=40.0,
):
    return RealStrategyInput(
        step=0,
        moisture_observation_index=moisture,
        weather_forecast={
            "temperature_c": 30.0,
            "rain_mm": 0.0,
            "radiation_mj_m2": 18.0,
            "humidity_percent": 60.0,
        },
        water_budget_liters=budget,
        config={
            "initial_index": 45.0,
            "target_index": target,
            "lower_index": lower,
            "upper_index": 60.0,
        },
        model=FakeSoilModel(),
    )


def test_no_irrigation_when_predicted_moisture_is_at_or_above_target():
    strategy = SIDSSReal()

    item = make_input(
        moisture=52.0,
    )

    result = strategy.decide(item)

    assert result.irrigation_liters == 0.0
    assert result.used_fallback is False


def test_drier_soil_does_not_receive_less_water():
    wetter_strategy = SIDSSReal()
    drier_strategy = SIDSSReal()

    wetter = wetter_strategy.decide(
        make_input(moisture=45.0)
    )

    drier = drier_strategy.decide(
        make_input(moisture=35.0)
    )

    assert (
        drier.irrigation_liters
        >= wetter.irrigation_liters
    )


def test_missing_sensor_uses_internal_model_estimate():
    strategy = SIDSSReal()

    first = strategy.decide(
        make_input(moisture=45.0)
    )

    missing_item = make_input(
        moisture=None,
    )

    second = strategy.decide(
        missing_item
    )

    assert first.used_fallback is False
    assert second.used_fallback is True
    assert math.isfinite(
        second.irrigation_liters
    )

    assert (
        "fallback"
        in second.reason.lower()
    )


def test_irrigation_never_exceeds_daily_budget():
    strategy = SIDSSReal()

    item = make_input(
        moisture=20.0,
        budget=25.0,
    )

    result = strategy.decide(item)

    assert result.irrigation_liters >= 0.0
    assert result.irrigation_liters <= 25.0

    result.validate(
        item.water_budget_liters
    )