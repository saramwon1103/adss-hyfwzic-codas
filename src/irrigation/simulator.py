import numpy as np
import pandas as pd

from .interfaces import (
    WEATHER,
    RealStrategyInput,
)


def _sensor_observation(
    theta,
    row,
    cfg,
):
    """
    Measurement layer.

    theta:
        latent moisture state of the simulated environment.

    When the sensor is available:
        the strategy observes the current environment state
        through this measurement layer.

    When the sensor is unavailable:
        the strategy receives None and must use its fallback.
    """

    if bool(
        row.sensor_missing
    ):
        return None

    # Optional deterministic bias exists mainly so that
    # the measurement layer can be explicitly unit-tested.
    # Production config currently defaults to zero.
    bias = float(
        cfg.get(
            "sensor_bias_points",
            0.0,
        )
    )

    return float(
        np.clip(
            theta + bias,
            0.0,
            100.0,
        )
    )


def simulate_real(
    frame,
    model,
    cfg,
    factory,
    disturbances=None,
):
    """
    Replay one irrigation strategy on one scenario.

    The environment and the strategy are deliberately
    separated:

        latent state
            ↓
        sensor observation
            ↓
        strategy decision
            ↓
        soil-response model
            +
        environment disturbance
            ↓
        next latent state

    disturbances:
        Exogenous moisture deviations derived only from
        pre-evaluation training residuals.

    The strategy never receives disturbance directly.
    """

    if frame.empty:
        raise ValueError(
            "Simulation frame cannot be empty."
        )

    # -----------------------------------------------------
    # Disturbance sequence
    # -----------------------------------------------------

    if disturbances is None:
        disturbances = np.zeros(
            len(frame),
            dtype=float,
        )

    disturbances = np.asarray(
        disturbances,
        dtype=float,
    )

    if (
        len(disturbances)
        != len(frame)
    ):
        raise ValueError(
            "Disturbance sequence must match "
            "simulation length."
        )

    if not np.isfinite(
        disturbances
    ).all():
        raise ValueError(
            "Disturbances must be finite."
        )

    # -----------------------------------------------------
    # Strategy + latent environment initialization
    # -----------------------------------------------------

    strategy = factory()

    theta = float(
        cfg[
            "initial_index"
        ]
    )

    rows = []

    # -----------------------------------------------------
    # Simulation loop
    # -----------------------------------------------------

    for (
        step,
        row,
    ) in (
        frame
        .reset_index(
            drop=True
        )
        .iterrows()
    ):

        # -------------------------------------------------
        # Information visible to the strategy
        # -------------------------------------------------

        observation = (
            _sensor_observation(
                theta,
                row,
                cfg,
            )
        )

        forecast = {
            c:
                float(
                    row[
                        f"{c}_forecast"
                    ]
                )
            for c
            in WEATHER
        }

        item = (
            RealStrategyInput(
                step,
                observation,
                forecast,
                float(
                    row.water_budget_liters
                ),
                dict(cfg),
                model,
            )
        )

        # -------------------------------------------------
        # Strategy action
        # -------------------------------------------------

        action = (
            strategy.decide(
                item
            )
        )

        action.validate(
            item.water_budget_liters
        )

        # -------------------------------------------------
        # Actual environment forcing
        # -------------------------------------------------

        actual_weather = {
            c:
                float(row[c])
            for c
            in WEATHER
        }

        # Environment without irrigation:
        # used only for excess-water diagnostics.
        predicted_without = (
            model.predict(
                theta,
                0.0,
                actual_weather,
            )
        )

        # Base surrogate prediction after chosen action.
        modeled_end = (
            model.predict(
                theta,
                action.irrigation_liters,
                actual_weather,
            )
        )

        # -------------------------------------------------
        # Exogenous environment disturbance
        #
        # This is the key separation:
        #
        # strategy knows model prediction,
        # but does NOT know disturbance beforehand.
        # -------------------------------------------------

        disturbance = float(
            disturbances[
                step
            ]
        )

        end = float(
            np.clip(
                modeled_end
                + disturbance,
                0.0,
                100.0,
            )
        )

        # -------------------------------------------------
        # Excess irrigation diagnostic
        # -------------------------------------------------

        gain = float(
            model.irrigation_gain
        )

        if gain > 1e-8:

            safe_water = max(
                0.0,
                (
                    cfg[
                        "upper_index"
                    ]
                    -
                    predicted_without
                )
                / gain,
            )

        else:

            safe_water = float(
                "inf"
            )

        excess = max(
            0.0,
            action.irrigation_liters
            - safe_water,
        )

        # -------------------------------------------------
        # Feasible water demand diagnostic
        # -------------------------------------------------

        demand = (
            model.dose_to_target(
                theta,
                cfg[
                    "target_index"
                ],
                actual_weather,
                item.water_budget_liters,
            )
        )

        # -------------------------------------------------
        # Audit extrapolation
        #
        # Historical future moisture/targets never enter
        # the policy.
        # -------------------------------------------------

        actual_inputs = {
            "soil_moisture_index":
                theta,

            "irrigation_liters":
                action.irrigation_liters,

            **actual_weather,
        }

        lo = (
            model
            .training_report[
                "training_feature_min"
            ]
        )

        hi = (
            model
            .training_report[
                "training_feature_max"
            ]
        )

        out_of_range = any(
            (
                actual_inputs[c]
                < lo[c] - 1e-9
            )
            or
            (
                actual_inputs[c]
                > hi[c] + 1e-9
            )
            for c
            in actual_inputs
        )

        # -------------------------------------------------
        # Trace
        # -------------------------------------------------

        rows.append(
            {
                "timestamp":
                    row.timestamp,

                "sector_id":
                    row.sector_id,

                "alternative":
                    strategy.name,

                "moisture_start_index":
                    theta,

                "moisture_observation_index":
                    observation,

                "moisture_end_index":
                    end,

                "irrigation_liters":
                    action.irrigation_liters,

                "water_budget_liters":
                    item.water_budget_liters,

                "modeled_excess_irrigation_liters":
                    excess,

                "modeled_feasible_demand_liters":
                    demand,

                "used_fallback":
                    action.used_fallback,

                "sensor_missing":
                    bool(
                        row.sensor_missing
                    ),

                "demand_event":
                    bool(
                        row.demand_event
                    ),

                "outside_training_range":
                    out_of_range,

                "environment_disturbance_points":
                    disturbance,

                "reason":
                    action.reason,
            }
        )

        # -------------------------------------------------
        # Advance environment
        # -------------------------------------------------

        theta = end

    return pd.DataFrame(
        rows
    )