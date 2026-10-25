"""Chronological constrained ridge; observational surrogate, not a causal soil model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .interfaces import WEATHER, MODEL_FEATURES


@dataclass
class SoilResponseModel:
    feature_names: list
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients_standardized: np.ndarray
    alpha: float
    training_report: dict

    @property
    def coefficients_raw(self):
        return (
            self.coefficients_standardized[1:]
            / self.feature_scale
        )

    @property
    def irrigation_gain(self):
        return float(
            self.coefficients_raw[1]
        )

    def predict(
        self,
        moisture,
        liters,
        weather,
    ):
        values = np.array(
            [
                moisture,
                liters,
                *[
                    weather[c]
                    for c in WEATHER
                ],
            ],
            dtype=float,
        )

        if (
            not np.isfinite(values).all()
            or liters < 0
        ):
            raise ValueError(
                "Model inputs must be finite; "
                "water must be nonnegative."
            )

        raw = float(
            self.coefficients_standardized[0]
            +
            (
                (
                    values
                    - self.feature_mean
                )
                / self.feature_scale
            )
            @
            self.coefficients_standardized[1:]
        )

        return float(
            np.clip(
                raw,
                0,
                100,
            )
        )

    def dose_to_target(
        self,
        moisture,
        target,
        weather,
        budget,
    ):
        no_water = self.predict(
            moisture,
            0.0,
            weather,
        )

        if self.irrigation_gain <= 1e-8:
            # Do not manufacture an unsupported
            # irrigation response.
            return 0.0

        return float(
            np.clip(
                (
                    target
                    - no_water
                )
                / self.irrigation_gain,
                0,
                budget,
            )
        )

    def to_dict(self):
        return {
            "feature_names":
                self.feature_names,

            "feature_mean":
                self.feature_mean.tolist(),

            "feature_scale":
                self.feature_scale.tolist(),

            "coefficients_standardized":
                self.coefficients_standardized.tolist(),

            "coefficients_raw":
                dict(
                    zip(
                        self.feature_names,
                        map(
                            float,
                            self.coefficients_raw,
                        ),
                    )
                ),

            "alpha":
                self.alpha,

            "training_report":
                self.training_report,

            "causal_validity":
                (
                    "observational surrogate only; "
                    "historical irrigation is confounded "
                    "with farmer decisions"
                ),
        }


def _training_transitions(
    daily,
    evaluation_start,
):
    """
    Return exactly the chronological transitions
    eligible for model training.

    Both the input day and its next-day target must occur
    strictly before the evaluation period.
    """

    start = pd.Timestamp(
        evaluation_start
    )

    cols = [
        "soil_moisture_index",
        "historical_irrigation_liters",
        *WEATHER,
    ]

    mask = (
        (daily.timestamp < start)
        &
        (
            daily.timestamp
            + pd.Timedelta(days=1)
            < start
        )
    )

    train = (
        daily.loc[mask]
        .dropna(
            subset=(
                cols
                + [
                    "next_soil_moisture_index"
                ]
            )
        )
        .copy()
    )

    return train


def training_residuals(
    daily,
    model,
    evaluation_start,
):
    """
    Build an empirical disturbance library using only
    pre-evaluation training transitions.

    residual =
        observed_next_index
        - predicted_next_index

    These residuals are used only as an empirical
    environment-disturbance library.

    They must NOT be reported as holdout/model accuracy,
    because the model was fitted using these transitions.
    """

    train = _training_transitions(
        daily,
        evaluation_start,
    )

    if train.empty:
        raise ValueError(
            "No valid pre-evaluation transitions "
            "for residual estimation."
        )

    rows = []

    for _, row in train.iterrows():

        weather = {
            c: float(row[c])
            for c in WEATHER
        }

        predicted = model.predict(
            float(
                row.soil_moisture_index
            ),
            float(
                row.historical_irrigation_liters
            ),
            weather,
        )

        observed = float(
            row.next_soil_moisture_index
        )

        residual = (
            observed
            - predicted
        )

        rows.append(
            {
                "timestamp":
                    row.timestamp,

                "observed_next_index":
                    observed,

                "predicted_next_index":
                    predicted,

                "residual":
                    residual,
            }
        )

    result = pd.DataFrame(
        rows
    )

    expected_rows = int(
        model.training_report[
            "training_rows"
        ]
    )

    if len(result) != expected_rows:
        raise ValueError(
            "Residual training rows do not match "
            "the fitted model training rows."
        )

    if not np.isfinite(
        result["residual"].to_numpy()
    ).all():
        raise ValueError(
            "Training residuals must be finite."
        )

    return result


def make_disturbance_sequence(
    residuals,
    length,
):
    """
    Create a deterministic disturbance sequence
    for counterfactual replay.

    Residuals are replayed chronologically.
    If an evaluation period is longer than the residual
    library, the library is repeated deterministically.

    No random sampling is used so that runs are fully
    reproducible.
    """

    length = int(length)

    if length < 1:
        raise ValueError(
            "Disturbance length must be positive."
        )

    if (
        residuals.empty
        or "residual"
        not in residuals.columns
    ):
        raise ValueError(
            "Residual library must contain "
            "at least one residual value."
        )

    values = residuals[
        "residual"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Residual library must be finite."
        )

    return np.resize(
        values,
        length,
    ).astype(
        float,
        copy=False,
    )


def fit_response_model(
    daily,
    evaluation_start,
    alpha=10.0,
):
    """
    Chronological constrained ridge.

    No holdout targets are used in:
    - scaler
    - model fit
    - feature bounds

    Constraints:
    - 0 <= autoregressive coefficient <= 1
    - irrigation effect >= 0

    The feasible active-set combinations are enumerated,
    solved, and the lowest-loss feasible solution is used.
    """

    import itertools

    start = pd.Timestamp(
        evaluation_start
    )

    cols = [
        "soil_moisture_index",
        "historical_irrigation_liters",
        *WEATHER,
    ]

    train = _training_transitions(
        daily,
        start,
    )

    if (
        len(train) < 30
        or
        train.historical_irrigation_liters
        .gt(0)
        .sum()
        < 5
    ):
        raise ValueError(
            "Need >=30 valid pre-evaluation transitions "
            "and >=5 irrigated training days. "
            "Select a later evaluation start."
        )

    X = train[
        cols
    ].to_numpy(
        dtype=float
    )

    y = train[
        "next_soil_moisture_index"
    ].to_numpy(
        dtype=float
    )

    mean = X.mean(0)

    scale = X.std(0)

    scale[
        scale < 1e-8
    ] = 1.0

    A = np.c_[
        np.ones(len(X)),
        (
            X - mean
        )
        / scale,
    ]

    penalty = np.r_[
        0.0,
        np.ones(
            X.shape[1]
        ),
    ]

    best = None

    for (
        ar_status,
        water_status,
    ) in itertools.product(
        [
            "free",
            "lo",
            "hi",
        ],
        [
            "free",
            "lo",
        ],
    ):

        fixed = {}

        if ar_status != "free":
            fixed[1] = (
                0.0
                if ar_status == "lo"
                else scale[0]
            )

        if water_status == "lo":
            fixed[2] = 0.0

        free = [
            j
            for j
            in range(
                A.shape[1]
            )
            if j not in fixed
        ]

        B = A[
            :,
            free,
        ]

        adjusted = (
            y
            -
            sum(
                (
                    A[:, j] * v
                    for j, v
                    in fixed.items()
                ),
                start=np.zeros_like(y),
            )
        )

        coefficients = np.zeros(
            A.shape[1]
        )

        for j, value in fixed.items():
            coefficients[j] = value

        coefficients[
            free
        ] = np.linalg.solve(
            B.T @ B
            +
            alpha
            * np.diag(
                penalty[free]
            ),
            B.T @ adjusted,
        )

        if not (
            -1e-9
            <= coefficients[1]
            <= scale[0] + 1e-9
            and
            coefficients[2]
            >= -1e-9
        ):
            continue

        loss = float(
            np.square(
                A @ coefficients
                - y
            ).sum()
            +
            alpha
            * np.sum(
                coefficients[1:]
                ** 2
            )
        )

        if (
            best is None
            or loss < best[0]
        ):
            best = (
                loss,
                coefficients,
            )

    if best is None:
        raise ValueError(
            "Unable to fit a feasible "
            "response model."
        )

    report = {
        "training_rows":
            len(train),

        "training_first_day":
            str(
                train.timestamp
                .min()
                .date()
            ),

        "training_last_input_day":
            str(
                train.timestamp
                .max()
                .date()
            ),

        "training_last_target_day":
            str(
                (
                    train.timestamp.max()
                    + pd.Timedelta(days=1)
                ).date()
            ),

        "evaluation_start":
            str(
                start.date()
            ),

        "irrigated_training_days":
            int(
                train
                .historical_irrigation_liters
                .gt(0)
                .sum()
            ),

        "water_budget_liters":
            float(
                train
                .historical_irrigation_liters
                .quantile(0.95)
            ),

        "water_budget_rule":
            (
                "95th percentile of all valid "
                "training-day nominal volumes, "
                "including zeros"
            ),

        "training_volume_max_liters":
            float(
                train
                .historical_irrigation_liters
                .max()
            ),

        "training_feature_min":
            dict(
                zip(
                    MODEL_FEATURES,
                    map(
                        float,
                        X.min(0),
                    ),
                )
            ),

        "training_feature_max":
            dict(
                zip(
                    MODEL_FEATURES,
                    map(
                        float,
                        X.max(0),
                    ),
                )
            ),

        "constraints":
            (
                "AR in [0,1]; "
                "water coefficient >=0; "
                "alpha=10 fixed before holdout"
            ),

        "target":
            (
                "next local-midnight valid "
                "probe index; no precomputed "
                "24h targets"
            ),
    }

    if (
        report[
            "water_budget_liters"
        ]
        <= 0
    ):
        raise ValueError(
            "Training water budget is zero; "
            "no mock budget substituted."
        )

    return SoilResponseModel(
        MODEL_FEATURES.copy(),
        mean,
        scale,
        best[1],
        float(alpha),
        report,
    )