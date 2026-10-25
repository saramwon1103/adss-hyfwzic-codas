"""Real-input Mode2 orchestration and CLI in the repository's src/ layout."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ROOT,
    PAPER_DATA,
    IRRIGATION_DATA,
    IRRIGATION_OUTPUT,
)

from src.core.codas import (
    crisp_codas,
)

from src.core.hy_fwzic import (
    compute_weights,
)

from src.analysis.sensitivity import (
    analyze,
    weight_scenarios,
)

from src.common.outputs import (
    export_codas,
)

from src.common.reproducibility import (
    write_manifest,
)

from .interfaces import (
    WEATHER,
    RealStrategyInput,
    RealStrategyOutput,
)

from .preprocess import (
    prepare_real_run,
    DEFAULT_START,
    DEFAULT_END,
    SOURCE_DIR,
)

from .soil_response import (
    training_residuals,
    make_disturbance_sequence,
)

from .strategies import (
    REAL_STRATEGIES,
)

from .scenario_generator import (
    real_scenario,
    SCENARIOS,
)

from .simulator import (
    simulate_real,
)

from .kpi_evaluator import (
    real_kpis,
)

from .build_performance_matrix import (
    build_performance_matrix,
)


def run_real_mode2(
    scenario="Normal",
    sector=1,
    start=DEFAULT_START,
    end=DEFAULT_END,
):
    """
    Execute one complete real-data Mode 2 experiment.

    Flow:

        real data
        → preprocessing
        → training-only surrogate
        → training residual library
        → deterministic disturbance replay
        → scenario
        → three ADSS-like strategies
        → KPI matrix
        → Hy-FWZIC weights
        → CODAS ranking
        → sensitivity analysis
    """

    (
        base,
        model,
        cfg,
        quality,
        validation,
        daily,
    ) = prepare_real_run(
        sector,
        start,
        end,
    )

    # -----------------------------------------------------
    # Scenario under evaluation
    # -----------------------------------------------------

    frame = real_scenario(
        base,
        scenario,
    )

    # -----------------------------------------------------
    # Empirical disturbance library
    #
    # IMPORTANT:
    # Only pre-evaluation training transitions are used.
    # No holdout/evaluation outcome is used to generate
    # these disturbances.
    # -----------------------------------------------------

    residual_library = (
        training_residuals(
            daily,
            model,
            start,
        )
    )

    disturbances = (
        make_disturbance_sequence(
            residual_library,
            len(frame),
        )
    )

    # -----------------------------------------------------
    # Paired missing-sensor robustness experiment
    #
    # reference and outage share:
    # - same weather
    # - same water budget
    # - same disturbance sequence
    #
    # They differ only in imposed sensor availability.
    # -----------------------------------------------------

    baseline = real_scenario(
        base,
        "Normal",
    )

    for col in (
        WEATHER
        +
        [
            f"{c}_forecast"
            for c in WEATHER
        ]
        +
        [
            "water_budget_liters",
            "demand_event",
        ]
    ):
        baseline[col] = (
            frame[col]
        )

    stressed = (
        baseline.copy()
    )

    # Controlled sensor outage:
    # evaluation steps 10..18 inclusive = 9 days.
    stressed.loc[
        10:18,
        "sensor_missing",
    ] = True

    if (
        "scenario_sensor_outage"
        in stressed.columns
    ):
        stressed.loc[
            10:18,
            "scenario_sensor_outage",
        ] = True

    # -----------------------------------------------------
    # Run every alternative under exactly the same
    # environment disturbance sequence.
    # -----------------------------------------------------

    records = []
    traces = {}

    for (
        name,
        factory,
    ) in REAL_STRATEGIES.items():

        trace = simulate_real(
            frame,
            model,
            cfg,
            factory,
            disturbances=disturbances,
        )

        reference = simulate_real(
            baseline,
            model,
            cfg,
            factory,
            disturbances=disturbances,
        )

        outage = simulate_real(
            stressed,
            model,
            cfg,
            factory,
            disturbances=disturbances,
        )

        traces[
            name
        ] = trace

        records.append(
            {
                "alternative":
                    name,

                **real_kpis(
                    trace,
                    reference,
                    outage,
                    cfg,
                ),
            }
        )

    # -----------------------------------------------------
    # KPI performance matrix
    # -----------------------------------------------------

    matrix = (
        build_performance_matrix(
            records
        )
    )

    # -----------------------------------------------------
    # Criteria metadata
    # -----------------------------------------------------

    criteria = (
        pd.read_csv(
            IRRIGATION_DATA
            / "criteria.csv"
        )
        .set_index(
            "criterion"
        )
    )

    # -----------------------------------------------------
    # Illustrative / synthetic decision-maker preferences
    #
    # Do not call these real expert judgments unless
    # actual experts are later elicited.
    # -----------------------------------------------------

    prefs = pd.read_csv(
        IRRIGATION_DATA
        / "preferences.csv",
        index_col="expert",
    )

    # -----------------------------------------------------
    # Hy-FWZIC
    # -----------------------------------------------------

    weights = (
        compute_weights(
            prefs,
            pd.read_csv(
                PAPER_DATA
                / "linguistic_mapping.csv"
            ),
        )
        .set_index(
            "criterion"
        )
        .weight
    )

    # -----------------------------------------------------
    # CODAS
    # -----------------------------------------------------

    def rank(w):
        return crisp_codas(
            matrix,
            w,
            criteria.direction,
        )

    result = rank(
        weights
    )

    # -----------------------------------------------------
    # Sensitivity
    # -----------------------------------------------------

    sens, summary = analyze(
        rank,
        weights,
        weight_scenarios(
            weights,
            high=0.4,
        ),
    )

    # -----------------------------------------------------
    # Quality / audit metadata
    # -----------------------------------------------------

    quality[
        "scenario"
    ] = scenario

    quality[
        "scenario_inputs_modified"
    ] = (
        scenario != "Normal"
    )

    quality[
        "disturbance_source"
    ] = (
        "pre-evaluation training residual replay"
    )

    quality[
        "disturbance_library_rows"
    ] = int(
        len(
            residual_library
        )
    )

    quality[
        "disturbance_sequence_rows"
    ] = int(
        len(
            disturbances
        )
    )

    quality[
        "disturbance_mean_points"
    ] = float(
        np.mean(
            disturbances
        )
    )

    quality[
        "disturbance_mae_points"
    ] = float(
        np.mean(
            np.abs(
                disturbances
            )
        )
    )

    quality[
        "disturbance_min_points"
    ] = float(
        np.min(
            disturbances
        )
    )

    quality[
        "disturbance_max_points"
    ] = float(
        np.max(
            disturbances
        )
    )

    quality[
        "out_of_training_range_steps"
    ] = {
        name:
            int(
                trace
                .outside_training_range
                .sum()
            )
        for (
            name,
            trace,
        )
        in traces.items()
    }

    quality[
        "weak_irrigation_response"
    ] = bool(
        model.irrigation_gain
        < 0.01
    )

    quality[
        "water_effect_at_daily_budget_points"
    ] = float(
        model.irrigation_gain
        *
        model.training_report[
            "water_budget_liters"
        ]
    )

    quality[
        "holdout_error_exceeds_daily_water_effect"
    ] = bool(
        quality[
            "holdout_mae_points"
        ]
        >
        quality[
            "water_effect_at_daily_budget_points"
        ]
    )

    quality[
        "non_discriminating_kpis"
    ] = (
        matrix.columns[
            (
                matrix.max()
                - matrix.min()
            )
            < 1e-10
        ]
        .tolist()
    )

    quality[
        "initial_target_gap_points"
    ] = float(
        cfg[
            "target_index"
        ]
        -
        cfg[
            "initial_index"
        ]
    )

    return {
        "dataset":
            frame,

        "matrix":
            matrix,

        "criteria":
            criteria,

        "weights":
            weights,

        "result":
            result,

        "traces":
            traces,

        "sensitivity":
            sens,

        "sensitivity_summary":
            summary,

        "scenario":
            scenario,

        "is_synthetic":
            False,

        "source_kind":
            "uploaded_real_data",

        "outcomes_modeled":
            True,

        "model":
            model,

        "config":
            cfg,

        "quality":
            quality,

        "validation":
            validation,

        "daily_source":
            daily,

        "training_residuals":
            residual_library,

        "disturbances":
            disturbances,
    }


def export_run(
    run,
    out,
):
    """
    Export all reproducible Mode 2 artifacts.
    """

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    export_codas(
        run[
            "result"
        ],
        out,
        run[
            "matrix"
        ].columns,
    )

    run[
        "dataset"
    ].to_csv(
        out
        / "scenario_dataset.csv",
        index=False,
    )

    run[
        "matrix"
    ].to_csv(
        out
        / "performance_matrix.csv"
    )

    run[
        "weights"
    ].to_csv(
        out
        / "weights.csv"
    )

    run[
        "sensitivity"
    ].to_csv(
        out
        / "sensitivity.csv",
        index=False,
    )

    run[
        "sensitivity_summary"
    ].to_csv(
        out
        / "stability.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Strategy traces
    # -----------------------------------------------------

    for (
        name,
        trace,
    ) in run[
        "traces"
    ].items():

        filename = (
            name
            .lower()
            .replace(
                " ",
                "_",
            )
            .replace(
                "(",
                "",
            )
            .replace(
                ")",
                "",
            )
            +
            "_trace.csv"
        )

        trace.to_csv(
            out
            / filename,
            index=False,
        )

    if not run[
        "is_synthetic"
    ]:

        # Holdout validation is kept separate from
        # training residual replay.
        run[
            "validation"
        ].to_csv(
            out
            / "historical_holdout_validation.csv",
            index=False,
        )

        # Training residual library.
        run[
            "training_residuals"
        ].to_csv(
            out
            / "training_residuals.csv",
            index=False,
        )

        # Exact disturbance sequence used in simulation.
        pd.DataFrame(
            {
                "step":
                    np.arange(
                        len(
                            run[
                                "disturbances"
                            ]
                        )
                    ),

                "disturbance_points":
                    run[
                        "disturbances"
                    ],
            }
        ).to_csv(
            out
            / "disturbance_sequence.csv",
            index=False,
        )

        (
            out
            / "soil_response_model.json"
        ).write_text(
            json.dumps(
                run[
                    "model"
                ].to_dict(),
                indent=2,
            ),
            encoding="utf-8",
        )

        (
            out
            / "data_quality.json"
        ).write_text(
            json.dumps(
                run[
                    "quality"
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

        (
            out
            / "simulation_config.json"
        ).write_text(
            json.dumps(
                run[
                    "config"
                ],
                indent=2,
            ),
            encoding="utf-8",
        )


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--scenario",
        choices=[
            *SCENARIOS,
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--sector",
        type=int,
        choices=[
            1,
            2,
        ],
        default=1,
    )

    parser.add_argument(
        "--start",
        default=DEFAULT_START,
    )

    parser.add_argument(
        "--end",
        default=DEFAULT_END,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=IRRIGATION_OUTPUT,
    )

    args = (
        parser.parse_args()
    )

    output = (
        args.output
        .resolve()
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = []

    scenarios = (
        SCENARIOS
        if args.scenario
        == "all"
        else [
            args.scenario
        ]
    )

    for scenario in scenarios:

        run = run_real_mode2(
            scenario,
            args.sector,
            args.start,
            args.end,
        )

        out = (
            output
            /
            scenario
            .lower()
            .replace(
                " ",
                "_",
            )
        )

        export_run(
            run,
            out,
        )

        run[
            "daily_source"
        ].to_csv(
            output
            / "processed_irrigation.csv",
            index=False,
        )

        run[
            "dataset"
        ].to_csv(
            output
            / "evaluation_input.csv",
            index=False,
        )

        q = run[
            "quality"
        ]

        write_manifest(
            out,
            ROOT,
            "Mode2",
            (
                list(
                    IRRIGATION_DATA
                    .glob("*.csv")
                )
                +
                [
                    SOURCE_DIR
                    /
                    (
                        f"sector_"
                        f"{args.sector}"
                        "_10min.csv"
                    ),

                    SOURCE_DIR
                    / "provenance.json",
                ]
            ),
            {
                "source":
                    "real",

                "scenario":
                    scenario,

                "sector":
                    args.sector,

                "start":
                    args.start,

                "end":
                    args.end,

                "input_is_synthetic":
                    False,

                "outcomes_modeled":
                    True,

                "config":
                    run[
                        "config"
                    ],

                "disturbance_source":
                    (
                        "pre-evaluation "
                        "training residual replay"
                    ),

                "tau":
                    0.02,

                "threshold_rule":
                    "close",

                "normalization":
                    "min-max",
            },
        )

        winners = (
            run[
                "result"
            ]
            .ranking
            .loc[
                run[
                    "result"
                ]
                .ranking[
                    "rank"
                ]
                == 1,
                "alternative",
            ]
            .tolist()
        )

        summary.append(
            {
                "scenario":
                    scenario,

                "winners":
                    winners,

                "source":
                    "real",

                "input_is_synthetic":
                    False,

                "outcomes_modeled":
                    True,
            }
        )

        print(
            f"\n{scenario} "
            "| source=real "
            "| input_is_synthetic=False "
            "| outcomes_modeled=True"
        )

        print(
            f"Sector {args.sector}: "
            f"{args.start}..{args.end}, "
            f"{len(run['dataset'])} days "
            "| L / probe-index points"
        )

        print(
            "Holdout MAE="
            f"{q['holdout_mae_points']:.4f}; "
            "persistence MAE="
            f"{q['persistence_mae_points']:.4f}"
        )

        print(
            "Training residual disturbance: "
            f"n={q['disturbance_library_rows']}, "
            f"replay={q['disturbance_sequence_rows']}, "
            f"MAE="
            f"{q['disturbance_mae_points']:.4f}"
        )

        print(
            run[
                "result"
            ]
            .ranking[
                [
                    "alternative",
                    "H",
                    "rank",
                ]
            ]
            .to_string(
                index=False
            )
        )

    (
        output
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()