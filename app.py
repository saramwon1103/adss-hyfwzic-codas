"""Local Streamlit UI; no hosted service or external credentials required."""

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from src.core.codas import crisp_codas
from src.irrigation.scenario_generator import SCENARIOS
from src.pipelines import run_mode1, run_mode2
from src.visualization.tables import winner_names


# ============================================================
# Sensitivity helpers
# ============================================================

def reweight_one_criterion(
    weights,
    criterion,
    new_weight,
):
    """
    Set one criterion to a requested weight and redistribute
    the remaining mass proportionally across all other criteria.

    The result always sums to 1.
    """

    weights = (
        weights
        .astype(float)
        .copy()
    )

    if criterion not in weights.index:
        raise ValueError(
            f"Unknown criterion: {criterion}"
        )

    new_weight = float(
        new_weight
    )

    if not 0.0 < new_weight < 1.0:
        raise ValueError(
            "Sensitivity weight must be "
            "between 0 and 1."
        )

    others = (
        weights
        .drop(
            index=criterion
        )
    )

    old_other_total = float(
        others.sum()
    )

    if old_other_total <= 0:
        raise ValueError(
            "Cannot redistribute weights because "
            "the remaining criteria have zero "
            "total weight."
        )

    remaining_mass = (
        1.0
        - new_weight
    )

    adjusted = (
        others
        / old_other_total
        * remaining_mass
    )

    result = (
        weights.copy()
    )

    result.loc[
        criterion
    ] = new_weight

    result.loc[
        adjusted.index
    ] = adjusted

    # Protect against tiny floating-point drift.
    result = (
        result
        / result.sum()
    )

    return result


def ranking_comparison(
    baseline,
    sensitivity,
):
    """
    Compare baseline CODAS ranking with a sensitivity run.
    """

    left = (
        baseline[
            [
                "alternative",
                "H",
                "rank",
            ]
        ]
        .rename(
            columns={
                "H":
                    "baseline_H",

                "rank":
                    "baseline_rank",
            }
        )
    )

    right = (
        sensitivity[
            [
                "alternative",
                "H",
                "rank",
            ]
        ]
        .rename(
            columns={
                "H":
                    "sensitivity_H",

                "rank":
                    "sensitivity_rank",
            }
        )
    )

    comparison = (
        left.merge(
            right,
            on="alternative",
            how="inner",
        )
    )

    comparison[
        "rank_change"
    ] = (
        comparison[
            "baseline_rank"
        ]
        -
        comparison[
            "sensitivity_rank"
        ]
    )

    return (
        comparison
        .sort_values(
            [
                "sensitivity_rank",
                "alternative",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="DSS | Thắng + Quyên",
    page_icon="🌱",
    layout="wide",
)

st.title(
    "Agriculture Decision Support Lab"
)

st.caption(
    "DSS coursework · Thắng + Quyên · "
    "paper reproduction + irrigation proof-of-concept"
)

mode = st.sidebar.radio(
    "Mode",
    [
        "Mode 1 · Paper reproduction",
        "Mode 2 · Irrigation simulation",
    ],
)

st.sidebar.markdown(
    "**Scope**: Hy-FWZIC, CODAS, scenarios, "
    "proxy strategies, KPI, sensitivity and validation."
)


# ============================================================
# MODE 1
# ============================================================

if mode.startswith(
    "Mode 1"
):

    st.header(
        "13 ADSS · Hy-FWZIC → Hyperbolic CODAS"
    )

    source = (
        st.sidebar.selectbox(
            "Weight source",
            [
                "published",
                "computed",
            ],
        )
    )

    tau = (
        st.sidebar.slider(
            "τ (paper Eq.16, close rule)",
            min_value=0.0,
            max_value=0.05,
            value=0.02,
            step=0.001,
        )
    )

    run = run_mode1(
        tau,
        source,
    )

    a, b, c = (
        st.columns(3)
    )

    a.metric(
        "Top alternative",
        run[
            "result"
        ]
        .ranking
        .iloc[0]
        .alternative,
    )

    b.metric(
        "Table 8 score matches",
        f"{run['comparison'].score_within_tolerance.sum()}/13",
    )   

    c.metric(
        "Table 10 score matches",
        f"{run['sensitivity_comparison'].score_within_tolerance.sum()}/104",
    )

    st.info(
        "Paper reference: DOI "
        "10.1016/j.compag.2024.109618. "
        "Tolerance 0.0011 for printed H; "
        "ties use competition ranking. "
        "τ=0.02 is a documented reproduction assumption."
    )

    t1, t2, t3, t4 = (
        st.tabs(
            [
                "Data & weights",
                "Ranking",
                "Sensitivity & stability",
                "Validation & audit",
            ]
        )
    )

    # --------------------------------------------------------
    # Mode 1 · Data & weights
    # --------------------------------------------------------

    with t1:

        st.dataframe(
            run[
                "matrix"
            ],
            width="stretch",
        )

        st.bar_chart(
            run[
                "weights"
            ]
        )

        st.dataframe(
            run[
                "weight_comparison"
            ],
            width="stretch",
        )

    # --------------------------------------------------------
    # Mode 1 · Ranking
    # --------------------------------------------------------

    with t2:

        st.dataframe(
            run[
                "result"
            ].ranking,
            hide_index=True,
            width="stretch",
        )

        st.bar_chart(
            run[
                "result"
            ]
            .ranking
            .set_index(
                "alternative"
            )
            .H
        )

    # --------------------------------------------------------
    # Mode 1 · Sensitivity
    # --------------------------------------------------------

    with t3:

        choice = (
            st.selectbox(
                "Weight scenario",
                run[
                    "sensitivity_summary"
                ].scenario,
            )
        )

        st.dataframe(
            run[
                "sensitivity_summary"
            ],
            hide_index=True,
            width="stretch",
        )

        st.dataframe(
            run[
                "sensitivity"
            ].query(
                "scenario == @choice"
            ),
            hide_index=True,
            width="stretch",
        )

    # --------------------------------------------------------
    # Mode 1 · Validation
    # --------------------------------------------------------

    with t4:

        st.dataframe(
            run[
                "comparison"
            ],
            hide_index=True,
            width="stretch",
        )

        st.dataframe(
            run[
                "sensitivity_comparison"
            ],
            hide_index=True,
            width="stretch",
        )

        st.dataframe(
            run[
                "result"
            ].relative,
            width="stretch",
        )

        st.download_button(
            "Download validation CSV",
            run[
                "comparison"
            ].to_csv(
                index=False
            ),
            "paper_validation.csv",
            "text/csv",
        )


# ============================================================
# MODE 2
# ============================================================

else:

    st.header(
        "Uploaded Arnesano data · open-field tomato"
    )

    # --------------------------------------------------------
    # Sidebar controls
    # --------------------------------------------------------

    scenario = (
        st.sidebar.selectbox(
            "Operational scenario",
            SCENARIOS,
        )
    )

    sector = (
        st.sidebar.selectbox(
            "Tomato sector",
            [
                1,
                2,
            ],
        )
    )

    period = (
        st.sidebar.date_input(
            "Evaluation period",
            (
                date(
                    2025,
                    6,
                    1,
                ),
                date(
                    2025,
                    7,
                    13,
                ),
            ),
            min_value=date(
                2025,
                4,
                1,
            ),
            max_value=date(
                2025,
                9,
                24,
            ),
        )
    )

    if len(
        period
    ) != 2:

        st.info(
            "Select both evaluation "
            "start and end."
        )

        st.stop()

    # --------------------------------------------------------
    # Execute Mode 2
    # --------------------------------------------------------

    try:

        run = (
            run_mode2(
                scenario,
                sector=sector,
                start=(
                    period[
                        0
                    ].isoformat()
                ),
                end=(
                    period[
                        1
                    ].isoformat()
                ),
            )
        )

    except (
        ValueError,
        FileNotFoundError,
    ) as error:

        st.error(
            str(
                error
            )
        )

        st.stop()

    q = (
        run[
            "quality"
        ]
    )

    # --------------------------------------------------------
    # Research-integrity notices
    # --------------------------------------------------------

    st.info(
        "REAL INPUT DATA: uploaded sensor/valve series plus "
        "historical reanalysis weather. Volumes are nominal-flow "
        "litres; soil readings are relative probe-index points, "
        "not calibrated VWC. No mock fallback."
    )

    st.warning(
        "MODELLED OUTCOMES: the actions of SIDSS-like, FDSS-like "
        "and MRGCD-like were not observed in the field. "
        "They are reconstructed proxy strategies inspired by "
        "the original ADSS descriptions, not the original systems. "
        "Expert preferences remain illustrative. "
        "Drought and sensor-outage cases are what-if transformations."
    )

    if not q[
        "model_beats_persistence_mae"
    ]:

        st.warning(
            "Validation: the fitted response model does not beat "
            "simple persistence on this holdout. "
            "Treat rankings as exploratory."
        )

    if q[
        "weak_irrigation_response"
    ]:

        st.caption(
            "The learned irrigation response is weak. "
            "Small score differences and ties do not establish "
            "a reliable field winner."
        )

    if (
        max(
            q[
                "out_of_training_range_steps"
            ].values()
        )
        > 0
    ):

        st.warning(
            "Extrapolation: some simulated inputs are outside "
            "the training feature ranges; see Source & model "
            "validation for counts. Rankings are exploratory."
        )

    # --------------------------------------------------------
    # Main headline metrics
    # --------------------------------------------------------

    winners = (
        winner_names(
            run[
                "result"
            ]
        )
    )

    a, b, c = (
        st.columns(3)
    )

    a.metric(
        "Top alternative(s)",
        ", ".join(
            winners
        ),
    )

    b.metric(
        "Daily steps",
        len(
            run[
                "dataset"
            ]
        ),
    )

    c.metric(
        "Sector",
        (
            f"ARNESANO_TOMATO_"
            f"{sector}"
        ),
    )

    # --------------------------------------------------------
    # Mode 2 tabs
    # --------------------------------------------------------

    (
        t1,
        t2,
        t3,
        t4,
        t5,
        t6,
    ) = st.tabs(
        [
            "Experiment flow",
            "Data & weights",
            "KPI & ranking",
            "Scenario traces",
            "Sensitivity lab",
            "Source & model validation",
        ]
    )

    # ========================================================
    # TAB 1 · EXPERIMENT FLOW
    # ========================================================

    with t1:

        st.subheader(
            "Mode 2 decision flow"
        )

        st.caption(
            "Follow the complete path from real operational "
            "input to the final MCDM ranking."
        )

        f1, f2, f3, f4 = (
            st.columns(4)
        )

        f1.metric(
            "Scenario",
            scenario,
        )

        f2.metric(
            "Evaluation days",
            len(
                run[
                    "dataset"
                ]
            ),
        )

        f3.metric(
            "Training residuals",
            len(
                run[
                    "training_residuals"
                ]
            ),
        )

        f4.metric(
            "Disturbance replay",
            len(
                run[
                    "disturbances"
                ]
            ),
        )

        st.markdown(
            """
### 1 · Real operational input

Sensor, valve and reanalysis-weather data are loaded for the
selected sector and evaluation period.

↓

### 2 · Preprocessing

Raw observations are converted to daily decision inputs.
Weather forecasts use one-day persistence.

↓

### 3 · Soil-response surrogate

A constrained chronological ridge model is fitted only using
transitions strictly before the evaluation period.

↓

### 4 · Training-residual disturbance replay

Pre-evaluation training residuals are used as a deterministic
environment-disturbance sequence.

Evaluation-period future soil-moisture outcomes are **not**
used to construct this sequence.

↓

### 5 · Run three proxy irrigation strategies

The same scenario and same disturbance sequence are passed to:

- SIDSS-like
- FDSS-like
- MRGCD-like

↓

### 6 · KPI evaluation

Each strategy is converted to a performance profile:

- water use
- soil-moisture MAE
- over-irrigation
- replanning response
- missing-data robustness

↓

### 7 · Hy-FWZIC criteria weighting

Illustrative linguistic decision-maker preferences are converted
to criterion weights.

↓

### 8 · Baseline CODAS ranking

The KPI performance matrix and criterion weights are combined
to obtain the baseline ranking.

↓

### 9 · Sensitivity analysis

Criterion weights are perturbed and CODAS is executed again.

↓

### 10 · Stability interpretation

The new ranking is compared with the baseline to identify:

- stable winners
- ties
- rank changes
- possible rank reversal
"""
        )

        # ----------------------------------------------------
        # Step details
        # ----------------------------------------------------

        with st.expander(
            "Step 1–2 · Scenario input and preprocessing"
        ):

            st.caption(
                f"Evaluation: "
                f'{q["evaluation_start"]} '
                f'to {q["evaluation_end"]}.'
            )

            st.dataframe(
                run[
                    "dataset"
                ].head(
                    20
                ),
                hide_index=True,
                width="stretch",
            )

        with st.expander(
            "Step 3–4 · Surrogate and disturbance"
        ):

            e1, e2, e3 = (
                st.columns(3)
            )

            e1.metric(
                "Holdout MAE",
                (
                    f'{q["holdout_mae_points"]:.3f}'
                ),
            )

            e2.metric(
                "Persistence MAE",
                (
                    f'{q["persistence_mae_points"]:.3f}'
                ),
            )

            e3.metric(
                "Disturbance MAE",
                (
                    f'{q["disturbance_mae_points"]:.3f}'
                ),
            )

            st.markdown(
                "**Training residual library**"
            )

            st.dataframe(
                run[
                    "training_residuals"
                ].head(
                    20
                ),
                hide_index=True,
                width="stretch",
            )

            disturbance_frame = (
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
                )
            )

            st.markdown(
                "**Disturbance sequence used in the evaluation**"
            )

            st.dataframe(
                disturbance_frame,
                hide_index=True,
                width="stretch",
            )

            st.line_chart(
                disturbance_frame
                .set_index(
                    "step"
                )
            )

        with st.expander(
            "Step 5 · Strategy simulation"
        ):

            strategy_trace = (
                pd.concat(
                    run[
                        "traces"
                    ].values(),
                    ignore_index=True,
                )
            )

            trace_columns = [
                "timestamp",
                "alternative",
                "moisture_start_index",
                "moisture_observation_index",
                "irrigation_liters",
                "environment_disturbance_points",
                "moisture_end_index",
                "used_fallback",
            ]

            available_trace_columns = [
                col
                for col
                in trace_columns
                if col
                in strategy_trace.columns
            ]

            st.dataframe(
                strategy_trace[
                    available_trace_columns
                ].head(
                    50
                ),
                hide_index=True,
                width="stretch",
            )

        with st.expander(
            "Step 6 · KPI performance matrix"
        ):

            st.dataframe(
                run[
                    "matrix"
                ],
                width="stretch",
            )

        with st.expander(
            "Step 7 · Hy-FWZIC weights"
        ):

            st.dataframe(
                run[
                    "weights"
                ]
                .rename(
                    "weight"
                )
                .to_frame(),
                width="stretch",
            )

            st.bar_chart(
                run[
                    "weights"
                ]
            )

        with st.expander(
            "Step 8 · Baseline CODAS ranking"
        ):

            st.dataframe(
                run[
                    "result"
                ].ranking,
                hide_index=True,
                width="stretch",
            )

            st.bar_chart(
                run[
                    "result"
                ]
                .ranking
                .set_index(
                    "alternative"
                )
                .H
            )

        st.info(
            "Continue to the Sensitivity lab tab to perturb one "
            "criterion weight and observe how CODAS re-ranks the "
            "three alternatives."
        )

    # ========================================================
    # TAB 2 · DATA & WEIGHTS
    # ========================================================

    with t2:

        st.caption(
            f'Evaluation: {q["evaluation_start"]} '
            f'to {q["evaluation_end"]}. '
            "Forecasts are previous-day persistence. "
            "Model fitted before evaluation."
        )

        st.subheader(
            "Scenario input"
        )

        st.dataframe(
            run[
                "dataset"
            ],
            hide_index=True,
            width="stretch",
        )

        st.subheader(
            "Criteria"
        )

        st.dataframe(
            run[
                "criteria"
            ],
            width="stretch",
        )

        st.caption(
            "Weights use illustrative decision-maker judgments. "
            "This dataset itself supplies no expert preferences."
        )

        st.subheader(
            "Hy-FWZIC weights"
        )

        st.bar_chart(
            run[
                "weights"
            ]
        )

        st.dataframe(
            run[
                "weights"
            ]
            .rename(
                "weight"
            )
            .to_frame(),
            width="stretch",
        )

        st.download_button(
            "Download scenario input CSV",
            run[
                "dataset"
            ].to_csv(
                index=False
            ),
            "scenario_input_real.csv",
            "text/csv",
        )

    # ========================================================
    # TAB 3 · KPI & RANKING
    # ========================================================

    with t3:

        st.subheader(
            "Performance matrix"
        )

        st.dataframe(
            run[
                "matrix"
            ],
            width="stretch",
        )

        constants = (
            run[
                "matrix"
            ]
            .columns[
                (
                    run[
                        "matrix"
                    ].max()
                    -
                    run[
                        "matrix"
                    ].min()
                )
                < 1e-10
            ]
            .tolist()
        )

        if constants:

            st.caption(
                "Non-discriminating KPI columns in this run: "
                +
                ", ".join(
                    constants
                )
            )

        st.caption(
            "Water = litres; soil error = probe-index points. "
            "Excess water and response are model-based proxies. "
            "Ties are retained; robustness=1 can mean no additional "
            "measured degradation under this paired stress test."
        )

        st.subheader(
            "Baseline CODAS ranking"
        )

        st.dataframe(
            run[
                "result"
            ].ranking,
            hide_index=True,
            width="stretch",
        )

        st.bar_chart(
            run[
                "result"
            ]
            .ranking
            .set_index(
                "alternative"
            )
            .H
        )

        st.download_button(
            "Download performance matrix",
            run[
                "matrix"
            ].to_csv(),
            "performance_matrix_real.csv",
            "text/csv",
        )

    # ========================================================
    # TAB 4 · SCENARIO TRACES
    # ========================================================

    with t4:

        traces = (
            pd.concat(
                run[
                    "traces"
                ].values(),
                ignore_index=True,
            )
        )

        st.caption(
            "Closed-loop surrogate trajectories, not measured "
            "field outcomes of these alternative policies."
        )

        st.subheader(
            "Moisture trajectory"
        )

        st.line_chart(
            traces.pivot(
                index="timestamp",
                columns="alternative",
                values="moisture_end_index",
            )
        )

        st.subheader(
            "Irrigation decisions"
        )

        st.bar_chart(
            traces.pivot(
                index="timestamp",
                columns="alternative",
                values="irrigation_liters",
            )
        )

        name = (
            st.selectbox(
                "Inspect strategy trace",
                list(
                    run[
                        "traces"
                    ]
                ),
            )
        )

        st.dataframe(
            run[
                "traces"
            ][
                name
            ],
            hide_index=True,
            width="stretch",
        )

    # ========================================================
    # TAB 5 · INTERACTIVE SENSITIVITY LAB
    # ========================================================

    with t5:

        st.subheader(
            "Interactive sensitivity experiment"
        )

        st.caption(
            "Change the importance of one criterion. "
            "The remaining criterion weights are redistributed "
            "proportionally and CODAS is executed again."
        )

        st.info(
            "This is a sensitivity weight perturbation. "
            "It does not represent a new expert elicitation or "
            "a new Hy-FWZIC survey."
        )

        baseline_weights = (
            run[
                "weights"
            ]
            .astype(float)
            .copy()
        )

        criterion = (
            st.selectbox(
                "Criterion to stress",
                list(
                    baseline_weights.index
                ),
                key=(
                    "mode2_sensitivity_criterion"
                ),
            )
        )

        baseline_weight = float(
            baseline_weights.loc[
                criterion
            ]
        )

        default_slider_weight = float(
            min(
                max(
                    baseline_weight,
                    0.05,
                ),
                0.60,
            )
        )

        new_weight = (
            st.slider(
                "New criterion weight",
                min_value=0.05,
                max_value=0.60,
                value=(
                    default_slider_weight
                ),
                step=0.01,
                key=(
                    "mode2_sensitivity_weight"
                ),
            )
        )

        sensitivity_weights = (
            reweight_one_criterion(
                baseline_weights,
                criterion,
                new_weight,
            )
        )

        sensitivity_result = (
            crisp_codas(
                run[
                    "matrix"
                ],
                sensitivity_weights,
                run[
                    "criteria"
                ].direction,
            )
        )

        comparison = (
            ranking_comparison(
                run[
                    "result"
                ].ranking,
                sensitivity_result.ranking,
            )
        )

        baseline_winners = (
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

        sensitivity_winners = (
            sensitivity_result
            .ranking
            .loc[
                sensitivity_result
                .ranking[
                    "rank"
                ]
                == 1,
                "alternative",
            ]
            .tolist()
        )

        winner_changed = (
            set(
                baseline_winners
            )
            !=
            set(
                sensitivity_winners
            )
        )

        rank_changed = bool(
            (
                comparison[
                    "baseline_rank"
                ]
                !=
                comparison[
                    "sensitivity_rank"
                ]
            ).any()
        )

        s1, s2, s3, s4 = (
            st.columns(4)
        )

        s1.metric(
            "Criterion",
            criterion,
        )

        s2.metric(
            "Baseline weight",
            f"{baseline_weight:.3f}",
        )

        s3.metric(
            "New weight",
            f"{new_weight:.3f}",
        )

        s4.metric(
            "Winner changed?",
            (
                "YES"
                if winner_changed
                else "NO"
            ),
        )

        # ----------------------------------------------------
        # Sensitivity flow
        # ----------------------------------------------------

        st.markdown(
            "### Sensitivity flow"
        )

        st.code(
            (
                f"Scenario: {scenario}\n"
                "\n"
                "KPI performance matrix\n"
                "        ↓\n"
                "Baseline Hy-FWZIC weights\n"
                "        ↓\n"
                "Baseline CODAS ranking\n"
                "        ↓\n"
                f"Stress criterion: {criterion}\n"
                f"{baseline_weight:.4f} "
                f"→ {new_weight:.4f}\n"
                "        ↓\n"
                "Redistribute remaining weight mass\n"
                "        ↓\n"
                "Run CODAS again\n"
                "        ↓\n"
                "Compare rank and winner stability"
            ),
            language=None,
        )

        # ----------------------------------------------------
        # Weight comparison
        # ----------------------------------------------------

        left, right = (
            st.columns(2)
        )

        with left:

            st.markdown(
                "**Baseline weights**"
            )

            st.dataframe(
                baseline_weights
                .rename(
                    "baseline_weight"
                )
                .to_frame(),
                width="stretch",
            )

        with right:

            st.markdown(
                "**Sensitivity weights**"
            )

            st.dataframe(
                sensitivity_weights
                .rename(
                    "sensitivity_weight"
                )
                .to_frame(),
                width="stretch",
            )

        # ----------------------------------------------------
        # Weight comparison table
        # ----------------------------------------------------

        weight_comparison = (
            pd.DataFrame(
                {
                    "baseline_weight":
                        baseline_weights,

                    "sensitivity_weight":
                        sensitivity_weights,
                }
            )
        )

        weight_comparison[
            "delta"
        ] = (
            weight_comparison[
                "sensitivity_weight"
            ]
            -
            weight_comparison[
                "baseline_weight"
            ]
        )

        st.markdown(
            "### Weight perturbation"
        )

        st.dataframe(
            weight_comparison,
            width="stretch",
        )

        # ----------------------------------------------------
        # Baseline ranking
        # ----------------------------------------------------

        st.markdown(
            "### Baseline CODAS"
        )

        st.dataframe(
            run[
                "result"
            ].ranking,
            hide_index=True,
            width="stretch",
        )

        # ----------------------------------------------------
        # Sensitivity ranking
        # ----------------------------------------------------

        st.markdown(
            "### CODAS after perturbation"
        )

        st.dataframe(
            sensitivity_result.ranking,
            hide_index=True,
            width="stretch",
        )

        # ----------------------------------------------------
        # Ranking comparison
        # ----------------------------------------------------

        st.markdown(
            "### Baseline → sensitivity comparison"
        )

        st.dataframe(
            comparison,
            hide_index=True,
            width="stretch",
        )

        st.bar_chart(
            sensitivity_result
            .ranking
            .set_index(
                "alternative"
            )
            .H
        )

        if winner_changed:

            st.warning(
                "Winner-set reversal detected: "
                "the rank-1 alternative set changed "
                "under this weight perturbation."
            )

        elif rank_changed:

            st.warning(
                "The winner set is unchanged, but at least "
                "one alternative changed rank."
            )

        else:

            st.success(
                "No ranking change under this "
                "weight perturbation."
            )

        # ----------------------------------------------------
        # Predefined sensitivity scenarios
        # ----------------------------------------------------

        with st.expander(
            "Predefined sensitivity scenarios"
        ):

            st.caption(
                "These are the automated sensitivity runs "
                "already generated by the Mode 2 pipeline."
            )

            st.dataframe(
                run[
                    "sensitivity_summary"
                ],
                hide_index=True,
                width="stretch",
            )

            st.dataframe(
                run[
                    "sensitivity"
                ],
                hide_index=True,
                width="stretch",
            )

    # ========================================================
    # TAB 6 · SOURCE & MODEL VALIDATION
    # ========================================================

    with t6:

        v1, v2, v3 = (
            st.columns(3)
        )

        v1.metric(
            "Holdout MAE (index points)",
            f'{q["holdout_mae_points"]:.3f}',
        )

        v2.metric(
            "Persistence MAE (index points)",
            f'{q["persistence_mae_points"]:.3f}',
        )

        v3.metric(
            "Valid holdout pairs",
            q[
                "holdout_pairs"
            ],
        )

        st.caption(
            "One-step replay uses historical water actions to "
            "validate predictions. Future holdout targets do not "
            "enter training or policy inputs."
        )

        validation = (
            run[
                "validation"
            ]
            .set_index(
                "timestamp"
            )
        )

        st.line_chart(
            validation[
                [
                    "observed_next_index",
                    "predicted_next_index",
                    "persistence_next_index",
                ]
            ]
        )

        st.subheader(
            "Holdout validation table"
        )

        st.dataframe(
            run[
                "validation"
            ],
            hide_index=True,
            width="stretch",
        )

        st.subheader(
            "Data-quality and simulation audit"
        )

        st.json(
            q
        )

        st.subheader(
            "Soil-response surrogate"
        )

        st.json(
            run[
                "model"
            ].to_dict()
        )