import pandas as pd

from src.core.fuzzy import (
    HyFS,
    score,
    scalar_multiply,
    hyfwa,
    defuzzify,
)


def build_linguistic_map(mapping_df: pd.DataFrame) -> dict[str, HyFS]:
    """
    Convert linguistic_mapping.csv into:

        {
            "VI": HyFS(...),
            "I": HyFS(...),
            ...
        }

    The numeric HyFS values come from CSV, not hard-coded here.
    """

    linguistic_map = {}

    for _, row in mapping_df.iterrows():
        linguistic_map[row["code"]] = HyFS(
            mu=float(row["mu"]),
            nu=float(row["nu"]),
        )

    return linguistic_map


def compute_hy_fwzic(
    expert_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce Hy-FWZIC criteria weighting.

    Parameters
    ----------
    expert_df:
        Table 4 expert judgments loaded from CSV.

    mapping_df:
        Linguistic -> HyFS mapping loaded from CSV.

    Returns
    -------
    pd.DataFrame
        criterion, mu, nu, crisp, weight, rank
    """

    linguistic_map = build_linguistic_map(mapping_df)

    # All columns except "criterion" are decision makers
    expert_columns = [
        column
        for column in expert_df.columns
        if column != "criterion"
    ]

    # ---------------------------------------------------------
    # STEP 1
    # Calculate total score for each decision maker
    # ---------------------------------------------------------

    expert_total_scores = {}

    for expert in expert_columns:

        total_score = 0.0

        for linguistic_value in expert_df[expert]:

            h = linguistic_map[linguistic_value]

            total_score += score(h)

        expert_total_scores[expert] = total_score

    # ---------------------------------------------------------
    # STEP 2
    # Equal expert weights for reproduction
    #
    # Do NOT hard-code [0.2, 0.2, ...].
    # Determine automatically from number of experts.
    # ---------------------------------------------------------

    number_of_experts = len(expert_columns)

    expert_weights = [
        1.0 / number_of_experts
        for _ in expert_columns
    ]

    # ---------------------------------------------------------
    # STEP 3
    # Process each criterion
    # ---------------------------------------------------------

    results = []

    for _, row in expert_df.iterrows():

        criterion = row["criterion"]

        scaled_values = []

        for expert in expert_columns:

            linguistic_value = row[expert]

            original_hyfs = linguistic_map[linguistic_value]

            # lambda = reciprocal of total expert score
            lam = 1.0 / expert_total_scores[expert]

            scaled_hyfs = scalar_multiply(
                lam,
                original_hyfs,
            )

            scaled_values.append(scaled_hyfs)

        # Aggregate all experts using HyFWA
        aggregated = hyfwa(
            scaled_values,
            expert_weights,
        )

        # Defuzzification
        crisp_value = defuzzify(aggregated)

        results.append(
            {
                "criterion": criterion,
                "mu": aggregated.mu,
                "nu": aggregated.nu,
                "crisp": crisp_value,
            }
        )

    result_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # STEP 4
    # Normalize crisp values to final criteria weights
    # ---------------------------------------------------------

    crisp_sum = result_df["crisp"].sum()

    result_df["weight"] = (
        result_df["crisp"] / crisp_sum
    )

    # Highest weight = rank 1
    result_df["rank"] = (
        result_df["weight"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return result_df