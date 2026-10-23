from pathlib import Path

import pandas as pd

from src.core.hy_fwzic import compute_hy_fwzic
from src.paper.load_paper_data import (
    load_linguistic_mapping,
    load_expert_judgments,
    load_expected_weights,
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper"


# =========================================================
# HY-FWZIC PAPER REPRODUCTION
# =========================================================

def reproduce_hy_fwzic() -> pd.DataFrame:
    """
    Reproduce the Hy-FWZIC criteria weights reported in the paper.

    Pipeline:

        Paper input CSV
            ↓
        Hy-FWZIC implementation
            ↓
        Actual results
            ↓
        Compare with paper expected results
            ↓
        PASS / FAIL
            ↓
        Save comparison CSV

    Returns
    -------
    pd.DataFrame
        Comparison table containing actual and expected results.
    """

    # -----------------------------------------------------
    # STEP 1
    # Load input data reconstructed from the paper
    # -----------------------------------------------------

    mapping_df = load_linguistic_mapping()

    expert_df = load_expert_judgments()

    # -----------------------------------------------------
    # STEP 2
    # Load expected results reported in Table 5
    # -----------------------------------------------------

    expected_df = load_expected_weights()

    # -----------------------------------------------------
    # STEP 3
    # Run our Hy-FWZIC implementation
    # -----------------------------------------------------

    actual_df = compute_hy_fwzic(
        expert_df=expert_df,
        mapping_df=mapping_df,
    )

    # -----------------------------------------------------
    # STEP 4
    # Rename actual result columns
    # -----------------------------------------------------

    actual_df = actual_df.rename(
        columns={
            "mu": "actual_mu",
            "nu": "actual_nu",
            "crisp": "actual_crisp",
            "weight": "actual_weight",
            "rank": "actual_rank",
        }
    )

    # -----------------------------------------------------
    # STEP 5
    # Merge actual results with expected results
    # -----------------------------------------------------

    comparison_df = actual_df.merge(
        expected_df,
        on="criterion",
        how="inner",
        validate="one_to_one",
    )

    # -----------------------------------------------------
    # STEP 6
    # Check whether all expected criteria were reproduced
    # -----------------------------------------------------

    if len(comparison_df) != len(expected_df):
        raise ValueError(
            "The number of reproduced criteria does not match "
            "the number of expected criteria."
        )

    # -----------------------------------------------------
    # STEP 7
    # Calculate reproduction differences
    # -----------------------------------------------------

    comparison_df["weight_difference"] = (
        comparison_df["actual_weight"]
        - comparison_df["expected_weight"]
    ).abs()

    comparison_df["rank_match"] = (
        comparison_df["actual_rank"]
        == comparison_df["expected_rank"]
    )

    # Numerical tolerance used for weight comparison
    tolerance = 1e-6

    comparison_df["weight_match"] = (
        comparison_df["weight_difference"]
        <= tolerance
    )

    # Criterion passes only when:
    # - weight matches
    # - rank matches
    comparison_df["status"] = (
        comparison_df["weight_match"]
        & comparison_df["rank_match"]
    ).map(
        {
            True: "PASS",
            False: "FAIL",
        }
    )

    # -----------------------------------------------------
    # STEP 8
    # Calculate summary BEFORE removing helper columns
    #
    # Important:
    # weight_match and rank_match still exist here.
    # This avoids the previous KeyError.
    # -----------------------------------------------------

    max_difference = comparison_df[
        "weight_difference"
    ].max()

    all_weights_match = comparison_df[
        "weight_match"
    ].all()

    all_ranks_match = comparison_df[
        "rank_match"
    ].all()

    overall_pass = (
        all_weights_match
        and all_ranks_match
    )

    # -----------------------------------------------------
    # STEP 9
    # Select columns for display and CSV output
    # -----------------------------------------------------

    comparison_df = comparison_df[
        [
            "criterion",
            "actual_mu",
            "actual_nu",
            "actual_crisp",
            "actual_weight",
            "expected_weight",
            "weight_difference",
            "actual_rank",
            "expected_rank",
            "status",
        ]
    ]

    # -----------------------------------------------------
    # STEP 10
    # Prepare formatted version for terminal display
    # -----------------------------------------------------

    display_df = comparison_df.copy()

    numeric_columns = [
        "actual_mu",
        "actual_nu",
        "actual_crisp",
        "actual_weight",
        "expected_weight",
        "weight_difference",
    ]

    for column in numeric_columns:
        display_df[column] = display_df[column].map(
            lambda value: f"{value:.9f}"
        )

    # -----------------------------------------------------
    # STEP 11
    # Print comparison table
    # -----------------------------------------------------

    print()
    print("=" * 120)
    print("MODE 1 - Hy-FWZIC Paper Reproduction")
    print("=" * 120)

    print(
        display_df.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # STEP 12
    # Print reproduction summary
    # -----------------------------------------------------

    print()
    print("-" * 120)
    print("REPRODUCTION SUMMARY")
    print("-" * 120)

    print(
        f"Criteria reproduced : "
        f"{len(comparison_df)}"
    )

    print(
        f"Maximum weight error: "
        f"{max_difference:.12f}"
    )

    print(
        f"Weights match       : "
        f"{all_weights_match}"
    )

    print(
        f"Ranks match         : "
        f"{all_ranks_match}"
    )

    if overall_pass:
        print(
            "Overall result      : PASS "
            "- Hy-FWZIC reproduction matches the paper."
        )
    else:
        print(
            "Overall result      : FAIL "
            "- Differences detected."
        )

    # -----------------------------------------------------
    # STEP 13
    # Save reproduction results to CSV
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "hy_fwzic_reproduction.csv"
    )

    comparison_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved result to     : {output_path}"
    )

    print("=" * 120)
    print()

    # -----------------------------------------------------
    # Return DataFrame for possible future reuse
    # -----------------------------------------------------

    return comparison_df


# =========================================================
# MAIN
# =========================================================

def main():
    reproduce_hy_fwzic()


if __name__ == "__main__":
    main()