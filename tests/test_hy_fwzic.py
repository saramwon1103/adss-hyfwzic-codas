import numpy as np

from src.core.hy_fwzic import compute_hy_fwzic
from src.paper.load_paper_data import (
    load_linguistic_mapping,
    load_expert_judgments,
    load_expected_weights,
)


def test_paper_hy_fwzic_reproduction():

    # ---------------------------------------
    # ARRANGE
    # Load input data from CSV
    # ---------------------------------------

    mapping_df = load_linguistic_mapping()

    expert_df = load_expert_judgments()

    expected_df = load_expected_weights()

    # ---------------------------------------
    # ACT
    # Run our Hy-FWZIC implementation
    # ---------------------------------------

    actual_df = compute_hy_fwzic(
        expert_df=expert_df,
        mapping_df=mapping_df,
    )

    # ---------------------------------------
    # ASSERT
    # Compare our result with paper result
    # ---------------------------------------

    comparison = actual_df.merge(
        expected_df,
        on="criterion",
        how="inner",
    )

    assert len(comparison) == len(expected_df)

    assert np.allclose(
        comparison["weight"],
        comparison["expected_weight"],
        atol=1e-6,
    )

    assert np.array_equal(
        comparison["rank"],
        comparison["expected_rank"],
    )