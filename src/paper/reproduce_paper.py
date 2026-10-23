"""Integrated Mode 1 reproduction: Hy-FWZIC -> CODAS -> sensitivity."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.sensitivity import analyze
from src.common.outputs import export_codas
from src.config import PAPER_OUTPUT
from src.core.codas import hyperbolic_codas
from src.core.hy_fwzic import compute_hy_fwzic
from src.paper.load_paper_data import (
    load_expected_weights,
    load_expert_judgments,
    load_linguistic_mapping,
    load_mode1,
)
from src.paper.validate_paper import compare_results


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PAPER_OUTPUT


def _weight_reproduction_table(actual_df: pd.DataFrame) -> pd.DataFrame:
    """Compare calculated Hy-FWZIC weights with Table 5 expected values."""
    expected_df = load_expected_weights()
    actual = actual_df.rename(
        columns={
            "mu": "actual_mu",
            "nu": "actual_nu",
            "crisp": "actual_crisp",
            "weight": "actual_weight",
            "rank": "actual_rank",
        }
    )
    comparison = actual.merge(expected_df, on="criterion", how="inner", validate="one_to_one")
    if len(comparison) != len(expected_df):
        raise ValueError("The number of reproduced criteria does not match expected criteria")

    comparison["weight_difference"] = (
        comparison["actual_weight"] - comparison["expected_weight"]
    ).abs()
    comparison["weight_match"] = comparison["weight_difference"] <= 1e-6
    comparison["rank_match"] = comparison["actual_rank"] == comparison["expected_rank"]
    comparison["status"] = (comparison["weight_match"] & comparison["rank_match"]).map(
        {True: "PASS", False: "FAIL"}
    )
    return comparison


def reproduce_hy_fwzic(save: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Reproduce Table 5 and optionally print/save a readable comparison."""
    actual_df = compute_hy_fwzic(
        expert_df=load_expert_judgments(),
        mapping_df=load_linguistic_mapping(),
    )
    full = _weight_reproduction_table(actual_df)

    display_columns = [
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
    comparison = full[display_columns].copy()

    if verbose:
        display = comparison.copy()
        for column in [
            "actual_mu",
            "actual_nu",
            "actual_crisp",
            "actual_weight",
            "expected_weight",
            "weight_difference",
        ]:
            display[column] = display[column].map(lambda value: f"{value:.9f}")

        max_difference = float(full["weight_difference"].max())
        all_weights_match = bool(full["weight_match"].all())
        all_ranks_match = bool(full["rank_match"].all())

        print("\n" + "=" * 120)
        print("MODE 1 - Hy-FWZIC Paper Reproduction")
        print("=" * 120)
        print(display.to_string(index=False))
        print("\n" + "-" * 120)
        print("REPRODUCTION SUMMARY")
        print("-" * 120)
        print(f"Criteria reproduced : {len(comparison)}")
        print(f"Maximum weight error: {max_difference:.12f}")
        print(f"Weights match       : {all_weights_match}")
        print(f"Ranks match         : {all_ranks_match}")
        print(
            "Overall result      : "
            + ("PASS - Hy-FWZIC reproduction matches the paper." if all_weights_match and all_ranks_match
               else "FAIL - Differences detected.")
        )

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "hy_fwzic_reproduction.csv"
        comparison.to_csv(output_path, index=False)
        if verbose:
            print(f"\nSaved result to     : {output_path}")
            print("=" * 120 + "\n")

    return comparison


def run_mode1(tau: float = 0.02, source: str = "published") -> dict:
    """Run the complete Mode 1 paper reproduction pipeline.

    Parameters
    ----------
    tau:
        Eq. (16) close-threshold used by the reproduced CODAS implementation.
    source:
        ``published`` uses Table 5 weights as the CODAS input; ``computed``
        uses weights independently recomputed from Table 4 via Hy-FWZIC.

    Returns
    -------
    dict
        Inputs, calculated weights, CODAS result, sensitivity results and
        validation tables consumed by tests/UI.
    """
    if source not in {"published", "computed"}:
        raise ValueError("source must be 'published' or 'computed'")

    data = load_mode1()
    criteria = data["criteria"]

    computed_weights = compute_hy_fwzic(
        data["expert_judgments"], data["linguistic_mapping"]
    )
    computed_series = computed_weights.set_index("criterion")["weight"]
    published_series = criteria["weight"]
    weights = published_series if source == "published" else computed_series

    rank_function = lambda w: hyperbolic_codas(
        data["matrix"],
        w,
        criteria["direction"],
        data["mapping"],
        tau=tau,
        threshold_rule="close",
    )

    result = rank_function(weights)
    sensitivity, sensitivity_summary = analyze(
        rank_function,
        weights,
        data["sensitivity_scenarios"],
    )
    comparison, sensitivity_comparison, weight_comparison = compare_results(
        result,
        sensitivity,
        computed_weights,
        criteria,
    )

    return {
        **data,
        "source": source,
        "tau": tau,
        "computed_weights": computed_weights,
        "weights": weights,
        "result": result,
        "comparison": comparison,
        "sensitivity": sensitivity,
        "sensitivity_summary": sensitivity_summary,
        "sensitivity_comparison": sensitivity_comparison,
        "weight_comparison": weight_comparison,
    }


def _export_mode1(run: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    export_codas(run["result"], output, run["matrix"].columns)
    run["computed_weights"].to_csv(output / "computed_weights.csv", index=False)
    run["comparison"].to_csv(output / "ranking_validation.csv", index=False)
    run["sensitivity"].to_csv(output / "sensitivity.csv", index=False)
    run["sensitivity_summary"].to_csv(output / "stability.csv", index=False)
    run["sensitivity_comparison"].to_csv(output / "sensitivity_validation.csv", index=False)
    run["weight_comparison"].to_csv(output / "weight_validation.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", choices=["published", "computed"], default="published")
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    # Keep Quyên's explicit Table 5 audit visible in the CLI.
    reproduce_hy_fwzic(save=True, verbose=True)

    run = run_mode1(tau=args.tau, source=args.weights)
    _export_mode1(run, args.output.resolve())

    print("MODE 1 - CODAS / Sensitivity")
    print(f"Weight source       : {args.weights}")
    print(f"Top alternative     : {run['result'].ranking.iloc[0].alternative}")
    print(f"Table 8 score match : {int(run['comparison'].score_within_tolerance.sum())}/13")
    print(
        "Table 10 score match: "
        f"{int(run['sensitivity_comparison'].score_within_tolerance.sum())}/104"
    )
    print(f"Saved integrated outputs to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
