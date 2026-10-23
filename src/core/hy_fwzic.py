"""Hy-FWZIC criterion weighting used by paper reproduction and Mode 2."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.fuzzy import HyFS, defuzzify, hyfwa, scalar_multiply, score


def build_linguistic_map(mapping_df: pd.DataFrame) -> dict[str, HyFS]:
    """Build ``linguistic code -> HyFS`` from the mapping CSV."""
    required = {"code", "mu", "nu"}
    missing = required.difference(mapping_df.columns)
    if missing:
        raise ValueError(f"linguistic mapping missing columns: {sorted(missing)}")
    if mapping_df["code"].duplicated().any():
        raise ValueError("linguistic mapping codes must be unique")

    return {
        str(row["code"]): HyFS(mu=float(row["mu"]), nu=float(row["nu"]))
        for _, row in mapping_df.iterrows()
    }


def _normalize_expert_matrix(
    expert_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize supported expert-matrix layouts to criterion x expert codes.

    Supported inputs:
    - Paper layout: ``criterion, DM1, DM2, ...`` with VI/I/MI/SI/LI values.
    - Mode 2 layout: expert index/rows and criterion columns, with either
      linguistic codes or numeric Likert values 1..5.
    """
    if expert_df.empty:
        raise ValueError("expert judgments must not be empty")

    codes = set(mapping_df["code"].astype(str))
    numeric_to_code = {}
    if "numerical_value" in mapping_df.columns:
        numeric_to_code = {
            float(row["numerical_value"]): str(row["code"])
            for _, row in mapping_df.iterrows()
        }

    # Paper layout is already criterion x expert.
    if "criterion" in expert_df.columns:
        normalized = expert_df.copy()
        if normalized["criterion"].duplicated().any():
            raise ValueError("criterion labels must be unique")
        value_columns = [c for c in normalized.columns if c != "criterion"]
    else:
        # Mode 2 preferences are expert x criterion; transpose them.
        if expert_df.columns.duplicated().any():
            raise ValueError("criterion labels must be unique")
        if expert_df.index.has_duplicates:
            raise ValueError("expert labels must be unique")
        normalized = expert_df.T.reset_index().rename(columns={"index": "criterion"})
        value_columns = [c for c in normalized.columns if c != "criterion"]

    if not value_columns:
        raise ValueError("at least one expert is required")

    def to_code(value):
        if pd.isna(value):
            raise ValueError("expert judgments must not contain missing values")
        text = str(value).strip()
        if text in codes:
            return text
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown expert judgment: {value!r}") from exc
        if number not in numeric_to_code:
            raise ValueError(f"numeric expert judgment not declared in mapping: {value!r}")
        return numeric_to_code[number]

    normalized[value_columns] = normalized[value_columns].map(to_code)
    return normalized


def compute_hy_fwzic(
    expert_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Hy-FWZIC weights from expert judgments.

    The reproduction uses equal expert aggregation weights because the supplied
    paper defines expert weights in Eq. (6) but does not publish separate
    values for the five decision makers.  This assumption is documented by the
    project and reproduces Table 5.
    """
    normalized = _normalize_expert_matrix(expert_df, mapping_df)
    linguistic_map = build_linguistic_map(mapping_df)
    expert_columns = [c for c in normalized.columns if c != "criterion"]

    # Eq. (2) totals per expert, used by Eq. (4) scaling.
    expert_total_scores: dict[object, float] = {}
    for expert in expert_columns:
        total = sum(score(linguistic_map[value]) for value in normalized[expert])
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"invalid total score for expert {expert!r}: {total}")
        expert_total_scores[expert] = float(total)

    number_of_experts = len(expert_columns)
    expert_weights = [1.0 / number_of_experts] * number_of_experts

    results = []
    for _, row in normalized.iterrows():
        scaled_values = []
        for expert in expert_columns:
            original = linguistic_map[row[expert]]
            lam = 1.0 / expert_total_scores[expert]
            scaled_values.append(scalar_multiply(lam, original))

        aggregated = hyfwa(scaled_values, expert_weights)
        results.append(
            {
                "criterion": str(row["criterion"]),
                "mu": aggregated.mu,
                "nu": aggregated.nu,
                "crisp": defuzzify(aggregated),
            }
        )

    result_df = pd.DataFrame(results)
    crisp_sum = float(result_df["crisp"].sum())
    if not np.isfinite(crisp_sum) or crisp_sum <= 0:
        raise ValueError("sum of crisp criterion values must be positive and finite")

    result_df["weight"] = result_df["crisp"] / crisp_sum
    result_df["rank"] = (
        result_df["weight"].rank(ascending=False, method="min").astype(int)
    )
    return result_df


def compute_weights(
    expert_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compatibility/public API for criterion weighting.

    Mode 1 can pass the paper's linguistic matrix; Mode 2 can pass an
    expert-by-criterion matrix containing numeric Likert values.  Both routes
    use the exact same Hy-FWZIC implementation above.
    """
    return compute_hy_fwzic(expert_df=expert_df, mapping_df=mapping_df)
