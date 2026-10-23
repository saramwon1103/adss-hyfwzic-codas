"""Load immutable paper-reproduction inputs from ``data/paper``."""
from pathlib import Path

import pandas as pd

from src.config import PAPER_DATA, PAPER_EXPECTED

# Kept for backwards compatibility with earlier imports.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_DATA_DIR = PAPER_DATA
EXPECTED_DIR = PAPER_EXPECTED


def load_linguistic_mapping() -> pd.DataFrame:
    """Load the linguistic -> HyFS mapping used by Hy-FWZIC."""
    return pd.read_csv(PAPER_DATA / "linguistic_mapping.csv")


def load_expert_judgments() -> pd.DataFrame:
    """Load the paper expert decision matrix (Table 4 reconstruction)."""
    return pd.read_csv(PAPER_DATA / "expert_judgments.csv")


def load_expected_weights() -> pd.DataFrame:
    """Load expected Table 5 weights/ranks used only for validation."""
    return pd.read_csv(PAPER_EXPECTED / "expected_weights.csv")


def load_mode1() -> dict:
    """Load every input needed by the integrated Mode 1 pipeline.

    Returns a dictionary so callers/tests can inspect each immutable input
    independently. Published weights remain separate from computed weights;
    expected outputs are not used as algorithm inputs.
    """
    criteria = pd.read_csv(PAPER_DATA / "published_weights.csv")
    if criteria["criterion"].duplicated().any():
        raise ValueError("published weight criterion labels must be unique")
    criteria = criteria.set_index("criterion")

    matrix = pd.read_csv(PAPER_DATA / "adss_performance.csv", index_col="alternative")
    mapping = pd.read_csv(PAPER_DATA / "performance_mapping.csv")
    linguistic_mapping = load_linguistic_mapping()
    expert_judgments = load_expert_judgments()
    sensitivity_scenarios = pd.read_csv(
        PAPER_DATA / "sensitivity_scenarios.csv", index_col="scenario"
    )

    expected_weights = load_expected_weights()
    expected_ranking = pd.read_csv(PAPER_EXPECTED / "expected_ranking.csv")
    expected_distances = pd.read_csv(PAPER_EXPECTED / "expected_distances.csv")
    expected_sensitivity = pd.read_csv(PAPER_EXPECTED / "expected_sensitivity.csv")

    return {
        "criteria": criteria,
        "matrix": matrix,
        "mapping": mapping,
        "linguistic_mapping": linguistic_mapping,
        "expert_judgments": expert_judgments,
        "sensitivity_scenarios": sensitivity_scenarios,
        "expected_weights": expected_weights,
        "expected_ranking": expected_ranking,
        "expected_distances": expected_distances,
        "expected_sensitivity": expected_sensitivity,
    }
