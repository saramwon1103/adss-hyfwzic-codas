from pathlib import Path

import pandas as pd


# Root của project:
# D:/adss-hyfwzic-codas
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PAPER_DATA_DIR = PROJECT_ROOT / "data" / "paper"
EXPECTED_DIR = PAPER_DATA_DIR / "expected"


def load_linguistic_mapping() -> pd.DataFrame:
    """
    Load Table 2 linguistic -> HyFS mapping.
    """
    path = PAPER_DATA_DIR / "linguistic_mapping.csv"

    return pd.read_csv(path)


def load_expert_judgments() -> pd.DataFrame:
    """
    Load Table 4 expert decision matrix.
    """
    path = PAPER_DATA_DIR / "expert_judgments.csv"

    return pd.read_csv(path)


def load_expected_weights() -> pd.DataFrame:
    """
    Load Table 5 expected Hy-FWZIC results reported by the paper.
    """
    path = EXPECTED_DIR / "expected_weights.csv"

    return pd.read_csv(path)