"""Repository paths; resolved from source location, independent of working directory."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PAPER_DATA = ROOT / 'data/paper'
PAPER_EXPECTED = PAPER_DATA / 'expected'
IRRIGATION_DATA = ROOT / 'data/irrigation'
IRRIGATION_RAW = IRRIGATION_DATA / 'raw'
IRRIGATION_PROCESSED = IRRIGATION_DATA / 'processed'
PAPER_OUTPUT = ROOT / 'outputs/paper'
IRRIGATION_OUTPUT = ROOT / 'outputs/irrigation'
