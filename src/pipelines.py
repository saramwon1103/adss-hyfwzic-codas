"""Small public entry points; domain logic lives in src/paper and src/irrigation."""
from .config import ROOT
from .paper.load_paper_data import load_mode1
from .paper.reproduce_paper import run_mode1
from .common.outputs import export_codas


def run_mode2(scenario='Normal',seed=2026,data=None,config=None,*,source='real',sector=1,start='2025-06-01',end='2025-07-13'):
    if source!='real' or data is not None or config is not None:
        raise ValueError('This repository uses real L/index data. Legacy mm/VWC or mock inputs are not accepted.')
    from .irrigation.run_irrigation import run_real_mode2
    return run_real_mode2(scenario,sector,start,end)
