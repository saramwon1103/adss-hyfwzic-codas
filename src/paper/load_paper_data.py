import pandas as pd
from src.config import PAPER_DATA


def load_mode1():
    d=PAPER_DATA
    return {'matrix':pd.read_csv(d/'adss_performance.csv',index_col='alternative'),
            'criteria':pd.read_csv(d/'published_weights.csv').set_index('criterion'),
            'mapping':pd.read_csv(d/'performance_mapping.csv'),
            'linguistic_mapping':pd.read_csv(d/'linguistic_mapping.csv'),
            'judgments':pd.read_csv(d/'expert_judgments.csv',index_col='expert'),
            'scenarios':pd.read_csv(d/'sensitivity_scenarios.csv',index_col='scenario')}
