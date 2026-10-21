from pathlib import Path
import pandas as pd

def export_codas(result, output, criterion_labels):
    output.mkdir(parents=True, exist_ok=True)
    result.ranking.to_csv(output/'ranking.csv', index=False)
    result.relative.to_csv(output/'relative_assessment.csv')
    labels = list(criterion_labels)
    for stage in ['normalized', 'weighted']:
        values = getattr(result, stage)
        columns = ([f'{c}_{p}' for c in labels for p in ['mu','nu']]
                   if values.ndim == 3 else labels)
        pd.DataFrame(values.reshape(len(values), -1), index=result.relative.index,
                     columns=columns).to_csv(output/f'{stage}_matrix.csv')
    nis = result.nis
    if nis.ndim == 2:
        pd.DataFrame(nis, columns=['mu','nu'], index=labels).to_csv(output/'nis.csv')
    else:
        pd.Series(nis, index=labels, name='nis').to_csv(output/'nis.csv', index_label='criterion')
