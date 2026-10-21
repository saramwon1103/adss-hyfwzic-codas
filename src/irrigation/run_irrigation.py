"""Real-input Mode2 orchestration and CLI in the repository's src/ layout."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import ROOT,PAPER_DATA,IRRIGATION_DATA,IRRIGATION_OUTPUT
from src.core.codas import crisp_codas
from src.core.hy_fwzic import compute_weights
from src.analysis.sensitivity import analyze,weight_scenarios
from src.common.outputs import export_codas
from src.common.reproducibility import write_manifest
from .interfaces import WEATHER,RealStrategyInput,RealStrategyOutput
from .preprocess import prepare_real_run,DEFAULT_START,DEFAULT_END,SOURCE_DIR
from .strategies import REAL_STRATEGIES
from .scenario_generator import real_scenario,SCENARIOS
from .simulator import simulate_real
from .kpi_evaluator import real_kpis
from .build_performance_matrix import build_performance_matrix

def run_real_mode2(scenario='Normal',sector=1,start=DEFAULT_START,end=DEFAULT_END):
    base,model,cfg,quality,validation,daily=prepare_real_run(sector,start,end)
    frame=real_scenario(base,scenario)
    # Paired robustness retains natural historical sensor missingness in BOTH runs.
    baseline=real_scenario(base,'Normal')
    for col in WEATHER+[f'{c}_forecast' for c in WEATHER]+['water_budget_liters','demand_event']:
        baseline[col]=frame[col]
    stressed=baseline.copy();stressed.loc[10:18,'sensor_missing']=True
    records,traces=[],{}
    for name,factory in REAL_STRATEGIES.items():
        trace=simulate_real(frame,model,cfg,factory)
        reference=simulate_real(baseline,model,cfg,factory)
        outage=simulate_real(stressed,model,cfg,factory)
        traces[name]=trace
        records.append({'alternative':name,**real_kpis(trace,reference,outage,cfg)})
    matrix=build_performance_matrix(records)
    d=IRRIGATION_DATA
    criteria=pd.read_csv(d/'criteria.csv').set_index('criterion')
    prefs=pd.read_csv(d/'preferences.csv',index_col='expert')
    weights=compute_weights(prefs,pd.read_csv(PAPER_DATA/'linguistic_mapping.csv')).set_index('criterion').weight
    rank=lambda w:crisp_codas(matrix,w,criteria.direction)
    result=rank(weights);sens,summary=analyze(rank,weights,weight_scenarios(weights,high=.4))
    quality['scenario']=scenario
    quality['scenario_inputs_modified']=scenario!='Normal'
    quality['out_of_training_range_steps']={name:int(t.outside_training_range.sum()) for name,t in traces.items()}
    quality['weak_irrigation_response']=bool(model.irrigation_gain<.01)
    quality['water_effect_at_daily_budget_points']=float(model.irrigation_gain*model.training_report['water_budget_liters'])
    quality['holdout_error_exceeds_daily_water_effect']=bool(quality['holdout_mae_points']>quality['water_effect_at_daily_budget_points'])
    quality['non_discriminating_kpis']=matrix.columns[(matrix.max()-matrix.min())<1e-10].tolist()
    quality['initial_target_gap_points']=float(cfg['target_index']-cfg['initial_index'])
    return {'dataset':frame,'matrix':matrix,'criteria':criteria,'weights':weights,'result':result,
            'traces':traces,'sensitivity':sens,'sensitivity_summary':summary,'scenario':scenario,
            'is_synthetic':False,'source_kind':'uploaded_real_data','outcomes_modeled':True,
            'model':model,'config':cfg,'quality':quality,'validation':validation,'daily_source':daily}

def export_run(run,out):
    export_codas(run['result'],out,run['matrix'].columns)
    run['dataset'].to_csv(out/'scenario_dataset.csv',index=False)
    run['matrix'].to_csv(out/'performance_matrix.csv')
    run['weights'].to_csv(out/'weights.csv')
    run['sensitivity'].to_csv(out/'sensitivity.csv',index=False)
    run['sensitivity_summary'].to_csv(out/'stability.csv',index=False)
    for name,trace in run['traces'].items():
        filename=name.lower().replace(' ','_').replace('(','').replace(')','')+'_trace.csv'
        trace.to_csv(out/filename,index=False)
    if not run['is_synthetic']:
        run['validation'].to_csv(out/'historical_holdout_validation.csv',index=False)
        (out/'soil_response_model.json').write_text(json.dumps(run['model'].to_dict(),indent=2),encoding='utf-8')
        (out/'data_quality.json').write_text(json.dumps(run['quality'],indent=2),encoding='utf-8')
        (out/'simulation_config.json').write_text(json.dumps(run['config'],indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=[*SCENARIOS,'all'],default='all')
    parser.add_argument('--sector',type=int,choices=[1,2],default=1)
    parser.add_argument('--start',default=DEFAULT_START)
    parser.add_argument('--end',default=DEFAULT_END)
    parser.add_argument('--output',type=Path,default=IRRIGATION_OUTPUT)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    summary=[]
    for scenario in SCENARIOS if args.scenario=='all' else [args.scenario]:
        run=run_real_mode2(scenario,args.sector,args.start,args.end)
        out=output/scenario.lower().replace(' ','_');export_run(run,out)
        run['daily_source'].to_csv(output/'processed_irrigation.csv',index=False)
        run['dataset'].to_csv(output/'evaluation_input.csv',index=False)
        q=run['quality']
        write_manifest(out,ROOT,'Mode2',list(IRRIGATION_DATA.glob('*.csv'))+
                       [SOURCE_DIR/f'sector_{args.sector}_10min.csv',SOURCE_DIR/'provenance.json'],
                       {'source':'real','scenario':scenario,'sector':args.sector,'start':args.start,'end':args.end,
                        'input_is_synthetic':False,'outcomes_modeled':True,'config':run['config'],
                        'tau':.02,'threshold_rule':'close','normalization':'min-max'})
        winners=run['result'].ranking.loc[run['result'].ranking['rank']==1,'alternative'].tolist()
        summary.append({'scenario':scenario,'winners':winners,'source':'real','input_is_synthetic':False,'outcomes_modeled':True})
        print(f'\n{scenario} | source=real | input_is_synthetic=False | outcomes_modeled=True')
        print(f'Sector {args.sector}: {args.start}..{args.end}, {len(run["dataset"])} days | L / probe-index points')
        print(f'Holdout MAE={q["holdout_mae_points"]:.4f}; persistence MAE={q["persistence_mae_points"]:.4f}')
        print(run['result'].ranking[['alternative','H','rank']].to_string(index=False))
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

if __name__=='__main__':main()
