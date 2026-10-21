import argparse,json
from pathlib import Path
from src.config import ROOT,PAPER_DATA,PAPER_OUTPUT
from src.core.codas import hyperbolic_codas
from src.core.hy_fwzic import compute_weights
from src.analysis.sensitivity import analyze
from src.common.outputs import export_codas
from src.common.reproducibility import write_manifest
from .load_paper_data import load_mode1
from .validate_paper import compare_results


def run_mode1(tau=.02,source='published'):
    data=load_mode1();criteria=data['criteria']
    computed=compute_weights(data['judgments'],data['linguistic_mapping'])
    if source not in ['published','computed']:raise ValueError('source must be published or computed.')
    weights=criteria.weight if source=='published' else computed.set_index('criterion').weight
    rank=lambda w:hyperbolic_codas(data['matrix'],w,criteria.direction,data['mapping'],tau)
    result=rank(weights)
    details,summary=analyze(rank,weights,data['scenarios'])
    comparison,sc,wc=compare_results(result,details,computed,criteria)
    return {**data,'weights':weights,'weight_comparison':wc,'result':result,'comparison':comparison,
            'sensitivity':details,'sensitivity_summary':summary,'sensitivity_comparison':sc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=PAPER_OUTPUT)
    parser.add_argument('--weights', choices=['published','computed'], default='published')
    parser.add_argument('--tau', type=float, default=.02)
    args = parser.parse_args()
    run = run_mode1(args.tau, args.weights)
    out = args.output.resolve()
    export_codas(run['result'], out, run['matrix'].columns)
    for key in ['weight_comparison','comparison','sensitivity','sensitivity_summary','sensitivity_comparison']:
        run[key].to_csv(out/f'{key}.csv', index=False)
    report = {'weight_source': args.weights, 'tau': args.tau, 'threshold_rule':'close (paper Eq.16)',
              'winner': run['result'].ranking.iloc[0].alternative,
              'base_scores_passed': int(run['comparison'].score_within_tolerance.sum()),
              'base_scores_total': 13,
              'sensitivity_scores_passed': int(run['sensitivity_comparison'].score_within_tolerance.sum()),
              'sensitivity_scores_total':104,
              'sensitivity_rank_matches': int(run['sensitivity_comparison'].rank_match.sum()),
              'max_weight_error': float(run['weight_comparison'].abs_error.max()),
              'note':'Reference H is printed to 3 decimals; absolute tolerance 0.0011. Exact printed ties/ranks are compared and discrepancies retained.'}
    (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    write_manifest(out, ROOT, 'Mode1', list(PAPER_DATA.rglob('*.csv')),
                   {'weights':args.weights,'tau':args.tau,'threshold_rule':'close','score_tolerance':.0011})
    print(run['result'].ranking.to_string(index=False))
    print(json.dumps(report,indent=2))
    return 0 if report['base_scores_passed']==13 and report['sensitivity_scores_passed']==104 and report['max_weight_error']<1e-8 else 1

if __name__=='__main__':raise SystemExit(main())
