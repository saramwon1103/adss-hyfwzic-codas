"""Local Streamlit UI; no hosted service or external credentials required."""
import pandas as pd
import streamlit as st
from src.visualization.tables import winner_names
from src.pipelines import run_mode1, run_mode2
from src.irrigation.scenario_generator import SCENARIOS

st.set_page_config(page_title='DSS | Thắng', page_icon='🌱', layout='wide')
st.title('Agriculture Decision Support Lab')
st.caption('DSS coursework · Thắng + shared integration · v2.0 · GitHub src layout')
mode = st.sidebar.radio('Mode', ['Mode 1 · Paper reproduction', 'Mode 2 · Irrigation simulation'])
st.sidebar.markdown('**Scope**: CODAS, scenarios, strategies, KPI, sensitivity and shared integration.')

if mode.startswith('Mode 1'):
    st.header('13 ADSS · Hy-FWZIC → Hyperbolic CODAS')
    source = st.sidebar.selectbox('Weight source', ['published','computed'])
    tau = st.sidebar.slider('τ (paper Eq.16, close rule)', min_value=0., max_value=.05, value=.02, step=.001)
    run = run_mode1(tau, source)
    a,b,c=st.columns(3)
    a.metric('Top alternative', run['result'].ranking.iloc[0].alternative)
    b.metric('Table 8 score matches', f"{run['comparison'].score_within_tolerance.sum()}/13")
    c.metric('Table 10 score matches', f"{run['sensitivity_comparison'].score_within_tolerance.sum()}/104")
    st.info('Paper reference: DOI 10.1016/j.compag.2024.109618. Tolerance 0.0011 for printed H; ties use competition ranking. τ=0.02 is a documented reproduction assumption.')
    t1,t2,t3,t4=st.tabs(['Data & weights','Ranking','Sensitivity & stability','Validation & audit'])
    with t1:
        st.dataframe(run['matrix'], width='stretch')
        st.bar_chart(run['weights'])
        st.dataframe(run['weight_comparison'], width='stretch')
    with t2:
        st.dataframe(run['result'].ranking, hide_index=True, width='stretch')
        st.bar_chart(run['result'].ranking.set_index('alternative').H)
    with t3:
        choice=st.selectbox('Weight scenario',run['sensitivity_summary'].scenario)
        st.dataframe(run['sensitivity_summary'],hide_index=True,width='stretch')
        st.dataframe(run['sensitivity'].query('scenario == @choice'),hide_index=True,width='stretch')
    with t4:
        st.dataframe(run['comparison'],hide_index=True,width='stretch')
        st.dataframe(run['sensitivity_comparison'],hide_index=True,width='stretch')
        st.dataframe(run['result'].relative,width='stretch')
        st.download_button('Download validation CSV',run['comparison'].to_csv(index=False),'paper_validation.csv','text/csv')
else:
    from datetime import date
    st.header('Uploaded Arnesano data · open-field tomato')
    scenario=st.sidebar.selectbox('Operational scenario',SCENARIOS)
    sector=st.sidebar.selectbox('Tomato sector',[1,2])
    period=st.sidebar.date_input('Evaluation period',(date(2025,6,1),date(2025,7,13)),
                                min_value=date(2025,4,1),max_value=date(2025,9,24))
    if len(period)!=2:
        st.info('Select both evaluation start and end.');st.stop()
    try:
        run=run_mode2(scenario,sector=sector,start=period[0].isoformat(),end=period[1].isoformat())
    except (ValueError,FileNotFoundError) as error:
        st.error(str(error));st.stop()
    q=run['quality']
    st.info('REAL INPUT DATA: uploaded sensor/valve series plus historical reanalysis weather. Volumes are nominal-flow litres; soil readings are relative probe-index points, not calibrated VWC. No mock fallback.')
    st.warning('MODELLED OUTCOMES: actions of these three heuristics were not observed in the field. SIDSS is still a stub; expert preferences remain illustrative. Drought and sensor-outage cases are what-if transformations.')
    if not q['model_beats_persistence_mae']:
        st.warning('Validation: the fitted response model does not beat simple persistence on this holdout. Treat rankings as exploratory.')
    if q['weak_irrigation_response']:
        st.caption('The learned irrigation response is weak. Small score differences and ties do not establish a reliable field winner.')
    if max(q['out_of_training_range_steps'].values())>0:
        st.warning('Extrapolation: some simulated inputs are outside the training feature ranges; see Source & model validation for counts. Rankings are exploratory.')
    winners=winner_names(run['result'])
    a,b,c=st.columns(3)
    a.metric('Top alternative(s)',', '.join(winners))
    b.metric('Daily steps',len(run['dataset']))
    c.metric('Sector',f'ARNESANO_TOMATO_{sector}')
    t1,t2,t3,t4,t5=st.tabs(['Data & weights','KPI & ranking','Scenario traces','Sensitivity & stability','Source & model validation'])
    with t1:
        st.caption(f'Evaluation: {q["evaluation_start"]} to {q["evaluation_end"]}. Forecasts are previous-day persistence. Model fitted before evaluation.')
        st.dataframe(run['dataset'],hide_index=True,width='stretch')
        st.dataframe(run['criteria'],width='stretch')
        st.caption('Weights use the previous illustrative expert judgments; this dataset supplies no expert preferences.')
        st.bar_chart(run['weights'])
        st.download_button('Download scenario input CSV',run['dataset'].to_csv(index=False),'scenario_input_real.csv','text/csv')
    with t2:
        st.dataframe(run['matrix'],width='stretch')
        st.dataframe(run['result'].ranking,hide_index=True,width='stretch')
        st.bar_chart(run['result'].ranking.set_index('alternative').H)
        constants=run['matrix'].columns[run['matrix'].max()-run['matrix'].min()<1e-10].tolist()
        if constants:st.caption('Non-discriminating KPI columns in this run: '+', '.join(constants))
        st.caption('Water = litres; soil error = probe-index points. Excess water and response are model-based proxies. Ties are retained; robustness=1 can mean no additional measured degradation in this scenario.')
        st.download_button('Download performance matrix',run['matrix'].to_csv(),'performance_matrix_real.csv','text/csv')
    with t3:
        traces=pd.concat(run['traces'].values())
        st.caption('Closed-loop surrogate trajectories, not measured outcomes of these alternative policies.')
        st.line_chart(traces.pivot(index='timestamp',columns='alternative',values='moisture_end_index'))
        st.bar_chart(traces.pivot(index='timestamp',columns='alternative',values='irrigation_liters'))
        name=st.selectbox('Inspect strategy trace',list(run['traces']))
        st.dataframe(run['traces'][name],hide_index=True,width='stretch')
    with t4:
        st.dataframe(run['sensitivity_summary'],hide_index=True,width='stretch')
        st.dataframe(run['sensitivity'],hide_index=True,width='stretch')
    with t5:
        a,b,c=st.columns(3)
        a.metric('Holdout MAE (index points)',f'{q["holdout_mae_points"]:.3f}')
        b.metric('Persistence MAE (index points)',f'{q["persistence_mae_points"]:.3f}')
        c.metric('Valid holdout pairs',q['holdout_pairs'])
        st.caption('One-step replay uses historical water actions to validate predictions. Future holdout targets do not enter training or policy inputs.')
        v=run['validation'].set_index('timestamp')
        st.line_chart(v[['observed_next_index','predicted_next_index','persistence_next_index']])
        st.dataframe(run['validation'],hide_index=True,width='stretch')
        st.json(q)
        st.json(run['model'].to_dict())
