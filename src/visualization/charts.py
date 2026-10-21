"""Generate scientific output figures from computed results (no imagegen)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.pipelines import ROOT, run_mode1, run_mode2

def generate_figures():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'#f6f8fb','axes.facecolor':'white',
                         'axes.titleweight':'bold','axes.labelcolor':'#334155','text.color':'#0f172a'})
    out=ROOT/'outputs/figures';out.mkdir(parents=True,exist_ok=True)
    run=run_mode1();r=run['result'].ranking.sort_values('H')
    fig,ax=plt.subplots(figsize=(10,6.6),layout='constrained')
    colors=['#059669' if x=='A8' else '#94a3b8' if x!='A11' else '#e76f51' for x in r.alternative]
    ax.barh(r.alternative,r.H,color=colors,height=.7)
    for i,v in enumerate(r.H):ax.text(v+(.06 if v>=0 else -.06),i,f'{v:.3f}',va='center',ha='left' if v>=0 else 'right',fontsize=9)
    ax.axvline(0,color='#cbd5e1',lw=1);ax.set_xlim(-4.4,4.7)
    ax.set_xlabel('CODAS H score (higher is better)');ax.set_title('Mode 1 | Reproduction of the 13-ADSS ranking',loc='left',pad=18)
    fig.text(.5,.012,'Table 8 reference: 13/13 H values within 0.0011 | paper close rule, tau=0.02',ha='center',fontsize=9,color='#475569')
    fig.get_layout_engine().set(rect=(0,.045,1,1))
    fig.savefig(out/'mode1_ranking.png',dpi=170,bbox_inches='tight');plt.close(fig)

    s=run['sensitivity'].pivot(index='alternative',columns='scenario',values='rank').reindex(run['result'].ranking.alternative)
    fig,ax=plt.subplots(figsize=(9,7),layout='constrained')
    im=ax.imshow(s,vmin=1,vmax=13,cmap='YlGnBu_r',aspect='auto')
    ax.set_xticks(range(8),s.columns);ax.set_yticks(range(13),s.index)
    for i in range(13):
     for j in range(8):ax.text(j,i,int(s.iloc[i,j]),ha='center',va='center',color='white' if s.iloc[i,j]<5 else '#0f172a',fontsize=9)
    ax.set_title('Mode 1 | Sensitivity ranks (Table 10)',loc='left',pad=15)
    ax.set_xlabel('Weight scenario | emphasized criterion gets 0.3; others 0.1')
    fig.colorbar(im,ax=ax,label='Rank (1 is best)',shrink=.7)
    fig.savefig(out/'mode1_sensitivity.png',dpi=170,bbox_inches='tight');plt.close(fig)


    colors={'SIDSS-like (stub)':'#059669','FDSS-like':'#2563eb','MRGCD-like':'#e76f51'}
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,layout='constrained')
    for ax,scenario in zip(axes,['Normal','Drought','Missing Sensor']):
     rr=run_mode2(scenario)
     for name,t in rr['traces'].items():
      ax.plot(t.timestamp,t.moisture_end_index,label=name,color=colors[name],lw=1.8)
     ax.axhspan(40,60,color='#86efac',alpha=.13,label='Source operating band 40..60')
     ax.axhline(50,c='#64748b',ls='--',lw=1)
     if scenario=='Missing Sensor':ax.axvspan(rr['dataset'].timestamp.iloc[10],rr['dataset'].timestamp.iloc[19],color='#fbbf24',alpha=.2,label='Added sensor outage')
     if scenario=='Drought':ax.axvline(rr['dataset'].timestamp.iloc[10],c='#9a3412',ls=':',lw=1.5,label='What-if drought starts')
     ax.set_title(scenario,loc='left');ax.set_ylabel('Relative probe-index points');ax.set_ylim(20,65);ax.grid(alpha=.15)
     ax.legend(loc='lower left',ncol=3,fontsize=8)
    axes[-1].set_xlabel('Local evaluation date (2025)')
    fig.suptitle('Mode 2 | Uploaded field inputs, modelled policy outcomes\nSector 1 | 2025-06-01 to 2025-07-13 | 43 days | SIDSS remains a stub',fontsize=13)
    fig.savefig(out/'mode2_scenario_traces.png',dpi=170,bbox_inches='tight');plt.close(fig)

    rr=run_mode2('Normal');v=rr['validation'];q=rr['quality']
    fig,ax=plt.subplots(figsize=(12,5.5),layout='constrained')
    ax.plot(v.timestamp,v.observed_next_index,label='Observed next-boundary sensor index',color='#0f172a',lw=1.8)
    ax.plot(v.timestamp,v.predicted_next_index,label=f'Surrogate one-step prediction (MAE {q["holdout_mae_points"]:.2f})',color='#2563eb')
    ax.plot(v.timestamp,v.persistence_next_index,label=f'Persistence baseline (MAE {q["persistence_mae_points"]:.2f})',color='#94a3b8',ls='--')
    ax.set_ylabel('Relative probe-index points');ax.set_title('Historical-action holdout validation | 39 valid daily pairs\nReplay of recorded irrigation for prediction validation; no counterfactual field measurements',loc='left',fontsize=12,pad=14)
    ax.legend(ncol=1,fontsize=9);ax.grid(alpha=.15)
    fig.savefig(out/'mode2_holdout_validation.png',dpi=170,bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    rr=run_mode2('Drought');m=rr['matrix'];names=['SIDSS stub','FDSS-like','MRGCD-like']
    for ax,col,title in zip(axes,['water_use_liters','soil_moisture_mae_points'],['Proposed water use (L)','Modelled soil error (index points)']):
     vals=m[col];ax.bar(names,vals,color=list(colors.values()),width=.65)
     for i,v in enumerate(vals):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom',fontsize=9)
     ax.set_title(title,loc='left',fontsize=11);ax.margins(y=.2)
    fig.suptitle('Mode 2 | What-if drought on uploaded field data\nModel-based trade-off, not a measured comparison of the original DSS products',fontsize=12)
    fig.savefig(out/'mode2_drought_tradeoff.png',dpi=170,bbox_inches='tight');plt.close(fig)
    print('Saved five figures to',out)
