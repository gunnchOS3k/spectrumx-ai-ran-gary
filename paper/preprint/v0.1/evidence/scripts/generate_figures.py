from pathlib import Path
import json,csv,numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
E=Path(__file__).resolve().parents[1];OUT=E/'generated';FIG=OUT/'figures';TAB=OUT/'tables'
for d in [OUT,FIG,TAB]:d.mkdir(parents=True,exist_ok=True)
def get(name,file):return json.loads((E/'data'/name/('rq2_cross_layer_continuity_'+file+'.json')).read_text())
def ci(v):
 x=np.array(v);m=float(x.mean());sd=float(x.std(ddof=1));h=float(t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x)));return dict(mean=m,ci_low=m-h,ci_high=m+h,sd=sd,n=len(x))
def esc(s):return s.replace('_',r'\_')
def tab(name,header,rows,align):
 (TAB/name).write_text('\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(header)+r' \\'+'\n\\midrule\n'+'\n'.join(' & '.join(str(c) for c in r)+r' \\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n')
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
H=get('original_reproduced','heldout'); D=get('original_reproduced','domain_shift'); A=get('original_reproduced','ablation');R=get('revised_reproduced','heldout')
metrics=['service_continuity_utility','predicted_latency_ms','energy_use_j','fairness','n_network_switches','n_placement_switches','switch_cost','compute_time_ms']
rows=[]
for name,data in H['policies'].items():
 c=ci(data['seed_means']['service_continuity_utility']);rows.append([esc(name),f"{c['mean']:.4f}",f"[{c['ci_low']:.4f}, {c['ci_high']:.4f}]",f"{data['predicted_latency_ms']['mean']:.2f}",f"{data['switch_cost']['mean']:.4f}"])
tab('heldout.tex',['Policy','$\\bar U$','95\\% seed CI','$\\widehat L$ (ms)','$C$ / episode'],rows,'lrrrr')
flat=[]
for vintage in ['original_reproduced','revised_reproduced']:
 for family in ['train','heldout']:
  doc=get(vintage,family)
  for policy,dat in doc['policies'].items():
   for metric,summ in dat.items():
    if metric=='seed_means':continue
    flat.append({'vintage':vintage,'family':family,'policy':policy,'metric':metric,**summ})
with (TAB/'summary_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
comparisons=[]
for left,right in [('twin_informed','no_adaptation'),('information_equivalent','no_adaptation'),('information_equivalent','twin_informed'),('information_equivalent','oracle')]:
 x=np.array(H['policies'][left]['seed_means']['service_continuity_utility'])-np.array(H['policies'][right]['seed_means']['service_continuity_utility']);comparisons.append({'left':left,'right':right,**ci(x),'seed_differences':x.tolist()})
x=np.array(A['no_switch_cost']['policies']['twin_informed']['seed_means']['service_continuity_utility'])-np.array(H['policies']['twin_informed']['seed_means']['service_continuity_utility']);comparisons.append({'left':'twin_no_switch_cost','right':'twin_full',**ci(x),'seed_differences':x.tolist()})
(OUT/'paired_statistics.json').write_text(json.dumps(comparisons,indent=2)+'\n')
tab('paired.tex',['Paired comparison','$\\Delta U$','95\\% seed CI'],[[esc(c['left'])+' -- '+esc(c['right']),f"{c['mean']:.5f}",f"[{c['ci_low']:.5f}, {c['ci_high']:.5f}]"] for c in comparisons],'lrr')
rows=[]
for name,d in D.items():
 for pol in ['no_adaptation','static','edge_only','twin_informed','information_equivalent','oracle']:
  v=ci(d['policies'][pol]['seed_means']['service_continuity_utility']);rows.append([esc(name),esc(pol),f"{v['mean']:.4f}",f"[{v['ci_low']:.4f}, {v['ci_high']:.4f}]"])
tab('domain_shift.tex',['Family','Policy','$\\bar U$','95\\% seed CI'],rows,'llrr')
rows=[]
for name,doc in A.items():
 d=doc['policies']['twin_informed'];v=ci(d['seed_means']['service_continuity_utility']);rows.append([esc(name),f"{v['mean']:.4f}",f"[{v['ci_low']:.4f}, {v['ci_high']:.4f}]",f"{d['switch_cost']['mean']:.4f}"])
tab('ablations.tex',['Intervention','$\\bar U$','95\\% seed CI','$C$ / episode'],rows,'lrrr')
rows=[]
for pol in R['policies']:
 v=ci(R['policies'][pol]['seed_means']['service_continuity_utility']);rows.append([esc(pol),f"{v['mean']:.4f}",f"[{v['ci_low']:.4f}, {v['ci_high']:.4f}]"])
tab('revised.tex',['Policy','$\\bar U$','95\\% seed CI'],rows,'lrr')
# Figure 1: held-out seed means, paired design shown directly.
order=['local_only','cloud_only','twin_informed','rule_based','no_adaptation','optimization_based','edge_only','static','oracle','information_equivalent'];fig,ax=plt.subplots(figsize=(7.1,3.2));ys=np.arange(len(order))
for y,pol in zip(ys,order):
 v=H['policies'][pol]['seed_means']['service_continuity_utility'];c=ci(v);ax.errorbar(c['mean'],y,xerr=c['ci_high']-c['mean'],fmt='o',color='#193e59',capsize=3);ax.scatter(v,np.repeat(y,len(v)),marker='|',color='#b16a31',s=40,zorder=4)
ax.set_yticks(ys,order);ax.set_xlabel('Net continuity utility (dimensionless)');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(FIG/'heldout.pdf');fig.savefig(FIG/'heldout.png',dpi=200);plt.close(fig)
# Figure 2: gross continuity and switching contribution from exact episode sums.
order=['no_adaptation','edge_only','static','twin_informed','information_equivalent','oracle'];fig,axes=plt.subplots(1,2,figsize=(7.1,2.6))
for i,pol in enumerate(order):
 d=H['policies'][pol];net=d['service_continuity_utility']['mean'];cost=d['switch_cost']['mean']/16;axes[0].bar(i,net,color='#193e59');axes[0].bar(i,cost,bottom=net,color='#b16a31')
for ax in axes:ax.set_xticks(np.arange(len(order)),[s.replace('_','\n') for s in order],fontsize=6);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylim(.75,.85);axes[0].set_ylabel('Gross = net + per-slot cost');axes[0].plot([],[],color='#193e59',lw=6,label='Net utility');axes[0].plot([],[],color='#b16a31',lw=6,label='Mean slot cost');axes[0].legend(fontsize=6)
for i,pol in enumerate(order):
 d=H['policies'][pol];axes[1].bar(i-.16,d['n_network_switches']['mean'],width=.32,color='#193e59');axes[1].bar(i+.16,d['n_placement_switches']['mean'],width=.32,color='#b16a31')
axes[1].set_ylabel('Switches per episode');axes[1].plot([],[],color='#193e59',lw=6,label='Network');axes[1].plot([],[],color='#b16a31',lw=6,label='Placement');axes[1].legend(fontsize=6);fig.tight_layout();fig.savefig(FIG/'switch_cost.pdf');fig.savefig(FIG/'switch_cost.png',dpi=200);plt.close(fig)
# Figure 3: distribution shift with original paired seed means.
fig,ax=plt.subplots(figsize=(7.1,2.7));families=['held_out','high_blockage','energy_constrained','outage_heavy'];colors=['#193e59','#b16a31','#3d7961']
for i,(pol,col) in enumerate(zip(['no_adaptation','twin_informed','information_equivalent'],colors)):
 vals=[];errs=[]
 for fam in families:
  d=H if fam=='held_out' else D[fam];c=ci(d['policies'][pol]['seed_means']['service_continuity_utility']);vals.append(c['mean']);errs.append(c['ci_high']-c['mean'])
 ax.errorbar(np.arange(4)+(i-1)*.1,vals,yerr=errs,fmt='o-',color=col,capsize=3,label=pol)
ax.set_xticks(range(4),['Held out','High blockage','Energy constrained','Outage heavy']);ax.set_ylabel('Net continuity utility');ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(FIG/'domain_shift.pdf');fig.savefig(FIG/'domain_shift.png',dpi=200);plt.close(fig)
# Figure 4: algebraic post-hoc cost rescaling on fixed original action traces.
fig,ax=plt.subplots(figsize=(3.5,2.6));lam=np.linspace(0,2,101)
for pol,col in zip(['no_adaptation','twin_informed','information_equivalent'],colors):
 d=H['policies'][pol];u=d['service_continuity_utility']['mean'];c=d['switch_cost']['mean']/16;ax.plot(lam,u+(1-lam)*c,color=col,label=pol)
dt=H['policies']['twin_informed'];db=H['policies']['no_adaptation'];ct=dt['switch_cost']['mean']/16;cb=db['switch_cost']['mean']/16;threshold=(dt['service_continuity_utility']['mean']+ct-db['service_continuity_utility']['mean']-cb)/(ct-cb)
ax.axvline(threshold,color='gray',ls=':',lw=1);ax.set_xlabel('Switch-cost multiplier (fixed actions)');ax.set_ylabel('Net continuity utility');ax.legend(fontsize=6);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(FIG/'cost_rescaling.pdf');fig.savefig(FIG/'cost_rescaling.png',dpi=200);plt.close(fig)
