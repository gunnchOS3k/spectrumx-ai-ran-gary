"""Finite optimizer component-domain check using existing seed/N inputs, no new outcomes."""
from pathlib import Path
import importlib.util,json,sys,inspect
from scipy.optimize._slsqp_py import _minimize_slsqp
E=Path(__file__).resolve().parents[1];G=E/'generated';G.mkdir(exist_ok=True);rows=[]
for v in ['original','revised']:
 name='bound_'+v;s=importlib.util.spec_from_file_location(name,E/'source'/v/'digital_programme.py');m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m)
 for seed in list(range(5))+list(range(100,105)):
  for n in range(4,9):
   ss=m.Slot(20.,1.,2.,20.,80.,n,[1]*n,False,['terrestrial','local_edge_wifi','degraded_local','ntn_fallback'],.5,1.,.6,0.,0.,'degraded_ok')
   x,note=m._optimize_shares(ss if v=='original' else m.action_space_from_slot(ss),seed)
   assert note=='slsqp'
   rows.append({'vintage':v,'seed':seed,'n_users':n,'shares':x,'backend':note,'budget_residual':20-sum(x),'max_share_error':max(abs(a-20/n) for a in x),'strict_budget_feasible':sum(x)<=20})
res=max(abs(a['budget_residual']) for a in rows);du=res/20
# Lipschitz bound for fixed actions: |d gross/du| <= .4*L*alpha*beta*max_fidelity_latency/500+.2*q*alpha/100.
# All source families L<=140,q<=18,alpha<=10,beta<=1.25,fidelity latency multiplier<=1.05.
K=.4*140*10*1.25*1.05/500+.2*18*10/100
bound=K*du
assert bound<1e-6
result={'inputs':'Existing seeds0..4 and100..104, n_users4..8,B20; all frozen optimizer input combinations, two sources separately. No new seed or policy outcome.', 'component_solves':len(rows),'max_absolute_budget_residual':res,'strict_budget_overshoots':sum(not a['strict_budget_feasible'] for a in rows),'max_share_error':max(a['max_share_error'] for a in rows),'max_utilization_error':du,'fixed_action_primary_score_lipschitz_constant':K,'worst_absolute_slot_score_impact_bound':bound,'bound_conditions':'Fixed network/placement/fidelity/checkpoint decisions (shares do not select them), source priors and score; no physical interpretation. Does not remove discrete-policy or observation confounding.','paired_negative_margin_minimum':.010294622848085685,'material_to_preserved_negative_direction':False,'scipy_slsqp_local_signature':str(inspect.signature(_minimize_slsqp)),'exact_mathematical_optimum_uniform':True,'numerical_solutions_exactly_optimal_or_strictly_feasible':False,'rows':rows,'human_approval':False}
(G/'OPTIMIZER_BOUNDS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
