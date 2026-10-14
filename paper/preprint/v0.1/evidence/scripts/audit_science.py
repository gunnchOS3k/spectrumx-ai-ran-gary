"""Independent arithmetic/model assertions; no campaign, new seed or frozen-file mutation."""
from pathlib import Path
from dataclasses import asdict,replace
import json,importlib.util,sys,statistics,math,hashlib,copy
from collections import Counter
import numpy as np
from scipy.stats import t
E=Path(__file__).resolve().parents[1];G=E/'generated';G.mkdir(exist_ok=True)
SHAS={'original':'86a5a6f3000fce7593ee1bdfe02bae6bc37d2498','revised':'b44357f764d440f983132bd44a9c23c180591f70'}
def module(v):
 f=E/'source'/v/'digital_programme.py';name='audit_frozen_'+v;s=importlib.util.spec_from_file_location(name,f);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
# Exclusion is exact-key timing only; dictionary key sets, list lengths and all other values remain compared.
def scientific(o):
 if isinstance(o,dict):return {k:scientific(v) for k,v in o.items() if k!='compute_time_ms'}
 if isinstance(o,list):return [scientific(v) for v in o]
 return o
def stats(vals):
 assert len(vals)==5;m=statistics.fmean(vals);sd=statistics.stdev(vals);h=float(t.ppf(.975,4))*sd/math.sqrt(5)
 return {'n':5,'mean':m,'sample_sd':sd,'ci95':[m-h,m+h]}
def paired(a,b):
 assert len(a)==len(b)==5;d=[x-y for x,y in zip(a,b)];r=stats(d);r.update(seed_differences=d,paired_dz=r['mean']/r['sample_sd'] if r['sample_sd'] else None);return r
def get(v,s):return json.loads((E/'data'/v/('rq2_cross_layer_continuity_'+s+'.json')).read_text())
allstats={};parity={};summaries=0;contracts={};selection={};solves={};interventions={}
for v in SHAS:
 m=module(v);proto=m.load_protocol(E/'source'/v/'experiment_protocol.yaml');assert proto['split']['train_seeds']==list(range(5)) and proto['split']['held_out_seeds']==list(range(100,105))
 parity[v]={}
 for suffix in ['train','heldout','domain_shift','ablation','summary']:
  a=get(v,suffix);b=get(v+'_reproduced',suffix)
  assert scientific(a)==scientific(b),(v,suffix,'historical non-timing parity')
  parity[v][suffix]={'all_nontiming_fields_equal':True,'archived_sha256':hashlib.sha256((E/'data'/v/f'rq2_cross_layer_continuity_{suffix}.json').read_bytes()).hexdigest()}
  docs=a if suffix in ['domain_shift','ablation'] else {suffix:a} if suffix!='summary' else {}
  for label,d in docs.items():
   assert d['seeds'] in [list(range(5)),list(range(100,105))] and d['n_episodes_per_seed']==24 and d['n_slots_per_episode']==16
   for pol,p in d['policies'].items():
    for key,vals in p['seed_means'].items():
     if key=='compute_time_ms':continue
     s=stats(vals);ref=p[key];assert ref['n']==5 and abs(ref['mean']-s['mean'])<1e-12 and abs(ref['sd']-s['sample_sd'])<1e-12
     critical=2.776 if v=='original' else float(t.ppf(.975,4));h=critical*s['sample_sd']/5**.5
     assert abs(ref['ci_low']-(s['mean']-h))<1e-10 and abs(ref['ci_high']-(s['mean']+h))<1e-10
     summaries+=1
   if d.get('effect_sizes') and 'no_adaptation' in d['policies']:
    base=d['policies']['no_adaptation']['seed_means']['service_continuity_utility']
    for pol,obj in d['effect_sizes'].items():
     a1=d['policies'][pol]['seed_means']['service_continuity_utility'];sdpooled=((statistics.variance(a1)+statistics.variance(base))/2)**.5;val=(statistics.fmean(a1)-statistics.fmean(base))/sdpooled if sdpooled else 0.
     assert abs(obj['cohens_d_vs_no_adaptation_continuity']-val)<1e-10
 h=get(v,'heldout');A=get(v,'ablation');pol=h['policies'];arr=lambda p:pol[p]['seed_means']['service_continuity_utility']
 comps=[('twin_informed','no_adaptation'),('information_equivalent','twin_informed'),('information_equivalent','oracle')]
 if v=='revised':comps += [('adaptive_fidelity','fixed_target_fidelity'),('adaptive_checkpoint','checkpoint_disabled')]
 cs={a+' minus '+b:paired(arr(a),arr(b)) for a,b in comps}
 cs['no_switch_cost minus full']=paired(A['no_switch_cost']['policies']['twin_informed']['seed_means']['service_continuity_utility'],arr('twin_informed'))
 allstats[v]={'heldout_seeds':h['seeds'],'primary_paired':cs,'marginal':{p:stats(arr(p)) for p in pol},'interpretation':'Conditional simulation n=5seed means; whole-policy comparisons; not population effects or equivalence proof.'}
 # Recover exactly existing state draws, call only linear selector. No optimizer/policy campaign, no new numerical utility estimate.
 counts={};examples={};fields={};lastslot=None
 families=[('in_distribution',proto['split']['train_seeds']),('held_out',proto['split']['held_out_seeds'])]+[(f,proto['split']['held_out_seeds']) for f in ['high_blockage','energy_constrained','outage_heavy']]
 for family,seeds in families:
  tally=Counter()
  for seed in seeds:
   rng=np.random.default_rng(seed)
   for _ in range(24*16):
    slot=m.generate_slot(rng,family);lastslot=slot
    assert slot.latency_ms>0 and slot.edge_capacity>0
    act=m.policy_information_equivalent(slot,None,seed,proto) if v=='original' else m.call_policy('information_equivalent',slot,None,seed,proto)
    alpha={'terrestrial':1.,'local_edge_wifi':1.1,'degraded_local':1.6,'ntn_fallback':2.2,'offline_continuation':10.}
    expected=min(slot.available_networks,key=alpha.__getitem__)
    assert act.placement=='edge' and act.network==expected and act.shares==[20/slot.n_users]*slot.n_users and act.power==[1.]*slot.n_users
    if v=='revised':assert act.fidelity_level=='target' and act.checkpoint_action==act.recover_action=='none'
    # Independent score inequalities, not copied invocation of scoring code.
    L=slot.latency_ms;bonus=.25*(slot.continuity_class=='strict')+.15*(L>40)
    for n in slot.available_networks:
     assert .004*L*alpha[n]*.15+bonus+.002*max(50-slot.energy_budget,0)>0
     assert .004*L*alpha[n]*.25+bonus>0
    tally[act.network]+=1
  counts[family]={'state_assertions':sum(tally.values()),'selected_network_counts':dict(tally),'edge_placements':sum(tally.values())}
 # Separate focused boundary/ablation assertions; do not extrapolate positive-latency theorem to zero-latency ties.
 cases=[]
 slot=replace(lastslot,latency_ms=20.,packet_loss_pct=2.,energy_budget=80.,continuity_class='degraded_ok',terrestrial_outage=False,available_networks=['terrestrial','local_edge_wifi','degraded_local','ntn_fallback'],edge_capacity=.5)
 call=lambda s:m.policy_information_equivalent(s,None,100,proto) if v=='original' else m.call_policy('information_equivalent',s,None,100,proto)
 for name in ['full','no_edge_observations','no_twin_context','no_continuity_objective','no_energy_constraint','no_local_edge','no_switch_cost']:
  use=m._ablate_slot(slot,name);a=call(use);cases.append({'ablation':name,'placement':a.placement,'network':a.network,'latency_for_scoring':use.latency_ms,'loss_for_scoring':use.packet_loss_pct})
 assert cases[1]['placement']=='cloud' and cases[1]['latency_for_scoring']==0 and cases[1]['loss_for_scoring']==0
 assert next(c for c in cases if c['ablation']=='no_local_edge')['placement']!='edge'
 # Illustrative outside-domain tie: loss/scores vary but cloud first wins with zero latency and no bonuses.
 assert call(replace(slot,latency_ms=0.)).placement=='cloud'
 selection[v]={'reconstructed_existing_states':sum(x['state_assertions'] for x in counts.values()),'families':counts,'scope':'NEW focused source assertions on regenerated historical input states, not retained action traces or a new performance campaign','retained_action_traces_available':False,'focused_ablated_boundary_cases':cases,'fidelity_in_selection':'No fidelity score dimension; revised returns target/none/none by Action defaults.' if v=='revised' else 'No fidelity action in original.'}
 # Existing optimum domain N4–8,B20; ten tiny optimizer calls across both versions, not seeded outcome expansion.
 solved=[]
 for n in range(4,9):
  ss=replace(slot,n_users=n,user_priorities=[1]*n)
  target=ss if v=='original' else m.action_space_from_slot(ss)
  x,note=m._optimize_shares(target,100);assert note=='slsqp';assert abs(sum(x)-20)<1e-5  # Approximate numerical check; strict budget feasibility is recorded separately below.
  error=max(abs(y-20/n) for y in x);assert error<1e-5
  optimum=-n*math.log1p(20/n)+.2;actual=-sum(math.log1p(y) for y in x)+.01*sum(x);assert abs(actual-optimum)<1e-6  # The analytical optimum is exact; source SLSQP defaults need not return an exact feasible optimum.
  solved.append({'n_users':n,'budget':20,'backend':note,'max_share_error_vs_uniform':error,'objective_gap':actual-optimum,'budget_residual':20-sum(x),'strict_budget_feasible':sum(x)<=20})
 solves[v]=solved
 # An unavailable action receives a finite score; demonstrate with a frozen action and modified available set.
 action=m.Action([5.]*4,[1.]*4,'terrestrial','cloud','audit_unavailable')
 absent=replace(slot,available_networks=['degraded_local'],terrestrial_outage=True,n_users=4)
 out=m.predict_metrics(absent,action,None,network_switch_penalty=.05,placement_switch_penalty=.03,apply_switch=True)
 assert math.isfinite(out['service_continuity_utility']) and out['service_continuity_utility']>0
 interventions[v]={'unavailable_action_positive_score':out['service_continuity_utility'],'truth_changed_by_no_edge_observations':True,'focused_case_scope':'illustrative deterministic assertions, not additional primary outcome samples'}
 if v=='revised':
  contracts={p:sorted(m.POLICY_REQUIRED_FIELDS[p]) for p in m.POLICY_REQUIRED_FIELDS}
  obs=m.observation_from_slot(slot,'information_equivalent')
  try:obs.get('mobility');raise AssertionError('Mask failed')
  except PermissionError:pass
  params=m.resolve_synthetic_params(proto)
  assert params['fidelity_continuity_factor']==m.FIDELITY_CONTINUITY_FACTOR and params['fidelity_energy_factor']==m.FIDELITY_ENERGY_FACTOR
# Posthoc score arithmetic uses raw aggregates, not newly generated actions.
h=get('original','heldout')['policies'];ct=h['twin_informed']['switch_cost']['mean']/16;cb=h['no_adaptation']['switch_cost']['mean']/16
ut=h['twin_informed']['service_continuity_utility']['mean'];ub=h['no_adaptation']['service_continuity_utility']['mean'];cross=(ut+ct-ub-cb)/(ct-cb)
assert abs(cross-.3376618519185418)<1e-12
# No recorded JSON includes per-slot actions: root outputs retain only summaries/seed means.
report={'experimental_shas':SHAS,'source_sha256':{v:hashlib.sha256((E/'source'/v/'digital_programme.py').read_bytes()).hexdigest() for v in SHAS},'historical_non_timing_parity':parity,'independently_checked_summary_metrics':summaries,'statistics':allstats,'selector':selection,'optimizer':solves,'revised_required_observations':contracts,'truth_and_feasibility':interventions,'switching':{'crossing_lambda':cross,'gross_twin':ut+ct,'gross_fallback':ub+cb,'extra_mean_slot_charge':ct-cb,'gross_gain':ut+ct-ub-cb,'plot_lambda_domain':[0,2],'clipping_inactive_proof':'0<=gross<=1;original slot charge<=.08,lambda<=2 implies rawU>=-.16>-1'},'scientific_presentation':'OPTION B: reproducible methodological study; no causal twin-information superiority','human_scientific_approval':False,'new_campaign_executed':False,'new_seeds_added':False,'new_assertion_scope':'Input reconstruction and selector calls only; bounded boundary cases;10 tiny SLSQP problems.'}
(G/'SCIENTIFIC_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'summary_metrics_checked':summaries,'original_delta':allstats['original']['primary_paired']['twin_informed minus no_adaptation'],'revised_delta':allstats['revised']['primary_paired']['twin_informed minus no_adaptation'],'selector_state_assertions':{v:selection[v]['reconstructed_existing_states'] for v in SHAS},'optimizer_calls':10,'new_campaign':False},indent=2))
