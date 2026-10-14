"""One bounded instruction acceptance rerun of the retained held-out protocol, no new seeds."""
from pathlib import Path
import json,sys,importlib.util,hashlib,math,copy,platform,statistics
from collections import Counter
from verify_integrity import source_objects
E=Path(__file__).resolve().parents[1]
def science(x):
 if isinstance(x,dict):return {k:science(v) for k,v in x.items() if k!='compute_time_ms'}
 if isinstance(x,list):return [science(v) for v in x]
 return x
def timing(d):
 out={}
 for pol,p in d['policies'].items():
  vals=p['seed_means']['compute_time_ms'];s=p['compute_time_ms']
  assert len(vals)==5 and all(math.isfinite(x) and x>=0 for x in vals)
  assert all(math.isfinite(x) for x in s.values()) and s['mean']>=0 and s['sd']>=0 and s['n']==5
  assert math.isclose(s['mean'],statistics.fmean(vals),abs_tol=1e-12)
  assert math.isclose(s['sd'],statistics.stdev(vals),abs_tol=1e-12)
  out[pol]={'seed_means':vals,'summary':s}
 return out
def compare(a,b):
 assert set(a)==set(b);ta,tb=timing(a),timing(b)
 assert science(a)==science(b),'Scientific payload mismatch'
 return {'non_timing_exact_equal':True,'timing_separately_validated_and_preserved':{'archived':ta,'fresh':tb},'only_excluded_key':'compute_time_ms'}
def mutation_checks(a):
 b=copy.deepcopy(a);p=next(iter(b['policies']));b['policies'][p]['service_continuity_utility']['mean']+=.01
 try:compare(a,b);raise RuntimeError('Scientific mutation incorrectly accepted')
 except AssertionError:pass
 b=copy.deepcopy(a);p=next(iter(b['policies']));v=b['policies'][p]['seed_means']['compute_time_ms'];delta=.1
 b['policies'][p]['seed_means']['compute_time_ms']=[x+delta for x in v]
 for k in ['mean','ci_low','ci_high']:b['policies'][p]['compute_time_ms'][k]+=delta
 compare(a,b)
 b=copy.deepcopy(a);b['policies'][p]['seed_means']['compute_time_ms'][0]=-1
 try:compare(a,b);raise RuntimeError('Invalid timing accepted')
 except AssertionError:pass
 b=copy.deepcopy(a);b['seeds']=list(reversed(b['seeds']))
 try:compare(a,b);raise RuntimeError('Changed seed order accepted')
 except AssertionError:pass
 return {'scientific_mutation_rejected':True,'benign_timing_change_accepted_and_validated':True,'invalid_timing_rejected':True,'seed_order_mutation_rejected':True}
if __name__=='__main__':
 source_objects();out=E/'generated/heldout_acceptance';out.mkdir(parents=True,exist_ok=True);report={'campaign_expanded':False,'seeds_added':False,'python':sys.version,'platform':platform.platform(),'versions':{}}
 for v in ['original','revised']:
  f=E/'source'/v/'digital_programme.py';name='acceptance_'+v;spec=importlib.util.spec_from_file_location(name,f);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
  proto=m.load_protocol(E/'source'/v/'experiment_protocol.yaml');seeds=proto['split']['held_out_seeds'];assert seeds==[100,101,102,103,104]
  archived=json.loads((E/'data'/v/'rq2_cross_layer_continuity_heldout.json').read_text())
  backend=Counter();actual_optimizer=m._optimize_shares
  def counted_optimizer(*args,**kwargs):
   result=actual_optimizer(*args,**kwargs);backend[result[1]]+=1;return result
  m._optimize_shares=counted_optimizer
  fresh=m.run_family(proto,family='held_out',seeds=seeds)
  assert backend['slsqp']>0 and set(backend)=={'slsqp'},dict(backend)
  (out/f'{v}.json').write_text(json.dumps(fresh,indent=2)+'\n')
  report['versions'][v]={'seeds':seeds,'optimizer_backends':dict(backend),'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),**compare(archived,fresh),'mutation_tests':mutation_checks(archived)}
 (out/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'heldout_non_timing_parity':{v:r['non_timing_exact_equal'] for v,r in report['versions'].items()},'mutation_checks_per_version':4,'new_seeds':False}))
