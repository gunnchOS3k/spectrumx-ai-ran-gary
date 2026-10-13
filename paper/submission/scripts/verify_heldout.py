#!/usr/bin/env python3
"""Reproduce each frozen Paper-II held-out vintage without changing judged code."""
from pathlib import Path
import importlib.util,json,sys,statistics,hashlib
from scipy.stats import t
p=Path(__file__).resolve().parents[1];report={}
for vintage in ('original','revised'):
 f=p/'evidence'/vintage/'digital_programme.py';name='paper_ii_frozen_'+vintage
 spec=importlib.util.spec_from_file_location(name,f);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 proto=m.load_protocol(p/'evidence'/vintage/'experiment_protocol.yaml')
 fresh=m.run_family(proto,family='held_out',seeds=proto['split']['held_out_seeds'])
 old=json.loads((p/'evidence'/vintage/'rq2_cross_layer_continuity_heldout.json').read_text())
 def strip(obj):
  if isinstance(obj,dict):return {k:strip(v) for k,v in obj.items() if k!='compute_time_ms'}
  if isinstance(obj,list):return [strip(v) for v in obj]
  return obj
 assert strip(fresh)==strip(old),vintage+' non-timing payload mismatch'
 outfile=p/'audit'/('fresh_'+vintage+'_heldout_2026-10-13.json');outfile.write_text(json.dumps(fresh,indent=2)+'\n')
 a=fresh['policies']['twin_informed']['seed_means']['service_continuity_utility'];b=fresh['policies']['no_adaptation']['seed_means']['service_continuity_utility'];delta=[x-y for x,y in zip(a,b)]
 mu=statistics.mean(delta);sd=statistics.stdev(delta);half=t.ppf(.975,len(delta)-1)*sd/len(delta)**.5
 report[vintage]={'source_sha':{'original':'86a5a6f3000fce7593ee1bdfe02bae6bc37d2498','revised':'b44357f764d440f983132bd44a9c23c180591f70'}[vintage],'fresh_heldout_nontiming_exact':True,'seeds':proto['split']['held_out_seeds'],'paired_twin_vs_noadapt':{'n':len(delta),'mean':mu,'sd':sd,'ci95':[mu-half,mu+half]},'source_snapshot_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'fresh_output_sha256':hashlib.sha256(outfile.read_bytes()).hexdigest(),'python':sys.version,'numpy':__import__('numpy').__version__,'scipy':__import__('scipy').__version__,'human_validation':False}
(p/'audit/independent_current_reproduction.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
