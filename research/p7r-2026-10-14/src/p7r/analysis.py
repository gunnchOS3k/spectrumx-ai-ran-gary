"""Named measurement estimands; no scientific data or causal inference created."""
import math,statistics
from .contracts import ContractError

def transactions(rows,registered_attempts):
 if type(registered_attempts) is not int or registered_attempts<len(rows) or registered_attempts<1:raise ContractError('All attempted population required')
 for r in rows:
  if type(r['duration_ns']) is not int or r['duration_ns']<=0 or type(r['received_application_bytes']) is not int or r['received_application_bytes']<0:raise ContractError('Time/body byte units')
 ok=[r for r in rows if r['failure'] is None and r['http_status'] is not None and 200<=r['http_status']<300]
 delays=[r['duration_ns']/1e6 for r in ok]
 variation=[abs(b-a) for a,b in zip(delays,delays[1:])]
 return {'all_attempts':registered_attempts,'recorded_attempts':len(rows),'missing_attempts':registered_attempts-len(rows),'successes':len(ok),'success_fraction':len(ok)/registered_attempts,'successful_transaction_mean_ms':statistics.mean(delays) if delays else None,'successful_transaction_max_ms':max(delays) if delays else None,'mean_adjacent_successful_transaction_duration_difference_ms':statistics.mean(variation) if variation else None,'jitter_definition':'Absolute adjacent successful application-transaction duration variation;not packet IPDV/RFC3550 jitter','application_response_goodput_bits_s':[r['received_application_bytes']*8/(r['duration_ns']/1e9) for r in ok],'goodput_scope':'Received HTTP body per end-to-end transaction;not link capacity,wire bytes or upload persisted unique bytes','packet_loss':None,'failure_fraction':1-len(ok)/registered_attempts}

def paired_session_summary(records):
 groups={}
 for r in records:
  if set(r)!= {'unit','policy','successes','eligible','task_manifest_sha256'} or r['policy'] not in {'mpc','hysteresis'} or type(r['eligible']) is not int or type(r['successes']) is not int or not 0<=r['successes']<=r['eligible'] or not r['eligible']:raise ContractError('Paired session counts')
  g=groups.setdefault(r['unit'],{})
  if r['policy'] in g:raise ContractError('Duplicate policy unit')
  g[r['policy']]=r
 differences=[]
 for k,g in groups.items():
  if set(g)!= {'mpc','hysteresis'}:raise ContractError('Missing policy retained as integrity event;cannot silently delete pair')
  a,b=g['mpc'],g['hysteresis']
  if a['eligible']!=b['eligible'] or a['task_manifest_sha256']!=b['task_manifest_sha256']:raise ContractError('Unmatched eligible task populations')
  differences.append(a['successes']/a['eligible']-b['successes']/b['eligible'])
 if not differences:raise ContractError('Nonempty independent session/block units')
 mean=statistics.mean(differences);sd=statistics.stdev(differences) if len(differences)>1 else None
 return {'independent_unit_count':len(differences),'paired_fraction_differences':differences,'mean_difference':mean,'sample_sd':sd,'confidence_interval':None,'interpretation':'Descriptive paired session/block differences only;CI design/independence/randomization require approved prospective protocol;packets/tasks are not independent sessions','causal_superiority_established':False,'equivalence_established':False}
