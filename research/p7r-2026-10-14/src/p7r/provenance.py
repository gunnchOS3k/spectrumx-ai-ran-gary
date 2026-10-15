"""Trace import/QA and evidence boundaries; metadata validation is not authenticity proof."""
from pathlib import Path
import json,math
from .contracts import ContractError,CATEGORIES,sha,digest,sensor

def strict_json(line):
 def unique(pairs):
  d={}
  for k,v in pairs:
   if k in d:raise ContractError('Duplicate JSON key')
   d[k]=v
  return d
 def invalid_constant(x):raise ContractError('Nonfinite JSON number')
 return json.loads(line,object_pairs_hook=unique,parse_constant=invalid_constant)

def ingest(raw,metadata):
 data=Path(raw).read_bytes()
 if len(data)>1048576:raise ContractError('Bounded import')
 keys={'raw_sha256','origin','source_revision','evidence_category','source_kind','rights_reference','local_analysis_permitted','clock_domain','unit','network_technology','independent_unit','acquisition_protocol','authorization_reference'}
 if set(metadata)!=keys or metadata['raw_sha256']!=sha(data):raise ContractError('Provenance fields/raw hash')
 if metadata['evidence_category'] not in CATEGORIES or metadata['unit']!='ns' or not metadata['clock_domain']:raise ContractError('Category/clock/units')
 if metadata['local_analysis_permitted'] is not True or not metadata['rights_reference']:raise ContractError('No source rights attestation')
 if metadata['source_kind'] not in {'ORIGINAL_ENGINEERING_FIXTURE','OWNER_CONTROLLED_MEASUREMENT','EXTERNAL_MEASUREMENT_DATASET'}:raise ContractError('Source kind')
 if metadata['source_kind']=='ORIGINAL_ENGINEERING_FIXTURE' and metadata['evidence_category']!='CONSTRUCTED_ENGINEERING_VERIFICATION':raise ContractError('Engineering fixture cannot become empirical evidence')
 if metadata['source_kind']!='ORIGINAL_ENGINEERING_FIXTURE' and metadata['evidence_category']!='REAL_MEASUREMENT_REPLAY':raise ContractError('Import/replay of measurements is not a new live experiment')
 if metadata['network_technology'] in {'ACTUAL_6G','6G'}:raise ContractError('No validated actual6G radio evidence in this milestone')
 if metadata['source_kind']!='ORIGINAL_ENGINEERING_FIXTURE' and any(not metadata[k] for k in ['origin','source_revision','independent_unit','acquisition_protocol','authorization_reference']):raise ContractError('Unidentified measurement context/authorization')
 rows=[strict_json(l) for l in data.decode().splitlines() if l]
 if not rows or len(rows)>1024:raise ContractError('Bounded nonempty trace')
 previous=-1
 for i,r in enumerate(rows):
  if set(r)!={'seq','timestamp_ns','event','value'} or type(r['seq']) is not int or r['seq']!=i or type(r['timestamp_ns']) is not int or r['timestamp_ns']<previous or not isinstance(r['event'],str):raise ContractError('Trace clock/order/schema')
  previous=r['timestamp_ns']
 return {'rows':rows,'provenance':metadata,'metadata_sha256':digest(metadata),'raw_sha256':sha(data),'external_authenticity_independently_verified':False,'validation_scope':'Bytes/schema/rights attestation/clock metadata only;not physical sensor calibration'}

def android_event(r):
 required={'schema','seq','elapsed_ns','uptime_ms','kind','data','capture_class'}
 if set(r)!=required or r['schema']!='p7r.android.event.v1' or r['capture_class'] not in {'UNREVIEWED_DEVICE_CAPTURE','CONSTRUCTED_ANDROID_FIXTURE'}:raise ContractError('Android envelope')
 if type(r['seq']) is not int or r['seq']<0 or type(r['elapsed_ns']) is not int or r['elapsed_ns']<0 or type(r['uptime_ms']) is not int or r['uptime_ms']<0 or r['uptime_ms']*1000000>r['elapsed_ns']:raise ContractError('Android clock domain/order')
 if r['kind'] not in {'session_start','snapshot','network_available','network_lost','network_capabilities','task_result','session_stop','collector_error'}:raise ContractError('Android event kind')
 forbidden={'imei','imsi','ssid','bssid','phone_number','student_id','location','contacts','serial','android_id'}
 def visit(x):
  if isinstance(x,dict):
   if forbidden&set(x):raise ContractError('Excluded identifier/private telemetry')
   for v in x.values():visit(v)
  elif isinstance(x,list):
   for v in x:visit(v)
 visit(r['data']);return True

def iperf_summary(x,category):
 if category not in CATEGORIES:raise ContractError('Category')
 if 'error' in x:return {'failure':x['error'],'packet_loss':None,'evidence_category':category}
 protocol=x['start']['test_start']['protocol'];end=x['end'];r=end.get('sum_received',end.get('sum',{}))
 for k in ['bytes','seconds','bits_per_second']:
  if type(r.get(k)) not in (int,float) or not math.isfinite(r[k]) or r[k]<0:raise ContractError('iperf units/counters')
 result={'protocol':protocol,'application_receiver_bytes':r['bytes'],'interval_s':r['seconds'],'receiver_bits_per_second':r['bits_per_second'],'packet_loss':None,'packet_loss_reason':'Only UDP sequence receiver statistics identify this metric','evidence_category':category}
 if protocol=='UDP' and all(k in r for k in ['packets','lost_packets','jitter_ms']):
  if not 0<=r['lost_packets']<=r['packets'] or r['jitter_ms']<0:raise ContractError('UDP counters')
  result.update(packet_loss=r['lost_packets']/r['packets'] if r['packets'] else None,jitter_ms=r['jitter_ms'],packet_loss_reason=None)
 return result


def evaluate_android_task(event,expected_hashes):
 android_event(event)
 if event['kind']!='task_result':raise ContractError('Task event required')
 d=event['data'];taskid=d['task_id']
 if taskid not in expected_hashes:raise ContractError('Independent task catalog identity required')
 from .contracts import Task,evaluate
 task=Task(taskid,expected_hashes[taskid],d['released_ns'],d['released_ns']+5000000000,'Android_elapsedRealtimeNanos_boot',30000000000,taskid not in {'original-java-sum','original-endpoint-sum'})
 result={'task_id':taskid,'clock_domain':d['clock_domain'],'finished_ns':d['finished_ns'],'generated_ns':d['released_ns'],'observed_output_sha256':d['observed_output_sha256'],'durable_reopened':d['durable_reopened'],'success_reported':d['failure'] is None,'failure':d['failure'],'evidence_category':'CONSTRUCTED_ENGINEERING_VERIFICATION' if event['capture_class']=='CONSTRUCTED_ANDROID_FIXTURE' else 'UNCLASSIFIED_PENDING_CAPTURE_REVIEW','integration_class':'ORIGINAL_EXECUTABLE_ANDROID_BENCHMARK_NOT_PRODUCTION'}
 return evaluate(task,result)

def android_capture(path):
 p=Path(path)
 if p.stat().st_size>1048576:raise ContractError('Bounded capture')
 rows=[strict_json(l) for l in p.read_text().splitlines() if l]
 if not 2<=len(rows)<=301:raise ContractError('Bounded nonempty session')
 previous=-1
 capture_class=rows[0]['capture_class']
 for i,r in enumerate(rows):
  android_event(r)
  if r['seq']!=i or r['elapsed_ns']<previous or r['capture_class']!=capture_class:raise ContractError('Sequence/boot-clock reset/mixed capture class')
  previous=r['elapsed_ns']
  if r['kind']=='task_result' and not 0<=r['data']['released_ns']<=r['data']['finished_ns']<=r['elapsed_ns']:raise ContractError('Future/inverted task timestamps')
 if rows[0]['kind']!='session_start' or rows[-1]['kind']!='session_stop':raise ContractError('Incomplete capture retained as integrity failure')
 return {'events':rows,'capture_class':capture_class,'raw_sha256':sha(p.read_bytes()),'physical_source_independently_verified':False,'clock_accuracy_validated':False,'actual_6g_radio_validation':False}
