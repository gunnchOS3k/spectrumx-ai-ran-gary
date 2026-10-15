"""Offline capture/export review only;never invokes ADB or an endpoint."""
from pathlib import Path
import argparse,base64,hashlib,json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'p7r-2026-10-14/src'))
from p7r.provenance import android_capture,evaluate_android_task
from p7r.contracts import ContractError
H=lambda b:hashlib.sha256(b).hexdigest()
EXPECTED={'original-java-sum':H(b'55\n'),'original-file-edit':H(b'Draft note\nEdited locally.\n'),'original-endpoint-sum':H(b'55\n')}

def decode_instrumentation(raw):
 if len(raw)>262144:raise ContractError('Bounded instrumentation output')
 text=raw.decode();fields={}
 for key in ['p7r_capture_base64','p7r_capture_sha256']:
  prefix='INSTRUMENTATION_RESULT: '+key+'=';values=[l[len(prefix):] for l in text.splitlines() if l.startswith(prefix)]
  if len(values)!=1:raise ContractError('Missing or duplicate export field')
  fields[key]=values[0]
 body=base64.b64decode(fields['p7r_capture_base64'],validate=True)
 if len(body)>65536 or H(body)!=fields['p7r_capture_sha256']:raise ContractError('Capture integrity mismatch')
 codes=[l for l in text.splitlines() if l.startswith('INSTRUMENTATION_CODE: ')]
 if len(codes)!=1:raise ContractError('Missing/ambiguous terminal status;retain raw log')
 return body,{'raw_stdout_sha256':H(raw),'capture_sha256':H(body),'instrumentation_terminal_status':codes[0],'execution_success_claimed':codes[0]=='INSTRUMENTATION_CODE: -1','physical_authenticity_verified':False}

def review(path,kind,reference=None):
 capture=android_capture(path);rows=capture['events']
 if capture['capture_class']!='UNREVIEWED_DEVICE_CAPTURE':raise ContractError('Physical acceptance review requires actual unreviewed capture,not constructed fixture label')
 start=rows[0]['data'].get('window_start_elapsed_ns');end=rows[-1]['elapsed_ns']
 if type(start) is not int or not 0<=start<=rows[0]['elapsed_ns']<=end or end-start>120000000000:raise ContractError('Acceptance window missing or exceeded;retain failed session')
 snapshots=sum(r['kind']=='snapshot' for r in rows)
 if snapshots>120:raise ContractError('Snapshot cap')
 tasks=[r for r in rows if r['kind']=='task_result'];ids=[r['data']['task_id'] for r in tasks];errors=[r for r in rows if r['kind']=='collector_error']
 if len(ids)!=len(set(ids)):raise ContractError('Repeated task/attempt')
 required={'original-java-sum','original-file-edit'} if kind=='local' else {'original-endpoint-sum'}
 if set(ids)!=required:raise ContractError('Missing/unexpected task attempt;raw retained')
 for r in tasks:
  if r['data']['released_ns']<start:raise ContractError('Task from another window')
 results=[evaluate_android_task(r,EXPECTED) for r in tasks]
 body=sum(r['data'].get('attempted_application_body_bytes',0) for r in tasks)
 if kind=='probe':
  if reference is None:raise ContractError('Original local capture required')
  local=review(reference,'local')
  if local['window_start_elapsed_ns']!=start or rows[0]['elapsed_ns']<local['last_elapsed_ns'] or rows[0]['elapsed_ns']-start>90000000000 or snapshots+local['snapshots']>120 or body>16384:raise ContractError('Cross-capture/window/snapshot/body mismatch')
  d=tasks[0]['data']
  if type(d['attempted_application_body_bytes']) is not int or not 0<=body<=16384 or type(d['received_application_body_bytes']) is not int or not 0<=d['received_application_body_bytes']<=4096:raise ContractError('Body byte bounds')
 elif body!=0:raise ContractError('Local phase must not have network task traffic')
 return {'capture_sha256':capture['raw_sha256'],'window_start_elapsed_ns':start,'last_elapsed_ns':end,'window_elapsed_ns':end-start,'snapshots':snapshots,'task_results':results,'attempted_application_body_bytes':body,'collector_error_count':len(errors),'structural_integrity_passed':True,'original_task_checks_passed':all(r['success'] for r in results) and not errors,'physical_authenticity_verified':False,'clock_accuracy_validated':False,'network_path_independently_verified':False,'measurement_acceptance_requires_owner_review':True,'publication_authorized':False}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--decode-instrumentation',type=Path);p.add_argument('--capture',type=Path);p.add_argument('--kind',choices=['local','probe']);p.add_argument('--reference-local',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('Choose a new output directory')
 if bool(a.decode_instrumentation)==bool(a.capture):p.error('Choose capture review OR instrumentation decoding')
 if a.capture and not a.kind:p.error('Capture review requires kind')
 a.output.mkdir(parents=True)
 try:
  if a.decode_instrumentation:
   b,r=decode_instrumentation(a.decode_instrumentation.read_bytes());(a.output/'probe.jsonl').write_bytes(b)
  else:r=review(a.capture,a.kind,a.reference_local)
  (a.output/'REVIEW.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
 except Exception as e:
  (a.output/'INTEGRITY_FAILURE.json').write_text(json.dumps({'error':type(e).__name__,'reason':str(e),'raw_input_unchanged':True,'publication_authorized':False},indent=2)+'\n');raise
