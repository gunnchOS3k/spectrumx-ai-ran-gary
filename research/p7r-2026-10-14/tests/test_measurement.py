import copy,json,tempfile,unittest,hashlib,subprocess,sys
from pathlib import Path
from dataclasses import replace
from p7r.contracts import *
from p7r.workloads import execute,execute_device_os_core
from p7r.endpoint import Endpoint,request,interrupted_upload,MAX_BODY
from p7r.provenance import ingest,android_event,evaluate_android_task,iperf_summary
from p7r.capabilities import capability,matched_contract,research_guard,ACTIONS
ROOT=Path(__file__).resolve().parents[1]

class Measurement(unittest.TestCase):
 def record(self):
  t=Task('a',sha(b'original'),10,20,'same_clock',100,True)
  r={'task_id':'a','clock_domain':'same_clock','finished_ns':15,'generated_ns':10,'observed_output_sha256':sha(b'original'),'durable_reopened':True,'success_reported':True,'failure':None,'evidence_category':'CONSTRUCTED_ENGINEERING_VERIFICATION','integration_class':'ORIGINAL_TEST'};return t,r
 def test_independent_correct_output(self):t,r=self.record();self.assertTrue(evaluate(t,r)['success'])
 def test_wrong_bytes_fail_despite_reported_success(self):
  t,r=self.record();r['observed_output_sha256']=sha(b'wrong');self.assertFalse(evaluate(t,r)['success'])
 def test_late_stale_nondurable_and_failure_retained(self):
  t,r=self.record()
  for k,v in [('finished_ns',21),('durable_reopened',False),('failure','offline')]:
   bad={**r,k:v};self.assertFalse(evaluate(t,bad)['success'])
  self.assertFalse(evaluate(replace(t,max_age_ns=1),r)['success'])
 def test_clock_domain_and_order_reject(self):
  t,r=self.record();r['clock_domain']='other';self.assertRaises(ContractError,evaluate,t,r)
  r['clock_domain']='same_clock';r['finished_ns']=9;self.assertRaises(ContractError,evaluate,t,r)
 def test_all_exogenous_tasks_count_and_duplicates_reject(self):
  t,r=self.record();b=replace(t,task_id='b');self.assertEqual(all_task_fraction([t,b],[r]),.5);self.assertEqual(all_task_fraction([t,b],[]),0);self.assertRaises(ContractError,all_task_fraction,[t,b],[r,r])
 def test_sensor_unavailable_not_zero(self):
  self.assertIsNone(sensor(None,'dBm','PERMISSION_DENIED','No permission')['value']);self.assertRaises(ContractError,sensor,0,'dBm','UNAVAILABLE','missing');self.assertRaises(ContractError,sensor,float('nan'),'J')
 def test_clock_offset_interval_no_oneway_inference(self):
  x=clock_bounds(100,220,230,150);self.assertEqual(x['server_minus_client_offset_interval_ns'],[80,120]);self.assertIsNone(x['one_way_latency_ns'])
 def test_clock_drift_or_order_reject(self):self.assertRaises(ContractError,clock_bounds,100,220,230,90);self.assertRaises(ContractError,clock_bounds,100,220,400,150)
 def test_original_executable_workloads(self):
  for kind in ['program','cached_lesson','document','restore']:
   with tempfile.TemporaryDirectory() as p:
    x=execute(kind,p);self.assertTrue(x['evaluation']['success']);self.assertEqual(x['result']['integration_class'],'ORIGINAL_EXECUTABLE_BENCHMARK_NOT_PRODUCTION_APP')
 def test_no_private_or_nonempty_work_directory(self):
  with tempfile.TemporaryDirectory() as p:
   (Path(p)/'keep').write_text('unchanged');self.assertRaises(ContractError,execute,'document',p);self.assertEqual((Path(p)/'keep').read_text(),'unchanged')
 def test_actual_pinned_device_os_core(self):
  f=ROOT/'vendor/device_os/offline_sync.py';h=json.loads((ROOT/'vendor/device_os/PROVENANCE.json').read_text())['sha256'];x=execute_device_os_core(f,h);self.assertTrue(x['correct']);self.assertTrue(x['duplicate_content_preserved']);self.assertFalse(x['production_application_executed']);self.assertFalse(x['durable_cloud_receipt_verified'])
 def test_core_source_mutation_reject(self):self.assertRaises(ContractError,execute_device_os_core,ROOT/'vendor/device_os/offline_sync.py','0'*64)
 def test_live_loopback_upload_pause_resume_and_dedup(self):
  with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
   with Endpoint(a) as e:
    data=b'original nonpersonal chunk\n'*100;x=interrupted_upload(e.url,data,b);self.assertTrue(x['correct']);self.assertTrue(x['progress_preserved']);self.assertFalse(x['actual_wireless_outage']);self.assertEqual(e.body_bytes_received,len(data)+len(data)//2)
 def test_loopback_latency_bytes_and_packet_loss_unknown(self):
  with tempfile.TemporaryDirectory() as p:
   with Endpoint(p) as e:
    r,b=request(e.url,'/ping');self.assertIsNone(r['failure']);self.assertGreater(r['duration_ns'],0);self.assertEqual(r['received_application_bytes'],len(b));self.assertIsNone(r['packet_loss']);self.assertEqual(r['observed_transport'],'LOOPBACK')
 def test_bounded_remote_sum_actual_server_compute(self):
  with tempfile.TemporaryDirectory() as p:
   with Endpoint(p) as e:
    r,b=request(e.url,'/compute',json.dumps({'values':[1,2,3,4]}).encode());self.assertIsNone(r['failure']);self.assertEqual(json.loads(b)['sum'],10)
 def test_nonlocal_endpoint_refused_before_dns(self):
  for u in ['https://example.com','http://localhost:80','http://127.0.0.1.evil:80','http://user@127.0.0.1:80','ftp://127.0.0.1:80']:self.assertRaises(ContractError,request,u,'/ping')
 def test_connection_failure_is_retained(self):
  with tempfile.TemporaryDirectory() as p:
   with Endpoint(p) as e:u=e.url
   r,b=request(u,'/ping');self.assertIsNotNone(r['failure']);self.assertGreaterEqual(r['duration_ns'],0);self.assertEqual(r['received_application_bytes'],0)
 def test_server_rejects_offset_and_path_traversal(self):
  with tempfile.TemporaryDirectory() as p:
   with Endpoint(p) as e:
    self.assertEqual(request(e.url,'/upload/a/5',b'x')[0]['http_status'],409);self.assertEqual(request(e.url,'/upload/../../secret/0',b'x')[0]['http_status'],404)
 def test_transfer_size_and_malformed_compute_reject(self):
  self.assertRaises(ContractError,request,'http://127.0.0.1:1234','/compute',b'x'*(MAX_BODY+1))
  with tempfile.TemporaryDirectory() as p:
   with Endpoint(p) as e:self.assertEqual(request(e.url,'/compute',b'{broken')[0]['http_status'],400)
 def metadata(self,raw):return {'raw_sha256':sha(raw),'origin':'original author fixture','source_revision':'constructed-v1','evidence_category':'CONSTRUCTED_ENGINEERING_VERIFICATION','source_kind':'ORIGINAL_ENGINEERING_FIXTURE','rights_reference':'original owned nonpersonal test content','local_analysis_permitted':True,'clock_domain':'fixture_monotonic','unit':'ns','network_technology':'LOOPBACK','independent_unit':'fixture_not_sample','acquisition_protocol':'constructed','authorization_reference':None}
 def trace(self):return b'{"seq":0,"timestamp_ns":1,"event":"start","value":null}\n{"seq":1,"timestamp_ns":2,"event":"stop","value":null}\n'
 def test_trace_schema_hash_and_metadata_binding(self):
  raw=self.trace()
  with tempfile.TemporaryDirectory() as p:
   f=Path(p)/'x.jsonl';f.write_bytes(raw);x=ingest(f,self.metadata(raw));self.assertEqual(len(x['rows']),2);self.assertFalse(x['external_authenticity_independently_verified'])
 def test_trace_hash_mutation_reject(self):
  raw=self.trace()
  with tempfile.TemporaryDirectory() as p:
   f=Path(p)/'x';f.write_bytes(raw+b' ');self.assertRaises(ContractError,ingest,f,self.metadata(raw))
 def test_trace_rights_and_fabricated_evidence_reject(self):
  raw=self.trace()
  with tempfile.TemporaryDirectory() as p:
   f=Path(p)/'x';f.write_bytes(raw)
   for k,v in [('local_analysis_permitted',False),('evidence_category','LIVE_CONTROLLED_WIRELESS'),('network_technology','6G')]:
    m=self.metadata(raw);m[k]=v;self.assertRaises(ContractError,ingest,f,m)
 def test_imported_measured_trace_is_replay_not_live(self):
  raw=self.trace()
  with tempfile.TemporaryDirectory() as p:
   f=Path(p)/'x';f.write_bytes(raw);m=self.metadata(raw);m.update(source_kind='EXTERNAL_MEASUREMENT_DATASET',evidence_category='LIVE_CONTROLLED_WIRELESS',authorization_reference='fixture-attestation-only');self.assertRaises(ContractError,ingest,f,m)
 def test_clock_reset_or_duplicate_sequence_reject(self):
  for raw in [b'{"seq":0,"timestamp_ns":2,"event":"a","value":0}\n{"seq":1,"timestamp_ns":1,"event":"b","value":0}\n',b'{"seq":2,"timestamp_ns":1,"event":"a","value":0}\n']:
   with tempfile.TemporaryDirectory() as p:
    f=Path(p)/'x';f.write_bytes(raw);self.assertRaises(ContractError,ingest,f,self.metadata(raw))
 def event(self):return {'schema':'p7r.android.event.v1','seq':0,'elapsed_ns':2000000,'uptime_ms':1,'kind':'snapshot','data':{'network':{'transports':['WIFI']}},'capture_class':'CONSTRUCTED_ANDROID_FIXTURE'}
 def test_android_contract_and_missing_telemetry(self):self.assertTrue(android_event(self.event()))
 def test_android_clock_and_private_identifiers_reject(self):
  e=self.event();e['uptime_ms']=3;self.assertRaises(ContractError,android_event,e);e=self.event();e['data']['network']['ssid']='private';self.assertRaises(ContractError,android_event,e)
 def test_device_claimed_correctness_not_used_as_oracle(self):
  e=self.event();e['kind']='task_result';e['data']={'task_id':'original-java-sum','released_ns':1,'finished_ns':10,'clock_domain':'Android_elapsedRealtimeNanos_boot','observed_output_sha256':sha(b'wrong'),'correct':True,'expected_sha256':sha(b'wrong'),'durable_reopened':False,'failure':None};self.assertFalse(evaluate_android_task(e,{'original-java-sum':sha(b'55\n')})['success'])
 def test_iperf_tcp_does_not_manufacture_packet_loss(self):
  x=iperf_summary({'start':{'test_start':{'protocol':'TCP'}},'end':{'sum_received':{'bytes':100,'seconds':1,'bits_per_second':800}}},'CONSTRUCTED_ENGINEERING_VERIFICATION');self.assertIsNone(x['packet_loss'])
 def test_iperf_udp_counters_and_failure(self):
  x=iperf_summary({'start':{'test_start':{'protocol':'UDP'}},'end':{'sum':{'bytes':90,'seconds':1,'bits_per_second':720,'packets':10,'lost_packets':1,'jitter_ms':2}}},'CONSTRUCTED_ENGINEERING_VERIFICATION');self.assertEqual(x['packet_loss'],.1);self.assertIsNotNone(iperf_summary({'error':'failed'},'CONSTRUCTED_ENGINEERING_VERIFICATION')['failure'])
 def test_evidence_claim_boundaries(self):
  self.assertFalse(claim_allowed('REAL_MEASUREMENT_REPLAY','WIFI','LIVE_PHYSICAL_EXPERIMENT'));self.assertFalse(claim_allowed('LIVE_CONTROLLED_WIRELESS','WIFI','MEASURED_NTN'));self.assertFalse(claim_allowed('LIVE_CONTROLLED_WIRELESS','NR','ACTUAL_6G_RADIO_PERFORMANCE'));self.assertFalse(claim_allowed('CONSTRUCTED_ENGINEERING_VERIFICATION','LOOPBACK','LIVE_PHYSICAL_EXPERIMENT'))
 def test_capability_does_not_grant_carrier_control(self):
  for action in ['RAN_CONFIG','RADIO_SCHEDULING','PHYSICAL_HANDOVER','NTN_FALLBACK']:self.assertFalse(capability('ANDROID_UNPRIVILEGED',action)['implemented_and_tested_here'])
 def test_matched_contract_and_mismatch(self):
  a={k:[] for k in ['observations','eligible_actions','resource_limits','compute_limits','switch_costs','endpoint_scope','missing_observation_rule']};self.assertTrue(matched_contract(a,copy.deepcopy(a)));b=copy.deepcopy(a);b['compute_limits']=[2];self.assertRaises(ContractError,matched_contract,a,b)
 def test_execution_guards_no_config_output(self):
  for mode in ['PILOT_EXPERIMENT','HELDOUT_EVALUATION','PHYSICAL_EXPERIMENT','LIVE_WIRELESS_EXPERIMENT']:
   with tempfile.TemporaryDirectory() as p:
    f=Path(p)/'must-not-exist';r=subprocess.run([sys.executable,str(ROOT/'p7r_run.py'),'--mode',mode,'--config',str(Path(p)/'reserved'),'--output',str(f)],capture_output=True,text=True);self.assertEqual(r.returncode,2);self.assertFalse(f.exists())
 def test_android_manifest_and_source_do_not_autorun_network_probe(self):
  import xml.etree.ElementTree as ET
  root=ET.parse(ROOT/'android/AndroidManifest.xml').getroot();ns={'a':'http://schemas.android.com/apk/res/android'};perms={p.attrib['{'+ns['a']+'}name'] for p in root.findall('uses-permission')};self.assertNotIn('android.permission.ACCESS_FINE_LOCATION',perms);activity=(ROOT/'android/src/com/gunnchos/research/p7r/CollectorActivity.java').read_text();self.assertNotIn('ScopedNetworkProbe.transfer',activity);self.assertIn('stopCapture()',activity)
 def test_json_schemas_are_valid_and_match_fixture(self):
  from jsonschema import Draft202012Validator
  for n in ['trace-record.schema.json','android-event.schema.json']:Draft202012Validator.check_schema(json.loads((ROOT/'schemas'/n).read_text()))
  Draft202012Validator(json.loads((ROOT/'schemas/android-event.schema.json').read_text())).validate(self.event())
 def test_trace_duplicate_keys_nonfinite_and_bool_sequence_reject(self):
  for raw in [b'{"seq":0,"seq":1,"timestamp_ns":1,"event":"a","value":0}\n',b'{"seq":0,"timestamp_ns":1,"event":"a","value":NaN}\n',b'{"seq":false,"timestamp_ns":1,"event":"a","value":0}\n']:
   with tempfile.TemporaryDirectory() as p:
    f=Path(p)/'x';f.write_bytes(raw);self.assertRaises(ContractError,ingest,f,self.metadata(raw))
 def test_transaction_analysis_retains_missing_and_failed_attempts(self):
  from p7r.analysis import transactions
  rows=[{'duration_ns':1000000,'received_application_bytes':100,'http_status':200,'failure':None},{'duration_ns':2000000,'received_application_bytes':0,'http_status':None,'failure':'offline'}];x=transactions(rows,3);self.assertEqual(x['success_fraction'],1/3);self.assertEqual(x['missing_attempts'],1);self.assertIsNone(x['packet_loss']);self.assertEqual(x['application_response_goodput_bits_s'],[800000])
 def test_paired_session_analysis_not_packet_pseudoreplication(self):
  from p7r.analysis import paired_session_summary
  rows=[{'unit':u,'policy':p,'successes':a,'eligible':10,'task_manifest_sha256':'original-taskset'} for u,p,a in [('block1','mpc',5),('block1','hysteresis',7),('block2','mpc',8),('block2','hysteresis',8)]];x=paired_session_summary(rows);self.assertAlmostEqual(x['mean_difference'],-.1);self.assertEqual(x['independent_unit_count'],2);self.assertFalse(x['causal_superiority_established']);self.assertRaises(ContractError,paired_session_summary,rows[:-1])
 def test_paired_task_population_mismatch_reject(self):
  from p7r.analysis import paired_session_summary
  a={'unit':'block','policy':'mpc','successes':1,'eligible':2,'task_manifest_sha256':'A'};b={**a,'policy':'hysteresis','task_manifest_sha256':'B'};self.assertRaises(ContractError,paired_session_summary,[a,b])
 def capture_rows(self):
  a=self.event();a['kind']='session_start';b=copy.deepcopy(a);b.update(seq=1,elapsed_ns=3000000,uptime_ms=2,kind='session_stop');return [a,b]
 def test_android_capture_complete_sequence(self):
  from p7r.provenance import android_capture
  with tempfile.TemporaryDirectory() as p:
   f=Path(p)/'x';f.write_text(''.join(json.dumps(x)+'\n' for x in self.capture_rows()));r=android_capture(f);self.assertFalse(r['physical_source_independently_verified']);self.assertFalse(r['clock_accuracy_validated'])
 def test_android_capture_ambiguous_or_nonfinite_json_reject(self):
  from p7r.provenance import android_capture
  valid=''.join(json.dumps(x)+'\n' for x in self.capture_rows())
  for raw in [valid.replace('"seq": 0','"seq": 0, "seq": 0',1),valid.replace('"data": {','"data": {"unvalidated": NaN,',1)]:
   with tempfile.TemporaryDirectory() as p:
    f=Path(p)/'x';f.write_text(raw);self.assertRaises(ContractError,android_capture,f)
 def test_android_capture_boot_reset_incomplete_or_mixed_class_reject(self):
  from p7r.provenance import android_capture
  for kind in ['reset','incomplete','mixed']:
   rows=self.capture_rows()
   if kind=='reset':rows[-1]['elapsed_ns']=1;rows[-1]['uptime_ms']=0
   elif kind=='incomplete':rows[-1]['kind']='snapshot'
   else:rows[-1]['capture_class']='UNREVIEWED_DEVICE_CAPTURE'
   with tempfile.TemporaryDirectory() as p:
    f=Path(p)/'x';f.write_text(''.join(json.dumps(x)+'\n' for x in rows));self.assertRaises(ContractError,android_capture,f)
 def test_android_capture_future_task_reject(self):
  from p7r.provenance import android_capture
  rows=self.capture_rows();task=copy.deepcopy(rows[0]);task.update(seq=1,kind='task_result');task['data']={'released_ns':1,'finished_ns':999999999};rows[1]['seq']=2;rows.insert(1,task)
  with tempfile.TemporaryDirectory() as p:
   f=Path(p)/'x';f.write_text(''.join(json.dumps(x)+'\n' for x in rows));self.assertRaises(ContractError,android_capture,f)
