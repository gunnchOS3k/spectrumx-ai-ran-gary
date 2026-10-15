import base64,copy,hashlib,json,tempfile,unittest
from pathlib import Path
import capture_review as c
from p7r.contracts import ContractError
class Review(unittest.TestCase):
 def fixture(self,kind='local'):
  start=100;rows=[]
  def add(k,t,data):rows.append({'schema':'p7r.android.event.v1','seq':len(rows),'elapsed_ns':t,'uptime_ms':0,'kind':k,'data':data,'capture_class':'UNREVIEWED_DEVICE_CAPTURE'})
  add('session_start',100 if kind=='local' else 200,{'window_start_elapsed_ns':start})
  for i,k in enumerate(['original-java-sum','original-file-edit'] if kind=='local' else ['original-endpoint-sum']):
   t=120+i if kind=='local' else 210;d={'task_id':k,'released_ns':t,'finished_ns':t+1,'clock_domain':'Android_elapsedRealtimeNanos_boot','observed_output_sha256':c.EXPECTED[k],'durable_reopened':k=='original-file-edit','failure':None}
   if kind=='probe':d.update(attempted_application_body_bytes=32,received_application_body_bytes=10)
   add('task_result',t+2,d)
  add('session_stop',150 if kind=='local' else 220,{});return rows
 def file(self,p,rows):p.write_text(''.join(json.dumps(r)+'\n' for r in rows));return p
 def stdout(self,body):return ('INSTRUMENTATION_RESULT: p7r_capture_base64='+base64.b64encode(body).decode()+'\nINSTRUMENTATION_RESULT: p7r_capture_sha256='+c.H(body)+'\nINSTRUMENTATION_CODE: -1\n').encode()
 def test_export_roundtrip(self):
  b=b'original nonpersonal fixture\n';raw=self.stdout(b);body,result=c.decode_instrumentation(raw);self.assertEqual(b,body);self.assertTrue(result['execution_success_claimed']);self.assertFalse(result['physical_authenticity_verified'])
 def test_export_corruption_rejected(self):self.assertRaises(ContractError,c.decode_instrumentation,self.stdout(b'original').replace(c.H(b'original').encode(),b'0'*64))
 def test_duplicate_and_incomplete_export_rejected(self):
  self.assertRaises(ContractError,c.decode_instrumentation,self.stdout(b'a')+b'INSTRUMENTATION_RESULT: p7r_capture_sha256=x\n');self.assertRaises(ContractError,c.decode_instrumentation,self.stdout(b'a').replace(b'INSTRUMENTATION_CODE: -1\n',b''))
 def test_local_correctness_is_independent(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'x';rows=self.fixture();rows[1]['data']['observed_output_sha256']=c.H(b'wrong');rows[1]['data']['correct']=True;r=c.review(self.file(p,rows),'local');self.assertFalse(r['original_task_checks_passed']);self.assertTrue(r['structural_integrity_passed'])
 def test_local_and_probe_combination(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);a=self.file(p/'local',self.fixture());b=self.file(p/'probe',self.fixture('probe'));r=c.review(b,'probe',a);self.assertTrue(r['original_task_checks_passed']);self.assertFalse(r['clock_accuracy_validated'])
 def test_combined_window_exceeded_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   rows=self.fixture();rows[-1]['elapsed_ns']=120000000101;self.assertRaises(ContractError,c.review,self.file(Path(tmp)/'x',rows),'local')
 def test_duplicate_or_missing_tasks_rejected(self):
  for mutation in ['duplicate','missing']:
   with tempfile.TemporaryDirectory() as tmp:
    rows=self.fixture()
    if mutation=='duplicate':rows[2]['data']['task_id']=rows[1]['data']['task_id']
    else:rows[2]['kind']='snapshot'
    self.assertRaises(ContractError,c.review,self.file(Path(tmp)/'x',rows),'local')
 def test_fixture_category_cannot_become_measurement(self):
  with tempfile.TemporaryDirectory() as tmp:
   rows=self.fixture()
   for r in rows:r['capture_class']='CONSTRUCTED_ANDROID_FIXTURE'
   self.assertRaises(ContractError,c.review,self.file(Path(tmp)/'x',rows),'local')
 def test_probe_requires_same_window_reference(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);rows=self.fixture('probe');rows[0]['data']['window_start_elapsed_ns']=101;a=self.file(p/'local',self.fixture());b=self.file(p/'probe',rows);self.assertRaises(ContractError,c.review,b,'probe',a)
 def test_probe_payload_cap_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);rows=self.fixture('probe');rows[1]['data']['attempted_application_body_bytes']=16385;a=self.file(p/'local',self.fixture());b=self.file(p/'probe',rows);self.assertRaises(ContractError,c.review,b,'probe',a)
