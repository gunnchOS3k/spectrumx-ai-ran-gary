#!/usr/bin/env python3
"""One fixed constructed engineering record,never a field/pilot campaign."""
from pathlib import Path
import argparse,sys,json,tempfile,hashlib
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from p7r.workloads import execute,execute_device_os_core
from p7r.endpoint import Endpoint,request,interrupted_upload
from p7r.contracts import sha
from p7r.analysis import transactions
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
def save(n,x):(a.output/n).write_text(json.dumps(x,indent=2)+'\n')
work=[]
for kind in ['program','cached_lesson','document','restore']:
 with tempfile.TemporaryDirectory(prefix='p7r-original-') as tmp:work.append(execute(kind,tmp))
 save('EXECUTABLE_WORKLOADS.json',work)
provenance=json.loads((ROOT/'vendor/device_os/PROVENANCE.json').read_text());core=execute_device_os_core(ROOT/'vendor/device_os/offline_sync.py',provenance['sha256']);save('DEVICE_OS_CORE_INTEGRATION.json',core)
with tempfile.TemporaryDirectory() as server,tempfile.TemporaryDirectory() as client:
 with Endpoint(server) as e:
  upload=interrupted_upload(e.url,b'Original nonpersonal upload\n'*64,client)
  ping,pb=request(e.url,'/ping');comp,cb=request(e.url,'/compute',json.dumps({'values':list(range(1,11))}).encode())
  save('CONTROLLED_LOOPBACK_TRANSACTIONS.json',{'upload':upload,'ping':{'observation':ping,'server':json.loads(pb)},'remote_sum':{'observation':comp,'response':json.loads(cb),'correct':json.loads(cb)['sum']==55},'server_received_body_bytes':e.body_bytes_received,'server_events':e.events,'analysis':transactions(upload['attempts']+[ping,comp],6),'external_traffic':False,'evidence_category':'CONSTRUCTED_ENGINEERING_VERIFICATION'})
assert all(x['evaluation']['success'] for x in work) and core['correct'] and upload['correct']
save('ENGINEERING_DEMO_STATUS.json',{'evidence_category':'CONSTRUCTED_ENGINEERING_VERIFICATION','fixed_original_workloads':4,'core_library_integration':True,'loopback_upload_resume':True,'wireless_measurements':0,'device_measurements':0,'prospective_campaign_executed':False,'physical_calibration':False,'publication_authorized':False})
print('Fixed engineering demonstration passed;no device/external/wireless measurements')
