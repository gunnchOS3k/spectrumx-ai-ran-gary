#!/usr/bin/env python3
"""Sequential pure-function engineering requests; never future batches/state."""
from pathlib import Path
import sys,time,json,resource
from dataclasses import asdict
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import p7a
from p7a.session import VERSION,MAX_REQUEST
from fair_control.controllers import Policy,Hardware
from fair_control.dynamics import Model
from fair_control.p6_control import decide_primary
from fair_control.core import digest,ContractError
print(json.dumps({'ready':VERSION}),flush=True)
family=None
for seq in range(64):
    line=sys.stdin.buffer.readline(MAX_REQUEST+2)
    if not line:break
    w=time.perf_counter();cpu=time.process_time()
    if len(line)>MAX_REQUEST or not line.endswith(b'\n'):raise ContractError('Bounded complete request')
    x=json.loads(line)
    if set(x)!={'version','request_id','policy','hardware','model','estimates'} or x['version']!=VERSION or x['request_id']!=seq:raise ContractError('Sequential protocol')
    p=x['policy'];p['observations']=tuple(p['observations']);policy=Policy(**p)
    if family is None:family=policy.family
    if policy.family!=family or family not in {'mpc','hysteresis'}:raise ContractError('One eligible family per session')
    h=x['hardware'];h['placement_cpu']=tuple(tuple(a) for a in h['placement_cpu'])
    decision=decide_primary(policy,x['estimates'],Hardware(**h),Model(**x['model']))
    reply={'version':VERSION,'request_id':seq,'input_sha256':digest(x['estimates']),'decision':asdict(decision),'worker_wall_s':time.perf_counter()-w,'worker_cpu_s':time.process_time()-cpu,'peak_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    print(json.dumps(reply,allow_nan=False),flush=True)
    del x,p,h,policy,decision,reply
