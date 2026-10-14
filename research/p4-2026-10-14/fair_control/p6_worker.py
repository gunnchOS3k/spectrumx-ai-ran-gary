#!/usr/bin/env python3
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from dataclasses import asdict
from fair_control.controllers import Policy,Hardware
from fair_control.p6_control import decide_primary as decide
from fair_control.dynamics import Model
print('READY',flush=True)
line=sys.stdin.readline();start=time.perf_counter();x=json.loads(line)
p=x['policy'];p['observations']=tuple(p['observations']);h=x['hardware'];h['placement_cpu']=tuple(tuple(a) for a in h['placement_cpu'])
d=decide(Policy(**p),x['estimates'],Hardware(**h),Model(**x['model']))
print(json.dumps({'decision':asdict(d),'worker_decision_s':time.perf_counter()-start},allow_nan=False),flush=True)
