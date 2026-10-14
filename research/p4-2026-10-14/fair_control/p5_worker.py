#!/usr/bin/env python3
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from dataclasses import asdict
from fair_control.controllers import Policy,Hardware,decide
from fair_control.dynamics import Model
x=json.load(sys.stdin);p=x['policy'];p['observations']=tuple(p['observations']);h=x['hardware'];h['placement_cpu']=tuple(tuple(v) for v in h['placement_cpu'])
print(json.dumps(asdict(decide(Policy(**p),x['estimates'],Hardware(**h),Model(**x['model']))),allow_nan=False))
