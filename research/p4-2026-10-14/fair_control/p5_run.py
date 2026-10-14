#!/usr/bin/env python3
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from fair_control.runner import run,ContractError
p=argparse.ArgumentParser(description='Engineering fixture only; all research modes disabled')
p.add_argument('--mode',default='ENGINEERING_TEST');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
c=json.loads((ROOT/'fixtures/p5_offline.json').read_text());c['mode']=a.mode
try:r=run(c,a.output)
except ContractError as e:p.exit(2,str(e)+'\n')
print(json.dumps({'payload_sha256':r['payload_sha256'],'primary':r['payload']['primary_outcome'],'campaign_executed':False}))
