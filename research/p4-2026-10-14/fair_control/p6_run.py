#!/usr/bin/env python3
"""Research-execution refusal interface. All modes remain disabled."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from fair_control.p6_configuration import execution_guard,ContractError
p=argparse.ArgumentParser();p.add_argument('--mode',required=True);p.add_argument('--config',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
try:execution_guard(a.mode,None,a.output)
except ContractError as ex:p.exit(2,str(ex)+'\n')
p.exit(2,'No research execution path exists in P6\n')
