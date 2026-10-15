#!/usr/bin/env python3
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',required=True);p.add_argument('--config',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();p.exit(2,'P7-R prospective physical/pilot/heldout execution is disabled before config or output access\n')
