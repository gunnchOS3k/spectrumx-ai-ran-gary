#!/usr/bin/env python3
"""Source-disabled research guard; benchmark/validation are separate fixed tools."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',required=True);p.add_argument('--config',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
p.exit(2,'P7-A research execution disabled before config/output access; no authorization token or research runner exists\n')
