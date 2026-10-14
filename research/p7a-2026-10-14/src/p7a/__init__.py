"""P7-A constructed runtime diagnostics; no research execution implementation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
P6=ROOT.parent/'p4-2026-10-14/fair_control'
sys.path.insert(0,str(P6/'src'))
