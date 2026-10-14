"""Extend existing P5 source hashing without rewriting P5 provenance records."""
from pathlib import Path
import hashlib
from .runner import tree_manifest
from .core import digest,ContractError
ROOT=Path(__file__).resolve().parents[2]
def manifest(root=ROOT):
    p=tree_manifest(root)
    for f in ['p6_check.py','p6_worker.py','p6_draft.py','p6_run.py']:
        q=root/f
        if q.exists():p['files'][f]=hashlib.sha256(q.read_bytes()).hexdigest()
    for q in sorted((root/'p6').rglob('*')):
        if q.is_file() and '__pycache__' not in q.parts:p['files'][str(q.relative_to(root))]=hashlib.sha256(q.read_bytes()).hexdigest()
    p['source_tree_sha256']=digest(p['files']);return p

def verify(expected,root=ROOT):
    if expected!=manifest(root):raise ContractError('P6 source or document provenance changed/incomplete')
    return True
