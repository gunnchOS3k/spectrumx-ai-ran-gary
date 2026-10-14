"""Portable exact SHA256 membership and captured source-object checks."""
from pathlib import Path
import hashlib,json,sys
E=Path(__file__).resolve().parents[1]
def verify(root):
 root=Path(root);manifest=root/'SHA256SUMS';entries={}
 for line in manifest.read_text().splitlines():
  h,p=line.split('  ',1);assert p not in entries and len(h)==64
  assert not Path(p).is_absolute() and '..' not in Path(p).parts
  entries[p]=h
 actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p!=manifest and 'generated' not in p.relative_to(root).parts and '__pycache__' not in p.parts}
 assert actual==set(entries),{'missing':list(set(entries)-actual),'extra':list(actual-set(entries))}
 for p,h in entries.items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p
 return len(entries)
def source_objects():
 d=json.loads((E/'source/MANIFEST.json').read_text())
 for p,m in d.items():assert hashlib.sha256((E/p).read_bytes()).hexdigest()==m['sha256'],p
 return len(d)
if __name__=='__main__':
 print(json.dumps({'manifest_payloads_verified':verify(Path(sys.argv[1]) if len(sys.argv)>1 else E),'frozen_objects_verified':source_objects()}))
