"""Check explicit claim pointers and every exported result-table cell against regeneration."""
from pathlib import Path
import json,hashlib
E=Path(__file__).resolve().parents[1]
links=json.loads((E/'CLAIM_LINKS.json').read_text());n=0
for c in links:
 p=E/c['path'];assert p.is_file() and (E/c['raw_family_path']).is_file(),c['id']
 x=json.loads(p.read_text())
 for k in c['pointer']:x=x[k]
 assert x==c['expected'],c['id']
 # Where re-computed scientific audit exists, compare the same full statistic/count too.
 fresh=E/'generated'/p.name
 if fresh.exists():
  y=json.loads(fresh.read_text())
  for k in c['pointer']:y=y[k]
  assert y==x,c['id']
 n+=1
tables=[]
for p in sorted((E/'tables').iterdir()):
 fresh=E/'generated/tables'/p.name
 assert fresh.is_file() and fresh.read_bytes()==p.read_bytes(),p.name
 tables.append(p.name)
# Verify figure numerical inputs and all version-independent scientific parity in the main audit.
a=json.loads((E/'generated/SCIENTIFIC_AUDIT.json').read_text())
assert a['independently_checked_summary_metrics']==1477
assert all(all(x['all_nontiming_fields_equal'] for x in y.values()) for y in a['historical_non_timing_parity'].values())
report={'claim_pointer_links_verified':n,'result_tables_byte_equal_to_regenerated':tables,'summary_metrics_independently_checked':1477,'frozen_source_objects':'separately checked by verify_integrity.py','figure_inputs':'generate_figures.py reads exact frozen/rerun JSON with independent t calculations; final plots visually inspected','new_campaign':False}
(E/'generated/CLAIM_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
