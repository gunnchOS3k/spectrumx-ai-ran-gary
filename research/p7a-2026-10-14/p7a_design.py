#!/usr/bin/env python3
"""Export or verify review-only overlay; cannot execute a research design."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from p7a.design import make_candidate,validate,file_hash,validate_decisions
from fair_control.core import digest,ContractError
p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);p.add_argument('--base-config',type=Path,required=True);p.add_argument('--export',action='store_true');p.add_argument('--verify',action='store_true');a=p.parse_args()
base=json.loads(a.base_config.read_text());protocol=file_hash(a.review/'P7A_PILOT_PROTOCOL.md');decisions=file_hash(a.review/'P7A_OWNER_DECISION_RECORD.json');record=json.loads((a.review/'P7A_OWNER_DECISION_RECORD.json').read_text())
validate_decisions(record)
if a.export and not a.verify:
 pin=json.loads((ROOT/'P7A_SOURCE_REVISION.json').read_text());c=make_candidate(pin['implementation_content_sha'],base,protocol,decisions);validate(c,base,protocol,decisions)
 seal={'candidate_canonical_sha256':digest(c),'source':c['source'],'authorization':c['authorization']}
 for n,x in [('P7A_PILOT_FREEZE_CANDIDATE.json',c),('P7A_FREEZE_SEAL.json',seal)]:
  (a.review/n).write_text(json.dumps(x,indent=2)+'\n')
elif a.verify and not a.export:
 c=json.loads((a.review/'P7A_PILOT_FREEZE_CANDIDATE.json').read_text());seal=json.loads((a.review/'P7A_FREEZE_SEAL.json').read_text());validate(c,base,protocol,decisions)
 if seal!={'candidate_canonical_sha256':digest(c),'source':c['source'],'authorization':c['authorization']}:raise ContractError('Candidate seal changed')
else:p.error('Choose one --export or --verify')
print(json.dumps({'candidate_canonical_sha256':digest(c),'scientific_protocol_approved':False,'pilot_execution_authorized':False,'heldout_execution_authorized':False,'publication_authorized':False}))
