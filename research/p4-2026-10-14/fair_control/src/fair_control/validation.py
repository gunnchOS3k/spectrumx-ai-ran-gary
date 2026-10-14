from pathlib import Path
import json,math,hashlib,platform,sys
from jsonschema import Draft202012Validator
from .core import ContractError,canonical,digest
ROOT=Path(__file__).resolve().parents[2]

def validate_config(config:dict)->None:
    schema=json.loads((ROOT/'schemas/config.schema.json').read_text());Draft202012Validator.check_schema(schema);Draft202012Validator(schema).validate(config)
    if config['authorization']['campaign_execution'] is not False or config['authorization']['publication'] is not False:
        raise ContractError('Preparation config cannot authorize execution/publication')
    if config['heldout']['seed_ids'] is not None or config['heldout']['outcome_paths']:
        raise ContractError('Preparation must not contain reserved seed/outcome data')
    fields=[tuple(sorted(p['observations'])) for p in config['policies']]
    if len({p['id'] for p in config['policies']})!=len(config['policies']) or len(set(fields))!=1:
        raise ContractError('Primary policy IDs unique and observation contracts identical')
    for p in config['policies']:
        if len(set(p['observations']))!=len(p['observations']):raise ContractError('Duplicate field')
    # JSON Schema number admits nonfinite Python floats in some paths; reject canonically too.
    canonical(config)

def validate_result(record:dict)->None:
    schema=json.loads((ROOT/'schemas/results.schema.json').read_text());Draft202012Validator(schema).validate(record);canonical(record)
    if record['evidence_class']!='ENGINEERING_VALIDATION':raise ContractError('No prospective results may be produced in preparation')
    if record['campaign_executed'] or record['publication_authorized']:raise ContractError('Forbidden result state')

def provenance(root:Path)->dict:
    files={}
    for d in ['src','schemas','configs','tests']:
        for p in sorted((root/d).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:files[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return {'evidence_class':'ENGINEERING_VALIDATION','files':files,'source_tree_sha256':digest(files),'python':sys.version,'platform':platform.platform(),'campaign_executed':False,'publication_authorized':False}
