"""P5 engineering-only runner. Research modes are hard-disabled in source."""
from pathlib import Path
from dataclasses import asdict
import json,subprocess,hashlib,sys,platform,importlib.metadata,copy
from .core import Action,Allocation,TruthFrame,PathTruth,Resources,ContractError,canonical,digest
from .tasks import TaskSpec,TaskRequirement,Content
from .dynamics import Environment,Model
from .controllers import Hardware,Policy,decide,validate_estimates,FIELDS
MODES=('ENGINEERING_TEST','PILOT_EXPERIMENT','CONFIRMATORY_EXPERIMENT','HELDOUT_EVALUATION')
AUTH={'scientific_protocol_approved':False,'pilot_execution':False,'confirmatory_execution':False,'heldout_execution':False,'human_scientific_approval':False,'publication':False}
ROOT=Path(__file__).resolve().parents[2]

def validate_fixture(config):
    if set(config)!= {'version','mode','fixture_id','seed_manifest','authorization','model','hardware','energy_j','tasks','contents','frames','policy'}:raise ContractError('Engineering fixture fields')
    if config['version']!='p5-engineering-fixture-0.1' or config['mode']!='ENGINEERING_TEST':raise ContractError('Research execution disabled in P5')
    if config['authorization']!=AUTH:raise ContractError('No config may grant research authorization')
    if config['fixture_id']!='constructed-offline-copy' or config['seed_manifest']!={'kind':'CONSTRUCTED_NO_RANDOMNESS','seed_ids':[],'heldout_paths':[]}:raise ContractError('No reserved seeds/holdout paths')
    # Bundled fixture identity is pinned; arbitrary user research scenarios cannot pass as engineering.
    allowed=json.loads((ROOT/'fixtures/p5_offline.json').read_text())
    if digest(config)!=digest(allowed):raise ContractError('Only source-controlled bounded engineering fixture allowed')
    if len(config['frames'])>8:raise ContractError('Engineering frame bound')
    canonical(config)

def tree_manifest(root=ROOT):
    files={}
    for d in ('src','tests','configs','schemas','fixtures','p5'):
        if not (root/d).exists():continue
        for p in sorted((root/d).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:files[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ['requirements.txt','prepare.py','p5_check.py','p5_run.py']:
        p=root/name
        if p.exists():files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    return {'files':files,'source_tree_sha256':digest(files)}

def verify_manifest(manifest,root=ROOT):
    actual=tree_manifest(root)
    if manifest!=actual:raise ContractError('Incomplete/changed source provenance')
    return True

def revision(root=ROOT):
    try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError,FileNotFoundError):
        p=root/'SOURCE_REVISION.json'
        if not p.exists():raise ContractError('Clean copy requires retained SOURCE_REVISION.json')
        x=json.loads(p.read_text());sha=x['publication_repository_sha']
        if len(sha)!=40 or any(c not in '0123456789abcdef' for c in sha):raise ContractError('Invalid source revision')
        return sha

def run(config,output=None):
    validate_fixture(config) # guard before any output or environment construction
    model=Model(**config['model']);hw=Hardware(**{**config['hardware'],'placement_cpu':tuple(tuple(x) for x in config['hardware']['placement_cpu'])})
    specs=[TaskSpec(requirement=TaskRequirement(**s['requirement']),**{k:tuple(v) if k=='content_keys' else v for k,v in s.items() if k!='requirement'}) for s in config['tasks']]
    env=Environment(specs,config['energy_j'],[Content(**c) for c in config['contents']],model);policy=Policy(**config['policy']);decisions=[]
    for f in config['frames']:
        t=f['truth'];truth=TruthFrame(t['frame_id'],tuple(t['task_ids']),tuple(PathTruth(**p) for p in t['paths']),tuple(t['placement_cpu'].items()),Resources(**t['resources']),t['duration_s'])
        # Exogenous estimates are authored separately in this engineering fixture, never copied from truth here.
        est=copy.deepcopy(f['estimates']);est['timestamp_s']=env.now_s;est['remaining_energy_j']=env.energy_j;est['previous_action']=asdict(env.previous) if env.previous else None
        est['task_queue']=[]
        for s in specs:
            v=env.tasks[s.task_id]
            if v.state.value in {'CREATED','FAILED','COMPLETED'}:continue
            est['task_queue'].append({'task_id':s.task_id,'remaining_mcycles':s.work_mcycles-v.progress_mcycles,'remaining_mbit':s.input_mbit-v.delivered_mbit,'memory_mb':s.memory_mb,'deadline_s':s.requirement.deadline_s,'minimum_quality':s.requirement.minimum_quality,'content_age_s':env._age(s,env.now_s) if all(k in env.contents for k in s.content_keys) else s.requirement.max_age_s+1,'max_age_s':s.requirement.max_age_s,'local_eligible':s.local_eligible,'network_required':s.network_required})
        decision=decide(policy,est,hw,model);decisions.append(asdict(decision));env.step(truth,decision.action,f['latency_s'])
    p=tree_manifest();source=revision();component=lambda name:hashlib.sha256((ROOT/'src/fair_control'/name).read_bytes()).hexdigest()
    timing=[{'elapsed_s':d.pop('elapsed_s')} for d in decisions] # retained separately, deterministic scientific/engineering payload
    payload={'schema_version':'p5-engineering-result-0.1','mode':'ENGINEERING_TEST','fixture_id':config['fixture_id'],'source_sha':source,'configuration_sha256':digest(config),'controller_revision':component('controllers.py'),'model_revision':component('dynamics.py'),'environment_revision':component('tasks.py'),'seed_manifest':config['seed_manifest'],'policy_information_declaration':list(FIELDS),'resource_budget_declaration':config['hardware'],'compute_budget_declaration':asdict(policy),'primary_outcome':{'schema':'all_exogenous_tasks_correct_deadline_fresh_quality_durable_fraction','scheduled':len(specs),'successes':len(env.observed),'fraction':env.evaluator.fraction(env.observed)},'secondary_outcome':{'energy_used_j':env.initial_energy_j-env.energy_j,'queue_length':env.queue_length,'failures':{k:t.failure for k,t in env.tasks.items() if t.failure},'invalid_actions':sum(not t.get('accepted',True) for t in env.trace)},'decisions':decisions,'checkpoint':env.snapshot(),'source_tree_sha256':p['source_tree_sha256'],'errors':[],'campaign_executed':False,'authorization':AUTH}
    record={'payload':payload,'payload_sha256':digest(payload),'timing_provenance':timing,'environment':{'python':sys.version,'platform':platform.platform(),'packages':{k:importlib.metadata.version(k) for k in ['jsonschema','numpy','scipy']},'fresh_installation_tested':False},'manifest':p}
    validate_output(record)
    if output:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        for name,value in [('ENGINEERING_RUN.json',record),('RUN_STATE.json',{'mode':'ENGINEERING_TEST','state':'COMPLETED','payload_sha256':record['payload_sha256'],'authorization':AUTH}),('SOURCE_PROVENANCE.json',p)]:
            tmp=output/(name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(output/name)
    return record

def validate_output(record):
    p=record['payload']
    if p['mode']!='ENGINEERING_TEST' or p['campaign_executed'] is not False or p['authorization']!=AUTH or record['payload_sha256']!=digest(p):raise ContractError('Invalid output integrity/auth')
    if p['seed_manifest']!={'kind':'CONSTRUCTED_NO_RANDOMNESS','seed_ids':[],'heldout_paths':[]}:raise ContractError('Holdout access')
    n=p['primary_outcome']['scheduled'];s=p['primary_outcome']['successes']
    if type(n) is not int or type(s) is not int or not 0<=s<=n or n<1 or p['primary_outcome']['fraction']!=s/n:raise ContractError('Primary numerator/denominator')
    for key in ['source_sha','configuration_sha256','controller_revision','model_revision','environment_revision','source_tree_sha256']:
        v=p[key];length=40 if key=='source_sha' else 64
        if not isinstance(v,str) or len(v)!=length or any(c not in '0123456789abcdef' for c in v):raise ContractError('Incomplete provenance')
    canonical(record)
