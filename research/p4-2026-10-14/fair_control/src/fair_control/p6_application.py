"""Original isolated reference workloads. No production WAIKE/Device OS import.

Actual CPython/file operations validate adapter mechanics. All content/parameters
are constructed and results are not calibration or learner/application evidence.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import json,hashlib,os,subprocess,sys,tempfile
from .core import ContractError,number,digest,canonical,TaskRequirement,TaskOutcome,completion_fraction

TEXT='Original constructed lesson: add the integers 1 through 10.\n'
NOTE='Draft note\nEdited locally.\n'
PROGRAM='print(sum(range(1, 11)))\n'

def bytes_hash(b):return hashlib.sha256(b).hexdigest()

def atomic_write(path:Path,data:bytes):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    # File fsync/rename checked on this host only; no hardware crash/directory fsync guarantee.

@dataclass(frozen=True)
class ReferenceResult:
    task_id:str
    output_utf8:str
    persisted_utf8:str|None
    virtual_finished_s:float
    generated_s:float
    quality:float
    failure:str|None=None
    evidence_class:str='CONSTRUCTED_REFERENCE_ADAPTER_CHECK'

@dataclass(frozen=True)
class ReferenceTask:
    task_id:str
    adapter:str
    eligibility:str
    required_input:str
    expected_output_utf8:str
    deadline_s:float
    quality_minimum:float
    freshness_limit_s:float
    requires_durable:bool
    compute_mcycles_assumption:float
    memory_mb_assumption:float
    network_required:bool
    restore_semantics:str
    failure_conditions:tuple[str,...]
    rights_status:str='ORIGINAL_CONSTRUCTED_FIXTURE_LOCAL_ENGINEERING_ONLY'
    owner_scientific_approval:bool=False
    def __post_init__(self):
        if self.adapter not in {'program','cached_lesson','document','restore','sync','unsupported'}:raise ContractError('Unknown reference adapter')
        if self.owner_scientific_approval is not False:raise ContractError('No implicit task approval')
        for k in ['deadline_s','quality_minimum','freshness_limit_s','compute_mcycles_assumption','memory_mb_assumption']:number(getattr(self,k),k)
        if self.quality_minimum>1:raise ContractError('Quality domain')
        if any(type(x) is not bool for x in [self.requires_durable,self.network_required]):raise ContractError('Reference flags')
        if type(self.failure_conditions) is not tuple:raise ContractError('Immutable failure conditions')
    def requirement(self):return TaskRequirement(self.task_id,0,self.deadline_s,self.freshness_limit_s,self.quality_minimum,self.requires_durable)


def evaluate(task:ReferenceTask,result:ReferenceResult):
    if result.task_id!=task.task_id or result.evidence_class!='CONSTRUCTED_REFERENCE_ADAPTER_CHECK':raise ContractError('Reference identity/class')
    number(result.virtual_finished_s,'virtual time');number(result.generated_s,'generated time')
    if result.generated_s>result.virtual_finished_s:raise ContractError('Future content')
    # Fixed expected result is external to the implementation and policy score.
    correct=result.failure is None and result.output_utf8==task.expected_output_utf8
    durable=result.persisted_utf8==result.output_utf8
    o=TaskOutcome(result.task_id,result.virtual_finished_s,result.virtual_finished_s-result.generated_s,result.quality,correct,durable)
    return {'success':completion_fraction([task.requirement()],[o])==1.,'outcome':asdict(o),'observed_output_sha256':bytes_hash(result.output_utf8.encode()),'expected_output_sha256':bytes_hash(task.expected_output_utf8.encode()),'evidence_class':result.evidence_class}

def run_reference(task:ReferenceTask,root:Path, *, content_utf8=TEXT,expected_content_sha256=None,virtual_now_s=1.,generated_s=0.,corrupt_checkpoint=False,network_up=False):
    """Fixed small adapters in a caller-owned empty temporary folder, never arbitrary programs."""
    root=Path(root)
    if not root.is_dir() or any(root.iterdir()):raise ContractError('Empty isolated work directory required')
    number(virtual_now_s,'virtual now');number(generated_s,'generated')
    if type(network_up) is not bool:raise ContractError('Network flag')
    output='';persisted=None;failure=None
    if task.adapter=='program':
        p=root/'reference.py';p.write_text(PROGRAM)
        try:r=subprocess.run([sys.executable,'-I',str(p)],cwd=root,capture_output=True,text=True,timeout=2.)
        except subprocess.TimeoutExpired:failure='engineering_process_timeout'
        else:
            output=r.stdout
            if r.returncode:failure='reference_process_error'
    elif task.adapter=='cached_lesson':
        expected_content_sha256=expected_content_sha256 or bytes_hash(TEXT.encode())
        atomic_write(root/'lesson.txt',content_utf8.encode());data=(root/'lesson.txt').read_bytes()
        if bytes_hash(data)!=expected_content_sha256:failure='corrupt_content'
        output=data.decode();atomic_write(root/'progress.json',canonical({'lesson_sha256':bytes_hash(data),'completed_step':1}))
        persisted=output if json.loads((root/'progress.json').read_text())['completed_step']==1 else None
    elif task.adapter=='document':
        atomic_write(root/'note.txt',b'Draft note\n');atomic_write(root/'note.txt',NOTE.encode());output=NOTE;persisted=(root/'note.txt').read_text()
    elif task.adapter=='restore':
        state={'version':'constructed-note-0.1','text':'Draft note\n','progress':1};wrapped={'state':state,'sha256':digest(state)};atomic_write(root/'checkpoint.json',canonical(wrapped))
        if corrupt_checkpoint:
            wrapped['state']['text']='corrupted';atomic_write(root/'checkpoint.json',canonical(wrapped))
        loaded=json.loads((root/'checkpoint.json').read_text())
        if digest(loaded['state'])!=loaded['sha256']:failure='checkpoint_integrity'
        else:
            # New read from disk, not reuse of volatile state.
            output=loaded['state']['text']+'Edited locally.\n';atomic_write(root/'restored.txt',output.encode());persisted=(root/'restored.txt').read_text()
    elif task.adapter=='sync':
        # Original idempotent reference transport; no production sync service or private records.
        ledger={};state={};events=[{'id':'m1','key':'note','value':'offline edit'},{'id':'m1','key':'note','value':'offline edit'}]
        atomic_write(root/'outbox.json',canonical(events))
        if not network_up:failure='network_required'
        else:
            for ev in json.loads((root/'outbox.json').read_text()):
                if ev['id'] not in ledger:state[ev['key']]=ev['value'];ledger[ev['id']]=True
            output=json.dumps({'applied':len(ledger),'note':state['note']},sort_keys=True,separators=(',',':'));atomic_write(root/'receipt.json',output.encode());persisted=(root/'receipt.json').read_text()
    else:failure='unsupported_workload'
    return ReferenceResult(task.task_id,output,persisted,virtual_now_s,generated_s,1.,failure)

def catalog():
    common={'eligibility':'Original nonpersonal constructed input only','required_input':'Bundled authored fixture','deadline_s':5.,'quality_minimum':1.,'freshness_limit_s':30.,'compute_mcycles_assumption':5.,'memory_mb_assumption':8.,'network_required':False,'restore_semantics':'Explicit byte-integrity check; timing/demand remain synthetic assumptions','failure_conditions':('incorrect','late','stale','corrupt','nondurable','unsupported')}
    tasks=[ReferenceTask('known-program','program',expected_output_utf8='55\n',requires_durable=False,**common),ReferenceTask('cached-lesson','cached_lesson',expected_output_utf8=TEXT,requires_durable=True,**common),ReferenceTask('local-document','document',expected_output_utf8=NOTE,requires_durable=True,**common),ReferenceTask('restore-note','restore',expected_output_utf8=NOTE,requires_durable=True,**common),ReferenceTask('idempotent-sync','sync',expected_output_utf8='{"applied":1,"note":"offline edit"}',requires_durable=True,**{**common,'network_required':True}),ReferenceTask('unsupported-server','unsupported',expected_output_utf8='server-only completion',requires_durable=False,**common)]
    return {'catalog_version':'p6-reference-tasks-0.1','evidence_class':'ORIGINAL_CONSTRUCTED_REFERENCE_WORKLOADS','owner_scientific_approval':False,'tasks':[{**asdict(t),'failure_conditions':list(t.failure_conditions)} for t in tasks]}

def catalog_tasks():return tuple(ReferenceTask(**{**x,'failure_conditions':tuple(x['failure_conditions'])}) for x in catalog()['tasks'])
