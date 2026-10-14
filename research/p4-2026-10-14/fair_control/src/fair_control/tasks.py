"""Constructed task semantics, independent byte-outcome evaluator, lifecycle."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
from .core import ContractError, TaskRequirement, TaskOutcome, completion_fraction, number, digest

class State(str, Enum):
    CREATED='CREATED'; QUEUED='QUEUED'; READY='READY'; RUNNING='RUNNING'
    SUSPENDED='SUSPENDED'; BLOCKED='BLOCKED'; FAILED='FAILED'; COMPLETED='COMPLETED'
TERMINAL={State.FAILED, State.COMPLETED}
TRANSITIONS={
 State.CREATED:{State.QUEUED,State.FAILED},
 State.QUEUED:{State.READY,State.BLOCKED,State.FAILED},
 State.READY:{State.RUNNING,State.BLOCKED,State.FAILED},
 State.RUNNING:{State.SUSPENDED,State.BLOCKED,State.COMPLETED,State.FAILED},
 State.SUSPENDED:{State.READY,State.BLOCKED,State.FAILED},
 State.BLOCKED:{State.READY,State.SUSPENDED,State.FAILED},
 State.FAILED:set(), State.COMPLETED:set()}

def sha(text:str)->str: return hashlib.sha256(text.encode()).hexdigest()

@dataclass(frozen=True)
class Content:
    key:str
    value:str
    generated_s:float
    def __post_init__(self):
        if not self.key or not isinstance(self.value,str):raise ContractError('Invalid content')
        number(self.generated_s,'generated_s')

@dataclass(frozen=True)
class TaskSpec:
    requirement:TaskRequirement
    work_mcycles:float
    memory_mb:float
    payload:str
    expected_sha256:str
    operation:str='copy'
    content_keys:tuple[str,...]=()
    input_mbit:float=0.
    network_required:bool=False
    local_eligible:bool=True
    checkpointable:bool=True
    def __post_init__(self):
        if not isinstance(self.requirement,TaskRequirement):raise ContractError('Task requirement')
        number(self.work_mcycles,'work_mcycles',positive=True);number(self.memory_mb,'memory_mb',positive=True)
        number(self.input_mbit,'input_mbit')
        if type(self.content_keys) is not tuple or any(not isinstance(k,str) or not k for k in self.content_keys) or len(set(self.content_keys))!=len(self.content_keys):raise ContractError('Content dependencies')
        if not isinstance(self.payload,str) or len(self.expected_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.expected_sha256):raise ContractError('Expected digest required independently')
        if any(type(x) is not bool for x in [self.network_required,self.local_eligible,self.checkpointable]):raise ContractError('Boolean task flags')
    @property
    def task_id(self):return self.requirement.task_id

@dataclass(frozen=True)
class Observed:
    task_id:str
    finished_s:float
    output:str
    content_age_s:float
    quality:float
    durable:bool
    def __post_init__(self):
        if not isinstance(self.output,str):raise ContractError('Observed bytes required as UTF-8 text')
        TaskOutcome(self.task_id,self.finished_s,self.content_age_s,self.quality,True,self.durable)

class Evaluator:
    """No policy predictions or connectivity flags enter the success definition."""
    def __init__(self,specs):
        self.specs=tuple(specs)
        if not self.specs or len({s.task_id for s in self.specs})!=len(self.specs):raise ContractError('Unique exogenous task population')
    def outcomes(self,observed):
        ids=[o.task_id for o in observed];known={s.task_id:s for s in self.specs}
        if len(ids)!=len(set(ids)) or not set(ids)<=set(known):raise ContractError('Duplicate/unknown completion')
        return tuple(TaskOutcome(o.task_id,o.finished_s,o.content_age_s,o.quality,sha(o.output)==known[o.task_id].expected_sha256,o.durable) for o in observed)
    def fraction(self,observed):return completion_fraction([s.requirement for s in self.specs],self.outcomes(observed))
    def accepts(self,o):
        s=next(s for s in self.specs if s.task_id==o.task_id)
        return Evaluator([s]).fraction([o])==1.

@dataclass
class TaskRuntime:
    state:State=State.CREATED
    progress_mcycles:float=0.
    delivered_mbit:float=0.
    latency_paid_s:float=0.
    placement:str|None=None
    network:str|None=None
    checkpoint:dict|None=None
    failure:str|None=None
    history:list=field(default_factory=list)
    def move(self,new:State,at:float,reason:str):
        if new not in TRANSITIONS[self.state]:raise ContractError(f'Invalid task transition {self.state}->{new}')
        number(at,'transition timestamp')
        if self.history and at<self.history[-1]['at_s']:raise ContractError('Noncausal transition')
        self.history.append({'from':self.state.value,'to':new.value,'at_s':at,'reason':reason});self.state=new
    def fail(self,at,reason):
        if self.state not in TERMINAL:self.move(State.FAILED,at,reason);self.failure=reason

def execute_payload(spec:TaskSpec,payload:str|None=None)->str:
    # Synthetic deterministic backend; reference digests belong to independently authored specs.
    payload=spec.payload if payload is None else payload
    if spec.operation=='copy':return payload
    if spec.operation=='reverse':return payload[::-1]
    if spec.operation=='upper':return payload.upper()
    raise ContractError('Unsupported workload')
