"""Common truth, observation, execution constraints and accounting contracts.

All bundled data are constructed engineering fixtures. Python observation masks
are cooperation/audit contracts, not a security sandbox against hostile code.
No stochastic generator, reserved holdout, task emulator or campaign is provided.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections.abc import Mapping, Iterator, Callable, Sequence
from types import MappingProxyType
from typing import Any
import math, time, json, hashlib

class ContractError(ValueError): pass
class ObservationDenied(ContractError): pass
class ComputeExceeded(ContractError): pass

def number(x: float, name: str, *, positive: bool=False) -> float:
    if isinstance(x, bool) or not isinstance(x,(int,float)) or not math.isfinite(x):
        raise ContractError(f'{name}: finite number required')
    if x < 0 or (positive and x == 0): raise ContractError(f'{name}: nonnegative/positive domain')
    return float(x)

def freeze(x: Any) -> Any:
    if isinstance(x,Mapping): return MappingProxyType({k:freeze(v) for k,v in x.items()})
    if isinstance(x,(list,tuple)): return tuple(freeze(v) for v in x)
    if isinstance(x,float) and not math.isfinite(x): raise ContractError('Nonfinite observation')
    if x is None or isinstance(x,(str,bool,int,float)):return x
    raise ContractError('Only JSON-compatible observation values permitted')

def canonical(x: Any) -> bytes:
    return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def digest(x: Any) -> str: return hashlib.sha256(canonical(x)).hexdigest()

@dataclass(frozen=True)
class PathTruth:
    name: str
    up: bool
    capacity_mbps: float
    gateway_up: bool=True
    terminal_power: bool=True
    visible: bool=True
    def __post_init__(self):
        number(self.capacity_mbps,'capacity_mbps')
        if not isinstance(self.name,str) or not self.name: raise ContractError('Empty path name')
        if any(type(v) is not bool for v in (self.up,self.gateway_up,self.terminal_power,self.visible)):
            raise ContractError('Availability flags must be Boolean')
    @property
    def usable(self): return self.up and self.gateway_up and self.terminal_power and self.visible

@dataclass(frozen=True)
class Resources:
    bandwidth_mbps: float
    cpu_mcycles_s: float
    memory_mb: float
    power_w: float
    energy_j: float
    def __post_init__(self):
        for n,v in asdict(self).items(): number(v,n)

@dataclass(frozen=True)
class TruthFrame:
    frame_id: str
    task_ids: tuple[str,...]
    paths: tuple[PathTruth,...]
    placement_cpu: tuple[tuple[str,float],...]
    resources: Resources
    duration_s: float
    def __post_init__(self):
        if type(self.task_ids) is not tuple or type(self.paths) is not tuple or type(self.placement_cpu) is not tuple:
            raise ContractError('Immutable tuples required')
        if not self.frame_id or not self.task_ids or len(set(self.task_ids))!=len(self.task_ids):
            raise ContractError('Unique nonempty frame/task identifiers required')
        if not isinstance(self.resources,Resources) or any(not isinstance(p,PathTruth) for p in self.paths): raise ContractError('Invalid truth component')
        if any(type(p) is not tuple or len(p)!=2 for p in self.placement_cpu):raise ContractError('Immutable placement pairs required')
        if len({p.name for p in self.paths})!=len(self.paths): raise ContractError('Duplicate paths')
        if len(dict(self.placement_cpu))!=len(self.placement_cpu): raise ContractError('Duplicate placement')
        for p,n in self.placement_cpu:
            if p not in {'local','edge','cloud'}: raise ContractError('Unknown placement')
            number(n,'placement_cpu')
        number(self.duration_s,'duration_s',positive=True)

class Observation(Mapping):
    """Only supplied estimate fields; never retains the TruthFrame or future truth."""
    def __init__(self, estimates: Mapping, allowed: Sequence[str]):
        fields=tuple(allowed)
        if len(set(fields))!=len(fields) or any(k not in estimates for k in fields):
            raise ContractError('Duplicate/missing declared observation')
        self._values=freeze({k:estimates[k] for k in fields});self._reads=[]
    def __getitem__(self,key):
        if key not in self._values: raise ObservationDenied(str(key))
        self._reads.append(key);return self._values[key]
    def __iter__(self)->Iterator: return iter(self._values)
    def __len__(self): return len(self._values)
    @property
    def consumed_fields(self): return tuple(self._reads)

@dataclass(frozen=True)
class Allocation:
    task_id: str
    bandwidth_mbps: float
    cpu_mcycles_s: float
    memory_mb: float
    power_w: float

@dataclass(frozen=True)
class Action:
    network: str|None
    placement: str
    allocations: tuple[Allocation,...]=()
    fidelity: str='target'
    @classmethod
    def defer(cls): return cls(None,'defer')

@dataclass(frozen=True)
class TransitionCost:
    stall_s: float=0.
    energy_j: float=0.
    def __post_init__(self):
        number(self.stall_s,'stall_s');number(self.energy_j,'transition_energy_j')

def transition(previous: Action|None, action: Action, *, network: TransitionCost, placement: TransitionCost) -> TransitionCost:
    if previous is None: return TransitionCost()
    dn=previous.network!=action.network;dp=previous.placement!=action.placement
    return TransitionCost(network.stall_s*dn+placement.stall_s*dp,network.energy_j*dn+placement.energy_j*dp)

def validate_action(truth: TruthFrame, action: Action, *, cost: TransitionCost=TransitionCost()) -> tuple[str,...]:
    """Final execution-side check; do not expose this truth oracle to candidate selection."""
    errors=[]
    if not isinstance(action,Action) or type(action.allocations) is not tuple: return ('invalid_action_contract',)
    if any(not isinstance(a,Allocation) for a in action.allocations):return ('invalid_allocation_contract',)
    if action.placement=='defer':
        return () if action.network is None and not action.allocations and action.fidelity=='target' else ('defer_must_use_no_resource',)
    if action.fidelity not in {'target','degraded'}: errors.append('unknown_fidelity')
    placements=dict(truth.placement_cpu)
    if action.placement not in placements: errors.append('unknown_placement')
    path=next((p for p in truth.paths if p.name==action.network),None)
    if action.placement!='local' and (path is None or not path.usable): errors.append('unavailable_path')
    if action.placement=='local' and action.network is not None: errors.append('local_has_no_network')
    ids=[a.task_id for a in action.allocations]
    if len(ids)!=len(set(ids)): errors.append('duplicate_task')
    if not set(ids)<=set(truth.task_ids): errors.append('unknown_task')
    if not ids: errors.append('empty_execution')
    totals={k:0. for k in ['bandwidth_mbps','cpu_mcycles_s','memory_mb','power_w']}
    for a in action.allocations:
        for k in totals:
            try: totals[k]+=number(getattr(a,k),k)
            except ContractError: errors.append('invalid_'+k)
    # No tolerance forgiveness: a caller must project/quantize before committing.
    for k,total in totals.items():
        if total>getattr(truth.resources,k): errors.append('over_'+k)
    if totals['cpu_mcycles_s']>placements.get(action.placement,0): errors.append('placement_cpu_exceeded')
    if path is not None and totals['bandwidth_mbps']>path.capacity_mbps: errors.append('path_capacity_exceeded')
    if action.placement=='local' and totals['bandwidth_mbps']!=0: errors.append('local_bandwidth_nonzero')
    if totals['power_w']*truth.duration_s+cost.energy_j>truth.resources.energy_j: errors.append('energy_exceeded')
    if cost.stall_s>=truth.duration_s: errors.append('transition_consumes_interval')
    return tuple(errors)

class ComputeLedger:
    def __init__(self,max_evaluations:int,deadline_s:float,clock:Callable[[],float]=time.perf_counter):
        if type(max_evaluations) is not int or max_evaluations<1: raise ContractError('Invalid evaluation limit')
        number(deadline_s,'deadline_s',positive=True)
        self.maximum=max_evaluations;self.deadline_s=deadline_s;self.clock=clock;self.started=clock();self.evaluations=0;self.exceeded=False
    def check(self):
        elapsed=self.clock()-self.started
        if not math.isfinite(elapsed) or elapsed<0: raise ContractError('Nonmonotonic/invalid compute clock')
        if self.evaluations>self.maximum or elapsed>self.deadline_s:
            self.exceeded=True;raise ComputeExceeded('decision budget exceeded')
    def charge(self):
        self.check()
        if self.evaluations>=self.maximum:
            self.exceeded=True;raise ComputeExceeded('model evaluation limit')
        self.evaluations+=1

def choose(candidates:Sequence[Action], score:Callable[[Action],float], ledger:ComputeLedger) -> Action:
    """Deterministic current-stage selection hook. Not a completed MPC/controller campaign."""
    best=Action.defer();value=-math.inf
    try:
        for a in candidates:
            ledger.charge();s=score(a);ledger.check()
            if isinstance(s,bool) or not math.isfinite(s): raise ContractError('Invalid predicted objective')
            if s>value: best,value=a,s # stable supplied catalog tie order
    except ComputeExceeded:
        return Action.defer() # same timeout/defer semantics for every controller
    return best

@dataclass(frozen=True)
class TaskRequirement:
    task_id: str
    released_s: float
    deadline_s: float
    max_age_s: float
    minimum_quality: float
    requires_durable: bool=True
    def __post_init__(self):
        for k in ['released_s','deadline_s','max_age_s','minimum_quality']: number(getattr(self,k),k)
        if self.deadline_s<self.released_s or self.minimum_quality>1: raise ContractError('Invalid task requirement')

@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    finished_s: float
    content_age_s: float
    quality: float
    correct: bool
    durable: bool
    def __post_init__(self):
        for k in ['finished_s','content_age_s','quality']: number(getattr(self,k),k)
        if self.quality>1 or type(self.correct) is not bool or type(self.durable) is not bool: raise ContractError('Invalid outcome')

def completion_fraction(requirements:Sequence[TaskRequirement],outcomes:Sequence[TaskOutcome])->float:
    """Independent outcome adapter contract; rejected/missing tasks remain failures."""
    ids=[r.task_id for r in requirements];oids=[o.task_id for o in outcomes]
    if not ids or len(ids)!=len(set(ids)) or len(oids)!=len(set(oids)) or not set(oids)<=set(ids):
        raise ContractError('Unknown/duplicate/empty task population')
    byid={o.task_id:o for o in outcomes};completed=0
    for r in requirements:
        o=byid.get(r.task_id)
        completed+=bool(o and o.correct and (o.durable or not r.requires_durable)
                        and r.released_s<=o.finished_s<=r.deadline_s and o.content_age_s<=r.max_age_s
                        and o.quality>=r.minimum_quality)
    return completed/len(ids)
