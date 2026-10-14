"""Causal, uncalibrated fluid task model. SI-derived units; no measured hardware."""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import copy, json
from .core import (Action,Allocation,TruthFrame,Resources,TransitionCost,ContractError,
                   transition,validate_action,number,digest,canonical)
from .tasks import State,TERMINAL,TRANSITIONS,TaskSpec,TaskRuntime,Content,Observed,Evaluator,execute_payload

@dataclass(frozen=True)
class Model:
    schema_version:str='p5-synthetic-0.1'
    mode:str='UNCALIBRATED_SYNTHETIC'
    network_stall_s:float=.05
    network_energy_j:float=.01
    placement_stall_s:float=.1
    placement_energy_j:float=.02
    degraded_quality:float=.5
    def __post_init__(self):
        if self.mode!='UNCALIBRATED_SYNTHETIC' or self.schema_version!='p5-synthetic-0.1':raise ContractError('P5 supports only labeled synthetic model')
        for k,v in asdict(self).items():
            if k not in {'schema_version','mode'}:number(v,k)
        if self.degraded_quality>1:raise ContractError('Quality range')
    def cost(self,previous,action):
        if action.placement=='defer':return TransitionCost()
        return transition(previous,action,network=TransitionCost(self.network_stall_s,self.network_energy_j),placement=TransitionCost(self.placement_stall_s,self.placement_energy_j))

class Environment:
    def __init__(self,specs,energy_j,contents=(),model=Model()):
        self.specs=tuple(specs);self.evaluator=Evaluator(self.specs);self.model=model
        self.tasks={s.task_id:TaskRuntime() for s in self.specs};contents=tuple(contents);self.contents={c.key:c for c in contents}
        if len(self.contents)!=len(tuple(contents)):raise ContractError('Duplicate content')
        self.now_s=0.;self.initial_energy_j=number(energy_j,'energy');self.energy_j=self.initial_energy_j
        self.previous=None;self.observed=[];self.attempts=[];self.trace=[]
        self._release()
    def _release(self):
        for s in self.specs:
            t=self.tasks[s.task_id];r=s.requirement
            if t.state==State.CREATED and r.released_s<=self.now_s:
                t.move(State.QUEUED,self.now_s,'arrival')
                t.move(State.READY,self.now_s,'queued')
            if t.state not in TERMINAL and r.deadline_s<self.now_s:t.fail(self.now_s,'deadline')
    @property
    def queue_length(self):return sum(t.state not in TERMINAL|{State.CREATED} for t in self.tasks.values())
    def snapshot(self):
        return {'version':'p5-state-0.1','specs_hash':digest([asdict(s) for s in self.specs]),'model':asdict(self.model),'now_s':self.now_s,'energy_j':self.energy_j,'initial_energy_j':self.initial_energy_j,'previous':asdict(self.previous) if self.previous else None,'tasks':{k:asdict(v) for k,v in self.tasks.items()},'contents':[asdict(v) for v in self.contents.values()],'observed':[asdict(o) for o in self.observed],'attempts':[asdict(o) for o in self.attempts],'trace':copy.deepcopy(self.trace)}
    @classmethod
    def restart(cls,specs,payload):
        canonical(payload)
        if set(payload)!= {'version','specs_hash','model','now_s','energy_j','initial_energy_j','previous','tasks','contents','observed','attempts','trace'} or payload['version']!='p5-state-0.1':raise ContractError('State version/fields')
        if payload['specs_hash']!=digest([asdict(s) for s in specs]):raise ContractError('State task contract mismatch')
        e=cls(specs,payload['initial_energy_j'],[Content(**c) for c in payload['contents']],Model(**payload['model']))
        e.now_s=number(payload['now_s'],'now_s');e.energy_j=number(payload['energy_j'],'energy_j')
        if e.energy_j>e.initial_energy_j:raise ContractError('Energy creation')
        if set(payload['tasks'])!=set(e.tasks):raise ContractError('State population mismatch')
        for k,v in payload['tasks'].items():
            v=copy.deepcopy(v);v['state']=State(v['state']);e.tasks[k]=TaskRuntime(**v)
        a=payload['previous'];e.previous=Action(a['network'],a['placement'],tuple(Allocation(**x) for x in a['allocations']),a['fidelity']) if a else None
        e.observed=[Observed(**x) for x in payload['observed']];e.attempts=[Observed(**x) for x in payload['attempts']];e.trace=copy.deepcopy(payload['trace']);e.assert_invariants();return e
    def assert_invariants(self):
        number(self.now_s,'now');number(self.energy_j,'energy')
        if self.energy_j>self.initial_energy_j:raise ContractError('Energy creation')
        ids=[o.task_id for o in self.observed]
        if len(ids)!=len(set(ids)):raise ContractError('Duplicate completion')
        self.evaluator.outcomes(self.observed);self.evaluator.outcomes(self.attempts)
        for s in self.specs:
            t=self.tasks[s.task_id]
            if not 0<=t.progress_mcycles<=s.work_mcycles or not 0<=t.delivered_mbit<=s.input_mbit:raise ContractError('Invalid progress')
            number(t.latency_paid_s,'latency')
            if t.placement not in {None,'local','edge','cloud'}:raise ContractError('Invalid execution placement')
            if t.network is not None and not isinstance(t.network,str):raise ContractError('Invalid execution network')
            if (t.state==State.COMPLETED)!=(s.task_id in ids):raise ContractError('Completion-state mismatch')
            if t.state==State.COMPLETED and not self.evaluator.accepts(next(o for o in self.observed if o.task_id==s.task_id)):raise ContractError('Invalid successful completion')
            state=State.CREATED;last=-1.
            for h in t.history:
                if h['from']!=state.value or State(h['to']) not in TRANSITIONS[state] or not last<=h['at_s']<=self.now_s:raise ContractError('Invalid lifecycle history')
                state=State(h['to']);last=h['at_s']
            if state!=t.state:raise ContractError('State/history mismatch')
            if t.checkpoint:self._verify_checkpoint(s.task_id,t.checkpoint)
        if self.queue_length<0:raise ContractError('Negative queue')
        spent=0.;last_time=0.
        for row in self.trace:
            for k in ('energy_j','cpu_mcycles','delivered_mbit','stall_s','maintenance_duration_s'):
                if k in row:number(row[k],k)
            if not last_time<=row['at_s']<=self.now_s:raise ContractError('Invalid trace time')
            last_time=row['at_s'];spent+=row['energy_j']
        # Floating addition order may differ; this is a ledger identity check, not budget forgiveness.
        import math
        if not math.isclose(spent+self.energy_j,self.initial_energy_j,rel_tol=1e-12,abs_tol=1e-12):raise ContractError('Energy ledger mismatch')
    def _age(self,s,at):
        return max([0.]+[at-self.contents[k].generated_s for k in s.content_keys])
    def _ready(self,s,t,at):
        if any(k not in self.contents or self.contents[k].generated_s>at for k in s.content_keys):return 'missing_content'
        if self._age(s,at)>s.requirement.max_age_s:return 'stale_content'
        if s.operation not in {'copy','reverse','upper'}:return 'unsupported_workload'
        return None
    def step(self,truth:TruthFrame,action:Action,latency_s:dict[str,float]):
        """One externally supplied current frame only; arrivals split at release boundaries by caller."""
        if not isinstance(truth,TruthFrame) or not isinstance(action,Action):raise ContractError('Typed truth/action required')
        self._release();end=self.now_s+truth.duration_s
        if any(self.now_s<s.requirement.released_s<end for s in self.specs):raise ContractError('Split frame at task arrival')
        if set(truth.task_ids)!=set(self.tasks):raise ContractError('Truth task population mismatch')
        if set(latency_s)!={p.name for p in truth.paths}:raise ContractError('Latency path contract')
        for v in latency_s.values():number(v,'latency_s')
        cost=self.model.cost(self.previous,action)
        frame=replace(truth,resources=replace(truth.resources,energy_j=min(truth.resources.energy_j,self.energy_j)))
        errors=list(validate_action(frame,action,cost=cost))
        for a in action.allocations:
            s=next((s for s in self.specs if s.task_id==a.task_id),None)
            if s is None:continue
            t=self.tasks[s.task_id]
            if t.state in TERMINAL or t.state==State.CREATED:errors.append('inactive_task')
            if a.memory_mb<s.memory_mb:errors.append('task_memory_requirement')
            if action.placement=='local' and (not s.local_eligible or s.network_required or s.input_mbit>t.delivered_mbit):errors.append('local_ineligible')
        accepted=not errors
        used_cpu=used_mbit=used_energy=stall=0.
        if accepted and action.placement!='defer':
            stall=cost.stall_s;used_energy=cost.energy_j+sum(a.power_w for a in action.allocations)*truth.duration_s
            self.energy_j-=used_energy;self.previous=action
            selected={a.task_id for a in action.allocations}
            for k,t in self.tasks.items():
                if t.state==State.RUNNING and k not in selected:t.move(State.SUSPENDED,self.now_s,'not_selected')
            for a in action.allocations:
                s=next(s for s in self.specs if s.task_id==a.task_id);t=self.tasks[s.task_id]
                reason=self._ready(s,t,self.now_s)
                if reason:
                    if reason=='missing_content':
                        if t.state!=State.BLOCKED:t.move(State.BLOCKED,self.now_s,reason)
                    else:t.fail(self.now_s,reason)
                    continue
                if t.state in {State.BLOCKED,State.SUSPENDED}:t.move(State.READY,self.now_s,'requirements_available')
                t.move(State.RUNNING,self.now_s,'allocation') if t.state==State.READY else None
                # Migration discards volatile progress; a durable checkpoint requires explicit timed restore.
                if t.placement is not None and t.placement!=action.placement:
                    t.progress_mcycles=0.;t.delivered_mbit=0.;t.latency_paid_s=0.
                if t.network!=action.network:t.latency_paid_s=0.
                t.placement=action.placement;t.network=action.network
                available=max(0.,min(end,s.requirement.deadline_s)-self.now_s-stall)
                latency=latency_s.get(action.network,0.) if action.placement!='local' else 0.
                unpaid=max(0.,latency-t.latency_paid_s);paid=min(available,unpaid);t.latency_paid_s+=paid;available-=paid
                missing=max(0.,s.input_mbit-t.delivered_mbit)
                delivery=min(missing,a.bandwidth_mbps*available)
                delivery_time=delivery/a.bandwidth_mbps if a.bandwidth_mbps else 0.
                t.delivered_mbit+=delivery;used_mbit+=delivery;available-=delivery_time
                if t.delivered_mbit<s.input_mbit:continue
                work=min(s.work_mcycles-t.progress_mcycles,a.cpu_mcycles_s*available);t.progress_mcycles+=work;used_cpu+=work
                if t.progress_mcycles>=s.work_mcycles:
                    finish=self.now_s+stall+paid+delivery_time+(work/a.cpu_mcycles_s if a.cpu_mcycles_s else 0.)
                    o=Observed(s.task_id,finish,execute_payload(s,''.join(self.contents[k].value for k in s.content_keys) if s.content_keys else None),self._age(s,finish),1. if action.fidelity=='target' else self.model.degraded_quality,True)
                    self.attempts.append(o)
                    if self.evaluator.accepts(o):t.move(State.COMPLETED,finish,'verified_completion');self.observed.append(o)
                    else:t.fail(finish,'outcome_rejected')
        else:
            for t in self.tasks.values():
                if t.state==State.RUNNING:t.move(State.SUSPENDED,self.now_s,'rejected_or_deferred')
        self.now_s=end;self._release()
        for s in self.specs:
            t=self.tasks[s.task_id]
            if t.state not in TERMINAL and s.requirement.deadline_s<=self.now_s:t.fail(self.now_s,'deadline')
        row={'at_s':self.now_s,'action':asdict(action),'accepted':accepted,'errors':sorted(set(errors)),'cpu_mcycles':used_cpu,'delivered_mbit':used_mbit,'energy_j':used_energy,'stall_s':stall,'remaining_energy_j':self.energy_j,'queue_length':self.queue_length}
        self.trace.append(row);self.assert_invariants();return row
    def _verify_checkpoint(self,task_id,c):
        if set(c)!= {'task_id','progress_mcycles','delivered_mbit','placement','network','created_s','spec_hash','sha256'}:raise ContractError('Checkpoint fields')
        s=next(s for s in self.specs if s.task_id==task_id)
        if c['task_id']!=task_id or c['spec_hash']!=digest(asdict(s)) or c['sha256']!=digest({k:v for k,v in c.items() if k!='sha256'}):raise ContractError('Invalid checkpoint digest/identity')
        if not 0<=c['progress_mcycles']<=s.work_mcycles or not 0<=c['delivered_mbit']<=s.input_mbit or not 0<=c['created_s']<=self.now_s:raise ContractError('Invalid checkpoint progress/time')
    def _maintenance(self,duration_s,energy_j):
        number(duration_s,'maintenance duration',positive=True);number(energy_j,'maintenance energy')
        if energy_j>self.energy_j:raise ContractError('Maintenance energy exhausted')
        self.energy_j-=energy_j;self.now_s+=duration_s
        for t in self.tasks.values():
            if t.state==State.RUNNING:t.move(State.SUSPENDED,self.now_s,'maintenance_stall')
        self._release()
        self.trace.append({'at_s':self.now_s,'maintenance_duration_s':duration_s,'energy_j':energy_j})
    def checkpoint(self,task_id,duration_s,energy_j):
        s=next(s for s in self.specs if s.task_id==task_id);t=self.tasks[task_id]
        if not s.checkpointable or t.state not in {State.RUNNING,State.SUSPENDED,State.READY}:raise ContractError('Checkpoint not permitted')
        self._maintenance(duration_s,energy_j)
        if t.state in TERMINAL:raise ContractError('Checkpoint missed deadline')
        c={'task_id':task_id,'progress_mcycles':t.progress_mcycles,'delivered_mbit':t.delivered_mbit,'placement':t.placement,'network':t.network,'created_s':self.now_s,'spec_hash':digest(asdict(s))};c['sha256']=digest(c);t.checkpoint=c
        self.assert_invariants();return copy.deepcopy(c)
    def crash(self,task_id):
        t=self.tasks[task_id]
        if t.state in TERMINAL|{State.CREATED}:raise ContractError('Inactive crash')
        if t.state!=State.BLOCKED:t.move(State.BLOCKED,self.now_s,'volatile_state_lost')
        t.progress_mcycles=0.;t.delivered_mbit=0.;t.latency_paid_s=0.;t.placement=None;t.network=None
        self.assert_invariants()
    def restore(self,task_id,duration_s,energy_j):
        t=self.tasks[task_id]
        if t.state!=State.BLOCKED or t.checkpoint is None:raise ContractError('No restorable state')
        self._verify_checkpoint(task_id,t.checkpoint);self._maintenance(duration_s,energy_j)
        if t.state in TERMINAL:raise ContractError('Restore missed deadline')
        c=t.checkpoint;t.progress_mcycles=c['progress_mcycles'];t.delivered_mbit=c['delivered_mbit'];t.placement=c['placement'];t.network=c['network'];t.latency_paid_s=0.
        t.move(State.READY,self.now_s,'checkpoint_restored');self.assert_invariants()
