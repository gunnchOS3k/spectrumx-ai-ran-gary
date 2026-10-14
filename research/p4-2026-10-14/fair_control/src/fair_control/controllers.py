"""Estimate-only finite catalog policies; bounded beam MPC and matched hysteresis.

API boundary, not a sandbox against malicious Python reflection. No future truth
or historical outcome file is accepted. Scores are predicted task counts, never
an outcome oracle. Every predictor invocation is charged before execution.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math, time, copy
from .core import (TruthFrame,PathTruth,Resources,validate_action,Observation,ObservationDenied,Action,Allocation,ComputeLedger,ComputeExceeded,
                   ContractError,TransitionCost,freeze,digest,number)
from .dynamics import Model
FIELDS=('timestamp_s','estimated_paths','task_queue','remaining_energy_j','previous_action','forecasts')

@dataclass(frozen=True)
class Hardware:
    cpu_mcycles_s:float=10.
    memory_mb:float=100.
    bandwidth_mbps:float=10.
    power_w:float=2.
    placement_cpu:tuple= (('local',5.),('edge',10.),('cloud',10.))
    def __post_init__(self):
        for k in ['cpu_mcycles_s','memory_mb','bandwidth_mbps','power_w']:number(getattr(self,k),k,positive=True)
        if type(self.placement_cpu) is not tuple or len(dict(self.placement_cpu))!=len(self.placement_cpu):raise ContractError('Placement contract')
        for k,v in self.placement_cpu:
            if k not in {'local','edge','cloud'}:raise ContractError('Placement')
            number(v,'placement CPU')

@dataclass(frozen=True)
class Policy:
    family:str
    observations:tuple=FIELDS
    max_evaluations:int=512
    deadline_s:float=.1
    horizon:int=3
    beam_width:int=4
    hysteresis:float=.05 # provisional prediction-task-count margin, not measured/approved
    def __post_init__(self):
        if self.family not in {'fallback','static','rule','greedy','hysteresis','mpc','fixed_target'}:raise ContractError('Unknown family')
        if type(self.observations) is not tuple or tuple(self.observations)!=FIELDS:raise ContractError('Common declared fields required')
        if type(self.max_evaluations) is not int or not 1<=self.max_evaluations<=512:raise ContractError('Evaluation limit')
        number(self.deadline_s,'wall deadline',positive=True);number(self.hysteresis,'hysteresis')
        if not self.deadline_s<=.1 or type(self.horizon) is not int or not 1<=self.horizon<=3 or type(self.beam_width) is not int or not 1<=self.beam_width<=4:raise ContractError('P4 planning bounds')

def action_from_dict(a):return Action(a['network'],a['placement'],tuple(Allocation(**x) for x in a['allocations']),a['fidelity']) if a else None

def thaw(v):
    if hasattr(v,'items'):return {k:thaw(x) for k,x in v.items()}
    if isinstance(v,tuple):return [thaw(x) for x in v]
    return v

def validate_estimates(x):
    if set(x)!=set(FIELDS):raise ObservationDenied('Undeclared estimate/future-truth field')
    number(x['timestamp_s'],'timestamp');number(x['remaining_energy_j'],'battery')
    tasks=x['task_queue'];ids=[]
    if len(tasks)>64 or len(x['estimated_paths'])>4:raise ContractError('Bounded engineering input size')
    for t in tasks:
        if set(t)!= {'task_id','remaining_mcycles','remaining_mbit','memory_mb','deadline_s','minimum_quality','content_age_s','max_age_s','local_eligible','network_required','work_mcycles','input_mbit','placement','network','latency_paid_s','content_available','supported'}:raise ContractError('Task estimate fields')
        if not isinstance(t['task_id'],str) or not t['task_id']:raise ContractError('Task ID')
        ids.append(t['task_id'])
        for k in ['remaining_mcycles','remaining_mbit','memory_mb','deadline_s','minimum_quality','content_age_s','max_age_s','work_mcycles','input_mbit','latency_paid_s']:number(t[k],k)
        if t['minimum_quality']>1 or type(t['local_eligible']) is not bool or any(type(t[k]) is not bool for k in ('network_required','content_available','supported')):raise ContractError('Task estimate types')
        if t['placement'] not in {None,'local','edge','cloud'} or (t['network'] is not None and not isinstance(t['network'],str)):raise ContractError('Task estimate execution state')
    if len(set(ids))!=len(ids):raise ContractError('Duplicate task estimates')
    frames=[{'timestamp_s':x['timestamp_s'],'duration_s':1.,'paths':x['estimated_paths']}]+list(x['forecasts'])
    if len(x['forecasts'])>2:raise ContractError('Forecast bound H=3')
    prev=-1.
    for f in frames:
        if set(f)!= {'timestamp_s','duration_s','paths'}:raise ObservationDenied('Forecast may contain predictions only')
        number(f['timestamp_s'],'forecast time');number(f['duration_s'],'predicted duration',positive=True)
        if f['timestamp_s']<prev:raise ContractError('Forecast time order')
        prev=f['timestamp_s'];names=[]
        for p in f['paths']:
            if set(p)!= {'name','up','capacity_mbps','latency_s'}:raise ObservationDenied('Path estimate fields')
            if not isinstance(p['name'],str) or not p['name'] or type(p['up']) is not bool:raise ContractError('Path estimate types')
            number(p['capacity_mbps'],'capacity');number(p['latency_s'],'latency');names.append(p['name'])
        if len(set(names))!=len(names):raise ContractError('Duplicate path estimates')
    if x['previous_action']:action_from_dict(x['previous_action'])
    freeze(x)

def catalog(x,hardware:Hardware, *, fixed_target=False,model=Model()):
    """Public hardware + estimates only. No true-feasibility query or hidden label."""
    tasks=list(x['task_queue']);paths=list(x['estimated_paths']);actions=[Action.defer()]
    for place,pcpu in hardware.placement_cpu:
        eligible=[t for t in tasks if t['memory_mb']<=hardware.memory_mb and (place!='local' or (t['local_eligible'] and not t['network_required'] and t['remaining_mbit']==0))]
        eligible.sort(key=lambda t:(t['deadline_s'],t['task_id']))
        selected=[];mem=0.
        for t in eligible:
            if mem+t['memory_mb']<=hardware.memory_mb:selected.append(t);mem+=t['memory_mb']
        if not selected:continue
        path_options=[None] if place=='local' else [p for p in paths if p['up']]
        for path in path_options:
            cap=0. if path is None else min(hardware.bandwidth_mbps,path['capacity_mbps'])
            for profile in ('equal','priority','deadline'):
                weights=[1. if profile=='equal' else (max(t['remaining_mcycles'],1e-9) if profile=='priority' else 1/max(t['deadline_s']-x['timestamp_s'],1e-9)) for t in selected]
                denom=sum(weights);cpu=min(hardware.cpu_mcycles_s,pcpu)
                def shares(total):
                    vals=[total*w/denom for w in weights[:-1]];vals.append(max(0.,total-sum(vals)))
                    while sum(vals)>total:vals[-1]=math.nextafter(vals[-1],0.)
                    return vals
                bs,cs,ps=shares(cap),shares(cpu),shares(hardware.power_w)
                alloc=tuple(Allocation(t['task_id'],b,c,t['memory_mb'],p) for t,b,c,p in zip(selected,bs,cs,ps))
                for fidelity in (('target',) if fixed_target else ('target','degraded')):actions.append(Action(path['name'] if path else None,place,alloc,fidelity))
    # Check public estimate feasibility; this frame contains no environment truth.
    if tasks:
        estimated=TruthFrame('ESTIMATE_ONLY',tuple(t['task_id'] for t in tasks),tuple(PathTruth(p['name'],p['up'],p['capacity_mbps']) for p in paths),hardware.placement_cpu,Resources(hardware.bandwidth_mbps,hardware.cpu_mcycles_s,hardware.memory_mb,hardware.power_w,x['remaining_energy_j']),1.)
        actions=[a for a in actions if not validate_action(estimated,a,cost=model.cost(action_from_dict(x['previous_action']),a))]
    # Stable lexicographic tie order, duplicate profiles collapsed by byte identity.
    unique={digest(asdict(a)):a for a in actions}
    return tuple(sorted(unique.values(),key=lambda a:(a.placement,a.network or '',a.fidelity,digest(asdict(a)))))

def predict(x,a,hardware,model,duration_s=1.):
    """Deterministic fluid forecast using declared estimate state only."""
    y=copy.deepcopy(x);previous=action_from_dict(x['previous_action']);cost=model.cost(previous,a)
    completed=0
    if a.placement=='defer':y['timestamp_s']+=duration_s;return completed,y
    if cost.stall_s>=duration_s or sum(z.power_w for z in a.allocations)*duration_s+cost.energy_j>x['remaining_energy_j']:
        y['timestamp_s']+=duration_s;return 0,y
    paths={p['name']:p for p in x['estimated_paths']}
    if a.placement!='local' and (a.network not in paths or not paths[a.network]['up']):y['timestamp_s']+=duration_s;return 0,y
    quality=1. if a.fidelity=='target' else model.degraded_quality
    remaining=[];byid={z.task_id:z for z in a.allocations}
    for t in y['task_queue']:
        z=byid.get(t['task_id']);success=False
        if z and t['content_available'] and t['supported']:
            if t['placement'] is not None and t['placement']!=a.placement:
                t['remaining_mcycles']=t['work_mcycles'];t['remaining_mbit']=t['input_mbit'];t['latency_paid_s']=0.
            if t['network']!=a.network:t['latency_paid_s']=0.
            t['placement']=a.placement;t['network']=a.network
            dt=max(0.,min(duration_s,t['deadline_s']-x['timestamp_s'])-cost.stall_s)
            latency=paths[a.network]['latency_s'] if a.network else 0.
            paid=min(dt,max(0.,latency-t['latency_paid_s']));t['latency_paid_s']+=paid;dt-=paid
            bits=min(t['remaining_mbit'],z.bandwidth_mbps*dt);t['remaining_mbit']-=bits;transfer_time=bits/z.bandwidth_mbps if z.bandwidth_mbps else 0.;dt-=transfer_time
            work=min(t['remaining_mcycles'],z.cpu_mcycles_s*dt);t['remaining_mcycles']-=work
            finish=x['timestamp_s']+cost.stall_s+paid+transfer_time+(work/z.cpu_mcycles_s if z.cpu_mcycles_s else 0.)
            success=t['remaining_mcycles']==0 and t['remaining_mbit']==0 and quality>=t['minimum_quality'] and t['content_age_s']+(finish-x['timestamp_s'])<=t['max_age_s'] and finish<=t['deadline_s']
        if success:completed+=1
        elif t['deadline_s']>x['timestamp_s']+duration_s:
            t['content_age_s']+=duration_s;remaining.append(t)
    y['task_queue']=remaining;y['timestamp_s']+=duration_s;y['remaining_energy_j']-=sum(z.power_w for z in a.allocations)*duration_s+cost.energy_j;y['previous_action']=asdict(a)
    return completed,y

@dataclass(frozen=True)
class Decision:
    action:Action
    status:str
    model_evaluations:int
    consumed_fields:tuple
    elapsed_s:float
    observation_sha256:str
    catalog_sha256:str
    predicted_completions:float|None
    error:str|None=None

def decide(policy:Policy,estimates,hardware=Hardware(),model=Model(),clock=time.perf_counter)->Decision:
    validate_estimates(estimates);o=Observation(estimates,policy.observations)
    ledger=ComputeLedger(policy.max_evaluations,policy.deadline_s,clock)
    action=Action.defer();status='SOLVED';score=None;error=None;options=()
    try:
        # Read all authorized inputs before catalog construction; report actual repeated reads too.
        x={k:thaw(o[k]) for k in policy.observations};ledger.check()
        options=catalog(x,hardware,fixed_target=policy.family=='fixed_target',model=model);ledger.check()
        if policy.family in {'static','fallback','rule'}:
            executable=[a for a in options if a.placement!='defer']
            if policy.family=='static':preferred=[a for a in executable if a.placement=='edge' and a.network=='TN' and a.fidelity=='target']
            elif policy.family=='fallback':preferred=[a for a in executable if a.placement=='cloud' and a.fidelity=='target']
            else:preferred=[a for a in executable if a.placement=='edge' and a.fidelity=='target']
            # Fixed preference is revalidated against the same current public estimated catalog.
            action=(preferred or executable or [Action.defer()])[0]
        elif policy.family=='mpc':
            beam=[(0.,x,None,())]
            for depth in range(policy.horizon):
                expanded=[]
                for value,state,first,path in beam:
                    if depth:
                        fs=x['forecasts']
                        if depth-1>=len(fs):continue
                        f=fs[depth-1];state=copy.deepcopy(state);state['estimated_paths']=f['paths'];state['timestamp_s']=f['timestamp_s'];dt=f['duration_s']
                    else:dt=1.
                    for i,a in enumerate(catalog(state,hardware,model=model)):
                        ledger.charge();v,next_state=predict(state,a,hardware,model,dt);ledger.check()
                        expanded.append((value+v,next_state,first or a,path+(i,)))
                if not expanded:break
                expanded.sort(key=lambda row:(-row[0],row[3]));beam=expanded[:policy.beam_width]
                score,state,action,path=beam[0]
            status='BOUNDED_SEARCH_COMPLETE' # finite beam approximation, not global optimizer convergence
        else:
            evaluated=[]
            for a in options:
                ledger.charge();v,_=predict(x,a,hardware,model);ledger.check();evaluated.append((v,a))
            score,action=max(evaluated,key=lambda row:row[0]) # stable catalog ties
            if policy.family=='hysteresis':
                previous=action_from_dict(x['previous_action'])
                same=[(v,a) for v,a in evaluated if previous and a.network==previous.network and a.placement==previous.placement and a.fidelity==previous.fidelity]
                if same:
                    v,a=max(same,key=lambda row:row[0])
                    if score-v<=policy.hysteresis:score,action=v,a
        ledger.check()
    except ComputeExceeded as ex:status='BUDGET_EXCEEDED';action=Action.defer();error=str(ex);score=None
    except (ContractError,ArithmeticError) as ex:status='MODEL_FAILURE';action=Action.defer();error=str(ex);score=None
    elapsed=clock()-ledger.started
    if not math.isfinite(elapsed) or elapsed<0:raise ContractError('Invalid elapsed')
    return Decision(action,status,ledger.evaluations,o.consumed_fields,elapsed,digest(estimates),digest([asdict(a) for a in options]),score,error)

def validate_pair(policies,views):
    if len(policies)!=2 or len(views)!=2:raise ContractError('Primary pair required')
    if policies[0].observations!=policies[1].observations or policies[0].max_evaluations!=policies[1].max_evaluations or policies[0].deadline_s!=policies[1].deadline_s:raise ContractError('Observation/compute opportunity mismatch')
    # Dynamic state may differ after causal actions; pairing identity applies to exogenous estimates.
    for x in views:validate_estimates(x)
    for k in ('timestamp_s','estimated_paths','forecasts'):
        if digest(views[0][k])!=digest(views[1][k]):raise ContractError('Unequal exogenous observations')
