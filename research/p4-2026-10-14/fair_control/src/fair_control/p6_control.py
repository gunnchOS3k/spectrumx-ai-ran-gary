"""P6 fixed-target primary variant using unchanged P5 model/catalog/predictor.

This is a recommended prospective amendment, not an approved protocol. No
fidelity/resource saving is inferred. Legacy P5 behavior remains byte-intact.
"""
from dataclasses import asdict
import math,time
from .core import Observation,ComputeLedger,ComputeExceeded,Action,ContractError,digest
from .controllers import (Policy,Hardware,Decision,FIELDS,thaw,validate_estimates,
                          catalog,predict,action_from_dict)
from .dynamics import Model

FIXED='FIXED_TARGET'
TRADEOFF='PROSPECTIVE_FIDELITY_RESOURCE_TRADEOFF'
def validate_fidelity(convention,efficiency_claim=False):
    if type(efficiency_claim) is not bool or efficiency_claim:raise ContractError('No verified fidelity/resource saving; efficiency claims unsupported')
    if convention!=FIXED:raise ContractError('Tradeoff option is prospective only; no justified resource-saving model')
    return True

def primary_catalog(x,hardware,model):return catalog(x,hardware,fixed_target=True,model=model)

def decide_primary(policy,estimates,hardware=Hardware(),model=Model(),clock=time.perf_counter):
    if policy.family not in {'mpc','hysteresis'}:raise ContractError('P6 primary families only')
    validate_estimates(estimates);o=Observation(estimates,policy.observations)
    ledger=ComputeLedger(policy.max_evaluations,policy.deadline_s,clock);action=Action.defer();status='SOLVED';score=None;error=None;options=()
    try:
        x={k:thaw(o[k]) for k in policy.observations};ledger.check();options=primary_catalog(x,hardware,model);ledger.check()
        if policy.family=='mpc':
            beam=[(0.,x,None,())]
            for depth in range(policy.horizon):
                expanded=[]
                for value,state,first,path in beam:
                    if depth:
                        if depth-1>=len(x['forecasts']):continue
                        f=x['forecasts'][depth-1]
                        state={**state,'estimated_paths':f['paths'],'timestamp_s':f['timestamp_s']};dt=f['duration_s']
                    else:dt=1.
                    for i,a in enumerate(primary_catalog(state,hardware,model)):
                        ledger.charge();v,nxt=predict(state,a,hardware,model,dt);ledger.check()
                        expanded.append((value+v,nxt,a if first is None else first,path+(i,)))
                if not expanded:break
                expanded.sort(key=lambda row:(-row[0],row[3]));beam=expanded[:policy.beam_width];score,state,action,path=beam[0]
            status='BOUNDED_SEARCH_COMPLETE'
        else:
            evaluated=[]
            for a in options:
                ledger.charge();v,_=predict(x,a,hardware,model);ledger.check();evaluated.append((v,a))
            score,action=max(evaluated,key=lambda row:row[0]);previous=action_from_dict(x['previous_action'])
            same=[(v,a) for v,a in evaluated if previous and (a.network,a.placement,a.fidelity)==(previous.network,previous.placement,previous.fidelity)]
            if same:
                v,a=max(same,key=lambda row:row[0])
                if score-v<=policy.hysteresis:score,action=v,a
        ledger.check()
    except ComputeExceeded as ex:status='BUDGET_EXCEEDED';action=Action.defer();error=str(ex);score=None
    except (ContractError,ArithmeticError) as ex:status='MODEL_FAILURE';action=Action.defer();error=str(ex);score=None
    elapsed=clock()-ledger.started
    if not math.isfinite(elapsed) or elapsed<0:raise ContractError('Invalid compute clock')
    return Decision(action,status,ledger.evaluations,o.consumed_fields,elapsed,digest(estimates),digest([asdict(a) for a in options]),score,error)
