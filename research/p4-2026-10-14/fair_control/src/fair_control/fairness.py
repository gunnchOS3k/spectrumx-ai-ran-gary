"""Executable matched-opportunity contract; constructed checks, no campaign loop."""
from dataclasses import asdict,dataclass
from .core import ContractError,digest
from .controllers import Policy,Hardware,validate_pair,decide
from .dynamics import Model

@dataclass(frozen=True)
class Opportunity:
    hardware:Hardware
    model:Model
    max_evaluations:int=512
    deadline_s:float=.1
    def identity(self):return digest(asdict(self))

def validate_opportunities(opportunities,policies,views):
    if len(opportunities)!=2 or opportunities[0].identity()!=opportunities[1].identity():raise ContractError('Unequal resource/cost/compute declarations')
    validate_pair(policies,views)
    for o,p in zip(opportunities,policies):
        if o.max_evaluations!=p.max_evaluations or o.deadline_s!=p.deadline_s:raise ContractError('Controller budget differs from common opportunity')

def matched_decisions(policies,views,opportunity):
    validate_opportunities([opportunity,opportunity],policies,views)
    return tuple(decide(p,x,opportunity.hardware,opportunity.model) for p,x in zip(policies,views))
