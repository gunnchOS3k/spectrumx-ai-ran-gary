"""Compute semantics are explicit; host time is never silently network latency."""
from .core import ContractError,number
SYNTHETIC='SYNTHETIC_COMPUTE_BUDGET'
DIAGNOSTIC='HOST_TIMING_DIAGNOSTIC'
COUPLED='PROPOSED_TIME_COUPLED_MODEL'

def convention(mode):
    if mode not in {SYNTHETIC,DIAGNOSTIC,COUPLED}:raise ContractError('Unknown compute accounting convention')
    return {'mode':mode,'model_calls_and_wall_budget_enforced':True,'host_timing_retained':True,'virtual_service_debit_s':0. if mode!=COUPLED else None,'host_is_radio_latency':False,'operational':mode!=COUPLED,'scientific_approval':False,'meaning':('Budget enforcement with separate host overhead diagnostics;no host-to-virtual calibration' if mode!=COUPLED else 'Unimplemented scientific time-coupling proposal;needs model,units,calibration/justification and approval')}

def require_operational(mode):
    x=convention(mode)
    if not x['operational']:raise ContractError('Proposed time-coupled model is not operational/approved')
    return x
