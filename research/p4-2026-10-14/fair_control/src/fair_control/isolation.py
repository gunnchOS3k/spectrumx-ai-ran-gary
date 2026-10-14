"""Optional process envelope. Cooperative policy budget + hard outer timeout.

Startup/process overhead has a separately declared allowance, shared across
policies. This is not a 100ms real-time guarantee and is not a hostile-code
sandbox. Kill/reap happens before returning a timeout/defer decision.
"""
from pathlib import Path
from dataclasses import asdict
import subprocess,sys,json,time
from .core import canonical,ContractError,number,Action,digest
from .controllers import Policy,Hardware,Decision,decide,action_from_dict
from .dynamics import Model

def process_call(request,timeout_s,worker=None):
    number(timeout_s,'outer process limit',positive=True)
    worker=worker or Path(__file__).resolve().parents[2]/'p5_worker.py'
    started=time.perf_counter()
    try:
        p=subprocess.run([sys.executable,str(worker)],input=canonical(request).decode(),capture_output=True,text=True,timeout=timeout_s)
    except subprocess.TimeoutExpired:
        # subprocess.run kills and waits for the direct child on timeout.
        return {'status':'PROCESS_TIMEOUT','decision':None,'outer_elapsed_s':time.perf_counter()-started}
    if p.returncode!=0:return {'status':'PROCESS_ERROR','decision':None,'outer_elapsed_s':time.perf_counter()-started,'error':p.stderr[:2000]}
    try:result=json.loads(p.stdout)
    except (ValueError,TypeError) as ex:return {'status':'PROCESS_ERROR','decision':None,'outer_elapsed_s':time.perf_counter()-started,'error':str(ex)}
    return {'status':'RETURNED','decision':result,'outer_elapsed_s':time.perf_counter()-started}

def decide_isolated(policy,estimates,hardware=Hardware(),model=Model(),startup_allowance_s=1.):
    number(startup_allowance_s,'process allowance',positive=True)
    request={'policy':asdict(policy),'estimates':estimates,'hardware':asdict(hardware),'model':asdict(model)}
    result=process_call(request,startup_allowance_s+policy.deadline_s)
    if result['decision']:
        d=result['decision'];d['action']=action_from_dict(d['action']);d['consumed_fields']=tuple(d['consumed_fields']);decision=Decision(**d)
    else:decision=Decision(Action.defer(),result['status'],0,(),result['outer_elapsed_s'],digest(estimates),digest([]),None,result.get('error','Process envelope expired'))
    return decision,{'outer_elapsed_s':result['outer_elapsed_s'],'startup_allowance_s':startup_allowance_s,'hard_outer_limit_s':startup_allowance_s+policy.deadline_s,'inner_wall_limit_s':policy.deadline_s,'process_status':result['status']}
