"""Measured accounting of engineering controller IPC/initialization overhead.

Child direct-process timeout only, not hostile sandbox/real-time certification.
No physical calibration. Same startup and decision budgets for eligible policies.
"""
from pathlib import Path
from dataclasses import asdict
import time,subprocess,sys,select,json,resource
from .core import ContractError,number,Action,digest,canonical
from .controllers import Policy,Hardware,Model,Decision,action_from_dict,validate_estimates

ROOT=Path(__file__).resolve().parents[2]
def _line(proc,limit_s):
    if limit_s<=0:return None
    ready,_,_=select.select([proc.stdout],[],[],limit_s)
    return proc.stdout.readline() if ready else None

def controlled_decision(policy,estimates,hardware=Hardware(),model=Model(),startup_allowance_s=1.,worker=None,accounting_mode='SYNTHETIC_COMPUTE_BUDGET'):
    from .p6_accounting import require_operational
    accounting=require_operational(accounting_mode)
    number(startup_allowance_s,'startup cap',positive=True)
    start=time.perf_counter();parent_cpu_start=time.process_time();usage_start=resource.getrusage(resource.RUSAGE_CHILDREN);validate_estimates(estimates)
    payload=canonical({'policy':asdict(policy),'estimates':estimates,'hardware':asdict(hardware),'model':asdict(model)})
    if len(payload)>65536:raise ContractError('Bounded process request')
    preparation=time.perf_counter()-start;spawn_start=time.perf_counter();proc=None;d=None;error=None;status='PROCESS_ERROR';child_time=None;startup=None;decision_wall=preparation
    try:
        proc=subprocess.Popen([sys.executable,str(worker or ROOT/'p6_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        ready=_line(proc,startup_allowance_s-(time.perf_counter()-spawn_start));startup=time.perf_counter()-spawn_start
        if ready!='READY\n' or startup>startup_allowance_s:status='STARTUP_TIMEOUT_OR_ERROR';error='Missing/late READY handshake'
        else:
            dispatched=time.perf_counter();failed_exit=False
            try:
                remaining=policy.deadline_s-preparation
                if remaining<=0:raise subprocess.TimeoutExpired(proc.args,remaining)
                out,err=proc.communicate(input=payload.decode()+'\n',timeout=remaining)
                failed_exit=proc.returncode!=0;line=out if not failed_exit else None
            except subprocess.TimeoutExpired:line=None
            decision_wall=preparation+time.perf_counter()-dispatched
            if failed_exit:status='PROCESS_ERROR';error=err[:2000]
            elif line is None or decision_wall>policy.deadline_s:status='DECISION_TIMEOUT';error='End-to-end decision budget exceeded'
            else:
                result=json.loads(line);child_time=number(result['worker_decision_s'],'worker time');x=result['decision']
                x['action']=action_from_dict(x['action']);x['consumed_fields']=tuple(x['consumed_fields']);d=Decision(**x)
                if d.model_evaluations>policy.max_evaluations or child_time>policy.deadline_s:status='DECISION_TIMEOUT';d=None;error='Invalid child accounting'
                else:status='RETURNED'
    except (ValueError,KeyError,OSError,ContractError) as ex:error=str(ex);status='PROCESS_ERROR';d=None
    finally:
        if proc is not None:
            if proc.poll() is None:proc.kill()
            proc.wait()
            for f in [proc.stdin,proc.stdout,proc.stderr]:
                if f:f.close()
    outer=time.perf_counter()-start
    if d is None:d=Decision(Action.defer(),status,0,(),decision_wall,digest(estimates),digest([]),None,error)
    usage_end=resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu=(usage_end.ru_utime+usage_end.ru_stime)-(usage_start.ru_utime+usage_start.ru_stime)
    report={'accounting_convention':accounting,'parent_cpu_s':time.process_time()-parent_cpu_start,'child_cpu_s':child_cpu,'cpu_accounting_scope':'Sequential Unix RUSAGE_CHILDREN delta; overlapping unrelated children invalidate attribution','process_status':status,'request_preparation_s':preparation,'startup_s':startup,'worker_decision_s':child_time,'decision_window_s':decision_wall,'outer_elapsed_s':outer,'startup_allowance_s':startup_allowance_s,'decision_allowance_s':policy.deadline_s,'simulated_compute_delay_debited':False,'physical_calibration':False,'direct_child_reaped':True,'actual_model_evaluations':d.model_evaluations if status=='RETURNED' else None,'evaluation_count_status':'RETURNED_LEDGER' if status=='RETURNED' else 'UNAVAILABLE_ON_KILLED_OR_FAILED_CHILD;zero_in_defer_is_placeholder' }
    return d,report
