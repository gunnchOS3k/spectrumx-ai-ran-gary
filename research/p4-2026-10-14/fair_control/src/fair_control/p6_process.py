"""Measured accounting of engineering controller IPC/initialization overhead.

Child direct-process timeout only, not hostile sandbox/real-time certification.
No physical calibration. Same startup and decision budgets for eligible policies.
"""
from pathlib import Path
from dataclasses import asdict
import time,subprocess,sys,select,json
from .core import ContractError,number,Action,digest,canonical
from .controllers import Policy,Hardware,Model,Decision,action_from_dict,validate_estimates

ROOT=Path(__file__).resolve().parents[2]
def _line(proc,limit_s):
    if limit_s<=0:return None
    ready,_,_=select.select([proc.stdout],[],[],limit_s)
    return proc.stdout.readline() if ready else None

def controlled_decision(policy,estimates,hardware=Hardware(),model=Model(),startup_allowance_s=1.,worker=None):
    number(startup_allowance_s,'startup cap',positive=True)
    start=time.perf_counter();validate_estimates(estimates)
    payload=canonical({'policy':asdict(policy),'estimates':estimates,'hardware':asdict(hardware),'model':asdict(model)})
    if len(payload)>65536:raise ContractError('Bounded process request')
    preparation=time.perf_counter()-start;spawn_start=time.perf_counter();proc=None;d=None;error=None;status='PROCESS_ERROR';child_time=None;startup=None;decision_wall=preparation
    try:
        proc=subprocess.Popen([sys.executable,str(worker or ROOT/'p6_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        ready=_line(proc,startup_allowance_s-(time.perf_counter()-spawn_start));startup=time.perf_counter()-spawn_start
        if ready!='READY\n' or startup>startup_allowance_s:status='STARTUP_TIMEOUT_OR_ERROR';error='Missing/late READY handshake'
        else:
            dispatched=time.perf_counter();proc.stdin.write(payload.decode()+'\n');proc.stdin.flush()
            line=_line(proc,policy.deadline_s-preparation)
            decision_wall=preparation+time.perf_counter()-dispatched
            if line is None or decision_wall>policy.deadline_s:status='DECISION_TIMEOUT';error='End-to-end decision budget exceeded'
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
    report={'process_status':status,'request_preparation_s':preparation,'startup_s':startup,'worker_decision_s':child_time,'decision_window_s':decision_wall,'outer_elapsed_s':outer,'startup_allowance_s':startup_allowance_s,'decision_allowance_s':policy.deadline_s,'simulated_compute_delay_debited':False,'physical_calibration':False,'direct_child_reaped':True}
    return d,report
