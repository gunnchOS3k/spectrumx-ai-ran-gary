"""Bounded experimental reuse of a pure estimate-only engineering worker.

Sequential current input only. Timeout/framing/identity errors kill and reap;
no reuse after failure. Not a campaign runner, hostile sandbox or real-time OS.
"""
from pathlib import Path
from dataclasses import asdict
import json,os,resource,select,subprocess,sys,time
from . import ROOT
from fair_control.core import ContractError,Action,canonical,digest,number
from fair_control.controllers import Policy,Hardware,Decision,action_from_dict,validate_estimates
from fair_control.dynamics import Model

VERSION='p7a-engineering-session-0.1'
MAX_REQUEST=65536
MAX_RESPONSE=262144

def child_cpu():
    r=resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime+r.ru_stime

class Session:
    def __init__(self,policy,hardware=Hardware(),model=Model(),startup_allowance_s=1.,worker=None,max_requests=64):
        if policy.family not in {'mpc','hysteresis'}:raise ContractError('Only matched primary families')
        if type(max_requests) is not int or not 1<=max_requests<=64:raise ContractError('Bounded current-request session')
        number(startup_allowance_s,'startup cap',positive=True)
        self.policy=policy;self.hardware=hardware;self.model=model;self.max_requests=max_requests
        self.next_id=0;self.closed=False;self.proc=None;self.buf=b'';self.summary=None
        self.began=time.perf_counter();self.cpu_began=time.process_time();self.child_began=child_cpu()
        try:
            self.proc=subprocess.Popen([sys.executable,str(worker or ROOT/'p7a_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0)
            os.set_blocking(self.proc.stdin.fileno(),False);os.set_blocking(self.proc.stdout.fileno(),False)
            r=self._read(self.began+startup_allowance_s)
            if r!={'ready':VERSION}:raise ContractError('Worker handshake identity')
            self.startup_s=time.perf_counter()-self.began
            if self.startup_s>startup_allowance_s:raise ContractError('Late startup')
        except Exception:
            self.close();raise
    def _read(self,deadline):
        while b'\n' not in self.buf:
            remain=deadline-time.perf_counter()
            if remain<=0:raise TimeoutError('Response budget exceeded')
            if not select.select([self.proc.stdout],[],[],remain)[0]:raise TimeoutError('Response budget exceeded')
            b=os.read(self.proc.stdout.fileno(),65536)
            if not b:raise ContractError('Worker exited/EOF')
            self.buf+=b
            if len(self.buf)>MAX_RESPONSE:raise ContractError('Response size')
        line,self.buf=self.buf.split(b'\n',1)
        if self.buf:raise ContractError('Unrequested/pipelined output')
        return json.loads(line)
    def _write(self,data,deadline):
        offset=0
        while offset<len(data):
            remain=deadline-time.perf_counter()
            if remain<=0 or not select.select([],[self.proc.stdin],[],max(0.,remain))[1]:raise TimeoutError('Input/IPC budget exceeded')
            offset+=os.write(self.proc.stdin.fileno(),data[offset:])
    def decide(self,estimates):
        if self.closed or self.next_id>=self.max_requests:raise ContractError('Closed/exhausted engineering session; retire before next request')
        begin=time.perf_counter();pcpu=time.process_time();status='RETURNED';error=None;actual=None;worker_wall=None;worker_cpu=None;rss=None
        validate_estimates(estimates)
        x={'version':VERSION,'request_id':self.next_id,'policy':asdict(self.policy),'hardware':asdict(self.hardware),'model':asdict(self.model),'estimates':estimates}
        payload=canonical(x)+b'\n'
        if len(payload)>MAX_REQUEST:raise ContractError('Request size')
        prepared=time.perf_counter()-begin;dispatched=time.perf_counter();deadline=begin+self.policy.deadline_s
        try:
            self._write(payload,deadline);r=self._read(deadline)
            if set(r)!={'version','request_id','input_sha256','decision','worker_wall_s','worker_cpu_s','peak_rss_raw'} or r['version']!=VERSION or r['request_id']!=self.next_id or r['input_sha256']!=digest(estimates):raise ContractError('Response identity/fields')
            x=r['decision'];x['action']=action_from_dict(x['action']);x['consumed_fields']=tuple(x['consumed_fields']);d=Decision(**x)
            worker_wall=number(r['worker_wall_s'],'worker wall');worker_cpu=number(r['worker_cpu_s'],'worker cpu');rss=number(r['peak_rss_raw'],'peak RSS')
            if d.observation_sha256!=digest(estimates) or not 0<=d.model_evaluations<=self.policy.max_evaluations or worker_wall>self.policy.deadline_s or time.perf_counter()>deadline:raise TimeoutError('Decision budget/accounting exceeded')
            if d.action.fidelity!='target':raise ContractError('Fixed-target primary')
            actual=d.model_evaluations
        except Exception as ex:
            status='DECISION_TIMEOUT' if isinstance(ex,TimeoutError) else 'PROCESS_ERROR';error=str(ex);self.close()
            d=Decision(Action.defer(),status,0,(),time.perf_counter()-begin,digest(estimates),digest([]),None,error)
        self.next_id+=1
        return d,{'process_status':status,'request_id':self.next_id-1,'request_preparation_s':prepared,'ipc_and_worker_elapsed_s':time.perf_counter()-dispatched,'decision_window_s':time.perf_counter()-begin,'parent_cpu_s':time.process_time()-pcpu,'worker_decision_s':worker_wall,'worker_cpu_s':worker_cpu,'child_peak_rss_raw':rss,'actual_model_evaluations':actual,'evaluation_count_status':'RETURNED_LEDGER' if actual is not None else 'UNKNOWN_AFTER_FAILURE','equal_max_model_calls':self.policy.max_evaluations,'equal_decision_allowance_s':self.policy.deadline_s,'virtual_service_debit_s':0.,'session_invalidated_on_failure':self.closed,'error':error}
    def close(self):
        if self.closed:return self.summary
        self.closed=True
        if self.proc:
            if self.proc.poll() is None:self.proc.kill()
            self.proc.wait()
            for f in [self.proc.stdin,self.proc.stdout,self.proc.stderr]:
                if f:f.close()
        self.summary={'wall_s':time.perf_counter()-self.began,'parent_cpu_s':time.process_time()-self.cpu_began,'child_cpu_s':child_cpu()-self.child_began,'startup_s':getattr(self,'startup_s',None),'direct_child_reaped':self.proc is None or self.proc.returncode is not None,'cpu_scope':'Sequential child, reaped RUSAGE_CHILDREN delta including startup/IPC/serialization/termination; unrelated overlapping children invalidate attribution'}
        return self.summary
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
