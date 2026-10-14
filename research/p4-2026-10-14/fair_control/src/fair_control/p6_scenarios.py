"""Causal Markov/common-dependency and congestion interfaces.

No RNG, future stream, campaign or realistic outage calibration. Tests supply
explicit uniform variates/current arrivals. Distribution parameters are assumed.
"""
from dataclasses import dataclass,asdict
import math,copy
from .core import ContractError,number,digest,PathTruth,Resources,TruthFrame

@dataclass(frozen=True)
class Gate:
    name:str
    fail_rate_s:float
    restore_rate_s:float
    def __post_init__(self):
        if not isinstance(self.name,str) or not self.name:raise ContractError('Gate name')
        number(self.fail_rate_s,'failure rate /s');number(self.restore_rate_s,'restoration rate /s')

@dataclass(frozen=True)
class Link:
    name:str
    capacity_mbps:float
    latency_s:float
    dependencies:tuple[str,...]
    def __post_init__(self):
        if not self.name or type(self.dependencies) is not tuple or len(set(self.dependencies))!=len(self.dependencies):raise ContractError('Link dependencies')
        number(self.capacity_mbps,'Mbps');number(self.latency_s,'latency s')

class NetworkProcess:
    def __init__(self,gates,links,duration_s=1.):
        self.gates=tuple(gates);self.links=tuple(links);self.duration_s=number(duration_s,'duration s',positive=True)
        if len({g.name for g in self.gates})!=len(self.gates) or len({l.name for l in self.links})!=len(self.links):raise ContractError('Duplicate component')
        known={g.name for g in self.gates}
        if any(not set(l.dependencies)<=known for l in self.links):raise ContractError('Unknown dependency')
        self.up={g.name:True for g in self.gates};self.now_s=0.;self.history=[]
    def advance(self,uniforms,background_mbps=None):
        if set(uniforms)!=set(self.up):raise ContractError('Exactly one current variate per gate')
        for u in uniforms.values():
            number(u,'uniform');
            if u>=1:raise ContractError('Uniform in [0,1)')
        background_mbps={} if background_mbps is None else background_mbps
        if not set(background_mbps)<={l.name for l in self.links}:raise ContractError('Unknown background path')
        for x in background_mbps.values():number(x,'background Mbps')
        for g in self.gates:
            rate=g.fail_rate_s if self.up[g.name] else g.restore_rate_s
            p=-math.expm1(-rate*self.duration_s)
            if uniforms[g.name]<p:self.up[g.name]=not self.up[g.name]
        paths=[]
        for l in self.links:
            up=all(self.up[g] for g in l.dependencies);cap=max(0.,l.capacity_mbps-background_mbps.get(l.name,0.))
            paths.append({'name':l.name,'up':up,'capacity_mbps':cap,'latency_s':l.latency_s})
        row={'timestamp_s':self.now_s,'duration_s':self.duration_s,'paths':paths,'component_up':dict(self.up),'evidence_class':'SYNTHETIC_CURRENT_FRAME'}
        self.history.append(copy.deepcopy(row));self.now_s+=self.duration_s;return row
    def snapshot(self):return {'version':'p6-network-state-0.1','definition_sha256':digest({'gates':[asdict(x) for x in self.gates],'links':[asdict(x) for x in self.links],'duration_s':self.duration_s}),'up':dict(self.up),'now_s':self.now_s,'history':copy.deepcopy(self.history)}
    def restore(self,x):
        if x['version']!='p6-network-state-0.1' or x['definition_sha256']!=self.snapshot()['definition_sha256'] or set(x['up'])!=set(self.up):raise ContractError('Network restart identity')
        if any(type(v) is not bool for v in x['up'].values()):raise ContractError('Restart gate flags')
        number(x['now_s'],'restart time');self.up=dict(x['up']);self.now_s=x['now_s'];self.history=copy.deepcopy(x['history'])

@dataclass
class FluidQueue:
    backlog_mbit:float=0.
    def advance(self,arrivals_mbit,capacity_mbps,duration_s):
        number(self.backlog_mbit,'backlog');number(arrivals_mbit,'Mbit');number(capacity_mbps,'Mbps');number(duration_s,'s',positive=True)
        available=self.backlog_mbit+arrivals_mbit;served=min(available,capacity_mbps*duration_s);self.backlog_mbit=available-served
        return {'served_mbit':served,'backlog_mbit':self.backlog_mbit,'queue_delay_s':self.backlog_mbit/capacity_mbps if capacity_mbps else None}

def poisson_count(rate_tasks_s,duration_s,u,cap=1024):
    """Inverse Poisson CDF with supplied current uniform; no random seed selection."""
    number(rate_tasks_s,'tasks/s');number(duration_s,'s',positive=True);number(u,'u')
    if u>=1 or type(cap) is not int or not 1<=cap<=1024:raise ContractError('Arrival bounds')
    mean=rate_tasks_s*duration_s
    if mean>50:raise ContractError('Bounded constructed arrival mean <=50')
    probability=math.exp(-mean);total=probability;k=0
    while u>=total and k<cap:
        k+=1;probability*=mean/k;total+=probability
    if k>=cap:raise ContractError('Arrival cap exceeded')
    return k

def burst_count(count,max_tasks=64):
    if type(count) is not int or type(max_tasks) is not int or not 0<=count<=max_tasks<=64:raise ContractError('Constructed burst cap')
    return count

class CausalEstimator:
    """Only delivered history; persistence prediction, never future truth."""
    def __init__(self,delay_s=0.,capacity_relative_error=0.,latency_bias_s=0.,flip_up=False):
        number(delay_s,'observation delay')
        if not isinstance(capacity_relative_error,(float,int)) or isinstance(capacity_relative_error,bool) or not math.isfinite(capacity_relative_error) or not -1<=capacity_relative_error<=1:raise ContractError('Capacity error [-1,1]')
        if not isinstance(latency_bias_s,(float,int)) or isinstance(latency_bias_s,bool) or not math.isfinite(latency_bias_s):raise ContractError('Finite latency error')
        if type(flip_up) is not bool:raise ContractError('Availability-error flag')
        self.delay_s=delay_s;self.capacity_relative_error=capacity_relative_error;self.latency_bias_s=latency_bias_s;self.flip_up=flip_up;self._delivered=[]
    def ingest(self,current_frame,delivered_at_s):
        number(delivered_at_s,'delivery time')
        f=copy.deepcopy(current_frame)
        if set(f)!= {'timestamp_s','duration_s','paths','component_up','evidence_class'} or f['evidence_class']!='SYNTHETIC_CURRENT_FRAME':raise ContractError('Only current synthetic frame')
        number(f['timestamp_s'],'frame time');number(f['duration_s'],'frame duration',positive=True)
        if f['timestamp_s']>delivered_at_s or (self._delivered and f['timestamp_s']<=self._delivered[-1]['frame']['timestamp_s']):raise ContractError('Future or reordered estimator input')
        for p in f['paths']:
            if set(p)!= {'name','up','capacity_mbps','latency_s'} or type(p['up']) is not bool:raise ContractError('Estimate path fields')
            number(p['capacity_mbps'],'capacity');number(p['latency_s'],'latency')
        self._delivered.append({'frame':f,'delivered_at_s':delivered_at_s})
    def observe(self,now_s,horizon=3,interval_s=1.):
        number(now_s,'observation time');number(interval_s,'interval',positive=True)
        if type(horizon) is not int or not 1<=horizon<=3:raise ContractError('Horizon bound')
        eligible=[x['frame'] for x in self._delivered if x['delivered_at_s']<=now_s and x['frame']['timestamp_s']<=now_s-self.delay_s]
        if not eligible:raise ContractError('No causal observation available; defer')
        f=eligible[-1];paths=copy.deepcopy(f['paths'])
        for p in paths:
            p['up']=not p['up'] if self.flip_up else p['up'];p['capacity_mbps']*=1+self.capacity_relative_error;p['latency_s']=max(0.,p['latency_s']+self.latency_bias_s)
        forecasts=[{'timestamp_s':now_s+i*interval_s,'duration_s':interval_s,'paths':copy.deepcopy(paths)} for i in range(1,horizon)]
        return {'timestamp_s':now_s,'estimated_paths':paths,'forecasts':forecasts,'provenance':{'estimator':'causal-persistence-0.1','source_timestamp_s':f['timestamp_s'],'observation_age_s':now_s-f['timestamp_s'],'delay_s':self.delay_s,'future_truth_access':False,'errors_are_synthetic':True}}
    def snapshot(self):return {'version':'p6-estimator-state-0.1','parameters':{'delay_s':self.delay_s,'capacity_relative_error':self.capacity_relative_error,'latency_bias_s':self.latency_bias_s,'flip_up':self.flip_up},'delivered':copy.deepcopy(self._delivered)}
    @classmethod
    def restart(cls,payload):
        if payload['version']!='p6-estimator-state-0.1':raise ContractError('Estimator state version')
        e=cls(**payload['parameters'])
        for x in payload['delivered']:e.ingest(x['frame'],x['delivered_at_s'])
        return e
