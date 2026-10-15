from dataclasses import dataclass,asdict
import hashlib,json,math

class ContractError(ValueError):pass
CATEGORIES=('CONSTRUCTED_ENGINEERING_VERIFICATION','SYNTHETIC_SIMULATION','REAL_MEASUREMENT_REPLAY','HARDWARE_IN_LOOP','LIVE_CONTROLLED_WIRELESS','INDEPENDENT_REPRODUCED_PHYSICAL')
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(b):return hashlib.sha256(b).hexdigest()
def ns(x):
 if type(x) is not int or x<0:raise ContractError('Nonnegative integer monotonic ns required')
 return x

def sensor(value,unit,status='AVAILABLE',reason=None):
 if status not in {'AVAILABLE','UNAVAILABLE','PERMISSION_DENIED','UNSUPPORTED'}:raise ContractError('Sensor status')
 if not isinstance(unit,str) or not unit:raise ContractError('Explicit units')
 if status=='AVAILABLE':
  if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):raise ContractError('Finite sensor value')
  if reason is not None:raise ContractError('Available sensor reason')
 elif value is not None or not reason:raise ContractError('Missing telemetry must be null with reason')
 return {'value':value,'unit':unit,'status':status,'reason':reason}

@dataclass(frozen=True)
class Task:
 task_id:str
 expected_sha256:str
 released_ns:int
 deadline_ns:int
 clock_domain:str
 max_age_ns:int
 durable_required:bool=True
 def __post_init__(self):
  for v in [self.released_ns,self.deadline_ns,self.max_age_ns]:ns(v)
  if not self.task_id or not self.clock_domain or self.deadline_ns<self.released_ns or len(self.expected_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.expected_sha256):raise ContractError('Task identity/clock/deadline/digest')
  if type(self.durable_required) is not bool:raise ContractError('Durability flag')

def evaluate(task,result):
 if result['task_id']!=task.task_id or result['clock_domain']!=task.clock_domain:raise ContractError('Task/clock identity')
 finish=ns(result['finished_ns']);generated=ns(result['generated_ns'])
 if finish<task.released_ns or generated>finish:raise ContractError('Invalid event clock order')
 if type(result['durable_reopened']) is not bool or type(result['success_reported']) is not bool:raise ContractError('Outcome boolean')
 return {'task_id':task.task_id,'success':result['success_reported'] and result['failure'] is None and result['observed_output_sha256']==task.expected_sha256 and finish<=task.deadline_ns and finish-generated<=task.max_age_ns and (result['durable_reopened'] or not task.durable_required),'latency_ns':finish-task.released_ns,'correct':result['observed_output_sha256']==task.expected_sha256,'evidence_category':result['evidence_category'],'integration_class':result['integration_class']}

def all_task_fraction(tasks,results):
 if not tasks or len({t.task_id for t in tasks})!=len(tasks):raise ContractError('Nonempty unique population')
 known={t.task_id:t for t in tasks};ids=[r['task_id'] for r in results]
 if len(ids)!=len(set(ids)) or not set(ids)<=set(known):raise ContractError('Duplicate/unknown outcome')
 return sum(evaluate(known[r['task_id']],r)['success'] for r in results)/len(tasks)

def clock_bounds(t1,t2,t3,t4):
 for x in [t1,t2,t3,t4]:ns(x)
 if t4<t1 or t3<t2:raise ContractError('Within-clock monotonic order')
 low=t3-t4;high=t2-t1
 if low>high:raise ContractError('Incompatible clocks/order or drift')
 return {'server_minus_client_offset_interval_ns':[low,high],'assumptions':'same clock rate over exchange and nonnegative path delays;not validated synchronization','one_way_latency_ns':None,'client_transaction_ns':t4-t1,'server_span_ns':t3-t2}

def claim_allowed(category,radio,claim):
 if category not in CATEGORIES:raise ContractError('Evidence category')
 if claim=='ACTUAL_6G_RADIO_PERFORMANCE':return False # no such evidence in this milestone
 if claim=='LIVE_PHYSICAL_EXPERIMENT':return category in {'HARDWARE_IN_LOOP','LIVE_CONTROLLED_WIRELESS','INDEPENDENT_REPRODUCED_PHYSICAL'} and radio not in {'LOOPBACK','EMULATED','UNKNOWN'}
 if claim=='MEASURED_NTN':return radio in {'AUTHENTIC_SATELLITE','MEASURED_NTN'} and category in {'REAL_MEASUREMENT_REPLAY','HARDWARE_IN_LOOP','LIVE_CONTROLLED_WIRELESS','INDEPENDENT_REPRODUCED_PHYSICAL'}
 return claim=='6G_RESEARCH_RELEVANCE' # documented motivation,not measured capability
