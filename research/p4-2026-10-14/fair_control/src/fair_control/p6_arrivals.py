"""Deterministic per-decision queue window; ALL exogenous tasks stay registered.

The P5 limit64 is an observation-list bound, not an episode/admission limit.
Overflow is unserved queued work, never a removed denominator observation.
"""
import copy
from .core import ContractError,canonical,digest,number
from .controllers import validate_estimates
VALID={'CREATED','QUEUED','READY','RUNNING','SUSPENDED','BLOCKED','FAILED','COMPLETED'}
TERMINAL={'FAILED','COMPLETED'}

class ArrivalLedger:
    def __init__(self,window_size=64):
        if type(window_size) is not int or not 1<=window_size<=64:raise ContractError('Observation window must be1..64')
        self.window_size=window_size;self.rows={};self._population_sealed=False
    def register(self,records):
        if self._population_sealed:raise ContractError('Fixed exogenous population already sealed')
        ids=[r['estimate']['task_id'] for r in records]
        if len(set(ids))!=len(ids) or set(ids)&set(self.rows):raise ContractError('Duplicate exogenous task')
        checked={}
        for r in records:
            if set(r)!= {'estimate','released_s','state'} or r['state'] not in VALID:raise ContractError('Arrival record contract')
            number(r['released_s'],'release')
            if r['estimate']['deadline_s']<r['released_s']:raise ContractError('Deadline before release')
            # Reuse the existing estimate validator on a single row; no hidden truth fields.
            base={'timestamp_s':0.,'estimated_paths':[],'task_queue':[r['estimate']],'remaining_energy_j':0.,'previous_action':None,'forecasts':[]};validate_estimates(base)
            checked[r['estimate']['task_id']]=copy.deepcopy(r)
        self.rows.update(checked)
    def register_burst(self,records):
        # Validate bounded batches transactionally; never truncate the full burst.
        if self._population_sealed:raise ContractError('Population sealed')
        staged=ArrivalLedger(self.window_size)
        from .p6_scenarios import burst_count
        for start in range(0,len(records),64):
            batch=records[start:start+64];burst_count(len(batch));staged.register(batch)
        if set(staged.rows)&set(self.rows):raise ContractError('Duplicate burst population')
        self.rows.update(staged.rows)
    def seal(self):
        if not self.rows:raise ContractError('Nonempty exogenous eligible population required')
        self._population_sealed=True
    def update(self,records):
        ids=[r['estimate']['task_id'] for r in records]
        if len(set(ids))!=len(ids) or not set(ids)<=set(self.rows):raise ContractError('Update identity')
        staged=ArrivalLedger(self.window_size);staged.register(records)
        for k,r in staged.rows.items():
            old=self.rows[k]
            if r['released_s']!=old['released_s'] or r['estimate']['deadline_s']!=old['estimate']['deadline_s']:raise ContractError('Cannot change exogenous task clock')
            if old['state'] in TERMINAL and r!=old:raise ContractError('Cannot revive/rewrite terminal outcome')
        self.rows.update(staged.rows)
    def view(self,base_estimates,now_s):
        if not self._population_sealed:raise ContractError('Seal exogenous workload before controller decisions')
        number(now_s,'now')
        eligible=[r for r in self.rows.values() if r['released_s']<=now_s and r['state'] not in TERMINAL|{'CREATED'}]
        eligible.sort(key=lambda r:(r['estimate']['deadline_s'],r['released_s'],r['estimate']['task_id']))
        visible=eligible[:self.window_size];overflow=eligible[self.window_size:]
        x=copy.deepcopy(base_estimates);x['timestamp_s']=now_s;x['task_queue']=[copy.deepcopy(r['estimate']) for r in visible];validate_estimates(x)
        return x,{'registered_eligible_denominator':len(self.rows),'visible_count':len(visible),'overflow_count':len(overflow),'overflow_ids':[r['estimate']['task_id'] for r in overflow],'ordering':'deadline,release,task_id','no_task_discarded':True}
    def snapshot(self):
        payload={'version':'p6-arrival-ledger-0.1','window_size':self.window_size,'sealed':self._population_sealed,'rows':[copy.deepcopy(self.rows[k]) for k in sorted(self.rows)]}
        return {'payload':payload,'sha256':digest(payload)}
    @classmethod
    def restart(cls,record):
        if set(record)!= {'payload','sha256'} or record['sha256']!=digest(record['payload']):raise ContractError('Arrival checkpoint integrity')
        x=record['payload']
        if x['version']!='p6-arrival-ledger-0.1' or type(x['sealed']) is not bool:raise ContractError('Arrival version/state')
        q=cls(x['window_size']);q.register(x['rows'])
        if x['sealed']:q.seal()
        return q
