"""Dimension/uncertainty-aware import contract. No real data acquired in P6."""
from dataclasses import dataclass,asdict
import hashlib,statistics,math
from .core import ContractError,number,digest
UNITS={'duration':'s','memory':'MB','work':'Mcycles','arrival_rate':'tasks/s','capacity':'Mbps','latency':'s','switch':'s','checkpoint':'s','recovery':'s','power':'W','energy':'J','deadline':'s'}
STATUSES={'MEASURED','LITERATURE_DERIVED','STANDARDS_BASED_REQUIREMENT','SYNTHETIC_ASSUMPTION','UNCALIBRATED'}
CONVERSIONS={'ms':('s',.001),'s':('s',1.),'bytes':('MB',1e-6),'MiB':('MB',1.048576),'MB':('MB',1.),'bit/s':('Mbps',1e-6),'Mbps':('Mbps',1.),'cycles':('Mcycles',1e-6),'Mcycles':('Mcycles',1.),'W':('W',1.),'J':('J',1.),'tasks/s':('tasks/s',1.)}
@dataclass(frozen=True)
class CalibrationRecord:
    parameter:str
    status:str
    units:str
    source_uri:str
    source_sha256:str|None
    method:str
    uncertainty:str
    applicability:str
    restrictions:str
    license_notice:str
    rights_review_status:str
    imported:bool=False
    value:float|None=None
    def __post_init__(self):
        if self.parameter not in UNITS or self.units!=UNITS[self.parameter] or self.status not in STATUSES:raise ContractError('Calibration class/dimension')
        for k in ['source_uri','method','uncertainty','applicability','restrictions','license_notice','rights_review_status']:
            if not isinstance(getattr(self,k),str) or not getattr(self,k):raise ContractError('Calibration metadata incomplete')
        if type(self.imported) is not bool:raise ContractError('Import flag')
        if self.value is not None:number(self.value,'calibration value')
        if self.source_sha256 is not None and (len(self.source_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.source_sha256)):raise ContractError('Calibration source identity')
        if self.imported and self.rights_review_status!='LOCAL_USE_VERIFIED':raise ContractError('No unreviewed dataset import')
        if self.status=='MEASURED' and (not self.imported or self.source_sha256 is None or self.value is None or self.source_uri.startswith('constructed:')):raise ContractError('Actual measured data/provenance required')

def convert(value,units,target):
    number(value,'input')
    if units not in CONVERSIONS or CONVERSIONS[units][0]!=target:raise ContractError('Incompatible units; no RTT-to-one-way assumption')
    return value*CONVERSIONS[units][1]

def inspect_reference(record,samples,source_bytes):
    if record.source_sha256!=hashlib.sha256(source_bytes).hexdigest():raise ContractError('Raw source hash mismatch')
    if record.rights_review_status!='LOCAL_USE_VERIFIED':raise ContractError('Reference source local use not verified')
    values=[]
    for x in samples:
        if set(x)!= {'value','units','metric_definition'} or x['metric_definition']!=record.parameter:raise ContractError('Ambiguous metric')
        values.append(convert(x['value'],x['units'],record.units))
    if not values:raise ContractError('Empty samples')
    return {'status':record.status,'source_sha256':record.source_sha256,'record_sha256':digest(asdict(record)),'units':record.units,'n':len(values),'mean':statistics.fmean(values),'sample_sd':statistics.stdev(values) if len(values)>1 else None,'uncertainty_metadata':record.uncertainty,'physical_calibration_established':False,'evidence_class':'CALIBRATION_ADAPTER_CHECK_NOT_FIELD_VALIDATION'}

def azure_duration_adapter(rows):
    """Documented Average(ms)/Count aggregates, not per-invocation percentiles.

Only constructed rows tested in P6; callers must independently settle source
hash/rights using CalibrationRecord before importing real dataset bytes.
"""
    if not rows:raise ContractError('Empty Azure duration table')
    numerator=0.;n=0
    for r in rows:
        if set(r)!= {'Average','Count'} or type(r['Count']) is not int or r['Count']<1:raise ContractError('Aggregate duration schema')
        duration=convert(r['Average'],'ms','s');numerator+=duration*r['Count'];n+=r['Count']
    return {'weighted_mean_duration_s':numerator/n,'reported_execution_count':n,'cold_start_included':False,'raw_invocation_variance_available':False,'restriction':'Aggregates cannot be treated as independent invocation observations or transferred to device Mcycles without calibration.'}

def one_way_latency_adapter(rows):
    vals=[]
    for r in rows:
        if set(r)!= {'value','units','semantics'} or r['semantics']!='ONE_WAY_END_TO_END':raise ContractError('RTT/propagation cannot silently become one-way end-to-end latency')
        vals.append(convert(r['value'],r['units'],'s'))
    if not vals:raise ContractError('Empty latency rows')
    return vals
