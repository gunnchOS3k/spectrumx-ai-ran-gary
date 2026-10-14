"""Unit/provenance validation for later calibration. No dataset is acquired here."""
from dataclasses import dataclass,asdict
import math, hashlib
from .core import ContractError,number,digest
UNITS={'network.capacity':'Mbps','network.latency':'s','compute.capacity':'Mcycles/s','task.work':'Mcycles','memory.capacity':'MB','power':'W','energy':'J','switch.duration':'s','switch.energy':'J'}
STATUSES={'MEASURED','LITERATURE_DERIVED','STANDARDS_REQUIREMENT','SYNTHETIC_ASSUMPTION','UNCALIBRATED'}

@dataclass(frozen=True)
class Calibration:
    parameter:str
    units:str
    status:str
    source:str
    source_sha256:str|None
    license_notice:str
    method:str
    uncertainty:str
    applicability:str
    limitations:str
    value:float|None=None
    def __post_init__(self):
        if self.parameter not in UNITS or self.units!=UNITS[self.parameter] or self.status not in STATUSES:raise ContractError('Calibration parameter/unit/status')
        if not all(isinstance(x,str) and x for x in [self.source,self.license_notice,self.method,self.uncertainty,self.applicability,self.limitations]):raise ContractError('Incomplete calibration provenance')
        if self.source_sha256 is not None and (len(self.source_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.source_sha256)):raise ContractError('Source digest')
        if self.value is not None:number(self.value,'calibration value')
        if self.status=='MEASURED' and (self.source_sha256 is None or self.value is None or self.source.startswith('constructed:')):raise ContractError('Measured requires actual source/number; constructed references are synthetic')

def reference_summary(record:Calibration,samples,source_bytes:bytes):
    """Data adapter checks identity/units; method/status must be supplied honestly."""
    if hashlib.sha256(source_bytes).hexdigest()!=record.source_sha256:raise ContractError('Calibration source hash mismatch')
    if not samples:raise ContractError('Empty reference data')
    vals=[]
    for row in samples:
        if set(row)!= {'value','units'} or row['units']!=record.units:raise ContractError('Reference unit mismatch')
        vals.append(number(row['value'],'reference sample'))
    mean=sum(vals)/len(vals);sd=math.sqrt(sum((v-mean)**2 for v in vals)/(len(vals)-1)) if len(vals)>1 else None
    return {'evidence_class':'CALIBRATION_ADAPTER_CHECK','status':record.status,'parameter':record.parameter,'units':record.units,'n':len(vals),'mean':mean,'sample_sd':sd,'source_sha256':record.source_sha256,'record_sha256':digest(asdict(record)),'uncertainty_limit':'Sample SD is descriptive; does not supply calibration validity or field applicability.'}

def power_duration_energy(power_w,duration_s):return number(power_w,'power W')*number(duration_s,'duration s')
