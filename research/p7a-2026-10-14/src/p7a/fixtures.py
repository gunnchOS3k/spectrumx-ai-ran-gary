"""Six fixed current-observation engineering inputs, not a scientific sample."""
import copy
from fair_control.controllers import validate_estimates
from fair_control.core import digest

def fixtures():
    rows=[]
    for name,n,online,network in [('empty',0,False,False),('local-1',1,False,False),('local-8',8,False,False),('local-64',64,False,False),('network-8',8,True,True),('network-64',64,True,True)]:
        paths=[{'name':p,'up':online,'capacity_mbps':c,'latency_s':l} for p,c,l in [('TN',10.,.02),('Wi-Fi',5.,.04),('NTN',3.,.2)]]
        tasks=[{'task_id':f'engineering-{i:03d}','remaining_mcycles':5.,'remaining_mbit':1. if network else 0.,'memory_mb':8.,'deadline_s':5.,'minimum_quality':1.,'content_age_s':1.,'max_age_s':30.,'local_eligible':True,'network_required':network,'work_mcycles':5.,'input_mbit':1. if network else 0.,'placement':None,'network':None,'latency_paid_s':0.,'content_available':True,'supported':True} for i in range(n)]
        x={'timestamp_s':1.,'estimated_paths':paths,'task_queue':tasks,'remaining_energy_j':100.,'previous_action':None,'forecasts':[{'timestamp_s':float(t),'duration_s':1.,'paths':copy.deepcopy(paths)} for t in [2,3]]}
        validate_estimates(x)
        rows.append({'id':name,'estimated_input':x,'sha256':digest(x),'evidence_class':'CONSTRUCTED_RUNTIME_INPUT_NOT_PILOT'})
    return rows
