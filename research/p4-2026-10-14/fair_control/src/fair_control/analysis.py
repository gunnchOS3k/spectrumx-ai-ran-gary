"""Statistical scaffolding. Tests use constructed arrays, not research outcomes."""
from __future__ import annotations
import math,statistics
from scipy.stats import t,norm
from .core import ContractError,number

def paired_summary(left:dict[str,float],right:dict[str,float])->dict:
    if set(left)!=set(right) or len(left)<2: raise ContractError('Matched independent unit IDs, n>=2 required')
    ids=sorted(left);d=[]
    for k in ids:
        a=number(left[k],'fraction');b=number(right[k],'fraction')
        if max(a,b)>1: raise ContractError('Fractions must be <=1')
        d.append(a-b)
    n=len(d);m=statistics.fmean(d);sd=statistics.stdev(d);half=float(t.ppf(.975,n-1))*sd/math.sqrt(n)
    return {'unit_ids':ids,'n':n,'mean':m,'sample_sd':sd,'ci95':[m-half,m+half],'paired_dz':m/sd if sd else None,'paired_differences':d,'interpretation':'Conditional paired unit means; zero inclusion is inconclusive, not equivalence.'}

def plan_size(sd:float,*,half_width:float=.01,effect_gap:float=.02,power:float=.9,cap:int=2500)->dict:
    number(sd,'pilot_sd');number(half_width,'half_width',positive=True);number(effect_gap,'effect_gap',positive=True)
    if not .5<power<1 or type(cap) is not int or cap<30: raise ContractError('Invalid power/cap')
    # Required size decreases as t degrees of freedom increase. Find the
    # smallest n satisfying both planning conditions,not a conservative first
    # jump based on df29 that can falsely reject a usable compute/sample cap.
    def required(n):
        critical=float(t.ppf(.975,n-1))
        return max(30,math.ceil((critical*sd/half_width)**2),math.ceil(((critical+float(norm.ppf(power)))*sd/effect_gap)**2))
    lo=30;hi=required(lo)
    while lo<hi:
        mid=(lo+hi)//2
        if mid>=required(mid):hi=mid
        else:lo=mid+1
    n=lo
    return {'n':n,'feasible_within_cap':n<=cap,'cap':cap,'pilot_sd_assumption':sd,'half_width_target':half_width,'effect_gap':effect_gap,'power_normal_approximation':power,'heldout_may_start':False,'scope':'Smallest n satisfying the stated t-precision/normal-power planning approximation; not an exact power guarantee. Independent pilot,human freeze and authorization still required.'}
