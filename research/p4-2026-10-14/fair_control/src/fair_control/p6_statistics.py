"""Prospective planning assumptions only. No future empirical outcomes read."""
import math
from scipy.stats import t,norm,nct,chi2
from .core import number,ContractError

def planning(sd,margin=.02,alternative=.04,half_width=.01,power=.9,cap=2500):
    number(sd,'hypothetical SD',positive=True);number(margin,'practical margin');number(alternative,'alternative',positive=True);number(half_width,'half width',positive=True)
    if alternative<=margin or not .5<power<1 or type(cap) is not int or not 30<=cap<=10000:raise ContractError('Planning domains')
    gap=alternative-margin
    # Lower end of TWO-SIDED 95% CI above margin corresponds to a one-sided
    # 2.5% rejection region. Noncentral-t planning assumes iid normal pair means.
    def metrics(n):
        q=float(t.ppf(.975,n-1));return q*sd/math.sqrt(n),float(nct.sf(q,n-1,gap*math.sqrt(n)/sd))
    lo=30;hi=cap
    if metrics(hi)[0]>half_width or metrics(hi)[1]<power:n=None
    else:
        while lo<hi:
            m=(lo+hi)//2;h,p=metrics(m)
            if h<=half_width and p>=power:hi=m
            else:lo=m+1
        n=lo
    h40,p40=metrics(40);lower=sd*math.sqrt(39/float(chi2.ppf(.975,39)));upper=sd*math.sqrt(39/float(chi2.ppf(.025,39)))
    return {'evidence_class':'HYPOTHETICAL_DESIGN_CALCULATION','sd_assumption':sd,'margin_proposal':margin,'alternative_proposal':alternative,'effect_gap':gap,'n_required_within_cap':n,'cap':cap,'power_target':power,'half_width_target':half_width,'at_required':dict(zip(['half_width','normal_noncentral_t_power'],metrics(n))) if n else None,'pilot_n_proposal':40,'pilot_half_width_if_sd_holds':h40,'pilot_power_if_sd_holds':p40,'sigma_interval_if_pilot_s_equals_assumed_sd':[lower,upper],'assumptions':'IID approximately normal paired-unit means; sample-size planning only,not guaranteed actual power; difference bounded [-1,1],possible heterogeneity/non-normality. No outcomes used.','scientific_approval':False,'pilot_execution_authorized':False}

def estimand(completions_a,completions_b,exogenous_counts):
    """Exact algebra on constructed unit counts; not called on prospective data."""
    if not(len(completions_a)==len(completions_b)==len(exogenous_counts)) or not exogenous_counts:raise ContractError('Matched units')
    diff=[]
    for a,b,m in zip(completions_a,completions_b,exogenous_counts):
        if any(type(v) is not int for v in [a,b,m]) or m<1 or not 0<=a<=m or not 0<=b<=m:raise ContractError('Fixed task denominator')
        diff.append(a/m-b/m)
    return {'paired_unit_differences':diff,'equal_unit_mean':sum(diff)/len(diff),'task_weighted_difference':(sum(completions_a)-sum(completions_b))/sum(exogenous_counts),'warning':'Distinct weighting estimands when counts vary; preselect equal independent seed-unit weighting.'}
