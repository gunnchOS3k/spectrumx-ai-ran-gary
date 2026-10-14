#!/usr/bin/env python3
"""Bounded fixed engineering timing; no environment evolution or task outcomes."""
from pathlib import Path
from dataclasses import asdict
import argparse,collections,hashlib,importlib.metadata,json,os,platform,resource,statistics,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import p7a
from p7a.fixtures import fixtures
from p7a.session import Session,child_cpu
from p7a.memory import resident_bytes
from fair_control.controllers import Policy
from fair_control.p6_process import controlled_decision
from fair_control.core import digest

def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def stats(xs):
    xs=[x for x in xs if x is not None]
    return {'n':len(xs),'min':min(xs),'median':statistics.median(xs),'mean':statistics.mean(xs),'max':max(xs)} if xs else None

def signature(d):
    x=asdict(d);x.pop('elapsed_s');return digest(x)

def run(output):
    output.mkdir(parents=True,exist_ok=False)
    fixed=fixtures();save(output/'FIXED_ENGINEERING_INPUTS.json',fixed)
    env={'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),'logical_cpus':os.cpu_count(),'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','scipy','jsonschema']},'cpu_name':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip() if sys.platform=='darwin' else platform.processor(),'load_before':list(os.getloadavg()),'exclusive_host':False,'fresh_dependency_install':False,'measurement_scope':'Sequential local fixed engineering requests; no environment step/task-completion outcome,seed or future workload','rss_method':'Darwin proc_pidinfo PROC_PIDTASKINFO.pti_resident_size bytes; raw ru_maxrss retained separately with no conversion','clock':vars(time.get_clock_info('perf_counter'))}
    results=[];sessions=[];global_start=time.perf_counter();global_cpu=time.process_time();global_child=child_cpu()
    # Exactly 6 inputs x 3 repetitions x 2 families x 2 execution mechanisms.
    # Fixed fixture order rotates by repetition; family order reversed in odd reps
    # for fresh mode. This is blocking, not randomization or scientific pairing.
    for rep in range(3):
        order=fixed[rep:]+fixed[:rep]
        for case in order:
            for family in (['mpc','hysteresis'] if rep%2==0 else ['hysteresis','mpc']):
                if time.perf_counter()-global_start>60:raise RuntimeError('Engineering benchmark wall cap; preserve partial JSONL')
                d,r=controlled_decision(Policy(family),case['estimated_input'])
                row={'mode':'FRESH_P6','rep':rep,'fixture':case['id'],'input_sha256':case['sha256'],'family':family,'decision_status':d.status,'action_sha256':digest(asdict(d.action)),'decision_signature':signature(d),'timing':r}
                results.append(row)
                with (output/'RAW_REQUESTS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    for family in ['mpc','hysteresis']:
        session=None
        try:
            for rep in range(3):
                for case in fixed[rep:]+fixed[:rep]:
                    if time.perf_counter()-global_start>60:raise RuntimeError('Engineering wall cap')
                    if session is None or session.closed:session=Session(Policy(family))
                    d,r=session.decide(case['estimated_input'])
                    rss=resident_bytes(session.proc.pid) if not session.closed else None
                    row={'mode':'REUSE_ENGINEERING','rep':rep,'fixture':case['id'],'input_sha256':case['sha256'],'family':family,'decision_status':d.status,'action_sha256':digest(asdict(d.action)),'decision_signature':signature(d),'timing':r,'resident_bytes_after_response':rss}
                    results.append(row)
                    with (output/'RAW_REQUESTS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    if session.closed:sessions.append({'family':family,**session.summary});session=None
        finally:
            if session:sessions.append({'family':family,**session.close()})
    env['load_after']=list(os.getloadavg());env['parent_peak_rss_raw']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    grouped=[]
    for mode in ['FRESH_P6','REUSE_ENGINEERING']:
        for family in ['mpc','hysteresis']:
            for case in fixed:
                rows=[x for x in results if x['mode']==mode and x['family']==family and x['fixture']==case['id']]
                grouped.append({'mode':mode,'family':family,'fixture':case['id'],'requests':len(rows),'process_status_counts':dict(collections.Counter(x['timing']['process_status'] for x in rows)),'decision_status_counts':dict(collections.Counter(x['decision_status'] for x in rows)),'decision_window_s':stats([x['timing']['decision_window_s'] for x in rows]),'startup_s':stats([x['timing'].get('startup_s') for x in rows]),'worker_s':stats([x['timing']['worker_decision_s'] for x in rows]),'calls':stats([x['timing']['actual_model_evaluations'] for x in rows]),'unavailable_calls':sum(x['timing']['actual_model_evaluations'] is None for x in rows),'resident_bytes':stats([x.get('resident_bytes_after_response') for x in rows])})
    fresh=[x for x in results if x['mode']=='FRESH_P6'];reuse=[x for x in results if x['mode']=='REUSE_ENGINEERING']
    fc=sum(x['timing']['parent_cpu_s']+x['timing']['child_cpu_s'] for x in fresh)
    fw=sum(x['timing']['outer_elapsed_s'] for x in fresh)
    rc=sum(x['parent_cpu_s']+x['child_cpu_s'] for x in sessions);rw=sum(x['wall_s'] for x in sessions)
    projected={}
    for mode,cpu,wall,n in [('FRESH_P6',fc,fw,len(fresh)),('REUSE_ENGINEERING',rc,rw,len(reuse))]:
        cpu_per=cpu/n;wall_per=wall/n
        projected[mode]={'measured_requests':n,'measured_parent_plus_child_cpu_s':cpu,'measured_wall_s':wall,'cpu_s_per_request':cpu_per,'wall_s_per_request':wall_per,'full_40x24x64x2_cpu_hours':cpu_per*122880/3600,'full_40x24x64x2_wall_hours':wall_per*122880/3600,'threefold_controller_only_cpu_hours':3*cpu_per*122880/3600,'smaller_8x24x64x2_threefold_cpu_hours':3*cpu_per*24576/3600,'later_867x24x64x2_cpu_hours':cpu_per*(867*24*64*2)/3600,'scope':'Equal-weight six engineering inputs; linear arithmetic,not confidence bounds or full-harness runtime. Threefold factor is an explicit unvalidated contingency assumption. Reuse includes observed restart/startup and per-session bookkeeping,not a proven long-run leak or host-load bound.'}
    parity=[]
    for case in fixed:
        for fam in ['mpc','hysteresis']:
            a=[x for x in fresh if x['fixture']==case['id'] and x['family']==fam]
            b=[x for x in reuse if x['fixture']==case['id'] and x['family']==fam]
            parity.append({'fixture':case['id'],'family':fam,'all_semantic_signatures_equal':len({x['decision_signature'] for x in a+b})==1,'all_action_hashes_equal':len({x['action_sha256'] for x in a+b})==1,'interpretation':'Timing-cutoff differences must be retained; equality on these inputs does not prove universal parity'})
    report={'evidence_class':'CONSTRUCTED_RUNTIME_DIAGNOSTIC_NOT_SCIENTIFIC_OUTCOMES','requests':len(results),'fixed_cases':len(fixed),'repetitions':3,'random_seeds':[],'sessions':sessions,'groups':grouped,'projections':projected,'semantic_comparison':parity,'environment':env,'elapsed_wall_s':time.perf_counter()-global_start,'parent_plus_children_cpu_s':time.process_time()-global_cpu+child_cpu()-global_child,'pilot_executed':False,'heldout_accessed':False,'publication_authorized':False,'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'input_catalog_sha256':digest(fixed)}
    save(output/'RUNTIME_DIAGNOSTICS.json',report)
    print(json.dumps({'requests':len(results),'projections':projected,'status_counts':dict(collections.Counter((x['mode']+'/'+x['timing']['process_status']) for x in results))},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args();run(a.output)
