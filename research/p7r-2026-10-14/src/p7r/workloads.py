"""Real executable original benchmark operations; engineering evidence only here."""
from pathlib import Path
from dataclasses import asdict
import importlib.util,json,os,resource,subprocess,sys,time
from .contracts import Task,evaluate,sha,ContractError
TEXT=b'Original nonpersonal lesson: sum the integers 1 through 10.\n'
NOTE=b'Draft note\nEdited locally.\n'
CLOCK='host_perf_counter_ns_single_process_boot'

def atomic(path,data):
 p=Path(path);tmp=p.with_suffix('.pending')
 with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)

def execute(kind,root,deadline_s=5.):
 root=Path(root)
 if not root.is_dir() or any(root.iterdir()):raise ContractError('Empty owner-controlled temporary directory')
 if kind not in {'program','cached_lesson','document','restore'}:raise ContractError('Known original workload only')
 if type(deadline_s) not in (int,float) or not 0<deadline_s<=10:raise ContractError('Bounded deadline')
 release=time.perf_counter_ns();cpu=time.process_time_ns();usage=resource.getrusage(resource.RUSAGE_CHILDREN)
 out=b'';persist=None;failure=None
 expected={'program':b'55\n','cached_lesson':TEXT,'document':NOTE,'restore':NOTE}[kind]
 if kind=='program':
  script=root/'original.py';script.write_text('print(sum(range(1,11)))\n')
  try:
   p=subprocess.run([sys.executable,'-I',str(script)],capture_output=True,cwd=root,timeout=min(2,deadline_s));out=p.stdout;failure=None if p.returncode==0 else 'program_exit_error'
  except subprocess.TimeoutExpired:failure='program_timeout'
 elif kind=='cached_lesson':
  atomic(root/'lesson.txt',TEXT);out=(root/'lesson.txt').read_bytes();atomic(root/'progress.json',json.dumps({'lesson_sha256':sha(out),'step':1}).encode());persist=out if json.loads((root/'progress.json').read_text())['step']==1 else None
 elif kind=='document':
  atomic(root/'note.txt',b'Draft note\n');atomic(root/'note.txt',NOTE);out=NOTE;persist=(root/'note.txt').read_bytes()
 else:
  state={'version':1,'text':'Draft note\n'};atomic(root/'checkpoint.json',json.dumps({'state':state,'hash':sha(json.dumps(state,sort_keys=True).encode())}).encode());loaded=json.loads((root/'checkpoint.json').read_text())
  if loaded['hash']!=sha(json.dumps(loaded['state'],sort_keys=True).encode()):failure='checkpoint_corrupt'
  else:out=(loaded['state']['text']+'Edited locally.\n').encode();atomic(root/'restored.txt',out);persist=(root/'restored.txt').read_bytes()
 finish=time.perf_counter_ns();u=resource.getrusage(resource.RUSAGE_CHILDREN)
 t=Task(kind,sha(expected),release,release+int(deadline_s*1e9),CLOCK,int(30e9),kind!='program')
 r={'task_id':kind,'clock_domain':CLOCK,'finished_ns':finish,'generated_ns':release,'observed_output_sha256':sha(out),'durable_reopened':persist==out if persist is not None else False,'success_reported':failure is None,'failure':failure,'evidence_category':'CONSTRUCTED_ENGINEERING_VERIFICATION','integration_class':'ORIGINAL_EXECUTABLE_BENCHMARK_NOT_PRODUCTION_APP','resource':{'parent_cpu_ns':time.process_time_ns()-cpu,'child_cpu_s':(u.ru_utime+u.ru_stime)-(usage.ru_utime+usage.ru_stime),'max_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'max_rss_unit':'OS_NATIVE_NOT_CONVERTED'},'progress_preservation':'Reopened host files;not crash/power-cut durability'}
 return {'task':asdict(t),'result':r,'evaluation':evaluate(t,r)}

def execute_device_os_core(module_file,expected_module_sha256):
 p=Path(module_file)
 if sha(p.read_bytes())!=expected_module_sha256:raise ContractError('Pinned licensed core source changed')
 name='p7r_owned_core';spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
 try:
  spec.loader.exec_module(module)
  a=module.OfflineSyncEngine(replica_id='original-a',now_ms=lambda:1000);b=module.OfflineSyncEngine(replica_id='original-b',now_ms=lambda:1000)
  a.put('original-nonpersonal-note',{'text':'offline edit'});pending=a.pending();b.sync_from_peer(pending);b.sync_from_peer(pending)
  actual=b.get('original-nonpersonal-note')
  return {'integration_class':'PINNED_DEVICE_OS_CORE_LIBRARY_EXECUTED','module_sha256':expected_module_sha256,'correct':actual=={'text':'offline edit'},'record_count':len(b.store),'duplicate_content_preserved':len(b.store)==1,'production_application_executed':False,'network_transport_executed':False,'durable_cloud_receipt_verified':False,'evidence_category':'CONSTRUCTED_ENGINEERING_VERIFICATION'}
 finally:sys.modules.pop(name,None)
