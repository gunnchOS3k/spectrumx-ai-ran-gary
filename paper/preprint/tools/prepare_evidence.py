from pathlib import Path
import json,shutil,hashlib,subprocess,sys,platform
R=Path(__file__).resolve().parents[3];P=R/'paper/preprint/v0.1';E=P/'evidence';V=P/'verification'
for d in [E/'source',E/'data',E/'prior_audit',E/'scripts',E/'environment',V]:d.mkdir(parents=True,exist_ok=True)
(P/'.gitattributes').write_text('* -text\n');(P/'.gitignore').write_text('*.aux\n*.log\n*.out\n*.bbl\n*.blg\n__pycache__/\nevidence/generated/\n')
SHAS={'original':'86a5a6f3000fce7593ee1bdfe02bae6bc37d2498','revised':'b44357f764d440f983132bd44a9c23c180591f70'}
manifest={};initial={}
for name,sha in SHAS.items():
 dest=E/'source'/name;dest.mkdir(exist_ok=True)
 for rel in ['src/airan_research/experiments/digital_programme.py','paper/artifacts/experiment_protocol.yaml','requirements.txt','LICENSE']:
  b=subprocess.check_output(['git','show',sha+':'+rel],cwd=R);target=dest/Path(rel).name;target.write_bytes(b);manifest[str(target.relative_to(E))]={'source_sha':sha,'git_path':rel,'sha256':hashlib.sha256(b).hexdigest()}
 for suffix in ['','_reproduced']:
  dd=E/'data'/(name+suffix);dd.mkdir(exist_ok=True)
  for f in (R/'paper/submission/evidence'/(name+suffix)).glob('rq2_*.json'):
   shutil.copyfile(f,dd/f.name)
   if not suffix:
    rel='results/experiments/'+f.name;b=subprocess.check_output(['git','show',sha+':'+rel],cwd=R);assert b==f.read_bytes();manifest[str((dd/f.name).relative_to(E))]={'source_sha':sha,'git_path':rel,'sha256':hashlib.sha256(b).hexdigest()}
predecessor='72044d4008dc7624b1246b880387a693c37150e7';b=subprocess.check_output(['git','show',predecessor+':paper/artifacts/experiment_protocol.yaml'],cwd=R);(E/'source/protocol_predecessor.yaml').write_bytes(b);manifest['source/protocol_predecessor.yaml']={'source_sha':predecessor,'git_path':'paper/artifacts/experiment_protocol.yaml','sha256':hashlib.sha256(b).hexdigest()}
(E/'source/MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');shutil.copyfile(R/'LICENSE',E/'LICENSE');shutil.copyfile(R/'LICENSE',P/'SOFTWARE_LICENSE_NOTICE.txt')
for f in (R/'paper/submission/audit').glob('*'):
 if f.is_file():shutil.copyfile(f,E/'prior_audit'/f.name)
shutil.copyfile(R/'paper/submission/reproducibility/requirements-lock.txt',E/'environment/HISTORICAL_REQUIREMENTS.txt')
(E/'environment/acceptance-requirements.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
(E/'environment/ACCEPTANCE_ENVIRONMENT.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'existing_environment_reused':True,'fresh_installation_tested':False,'cross_platform_tested':False,'numpy':__import__('numpy').__version__,'scipy':__import__('scipy').__version__,'matplotlib':__import__('matplotlib').__version__},indent=2)+'\n')
for name in ['7gc-digital-twin','spectrumx-ai-ran-gary','ntn-resilience-sim','gunnchos-research-portal']:
 rr=R.parent/name;initial[name]={'branch':subprocess.check_output(['git','branch','--show-current'],cwd=rr,text=True).strip(),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=rr,text=True).strip(),'status_at_recovery':'clean before new PaperII files' if name=='spectrumx-ai-ran-gary' else subprocess.check_output(['git','status','--porcelain'],cwd=rr,text=True).strip() or 'clean'}
(V/'RECOVERY.json').write_text(json.dumps({'checkpoints':initial,'experimental_shas':SHAS,'protocol_predecessor':predecessor,'preserved_submission_sha':initial['spectrumx-ai-ran-gary']['head'],'frozen_git_bytes_verified':len(manifest),'reset_or_push':False,'publication_authorized':False},indent=2)+'\n');print('Recovered',len(manifest),'frozen Git objects; historical outputs and environments retained.')
