from pathlib import Path
import json,subprocess,hashlib,zipfile,tempfile,sys,os
P=Path(__file__).resolve().parents[1]/'v0.1';V=P/'verification';A=Path(tempfile.mkdtemp(prefix='paper-ii-v01-acceptance-'));S=A/'source';E=A/'evidence';S.mkdir();E.mkdir()
for name,root in [('source',S),('evidence',E)]:
 with zipfile.ZipFile(P/f'paper-ii-v0.1-{name}.zip') as z:
  for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
  assert z.testzip() is None;z.extractall(root)
ENV=dict(os.environ,PATH='/Users/gunnchos/Library/TinyTeX/bin/universal-darwin:'+os.environ['PATH'],MPLCONFIGDIR=str(A/'mpl'),PYTHONDONTWRITEBYTECODE='1')
def run(args,cwd,log):
 with (V/log).open('w') as f:r=subprocess.run(args,cwd=cwd,env=ENV,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,(args,log,r.returncode)
run([sys.executable,'scripts/verify_integrity.py'],E,'CLEAN_EVIDENCE_INITIAL_INTEGRITY.txt')
run([sys.executable,str(E/'scripts/verify_integrity.py'),str(S)],E,'CLEAN_SOURCE_INITIAL_INTEGRITY.txt')
run(['sh','BUILD.sh'],S,'CLEAN_SOURCE_COMPILE.txt')
run(['pdftoppm','-scale-to','1700','-png','preprint.pdf',str(A/'page')],S,'CLEAN_SOURCE_RENDER.txt')
images=sorted(A.glob('page-*.png'));expected=sorted((V/'pages').glob('page-*.png'));assert len(images)==len(expected)==10
assert all(a.read_bytes()==b.read_bytes() for a,b in zip(images,expected)),'Clean extraction render changed'
# Finish portable instruction acceptance without full primary-result campaign.
for script in ['audit_science.py','optimizer_bounds.py','generate_figures.py','verify_claims.py','reproduce_heldout.py','verify_integrity.py']:
 print('Running',script,flush=True);run([sys.executable,'scripts/'+script],E,'CLEAN_'+script.replace('.py','.txt'))
# Regenerate plots from the source archive's own relative plot inputs as well.
run([sys.executable,'generate_figures.py'],S/'figure_source','CLEAN_SOURCE_FIGURE_REGENERATION.txt')
from PIL import Image
figures=[]
for p in (S/'figures').glob('*.png'):
 fresh=S/'figure_source/generated/figures'/p.name
 assert Image.open(p).tobytes()==Image.open(fresh).tobytes(),p.name
 figures.append(p.name)
report={'clean_directory':str(A),'source_build_succeeded':True,'page_count':len(images),'every_clean_page_raster_equals_visually_inspected_candidate':True,'source_figure_regeneration_pixel_equal':figures,'evidence_instructions_succeeded':True,'heldout_acceptance':json.loads((E/'generated/heldout_acceptance/REPORT.json').read_text()),'new_seeds_added':False,'new_campaign_executed':False,'python':sys.version,'fresh_environment_installation_tested':False,'publication_authorized':False}
(V/'CLEAN_ACCEPTANCE.json').write_text(json.dumps(report,indent=2)+'\n')
# Retain the actual acceptance outputs separately, without rewriting any historical raw file.
import shutil
D=V/'heldout_acceptance';shutil.copytree(E/'generated/heldout_acceptance',D,dirs_exist_ok=True)
for root in [P/'evidence/verification']:
 shutil.copytree(D,root/'heldout_acceptance',dirs_exist_ok=True)
 shutil.copy2(V/'CLEAN_ACCEPTANCE.json',root/'CLEAN_ACCEPTANCE.json')
print(json.dumps({'clean_directory':str(A),'pages':len(images),'heldout_parity':{v:x['non_timing_exact_equal'] for v,x in report['heldout_acceptance']['versions'].items()},'passed':True}))
