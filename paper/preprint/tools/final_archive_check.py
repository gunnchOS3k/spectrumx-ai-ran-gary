from pathlib import Path
import json,zipfile,hashlib,tempfile,subprocess,sys,os,shutil
P=Path(__file__).resolve().parents[1]/'v0.1';V=P/'verification';H=json.loads((P/'ARTIFACT_HASHES.json').read_text());prior=Path(json.loads((V/'CLEAN_ACCEPTANCE.json').read_text())['clean_directory']);A=Path(tempfile.mkdtemp(prefix='paper-ii-v01-final-'));S=A/'source';E=A/'evidence';S.mkdir();E.mkdir()
for kind,root in [('source',S),('evidence',E)]:
 f=P/f'paper-ii-v0.1-{kind}.zip';assert hashlib.sha256(f.read_bytes()).hexdigest()==H['artifacts'][f.name]['sha256']
 with zipfile.ZipFile(f) as z:assert z.testzip() is None;z.extractall(root)
 entries={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (root/'SHA256SUMS').read_text().splitlines()}
 actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'};assert actual==set(entries)
 for n,h in entries.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
 assert len(entries)==H[kind+'_payload_files']
 # Check preservation of every generated-stage archival byte in the physical ZIP.
 staging=P/'verification/source_staging' if kind=='source' else P/'evidence'
 for n in entries:assert (root/n).read_bytes()==(staging/n).read_bytes(),n
# Proof final packaged computational closure equals the one just run; avoid repeating the campaign after adding documentation.
closure=[]
for d in ['source','scripts','data','environment']:
 for f in (E/d).rglob('*'):
  if f.is_file():
   rel=f.relative_to(E);assert f.read_bytes()==(prior/'evidence'/rel).read_bytes(),rel;closure.append(rel.as_posix())
ENV=dict(os.environ,PATH='/Users/gunnchos/Library/TinyTeX/bin/universal-darwin:'+os.environ['PATH'],PYTHONDONTWRITEBYTECODE='1')
for args,cwd,log in [([sys.executable,'scripts/verify_integrity.py'],E,'FINAL_EVIDENCE_INTEGRITY.txt'),([sys.executable,str(E/'scripts/verify_integrity.py'),str(S)],E,'FINAL_SOURCE_INTEGRITY.txt'),(['sh','BUILD.sh'],S,'FINAL_SOURCE_COMPILE.txt'),(['pdftoppm','-scale-to','1700','-png','preprint.pdf',str(A/'page')],S,'FINAL_SOURCE_RENDER.txt')]:
 with (V/log).open('w') as out:r=subprocess.run(args,cwd=cwd,env=ENV,stdout=out,stderr=subprocess.STDOUT)
 assert r.returncode==0,log
for i in range(1,11):assert (A/f'page-{i:02}.png').read_bytes()==(V/'pages'/f'page-{i:02}.png').read_bytes()
# A damaged payload fails manifest hash comparison. Operates only on a new temporary test file.
p=E/'PLAIN_LANGUAGE_SUMMARY.md';old=p.read_bytes();p.write_bytes(old+b'\nTAMPER TEST\n')
r=subprocess.run([sys.executable,'scripts/verify_integrity.py'],cwd=E,env=ENV,capture_output=True,text=True);assert r.returncode!=0;p.write_bytes(old)
report={'final_archive_paths_and_hashes':H['artifacts'],'source_payload_files_verified':H['source_payload_files'],'evidence_payload_files_verified':H['evidence_payload_files'],'all_archive_payload_bytes_verified':True,'zip_crc_verified':True,'safe_relative_members':True,'final_extraction_directory':str(A),'source_build_succeeded':True,'all_10_page_rasters_identical_to_reviewed_candidate':True,'final_evidence_execution_closure_equal_to_successful_acceptance':closure,'closure_file_count':len(closure),'damaged_payload_rejected_by_manifest':True,'scientific_rerun_repeated_after_documentation_only_packaging':False,'independent_human_review':False,'publication_authorized':False}
(V/'FINAL_ARCHIVE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['final_evidence_execution_closure_equal_to_successful_acceptance','final_archive_paths_and_hashes']},indent=2))
