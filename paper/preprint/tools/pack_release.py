"""Deterministic bounded release archives; no network/publishing action."""
from pathlib import Path
import json,hashlib,zipfile,subprocess,shutil
P=Path(__file__).resolve().parents[1]/'v0.1';E=P/'evidence';REPO=P.parents[2]
sha=subprocess.check_output(['git','rev-parse','617f01c'],cwd=REPO,text=True).strip()
prov={'repository':'gunnchOS3k/spectrumx-ai-ran-gary','branch':'publication/paper-ii-owner-review-2026-10-12','original_experimental_sha':'86a5a6f3000fce7593ee1bdfe02bae6bc37d2498','revised_experimental_sha':'b44357f764d440f983132bd44a9c23c180591f70','protocol_predecessor_sha':'72044d4008dc7624b1246b880387a693c37150e7','historical_author_review_sha':'f7b07300b5204f4ea1cf44eb925d603fe9705117','recovery_checkpoint_sha':'ef9fd69006fbe26c78cbed0b0c84e778e48b5782','publication_scientific_content_sha':sha,'content_anchor_scope':'Exact manuscript TeX/PDF/bibliography/figure/table sources, scientific reports, frozen snapshots and audit/acceptance scripts. Packaging metadata, manifests, acceptance outputs and final readiness records are added later and individually hashed; final checkpoint is recorded in the portal outside these self-referential artifacts.','publication_authorized':False,'human_scientific_approval':False,'originality_human_verified':False,'article_license_approved':False,'article_doi':None,'evidence_doi':None,'local_candidate_date':'2026-10-14','seed_scope':'train0..4;heldout/domain100..104;5seedmeans/version;no new seeds','frozen_objects':json.loads((E/'source/MANIFEST.json').read_text())}
for root in [P,E]:(root/'PROVENANCE.json').write_text(json.dumps(prov,indent=2)+'\n')
def payloads(root):
 return [p for p in sorted(root.rglob('*')) if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts and 'generated' not in p.relative_to(root).parts]
def manifest(root):
 files=payloads(root);(root/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix()+'\n' for p in files));return len(files)
def ziproot(root,dest):
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for p in sorted(root.rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts and 'generated' not in p.relative_to(root).parts:
    i=zipfile.ZipInfo(p.relative_to(root).as_posix(),(2026,10,14,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o100644<<16;z.writestr(i,p.read_bytes())
# Source staging keeps scientific review docs and portable plot sources but excludes QA rasters, historical implementation trees and prior environment artifacts.
STAGE=P/'verification/source_staging';STAGE.mkdir(parents=True,exist_ok=True)
# Remove only this tool's old generated staging, never user/source/historical files.
for p in list(STAGE.iterdir()):
 if p.is_dir():shutil.rmtree(p)
 else:p.unlink()
for name in ['preprint.tex','preprint.pdf','references.bib','BUILD.sh','BUILD_INSTRUCTIONS.md','SOFTWARE_LICENSE_NOTICE.txt','PROVENANCE.json']:
 shutil.copy2(P/name,STAGE/name)
for folder in ['figures','tables','figure_source','metadata']:shutil.copytree(P/folder,STAGE/folder)
for p in P.glob('*.md'):shutil.copy2(p,STAGE/p.name)
counts={'source_payload_files':manifest(STAGE),'evidence_payload_files':manifest(E)}
ziproot(STAGE,P/'paper-ii-v0.1-source.zip');ziproot(E,P/'paper-ii-v0.1-evidence.zip')
art={}
for n in ['preprint.pdf','paper-ii-v0.1-source.zip','paper-ii-v0.1-evidence.zip']:
 f=P/n;art[n]={'path':str(f),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
(P/'ARTIFACT_HASHES.json').write_text(json.dumps({**counts,'artifacts':art,'publication_scientific_content_sha':sha,'publication_authorized':False},indent=2)+'\n')
print(json.dumps({**counts,'artifacts':art},indent=2))
