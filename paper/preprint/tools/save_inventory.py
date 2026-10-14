from pathlib import Path
import hashlib,json,subprocess
P=Path(__file__).resolve().parents[1]/'v0.1';repo=P.parents[2]
# Use Git's tracked+authorized untracked inventory, respecting generated/cache exclusions.
paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','paper/preprint/v0.1'],cwd=repo,text=True).splitlines()
items={}
for rel in sorted(set(paths)):
 f=repo/rel
 if f.name=='LOCAL_INVENTORY.json' or not f.is_file():continue
 items[f.relative_to(P).as_posix()]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size}
(P/'LOCAL_INVENTORY.json').write_text(json.dumps({'scope':'Every Git-tracked or nonignored release/QA file under v0.1, excluding this self-referential inventory; caches/build products/generated staging excluded by .gitignore. Archive payload manifests independently cover all shipped members.','files':items,'count':len(items),'publication_authorized':False},indent=2)+'\n')
print(json.dumps({'inventoried_files':len(items)}))
