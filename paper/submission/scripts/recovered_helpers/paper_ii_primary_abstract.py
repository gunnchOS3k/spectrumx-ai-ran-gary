import urllib.request,json,pathlib
p=pathlib.Path('/Users/gunnchos/Downloads/gunnchos-7gc-research-product-spine/repos/_publication_worktrees/spectrumx-ai-ran-gary/paper/submission/audit')
doi='10.1109/TVT.2020.3018817'
with urllib.request.urlopen('https://api.crossref.org/works/'+doi,timeout=25) as r:j=json.load(r)['message']
(p/'sun2020_crossref_full.json').write_text(json.dumps(j,indent=2)+'\n');print(j.get('abstract','NO_ABSTRACT'));print(j.get('resource'))
