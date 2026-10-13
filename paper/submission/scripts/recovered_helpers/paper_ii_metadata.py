import urllib.request,json,pathlib
out=pathlib.Path('/Users/gunnchos/Downloads/gunnchos-7gc-research-product-spine/repos/_publication_worktrees/spectrumx-ai-ran-gary/paper/submission/audit')
dois=['10.1109/COMST.2017.2745201','10.1109/JSAC.2016.2611964','10.1109/TVT.2020.3018817','10.1109/TII.2020.3016320','10.1109/LWC.2022.3146207','10.1109/TITS.2024.3522243','10.1145/3711682','10.1109/GLOBECOM52923.2024.10901362']
results=[]
for doi in dois:
 try:
  req=urllib.request.Request('https://api.crossref.org/works/'+doi,headers={'User-Agent':'PublicationEvidenceAudit/1.0'})
  with urllib.request.urlopen(req,timeout=25) as r: data=json.load(r)['message']
  results.append({'DOI':doi,'title':data.get('title'),'author':data.get('author'),'container-title':data.get('container-title'),'published':data.get('published'),'volume':data.get('volume'),'issue':data.get('issue'),'page':data.get('page'),'URL':data.get('URL'),'type':data.get('type')})
  print(doi,data.get('title'),data.get('published'),data.get('page'))
 except Exception as e:results.append({'DOI':doi,'error':str(e)});print(doi,str(e))
(out/'crossref_metadata.json').write_text(json.dumps(results,indent=2)+'\n')
