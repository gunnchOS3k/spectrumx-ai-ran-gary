#!/usr/bin/env python3
"""Derive standalone Paper-II manuscript from its canonical authoring source."""
from pathlib import Path
import shutil,hashlib,json
s=Path(__file__).resolve().parents[1];p=s.parent
text=(p/'manuscript.tex').read_text().replace('\\newcommand{\\ArtifactRoot}{submission/}','\\newcommand{\\ArtifactRoot}{}')
text=text.replace('\\input{owner_review_results}',(p/'owner_review_results.tex').read_text())
assert '\\input{owner_review_results}' not in text
(s/'manuscript.tex').write_text(text)
shutil.copy2(p/'references.bib',s/'references.bib')
source={str(x.relative_to(s)):hashlib.sha256(x.read_bytes()).hexdigest() for x in s.rglob('*') if x.is_file() and x.suffix in ('.tex','.bib','.cls','.bst','.pdf','.png') and x.name!='manuscript.pdf'}
(s/'SOURCE_ASSET_SHA256.json').write_text(json.dumps(source,indent=2)+'\n')
print('Standalone submission generated from canonical Paper II source.')
