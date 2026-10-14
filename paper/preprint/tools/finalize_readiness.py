from pathlib import Path
import json,hashlib,shutil,re,subprocess
P=Path(__file__).resolve().parents[1]/'v0.1';E=P/'evidence';V=P/'verification';repo=P.parents[2]
a=json.loads((V/'CLEAN_ACCEPTANCE.json').read_text());assert a['source_build_succeeded'] and a['evidence_instructions_succeeded']
for x in a['heldout_acceptance']['versions'].values():assert x['non_timing_exact_equal'] and set(x['optimizer_backends'])=={'slsqp'} and all(x['mutation_tests'].values())
# Final PDF structural checks complement actual visual inspection.
import fitz
pdf=fitz.open(P/'preprint.pdf');assert len(pdf)==10
text='\n'.join(page.get_text() for page in pdf)
assert '\ufffd' not in text and 'Publication authorization: false' in text
assert all('References' not in page.get_text() or page.number>=8 for page in pdf)
log=(P/'preprint.log').read_text();assert not re.search(r'Overfull|undefined|Missing character|^!',log,re.M)
raw=(P/'preprint.tex').read_bytes();assert not any(x<32 and x not in [9,10] for x in raw)
page_notes={1:'Title, confirmed identity, exploratory/nonpublic label, abstract and motivation checked.',2:'Related-work bounds and model equations1–7 checked; indicator glyph corrected and re-inspected.',3:'Energy/share model, SLSQP objective and residual bounds readable; all indicators re-inspected.',4:'Selector differences, conditions, zero-latency exception and baseline fairness checked after final symbol edit.',5:'Policy/priors tables and five-seed statistics readable.',6:'Held-out table/figure and switching decomposition text checked; Fig4 cross-reference repaired.',7:'Switching/count and retrospective algebra figures readable; negative signs fixed without broken line separation.',8:'Domain-shift figure, intervention table and revised harmful/neutral outcomes table checked.',9:'Conclusion, author/data rights and substantive AI disclosure checked; bibliography begins cleanly.',10:'Remaining bibliography entries readable, all10reference entries present; whitespace at end is normal.'}
visual={'page_count':10,'reviewer':'AI-assisted coordinating agent; not human peer/scientific review','method':'Every rendered page viewed at1700px; affected pages re-inspected after corrections; final clean rebuild rasters identical','pdf_sha256':hashlib.sha256((P/'preprint.pdf').read_bytes()).hexdigest(),'pages':[{'page':i,'sha256':hashlib.sha256((V/'pages'/f'page-{i:02}.png').read_bytes()).hexdigest(),'review_note':page_notes[i]} for i in range(1,11)],'missing_glyph_or_unresolved_reference_or_overfull_box_found':False,'remaining_layout_notes':'Some harmless underfull justification warnings; no clipped figures, missing symbols, unresolved citations/references or overfull boxes. Figure2 truncated left axis is explicitly labeled in caption; not hidden.'}
(V/'PDF_VISUAL_REVIEW.json').write_text(json.dumps(visual,indent=2)+'\n');shutil.copy2(V/'PDF_VISUAL_REVIEW.json',E/'verification/PDF_VISUAL_REVIEW.json')
keys=['SOURCE_SEMANTICS_AUDITED','LINEAR_SELECTOR_CHARACTERIZED','OPTIMIZER_VERIFIED','BASELINE_FAIRNESS_AUDITED','NEGATIVE_RESULTS_PRESERVED','STATISTICS_VERIFIED','CITATIONS_VERIFIED','CLAIMS_EVIDENCE_ALIGNED','NOVELTY_BOUNDARY_DOCUMENTED','CROSS_PAPER_OVERLAP_REVIEWED','MANUSCRIPT_COMPILES','SOURCE_ARCHIVE_BUILDS','REPRODUCIBILITY_ARCHIVE_VERIFIED','FIGURES_VERIFIED','OWNER_REVIEW_READY']
gates={k:True for k in keys}
gates.update({k:False for k in ['OWNER_SCIENTIFIC_APPROVAL','ORIGINALITY_HUMAN_VERIFIED','ARTICLE_LICENSE_APPROVED','PUBLICATION_AUTHORIZED']})
scopes={'OPTIMIZER_VERIFIED':'Objective/analytical optimum and numerical behavior independently checked;26strictovershoots retained; exact numerical feasibility is FALSE.','BASELINE_FAIRNESS_AUDITED':'Differences diagnosed, not made fair; historical results unchanged.','STATISTICS_VERIFIED':'n5conditional synthetic estimates and retrospective paired arithmetic; no population/equivalence claim.','CITATIONS_VERIFIED':'Ten cited identities and bounded claim support; Sun/Dai/Jain full-text access limits disclosed; not exhaustive originality verification.','REPRODUCIBILITY_ARCHIVE_VERIFIED':'Bounded instructions accepted on recorded existing environment, frozen byte hashes and shipped payload manifest; not fresh-install/cross-platform/human replication.','OWNER_REVIEW_READY':'Complete technical packet for owner scientific review only, subject to final package re-extraction hash check.','NOVELTY_BOUNDARY_DOCUMENTED':'No priority established.'}
for root in [P,E]:(root/'QUALITY_GATES.json').write_text(json.dumps({'classification':'READY FOR OWNER REVIEW','gates':gates,'scope_limits':scopes},indent=2)+'\n')
s=(P/'SCIENTIFIC_READINESS_DECISION.md').read_text().replace('final READY FOR OWNER REVIEW depends on compilation, page inspection and clean release-archive acceptance. Current packaging status: REVISIONS REQUIRED until those checks finish.','compilation, page inspection and clean release-archive acceptance have passed. Current classification: **READY FOR OWNER REVIEW** for this bounded exploratory packet. Human scientific review has not been completed.')
for root in [P,E]:(root/'SCIENTIFIC_READINESS_DECISION.md').write_text(s)
readiness='''# Paper II exploratory preprint v0.1 — READY FOR OWNER REVIEW

Publication authorization: **false**. Human scientific approval: **false**. Originality human verified: **false**. Article-license approval: **false**.

This is a complete local methodological/negative-result candidate. It is not a journal-strength demonstration of fair or superior adaptive control. Frozen original/revised source and existing journal-review manuscript are preserved; no large campaign, extra seed, source retuning or public action occurred.

## Artifact inventory

* preprint.pdf —10pages; neutral standard article, independent of IEEE template. Editable preprint.tex and references.bib —10citedworks.
* figures/ —4PDF/PNG pairs; figure_source/ —portable plotting code and bounded input JSON; tables/ —5TeX tables plus fullsummary CSV. Some exported tables are supplemental, rather than inserted to inflate the paper.
* paper-ii-v0.1-source.zip —editable article/build files, plot sources/inputs/exports/tables, local metadata, review reports and preserved license notice.
* paper-ii-v0.1-evidence.zip —both exact engine/protocol snapshots and predecessor;19Gitobject hash bindings;20retained JSON output files (original/revised and historical reruns), historical reports/locks, current environment pins, statistics, focused assertions, bounded fresh acceptance outputs, data dictionary, claim links, licenses and SHA256SUMS.
* ARTIFACT_HASHES.json —exact outer artifact sizes/hashes; each archive's SHA256SUMS enumerates every shipped payload other than itself. Outer hashes also bind the manifest bytes. Final local inventory hashes all retained release/verification files separately.
* PROVENANCE.json —full original/revised experimental SHAs, predecessor, accepted journal-review SHA and separate publication scientific-content anchor. Packaging/readiness/check records follow that anchor; final local commit is recorded by the portal to avoid self-hashing claims.
* SCIENTIFIC_CONTRIBUTION_AUDIT,CLAIM_TO_IMPLEMENTATION_MATRIX,CLAIM_TO_EVIDENCE,LINEAR_SELECTOR_SEMANTIC_AUDIT,MATHEMATICAL_CHARACTERIZATION,OPTIMIZATION_SEMANTICS,BASELINE_FAIRNESS_AUDIT,POLICY_INFORMATION_MATRIX,STATISTICAL_VERIFICATION,NEGATIVE_RESULTS_AND_LIMITATIONS —substantive source/evidence audit.
* CITATION_VERIFICATION,NOVELTY_ASSESSMENT,FIVE_REVIEWER_OBJECTIONS,SCIENTIFIC_READINESS_DECISION,SCIENTIFIC_REVIEW —review and novelty limits.
* RIGHTS_AND_DECLARATIONS,AI_USE_DISCLOSURE,OWNER_REVIEW_WORKSHEET,POLICY_CHECK,PLAIN_LANGUAGE_SUMMARY,REPRODUCIBILITY,DATA_DICTIONARY,CROSS_PAPER_INTEGRITY —author/reproducibility guidance. metadata/{arxiv,techrxiv,zenodo}.json are local drafts with null DOI/license/date where unapproved.
* PROSPECTIVE_FAIR_CONTROL_STUDY.md —20-item future design; NOT run, approved or independently preregistered.
* verification/ —actual compile, clean extraction, numerical, mutation, image and parity records; prior failed exact-optimizer assertion retained.

## Verified acceptance and exact scope

Clean extraction compiled without undocumented machine paths in the source, no IEEE assets, unresolved reference/citation, missing glyph or overfull box. Every page was viewed; corrected malformed cross-reference, indicator glyph and minus-sign line break. All10clean rebuild page rasters equal the inspected candidate. All4source-archive figure rasters regenerated identically. Relevant Python dependency/runtime paths are supplied to the local test driver; user-facing extracted instructions are relative.

19frozen Git object bytes verified.14explicit claim links, all6exported result tables,1477non-timing summary metrics checked. Original/revised families are separate. Selector assertions reconstructed9600existing states per version; zero-latency tie exception included.100small component solves bound maxbudget residual3.1882259854e-6;26strictovershoots are a documented limitation, not concealed failures. Slot-score impact bound2.9172267767e-7 under frozen/current-runtime conditions is immaterial to the negative direction.

The one bounded archive-instruction test reran only existing held-out seeds100–104,24×16slots/version; exact non-timing whole-payload parity passed with SLSQPbackend counts retained. Scientific-value and seed-order mutations fail; a valid timing-only change passes with separate validation, and negative timing fails. Timing and all provenance bytes remain retained. These are new acceptance checks on the existing local environment, not independent human reproduction. Prior fullfamily results/tests remain historical. No fresh dependency installation/cross-platform test was done.

## Unresolved scientific limits

n=5seedmeans/version; conditional authored priors, retrospective unadjusted comparisons, no broad population inference or equivalence claim. No physical RF/NTN/energy measurement or calibrated digital twin. Static feasibility, observation use, compute/action opportunities, score/optimizer mismatch and truth-changing interventions limit causal interpretation. Revised adaptive fidelity/checkpoint findings are whole-policy harmful outcomes, not isolated component effects. Switching rescaling is fixed-action algebra, not a measured/horizon-aware transition experiment. State-of-the-art controllers are not reproduced; novelty is not established. Full-text inspection gaps and five journal objections remain explicit.

## Owner decisions and release boundary

Review science, originality and usefulness; confirm author list/contributions, funding/sponsors/conflicts, prior dissemination, patent/IP and permissions. Choose article and evidence/tooling licenses separately from preserved MIT software. Review AI disclosure and platform suitability. TechRxiv remains temporarily closed October14; arXiv endorsement/category/account rate capacity remain unverified; Zenodo human research/AI accountability requirements need owner assessment. No DOI is invented/reserved. No absence declaration is presumed.

All15technical/scoped review gates are true in QUALITY_GATES.json; four human/license/publication gates remain false. Next executable milestone: owner reviews this exact packet and records edits/decisions in OWNER_REVIEW_WORKSHEET.md. A prospective fair-control campaign requires a separate approved milestone; it is not part of this release. No push, preprint posting, external record, ORCID action, journal submission, merge, contact, fee or agreement is authorized/performed.
'''
# Improve spacing without touching immutable scientific inputs.
for x,y in [('—10','— 10'),('—4','— 4'),('—5','— 5'),('—20','— 20'),('10pages','10 pages'),('10citedworks','10 cited works'),('4PDF','4 PDF'),('5TeX','5 TeX'),('fullsummary','full summary'),('19Gitobject','19 Git-object'),('20retained','20 retained'),('all10clean','all 10 clean'),('All4source','All 4 source'),('All10clean','All 10 clean'),('19frozen','19 frozen'),('14explicit','14 explicit'),('6exported','6 exported'),('1477non','1,477 non'),('reconstructed9600existing','reconstructed 9,600 existing'),('100small','100 small'),('maxbudget','max budget'),('26strict','26 strict'),('bound2','bound 2'),('seeds100','seeds 100'),('16slots','16 slots'),('SLSQPbackend','SLSQP backend'),('fullfamily','full-family'),('5seedmeans','5 seed means'),('October14','October 14'),('All15technical','All 15 technical')]:readiness=readiness.replace(x,y)
for root in [P,E]:(root/'PREPRINT_RELEASE_READINESS.md').write_text(readiness)
(P/'PREPRINT_COVER_NOTE.md').write_text('Optional local cover note — not a platform requirement or submitted message.\n\nController Semantics and Negative Results in a Synthetic Cross-Layer Orchestration Study, Edmund Gunn, Jr., gunnchOS Research. Exploratory preprint v0.1, not peer reviewed or posted. The manuscript characterizes an archived synthetic controller comparison, preserves negative/adverse/inconclusive findings and provides separately versioned source/statistical evidence. No claim of measured network performance, adaptive superiority or established novelty. Author scientific/originality/rights/license declarations remain pending. Publication authorization:false. This note must not be sent without separate authorization.\n')
# Validate standard source-file SHA at publication anchor; only packaging records change later.
anchor=json.loads((P/'PROVENANCE.json').read_text())['publication_scientific_content_sha'];core=[]
for rel in ['preprint.tex','preprint.pdf','references.bib','BUILD.sh']+[p.relative_to(P).as_posix() for d in ['figures','tables','figure_source'] for p in (P/d).rglob('*') if p.is_file()]:
 b=subprocess.check_output(['git','show',f'{anchor}:paper/preprint/v0.1/{rel}'],cwd=repo)
 assert b==(P/rel).read_bytes(),rel;core.append(rel)
(V/'PUBLICATION_ANCHOR_CHECK.json').write_text(json.dumps({'source_sha':anchor,'exact_core_files_compared_to_git':core,'passed':True},indent=2)+'\n')
print(json.dumps({'readiness':'READY FOR OWNER REVIEW','technical_gates_true':len(keys),'approval_gates_false':4,'anchored_core_files':len(core),'optimizer_backends':{v:r['optimizer_backends'] for v,r in a['heldout_acceptance']['versions'].items()}}))
