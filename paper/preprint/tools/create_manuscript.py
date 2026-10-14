from pathlib import Path
import re,shutil,json
R=Path(__file__).resolve().parents[3];P=R/'paper/preprint/v0.1';E=P/'evidence';S=R/'paper/submission'
for d in [P/'figures',P/'tables',P/'metadata',E/'figures',E/'tables']:d.mkdir(exist_ok=True)
a=json.loads((P/'verification/SCIENTIFIC_AUDIT.json').read_text());O=a['statistics']['original']['marginal'];N=a['statistics']['revised']['marginal']
s=(S/'manuscript.tex').read_text();body=s[s.index('\\section{Introduction}'):s.index('\\section*{Data, Code, and Declarations}')]
body=body.replace('\\IEEEPARstart{S}{ervice}','Service').replace('\\ArtifactRoot ','')
# Numeric corrections are display rounding from preserved source, not new outcome estimates.
for old,new in {'0.79303':f"{O['twin_informed']['mean']:.5f}",'0.80663':f"{O['no_adaptation']['mean']:.5f}",'0.82740':f"{O['information_equivalent']['mean']:.5f}",'0.82710':f"{O['oracle']['mean']:.5f}",'0.82315':f"{a['switching']['gross_twin']:.5f}",'0.81621':f"{a['switching']['gross_fallback']:.5f}",'0.48180':'0.48183','0.15330':'0.15333','0.00694':f"{a['switching']['gross_gain']:.5f}",'0.74197':f"{N['adaptive_fidelity']['mean']:.5f}",'0.79650':f"{N['fixed_target_fidelity']['mean']:.5f}",'0.62374':f"{N['adaptive_checkpoint']['mean']:.5f}",'0.68232':f"{N['checkpoint_disabled']['mean']:.5f}"}.items():body=body.replace(old,new)
# Scope clarification and stronger actually supported methodological characterization.
needle='First, we provide exact numerical reproduction'
body=body.replace(needle,'First, we preserve the existing exact numerical reproduction')
body=body.replace('Finally, the reproduced extension supplies a separate unfavorable fidelity/checkpoint result and a prospective design for resolving fairness questions.','Finally, separately checked source inequalities explain the linear selector\'s effective behavior, distinguish an exact share optimum from numerical residuals, and define a prospective fair comparison. The reproduced extension retains separate unfavorable fidelity/checkpoint outcomes.')
body=body.replace('Our distinction is the reproducible audit of a specific archived comparison:', 'Our distinction is the reproducible audit and mathematical characterization of a specific archived comparison:')
# Previously abstract/metadata-only Hui interpretation remains bounded, now selected full-text methods inspected.
needle='Our distinction is the reproducible audit and mathematical characterization'
add=r'''
Prediction and transition-aware service placement also precede this study. Wang \emph{et al.}~\cite{wang2016placement} model local and migration costs over a look-ahead window and examine prediction error; their author manuscript reports synthetic and mobility-trace evaluation. Our current-slot rules do not implement that horizon optimization or reproduce its performance. A lower paired utility cannot establish a new switching-control principle. Jain \emph{et al.}~\cite{jain1984fairness} supply the established resource-allocation index used here; fairness of shares is not fairness of delivered task service. These distinctions constrain both algorithmic and methodological novelty.

'''
body=body.replace(needle,add+needle)
# Add rigorous selector characterization adjacent to actual policies.
needle='\\subsection{Actual Baseline Semantics}'
character=r'''
\subsection{Linear-Selector Collapse and Its Domain}
\label{sec:selector}
Let $B_e=0.25\mathbb{1}[\mathrm{strict}]+0.15\mathbb{1}[\ell>40]$ and $P_c=0.002\max\{50-e,0\}$. For a fixed available network $n$, the implemented score differences are
\begin{align}
 S(n,e)-S(n,c)&=0.004\ell\alpha_n(0.15)+B_e+P_c,\label{eq:edgecloud}\\
 S(n,e)-S(n,l)&=0.004\ell\alpha_n(0.25)+B_e.\label{eq:edgelocal}
\end{align}
Here $e,c,l$ in score arguments denote edge, cloud and local placement. If edge is a candidate, $\ell>0$, $\alpha_n>0$, and the frozen nonnegative bonuses and placement coefficients apply, both differences are strictly positive. Edge therefore dominates the other placements. These conditions hold for every full original family; independently inspected revised full-family inputs have the same property. They are sufficient conditions, not a theorem for arbitrary weights, absent edge or all interventions.

Loss adds a constant across selection candidates. An outage removes terrestrial from the generated available set, so its mismatch penalty cannot distinguish the remaining candidates. With edge selected, the argmax minimizes $\alpha_n$ among available paths: terrestrial, otherwise local-edge Wi-Fi if available, otherwise degraded-local. The revised optional offline category has coefficient10 and does not beat these available alternatives. Shares are uniform, power is one; the revised linear action defaults to target fidelity and no checkpoint or recovery. Its linear score has no fidelity-choice dimension. Features change score magnitudes or margins without changing this argmax within the stated domain; latency and loss still affect the subsequently evaluated utility.

A focused source check reconstructs the existing five-seed input sequences for train, held-out and three shifted families, preserving each version's random-draw schedule. It calls only the respective selector and independent simplified-rule/inequality assertions. All 9,600 states in each version pass separately. No per-slot action traces were retained in the original summary files; these are reconstructed-input assertions, not a new held-out performance study or additional seed replicates.

The strict result excludes zero latency. Under \pol{no\_edge\_observations}, $\ell=q=0$. With non-strict class and $e\ge50$, all candidate scores tie. Enumeration starts with cloud and the first available network, and replacement uses strict greater-than, preserving that first candidate. Strictness or low energy can change the remaining ties. A new boundary assertion demonstrates this exception. Historical ablation outputs evaluate the procedural policy; the linear boundary check is not an archived linear-policy ablation outcome. Removing edge also removes the dominance premise.

'''
body=body.replace(needle,character+needle)
# Exact optimizer reasoning with a separately disclosed residual failure.
needle='\\subsection{Linear Selector and Greedy Model Evaluator}'
opt=r'''
For the fixed $B=20$ domain, every feasible coordinate is below99, so $\partial f/\partial x_i=-1/(1+x_i)+0.01<0$. The exact optimum uses the full budget; strict convexity and equal derivatives then yield the unique allocation $x_i^*=20/N_t$. This is an allocation subproblem, not maximization of the reported network/placement utility. Distributing an unchanged sum differently does not improve that primary score.

Numerical output is approximate. The source supplies no tolerance override or post-solve budget projection. A focused component audit checks all existing seed values0--4 and100--104 with each $N=4,\ldots,8$, separately for both versions (100 small solves). Maximum absolute budget residual is $3.1883\times10^{-6}$ and maximum share deviation from the exact optimum is $2.5557\times10^{-6}$; 26 returned sums exceed the budget strictly. The original exact-budget audit assertion failed and its log is retained. SLSQP reports success while the source accepts small residuals; no exact numerical feasibility is asserted.

At fixed discrete/fidelity/checkpoint choices, these source bounds give $|\Delta u|\le1.5942\times10^{-7}$. Bounding the score derivative by1.83 gives absolute slot-score impact below $2.918\times10^{-7}$. The choices themselves do not depend on optimized shares. This conditional bound is too small to reverse either retained negative comparison; it does not repair broader fairness or resource-feasibility limitations. No historical allocation or result is projected or replaced.

'''
body=body.replace(needle,opt+needle)
body=body.replace('Five original and 31 revised tests pass.', 'The recovered historical records report five original and31 revised passing tests; those tests were not re-executed merely to relabel them as new validation. New focused arithmetic checks verify1,477 non-timing summaries, selector assertions and optimizer component behavior; their separate scripts and logs are supplied.')
body=body.replace('Source reruns used Python 3.12.14, NumPy 2.3.5, and SciPy 1.18.1.','The recovered historical source reruns used Python3.12.14, NumPy2.3.5 and SciPy1.18.1. The new focused audit uses an existing Python3.14.7 environment; its exact dependencies are retained separately.')
body=body.replace('All orchestration evidence is synthetic.','All orchestration evidence is synthetic. The present contribution is selected as a methodological investigation, rather than a fair causal demonstration of controller superiority.')
# Replace shifted-family numeric prose directly from retained arrays.
D=json.loads((E/'data/original/rq2_cross_layer_continuity_domain_shift.json').read_text())
def val(f,p):return D[f]['policies'][p]['service_continuity_utility']['mean']
body=re.sub(r'High-blockage procedural utility.*?These descriptive exploratory outcomes establish no uniform superiority \(Fig\.~\\ref\{fig:shift\}\)\.',f"High-blockage procedural utility is {val('high_blockage','twin_informed'):.5f} versus cloud fallback {val('high_blockage','no_adaptation'):.5f}; outage-heavy values are {val('outage_heavy','twin_informed'):.5f} versus {val('outage_heavy','no_adaptation'):.5f}. The linear selector is {val('high_blockage','information_equivalent'):.5f} in high blockage and {val('outage_heavy','information_equivalent'):.5f} in outage heavy. These descriptive exploratory outcomes establish no uniform superiority (Fig.~\\ref{{fig:shift}}).",body,flags=re.S)
body=body.replace('0.79098',f"{val('energy_constrained','twin_informed'):.5f}")
# Explicit paired extension uncertainties, preserving whole-policy confounding.
needle='Both are unfavorable adaptive outcomes.'
z=N # means above; intervals are from independently recomputed paired arrays.
f=a['statistics']['revised']['primary_paired']['adaptive_fidelity minus fixed_target_fidelity'];c=a['statistics']['revised']['primary_paired']['adaptive_checkpoint minus checkpoint_disabled']
body=body.replace(needle,needle+f" Paired differences are {f['mean']:.5f} with95\\% interval $[{f['ci95'][0]:.5f},{f['ci95'][1]:.5f}]$, and {c['mean']:.5f} with interval $[{c['ci95'][0]:.5f},{c['ci95'][1]:.5f}]$, respectively. These additional paired calculations are post hoc.")
body=body.replace('Current selected tests are recorded separately', 'Current selected tests are recorded separately')
body=body.replace('Vertical mark: procedural/cloud-fallback mean crossing. No retuning or retraining.', 'Vertical mark: procedural/cloud-fallback mean crossing. Aggregate score accounting with fixed actions, not a new sweep or measured transition cost.')
body=body.replace('Right: switches per episode. Charges are configured dimensionless penalties.', 'Right: switches per episode. Left vertical axis begins at0.75. Charges are configured dimensionless penalties.')
body=body.replace('The study provides a transparent audit and reproducibility package for this synthetic case, and a prospective path to a fairer, physically meaningful comparison.','The selector characterization and numerical audit further show where apparently rich decisions reduce to coefficient ordering and an approximately uniform subproblem. The study provides a transparent, implementation-specific reproducibility case, with no claim of established novelty, and a prospective path to a fairer comparison.')
body+=r'''
\section*{Data, Code, and Author-Review Declarations}
The separate bounded evidence archive includes both frozen source/protocol vintages, original and reproduced summaries, seeds, checksums, independent mathematical/statistical scripts and environment records. Exact revisions are supplied in \texttt{PROVENANCE.json}; original86a5a6f, revisedb44357f and manuscript/preprint revisions are distinct. No permanent evidence DOI or public preprint record has been created. Repository software MIT notices are retained; article/data permissions and redistribution rights require owner confirmation. Existing journal-review files remain preserved separately.

Confirmed author: Edmund Gunn, Jr.; gunnchOS Research; egunnjr@gunnchos.com; ORCID0009-0004-6546-2609. Additional authors, contributions, funding/sponsors, conflicts, prior dissemination, rights and patent implications are pending. No absence of funding, conflicts or additional authors is presumed. No university affiliation, admission, supervision or endorsement is claimed.

\section*{AI-Use Disclosure}
OpenAI ChatGPT/Codex tools assisted substantively with drafting/revision, source and policy audits, mathematical exposition, independent statistical and assertion code, literature/metadata investigation, programmatic plots, LaTeX and reproducibility packaging. The abstract, Sections1--8, equations, code, figures and declarations include AI-assisted material. Plots are rendered from synthetic artifacts, not invented measured data. The historical tool/model inventory remains subject to owner confirmation. AI systems are not authors. Edmund Gunn, Jr. must verify the complete text, methods, computations, citations and rights before any release. Human scientific approval, originality verification, article-license approval and publication authorization remain false. No independent human peer review is claimed.
'''
title='Controller Semantics and Negative Results in a Synthetic Cross-Layer Orchestration Study'
abstract=r'''Digital-twin context can guide network and computation choices, but a controller label does not establish information equivalence, meaningful optimization, or improved service. This article audits two preserved synthetic cross-layer orchestration implementations and their negative results. In the original five-seed held-out study, the procedural twin policy underperforms cloud fallback by0.01360 utility units, with a post hoc paired95\% interval[-0.01478,-0.01242]. A separately reproduced revision retains a negative difference of0.01255 and unfavorable adaptive-fidelity and checkpoint outcomes. Source inequalities show that the full linear selector chooses edge placement on the available path with the smallest configured penalty; focused assertions confirm this behavior on9,600 reconstructed input states in each version, separately. A zero-latency transformation instead creates score ties. The symmetric SLSQP share objective has a uniform mathematical optimum, but optimizes a different function from network/placement utility and returns small budget residuals. A bounded numerical analysis finds these residuals too small to reverse the retained negative comparisons. Switching accounting, incomplete action feasibility, unequal consumed information/computation, and truth-changing interventions limit causal interpretation. The contribution is a reproducible methodological characterization of this model, with no measured network, general adaptive superiority, or established novelty claim.'''
header=r'''\documentclass[10pt,twocolumn]{article}
\usepackage[a4paper,margin=18mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,amssymb,booktabs,array,graphicx,url}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[hidelinks]{hyperref}
\setlength{\columnsep}{7mm}
\newcommand{\pol}[1]{\texttt{#1}}
\title{TITLE}
\author{Edmund Gunn, Jr.\\gunnchOS Research\\\texttt{egunnjr@gunnchos.com}\\ORCID: \href{https://orcid.org/0009-0004-6546-2609}{0009-0004-6546-2609}}
\date{gunnchOS Research Exploratory Preprint --- v0.1\\Local candidate for scientific review, 14 October 2026\\Not peer reviewed. Not posted. Publication authorization: false.}
\begin{document}
\maketitle
\begin{abstract}
ABSTRACT
\end{abstract}
\noindent\textbf{Keywords:} Cross-layer control, controller semantics, digital twin, switching costs, negative results, reproducibility.
'''.replace('TITLE',title).replace('ABSTRACT',abstract)
# Ensure prose word separation around typed numeric additions.
for old,new in {'by0.':'by 0.','of0.':'of 0.','paired95':'paired 95','interval[':'interval [','on9,600':'on 9,600','below99':'below 99','inputs0':'inputs 0','values0':'values 0','degree4':'degree 4','SciPyquantile':'SciPy quantile','of100':'of 100','bounds100':'bounds 100','report100':'report 100','by1.83':'by 1.83','verify1,477':'verify 1,477','and31':'and 31','begin at0.75':'begin at 0.75','with95':'with 95','Python3.':'Python 3.','NumPy2.':'NumPy 2.','SciPy1.':'SciPy 1.','original86':'original 86','revisedb':'revised b','ORCID0009':'ORCID 0009','Sections1':'Sections 1'}.items():header=header.replace(old,new);body=body.replace(old,new)
(P/'preprint.tex').write_text(header+'\n'+body+'\n\\bibliographystyle{plainnat}\n\\bibliography{references}\n\\end{document}\n')
bib=(S/'references.bib').read_text()+r'''
@misc{wang2016placement,
 author={Shiqiang Wang and Rahul Urgaonkar and Ting He and Kevin Chan and Murtaza Zafer and Kin K. Leung},
 title={Dynamic Service Placement for Mobile Micro-Clouds with Predicted Future Costs},
 year={2016}, note={Author preprint, arXiv:1503.02735v2, September16; journal issue metadata not used here},
 doi={10.48550/arXiv.1503.02735},url={https://arxiv.org/abs/1503.02735}}
@techreport{jain1984fairness,
 author={Rajendra K. Jain and Dah-Ming W. Chiu and William R. Hawe},
 title={A Quantitative Measure of Fairness and Discrimination for Resource Allocation in Shared Computer Systems},
 institution={Digital Equipment Corporation},number={DEC-TR-301},year={1984},month=sep,
 url={https://www.cse.wustl.edu/~jain/papers/fairness.htm},note={Technical report; no DOI assigned in this bibliography}}
'''
(P/'references.bib').write_text(bib)
for f in (S/'figures').glob('*'):shutil.copyfile(f,P/'figures'/f.name);shutil.copyfile(f,E/'figures'/f.name)
for f in (S/'tables').glob('*'):shutil.copyfile(f,P/'tables'/f.name);shutil.copyfile(f,E/'tables'/f.name)
# Portable figure/table generator, reusing sound historical implementation; exclude archive/report sections and hardcoded test counts.
gen=(S/'scripts/build_evidence.py').read_text();start=gen.index('def get(');end=gen.index('# Recursively verify comparable numerical fields')
gen=gen[start:end];gen=gen.replace("P/'evidence'","E/'data'").replace("P/'tables'","TAB").replace("P/'figures'","FIG").replace("P/'audit'","OUT").replace('fixed action trace','fixed actions')
(E/'scripts/generate_figures.py').write_text("from pathlib import Path\nimport json,csv,numpy as np\nfrom scipy.stats import t\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nE=Path(__file__).resolve().parents[1];OUT=E/'generated';FIG=OUT/'figures';TAB=OUT/'tables'\nfor d in [OUT,FIG,TAB]:d.mkdir(parents=True,exist_ok=True)\n"+gen)
(P/'BUILD.sh').write_text('''#!/bin/sh
set -eu
cd "$(dirname "$0")"
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error preprint.tex
bibtex preprint
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error preprint.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error preprint.tex
''')
print('Neutral complete manuscript created from accepted author-review copy; frozen/journal files preserved.')
