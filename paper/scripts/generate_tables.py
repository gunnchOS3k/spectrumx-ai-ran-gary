#!/usr/bin/env python3
"""Generate Paper II tables/figures from experiment JSON. RESULT_PENDING if missing."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "paper" / "tables"
FIGS = ROOT / "paper" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
EXP = ROOT / "results" / "experiments"
HELD = EXP / "rq2_cross_layer_continuity_heldout.json"
TRAIN = EXP / "rq2_cross_layer_continuity_train.json"
SHIFT = EXP / "rq2_cross_layer_continuity_domain_shift.json"
ABL = EXP / "rq2_cross_layer_continuity_ablation.json"
TOY_CANDIDATES = [
    ROOT / "results" / "e2e" / "airan_toy.json",
    ROOT / "results" / "e2e" / "airan_policy_metrics.json",
    ROOT / "results" / "demo_airan_policy.json",
]


def pending(path: Path, msg: str) -> None:
    path.write_text(f"\\textbf{{RESULT\\_PENDING.}} {msg}\\par\n", encoding="utf-8")
    print("RESULT_PENDING", path)


def fmt(x: object, nd: int = 4) -> str:
    if isinstance(x, float):
        if x != x:  # nan
            return "nan"
        return f"{x:.{nd}f}"
    return str(x)


def write_policy_table(src: Path, stem: str, caption: str) -> None:
    tex = TABLES / f"{stem}.tex"
    md = TABLES / f"{stem}.md"
    csv_path = TABLES / f"{stem}.csv"
    if not src.exists():
        pending(tex, f"Missing {src.name}. Run scripts/run_paper_ii_heldout.py.")
        md.write_text(f"**RESULT_PENDING.** Missing `{src}`.\n", encoding="utf-8")
        return
    data = json.loads(src.read_text(encoding="utf-8"))
    policies = data.get("policies") or {}
    effects = data.get("effect_sizes") or {}
    rows = []
    for name, block in policies.items():
        u = block.get("service_continuity_utility") or {}
        lat = block.get("predicted_latency_ms") or {}
        sw = block.get("switch_cost") or {}
        cpu = block.get("compute_time_ms") or {}
        d = (effects.get(name) or {}).get("cohens_d_vs_no_adaptation_continuity", "")
        rows.append(
            {
                "policy": name,
                "continuity_mean": u.get("mean"),
                "continuity_ci_low": u.get("ci_low"),
                "continuity_ci_high": u.get("ci_high"),
                "latency_ms_mean": lat.get("mean"),
                "switch_cost_mean": sw.get("mean"),
                "compute_time_ms_mean": cpu.get("mean"),
                "cohens_d_vs_no_adaptation": d,
            }
        )
    fieldnames = list(rows[0].keys()) if rows else ["policy"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    md_lines = [
        f"# {stem} ({data.get('evidence_class', 'SYNTHETIC_SIM')})",
        "",
        f"Family `{data.get('family')}`; seeds `{data.get('seeds')}`. Timing is `{data.get('latency_class')}`.",
        "",
        "| policy | continuity mean [95% CI] | latency_ms | switch_cost | compute_time_ms | Cohen's d vs no_adaptation |",
        "|---|---|---|---|---|---|",
    ]
    tex_rows = []
    for r in rows:
        ci = f"{fmt(r['continuity_mean'])} [{fmt(r['continuity_ci_low'])}, {fmt(r['continuity_ci_high'])}]"
        md_lines.append(
            f"| `{r['policy']}` | {ci} | {fmt(r['latency_ms_mean'])} | {fmt(r['switch_cost_mean'])} | {fmt(r['compute_time_ms_mean'])} | {fmt(r['cohens_d_vs_no_adaptation'])} |"
        )
        tex_rows.append(
            f"{r['policy'].replace('_', '\\_')} & {fmt(r['continuity_mean'])} & {fmt(r['continuity_ci_low'])} & {fmt(r['continuity_ci_high'])} & {fmt(r['latency_ms_mean'])} & {fmt(r['switch_cost_mean'])} & {fmt(r['compute_time_ms_mean'])} & {fmt(r['cohens_d_vs_no_adaptation'])} \\\\"
        )
    md_lines.append("")
    md.write_text("\n".join(md_lines), encoding="utf-8")
    body = "\n".join(tex_rows) or "\\multicolumn{8}{c}{RESULT\\_PENDING} \\\\"
    tex.write_text(
        "\\begin{table}[h]\\centering\n"
        f"\\caption{{{caption}}}\n"
        "\\begin{tabular}{lrrrrrrr}\\toprule\n"
        "policy & mean $U$ & CI low & CI high & lat (ms) & switch cost & compute ms & $d$ vs no-adapt \\\\\n"
        f"\\midrule\n{body}\n\\bottomrule\\end{{tabular}}\n"
        "\\end{table}\n",
        encoding="utf-8",
    )
    print("wrote", tex, csv_path, md)


def write_shift_table() -> None:
    tex = TABLES / "rq2_domain_shift.tex"
    md = TABLES / "rq2_domain_shift.md"
    csv_path = TABLES / "rq2_domain_shift.csv"
    if not SHIFT.exists():
        pending(tex, "Missing domain-shift JSON.")
        md.write_text("**RESULT_PENDING.** Missing domain-shift JSON.\n", encoding="utf-8")
        return
    data = json.loads(SHIFT.read_text(encoding="utf-8"))
    rows = []
    for fam, block in data.items():
        policies = block.get("policies") or {}
        for name in ("no_adaptation", "rule_based", "twin_informed", "information_equivalent", "oracle"):
            if name not in policies:
                continue
            u = policies[name].get("service_continuity_utility") or {}
            rows.append({"family": fam, "policy": name, "continuity_mean": u.get("mean"), "ci_low": u.get("ci_low"), "ci_high": u.get("ci_high")})
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["family", "policy", "continuity_mean", "ci_low", "ci_high"])
        w.writeheader()
        w.writerows(rows)
    md_lines = ["# rq2_domain_shift (SYNTHETIC_SIM)", "", "| family | policy | continuity mean [95% CI] |", "|---|---|---|"]
    tex_rows = []
    for r in rows:
        md_lines.append(f"| `{r['family']}` | `{r['policy']}` | {fmt(r['continuity_mean'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] |")
        tex_rows.append(
            f"{str(r['family']).replace('_', '\\_')} & {str(r['policy']).replace('_', '\\_')} & {fmt(r['continuity_mean'])} & {fmt(r['ci_low'])} & {fmt(r['ci_high'])} \\\\"
        )
    md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    tex.write_text(
        "\\begin{table}[h]\\centering\n"
        "\\caption{Domain-shift continuity (SYNTHETIC\\_SIM; held-out seeds).}\n"
        "\\begin{tabular}{llrrr}\\toprule family & policy & mean $U$ & CI low & CI high \\\\\\midrule\n"
        + "\n".join(tex_rows)
        + "\n\\bottomrule\\end{tabular}\\end{table}\n",
        encoding="utf-8",
    )
    print("wrote", tex)


def write_ablation_table() -> None:
    tex = TABLES / "rq2_ablations.tex"
    md = TABLES / "rq2_ablations.md"
    csv_path = TABLES / "rq2_ablations.csv"
    if not ABL.exists():
        pending(tex, "Missing ablation JSON.")
        md.write_text("**RESULT_PENDING.** Missing ablation JSON.\n", encoding="utf-8")
        return
    data = json.loads(ABL.read_text(encoding="utf-8"))
    rows = []
    for name, block in data.items():
        pol = (block.get("policies") or {}).get("twin_informed") or {}
        u = pol.get("service_continuity_utility") or {}
        sw = pol.get("switch_cost") or {}
        rows.append({"ablation": name, "continuity_mean": u.get("mean"), "ci_low": u.get("ci_low"), "ci_high": u.get("ci_high"), "switch_cost_mean": sw.get("mean")})
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ablation", "continuity_mean", "ci_low", "ci_high", "switch_cost_mean"])
        w.writeheader()
        w.writerows(rows)
    md_lines = ["# rq2_ablations (SYNTHETIC_SIM)", "", "| ablation | continuity mean [95% CI] | switch_cost |", "|---|---|---|"]
    tex_rows = []
    for r in rows:
        md_lines.append(f"| `{r['ablation']}` | {fmt(r['continuity_mean'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] | {fmt(r['switch_cost_mean'])} |")
        tex_rows.append(
            f"{str(r['ablation']).replace('_', '\\_')} & {fmt(r['continuity_mean'])} & {fmt(r['ci_low'])} & {fmt(r['ci_high'])} & {fmt(r['switch_cost_mean'])} \\\\"
        )
    md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    tex.write_text(
        "\\begin{table}[h]\\centering\n"
        "\\caption{Twin-informed ablations on held-out seeds (SYNTHETIC\\_SIM).}\n"
        "\\begin{tabular}{lrrrr}\\toprule ablation & mean $U$ & CI low & CI high & switch cost \\\\\\midrule\n"
        + "\n".join(tex_rows)
        + "\n\\bottomrule\\end{tabular}\\end{table}\n",
        encoding="utf-8",
    )
    print("wrote", tex)


def write_toy() -> None:
    dest = TABLES / "rq2_toy.tex"
    src = next((p for p in TOY_CANDIDATES if p.exists()), None)
    if src is None:
        pending(dest, "Run \\texttt{python3 scripts/demo\\_airan\\_policy.py --toy} first.")
        return
    data = json.loads(src.read_text(encoding="utf-8"))
    dest.write_text(
        "\\begin{table}[h]\\centering\n"
        "\\caption{Legacy toy policy snapshot (SYNTHETIC\\_SIM; not competition IQ).}\n"
        "\\begin{tabular}{ll}\\toprule field & value \\\\\\midrule\n"
        f"mode & {data.get('mode')} \\\\\nseed & {data.get('seed')} \\\\\n"
        f"note & {data.get('note', 'toy')} \\\\\\bottomrule\\end{{tabular}}\\end{{table}}\n",
        encoding="utf-8",
    )
    print("wrote", dest, "from", src)


def write_figures() -> None:
    banner = FIGS / "README.md"
    png = FIGS / "rq2_heldout_continuity.png"
    if not HELD.exists():
        banner.write_text(
            "**RESULT_PENDING.** Held-out JSON missing; no figure generated.\n",
            encoding="utf-8",
        )
        print("RESULT_PENDING figure")
        return
    data = json.loads(HELD.read_text(encoding="utf-8"))
    names = []
    means = []
    lows = []
    highs = []
    for name, block in (data.get("policies") or {}).items():
        u = block.get("service_continuity_utility") or {}
        if u.get("mean") is None:
            continue
        names.append(name)
        means.append(float(u["mean"]))
        lows.append(float(u.get("ci_low", u["mean"])))
        highs.append(float(u.get("ci_high", u["mean"])))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        banner.write_text(
            "**RESULT_PENDING.** matplotlib not importable; CSV/TeX tables still generated from JSON.\n",
            encoding="utf-8",
        )
        print("RESULT_PENDING matplotlib")
        return
    yerr = [[m - lo for m, lo in zip(means, lows)], [hi - m for m, hi in zip(means, highs)]]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(range(len(names)), means, yerr=yerr, capsize=3, color="#4C78A8")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_ylabel("service_continuity_utility")
    ax.set_title("Held-out continuity (SYNTHETIC_SIM; 95% t CI over seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    banner.write_text(
        "Figures are generated from `results/experiments/*.json`.\n\n"
        f"- `{png.name}`: held-out continuity means with 95% t CIs. SYNTHETIC_SIM. Not RF.\n",
        encoding="utf-8",
    )
    print("wrote", png)


def main() -> int:
    write_toy()
    write_policy_table(
        TRAIN,
        "rq2_train_policies",
        "Train-family policy continuity (SYNTHETIC\\_SIM; t CI over train seeds). Not competition IQ.",
    )
    write_policy_table(
        HELD,
        "rq2_heldout_policies",
        "Held-out policy continuity (SYNTHETIC\\_SIM; t CI over held-out seeds). Not competition IQ.",
    )
    write_shift_table()
    write_ablation_table()
    write_figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
