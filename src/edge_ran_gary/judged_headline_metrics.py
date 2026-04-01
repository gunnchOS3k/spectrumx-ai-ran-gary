"""
Single source for judge-facing headline metrics shown in Streamlit.

Values come only from ``submissions/submission_metrics.csv`` (and the same row
logic as the app’s ``_pick_core_submission_row`` / active package row).  If a
field is missing, the UI shows an explicit placeholder — never invented numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


PLACEHOLDER = "—"
SOURCE_HINT = "Add row to `submissions/submission_metrics.csv`"


def _clean_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "—"}:
        return None
    return s


@dataclass(frozen=True)
class JudgedHeadlineBundle:
    """Headline strip for SpectrumX judges; each *_verified flag reflects CSV-backed data."""

    package: str
    package_verified: bool
    leaderboard_rank: str
    rank_verified: bool
    leaderboard_accuracy: str
    accuracy_verified: bool
    runtime_s: str
    runtime_verified: bool
    model_family: str
    model_verified: bool
    novelty_one_liner: str
    viz_why_one_liner: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "package_verified": self.package_verified,
            "leaderboard_rank": self.leaderboard_rank,
            "rank_verified": self.rank_verified,
            "leaderboard_accuracy": self.leaderboard_accuracy,
            "accuracy_verified": self.accuracy_verified,
            "runtime_s": self.runtime_s,
            "runtime_verified": self.runtime_verified,
            "model_family": self.model_family,
            "model_verified": self.model_verified,
        }


def build_judged_headlines(
    core_row: Optional[Mapping[str, Any]],
    active_pkg: Optional[str],
) -> JudgedHeadlineBundle:
    """
    Merge ``core_row`` (authoritative CSV pick) with optional ``active_pkg`` for package name.

    Numeric/report fields still come only from ``core_row`` when present.
    """
    row = dict(core_row or {})
    pkg = _clean_str(active_pkg) or _clean_str(row.get("submission")) or _clean_str(row.get("folder"))
    pkg_verified = pkg is not None

    def field(key: str, *aliases: str) -> tuple[str, bool]:
        for k in (key,) + aliases:
            if k in row:
                v = _clean_str(row.get(k))
                if v is not None:
                    return v, True
        return PLACEHOLDER, False

    rank, rank_v = field("leaderboard_rank", "rank", "lb_rank")
    acc, acc_v = field("leaderboard_accuracy", "accuracy", "lb_accuracy", "score")
    run, run_v = field("runtime", "runtime_per_sample", "runtime_sec")
    model, model_v = field("model_family", "model_class", "family")

    novelty = (
        "Compact, interpretable feature design with a fixed inference contract suitable for embedded-style spectrum sensing."
        if model_v
        else "Interpretable feature-based occupancy detection with a stable `evaluate(iq)` contract (see submission `main.py`)."
    )
    viz_why = (
        "Time–frequency views and scenario-linked panels make detector behavior auditable for judges without exposing raw competition IQ in-cloud."
    )

    return JudgedHeadlineBundle(
        package=pkg or PLACEHOLDER,
        package_verified=pkg_verified,
        leaderboard_rank=rank,
        rank_verified=rank_v,
        leaderboard_accuracy=acc,
        accuracy_verified=acc_v,
        runtime_s=run,
        runtime_verified=run_v,
        model_family=model,
        model_verified=model_v,
        novelty_one_liner=novelty,
        viz_why_one_liner=viz_why,
    )


def submission_metrics_path(repo_root: Path) -> Path:
    return repo_root / "submissions" / "submission_metrics.csv"


def headline_placeholder(active_pkg: Optional[str] = None) -> JudgedHeadlineBundle:
    """When CSV parsing is unavailable — still show structured placeholders."""
    return JudgedHeadlineBundle(
        package=_clean_str(active_pkg) or PLACEHOLDER,
        package_verified=bool(_clean_str(active_pkg)),
        leaderboard_rank=PLACEHOLDER,
        rank_verified=False,
        leaderboard_accuracy=PLACEHOLDER,
        accuracy_verified=False,
        runtime_s=PLACEHOLDER,
        runtime_verified=False,
        model_family=PLACEHOLDER,
        model_verified=False,
        novelty_one_liner=(
            "Interpretable feature-based occupancy detection with a stable `evaluate(iq)` contract "
            "(populate submission_metrics.csv for headline numbers)."
        ),
        viz_why_one_liner=(
            "Time–frequency microscope panels support visualization judging without shipping raw competition IQ in-cloud."
        ),
    )
