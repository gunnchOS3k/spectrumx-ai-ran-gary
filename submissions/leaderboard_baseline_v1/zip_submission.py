"""
Helper script to build the leaderboard submission zip.

Creates `leaderboard_baseline_v1.zip` in the repo root containing:
    - main.py
    - user_reqs.txt
    - README_submission.md
"""

from __future__ import annotations

import zipfile
from pathlib import Path


def build_zip() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    submission_dir = repo_root / "submissions" / "leaderboard_baseline_v1"
    zip_path = repo_root / "leaderboard_baseline_v1.zip"

    files_to_include = [
        submission_dir / "main.py",
        submission_dir / "user_reqs.txt",
        submission_dir / "README_submission.md",
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files_to_include:
            if file_path.is_file():
                # Store paths at the top level of the zip
                zf.write(file_path, arcname=file_path.name)

    return zip_path


if __name__ == "__main__":
    result = build_zip()
    print(f"Created submission archive: {result}")

