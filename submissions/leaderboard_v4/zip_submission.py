"""Create a zip submission for leaderboard v4."""

from __future__ import annotations

from pathlib import Path
import zipfile


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    submission_dir = repo_root / "submissions" / "leaderboard_v4"
    zip_path = repo_root / "submissions" / "leaderboard_v4.zip"

    files = [
        submission_dir / "main.py",
        submission_dir / "user_reqs.txt",
    ]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)

    print(f"Created zip: {zip_path}")


if __name__ == "__main__":
    main()
