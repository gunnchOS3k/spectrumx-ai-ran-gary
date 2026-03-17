"""Build leaderboard_v6.zip in the repo root."""
from pathlib import Path
import zipfile

here = Path(__file__).resolve().parent
root = here.parents[1]
out = root / "leaderboard_v6.zip"

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(here / "main.py", "main.py")
    zf.write(here / "user_reqs.txt", "user_reqs.txt")

print(f"Created {out}  ({out.stat().st_size} bytes)")
