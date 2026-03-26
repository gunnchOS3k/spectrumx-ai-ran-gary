"""Quick smoke test for v14 main.py on actual dataset files."""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "submissions" / "leaderboard_v14"))
import main as m  # noqa: E402

base = Path("competition_dataset/files")
user = next(p for p in base.iterdir() if p.is_dir())
vla  = user / "VLA_brutal"
gt   = pd.read_csv(vla / "groundtruth.csv")

correct = 0
total   = 0
for _, row in gt.iterrows():
    fp = vla / row["filename"]
    if not fp.exists():
        fp = (user / "trainingData") / row["filename"]
    if not fp.exists():
        continue
    pred  = m.evaluate(str(fp))
    label = int(row["label"])
    ok    = pred == label
    correct += ok
    total   += 1
    status = "OK" if ok else "WRONG"
    print(f"{row['filename']}: pred={pred} label={label} [{status}]")

print()
print(f"Accuracy on {total} labeled files: {correct}/{total} = {correct / total:.4f}")
