from pathlib import Path
import yaml

DIR = Path(__file__).resolve().parents[2] / "configs" / "campus_ai_ran_profiles"


def load_profile(site_id: str) -> dict:
    with (DIR / f"{site_id}.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)
