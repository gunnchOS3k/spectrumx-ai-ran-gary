#!/usr/bin/env python3
"""
Probe environment for NGC / Docker readiness. **Never prints or stores API key values.**

Required for NGC probes (optional): set environment variables (names only documented here):
  - NGC_API_KEY_AODT
  - NGC_API_KEY_AERIAL_RAN

Writes: data/aerial_omniverse/access_summary.json (boolean presence + probe outcomes only).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "aerial_omniverse"
OUT_FILE = OUT_DIR / "access_summary.json"
NGC_PROBE_URL = "https://catalog.ngc.nvidia.com/api/rest/resources?size=1"


def _docker_usable() -> bool:
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return True
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False


def _ngc_probe(api_key: str) -> str:
    if not api_key:
        return "skipped_no_key"
    try:
        req = Request(
            NGC_PROBE_URL,
            headers={"Authorization": f"Api-Key {api_key}"},
            method="GET",
        )
        with urlopen(req, timeout=25) as resp:
            return "ok" if 200 <= resp.status < 300 else f"http_{resp.status}"
    except HTTPError as e:
        return f"http_{e.code}"
    except URLError as e:
        return f"url_error:{type(e.reason).__name__ if e.reason else 'unknown'}"
    except Exception as e:
        return f"error:{type(e).__name__}"


def build_summary() -> Dict[str, Any]:
    key_aodt = os.environ.get("NGC_API_KEY_AODT", "").strip()
    key_ran = os.environ.get("NGC_API_KEY_AERIAL_RAN", "").strip()

    return {
        "access_summary_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "env_presence": {
            "NGC_API_KEY_AODT": bool(key_aodt),
            "NGC_API_KEY_AERIAL_RAN": bool(key_ran),
            "AWS_ACCESS_KEY_ID": bool(os.environ.get("AWS_ACCESS_KEY_ID", "").strip()),
            "AZURE_CLIENT_ID": bool(os.environ.get("AZURE_CLIENT_ID", "").strip()),
        },
        "checks": {
            "docker_usable": _docker_usable(),
            "ngc_catalog_probe_AODT_key": _ngc_probe(key_aodt),
            "ngc_catalog_probe_AERIAL_RAN_key": _ngc_probe(key_ran),
        },
        "notes": (
            "This file intentionally excludes secret values. "
            "Set NGC_API_KEY_* in the environment; never commit keys to the repo."
        ),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    OUT_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_FILE.relative_to(REPO_ROOT)}")
    if not summary["env_presence"]["NGC_API_KEY_AODT"] and not summary["env_presence"]["NGC_API_KEY_AERIAL_RAN"]:
        print(
            "Note: neither NGC_API_KEY_AODT nor NGC_API_KEY_AERIAL_RAN is set — NGC probes were skipped.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
