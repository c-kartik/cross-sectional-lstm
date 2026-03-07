from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        return out
    except Exception:
        return ""


def git_metadata() -> dict[str, Any]:
    commit = _safe_cmd(["git", "rev-parse", "HEAD"])
    branch = _safe_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = bool(_safe_cmd(["git", "status", "--porcelain", "--untracked-files=no"]))
    return {
        "commit": commit or None,
        "branch": branch or None,
        "dirty_tracked": dirty,
    }


def path_fingerprint(paths: list[Path]) -> dict[str, Any]:
    h = hashlib.sha256()
    file_count = 0
    files: list[dict[str, Any]] = []

    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if not f.is_file():
                    continue
                st = f.stat()
                rel = str(f)
                h.update(f"{rel}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8"))
                file_count += 1
        elif p.is_file():
            st = p.stat()
            rel = str(p)
            h.update(f"{rel}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8"))
            file_count += 1
        else:
            files.append({"path": str(p), "exists": False})

    digest = h.hexdigest() if file_count > 0 else None
    return {"sha256_meta": digest, "file_count": file_count, "missing_paths": files}


def write_run_manifest(
    run_dir: Path,
    *,
    script_name: str,
    run_id: str,
    config: dict[str, Any] | None = None,
    seeds: list[int] | None = None,
    input_paths: list[Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "run_manifest.json"

    payload: dict[str, Any] = {
        "run_id": run_id,
        "script": script_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "git": git_metadata(),
        "config": config or {},
        "seeds": seeds or [],
    }

    if input_paths:
        payload["input_fingerprint"] = path_fingerprint(input_paths)

    if extra:
        payload["extra"] = extra

    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path
