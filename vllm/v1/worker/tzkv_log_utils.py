# SPDX-License-Identifier: Apache-2.0
# TZKV research attack log/env utilities.

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in (
        "1", "true", "yes", "on"
    )


def _env_bool(name: str, default: str = "0") -> bool:
    # Compatibility with older attack hooks.
    return env_bool(name, default)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    # Compatibility with older attack hooks.
    return env_int(name, default)


def now_ts() -> float:
    return time.time()


def _now() -> float:
    # Compatibility with older attack hooks.
    return now_ts()


def get_log_path() -> Path:
    return Path(os.environ.get(
        "TZKV_LOG",
        os.environ.get(
            "TZKV_ATTACK_LOG",
            "/home/xx/tee-llm-thor/logs/tzkv_vllm_attacks.jsonl",
        ),
    ))


def write_event(event: str, fields: dict[str, Any] | None = None) -> None:
    rec: dict[str, Any] = {
        "ts": now_ts(),
        "event": event,
        "pid": os.getpid(),
    }

    if fields:
        clean: dict[str, Any] = {}
        for k, v in fields.items():
            try:
                json.dumps(v)
                clean[k] = v
            except Exception:
                clean[k] = str(v)
        rec.update(clean)

    path = get_log_path()

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # Logging must never break inference.
        pass


def log_event(event: str, fields: dict[str, Any] | None = None) -> None:
    # Compatibility with older attack hooks.
    write_event(event, fields)

# ---- Compatibility helpers for older TZKV attack hooks ----

def tzkv_log_enabled() -> bool:
    return env_bool("TZKV_LOG_ENABLE", "1")


def tzkv_event_enabled() -> bool:
    return env_bool("TZKV_EVENT_ENABLE", "1")


def tzkv_debug_enabled() -> bool:
    return env_bool("TZKV_DEBUG", "0")


def tzkv_should_log() -> bool:
    return tzkv_log_enabled()


def tzkv_log(event: str, fields: dict[str, Any] | None = None) -> None:
    write_event(event, fields)


def tzkv_log_event(event: str, fields: dict[str, Any] | None = None) -> None:
    write_event(event, fields)


def append_jsonl(path: str | Path, rec: dict[str, Any]) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
