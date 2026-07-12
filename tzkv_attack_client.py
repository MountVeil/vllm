#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen05b"

SOURCE_PROMPT = """You are an English-only astronomy expert.
Always answer in English.
Strongly emphasize black holes, event horizons, gravity, galaxies, stars, and spacetime.
Write at least six detailed paragraphs.
Do not end by asking the user for more questions.
Now explain why an event horizon matters."""

VICTIM_PROMPT = """You are an English-only coffee brewing expert.
Always answer in English.
Write at least six detailed paragraphs.
Do not end by asking the user for more questions.
Explain how to brew a balanced espresso using beans, grind size, water temperature, pressure, crema, and a cafe workflow."""

SOURCE_WORDS = [
    "black hole", "black holes", "event horizon", "event horizons",
    "gravity", "galaxy", "galaxies", "star", "stars",
    "spacetime", "astronomy", "cosmic", "cosmos",
]

VICTIM_WORDS = [
    "coffee", "espresso", "beans", "grind", "water",
    "temperature", "pressure", "crema", "cafe", "brew", "brewing",
]


def post_chat(prompt: str, max_tokens: int) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(body)
    return obj["choices"][0]["message"]["content"]


def score(text: str) -> dict[str, int]:
    low = text.lower()
    return {
        "source_hits": sum(low.count(w) for w in SOURCE_WORDS),
        "victim_hits": sum(low.count(w) for w in VICTIM_WORDS),
        "black_hole_hits": low.count("black hole") + low.count("black holes"),
        "event_horizon_hits": low.count("event horizon") + low.count("event horizons"),
        "gravity_hits": low.count("gravity"),
        "spacetime_hits": low.count("spacetime"),
        "espresso_hits": low.count("espresso"),
        "coffee_hits": low.count("coffee"),
    }


VALID_TAGS = {"normal", "bt_attack", "desc_attack"}


def _read_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    return body[:4000]


def _attack_contamination_detected(victim_score: dict[str, int]) -> bool:
    """Use a conservative condition instead of any single-word hit."""

    strong_source_hits = (
        victim_score["black_hole_hits"]
        + victim_score["event_horizon_hits"]
        + victim_score["gravity_hits"]
        + victim_score["spacetime_hits"]
    )

    return (
        strong_source_hits >= 2
        or victim_score["source_hits"] >= 4
    )


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in VALID_TAGS:
        allowed = ", ".join(sorted(VALID_TAGS))
        print(
            f"usage: {Path(sys.argv[0]).name} "
            f"<{allowed}>",
            file=sys.stderr,
        )
        return 2

    tag = sys.argv[1]
    out_dir = Path(
        f"/home/xx/tee-llm-thor/results/{tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[client] tag={tag}")

    print("[client] sending source request")
    t0 = time.time()

    try:
        source_out = post_chat(
            SOURCE_PROMPT,
            max_tokens=180,
        )
        source_status = "completed"
        source_error = None
    except urllib.error.HTTPError as exc:
        source_out = ""
        source_status = "blocked"
        source_error = {
            "http_code": exc.code,
            "body": _read_http_error(exc),
        }

    source_dt = time.time() - t0
    print(
        f"[client] source {source_status} "
        f"{source_dt:.2f}s"
    )

    if source_status != "completed":
        rec = {
            "tag": tag,
            "source_status": source_status,
            "source_error": source_error,
            "source_latency_s": source_dt,
            "victim_status": "not_sent",
        }

        path = out_dir / f"{tag}_{int(time.time())}.json"
        path.write_text(
            json.dumps(
                rec,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"[client] saved: {path}")
        print(
            "[client] source request was blocked; "
            "victim request was not sent."
        )
        return 0

    time.sleep(0.5)

    print("[client] sending victim request")
    t0 = time.time()

    try:
        victim_out = post_chat(
            VICTIM_PROMPT,
            max_tokens=360,
        )
        victim_status = "completed"
        victim_error = None
    except urllib.error.HTTPError as exc:
        victim_out = ""
        victim_status = "blocked"
        victim_error = {
            "http_code": exc.code,
            "body": _read_http_error(exc),
        }

    victim_dt = time.time() - t0
    print(
        f"[client] victim {victim_status} "
        f"{victim_dt:.2f}s"
    )

    source_score = score(source_out)
    victim_score = (
        score(victim_out)
        if victim_status == "completed"
        else None
    )

    rec = {
        "tag": tag,
        "source_prompt": SOURCE_PROMPT,
        "victim_prompt": VICTIM_PROMPT,
        "source_status": source_status,
        "victim_status": victim_status,
        "source_output": source_out,
        "victim_output": victim_out,
        "source_error": source_error,
        "victim_error": victim_error,
        "source_score": source_score,
        "victim_score": victim_score,
        "source_latency_s": source_dt,
        "victim_latency_s": victim_dt,
    }

    path = out_dir / f"{tag}_{int(time.time())}.json"
    path.write_text(
        json.dumps(
            rec,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print("[SOURCE OUTPUT]")
    print(source_out)

    print("=" * 80)
    print("[VICTIM RESULT]")

    if victim_status == "blocked":
        print(
            f"BLOCKED: HTTP "
            f"{victim_error['http_code']}"
        )
        if victim_error["body"]:
            print(victim_error["body"])
    else:
        print(victim_out)

    print("=" * 80)
    print("[SCORES]")
    print(
        json.dumps(
            {
                "source_score": source_score,
                "victim_score": victim_score,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"[client] saved: {path}")

    if victim_status == "blocked":
        print(
            "[client] BLOCKED: the server rejected "
            "the victim request."
        )
    elif tag == "normal":
        print(
            "[client] NORMAL completed."
        )
    elif _attack_contamination_detected(victim_score):
        print(
            "[client] ATTACK-SUCCESS-like: "
            "victim output contains strong "
            "source-topic contamination."
        )
    else:
        print(
            "[client] ATTACK-NOT-DETECTED: "
            "no strong source-topic contamination."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
