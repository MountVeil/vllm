#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
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


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    out_dir = Path(f"/home/xx/tee-llm-thor/results/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[client] tag={tag}")

    print("[client] sending source request")
    t0 = time.time()
    source_out = post_chat(SOURCE_PROMPT, max_tokens=180)
    source_dt = time.time() - t0
    print(f"[client] source done {source_dt:.2f}s")

    time.sleep(0.5)

    print("[client] sending victim request")
    t0 = time.time()
    victim_out = post_chat(VICTIM_PROMPT, max_tokens=360)
    victim_dt = time.time() - t0
    print(f"[client] victim done {victim_dt:.2f}s")

    rec = {
        "tag": tag,
        "source_prompt": SOURCE_PROMPT,
        "victim_prompt": VICTIM_PROMPT,
        "source_output": source_out,
        "victim_output": victim_out,
        "source_score": score(source_out),
        "victim_score": score(victim_out),
        "source_latency_s": source_dt,
        "victim_latency_s": victim_dt,
    }

    path = out_dir / f"{tag}_{int(time.time())}.json"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("[SOURCE OUTPUT]")
    print(source_out)
    print("=" * 80)
    print("[VICTIM OUTPUT]")
    print(victim_out)
    print("=" * 80)
    print("[SCORES]")
    print(json.dumps({
        "source_score": rec["source_score"],
        "victim_score": rec["victim_score"],
    }, ensure_ascii=False, indent=2))
    print(f"[client] saved: {path}")

    if rec["victim_score"]["source_hits"] > 0:
        print("[client] PASS-like: victim output contains source-topic terms.")
    else:
        print("[client] no source-topic hit detected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
