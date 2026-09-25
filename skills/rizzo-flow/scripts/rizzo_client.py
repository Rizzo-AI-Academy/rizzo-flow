#!/usr/bin/env python3
"""Minimal Rizzo Flow client, standard library only.

    python rizzo_client.py request.json              # native API, compact summary
    python rizzo_client.py request.json --raw        # full JSON response
    python rizzo_client.py request.json --systemone  # Jev/TypeSafe-compatible endpoint
    cat request.json | python rizzo_client.py -      # read the request from stdin
    python rizzo_client.py --health                  # is the server up?

The base URL defaults to $RIZZO_URL or http://127.0.0.1:8017; $RIZZO_API_KEY is sent as a bearer.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def call(url, body=None, key=None, timeout=300):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    request = urllib.request.Request(url, data, headers, method="GET" if data is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        sys.exit(f"HTTP {error.code}: {detail}")
    except urllib.error.URLError as error:
        sys.exit(
            f"Rizzo Flow is not reachable at {url} ({error.reason}). "
            "Start it from the repository with `uv run rizzo serve`."
        )


def summarize_native(response):
    for key, answer in response["answers"].items():
        value = next(answer[f] for f in ("choice", "score", "value") if f in answer)
        top = answer["uncertainty"]["top_probability"]
        ranked = sorted(answer["probabilities"].items(), key=lambda item: -item[1])[:3]
        shown = ", ".join(f"{name}={p:.3f}" for name, p in ranked)
        print(f"{key}: {value!r}  [{answer['status']}, top {top:.3f}]  {shown}")
    print(f"total {response['timing']['total_seconds'] * 1000:.0f} ms")


def summarize_systemone(response):
    for key, answer in response["answers"].items():
        kind = answer["type"]
        value = answer["noul"] if kind == "noul" else answer[kind]
        extra = f"  confidence {answer['confidence']:.3f}" if "confidence" in answer else ""
        print(f"{key}: {value!r}{extra}")
    print(f"model {response['model']}, input tokens {response['usage']['input_tokens']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("request", nargs="?", help="JSON file, or - for stdin")
    parser.add_argument("--url", default=os.environ.get("RIZZO_URL", "http://127.0.0.1:8017"))
    parser.add_argument("--systemone", action="store_true", help="use POST /v1/systemone")
    parser.add_argument("--raw", action="store_true", help="print the full JSON response")
    parser.add_argument("--health", action="store_true", help="only check GET /health")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    key = os.environ.get("RIZZO_API_KEY")
    if args.health:
        health = call(f"{base}/health", key=key, timeout=10)
        model = health.get("model", {})
        print(
            f"{health['status']}: {model.get('source')} {model.get('precision')} "
            f"weights={model.get('weights', 'base')} on {model.get('backend')}"
        )
        return
    if not args.request:
        parser.error("a request file (or -) is required")
    if args.request == "-":
        body = json.load(sys.stdin)
    else:
        with open(args.request, encoding="utf-8") as source:
            body = json.load(source)
    if args.systemone:
        body.setdefault("model", "rizzo-latest")
    route = "/v1/systemone" if args.systemone else "/v1/decisions"
    response = call(base + route, body, key)
    if args.raw:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    elif args.systemone:
        summarize_systemone(response)
    else:
        summarize_native(response)


if __name__ == "__main__":
    main()
