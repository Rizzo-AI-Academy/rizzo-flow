#!/usr/bin/env python3
"""Rizzo Flow skill selector: picks only the skills that fit a prompt. Stdlib only.

Usage:
    python router/select_skills.py "testo del prompt"     # stampa le skill scelte in JSON
    python router/select_skills.py --list                 # catalogo delle skill trovate
    python router/select_skills.py --hook                 # hook UserPromptSubmit (stdin JSON)

Catalogo: ~/.claude/skills/*/SKILL.md, <cwd>/.claude/skills/*/SKILL.md, plugin installati
(~/.claude/plugins/cache/**/SKILL.md) e, opzionale, RIZZO_SKILLS_EXTRA (JSON: [{"name","description"}]).

Come decide: le skill sono divise in gruppi di al massimo 24; per ogni gruppo una domanda `choice`
"quale skill serve?" con l'opzione extra "nessuna". Tutte le domande condividono lo stesso state
(il prompt), quindi una sola richiesta con prefisso condiviso. Si tengono le opzioni con
probabilita' >= RIZZO_SKILL_TAU (default 0.30), al massimo RIZZO_SKILL_MAX (default 3).

Env: RIZZO_URL, RIZZO_SKILL_TAU, RIZZO_SKILL_MAX, RIZZO_SKILLS_EXTRA, RIZZO_SKILLS=0 (disattiva).
"""

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from route_prompt import call_rizzo  # noqa: E402

TAU = float(os.environ.get("RIZZO_SKILL_TAU", "0.30"))
MAX_PICK = int(os.environ.get("RIZZO_SKILL_MAX", "3"))
GROUP = 24  # 26 slot: 24 skill + "none" + margine
NONE_ID = "none"
_FRONT = re.compile(r"^---\s*\n(.*?)\n---", re.S)


def _frontmatter(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    m = _FRONT.match(text)
    out = {}
    if m:
        for line in m.group(1).splitlines():
            k, sep, v = line.partition(":")
            if sep and k.strip() in ("name", "description"):
                out[k.strip()] = v.strip().strip("\"'")
    return out


def catalog(cwd: str = "") -> list[dict]:
    home = Path.home() / ".claude"
    roots = [home / "skills", Path(cwd or os.getcwd()) / ".claude" / "skills"]
    files = [p for r in roots if r.is_dir() for p in r.glob("*/SKILL.md")]
    files += list((home / "plugins" / "cache").glob("**/SKILL.md"))
    seen, out = set(), []
    for f in files:
        fm = _frontmatter(f)
        name, desc = fm.get("name") or f.parent.name, fm.get("description", "")
        if desc and name not in seen:
            seen.add(name)
            out.append({"name": name, "description": desc[:600], "path": str(f)})
    extra = os.environ.get("RIZZO_SKILLS_EXTRA", str(Path(__file__).resolve().parent / "skills_extra.json"))
    if extra and Path(extra).is_file():
        for s in json.loads(Path(extra).read_text(encoding="utf-8")):
            if s["name"] not in seen:
                seen.add(s["name"])
                out.append({"name": s["name"], "description": s["description"][:600]})
    return out


def build_payload(prompt: str, skills: list[dict]) -> dict:
    questions = {}
    for g in range(0, len(skills), GROUP):
        chunk = skills[g:g + GROUP]
        opts = [{"id": f"s{g + i}", "description": f"{s['name']}: {s['description']}"} for i, s in enumerate(chunk)]
        opts.append({"id": NONE_ID, "description": "None of these skills is needed for this request."})
        questions[f"group_{g // GROUP}"] = {
            "type": "choice",
            "instructions": "Which skill, if any, should the assistant load to carry out the user request in the evidence?",
            "options": opts,
            "policy": {"allow_abstain": False},
        }
    return {"state": {"user_request": prompt[:12000]}, "questions": questions}


def select(prompt: str, cwd: str = "") -> dict:
    """Skill scelte per il prompt. Non solleva mai: in caso di errore ok=False."""
    t0 = time.perf_counter()
    skills = catalog(cwd)
    if not skills:
        return {"ok": False, "reason": "nessuna skill trovata", "skills": []}
    try:
        resp = call_rizzo(build_payload(prompt, skills), timeout=4.5)
    except Exception as e:  # fail-open: il selettore non deve mai bloccare il lavoro
        return {"ok": False, "reason": f"rizzo non raggiungibile ({e})", "skills": []}
    picks = []
    for ans in resp["answers"].values():
        for oid, p in ans["probabilities"].items():
            if oid != NONE_ID and oid.startswith("s") and p >= TAU:
                picks.append((p, skills[int(oid[1:])]["name"]))
    picks.sort(reverse=True)
    return {"ok": True, "skills": [{"name": n, "p": round(p, 3)} for p, n in picks[:MAX_PICK]],
            "catalog_size": len(skills), "latency_ms": round((time.perf_counter() - t0) * 1000)}


def hook_text(prompt: str, cwd: str = "") -> str | None:
    d = select(prompt, cwd)
    if not d["ok"]:
        return None
    if not d["skills"]:
        return (f"[rizzo-skills] Nessuna delle {d['catalog_size']} skill installate serve per questo prompt: "
                "non caricarne nessuna.")
    names = ", ".join(f"{s['name']} (p={s['p']:.2f})" for s in d["skills"])
    return (f"[rizzo-skills] Skill pertinenti per questo prompt (scelte da Rizzo Flow su {d['catalog_size']}): "
            f"{names}. Carica SOLO queste (tool Skill) se servono; considera disattivate tutte le altre "
            "per questo turno, salvo richiesta esplicita dell'utente.")


def hook() -> int:
    if os.environ.get("RIZZO_SKILLS", "1") == "0":
        return 0
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return 0
    prompt = data.get("prompt", "")
    if not prompt.strip() or prompt.lstrip().startswith("/"):
        return 0
    text = hook_text(prompt, data.get("cwd", ""))
    if text:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                                 "additionalContext": text}}, ensure_ascii=False))
    return 0


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ["--hook"]:
        return hook()
    if args[:1] == ["--list"]:
        for s in catalog():
            print(f"{s['name']}: {s['description'][:100]}")
        return 0
    prompt = sys.stdin.read() if args in ([], ["-"]) else " ".join(args)
    print(json.dumps(select(prompt), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
