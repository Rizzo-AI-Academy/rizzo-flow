#!/usr/bin/env python3
"""Installa skill rizzo-flow + hook router a livello utente (tutti i progetti/sessioni).

    python router/install_global.py            # installa
    python router/install_global.py --remove   # rimuove l'hook (la skill resta)

- Copia skills/rizzo-flow in ~/.claude/skills/rizzo-flow
- Aggiunge gli hook UserPromptSubmit e PreToolUse (Agent|Task) a ~/.claude/settings.json (backup .bak, merge, idempotente)
"""
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLAUDE = Path.home() / ".claude"
SETTINGS = CLAUDE / "settings.json"
SCRIPT = (REPO / "router" / "route_prompt.py").as_posix()
CMD = f'python "{SCRIPT}" --hook'
PRE_CMD = f'python "{SCRIPT}" --pretool'
MARK = "route_prompt.py"


def load():
    if SETTINGS.exists():
        backup = SETTINGS.with_suffix(f".json.{time.strftime('%Y%m%d-%H%M%S')}.bak")
        shutil.copy2(SETTINGS, backup)
        print(f"backup: {backup}")
        return json.loads(SETTINGS.read_text(encoding="utf-8") or "{}")
    return {}


def strip(groups):
    out = []
    for g in groups:
        hooks = [h for h in g.get("hooks", []) if MARK not in h.get("command", "")]
        if hooks:
            out.append({**g, "hooks": hooks})
    return out


def main():
    remove = "--remove" in sys.argv
    CLAUDE.mkdir(exist_ok=True)
    s = load()
    hooks = s.setdefault("hooks", {})
    pre = strip(hooks.get("PreToolUse", []))
    groups = strip(hooks.get("UserPromptSubmit", []))
    if not remove:
        pre.append({"matcher": "Agent|Task", "hooks": [{"type": "command", "command": PRE_CMD, "timeout": 5}]})
        dst = CLAUDE / "skills" / "rizzo-flow"
        shutil.copytree(REPO / "skills" / "rizzo-flow", dst, dirs_exist_ok=True)
        print(f"skill: {dst}")
        groups.append({"hooks": [{"type": "command", "command": CMD, "timeout": 5}]})
    if groups:
        hooks["UserPromptSubmit"] = groups
    else:
        hooks.pop("UserPromptSubmit", None)
    if pre:
        hooks["PreToolUse"] = pre
    else:
        hooks.pop("PreToolUse", None)
    SETTINGS.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(("hook rimosso da " if remove else "hook installato in ") + str(SETTINGS))


if __name__ == "__main__":
    main()
