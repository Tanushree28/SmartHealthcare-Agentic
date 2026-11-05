"""Glossary entries the chat may cite (G1, G2, ...), loaded and checked.

Each entry becomes a Finding with agent "glossary", so the fact-checker treats
a definition exactly like a case fact: numbers and clinical words in an answer
must be backed by the entries it cites.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from common import paths

from .findings import Finding

DEFAULT_PATH = paths.PROJECT_ROOT / "configs" / "glossary.toml"
GLOSSARY_ID = re.compile(r"^G\d+$")


class GlossaryError(ValueError):
    pass


def load_glossary(path: Path | str = DEFAULT_PATH) -> list[Finding]:
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    out, seen = [], set()
    for i, e in enumerate(raw.get("entry", [])):
        gid = str(e.get("id", ""))
        where = f"entry #{i + 1} ({gid or '?'})"
        if not GLOSSARY_ID.match(gid):
            raise GlossaryError(f"{where}: id must look like G1, G2, ...")
        if gid in seen:
            raise GlossaryError(f"{where}: duplicate id")
        seen.add(gid)
        for name in ("term", "definition", "source"):
            if not str(e.get(name, "")).strip():
                raise GlossaryError(f"{where}: missing {name}")
        out.append(Finding(id=gid, agent="glossary", kind="glossary", key=e["term"],
                           statement=f"{e['term']}: {e['definition']}",
                           evidence={"source": e["source"]}, terms=tuple(e.get("terms", []))))
    return out
