"""The published results page (docs/index.html).

These run offline against the site templates -- no processed data needed. They
guard the invariants that actually broke while the page was being built.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_report  # noqa: E402


@pytest.fixture(scope="module")
def fragments():
    return (
        (build_report.SITE / "head.html").read_text(encoding="utf-8"),
        (build_report.SITE / "body.html").read_text(encoding="utf-8"),
        (build_report.SITE / "script.html").read_text(encoding="utf-8"),
    )


@pytest.fixture(scope="module")
def document(fragments):
    payload = {
        "results": [{"label": "random_forest / rr+morphology", "macro": 0.4,
                     "acc": 0.8, "per_class": [], "cm": [[1]]}],
        "pipeline_totals": {"annotations": 10, "kept": 8, "boundary": 1,
                            "nonbeat": 1, "unknown": 0, "records": 2},
    }
    return build_report.assemble(payload, *fragments)


def test_output_is_pure_ascii(document):
    """Regression test: em-dashes rendered as mojibake twice during development.

    The page is assembled from fragments, so anything non-ASCII depends on the
    charset being served correctly. Escaping to 7-bit removes that dependency.
    """
    offenders = sorted({c for c in document if ord(c) > 127})
    assert not offenders, f"non-ASCII in output: {offenders}"


def test_known_punctuation_is_escaped_rather_than_rejected(fragments):
    """Characters in the entity map are converted, not treated as errors."""
    head, body, script = fragments
    doc = build_report.assemble({}, head, body + "<p>an em dash \u2014 here</p>", script)
    assert "&#8212;" in doc
    assert "\u2014" not in doc


def test_assemble_rejects_non_ascii_outside_the_entity_map(fragments):
    """The last-resort guard must actually fire, not merely exist.

    A character the entity map does not cover has to stop the build rather than
    ship as mojibake -- which is exactly how the bug reached the page twice.
    """
    head, body, script = fragments
    exotic = "\u03c0\u65e5"  # pi, and a CJK character
    assert all(ch not in build_report.ENTITIES for ch in exotic)
    with pytest.raises(SystemExit, match="non-ASCII"):
        build_report.assemble({}, head, body + exotic, script)


def test_document_is_a_complete_html_page(document):
    """Unlike the artifact version, this one supplies its own head."""
    assert document.startswith("<!doctype html>")
    assert '<html lang="en">' in document
    assert '<meta charset="utf-8">' in document
    assert '<meta name="viewport"' in document
    assert f"<title>{build_report.DOC_TITLE}</title>" in document
    assert document.rstrip().endswith("</html>")


def test_every_chart_container_has_a_renderer(fragments):
    """A chart id in the markup with no matching draw call renders blank."""
    _, body, script = fragments
    ids = set(re.findall(r'<svg id="(c-[a-z]+)"', body))
    assert ids, "no chart containers found"
    for chart_id in ids:
        assert f'getElementById("{chart_id}")' in script, f"{chart_id} is never drawn"


def test_theme_is_defined_for_all_three_states(fragments):
    """System, explicit light and explicit dark must each resolve."""
    head = fragments[0]
    assert "--paper:" in head.split("@media")[0], "light tokens must sit on bare :root"
    assert "@media (prefers-color-scheme: dark)" in head
    assert ':root:not([data-theme="light"])' in head
    assert ':root[data-theme="dark"]' in head


def test_body_paints_its_own_background(fragments):
    """A transparent body borrows the host's ground and can render unreadable."""
    head = fragments[0]
    body_rule = re.search(r"\bbody\{([^}]*)\}", head)
    assert body_rule and "background:var(--paper)" in body_rule.group(1)


def test_page_is_self_contained_apart_from_google_fonts(document):
    """No CDN scripts or remote images -- it must work as a static file."""
    remote = re.findall(r'(?:src|href)="(https?://[^"]+)"', document)
    for url in remote:
        assert url.startswith(("https://fonts.googleapis.com",
                               "https://fonts.gstatic.com",
                               "https://github.com/")), f"unexpected remote asset: {url}"
    assert "<script src=" not in document


def test_theme_toggle_is_wired_up(fragments):
    body, script = fragments[1], fragments[2]
    assert 'id="theme-toggle"' in body
    assert 'getElementById("theme-toggle")' in script
    assert "localStorage" in script


def test_localstorage_access_is_guarded(fragments):
    """Private-mode browsers throw on localStorage; the page must still render."""
    script = fragments[2]
    section = script[script.index("smarthealth-theme") - 400:]
    assert section.count("try{") >= 2, "localStorage reads/writes must be in try/catch"
