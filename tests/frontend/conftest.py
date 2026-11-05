"""Fixtures for the frontend tests: the agentic fixtures (tiny ECG ensemble,
synthetic MIT-BIH record, synthetic MIMIC-III database) and an App on them.
No real patient data."""

from __future__ import annotations

import pytest

from agentic import orchestrator

from ..agentic.conftest import (  # noqa: F401  (fixtures, re-exported for this folder)
    chart_env, chart_sources, knowledge, mitdb_dir, tiny_bundle, tiny_classifier)


@pytest.fixture
def deps(knowledge, tiny_classifier, chart_sources, mitdb_dir):
    return orchestrator.Deps(knowledge=knowledge, classifier=tiny_classifier,
                             chart=chart_sources, record_dir=mitdb_dir)


@pytest.fixture
def app(deps, chart_env):
    from frontend.app import App
    from frontend.outcome import OutcomeLookup

    return App(deps, llm=None, outcome=OutcomeLookup(chart_env["out"]))
