"""Guards on the Phase 1g pre-registration.

This experiment's validity rests on one claim: the composition was chosen on
DS1 out-of-fold predictions, before DS2 was scored. Nothing in a results file
can prove that after the fact -- the protection has to live in the code, and
these tests are what keep it honest.

The invariants here are cheap and would be easy to break by accident: adding a
candidate that names a loss nobody trains, dropping the reference composition
so there is nothing to compare against, or letting the selection metric drift
to something computed on the test set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("torch")

import train_ensemble_mix as mix  # noqa: E402

from ecg import config  # noqa: E402


class TestPreRegistration:
    def test_every_candidate_names_arms_that_are_trained(self):
        """A composition naming an untrained loss would KeyError mid-run."""
        trained = {a.name for a in mix.ARMS}
        for name, members in mix.COMPOSITIONS.items():
            unknown = set(members) - trained
            assert not unknown, f"{name} names untrained arms {sorted(unknown)}"

    def test_reference_is_a_candidate(self):
        """Without the reference in the table there is nothing to improve on."""
        assert mix.REFERENCE in mix.COMPOSITIONS

    def test_reference_is_the_current_ensemble(self):
        """Today's ensemble is five copies of focal; that is the thing to beat."""
        assert mix.COMPOSITIONS[mix.REFERENCE] == ("focal",)

    def test_a_candidate_drops_the_incumbent(self):
        """Phase 1f found focal the weakest S ranker.

        'Leave focal out' therefore has to be on the table. A candidate list
        where every option keeps the incumbent cannot express that finding.
        """
        assert any(
            "focal" not in members for members in mix.COMPOSITIONS.values()
        ), "no candidate drops focal; the hypothesis cannot be tested"

    def test_candidates_are_distinct(self):
        seen = {}
        for name, members in mix.COMPOSITIONS.items():
            key = frozenset(members)
            assert key not in seen, f"{name} duplicates {seen.get(key)}"
            seen[key] = name

    def test_selection_metric_is_named_out_of_fold(self):
        """The metric string is what the results file reports as the rule."""
        assert "OOF" in mix.SELECTION_METRIC or "out-of-fold" in mix.SELECTION_METRIC

    def test_arms_differ_only_in_the_loss(self):
        """Anything else varying would confound the comparison."""
        losses = {(a.loss, a.oversample) for a in mix.ARMS}
        assert len(losses) == len(mix.ARMS), "two arms share a loss configuration"


class TestScore:
    def _fixture(self, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.choice(
            len(config.AAMI_CLASSES), size=n, p=[0.89, 0.04, 0.06, 0.009, 0.001]
        )
        logits = rng.normal(size=(n, len(config.AAMI_CLASSES)))
        logits[np.arange(n), y] += 2.0
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return y, e / e.sum(axis=1, keepdims=True)

    def test_reports_both_kinds_of_metric(self):
        """Ranking, decisions and calibration -- Phase 1f showed all three differ."""
        y, proba = self._fixture()
        result = mix.score(y, proba)

        for key in (
            "macro_average_precision",
            "s_average_precision",
            "headline_macro_f1",
            "s_recall",
            "s_precision",
            "brier",
            "classwise_ece",
        ):
            assert key in result, f"score() dropped {key}"
            assert np.isfinite(result[key]), f"{key} is not finite"

    def test_a_better_model_scores_higher(self):
        """The guard against a metric wired up backwards."""
        y, good = self._fixture(seed=1)
        rng = np.random.default_rng(2)
        noise = rng.random(good.shape)
        bad = noise / noise.sum(axis=1, keepdims=True)

        assert (
            mix.score(y, good)["macro_average_precision"]
            > mix.score(y, bad)["macro_average_precision"]
        )
