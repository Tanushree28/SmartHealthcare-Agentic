"""Group C -- AAMI label mapping (PlanDoc/PLAN.md)."""

from __future__ import annotations

import pytest

from ecg import config
from ecg.errors import UnknownSymbolError
from ecg.labels import (
    BEAT_SYMBOL_TO_AAMI,
    NON_BEAT_SYMBOLS,
    class_index,
    map_symbol,
    map_symbols,
)

# The canonical 15 MIT-BIH beat types (ANSI/AAMI EC57). Written out here as an
# independent expectation rather than read from the module under test -- a table
# that checks itself proves nothing.
CANONICAL = [
    ("N", "N"), ("L", "N"), ("R", "N"), ("e", "N"), ("j", "N"),
    ("A", "S"), ("a", "S"), ("J", "S"), ("S", "S"),
    ("V", "V"), ("E", "V"),
    ("F", "F"),
    ("/", "Q"), ("f", "Q"), ("Q", "Q"),
]


@pytest.mark.parametrize("symbol,expected", CANONICAL)
def test_c1_every_beat_symbol_maps_to_its_aami_class(symbol, expected):
    """C1: one mis-keyed entry silently mislabels a whole beat category."""
    result = map_symbol(symbol)

    assert result.outcome == "beat"
    assert result.aami_class == expected
    assert result.class_index == config.AAMI_CLASSES.index(expected)


def test_c1b_beat_table_covers_exactly_the_canonical_fifteen():
    assert set(BEAT_SYMBOL_TO_AAMI) == {s for s, _ in CANONICAL}


@pytest.mark.parametrize(
    "symbol",
    ["+", "~", "|", "!", "[", "]", "x", "(", ")", "p", "t", "u", '"', "="],
)
def test_c2_non_beat_symbols_are_excluded_not_classified(symbol):
    """C2: prevents phantom beats entering the dataset."""
    result = map_symbol(symbol)

    assert result.outcome == "excluded"
    assert result.aami_class is None
    assert result.class_index is None
    assert result.reason


def test_c2b_no_symbol_is_in_both_tables():
    """An overlap would make the mapping order-dependent and ambiguous."""
    assert set(BEAT_SYMBOL_TO_AAMI) & set(NON_BEAT_SYMBOLS) == set()


def test_c3_unknown_symbol_raises_in_strict_mode():
    """C3: a mapping gap must be visible, not quietly shrink the dataset."""
    with pytest.raises(UnknownSymbolError, match="refusing to guess"):
        map_symbol("§", strict=True)


def test_c3b_unknown_symbol_is_counted_in_lenient_mode():
    result = map_symbol("§", strict=False)

    assert result.outcome == "unknown"
    assert result.aami_class is None


def test_c4_aami_class_order_is_stable_and_documented():
    """C4: a reordering would invalidate every saved label array."""
    assert config.AAMI_CLASSES == ("N", "S", "V", "F", "Q")
    assert [class_index(c) for c in config.AAMI_CLASSES] == [0, 1, 2, 3, 4]


def test_class_index_rejects_a_non_aami_name():
    with pytest.raises(ValueError, match="not an AAMI class"):
        class_index("X")


def test_map_symbols_preserves_order():
    results = map_symbols(["N", "+", "V"])

    assert [r.outcome for r in results] == ["beat", "excluded", "beat"]
    assert [r.aami_class for r in results] == ["N", None, "V"]


def test_flutter_wave_is_excluded_with_a_reason():
    """'!' marks a flutter wave, not a classifiable ventricular ectopic beat."""
    result = map_symbol("!")

    assert result.outcome == "excluded"
    assert "flutter" in result.reason.lower()
