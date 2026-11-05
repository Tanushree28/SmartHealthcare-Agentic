"""Typed failures, so a broken input says what is broken."""

from __future__ import annotations


class MimicError(Exception):
    """Base class for MIMIC-III pipeline errors."""


class DataSourceMissingError(MimicError):
    """The requested MIMIC-III source is not on disk.

    For the full database this is an access blocker (credentialing), not a bug.
    """


class SchemaError(MimicError):
    """A table is missing, duplicated, lacks columns, or has unparseable values."""


class IntegrityError(MimicError):
    """Tables disagree: duplicate keys, orphan rows, or mismatched identifiers."""


class LeakageError(MimicError):
    """A post-index-time column or a cross-split patient was about to be used."""
