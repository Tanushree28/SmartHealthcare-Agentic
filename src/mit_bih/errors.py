"""Exception types for the preprocessing pipeline.

Each one exists so a failure surfaces with a diagnosable message instead of a
raw library traceback. See the edge-case list in PlanDoc/PLAN.md.
"""

from __future__ import annotations


class EcgError(Exception):
    """Base class for every error this package raises deliberately."""


class DataRootError(EcgError):
    """The data root is missing or does not contain what was expected."""


class LeadNotFoundError(EcgError):
    """No lead from the configured preference order exists in the record."""


class SamplingRateError(EcgError):
    """The record's sampling rate differs from the one the filters assume."""


class SignalTooShortError(EcgError):
    """The signal is shorter than the filter's required padding length."""


class InvalidSignalError(EcgError):
    """The signal contains NaN or inf and cannot be processed."""


class UnknownSymbolError(EcgError):
    """An annotation symbol is absent from the AAMI mapping table."""
