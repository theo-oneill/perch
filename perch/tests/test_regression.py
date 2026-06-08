"""Golden-output regression tests for ``PH.compute_hom``.

These tests assert that the generator table produced for each fixture
matches a frozen reference. References live at ``perch/tests/data/*.npz``
and are regenerated via ``python -m perch.tests.data._regenerate``.

The references were captured with ``pad_essential=False``, so the tests
here use the ``*_legacy`` fixtures to pin that behavior. The padded
default (``pad_essential='auto'``) has its own regression coverage in
``test_pad_essential.py``.

The essential-class death column is a platform-dependent sentinel
(``-DBL_MAX`` on the current cripser build), so essential rows are matched
on their well-defined columns (htype, birth, birthpix, h_id) rather than
numerically on death.
"""

from pathlib import Path

import numpy as np
import pytest


REF_DIR = Path(__file__).resolve().parent / "data"

# Threshold below which a death value is treated as the essential-class
# sentinel rather than a real number.
from perch.tests._fixtures import ESSENTIAL_SENTINEL as _ESSENTIAL_THRESHOLD

# Columns of the 10-column generator table that are well-defined even for
# the essential class. Skips columns 2 (sentinel death) and 6–8 (deathpix,
# which is bogus for the essential class).
_ESSENTIAL_SAFE_COLS = [0, 1, 3, 4, 5, 9]


def _load_reference(name):
    path = REF_DIR / f"{name}_generators.npz"
    if not path.exists():
        pytest.skip(
            f"reference {path.name} missing — "
            f"run `python -m perch.tests.data._regenerate`"
        )
    return np.load(path)["generators"]


def _compare_generators(actual, reference, rtol=1e-12, atol=0.0):
    """Compare two generator tables, splitting on the essential-class sentinel."""
    assert actual.shape == reference.shape, (
        f"generator-table shape mismatch: {actual.shape} vs {reference.shape}"
    )

    a_ess = actual[:, 2] < _ESSENTIAL_THRESHOLD
    r_ess = reference[:, 2] < _ESSENTIAL_THRESHOLD
    assert a_ess.sum() == r_ess.sum(), (
        f"essential-class count mismatch: {a_ess.sum()} vs {r_ess.sum()}"
    )

    # Finite-death rows: compare every column.
    np.testing.assert_allclose(actual[~a_ess], reference[~r_ess],
                               rtol=rtol, atol=atol)

    # Essential rows: skip the sentinel death and the bogus deathpix.
    if a_ess.any():
        np.testing.assert_allclose(actual[a_ess][:, _ESSENTIAL_SAFE_COLS],
                                   reference[r_ess][:, _ESSENTIAL_SAFE_COLS],
                                   rtol=rtol, atol=atol)


def test_regression_2d_two_peaks(ph_2d_two_peaks_legacy):
    _compare_generators(ph_2d_two_peaks_legacy.generators,
                        _load_reference("2d_two_peaks"))


def test_regression_3d_two_peaks(ph_3d_two_peaks_legacy):
    _compare_generators(ph_3d_two_peaks_legacy.generators,
                        _load_reference("3d_two_peaks"))


def test_regression_2d_ring(ph_2d_ring_legacy):
    _compare_generators(ph_2d_ring_legacy.generators,
                        _load_reference("2d_ring"))


def test_regression_3d_shell(ph_3d_shell_legacy):
    _compare_generators(ph_3d_shell_legacy.generators,
                        _load_reference("3d_shell"))
