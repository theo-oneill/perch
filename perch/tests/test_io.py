"""Round-trip and behavior tests for ``PH.export_generators`` / ``PH.load_from``.

These encode the *current* behavior of ``load_from``. In particular,
``load_from`` silently drops:

* rows whose death value is below ``np.nanmin(data)`` (this strips the
  essential class with the default ``flip_data=True`` convention), and
* rows whose birth column is NaN.

See ``TODO.md`` for an open question about whether the essential-class
strip should be revisited.
"""

from __future__ import annotations

import numpy as np
import pytest

from perch.ph import PH


_ESSENTIAL_THRESHOLD = -1e30


def _odir(tmp_path):
    """`load_from`/`export_generators` expect a trailing-slash directory."""
    return str(tmp_path) + "/"


def test_export_load_roundtrip_strips_essential(ph_2d_two_peaks, toy_2d_two_peaks, tmp_path):
    """Exporting then loading drops the essential class but preserves finite rows."""
    img, _ = toy_2d_two_peaks
    ph_2d_two_peaks.export_generators("gens.txt", odir=_odir(tmp_path))

    loaded = PH.load_from("gens.txt", odir=_odir(tmp_path), data=img)

    essential = ph_2d_two_peaks.generators[:, 2] < _ESSENTIAL_THRESHOLD
    expected_finite = ph_2d_two_peaks.generators[~essential]

    assert loaded.generators.shape[0] == expected_finite.shape[0]
    # Sort both by h_id (column 9) so row-by-row comparison is order-independent.
    a = loaded.generators[np.argsort(loaded.generators[:, 9])]
    e = expected_finite[np.argsort(expected_finite[:, 9])]
    np.testing.assert_allclose(a, e, rtol=1e-10)


def test_load_from_conv_fac_scales_birth_death(ph_2d_two_peaks, toy_2d_two_peaks, tmp_path):
    """`conv_fac` scales birth and death; other columns are untouched."""
    img, _ = toy_2d_two_peaks
    ph_2d_two_peaks.export_generators("gens.txt", odir=_odir(tmp_path))

    loaded = PH.load_from("gens.txt", odir=_odir(tmp_path),
                          data=img, conv_fac=2.0)

    essential = ph_2d_two_peaks.generators[:, 2] < _ESSENTIAL_THRESHOLD
    expected = ph_2d_two_peaks.generators[~essential].copy()
    expected[:, 1:3] *= 2.0

    a = loaded.generators[np.argsort(loaded.generators[:, 9])]
    e = expected[np.argsort(expected[:, 9])]
    np.testing.assert_allclose(a, e, rtol=1e-10)


def test_load_from_adds_h_id_for_9_column_file(toy_2d_two_peaks, tmp_path):
    """A 9-column text file (no h_id) gets the h_id column auto-appended."""
    img, _ = toy_2d_two_peaks
    # Two synthetic finite-death rows with deaths above min(img).
    rows = np.array([
        [0.0, 0.50, 0.10, 5.0, 5.0, 0.0, 3.0, 3.0, 0.0],
        [0.0, 0.30, 0.05, 7.0, 7.0, 0.0, 1.0, 1.0, 0.0],
    ])
    path = tmp_path / "gens9.txt"
    np.savetxt(path, rows)

    loaded = PH.load_from("gens9.txt", odir=_odir(tmp_path), data=img)
    assert loaded.generators.shape == (2, 10)
    np.testing.assert_array_equal(loaded.generators[:, 9], np.array([0, 1]))


def test_load_from_drops_rows_below_data_min(toy_2d_two_peaks, tmp_path):
    """Rows whose death is below ``min(data)`` are stripped on load."""
    img, _ = toy_2d_two_peaks
    valid =    [0.0, 0.50, 0.10,                    5.0, 5.0, 0.0, 3.0, 3.0, 0.0, 0.0]
    too_low =  [0.0, 0.50, float(np.nanmin(img)) - 1.0, 5.0, 5.0, 0.0, 3.0, 3.0, 0.0, 1.0]
    np.savetxt(tmp_path / "g.txt", np.array([valid, too_low]))

    loaded = PH.load_from("g.txt", odir=_odir(tmp_path), data=img)
    assert loaded.generators.shape == (1, 10)
    np.testing.assert_allclose(loaded.generators[0], valid)


def test_load_from_drops_nan_birth(toy_2d_two_peaks, tmp_path):
    """Rows whose birth is NaN are stripped on load."""
    img, _ = toy_2d_two_peaks
    valid =    [0.0, 0.50,        0.10, 5.0, 5.0, 0.0, 3.0, 3.0, 0.0, 0.0]
    nan_birth = [0.0, float("nan"), 0.10, 5.0, 5.0, 0.0, 3.0, 3.0, 0.0, 1.0]
    np.savetxt(tmp_path / "g.txt", np.array([valid, nan_birth]))

    loaded = PH.load_from("g.txt", odir=_odir(tmp_path), data=img)
    assert loaded.generators.shape == (1, 10)
    np.testing.assert_allclose(loaded.generators[0], valid)
