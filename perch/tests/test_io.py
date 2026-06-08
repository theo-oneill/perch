"""Round-trip and behavior tests for ``PH.export_generators`` / ``PH.load_from``.

``load_from`` is a *faithful* loader: it returns exactly the rows in the file
(optionally scaled by ``conv_fac``), so an ``export_generators`` -> ``load_from``
round trip is the identity and matches ``compute_hom``'s in-memory generators.
It no longer silently drops the essential ``-DBL_MAX`` row, NaN-origin rows, or
NaN-birth rows — cleaning is a filtering decision left to the caller (e.g. via
``PH.filter`` / structure selection).
"""

from __future__ import annotations

import numpy as np

from perch.ph import PH
from perch.tests._fixtures import ESSENTIAL_SENTINEL as _ESSENTIAL_THRESHOLD


def _odir(tmp_path):
    """`load_from`/`export_generators` expect a trailing-slash directory."""
    return str(tmp_path) + "/"


def test_export_load_roundtrip_is_faithful(ph_2d_two_peaks_legacy, toy_2d_two_peaks, tmp_path):
    """Exporting then loading preserves *every* row, including the legacy
    essential class with its ``-DBL_MAX`` sentinel death. The round trip is the
    identity. (Uses the legacy fixture precisely because it carries the
    sentinel row that the old ``load_from`` used to strip.)"""
    img, _ = toy_2d_two_peaks
    ph_2d_two_peaks_legacy.export_generators("gens.txt", odir=_odir(tmp_path))

    loaded = PH.load_from("gens.txt", odir=_odir(tmp_path), data=img)

    expected = ph_2d_two_peaks_legacy.generators
    assert loaded.generators.shape[0] == expected.shape[0]
    # The essential sentinel row round-trips unchanged.
    assert (loaded.generators[:, 2] < _ESSENTIAL_THRESHOLD).sum() == 1
    # Sort both by h_id (column 9) so row-by-row comparison is order-independent.
    a = loaded.generators[np.argsort(loaded.generators[:, 9])]
    e = expected[np.argsort(expected[:, 9])]
    np.testing.assert_allclose(a, e, rtol=1e-10)


def test_load_from_conv_fac_scales_birth_death(ph_2d_two_peaks, toy_2d_two_peaks, tmp_path):
    """`conv_fac` scales birth and death on every row; other columns are
    untouched. Uses the default fixture so all deaths are finite and scale
    cleanly (the legacy -DBL_MAX death would overflow under conv_fac — see
    test_pad_essential for the no-overflow contract under the new default)."""
    img, _ = toy_2d_two_peaks
    ph_2d_two_peaks.export_generators("gens.txt", odir=_odir(tmp_path))

    loaded = PH.load_from("gens.txt", odir=_odir(tmp_path),
                          data=img, conv_fac=2.0)

    expected = ph_2d_two_peaks.generators.copy()
    expected[:, 1:3] *= 2.0

    assert loaded.generators.shape[0] == expected.shape[0]
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


def test_load_from_keeps_rows_below_data_min(toy_2d_two_peaks, tmp_path):
    """Faithful load: a row whose death is below ``min(data)`` is kept, not
    stripped (the old loader dropped it; cleaning is now the caller's job)."""
    img, _ = toy_2d_two_peaks
    valid =    [0.0, 0.50, 0.10,                    5.0, 5.0, 0.0, 3.0, 3.0, 0.0, 0.0]
    too_low =  [0.0, 0.50, float(np.nanmin(img)) - 1.0, 5.0, 5.0, 0.0, 3.0, 3.0, 0.0, 1.0]
    np.savetxt(tmp_path / "g.txt", np.array([valid, too_low]))

    loaded = PH.load_from("g.txt", odir=_odir(tmp_path), data=img)
    assert loaded.generators.shape == (2, 10)
    a = loaded.generators[np.argsort(loaded.generators[:, 9])]
    np.testing.assert_allclose(a, np.array([valid, too_low]))


def test_load_from_keeps_nan_birth_row(toy_2d_two_peaks, tmp_path):
    """Faithful load: a row whose birth is NaN is kept, not stripped."""
    img, _ = toy_2d_two_peaks
    valid =    [0.0, 0.50,        0.10, 5.0, 5.0, 0.0, 3.0, 3.0, 0.0, 0.0]
    nan_birth = [0.0, float("nan"), 0.10, 5.0, 5.0, 0.0, 3.0, 3.0, 0.0, 1.0]
    np.savetxt(tmp_path / "g.txt", np.array([valid, nan_birth]))

    loaded = PH.load_from("g.txt", odir=_odir(tmp_path), data=img)
    assert loaded.generators.shape == (2, 10)
    # The valid row is unchanged; the NaN-birth row is preserved (NaN birth).
    by_id = loaded.generators[np.argsort(loaded.generators[:, 9])]
    np.testing.assert_allclose(by_id[0], valid)
    assert np.isnan(by_id[1][1])
