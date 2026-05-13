"""Round-trip tests for ``Structures.export_segmentation`` / ``Structures.load_from``.

These exercise the FITS-based persistence path: segment + hierarchy → write
``<fname>_perch_seg.fits`` → read back into a fresh ``Structures`` and assert
the maps and per-structure properties round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from perch.structures import Structures


def _odir(tmp_path):
    return str(tmp_path) + "/"


def test_export_and_load_roundtrip(strucs_2d_two_peaks_h0, tmp_path):
    s = strucs_2d_two_peaks_h0
    s.export_segmentation(fname="run", odir=_odir(tmp_path))
    loaded = Structures.load_from(odir=_odir(tmp_path), fname="run",
                                  verbose=False)

    assert loaded.n_struc == s.n_struc
    np.testing.assert_array_equal(loaded.struc_map, s.struc_map)
    np.testing.assert_array_equal(loaded.level_map, s.level_map)

    # Order is not guaranteed to match; sort by id_ph and compare row-wise.
    a = np.argsort(loaded.id_ph)
    b = np.argsort(s.id_ph)
    np.testing.assert_allclose(loaded.birth[a], s.birth[b])
    np.testing.assert_array_equal(loaded.npix[a], s.npix[b])
    np.testing.assert_array_equal(np.array(loaded.level)[a].astype(int),
                                  np.array(s.level)[b].astype(int))


def test_export_and_load_preserves_parent_links(strucs_2d_two_peaks_h0, tmp_path):
    s = strucs_2d_two_peaks_h0
    s.export_segmentation(fname="run", odir=_odir(tmp_path))
    loaded = Structures.load_from(odir=_odir(tmp_path), fname="run",
                                  verbose=False)

    assert len(loaded.trunk) == len(s.trunk)
    assert len(loaded.leaves) == len(s.leaves)
    # Every structure with a parent in the original has one after reload.
    parents_before = {struc.id: struc.parent for struc in s.all_structures}
    parents_after = {struc.id: struc.parent for struc in loaded.all_structures}
    assert parents_before == parents_after


# ---------------------------------------------------------------------------
# Individual map exports (struc_map.fits, level_map.fits)
# ---------------------------------------------------------------------------

def test_export_struc_map_writes_readable_fits(strucs_2d_two_peaks_h0, tmp_path):
    from astropy.io import fits
    strucs_2d_two_peaks_h0.export_struc_map(fname="run", odir=_odir(tmp_path))
    out = tmp_path / "run_struc_map.fits"
    assert out.exists()
    with fits.open(out) as hdul:
        loaded = np.asarray(hdul[0].data, dtype=float)
    # NaN-aware comparison: same finite mask, same finite values.
    smap = np.asarray(strucs_2d_two_peaks_h0.struc_map, dtype=float)
    np.testing.assert_array_equal(np.isfinite(loaded), np.isfinite(smap))
    finite = np.isfinite(loaded)
    np.testing.assert_allclose(loaded[finite], smap[finite])


def test_export_level_map_writes_readable_fits(strucs_2d_two_peaks_h0, tmp_path):
    from astropy.io import fits
    strucs_2d_two_peaks_h0.export_level_map(fname="run", odir=_odir(tmp_path))
    out = tmp_path / "run_level_map.fits"
    assert out.exists()
    with fits.open(out) as hdul:
        loaded = np.asarray(hdul[0].data, dtype=float)
    lmap = np.asarray(strucs_2d_two_peaks_h0.level_map, dtype=float)
    np.testing.assert_array_equal(np.isfinite(loaded), np.isfinite(lmap))
    finite = np.isfinite(loaded)
    np.testing.assert_allclose(loaded[finite], lmap[finite])
