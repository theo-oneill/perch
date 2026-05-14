"""Tests for the ``*_coord`` properties on a Structures with a WCS attached.

The 3D coord-properties all work cleanly. The 2D path has three known
bugs (hardcoded 3-D coordinate handling in ``birthpix_coord``,
``centroid_coord``, and ``bbox_min_coord`` / ``bbox_max_coord``); those
tests are marked ``xfail`` so they are exercised but do not block the
suite. See TODO.md for the fix plan.
"""

from __future__ import annotations

import numpy as np

from astropy.coordinates import SkyCoord


# ---------------------------------------------------------------------------
# 3D path — all *_coord properties work
# ---------------------------------------------------------------------------

def test_3d_birthpix_coord_returns_per_structure_world_coord(
    strucs_3d_two_peaks_h0_wcs,
):
    s = strucs_3d_two_peaks_h0_wcs
    coords = s.birthpix_coord
    # 3D WCS returns a list — celestial SkyCoord and a spectral Quantity.
    assert len(coords) == 2  # [celestial, spectral]
    celestial, spectral = coords
    assert isinstance(celestial, SkyCoord)
    assert len(celestial) == s.n_struc
    assert spectral.shape == (s.n_struc,)


def test_3d_deathpix_coord_returns_per_structure_world_coord(
    strucs_3d_two_peaks_h0_wcs,
):
    s = strucs_3d_two_peaks_h0_wcs
    coords = s.deathpix_coord
    assert len(coords) == 2
    celestial, _ = coords
    assert len(celestial) == s.n_struc


def test_3d_centroid_coord_returns_per_structure_world_coord(
    strucs_3d_two_peaks_h0_wcs,
):
    s = strucs_3d_two_peaks_h0_wcs
    coords = s.centroid_coord
    celestial, _ = coords
    assert len(celestial) == s.n_struc


def test_3d_bbox_min_and_max_coord_return_per_structure_world_coord(
    strucs_3d_two_peaks_h0_wcs,
):
    s = strucs_3d_two_peaks_h0_wcs
    for prop in (s.bbox_min_coord, s.bbox_max_coord):
        celestial, _ = prop
        assert len(celestial) == s.n_struc


def test_3d_equiv_radius_coord_scales_with_pixel_size(
    strucs_3d_two_peaks_h0_wcs,
):
    """Equivalent radius in world units = px_radius * pixel-scale."""
    s = strucs_3d_two_peaks_h0_wcs
    r_pix = s.equiv_radius
    pix_scale = abs(np.diag(s.wcs.pixel_scale_matrix)[0])
    np.testing.assert_allclose(s.equiv_radius_coord, r_pix * pix_scale)


def test_3d_volume_coord_is_positive(strucs_3d_two_peaks_h0_wcs):
    s = strucs_3d_two_peaks_h0_wcs
    vols = s.volume_coord
    assert vols.shape == (s.n_struc,)
    assert np.all(vols > 0)


def test_3d_geom_cent_coord_returns_per_structure_world_coord(
    strucs_3d_two_peaks_h0_wcs,
):
    s = strucs_3d_two_peaks_h0_wcs
    coords = s.geom_cent_coord
    celestial, _ = coords
    assert len(celestial) == s.n_struc


def test_3d_birthpix_coord_round_trips_through_wcs(
    strucs_3d_two_peaks_h0_wcs,
):
    """world_to_pixel(pixel_to_world(birthpix)) ≈ birthpix."""
    s = strucs_3d_two_peaks_h0_wcs
    coords = s.birthpix_coord
    px = np.array(s.wcs.world_to_pixel(*coords))
    np.testing.assert_allclose(px.T, s.birthpix, atol=1e-6)


# ---------------------------------------------------------------------------
# 2D path — partially working; broken methods are xfail'd
# ---------------------------------------------------------------------------

def test_2d_deathpix_coord_returns_skycoord(strucs_2d_two_peaks_h0_wcs):
    s = strucs_2d_two_peaks_h0_wcs
    coords = s.deathpix_coord
    assert isinstance(coords, SkyCoord)
    assert len(coords) == s.n_struc


def test_2d_equiv_radius_coord_scales_with_pixel_size(
    strucs_2d_two_peaks_h0_wcs,
):
    s = strucs_2d_two_peaks_h0_wcs
    r_pix = s.equiv_radius
    pix_scale = abs(np.diag(s.wcs.pixel_scale_matrix)[0])
    np.testing.assert_allclose(s.equiv_radius_coord, r_pix * pix_scale)


def test_2d_volume_coord_is_positive(strucs_2d_two_peaks_h0_wcs):
    s = strucs_2d_two_peaks_h0_wcs
    vols = s.volume_coord
    assert vols.shape == (s.n_struc,)
    assert np.all(vols > 0)


def test_2d_birthpix_coord_returns_skycoord(strucs_2d_two_peaks_h0_wcs):
    s = strucs_2d_two_peaks_h0_wcs
    coords = s.birthpix_coord
    assert isinstance(coords, SkyCoord)
    assert len(coords) == s.n_struc


def test_2d_centroid_coord_returns_skycoord(strucs_2d_two_peaks_h0_wcs):
    s = strucs_2d_two_peaks_h0_wcs
    coords = s.centroid_coord
    assert isinstance(coords, SkyCoord)
    assert len(coords) == s.n_struc


def test_2d_geom_cent_coord_returns_skycoord(strucs_2d_two_peaks_h0_wcs):
    s = strucs_2d_two_peaks_h0_wcs
    coords = s.geom_cent_coord
    assert isinstance(coords, SkyCoord)
    assert len(coords) == s.n_struc


def test_2d_bbox_min_and_max_coord_return_skycoord(strucs_2d_two_peaks_h0_wcs):
    s = strucs_2d_two_peaks_h0_wcs
    for prop in (s.bbox_min_coord, s.bbox_max_coord):
        assert isinstance(prop, SkyCoord)
        assert len(prop) == s.n_struc


def test_2d_birthpix_coord_round_trips_through_wcs(strucs_2d_two_peaks_h0_wcs):
    """world_to_pixel(pixel_to_world(birthpix)) ≈ birthpix for the 2D path."""
    s = strucs_2d_two_peaks_h0_wcs
    coords = s.birthpix_coord
    px = np.array(s.wcs.world_to_pixel(coords))
    np.testing.assert_allclose(px.T, s.birthpix[:, :2], atol=1e-6)
