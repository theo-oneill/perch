"""Unit tests for the ``Structure`` class.

Most of these tests build ``Structure`` instances directly from a ``pi`` row
and hand-set ``_indices`` to isolate the property-computation logic from
``compute_segment``. The ``compute_segment`` path itself is exercised
against the session-shared PH fixtures further down.
"""

from __future__ import annotations

import numpy as np
import pytest

from perch.structure import Structure


def _make_structure(pi, img_shape=(8, 8), id=0, id_ph=0):
    return Structure(pi=pi, id=id, id_ph=id_ph, img_shape=img_shape)


def _set_indices(struc, indices_tuple):
    struc._indices = indices_tuple
    struc._npix = len(indices_tuple[0])


# ---------------------------------------------------------------------------
# Construction + basic properties
# ---------------------------------------------------------------------------

def test_construction_2d_basic_properties():
    pi = [0, 0.8, 0.2, 3, 4, 0, 1, 2, 0]
    s = _make_structure(pi, img_shape=(8, 8), id=5, id_ph=42)
    assert s.htype == 0
    assert s.birth == pytest.approx(0.8)
    assert s.death == pytest.approx(0.2)
    assert s.id == 5
    assert s.id_ph == 42
    np.testing.assert_array_equal(s.birthpix, [3, 4, 0])
    np.testing.assert_array_equal(s.deathpix, [1, 2, 0])
    assert s.birthpix.dtype.kind == "i"
    assert s.deathpix.dtype.kind == "i"


def test_construction_3d_ndim():
    s = _make_structure([0, 0.8, 0.2, 3, 4, 5, 0, 0, 0], img_shape=(8, 8, 8))
    assert s._ndim == 3
    np.testing.assert_array_equal(s.birthpix, [3, 4, 5])


def test_persistence_is_abs_diff_for_h0_and_h1():
    # H0: birth > death convention with flip_data=True
    s0 = _make_structure([0, 0.8, 0.2, 3, 4, 0, 1, 2, 0])
    assert s0.persistence == pytest.approx(0.6)
    # H1 in 2D: birth < death (saddle birth, interior death)
    s1 = _make_structure([1, 0.2, 0.8, 3, 4, 0, 1, 2, 0])
    assert s1.persistence == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Cache reset
# ---------------------------------------------------------------------------

def test_reset_cache_clears_cached_values():
    s = _make_structure([0, 0.8, 0.2, 3, 4, 0, 1, 2, 0])
    s._npix = 42
    s._geom_cent = np.array([1.0, 2.0])
    s._mask = np.ones((8, 8), bool)
    s._children = [1, 2]
    s._reset_cache()
    assert s._npix is None
    assert s._geom_cent is None
    assert s._mask is None
    assert s._children == []


# ---------------------------------------------------------------------------
# Geometry from hand-set indices
# ---------------------------------------------------------------------------

def test_bbox_2d_from_indices():
    s = _make_structure([0, 0.8, 0.2, 0, 0, 0, 0, 0, 0])
    _set_indices(s, (np.array([2, 3, 4]), np.array([5, 6, 5])))
    np.testing.assert_array_equal(s.bbox_min, [2, 5])
    np.testing.assert_array_equal(s.bbox_max, [4, 6])
    np.testing.assert_array_equal(s.bbox, [[2, 5], [4, 6]])


def test_bbox_3d_from_indices():
    s = _make_structure([0, 0.8, 0.2, 0, 0, 0, 0, 0, 0], img_shape=(8, 8, 8))
    _set_indices(s, (np.array([1, 4]), np.array([2, 6]), np.array([3, 7])))
    np.testing.assert_array_equal(s.bbox_min, [1, 2, 3])
    np.testing.assert_array_equal(s.bbox_max, [4, 6, 7])


def test_geom_cent_is_mean_of_indices():
    s = _make_structure([0, 0.8, 0.2, 0, 0, 0, 0, 0, 0])
    _set_indices(s, (np.array([1, 3]), np.array([2, 6])))
    np.testing.assert_allclose(s.geom_cent, [2.0, 4.0])


def test_equiv_radius_2d_inverts_circle_area():
    s = _make_structure([0, 0.8, 0.2, 0, 0, 0, 0, 0, 0], img_shape=(64, 64))
    R = 5.0
    s._npix = int(np.pi * R ** 2)
    assert s.equiv_radius == pytest.approx((s.npix / np.pi) ** 0.5)


def test_equiv_radius_3d_inverts_sphere_volume():
    s = _make_structure([0, 0.8, 0.2, 0, 0, 0, 0, 0, 0], img_shape=(32, 32, 32))
    R = 3.0
    s._npix = int(4 / 3 * np.pi * R ** 3)
    assert s.equiv_radius == pytest.approx(
        (3 * s.npix / (4 * np.pi)) ** (1 / 3)
    )


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def test_get_mask_marks_only_index_pixels():
    s = _make_structure([0, 0.8, 0.2, 0, 0, 0, 0, 0, 0], img_shape=(5, 5))
    indices = (np.array([1, 1, 2]), np.array([2, 3, 2]))
    _set_indices(s, indices)
    mask = s.get_mask()
    assert mask.shape == (5, 5)
    assert mask.dtype == bool
    assert int(mask.sum()) == 3
    assert mask[1, 2] and mask[1, 3] and mask[2, 2]
    assert not mask[0, 0]


# ---------------------------------------------------------------------------
# Image-dependent stats
# ---------------------------------------------------------------------------

def test_get_values_returns_image_at_indices():
    s = _make_structure([0, 0.8, 0.2, 0, 0, 0, 0, 0, 0], img_shape=(4, 4))
    img = np.arange(16, dtype=float).reshape(4, 4)
    _set_indices(s, (np.array([1, 2]), np.array([1, 2])))
    np.testing.assert_array_equal(s.get_values(img=img), [img[1, 1], img[2, 2]])


def test_pix_value_stats_match_hand_computed():
    s = _make_structure([0, 0.8, 0.2, 0, 0, 0, 0, 0, 0], img_shape=(4, 4))
    img = np.arange(1, 17, dtype=float).reshape(4, 4)
    _set_indices(s, (np.array([0, 1, 2]), np.array([0, 1, 2])))
    # values at (0,0)(1,1)(2,2) = 1, 6, 11
    s._calculate_pix_values(img=img)
    assert s.sum_val == pytest.approx(18)
    assert s.min_val == pytest.approx(1)
    assert s.max_val == pytest.approx(11)
    assert s.median_val == pytest.approx(6)


# ---------------------------------------------------------------------------
# compute_segment against real PH fixtures
# ---------------------------------------------------------------------------

def test_compute_segment_2d_h0_on_lower_peak(ph_2d_two_peaks, toy_2d_two_peaks):
    """The finite-death H0 generator segments to a blob containing its birthpix
    and all of whose pixels exceed the death threshold."""
    img, _ = toy_2d_two_peaks
    gens = ph_2d_two_peaks.generators
    h0_finite = gens[(gens[:, 0] == 0) & (gens[:, 2] > -1e30)]
    assert h0_finite.shape[0] == 1
    h_id = int(h0_finite[0, 9])
    struc = ph_2d_two_peaks.strucs.structures[h_id]

    struc.compute_segment(img)
    assert struc.indices is not None
    assert struc.npix > 0

    mask = struc.get_mask()
    bp = struc.birthpix
    assert mask[bp[0], bp[1]]
    np.testing.assert_array_less(struc.death, struc.get_values(img=img))


def test_compute_segment_2d_h1_on_ring(ph_2d_ring, toy_2d_ring):
    """H1 in 2D: filter is sublevel-set at birth → all segmented pixels lie
    below the birth threshold."""
    img, _, _ = toy_2d_ring
    gens = ph_2d_ring.generators
    h1 = gens[gens[:, 0] == 1]
    assert h1.shape[0] == 1
    h_id = int(h1[0, 9])
    struc = ph_2d_ring.strucs.structures[h_id]

    struc.compute_segment(img)
    assert struc.indices is not None
    assert struc.npix > 0
    np.testing.assert_array_less(struc.get_values(img=img), struc.birth)


def test_compute_segment_3d_h1_raises_not_implemented():
    s = _make_structure([1, 0.5, 0.1, 3, 4, 5, 0, 0, 0], img_shape=(8, 8, 8))
    img = np.zeros((8, 8, 8), dtype=np.float32)
    with pytest.raises(NotImplementedError):
        s.compute_segment(img)


# ---------------------------------------------------------------------------
# Hierarchy stubs on a fresh Structure
# ---------------------------------------------------------------------------

def test_hierarchy_defaults_are_empty():
    s = _make_structure([0, 0.8, 0.2, 0, 0, 0, 0, 0, 0])
    assert s.parent is None
    assert s.children == []
    assert s.descendants == []
    assert s.n_children == 0
    assert s.n_descendants == 0
    assert s.is_leaf is True


def test_hierarchy_with_children_set_marks_non_leaf():
    s = _make_structure([0, 0.8, 0.2, 0, 0, 0, 0, 0, 0])
    s._children = [1, 2, 3]
    assert s.n_children == 3
    assert s.is_leaf is False


# ---------------------------------------------------------------------------
# extreme_pix — htype-dependent argmax/argmin pixel
# ---------------------------------------------------------------------------

def test_extreme_pix_h0_returns_argmax_pixel():
    """For H0 structures, extreme_pix is the (row, col, ...) of the brightest
    pixel among the structure's indices."""
    s = _make_structure([0, 0.8, 0.2, 0, 0, 0, 0, 0, 0], img_shape=(4, 4))
    img = np.arange(1, 17, dtype=float).reshape(4, 4)
    # pixels at (0,0), (1,1), (2,2) → values 1, 6, 11; argmax → (2, 2)
    _set_indices(s, (np.array([0, 1, 2]), np.array([0, 1, 2])))
    s._calculate_extreme_pix(img=img)
    assert s.extreme_pix == (2, 2)


def test_extreme_pix_h2_returns_argmin_pixel():
    """For H2 (void) structures, extreme_pix is the pixel with the
    *minimum* value in the structure."""
    s = _make_structure([2, 0.8, 0.2, 0, 0, 0, 0, 0, 0], img_shape=(4, 4))
    img = np.arange(1, 17, dtype=float).reshape(4, 4)
    _set_indices(s, (np.array([0, 1, 2]), np.array([0, 1, 2])))
    s._calculate_extreme_pix(img=img)
    assert s.extreme_pix == (0, 0)  # value 1 is the minimum
