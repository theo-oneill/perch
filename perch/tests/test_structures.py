"""Unit tests for the ``Structures`` collection class.

Construction-path tests use small synthetic generator tables so the assertions
are independent of cripser. Aggregate-property and sorting tests run against
both synthetic tables and the session-shared PH fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from perch.structure import Structure
from perch.structures import Structures


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_generator_table(n=3):
    """Build a small (n, 10) generator table with strictly increasing births,
    increasing deaths, and distinct birth pixels."""
    rows = []
    for i in range(n):
        rows.append([
            0,                       # htype
            0.5 + 0.1 * i,           # birth
            0.10 + 0.05 * i,         # death
            i, i + 1, 0,             # birthpix
            0, 0, 0,                 # deathpix
            float(i),                # h_id
        ])
    return np.array(rows)


# ---------------------------------------------------------------------------
# Three construction paths
# ---------------------------------------------------------------------------

def test_construction_from_ndarray_assigns_ids():
    gens = _make_generator_table(3)
    strucs = Structures(structures=gens, img_shape=(8, 8))
    assert strucs.n_struc == 3
    assert sorted(strucs.structure_keys) == [0, 1, 2]
    for i in range(3):
        assert strucs.structures[i].id == i
        assert strucs.structures[i].id_ph == i
        assert strucs.structures[i].htype == 0


def test_construction_from_list_of_structures():
    sl = [
        Structure(pi=[0, 0.5 + 0.1 * i, 0.1, i, i, 0, 0, 0, 0],
                  id=i, id_ph=i, img_shape=(8, 8))
        for i in range(3)
    ]
    strucs = Structures(structures=sl, img_shape=(8, 8))
    assert strucs.n_struc == 3
    for i in range(3):
        assert strucs.structures[i] is sl[i]


def test_construction_from_dict_preserves_keys():
    d = {
        i: Structure(pi=[0, 0.5 + 0.1 * i, 0.1, i, i, 0, 0, 0, 0],
                     id=i, id_ph=i, img_shape=(8, 8))
        for i in range(3)
    }
    strucs = Structures(structures=d, img_shape=(8, 8))
    assert strucs.n_struc == 3
    assert set(strucs.structure_keys) == set(d.keys())


# ---------------------------------------------------------------------------
# Aggregate property arrays
# ---------------------------------------------------------------------------

def test_aggregate_arrays_match_input_table():
    gens = _make_generator_table(4)
    strucs = Structures(structures=gens, img_shape=(8, 8))

    np.testing.assert_array_equal(strucs.htype, gens[:, 0].astype(int))
    np.testing.assert_allclose(strucs.birth, gens[:, 1])
    np.testing.assert_allclose(strucs.death, gens[:, 2])
    np.testing.assert_allclose(strucs.persistence, np.abs(gens[:, 2] - gens[:, 1]))
    np.testing.assert_array_equal(strucs.id_ph, gens[:, 9].astype(int))
    assert strucs.birthpix.shape == (4, 3)
    assert strucs.deathpix.shape == (4, 3)
    np.testing.assert_array_equal(strucs.birthpix[2], [2, 3, 0])


def test_aggregate_arrays_on_real_fixture(ph_2d_two_peaks):
    s = ph_2d_two_peaks.strucs
    assert s.n_struc == len(ph_2d_two_peaks.generators)
    np.testing.assert_array_equal(np.sort(s.birth),
                                  np.sort(ph_2d_two_peaks.generators[:, 1]))
    np.testing.assert_array_equal(np.sort(s.htype),
                                  np.sort(ph_2d_two_peaks.generators[:, 0]).astype(int))


def test_norm_life_equals_persistence_over_birth():
    gens = _make_generator_table(3)
    strucs = Structures(structures=gens, img_shape=(8, 8))
    np.testing.assert_allclose(strucs.norm_life,
                               strucs.persistence / strucs.birth)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def test_sort_birth_ascending_and_descending():
    gens = _make_generator_table(4)
    strucs = Structures(structures=gens, img_shape=(8, 8))
    asc = strucs.sort_birth(invert=False)
    desc = strucs.sort_birth(invert=True)
    np.testing.assert_array_equal(np.array(strucs.birth)[asc],
                                  np.sort(strucs.birth))
    np.testing.assert_array_equal(np.array(strucs.birth)[desc],
                                  np.sort(strucs.birth)[::-1])


def test_sort_death_ascending():
    gens = _make_generator_table(4)
    strucs = Structures(structures=gens, img_shape=(8, 8))
    asc = strucs.sort_death(invert=False)
    np.testing.assert_array_equal(np.array(strucs.death)[asc],
                                  np.sort(strucs.death))


# ---------------------------------------------------------------------------
# DataFrame / generators round-trip
# ---------------------------------------------------------------------------

def test_make_df_has_required_columns():
    gens = _make_generator_table(3)
    strucs = Structures(structures=gens, img_shape=(8, 8))
    for s in strucs.all_structures:
        s._npix = 1
    strucs._make_df()
    required = {"ID_PH", "ID", "Htype", "Birth", "Death",
                "Birthpix_0", "Birthpix_1", "Birthpix_2",
                "Deathpix_0", "Deathpix_1", "Deathpix_2",
                "Npix", "Parent_ID", "Level"}
    assert required.issubset(set(strucs.df.columns))


def test_generators_method_reconstructs_input_table():
    gens = _make_generator_table(3)
    strucs = Structures(structures=gens, img_shape=(8, 8))
    for s in strucs.all_structures:
        s._npix = 1
    reconstructed = strucs.generators()
    np.testing.assert_allclose(reconstructed, gens)


# ---------------------------------------------------------------------------
# add_attributes — user-supplied per-structure properties
# ---------------------------------------------------------------------------

def test_add_attributes_propagates_to_each_structure_and_aggregate():
    """``add_attributes`` sets each non-ID column as both a per-structure
    attribute and an aggregate array on the Structures."""
    import pandas as pd

    gens = _make_generator_table(3)
    strucs = Structures(structures=gens, img_shape=(8, 8))
    df = pd.DataFrame({"ID": [0, 1, 2],
                       "custom_metric": [10.0, 20.0, 30.0]})
    strucs.add_attributes(df)

    for i in range(3):
        assert strucs.structures[i].custom_metric == pytest.approx(10.0 * (i + 1))
    np.testing.assert_array_equal(strucs.custom_metric, [10.0, 20.0, 30.0])


# ---------------------------------------------------------------------------
# Aggregate-mask methods on a segmented fixture
# ---------------------------------------------------------------------------

def test_get_mask_with_descendants_rolls_up_subtree(strucs_2d_two_peaks_h0):
    """``get_mask(s_include=[trunk_id])`` with descendants includes leaf pixels."""
    s = strucs_2d_two_peaks_h0
    trunk_id = s.trunk[0].id
    mask = s.get_mask(s_include=[trunk_id], use_descendants=True)
    assert mask.shape == s._imgshape
    # The trunk's segmentation is the whole image (essential class).
    assert int(mask.sum()) == int(np.prod(s._imgshape))


def test_get_mask_without_descendants_excludes_subtree(strucs_2d_two_peaks_h0):
    """Without descendants, asking for the trunk only returns pixels labelled
    with the trunk's id in struc_map — *not* the leaf's pixels (those are
    labelled with the leaf's id because it overwrote struc_map last)."""
    s = strucs_2d_two_peaks_h0
    trunk_id = s.trunk[0].id
    leaf_id = s.leaves[0].id
    mask = s.get_mask(s_include=[trunk_id], use_descendants=False)
    # All leaf-pixels should NOT be in the mask.
    leaf_pixels = (s.struc_map == leaf_id)
    assert not np.any(mask & leaf_pixels)


def test_get_struc_map_mask_preserves_struc_ids(strucs_2d_two_peaks_h0):
    """``get_struc_map_mask`` returns the struc_map subset for the requested
    ids, NaN elsewhere."""
    s = strucs_2d_two_peaks_h0
    leaf_id = s.leaves[0].id
    smap_mask = s.get_struc_map_mask(s_include=[leaf_id], use_descendants=False)
    # Where defined, values equal the leaf id; elsewhere NaN.
    defined = np.isfinite(smap_mask)
    assert defined.any()
    np.testing.assert_array_equal(np.unique(smap_mask[defined]).astype(int),
                                  [leaf_id])


def test_id_mask_returns_ids_for_selected_structures(strucs_2d_two_peaks_h0):
    """``id_mask`` carries each selected structure's id at its pixels,
    NaN elsewhere — no descendants rollup."""
    s = strucs_2d_two_peaks_h0
    leaf_id = s.leaves[0].id
    mask = s.id_mask(s_include=[leaf_id])
    assert mask.shape == s._imgshape
    defined = np.isfinite(mask)
    assert defined.any()
    np.testing.assert_array_equal(np.unique(mask[defined]).astype(int),
                                  [leaf_id])


def test_id_mask_default_covers_full_struc_map(strucs_2d_two_peaks_h0):
    """``id_mask()`` (no s_include) returns the full struc_map ids."""
    s = strucs_2d_two_peaks_h0
    mask = s.id_mask()
    defined = np.isfinite(mask)
    smap_defined = np.isfinite(s.struc_map)
    np.testing.assert_array_equal(defined, smap_defined)
    np.testing.assert_array_equal(mask[defined], s.struc_map[smap_defined])


def test_id_mask_raises_without_hierarchy():
    """``id_mask`` on a Structures without a struc_map raises RuntimeError."""
    gens = np.array([[0.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    strucs = Structures(structures=gens, img_shape=(8, 8))
    with pytest.raises(RuntimeError, match="hierarchy"):
        strucs.id_mask()


# ---------------------------------------------------------------------------
# remove_struc / remove_strucs
# ---------------------------------------------------------------------------

def _mutable_copy(strucs):
    """Return a deepcopy of an already-segmented Structures so tests can
    mutate it without leaking state to other session-scoped consumers.
    Avoids re-paying the JAX compile cost of a fresh compute_segment_hierarchy."""
    import copy
    return copy.deepcopy(strucs)


def test_remove_struc_drops_leaf_and_unlinks_parent(strucs_2d_two_peaks_h0):
    h0 = _mutable_copy(strucs_2d_two_peaks_h0)
    n_before = h0.n_struc
    leaf_id = h0.leaves[0].id
    trunk_id = h0.trunk[0].id

    h0.remove_struc(leaf_id)

    assert h0.n_struc == n_before - 1
    assert leaf_id not in h0.structures
    assert leaf_id not in h0.structures[trunk_id].children
    assert leaf_id not in h0.structures[trunk_id].descendants
    # The leaf's pixels in struc_map should now carry the parent's id.
    assert not np.any(h0.struc_map == leaf_id)


def test_remove_strucs_bulk_drops_multiple(strucs_2d_two_peaks_h0):
    h0 = _mutable_copy(strucs_2d_two_peaks_h0)
    all_ids = list(h0.structure_keys)
    leaf_ids = [s.id for s in h0.leaves]

    h0.remove_strucs(leaf_ids)

    assert h0.n_struc == len(all_ids) - len(leaf_ids)
    for lid in leaf_ids:
        assert lid not in h0.structures


# ---------------------------------------------------------------------------
# clear_struc_map
# ---------------------------------------------------------------------------

def test_clear_struc_map(strucs_2d_two_peaks_h0):
    h0 = _mutable_copy(strucs_2d_two_peaks_h0)
    assert h0.struc_map is not None
    h0.clear_struc_map()
    assert h0.struc_map is None
