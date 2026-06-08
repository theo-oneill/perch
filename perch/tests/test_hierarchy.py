"""Tests for ``Structures.compute_segment_hierarchy``.

The two-peaks fixture produces a clean hierarchy: the essential class (whole-
image segmentation) is the trunk, and the finite-death H0 generator is its
only leaf child. The session-scoped ``strucs_2d_two_peaks_h0`` fixture in
``conftest.py`` performs the segmentation once and shares the result here.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_struc_map_and_level_map_shapes(strucs_2d_two_peaks_h0):
    s = strucs_2d_two_peaks_h0
    assert s.struc_map is not None
    assert s.level_map is not None
    assert s.struc_map.shape == s._imgshape
    assert s.level_map.shape == s._imgshape


def test_trunk_and_leaves_single_branch(strucs_2d_two_peaks_h0):
    """Two-peaks fixture: 1 trunk (essential class), 1 leaf (lower peak)."""
    s = strucs_2d_two_peaks_h0
    assert len(s.trunk) == 1
    assert len(s.leaves) == 1
    trunk = s.trunk[0]
    leaf = s.leaves[0]
    assert trunk.id != leaf.id
    assert leaf.parent == trunk.id
    assert leaf.id in trunk.children
    # Birth of the trunk is the higher peak (amp=1.0); leaf is the lower peak.
    assert trunk.birth == pytest.approx(1.0)
    assert leaf.birth == pytest.approx(0.6, rel=1e-3)


def test_levels_are_one_and_two(strucs_2d_two_peaks_h0):
    """Trunk lives at level 1; its only child lives at level 2."""
    s = strucs_2d_two_peaks_h0
    levels = sorted(int(struc.level) for struc in s.all_structures)
    assert levels == [1, 2]


def test_descendants_propagate_transitively(strucs_2d_two_peaks_h0):
    """`descendants` is currently self-inclusive (see TODO.md). Trunk's list
    contains itself plus the leaf; the leaf's list is just itself."""
    s = strucs_2d_two_peaks_h0
    trunk = s.trunk[0]
    leaf = s.leaves[0]
    assert trunk.id in trunk.descendants
    assert leaf.id in trunk.descendants
    assert trunk.n_descendants == 2
    assert leaf.descendants == [leaf.id]
    assert leaf.n_descendants == 1


def test_parent_npix_is_at_least_child_npix(strucs_2d_two_peaks_h0):
    """The parent's segmentation must contain the child's."""
    s = strucs_2d_two_peaks_h0
    for struc in s.all_structures:
        if struc.parent is not None:
            assert s.structures[struc.parent].npix >= struc.npix


def test_struc_map_values_are_valid_ids(strucs_2d_two_peaks_h0):
    s = strucs_2d_two_peaks_h0
    finite = np.isfinite(s.struc_map)
    assert finite.any()
    for uid in np.unique(s.struc_map[finite]).astype(int):
        assert uid in s.structures


def test_struc_map_unique_id_count_matches_n_struc(strucs_2d_two_peaks_h0):
    """Every structure has at least one pixel in struc_map, and every pixel-id
    in struc_map is a structure id. Together: ``len(unique(struc_map)) == n_struc``."""
    s = strucs_2d_two_peaks_h0
    finite = s.struc_map[np.isfinite(s.struc_map)]
    n_unique = len(np.unique(finite))
    assert n_unique == s.n_struc, (
        f"struc_map has {n_unique} unique ids but Structures contains {s.n_struc} structures"
    )


def test_mixed_homology_groups_warns(ph_2d_ring, toy_2d_ring):
    """A Structures spanning multiple homology dimensions should warn."""
    img, _, _ = toy_2d_ring
    # Use the unfiltered collection (H0 + H1)
    mixed = ph_2d_ring.strucs
    with pytest.warns(UserWarning, match="multiple homology groups"):
        mixed.compute_segment_hierarchy(img_jnp=img, verbose=False,
                                        export=False, clobber=True)


# ---------------------------------------------------------------------------
# Diagnostic plot (opt-in via `pytest --perch-plots`).
# ---------------------------------------------------------------------------

def test_plot_segmentation_2d_two_peaks(strucs_2d_two_peaks_h0,
                                        toy_2d_two_peaks, plot_dir):
    if plot_dir is None:
        pytest.skip("pass --perch-plots to enable diagnostic plot output")
    from perch.tests import _plotting
    img, _ = toy_2d_two_peaks
    _plotting.plot_segmentation_2d(
        plot_dir / "2d_two_peaks_segmentation_legacy.png",
        img,
        strucs_2d_two_peaks_h0,
    )


# ---------------------------------------------------------------------------
# Segmentation assertions on the other three fixtures
# (struc_map shape + unique-id count, matching the 2D two-peaks coverage).
# ---------------------------------------------------------------------------

def test_3d_two_peaks_struc_map_shape_and_id_count(strucs_3d_two_peaks_h0,
                                                    toy_3d_two_peaks):
    img, _ = toy_3d_two_peaks
    s = strucs_3d_two_peaks_h0
    assert s.struc_map.shape == img.shape
    assert s.level_map.shape == img.shape
    finite = s.struc_map[np.isfinite(s.struc_map)]
    assert len(np.unique(finite)) == s.n_struc


def test_2d_ring_struc_map_shape_and_id_count(strucs_2d_ring_h1, toy_2d_ring):
    img, _, _ = toy_2d_ring
    s = strucs_2d_ring_h1
    assert s.struc_map.shape == img.shape
    assert s.level_map.shape == img.shape
    finite = s.struc_map[np.isfinite(s.struc_map)]
    assert len(np.unique(finite)) == s.n_struc
    # Only the H1 cycle was kept, so exactly one structure should be segmented.
    assert s.n_struc == 1


def test_3d_shell_struc_map_shape_and_id_count(strucs_3d_shell_h2, toy_3d_shell):
    img, _, _ = toy_3d_shell
    s = strucs_3d_shell_h2
    assert s.struc_map.shape == img.shape
    assert s.level_map.shape == img.shape
    finite = s.struc_map[np.isfinite(s.struc_map)]
    assert len(np.unique(finite)) == s.n_struc
    assert s.n_struc == 1


# ---------------------------------------------------------------------------
# Segmentation plots for the other three fixtures (opt-in via --perch-plots).
# ---------------------------------------------------------------------------

def test_plot_segmentation_3d_two_peaks(strucs_3d_two_peaks_h0,
                                        toy_3d_two_peaks, plot_dir):
    if plot_dir is None:
        pytest.skip("pass --perch-plots to enable diagnostic plot output")
    from perch.tests import _plotting
    img, _ = toy_3d_two_peaks
    _plotting.plot_segmentation_3d(
        plot_dir / "3d_two_peaks_segmentation_legacy.png",
        img,
        strucs_3d_two_peaks_h0,
    )


def test_plot_segmentation_2d_ring(strucs_2d_ring_h1, toy_2d_ring, plot_dir):
    if plot_dir is None:
        pytest.skip("pass --perch-plots to enable diagnostic plot output")
    from perch.tests import _plotting
    img, _, _ = toy_2d_ring
    _plotting.plot_segmentation_2d(
        plot_dir / "2d_ring_h1_segmentation.png",
        img,
        strucs_2d_ring_h1,
    )


def test_plot_segmentation_3d_shell(strucs_3d_shell_h2, toy_3d_shell, plot_dir):
    if plot_dir is None:
        pytest.skip("pass --perch-plots to enable diagnostic plot output")
    from perch.tests import _plotting
    img, _, _ = toy_3d_shell
    _plotting.plot_segmentation_3d(
        plot_dir / "3d_shell_h2_segmentation.png",
        img,
        strucs_3d_shell_h2,
    )
