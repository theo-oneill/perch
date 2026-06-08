"""Edge-case and robustness tests.

These cover the less-exercised branches of ``PH._prep_img`` (NaN-fill and
buff-pixel paths), the ``flip_data=False`` sublevel-set path, empty
``Structures`` collections, and ``verbose=True``.
"""

from __future__ import annotations

import numpy as np
import pytest

from perch.ph import PH
from perch.structures import Structures


# ---------------------------------------------------------------------------
# PH._prep_img branches via prep_img_kwargs
# ---------------------------------------------------------------------------

def test_compute_hom_with_buff_pix(toy_2d_two_peaks):
    """``buff_pix=True`` writes a sentinel value at a corner of the prepped
    image to anchor the essential class. The call should run without
    crashing and still produce the expected H0 generator pair. The path is
    deprecated in favor of ``pad_essential=``, so we also acknowledge the
    DeprecationWarning here."""
    img, _ = toy_2d_two_peaks
    with pytest.warns(DeprecationWarning, match="buff_pix"):
        ph = PH.compute_hom(data=img, verbose=False, pad_essential=False,
                            prep_img_kwargs={"buff_pix": True})
    h0 = ph.generators[ph.generators[:, 0] == 0]
    assert len(h0) >= 2


def test_compute_hom_with_fill_complete_on_nan_data():
    """``fill_complete=True`` replaces NaN pixels with a buffer value before
    PH; the call should run without crashing on a NaN-bearing image."""
    shape = (16, 16)
    y, x = np.indices(shape)
    img = np.exp(-((y - 4) ** 2 + (x - 5) ** 2) / (2 * 1.2 ** 2)).astype(np.float32)
    # Punch a hole of NaN values
    img[10:14, 10:14] = np.nan
    ph = PH.compute_hom(data=img, verbose=False,
                        prep_img_kwargs={"fill_complete": True})
    assert ph.generators is not None
    assert ph.generators.shape[1] == 10


def test_compute_hom_flip_data_false_runs(toy_2d_two_peaks):
    """``flip_data=False`` skips the negate-and-prep step; cripser runs
    directly on the input (sublevel-set PH of the data)."""
    img, _ = toy_2d_two_peaks
    ph = PH.compute_hom(data=img, verbose=False, flip_data=False)
    assert ph.generators is not None
    # Sublevel-set on a non-negative Gaussian image: births at the global
    # minimum (the zero corners). At least one H0 generator should appear.
    assert (ph.generators[:, 0] == 0).any()


def test_compute_hom_verbose_prints(toy_2d_two_peaks, capsys):
    """``verbose=True`` should print progress without crashing."""
    img, _ = toy_2d_two_peaks
    PH.compute_hom(data=img, verbose=True)
    out = capsys.readouterr().out
    assert "Computing PH" in out
    assert "Complete" in out


# ---------------------------------------------------------------------------
# Empty Structures
# ---------------------------------------------------------------------------

def test_empty_structures_construction():
    """Building a Structures from a (0, 10) array should work without crash."""
    empty = np.zeros((0, 10))
    strucs = Structures(structures=empty, img_shape=(8, 8))
    assert strucs.n_struc == 0
    assert list(strucs.structure_keys) == []
    assert strucs.all_structures == []


def test_empty_structures_segmentation_returns_without_error():
    """``compute_segment_hierarchy`` on an empty Structures hits the early-
    return guard and sets struc_map/level_map without raising."""
    empty = np.zeros((0, 10))
    strucs = Structures(structures=empty, img_shape=(8, 8))
    img = np.zeros((8, 8), dtype=np.float32)
    strucs.compute_segment_hierarchy(img_jnp=img, verbose=False, export=False)
    assert strucs.struc_map is not None
    assert strucs.struc_map.shape == (8, 8)
    assert strucs.trunk == []
    assert strucs.leaves == []


# ---------------------------------------------------------------------------
# Category-D feature paths not previously exercised
# ---------------------------------------------------------------------------

def test_compute_hom_with_buff_pix_3d(toy_3d_two_peaks):
    """``buff_pix=True`` 3D path in ``_prep_img``. Deprecated path — see
    ``test_compute_hom_with_buff_pix``."""
    img, _ = toy_3d_two_peaks
    with pytest.warns(DeprecationWarning, match="buff_pix"):
        ph = PH.compute_hom(data=img, verbose=False, pad_essential=False,
                            prep_img_kwargs={"buff_pix": True})
    assert ph.generators is not None
    assert (ph.generators[:, 0] == 0).any()


def test_remove_struc_drops_trunk_with_branch_path(strucs_2d_two_peaks_h0):
    """Removing a *trunk* (not a leaf) walks the branch-handling path in
    ``remove_struc``: level decrement for descendants, parent unlinking
    for children, and ``struc_map[trunk] → NaN``."""
    import copy
    h0 = copy.deepcopy(strucs_2d_two_peaks_h0)
    n_before = h0.n_struc
    trunk_id = h0.trunk[0].id
    leaf_id = h0.leaves[0].id
    leaf_level_before = h0.structures[leaf_id].level

    h0.remove_struc(trunk_id)

    assert h0.n_struc == n_before - 1
    assert trunk_id not in h0.structures
    # Former leaf is now an orphan (parent was the trunk we just removed).
    assert h0.structures[leaf_id].parent is None
    # Levels of all former descendants decremented by 1.
    assert h0.structures[leaf_id].level == leaf_level_before - 1
    # Trunk had no parent, so its pixels in struc_map become NaN.
    assert np.any(np.isnan(h0.struc_map))


def test_export_load_roundtrip_preserves_supp_props(strucs_2d_two_peaks_h0, tmp_path):
    """The hierarchy fixture is built with ``calc_supp_props=True``, so
    sum_val/centroid/bbox are written into the FITS table by
    ``export_segmentation`` and rebuilt by ``load_from``."""
    s = strucs_2d_two_peaks_h0
    odir = str(tmp_path) + "/"
    s.export_segmentation(fname="supp", odir=odir)

    from perch.structures import Structures
    loaded = Structures.load_from(odir=odir, fname="supp", verbose=False)

    # Sort both by id_ph so we can compare row-wise.
    a = np.argsort(loaded.id_ph)
    b = np.argsort(s.id_ph)
    # supplementary scalar stats
    np.testing.assert_allclose(
        np.array([loaded.structures[i]._sum_val for i in np.array(list(loaded.structure_keys))[a]]),
        np.array([s.structures[i]._sum_val for i in np.array(list(s.structure_keys))[b]]),
    )
    # centroids stored as coord stacks
    for sid_a, sid_b in zip(np.array(list(loaded.structure_keys))[a],
                            np.array(list(s.structure_keys))[b]):
        np.testing.assert_allclose(
            loaded.structures[sid_a]._centroid,
            s.structures[sid_b]._centroid,
        )


def test_structure_raises_on_missing_img(toy_2d_two_peaks):
    """All image-dependent ``Structure`` methods now raise ``ValueError``
    instead of printing a string and silently returning."""
    from perch.structure import Structure
    s = Structure(pi=[0, 0.8, 0.2, 3, 4, 0, 0, 0, 0],
                  id=0, id_ph=0, img_shape=(8, 8))
    s._indices = (np.array([0]), np.array([0]))
    s._npix = 1
    with pytest.raises(ValueError, match="img must be provided"):
        s.get_values(img=None)
    with pytest.raises(ValueError, match="img must be provided"):
        s._calculate_centroid(img=None)
    with pytest.raises(ValueError, match="img must be provided"):
        s._calculate_weight_cent(img=None)
    with pytest.raises(ValueError, match="img must be provided"):
        s._calculate_extreme_pix(img=None)


def test_structure_raises_on_missing_segmentation():
    """Methods that need an existing segmentation raise ``RuntimeError``
    when ``_indices`` is unset."""
    from perch.structure import Structure
    s = Structure(pi=[0, 0.8, 0.2, 3, 4, 0, 0, 0, 0],
                  id=0, id_ph=0, img_shape=(8, 8))
    with pytest.raises(RuntimeError, match="segmentation has not been computed"):
        s.get_mask()
    with pytest.raises(RuntimeError, match="segmentation has not been computed"):
        s._calculate_bbox()
    img = np.zeros((8, 8))
    with pytest.raises(RuntimeError, match="segmentation has not been computed"):
        s.get_values(img=img)


def test_structures_coord_props_raise_without_wcs():
    """All ``*_coord`` properties raise ``ValueError`` when no WCS is set."""
    gens = np.array([[0.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    strucs = Structures(structures=gens, img_shape=(8, 8))
    assert strucs.wcs is None
    for prop in ("birthpix_coord", "deathpix_coord", "geom_cent_coord",
                 "centroid_coord", "bbox_min_coord", "bbox_max_coord",
                 "equiv_radius_coord", "volume_coord"):
        with pytest.raises(ValueError, match="WCS"):
            getattr(strucs, prop)


def test_structures_export_raises_without_hierarchy(tmp_path):
    """The map-export methods raise ``RuntimeError`` when the maps
    were never populated by ``compute_segment_hierarchy``."""
    gens = np.array([[0.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    strucs = Structures(structures=gens, img_shape=(8, 8))
    odir = str(tmp_path) + "/"
    with pytest.raises(RuntimeError, match="hierarchy"):
        strucs.export_struc_map(fname="x", odir=odir)
    with pytest.raises(RuntimeError, match="hierarchy"):
        strucs.export_level_map(fname="x", odir=odir)


# ---------------------------------------------------------------------------
# Big-endian / FITS-loaded image support
# (FITS is spec'd big-endian; astropy hands back non-native dtypes.)
# ---------------------------------------------------------------------------

def test_pipeline_handles_explicit_big_endian(toy_2d_two_peaks):
    """Forcing non-native byteorder still produces equivalent generators
    and a valid segmentation through every entry point. Locks in the
    ``if not img.dtype.isnative`` guards in ``compute_segment`` and
    ``compute_segment_hierarchy``."""
    img, _ = toy_2d_two_peaks
    img_be = img.astype(img.dtype.newbyteorder(">"))
    assert not img_be.dtype.isnative

    # PH.compute_hom on big-endian: should match native output bit-for-bit.
    ph_be = PH.compute_hom(data=img_be, verbose=False)
    ph_native = PH.compute_hom(data=img, verbose=False)
    np.testing.assert_allclose(ph_be.generators, ph_native.generators)

    # Structure.compute_segment on big-endian.
    h0 = ph_be.generators[(ph_be.generators[:, 0] == 0)
                          & (ph_be.generators[:, 2] > -1e30)][0]
    struc = ph_be.strucs.structures[int(h0[9])]
    struc.compute_segment(img_be)
    assert struc.npix > 0

    # Structures.compute_segment_hierarchy on big-endian.
    h0_all = ph_be.filter(dimension=0)
    h0_all.compute_segment_hierarchy(img_jnp=img_be, verbose=False, export=False)
    assert h0_all.struc_map.shape == img.shape
    assert h0_all.n_struc == ph_be.filter(dimension=0).n_struc


def test_fits_load_pipeline_end_to_end(toy_2d_two_peaks, tmp_path):
    """Realistic astronomer workflow: write a FITS file, load it via
    astropy, run the whole PH + segmentation pipeline on what astropy
    returns. Validates that whatever byteorder astropy emits, perch
    handles it transparently."""
    from astropy.io import fits

    img, _ = toy_2d_two_peaks
    fits_path = tmp_path / "data.fits"
    fits.PrimaryHDU(img).writeto(fits_path)
    loaded = fits.getdata(fits_path)
    assert loaded.shape == img.shape

    ph = PH.compute_hom(data=loaded, verbose=False)
    h0 = ph.filter(dimension=0)
    h0.compute_segment_hierarchy(img_jnp=loaded, verbose=False, export=False)

    assert h0.n_struc >= 1
    assert h0.struc_map is not None
    assert h0.struc_map.shape == loaded.shape


def test_surface_area_save_points_writes_verts_file(ph_3d_shell, toy_3d_shell, tmp_path):
    """``_calculate_surface_area(save_points=True)`` writes a
    ``struc_<id_ph>_verts.txt`` file containing the marching-cubes vertices."""
    img, _, _ = toy_3d_shell
    h2 = ph_3d_shell.filter(dimension=2)
    h2.compute_segment_hierarchy(img_jnp=img, verbose=False, export=False)
    struc = h2.all_structures[0]
    struc.compute_segment(img)

    sdir = str(tmp_path) + "/"
    struc._calculate_surface_area(save_points=True, sdir=sdir)

    verts_file = tmp_path / f"struc_{struc.id_ph}_verts.txt"
    assert verts_file.exists()
    verts = np.loadtxt(verts_file)
    # Marching cubes on a non-trivial mask produces many vertices, each (z, y, x).
    assert verts.shape[1] == 3
    assert verts.shape[0] > 10
