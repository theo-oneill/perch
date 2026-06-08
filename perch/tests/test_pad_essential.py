"""Tests for the ``pad_essential`` kwarg on ``PH.compute_hom``.

These cover the new default behavior introduced with the kwarg: the H_0
generator born at ``nanmax(data)`` gets a finite, data-driven death by
running a second H_0-only cripser pass on a padded copy of the input.

Invariants verified here:

* Non-essential generator rows are bit-identical between ``False`` and
  ``'auto'`` — the patch only touches the originally-essential row.
* The infilled death is a property of the data, not of ``pad_value``;
  varying ``pad_value`` over a wide range produces the same death.
* ``'auto'`` picks ``'dilate'`` when NaN voxels exist, ``'bbox'``
  otherwise.
* ``flip_data=False`` rejects explicit ``'dilate'``/``'bbox'`` (the
  superlevel convention is required) and silently disables ``'auto'``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from perch.ph import PH


_ESSENTIAL_THRESHOLD = -1e30


def _essential_row(gens, target_birth):
    mask = (gens[:, 0] == 0) & (gens[:, 1] == target_birth)
    rows = gens[mask]
    assert rows.shape[0] == 1, (
        f"expected exactly one H_0 row born at {target_birth!r}, got {rows.shape[0]}"
    )
    return rows[0]


# ---------------------------------------------------------------------------
# Auto-resolution
# ---------------------------------------------------------------------------

def test_auto_picks_bbox_for_all_finite(toy_2d_two_peaks):
    img, _ = toy_2d_two_peaks
    ph = PH.compute_hom(data=img, verbose=False)
    assert ph.pad_essential == 'bbox'


def test_auto_picks_dilate_for_nan_halo(toy_3d_nan_halo):
    img, _ = toy_3d_nan_halo
    ph = PH.compute_hom(data=img, verbose=False)
    assert ph.pad_essential == 'dilate'


def test_auto_picks_bbox_for_all_finite_3d(toy_3d_two_peaks):
    """3D all-finite cube hits the bbox branch of 'auto'."""
    img, _ = toy_3d_two_peaks
    ph = PH.compute_hom(data=img, verbose=False)
    assert ph.pad_essential == 'bbox'
    target = np.nanmax(img)
    row = _essential_row(ph.generators, target)
    assert np.isfinite(row[2])
    assert row[2] > _ESSENTIAL_THRESHOLD


def test_auto_picks_dilate_for_nan_halo_2d():
    """2D NaN-haloed image hits the dilate branch of 'auto'."""
    y, x = np.indices((20, 20))
    img = np.exp(-((y - 10) ** 2 + (x - 10) ** 2) / (2 * 2.0 ** 2)).astype(np.float64)
    nan_mask = (y < 3) | (y > 16) | (x < 3) | (x > 16)
    img = np.where(nan_mask, np.nan, img)
    ph = PH.compute_hom(data=img, verbose=False)
    assert ph.pad_essential == 'dilate'
    target = np.nanmax(img)
    row = _essential_row(ph.generators, target)
    assert np.isfinite(row[2])
    assert row[2] > _ESSENTIAL_THRESHOLD


def test_dilate_handles_interior_nan_pixel():
    """A NaN-haloed 2D image with an additional NaN voxel inside the finite
    region still runs and still patches the originally-essential row to a
    finite death. The interior NaN gets filled by the dilation step (it sits
    on the dilated-edge of the surrounding finite voxels) but the patching
    logic continues to identify the originally-essential row by its
    original-frame birthpix, so the patch lands correctly.
    """
    y, x = np.indices((20, 20))
    img = np.exp(-((y - 10) ** 2 + (x - 10) ** 2) / (2 * 2.0 ** 2)).astype(np.float64)
    nan_mask = (y < 3) | (y > 16) | (x < 3) | (x > 16)
    img = np.where(nan_mask, np.nan, img)
    # Punch one extra NaN inside the finite region, well away from the peak
    # so it doesn't confuse identification of the maximum-birth voxel.
    img[6, 14] = np.nan

    auto = PH.compute_hom(data=img, verbose=False)
    legacy = PH.compute_hom(data=img, verbose=False, pad_essential=False)

    assert auto.pad_essential == 'dilate'
    target = np.nanmax(img)
    auto_row = _essential_row(auto.generators, target)
    assert np.isfinite(auto_row[2])
    assert auto_row[2] > _ESSENTIAL_THRESHOLD

    # Non-essential rows untouched by the patch.
    auto_ess = (auto.generators[:, 0] == 0) & (auto.generators[:, 1] == target)
    legacy_ess = (legacy.generators[:, 0] == 0) & (legacy.generators[:, 1] == target)
    np.testing.assert_array_equal(auto.generators[~auto_ess],
                                  legacy.generators[~legacy_ess])

    # Patched death pixel is in-bounds of the original frame.
    dp = auto_row[6:6 + auto.n_dim]
    assert (dp >= 0).all() and (dp < np.array(img.shape)).all()


def test_explicit_false_disables(toy_2d_two_peaks):
    img, _ = toy_2d_two_peaks
    ph = PH.compute_hom(data=img, verbose=False, pad_essential=False)
    assert ph.pad_essential is False


# ---------------------------------------------------------------------------
# The patched essential row has a finite death; sentinel is gone.
# ---------------------------------------------------------------------------

def test_essential_death_is_finite_under_auto(toy_2d_two_peaks):
    img, _ = toy_2d_two_peaks
    ph = PH.compute_hom(data=img, verbose=False)
    target = np.nanmax(img)
    row = _essential_row(ph.generators, target)
    assert np.isfinite(row[2])
    assert row[2] > _ESSENTIAL_THRESHOLD


def test_legacy_essential_death_is_sentinel(toy_2d_two_peaks):
    img, _ = toy_2d_two_peaks
    ph = PH.compute_hom(data=img, verbose=False, pad_essential=False)
    target = np.nanmax(img)
    row = _essential_row(ph.generators, target)
    assert row[2] < _ESSENTIAL_THRESHOLD


# ---------------------------------------------------------------------------
# Non-essential rows are untouched by the patch.
# ---------------------------------------------------------------------------

def test_non_essential_rows_unchanged_2d(toy_2d_two_peaks):
    img, _ = toy_2d_two_peaks
    target = np.nanmax(img)
    auto = PH.compute_hom(data=img, verbose=False).generators
    legacy = PH.compute_hom(data=img, verbose=False, pad_essential=False).generators

    auto_mask = (auto[:, 0] == 0) & (auto[:, 1] == target)
    legacy_mask = (legacy[:, 0] == 0) & (legacy[:, 1] == target)

    np.testing.assert_array_equal(auto[~auto_mask], legacy[~legacy_mask])


def test_non_essential_rows_unchanged_3d_dilate(toy_3d_nan_halo):
    img, _ = toy_3d_nan_halo
    target = np.nanmax(img)
    auto = PH.compute_hom(data=img, verbose=False).generators
    legacy = PH.compute_hom(data=img, verbose=False, pad_essential=False).generators

    assert auto.shape == legacy.shape
    auto_mask = (auto[:, 0] == 0) & (auto[:, 1] == target)
    legacy_mask = (legacy[:, 0] == 0) & (legacy[:, 1] == target)
    np.testing.assert_array_equal(auto[~auto_mask], legacy[~legacy_mask])


def test_total_generator_counts_per_dim_unchanged(toy_2d_ring):
    img, _, _ = toy_2d_ring
    auto = PH.compute_hom(data=img, verbose=False).generators
    legacy = PH.compute_hom(data=img, verbose=False, pad_essential=False).generators
    for dim in (0, 1):
        assert (auto[:, 0] == dim).sum() == (legacy[:, 0] == dim).sum()


# ---------------------------------------------------------------------------
# pad_value invariance — the infilled death is data-driven, not pad-driven.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [5.0, 10.0, 100.0, 1000.0])
def test_pad_value_does_not_affect_infilled_death_bbox(toy_2d_two_peaks, scale):
    img, _ = toy_2d_two_peaks
    target = np.nanmax(img)
    baseline = _essential_row(
        PH.compute_hom(data=img, verbose=False).generators, target
    )[2]
    custom = _essential_row(
        PH.compute_hom(data=img, verbose=False,
                       pad_value=scale * float(np.nanmax(img))).generators,
        target,
    )[2]
    assert custom == baseline


@pytest.mark.parametrize("scale", [5.0, 10.0, 100.0, 1000.0])
def test_pad_value_does_not_affect_infilled_death_dilate(toy_3d_nan_halo, scale):
    img, _ = toy_3d_nan_halo
    target = np.nanmax(img)
    baseline = _essential_row(
        PH.compute_hom(data=img, verbose=False).generators, target
    )[2]
    custom = _essential_row(
        PH.compute_hom(data=img, verbose=False,
                       pad_value=scale * float(np.nanmax(img))).generators,
        target,
    )[2]
    assert custom == baseline


# ---------------------------------------------------------------------------
# Death pixel is in-bounds in the original frame for both modes.
# ---------------------------------------------------------------------------

def test_essential_death_pixel_in_bounds_bbox(toy_2d_two_peaks):
    img, _ = toy_2d_two_peaks
    ph = PH.compute_hom(data=img, verbose=False)
    row = _essential_row(ph.generators, np.nanmax(img))
    dp = row[6:6 + ph.n_dim]
    assert (dp >= 0).all() and (dp < np.array(img.shape)).all()


def test_essential_death_pixel_value_matches_death_bbox(toy_2d_two_peaks):
    """The patched death pixel must index the merge voxel: ``img`` at that
    pixel equals the recorded death value. This pins the death-pixel column
    slice (cripser emits death pixel at fixed cols 6:6+n_dim, not 3+n_dim) —
    an off-by-n_dim slice on 2D input passes the in-bounds check above (clip
    masks it) but lands on the wrong voxel and fails here."""
    img, _ = toy_2d_two_peaks
    ph = PH.compute_hom(data=img, verbose=False)
    row = _essential_row(ph.generators, np.nanmax(img))
    dp = tuple(row[6:6 + ph.n_dim].astype(int))
    assert np.isclose(img[dp], row[2], rtol=1e-6)


def test_essential_death_pixel_in_bounds_dilate(toy_3d_nan_halo):
    """Death pixel of the patched essential row, in dilate mode, lands at a
    valid voxel index of the (un-padded) data array. We don't assert which
    voxel — cripser reports the merge-event location in the existing
    component, not necessarily the dilation-shell pixel."""
    img, _ = toy_3d_nan_halo
    ph = PH.compute_hom(data=img, verbose=False)
    row = _essential_row(ph.generators, np.nanmax(img))
    dp = row[6:6 + ph.n_dim]
    assert (dp >= 0).all() and (dp < np.array(img.shape)).all()


# ---------------------------------------------------------------------------
# Mode-specific error paths
# ---------------------------------------------------------------------------

def test_dilate_errors_on_all_finite(toy_2d_two_peaks):
    img, _ = toy_2d_two_peaks
    with pytest.raises(ValueError, match="dilate"):
        PH.compute_hom(data=img, verbose=False, pad_essential='dilate')


def test_unknown_mode_errors():
    img = np.ones((4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="pad_essential must be one of"):
        PH.compute_hom(data=img, verbose=False, pad_essential='nonsense')


def test_flip_data_false_errors_on_explicit_mode(toy_2d_two_peaks):
    img, _ = toy_2d_two_peaks
    with pytest.raises(ValueError, match="flip_data=True"):
        PH.compute_hom(data=img, verbose=False, flip_data=False,
                       pad_essential='bbox')


def test_flip_data_false_silently_disables_auto(toy_2d_two_peaks):
    img, _ = toy_2d_two_peaks
    ph = PH.compute_hom(data=img, verbose=False, flip_data=False)
    assert ph.pad_essential is False


# ---------------------------------------------------------------------------
# Deprecation warning on the old buff_pix path
# ---------------------------------------------------------------------------

def test_buff_pix_emits_deprecation_warning(toy_2d_two_peaks):
    img, _ = toy_2d_two_peaks
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PH.compute_hom(
            data=img,
            verbose=False,
            pad_essential=False,
            prep_img_kwargs={'buff_pix': True},
        )
    msgs = [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]
    assert any("buff_pix" in m and "pad_essential" in m for m in msgs)
