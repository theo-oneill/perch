"""End-to-end smoke tests for the persistent-homology pipeline.

These tests run ``PH.compute_hom`` on tiny synthetic images with known
topology and verify that the generators table, the wrapped ``Structures``
collection, and the filter API are all wired up correctly. They catch
regressions in the cripser binding, the flip-data preprocessing, and the
generators-table layout.
"""

import numpy as np


def _count_birthpix_matches(birthpix_arr, peak_coords, atol=1):
    """Count how many expected peaks have a birthpix within ``atol`` pixels."""
    matched = set()
    for bp in birthpix_arr:
        for i, exp in enumerate(peak_coords):
            if i in matched:
                continue
            if np.allclose(bp, exp, atol=atol):
                matched.add(i)
                break
    return len(matched)


def test_compute_hom_2d_generator_layout(toy_2d_two_peaks, ph_2d_two_peaks):
    img, _ = toy_2d_two_peaks
    ph = ph_2d_two_peaks

    assert ph.generators is not None
    assert ph.generators.ndim == 2
    assert ph.generators.shape[1] == 10  # 9 PH columns + appended h_id
    assert ph.img_shape == img.shape
    assert ph.n_dim == 2
    assert ph.max_Hi == 1  # n_dim - 1


def test_compute_hom_2d_finds_both_peaks(toy_2d_two_peaks, ph_2d_two_peaks):
    _, peaks = toy_2d_two_peaks
    ph = ph_2d_two_peaks

    h0 = ph.generators[ph.generators[:, 0] == 0]
    assert len(h0) >= 2, "expected at least one H0 generator per peak"

    # The two strongest H0 generators (largest birth, since flip_data=True
    # puts births at the peak intensities) should land on the two peaks.
    top2 = h0[np.argsort(-h0[:, 1])[:2]]
    birthpix_2d = top2[:, 3:5].astype(int)
    assert _count_birthpix_matches(birthpix_2d, peaks) == 2


def test_compute_hom_3d_finds_both_peaks(toy_3d_two_peaks, ph_3d_two_peaks):
    img, peaks = toy_3d_two_peaks
    ph = ph_3d_two_peaks

    assert ph.n_dim == 3
    assert ph.max_Hi == 2
    assert ph.img_shape == img.shape

    h0 = ph.generators[ph.generators[:, 0] == 0]
    assert len(h0) >= 2

    top2 = h0[np.argsort(-h0[:, 1])[:2]]
    birthpix_3d = top2[:, 3:6].astype(int)
    assert _count_birthpix_matches(birthpix_3d, peaks) == 2


def test_compute_hom_wraps_structures(ph_2d_two_peaks):
    ph = ph_2d_two_peaks
    assert ph.strucs is not None
    assert ph.strucs.n_struc == len(ph.generators)
    # Aggregate births from Structures should match the generators table.
    np.testing.assert_array_equal(
        np.sort(ph.strucs.birth),
        np.sort(ph.generators[:, 1]),
    )


def test_filter_by_dimension(ph_2d_two_peaks):
    ph = ph_2d_two_peaks
    h0 = ph.filter(dimension=0)
    assert h0 is not None
    assert h0.n_struc >= 1
    for s in h0.all_structures:
        assert s.htype == 0


def test_compute_hom_2d_ring_has_h1(toy_2d_ring, ph_2d_ring):
    """A 2D Gaussian ring births exactly one prominent H1 cycle whose
    deathpix sits at the ring's geometric center."""
    _, center, _ = toy_2d_ring
    ph = ph_2d_ring

    h1 = ph.generators[ph.generators[:, 0] == 1]
    assert len(h1) >= 1, "expected at least one H1 generator from the ring"

    persistence = np.abs(h1[:, 2] - h1[:, 1])
    top = h1[np.argmax(persistence)]
    assert persistence.max() > 0.5, "the ring's H1 cycle should be highly persistent"

    deathpix_2d = top[6:8].astype(int)
    assert np.allclose(deathpix_2d, center, atol=1), (
        f"H1 deathpix {tuple(deathpix_2d)} should be near ring center {center}"
    )


def test_compute_hom_3d_shell_has_h2(toy_3d_shell, ph_3d_shell):
    """A 3D Gaussian spherical shell encloses one prominent H2 cavity."""
    ph = ph_3d_shell

    h2 = ph.generators[ph.generators[:, 0] == 2]
    assert len(h2) >= 1, "expected at least one H2 generator from the shell"

    persistence = np.abs(h2[:, 2] - h2[:, 1])
    assert persistence.max() > 0.5, "the shell's H2 cavity should be highly persistent"


def test_filter_by_dimension_h1(ph_2d_ring):
    """`PH.filter(dimension=1)` returns only H1 structures."""
    h1 = ph_2d_ring.filter(dimension=1)
    assert h1.n_struc >= 1
    for s in h1.all_structures:
        assert s.htype == 1


def test_filter_by_dimension_h2(ph_3d_shell):
    """`PH.filter(dimension=2)` returns only H2 structures."""
    h2 = ph_3d_shell.filter(dimension=2)
    assert h2.n_struc >= 1
    for s in h2.all_structures:
        assert s.htype == 2


# ---------------------------------------------------------------------------
# PH.filter criteria (covers the branches in ph.py lines 301-323)
# ---------------------------------------------------------------------------

def test_filter_min_life_keeps_high_persistence(ph_2d_ring):
    """``min_life`` filters by ``|death - birth|``. The 15 finite-death noise
    H0 generators each have persistence < 0.1 and are dropped at threshold 0.5.
    Two survivors clear it: the H1 ring cycle (persistence ≈ 0.92) and the
    essential H0 (persistence ≈ 1.0 under the default ``pad_essential='auto'``;
    a much larger nominal value under legacy ``pad_essential=False``)."""
    high = ph_2d_ring.filter(min_life=0.5)
    assert high.n_struc == 2
    htypes = sorted(s.htype for s in high.all_structures)
    assert htypes == [0, 1]


def test_filter_max_life_keeps_low_persistence(ph_2d_ring):
    low = ph_2d_ring.filter(max_life=0.5)
    # everything except the essential H0 and the strong H1
    assert all(s.htype == 0 for s in low.all_structures)
    assert low.n_struc < ph_2d_ring.strucs.n_struc


def test_filter_min_birth_and_max_birth(ph_2d_ring):
    """Births in the ring fixture span roughly 0 (noise) to 0.92 (H1).
    A min_birth threshold should drop the lowest-birth noise rows."""
    above_half = ph_2d_ring.filter(min_birth=0.5)
    assert above_half.n_struc >= 1
    for s in above_half.all_structures:
        assert s.birth > 0.5

    below_half = ph_2d_ring.filter(max_birth=0.5)
    for s in below_half.all_structures:
        assert s.birth < 0.5


def test_filter_min_death_and_max_death(ph_2d_ring):
    """The H1 cycle dies at ~0; noise H0 generators die near 0 as well.
    Threshold above/below 0 should split cleanly."""
    above_neg1 = ph_2d_ring.filter(min_death=-1.0)
    assert all(s.death > -1.0 for s in above_neg1.all_structures)

    below_half = ph_2d_ring.filter(max_death=0.5)
    for s in below_half.all_structures:
        assert s.death < 0.5


def test_filter_min_life_norm_birth(ph_2d_ring):
    """``min_life_norm_birth`` is ``|life| / |birth|``. The H1 cycle has
    persistence ≈ birth, so the ratio ≈ 1.0. Noise rows have ratio ≪ 1."""
    s = ph_2d_ring.filter(min_life_norm_birth=0.5)
    assert s.n_struc >= 1
    for st in s.all_structures:
        ratio = abs(st.death - st.birth) / abs(st.birth)
        assert ratio > 0.5


def test_filter_min_life_norm_death(ph_2d_ring):
    """``min_life_norm_death`` uses ``|life| / |death|``. With finite-but-
    near-zero deaths most rows have huge ratios; a threshold of 1.0 keeps
    rows with persistence > |death|."""
    s = ph_2d_ring.filter(min_life_norm_death=1.0)
    for st in s.all_structures:
        ratio = abs(st.death - st.birth) / abs(st.death)
        assert ratio > 1.0


def test_filter_with_explicit_mask(ph_2d_two_peaks):
    """A boolean mask of length n_generators selects rows directly."""
    n = len(ph_2d_two_peaks.generators)
    mask = np.zeros(n, dtype=bool)
    mask[0] = True
    s = ph_2d_two_peaks.filter(mask=mask)
    assert s.n_struc == 1
    np.testing.assert_array_equal(s.all_structures[0].birthpix,
                                  ph_2d_two_peaks.generators[0, 3:6].astype(int))


# ---------------------------------------------------------------------------
# Diagnostic plot output (opt-in via `pytest --perch-plots [DIR]`).
# These tests are skipped by default; pass --perch-plots to write PNGs.
# ---------------------------------------------------------------------------

import pytest


def _require_plot_dir(plot_dir):
    if plot_dir is None:
        pytest.skip("pass --perch-plots to enable diagnostic plot output")


def test_plot_2d_two_peaks(toy_2d_two_peaks, ph_2d_two_peaks, plot_dir):
    _require_plot_dir(plot_dir)
    from perch.tests import _plotting
    img, peaks = toy_2d_two_peaks
    _plotting.plot_2d_peaks(plot_dir / "2d_two_peaks.png", img, peaks, ph_2d_two_peaks)


def test_plot_3d_two_peaks(toy_3d_two_peaks, ph_3d_two_peaks, plot_dir):
    _require_plot_dir(plot_dir)
    from perch.tests import _plotting
    img, peaks = toy_3d_two_peaks
    _plotting.plot_3d_peaks(plot_dir / "3d_two_peaks.png", img, peaks, ph_3d_two_peaks)


def test_plot_2d_ring(toy_2d_ring, ph_2d_ring, plot_dir):
    _require_plot_dir(plot_dir)
    from perch.tests import _plotting
    img, center, radius = toy_2d_ring
    _plotting.plot_2d_ring(plot_dir / "2d_ring.png", img, center, radius, ph_2d_ring)


def test_plot_3d_shell(toy_3d_shell, ph_3d_shell, plot_dir):
    _require_plot_dir(plot_dir)
    from perch.tests import _plotting
    img, center, radius = toy_3d_shell
    _plotting.plot_3d_shell(plot_dir / "3d_shell.png", img, center, radius, ph_3d_shell)
