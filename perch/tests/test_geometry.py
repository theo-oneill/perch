"""3D-geometry tests for ``Structure.surface_area``, ``volume``, ``sphericity``.

These exercise the ``skimage.measure.marching_cubes`` path. Voxelization
inflates the surface area relative to the smooth-sphere theoretical value,
so the tests assert tolerance bands rather than exact equality.
"""

from __future__ import annotations

import numpy as np
import pytest


def _segmented_h2_cavity(ph_3d_shell, toy_3d_shell):
    """Return the single H2 cavity structure with indices computed."""
    img, _, _ = toy_3d_shell
    h2 = ph_3d_shell.filter(dimension=2)
    h2.compute_segment_hierarchy(img_jnp=img, verbose=False, export=False)
    struc = h2.all_structures[0]
    struc.compute_segment(img)
    return struc


def test_volume_equals_npix(strucs_3d_two_peaks_h0, toy_3d_two_peaks):
    """``Structure.volume`` is the literal pixel count."""
    img, _ = toy_3d_two_peaks
    struc = strucs_3d_two_peaks_h0.all_structures[0]
    struc.compute_segment(img)
    assert struc.volume == struc.npix


def test_surface_area_positive_for_h2_cavity(ph_3d_shell, toy_3d_shell):
    """Marching-cubes surface area exists and is positive for the void."""
    struc = _segmented_h2_cavity(ph_3d_shell, toy_3d_shell)
    struc._calculate_surface_area(save_points=False)
    assert struc.surface_area > 0.0


def test_sphericity_of_spherical_cavity_is_near_one(ph_3d_shell, toy_3d_shell):
    """The shell's cavity is approximately spherical; voxelization should keep
    sphericity in the 0.80 — 1.0 band. (True sphere = 1.0; voxelization loss
    is typically 0.05 — 0.15.)"""
    struc = _segmented_h2_cavity(ph_3d_shell, toy_3d_shell)
    struc._calculate_sphericity(save_points=False)
    assert 0.80 < struc.sphericity <= 1.0


def test_sphericity_invariants_from_definition(ph_3d_shell, toy_3d_shell):
    """sphericity := π^(1/3) · (6V)^(2/3) / SA, by definition."""
    struc = _segmented_h2_cavity(ph_3d_shell, toy_3d_shell)
    struc._calculate_sphericity(save_points=False)
    expected = np.pi ** (1 / 3) * (6 * struc.volume) ** (2 / 3) / struc.surface_area
    assert struc.sphericity == pytest.approx(expected, rel=1e-12)
