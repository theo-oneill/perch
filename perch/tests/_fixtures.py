"""Shared fixture-data definitions.

The pytest fixtures in ``conftest.py`` and the golden-reference regenerator
in ``data/_regenerate.py`` both build their synthetic images from this
module, so the references and the fixtures cannot drift apart.

Underscore prefix keeps pytest from collecting this file.
"""

from __future__ import annotations

import numpy as np


def gaussian_field(shape, peaks_amps_sigmas, dtype=np.float32):
    """Sum of isotropic Gaussians on a grid."""
    grids = np.indices(shape)
    img = np.zeros(shape, dtype=dtype)
    for center, amp, sigma in peaks_amps_sigmas:
        r2 = sum((g - c) ** 2 for g, c in zip(grids, center))
        img += amp * np.exp(-r2 / (2 * sigma ** 2))
    return img


def shell_field(shape, center, radius, sigma=1.5, dtype=np.float32):
    """Gaussian shell of given radius around ``center``."""
    grids = np.indices(shape)
    r = np.sqrt(sum((g - c) ** 2 for g, c in zip(grids, center)))
    return np.exp(-(r - radius) ** 2 / (2 * sigma ** 2)).astype(dtype)


# ---- fixture parameters (single source of truth) -----------------------

_2D_TWO_PEAKS = dict(
    shape=(16, 16),
    peaks=[(4, 5), (11, 10)],
    amps=[1.0, 0.6],
    sigmas=[1.2, 1.2],
)

_3D_TWO_PEAKS = dict(
    shape=(12, 12, 12),
    peaks=[(3, 4, 5), (8, 7, 6)],
    amps=[1.0, 0.6],
    sigmas=[1.2, 1.2],
)

_2D_RING = dict(shape=(24, 24), center=(12, 12), radius=6.0, sigma=1.5)
_3D_SHELL = dict(shape=(24, 24, 24), center=(12, 12, 12), radius=6.0, sigma=1.5)


def make_2d_two_peaks():
    s = _2D_TWO_PEAKS
    spec = list(zip(s["peaks"], s["amps"], s["sigmas"]))
    return gaussian_field(s["shape"], spec), s["peaks"]


def make_3d_two_peaks():
    s = _3D_TWO_PEAKS
    spec = list(zip(s["peaks"], s["amps"], s["sigmas"]))
    return gaussian_field(s["shape"], spec), s["peaks"]


def make_2d_ring():
    s = _2D_RING
    return shell_field(s["shape"], s["center"], s["radius"], s["sigma"]), s["center"], s["radius"]


def make_3d_shell():
    s = _3D_SHELL
    return shell_field(s["shape"], s["center"], s["radius"], s["sigma"]), s["center"], s["radius"]


# Map fixture-name → builder. Used by ``data/_regenerate.py``.
BUILDERS = {
    "2d_two_peaks": make_2d_two_peaks,
    "3d_two_peaks": make_3d_two_peaks,
    "2d_ring":      make_2d_ring,
    "3d_shell":     make_3d_shell,
}


def make_wcs_2d(shape):
    """Tiny celestial 2D WCS for testing pixel→world transforms."""
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2 + 0.5, shape[0] / 2 + 0.5]
    w.wcs.crval = [10.0, -20.0]                 # arbitrary fiducial RA/Dec, degrees
    w.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0] # 1 arcsec / pixel
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def make_wcs_3d(shape):
    """Tiny 3D WCS (RA/Dec + frequency) for testing 3D pixel→world transforms."""
    from astropy.wcs import WCS
    w = WCS(naxis=3)
    w.wcs.crpix = [shape[2] / 2 + 0.5, shape[1] / 2 + 0.5, shape[0] / 2 + 0.5]
    w.wcs.crval = [10.0, -20.0, 1.0e9]
    w.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0, 1.0e6]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN", "FREQ"]
    return w
