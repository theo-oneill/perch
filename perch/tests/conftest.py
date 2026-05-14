"""Shared fixtures for the perch test suite."""

import os
from pathlib import Path

# Pin JAX to CPU before any test or module-under-test imports jax.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Pin matplotlib to the non-interactive Agg backend at the env-var level so
# any later `matplotlib.use("Agg")` is a no-op (and doesn't emit the
# deprecation warning that the strict filterwarnings would promote to error).
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest


# Repo-relative default so plots land in the same place regardless of cwd.
# perch/tests/conftest.py → perch/ → <repo>
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOT_DIR = _REPO_ROOT / "test_plots"


def pytest_addoption(parser):
    parser.addoption(
        "--perch-plots",
        action="store",
        nargs="?",
        const=str(DEFAULT_PLOT_DIR),
        default=None,
        metavar="DIR",
        help=("Write diagnostic plots from the test suite. "
              "Pass a directory, or use --perch-plots alone to write to "
              "<repo>/test_plots/ (gitignored)."),
    )


@pytest.fixture(scope="session")
def plot_dir(request):
    """Return a Path to write diagnostic plots into, or None if disabled."""
    val = request.config.getoption("--perch-plots")
    if val is None:
        return None
    p = Path(val).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


from perch.tests import _fixtures


@pytest.fixture(scope="session")
def toy_2d_two_peaks():
    """16x16 image with two well-separated Gaussian peaks at known pixels."""
    return _fixtures.make_2d_two_peaks()


@pytest.fixture(scope="session")
def toy_3d_two_peaks():
    """12x12x12 cube with two well-separated Gaussian peaks at known voxels."""
    return _fixtures.make_3d_two_peaks()


@pytest.fixture(scope="session")
def toy_2d_ring():
    """24x24 image with a Gaussian-profile ring (annulus) of radius 6.

    Carries one prominent H1 cycle whose deathpix should fall at the ring's
    geometric center.
    """
    return _fixtures.make_2d_ring()


@pytest.fixture(scope="session")
def toy_3d_shell():
    """24^3 cube with a Gaussian-profile spherical shell of radius 6.

    Carries one prominent H2 cycle (the enclosed void).
    """
    return _fixtures.make_3d_shell()


@pytest.fixture(scope="session")
def ph_2d_two_peaks(toy_2d_two_peaks):
    """PH of the 2D two-peaks image, computed once per session."""
    from perch.ph import PH
    img, _ = toy_2d_two_peaks
    return PH.compute_hom(data=img, verbose=False)


@pytest.fixture(scope="session")
def ph_3d_two_peaks(toy_3d_two_peaks):
    """PH of the 3D two-peaks cube, computed once per session."""
    from perch.ph import PH
    img, _ = toy_3d_two_peaks
    return PH.compute_hom(data=img, verbose=False)


@pytest.fixture(scope="session")
def ph_2d_ring(toy_2d_ring):
    """PH of the 2D ring image, computed once per session."""
    from perch.ph import PH
    img, _, _ = toy_2d_ring
    return PH.compute_hom(data=img, verbose=False)


@pytest.fixture(scope="session")
def ph_3d_shell(toy_3d_shell):
    """PH of the 3D shell cube, computed once per session."""
    from perch.ph import PH
    img, _, _ = toy_3d_shell
    return PH.compute_hom(data=img, verbose=False)


@pytest.fixture(scope="session")
def strucs_2d_two_peaks_h0(ph_2d_two_peaks, toy_2d_two_peaks):
    """H0 structures from the 2D two-peaks fixture with hierarchy segmented.

    Mutates a filtered Structures collection in place — paid once per session.
    The two-peaks layout produces a clean hierarchy: the essential class is the
    trunk (whole-image segmentation), the finite-death H0 generator is its
    only leaf child.
    """
    img, _ = toy_2d_two_peaks
    h0 = ph_2d_two_peaks.filter(dimension=0)
    h0.compute_segment_hierarchy(img_jnp=img, verbose=False, export=False)
    return h0


@pytest.fixture(scope="session")
def strucs_3d_two_peaks_h0(ph_3d_two_peaks, toy_3d_two_peaks):
    """H0 structures from the 3D two-peaks fixture with hierarchy segmented."""
    img, _ = toy_3d_two_peaks
    h0 = ph_3d_two_peaks.filter(dimension=0)
    h0.compute_segment_hierarchy(img_jnp=img, verbose=False, export=False)
    return h0


@pytest.fixture(scope="session")
def strucs_2d_ring_h1(ph_2d_ring, toy_2d_ring):
    """H1 structures from the 2D ring fixture with the (single-cycle) hierarchy
    segmented. Restricted to dimension 1 so the segmentation doesn't mix
    H0 noise with the true H1 cycle."""
    img, _, _ = toy_2d_ring
    h1 = ph_2d_ring.filter(dimension=1)
    h1.compute_segment_hierarchy(img_jnp=img, verbose=False, export=False)
    return h1


@pytest.fixture(scope="session")
def ph_2d_two_peaks_wcs(toy_2d_two_peaks):
    """PH of the 2D two-peaks image with a celestial WCS attached."""
    from perch.ph import PH
    img, _ = toy_2d_two_peaks
    wcs = _fixtures.make_wcs_2d(img.shape)
    return PH.compute_hom(data=img, wcs=wcs, verbose=False)


@pytest.fixture(scope="session")
def ph_3d_two_peaks_wcs(toy_3d_two_peaks):
    """PH of the 3D two-peaks cube with a celestial+freq WCS attached."""
    from perch.ph import PH
    img, _ = toy_3d_two_peaks
    wcs = _fixtures.make_wcs_3d(img.shape)
    return PH.compute_hom(data=img, wcs=wcs, verbose=False)


@pytest.fixture(scope="session")
def strucs_3d_shell_h2(ph_3d_shell, toy_3d_shell):
    """H2 structures from the 3D shell fixture with the (single-cavity)
    hierarchy segmented. Restricted to dimension 2 so the segmentation
    targets the enclosed void rather than discretization-noise H0/H1."""
    img, _, _ = toy_3d_shell
    h2 = ph_3d_shell.filter(dimension=2)
    h2.compute_segment_hierarchy(img_jnp=img, verbose=False, export=False)
    return h2


@pytest.fixture(scope="session")
def strucs_2d_two_peaks_h0_wcs(ph_2d_two_peaks_wcs, toy_2d_two_peaks):
    """Segmented 2D H0 collection that carries a celestial WCS."""
    img, _ = toy_2d_two_peaks
    h0 = ph_2d_two_peaks_wcs.filter(dimension=0)
    h0.compute_segment_hierarchy(img_jnp=img, verbose=False, export=False)
    return h0


@pytest.fixture(scope="session")
def strucs_3d_two_peaks_h0_wcs(ph_3d_two_peaks_wcs, toy_3d_two_peaks):
    """Segmented 3D H0 collection that carries a 3D WCS."""
    img, _ = toy_3d_two_peaks
    h0 = ph_3d_two_peaks_wcs.filter(dimension=0)
    h0.compute_segment_hierarchy(img_jnp=img, verbose=False, export=False)
    return h0
