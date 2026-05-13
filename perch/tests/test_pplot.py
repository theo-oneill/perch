"""Smoke tests for ``perch.pplot.pers_diagram``.

Persistence-diagram plotting is exercised against the standard PH fixtures.
These are smoke tests — they verify the call completes and produces axes
without crashing, not pixel-level layout.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from perch import pplot


def test_pers_diagram_smoke(ph_2d_two_peaks):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    pplot.pers_diagram(ph_2d_two_peaks, ax=ax)
    assert ax.get_xlabel() == "Birth"
    assert ax.get_ylabel() == "Death"
    # At least one scatter call should have plotted points.
    assert any(c.get_offsets().shape[0] > 0 for c in ax.collections)
    plt.close(fig)


def test_pers_diagram_with_dimensions_filter(ph_2d_ring):
    """Restricting to a specific dimension passes through the filter() path."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    pplot.pers_diagram(ph_2d_ring, ax=ax, dimensions=[1])
    # Only the H1 generator's point should appear (1 collection, 1 point).
    n_plotted = sum(c.get_offsets().shape[0] for c in ax.collections)
    assert n_plotted == 1
    plt.close(fig)


def test_pers_diagram_creates_axes_when_none(ph_3d_shell):
    """Calling with ax=None should produce its own figure."""
    pplot.pers_diagram(ph_3d_shell)
    # Close all figures matplotlib created internally.
    plt.close("all")
