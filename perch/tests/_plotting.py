"""Shared plotting helpers for the perch test suite.

These are imported by the diagnostic ``test_plot_*`` tests in
``test_ph.py`` when the user passes ``--perch-plots``. They are also
usable from notebooks or docs to produce the same figures.

The module is prefixed with an underscore so pytest does not collect it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib

# Force a non-interactive backend so tests never try to open a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _proxy(marker, mfc, mec, ms, mew, label):
    """Build a Line2D proxy artist for use in figure-level legends."""
    return Line2D([0], [0], linestyle="none", marker=marker,
                  mfc=mfc, mec=mec, ms=ms, mew=mew, label=label)


def _attach_marker_legend(fig, handles):
    """Place a horizontal marker legend below the subplots."""
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.03), ncol=len(handles),
               frameon=True, fontsize=9,
               handletextpad=0.4, columnspacing=1.4)


# cripser writes -DBL_MAX (~ -1.8e308) for the essential class. Anything
# more negative than this sentinel is treated as "infinite lifetime".
_ESSENTIAL_SENTINEL = -1e30

# Where to draw essential generators on the persistence diagram's death axis
# (purely cosmetic — the real lifetime is infinite).
_INF_PLOT_Y = -0.15

_DIM_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
_DIM_MARKERS = {0: "o", 1: "s", 2: "^"}


def is_essential(generators: np.ndarray) -> np.ndarray:
    """Boolean mask: which rows of a generator table are essential classes."""
    return generators[:, 2] < _ESSENTIAL_SENTINEL


def persistence_diagram(ax, gens, vmax, title, dims=(0, 1, 2)):
    """Plot a persistence diagram on ``ax``.

    Essential generators are drawn on a separate yellow band below
    ``death=0`` and labelled as having infinite lifetime.
    """
    for d in dims:
        mask = gens[:, 0] == d
        if not mask.any():
            continue
        b = gens[mask, 1]
        dth = gens[mask, 2]
        ess = dth < _ESSENTIAL_SENTINEL
        if (~ess).any():
            ax.scatter(b[~ess], dth[~ess],
                       c=_DIM_COLORS[d], marker=_DIM_MARKERS[d],
                       label=f"$H_{d}$", s=60,
                       edgecolors="black", linewidths=0.5)
        if ess.any():
            ax.scatter(b[ess], np.full(ess.sum(), _INF_PLOT_Y),
                       c=_DIM_COLORS[d], marker=_DIM_MARKERS[d], s=80,
                       edgecolors="black", linewidths=0.8,
                       label=f"$H_{d}$ (death=$\\infty$)")
    lim_lo = _INF_PLOT_Y - 0.05
    lim_hi = float(vmax) * 1.1
    ax.axhline(0, color="gray", lw=0.5)
    ax.plot([0, lim_hi], [0, lim_hi], "k--", lw=0.7, alpha=0.5)
    ax.axhspan(lim_lo, -0.02, color="lightyellow", alpha=0.4, zorder=0)
    ax.text(0.02, _INF_PLOT_Y,
            "death = $\\infty$ (plotted here for viz)",
            transform=ax.get_yaxis_transform(),
            fontsize=8, color="gray", va="center")
    ax.set_xlim(-0.05, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)


def _h0_overlay_2d(ax, img, gens):
    """Overlay recovered H0 birth/death pixels on a 2D image."""
    ax.imshow(img, origin="lower", cmap="magma")
    for g in gens[gens[:, 0] == 0]:
        bp = g[3:5].astype(int)
        if g[2] < _ESSENTIAL_SENTINEL:
            ax.plot(bp[1], bp[0], "*", mfc="yellow", mec="black", ms=22, mew=1.6)
        else:
            dp = g[6:8].astype(int)
            ax.plot(bp[1], bp[0], "o", mfc="none", mec="lime", ms=12, mew=2)
            ax.plot(dp[1], dp[0], "x", color="red", ms=10, mew=2)
            ax.plot([bp[1], dp[1]], [bp[0], dp[0]],
                    "-", color="white", lw=0.8, alpha=0.6)


def plot_2d_peaks(path, img, peaks, ph):
    """Diagnostic plot for the 2D two-peaks fixture."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    ax = axes[0]
    im = ax.imshow(img, origin="lower", cmap="magma")
    for (r, c) in peaks:
        ax.plot(c, r, "+", color="cyan", ms=14, mew=2)
    ax.set_title("2D fixture: two Gaussian peaks")
    plt.colorbar(im, ax=ax, fraction=0.046)

    _h0_overlay_2d(axes[1], img, ph.generators)
    axes[1].set_title("Recovered $H_0$")

    persistence_diagram(axes[2], ph.generators, vmax=float(img.max()),
                        title="Persistence diagram", dims=(0, 1))

    legend_items = [
        _proxy("+", "cyan", "cyan", 10, 2, "expected peak"),
        _proxy("o", "none", "lime", 9, 2, "$H_0$ birth (peak)"),
        _proxy("x", "red", "red", 9, 2, "$H_0$ death (saddle)"),
        _proxy("*", "yellow", "black", 14, 1.4,
               "$H_0$ essential ($\\infty$-lifetime)"),
    ]
    fig.tight_layout()
    _attach_marker_legend(fig, legend_items)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _setup_3d_axes(ax, shape, caption=None):
    """Set 3D axis limits/labels/aspect/view and an optional small caption."""
    ax.set_xlim(0, shape[2] - 1)
    ax.set_ylim(0, shape[1] - 1)
    ax.set_zlim(0, shape[0] - 1)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    try:
        ax.set_box_aspect((shape[2], shape[1], shape[0]))
    except Exception:
        pass
    ax.view_init(elev=22, azim=-55)
    if caption:
        ax.text2D(0.5, -0.08, caption,
                  transform=ax.transAxes, ha="center", va="top",
                  fontsize=8, style="italic", color="gray")


def _render_3d_volume(ax, img, threshold, alpha=0.35, cmap="magma", size=16):
    """Render voxels above ``threshold`` as a coloured 3D scatter."""
    mask = img > threshold
    zz, yy, xx = np.where(mask)
    vals = img[mask]
    ax.scatter(xx, yy, zz, c=vals, cmap=cmap, alpha=alpha, s=size,
               edgecolors="none", vmin=0.0, vmax=float(img.max()),
               depthshade=False)
    _setup_3d_axes(ax, img.shape, caption="scatter view of data cube")


def plot_3d_peaks(path, img, peaks, ph):
    """Diagnostic plot for the 3D two-peaks fixture."""
    fig = plt.figure(figsize=(15, 5))
    threshold = float(img.max()) * 0.15

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    _render_3d_volume(ax, img, threshold=threshold, alpha=0.35)
    for (z, y, x) in peaks:
        ax.scatter(x, y, z, marker="+", color="cyan", s=200,
                   linewidths=3, depthshade=False)
    ax.set_title("3D fixture: two Gaussian peaks")

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    _render_3d_volume(ax, img, threshold=threshold, alpha=0.25)
    for g in ph.generators[ph.generators[:, 0] == 0]:
        bp = g[3:6].astype(int)  # (z, y, x)
        if g[2] < _ESSENTIAL_SENTINEL:
            ax.scatter(bp[2], bp[1], bp[0], marker="*", s=350,
                       c="yellow", edgecolors="black", linewidths=1.5,
                       depthshade=False)
        else:
            ax.scatter(bp[2], bp[1], bp[0], marker="o", s=140,
                       facecolors="none", edgecolors="lime",
                       linewidths=2.5, depthshade=False)
    ax.set_title("Recovered $H_0$ birthpix")

    ax = fig.add_subplot(1, 3, 3)
    persistence_diagram(ax, ph.generators, vmax=float(img.max()),
                        title="Persistence diagram", dims=(0, 1, 2))

    legend_items = [
        _proxy("+", "cyan", "cyan", 12, 2.5, "expected peak"),
        _proxy("o", "none", "lime", 10, 2.5, "$H_0$ birth (peak)"),
        _proxy("*", "yellow", "black", 15, 1.4,
               "$H_0$ essential ($\\infty$-lifetime)"),
    ]
    fig.tight_layout()
    _attach_marker_legend(fig, legend_items)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_2d_ring(path, img, center, radius, ph):
    """Diagnostic plot for the 2D ring fixture (H1 test)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))

    ax = axes[0]
    im = ax.imshow(img, origin="lower", cmap="magma")
    ax.plot(center[1], center[0], "+", color="cyan", ms=14, mew=2)
    ax.set_title(f"2D fixture: Gaussian ring (r={radius})")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    ax.imshow(img, origin="lower", cmap="magma")
    h1 = ph.generators[ph.generators[:, 0] == 1]
    persistence = np.abs(h1[:, 2] - h1[:, 1])
    order = np.argsort(-persistence)
    for i, g in enumerate(h1[order]):
        bp = g[3:5].astype(int)
        dp = g[6:8].astype(int)
        color = "magenta" if i == 0 else "gray"
        alpha = 1.0 if i == 0 else 0.4
        ax.plot(bp[1], bp[0], "s", mfc="none", mec=color, ms=12, mew=2, alpha=alpha)
        ax.plot(dp[1], dp[0], "x", color=color, ms=12, mew=2, alpha=alpha)
    ax.set_title(f"Recovered $H_1$ ({len(h1)} total)")

    persistence_diagram(axes[2], ph.generators, vmax=float(img.max()),
                        title="Persistence diagram", dims=(0, 1))

    legend_items = [
        _proxy("+", "cyan", "cyan", 10, 2, "expected $H_1$ deathpix"),
        _proxy("s", "none", "magenta", 9, 2, "most-persistent $H_1$ birth (saddle)"),
        _proxy("x", "magenta", "magenta", 9, 2, "most-persistent $H_1$ death (interior min)"),
        _proxy("s", "none", "gray", 9, 2, "other $H_1$ generators"),
    ]
    fig.tight_layout()
    _attach_marker_legend(fig, legend_items)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_3d_shell(path, img, center, radius, ph):
    """Diagnostic plot for the 3D shell fixture (H2 test)."""
    fig = plt.figure(figsize=(15, 5))
    threshold = float(img.max()) * 0.3

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    _render_3d_volume(ax, img, threshold=threshold, alpha=0.18)
    ax.scatter(center[2], center[1], center[0], marker="+", color="cyan",
               s=240, linewidths=3, depthshade=False)
    ax.set_title(f"3D fixture: Gaussian shell (r={radius})")

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    _render_3d_volume(ax, img, threshold=threshold, alpha=0.12)
    # ground-truth center
    ax.scatter(center[2], center[1], center[0], marker="+", color="cyan",
               s=240, linewidths=3, depthshade=False)
    # recovered H2 deathpix(es)
    for g in ph.generators[ph.generators[:, 0] == 2]:
        dp = g[6:9].astype(int)
        ax.scatter(dp[2], dp[1], dp[0], marker="x", color="magenta",
                   s=180, linewidths=3, depthshade=False)
    ax.set_title("Recovered $H_2$ deathpix in cavity")

    ax = fig.add_subplot(1, 3, 3)
    persistence_diagram(ax, ph.generators, vmax=float(img.max()),
                        title="Persistence diagram", dims=(0, 1, 2))

    legend_items = [
        _proxy("+", "cyan", "cyan", 13, 2.8, "expected $H_2$ cavity center"),
        _proxy("x", "magenta", "magenta", 11, 2.8,
               "recovered $H_2$ deathpix"),
    ]
    fig.tight_layout()
    _attach_marker_legend(fig, legend_items)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_segmentation_2d(path, img, strucs):
    """Diagnostic plot of a segmented Structures collection.

    Three panels: the input image, the categorical ``struc_map`` (each
    structure id is its own color, NaN pixels gray), and the integer
    ``level_map`` (nesting depth). Structure ids are annotated at each
    structure's geometric centroid.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    # ---- Panel 1: original image ----
    ax = axes[0]
    im = ax.imshow(img, origin="lower", cmap="magma")
    ax.set_title("Original image")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # ---- Panel 2: struc_map (categorical) ----
    ax = axes[1]
    smap = np.asarray(strucs.struc_map, dtype=float)
    n = strucs.n_struc
    # tab10 gives 10 distinct colors; fall back to tab20 for larger collections.
    cmap_name = "tab10" if n <= 10 else "tab20"
    base_cmap = plt.get_cmap(cmap_name, max(n, 2))
    base_cmap.set_bad("0.85")  # gray for NaN pixels (unsegmented)
    masked = np.ma.array(smap, mask=~np.isfinite(smap))
    im = ax.imshow(masked, origin="lower", cmap=base_cmap,
                   vmin=-0.5, vmax=n - 0.5)
    ax.set_title(f"struc_map ({n} structures)")
    cb = plt.colorbar(im, ax=ax, fraction=0.046,
                      ticks=list(range(n)))
    cb.set_label("structure ID")
    # Annotate each structure id at the centroid of its pixels in the map.
    for sid in range(n):
        ys, xs = np.where(smap == sid)
        if len(ys) == 0:
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        ax.text(cx, cy, str(sid), color="white", ha="center", va="center",
                fontsize=10, fontweight="bold",
                path_effects=[])

    # ---- Panel 3: level_map ----
    ax = axes[2]
    lmap = np.asarray(strucs.level_map, dtype=float)
    finite_levels = lmap[np.isfinite(lmap)]
    if finite_levels.size:
        max_level = int(np.nanmax(finite_levels))
        min_level = int(np.nanmin(finite_levels))
    else:
        max_level, min_level = 1, 1
    level_cmap = plt.get_cmap("viridis",
                              max(max_level - min_level + 1, 1))
    level_cmap.set_bad("0.85")
    lmasked = np.ma.array(lmap, mask=~np.isfinite(lmap))
    im = ax.imshow(lmasked, origin="lower", cmap=level_cmap,
                   vmin=min_level - 0.5, vmax=max_level + 0.5)
    ax.set_title("level_map (hierarchy depth)")
    cb = plt.colorbar(im, ax=ax, fraction=0.046,
                      ticks=list(range(min_level, max_level + 1)))
    cb.set_label("nesting level")

    legend_items = [
        _proxy("s", "0.85", "0.85", 10, 1, "unsegmented (NaN)"),
    ]
    fig.tight_layout()
    _attach_marker_legend(fig, legend_items)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_segmentation_3d(path, img, strucs):
    """Diagnostic plot of a segmented 3D Structures collection.

    Three panels: a scatter of the original cube, the categorical
    ``struc_map`` (each id its own color), and the integer ``level_map``.
    The segmentation panels render only voxels that were actually assigned
    (NaN voxels are omitted from the scatter).
    """
    fig = plt.figure(figsize=(15, 5))
    threshold = float(img.max()) * 0.1

    # ---- Panel 1: original cube ----
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    _render_3d_volume(ax, img, threshold=threshold, alpha=0.35)
    ax.set_title("Original data cube")

    # ---- Panel 2: struc_map (categorical) ----
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    smap = np.asarray(strucs.struc_map, dtype=float)
    n = strucs.n_struc
    cmap_name = "tab10" if n <= 10 else "tab20"
    sc_cmap = plt.get_cmap(cmap_name, max(n, 2))
    finite_mask = np.isfinite(smap)
    if finite_mask.any():
        zz, yy, xx = np.where(finite_mask)
        ids = smap[finite_mask].astype(int)
        sc = ax.scatter(xx, yy, zz, c=ids, cmap=sc_cmap, alpha=0.22, s=18,
                        edgecolors="none", vmin=-0.5, vmax=n - 0.5,
                        depthshade=False)
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, shrink=0.6,
                          ticks=list(range(n)))
        cb.set_label("structure ID")
    _setup_3d_axes(ax, img.shape, caption="scatter view of struc_map")
    ax.set_title(f"struc_map ({n} structures)")

    # ---- Panel 3: level_map ----
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    lmap = np.asarray(strucs.level_map, dtype=float)
    finite_lev = np.isfinite(lmap)
    if finite_lev.any():
        zz, yy, xx = np.where(finite_lev)
        levs = lmap[finite_lev]
        max_level = int(levs.max())
        min_level = int(levs.min())
        lev_cmap = plt.get_cmap("viridis",
                                max(max_level - min_level + 1, 1))
        sc = ax.scatter(xx, yy, zz, c=levs, cmap=lev_cmap, alpha=0.22, s=18,
                        edgecolors="none",
                        vmin=min_level - 0.5, vmax=max_level + 0.5,
                        depthshade=False)
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, shrink=0.6,
                          ticks=list(range(min_level, max_level + 1)))
        cb.set_label("nesting level")
    _setup_3d_axes(ax, img.shape, caption="scatter view of level_map")
    ax.set_title("level_map (hierarchy depth)")

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def ensure_dir(p) -> Path:
    """Create a directory if needed and return it as a Path."""
    p = Path(p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p
