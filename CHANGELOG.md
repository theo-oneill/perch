# Changelog

All notable changes to perch are recorded here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/). Versions are derived from git
tags via setuptools-scm.

## [Unreleased]

### Changed (behavior — may alter existing output)

- **`PH.compute_hom` now gives the essential H₀ class a finite death by
  default.** A new `pad_essential=` keyword (default `'auto'`) runs a second,
  H₀-only cripser pass on a padded copy of the data and patches the
  originally-essential row's death (and death pixel) into the generators.
  Previously the essential class carried a `-DBL_MAX` sentinel death. All
  non-essential rows and the H₀/H₁/H₂ counts are unchanged. Pass
  `pad_essential=False` for the legacy sentinel behavior.
  - `'auto'` picks `'dilate'` when NaN voxels are present, else `'bbox'`.
  - `pad_value` defaults to `10 × nanmax(data)`; the infilled death is a
    property of the data, not of `pad_value`.
  - Only the superlevel convention (`flip_data=True`) is supported; explicit
    modes raise under `flip_data=False`, and `'auto'` silently disables there.

- **`PH.load_from` is now a faithful loader.** It returns exactly the rows in
  the file (optionally scaled by `conv_fac`), so an `export_generators` →
  `load_from` round trip is the identity and matches `compute_hom`'s in-memory
  generators. It no longer silently drops the essential `-DBL_MAX` row, rows
  with `death < nanmin(data)`, or NaN-birth rows — cleaning is left to the
  caller (e.g. `PH.filter`). This also removes the `conv_fac` overflow the old
  loader could trigger on the sentinel row.

### Deprecated

- `PH._prep_img(buff_pix=...)` now emits a `DeprecationWarning`; use
  `pad_essential=` on `PH.compute_hom` instead. It will be removed in a future
  release.
