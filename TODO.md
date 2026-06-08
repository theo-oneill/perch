# perch — open design questions

## Re-enable `oldestdeps` in CI when tox-uv resolver behaves

**Status:** removed from `.github/workflows/main.yml` after multiple
failed install attempts. The `tox.ini` `oldestdeps` factor itself is
still defined and works locally with `tox -e py310-test-oldestdeps`
on a host where the appropriate wheels are available.

**What kept breaking:**

Even after bumping ``numpy>=1.26`` and ``connected-components-3d>=3.10``
to wheels-available floors, the GHA runner kept landing in a state
where uv's isolated build env was building numpy from sdist (despite
manylinux wheels existing on PyPI) and the source build failed against
the runner's modern CPython (``_Py_HashDouble`` signature mismatch).
Looks like a `tox-uv` + `--resolution lowest-direct` + GHA-runner-glibc
interaction; not a real packaging bug in perch.

**Why it still matters:**

The whole point of an oldestdeps leg is to honestly test the declared
minimum-version dependencies. Without it in CI, a user installing
perch into an environment that pins to our declared bounds (conda-
forge solver, shared HPC modules, certain pipeline envs) could hit
silent breakage we never notice.

**Options to revisit:**

* **(a)** Wait for tox-uv to fix its wheel-preference behavior; re-add
  the leg as-is when uv prefers wheels over sdist in lowest-direct
  resolution.
* **(b)** Switch to plain `pip`-based tox envs (no `tox-uv`) for the
  oldestdeps leg only. uv elsewhere stays fast; oldestdeps uses pip
  which has more predictable wheel selection.
* **(c)** Force binary-only install: add `uv_install_args = --only-binary :all:`
  to the `oldestdeps` testenv. May not work if any dep doesn't ship
  wheels for our floor versions.
* **(d)** Use `mamba`/`conda` for the oldestdeps env. Heavyweight.

**Action:** revisit in a few months once uv/tox-uv tooling matures,
or pick one of (b)/(c) if oldestdeps coverage becomes a publication
blocker (e.g., JOSS reviewers ask about it).

---





A short, living list of decisions to revisit. Add items as you find them;
remove items once resolved (in code, in an issue, or by deciding "won't fix").

## `PH.load_from` silently strips the essential class

**Where:** `perch/ph.py` — the body of `PH.load_from`, around the
`base_struc = gens[:,2] < np.nanmin(self.data)` and
`base_struc = np.isnan(gens[:, 1])` filters.

**What happens today:** with the default `flip_data=True` convention,
`PH.compute_hom` represents the essential H₀ class with a sentinel
`death = -DBL_MAX`. When the generators table is round-tripped through
`export_generators` → `load_from`, the load step drops that row because
its death is below `np.nanmin(data)`. The same filter also drops rows
with a NaN birth. There is no warning, no flag, and no way to opt out.

**Why it might be wrong:** the essential class is meaningful topological
output, not noise. Round-tripping a `PH` object should not silently
discard it. A user who exports their generators and reloads them later
gets a `Structures` collection that is missing the highest-persistence
H₀ generator and may not realise it.

**Why the current code may have a reason:** the filter may have been
written to defend against malformed third-party generator tables — e.g.,
rows produced by a different pipeline that uses NaN/very-negative
sentinels for different reasons. Need to check the history and any
callers in the example notebooks.

**Options to consider:**
* **(a) Encode current behavior in tests, do nothing in code.** What we
  did for Step 2. Pros: keeps the change purely additive. Cons: leaves
  the foot-gun in place.
* **(b) Fix the code.** Preserve essential rows on load (`death` below
  the sentinel threshold is kept as-is). Strip only rows that are
  unambiguously malformed (e.g., birth NaN AND death NaN, or finite-but-
  below-min). Update `test_io.py` accordingly.
* **(c) Opt-in strip.** Keep the current behavior behind a `strip=True`
  default flag on `load_from`, so callers can pass `strip=False` to get
  the lossless round-trip. Probably the least disruptive.

**Related quirk:** `load_from(conv_fac=...)` multiplies the essential
class's `-DBL_MAX` death by `conv_fac` and triggers a
`RuntimeWarning: overflow encountered in multiply` before the strip
discards the overflowed row. Whichever option above is chosen should
make that warning go away too.

**Action:** decide between (a)/(b)/(c) and either open an issue, or do
the fix as a follow-up to Step 2. Update `test_io.py` to match the new
behavior if (b) or (c) is chosen.

---

## Default image-padding to give the essential class a finite lifetime

**Where:** `perch/ph.py` — new `pad_essential=` kwarg on
`PH.compute_hom`, plus a private `_pad_and_patch_essential` helper.
`PH._prep_img` stays as-is for now (see deprecation note below).

**What the absence costs us:** with the default `flip_data=True` and no
padding, the essential H₀ class has `death = -DBL_MAX`. Downstream
consequences already tracked elsewhere in this file:

* `load_from` silently drops essential rows.
* `conv_fac * (-DBL_MAX)` overflows during reload.
* `Structure.compute_segment` for the essential class filters
  `img > -DBL_MAX`, returning the whole image — in a hierarchy run, the
  essential class becomes a trivial trunk that swallows everything.
* `Structures.level_map` collapses to "1 outside inner structures, 2
  inside" — no useful depth information at the top of the tree.

**Strategy (chosen):** two-run approach from
`perch_dust_clouds/clean/run_perch/run_perch.py` and §X of the in-prep
paper. Run cripser once on the original cube. Run cripser again (H₀
only, `maxdim=0`) on a padded copy of the cube whose pad voxels are
filled with `10 × nanmax(data)`. Identify the unique generator row with
`birth == nanmax(data)` in both runs; copy that row's `death` (and
`death_pixel`) from the padded run into the original generators. All
other rows come from the original run untouched, so total H₀/H₁/H₂
counts are unchanged.

**API:**

```python
PH.compute_hom(..., pad_essential='auto', pad_value=None)
```

- `pad_essential='auto'` (new default): `'dilate'` if NaN voxels exist
  in `data`, else `'bbox'`.
- `pad_essential='dilate'`: binary-dilate the finite-valued region by
  one voxel, fill the new edge with `pad_value`. Array shape unchanged.
- `pad_essential='bbox'`: wrap the array in a 1-voxel shell of
  `pad_value`. Translate padded-run pixel coords back into the
  original frame before patching.
- `pad_essential=False`: legacy behavior (essential death =
  `-DBL_MAX`).
- `pad_value=None` (default): `10 * np.nanmax(data)`.

**Decisions made (2026-05-14):**

1. Default switches from legacy to `'auto'`. **This is a behavior
   change** — call it out explicitly in the commit message.
2. `pad_value` default is `10 × nanmax(data)`. The test suite must
   assert that varying this constant (e.g. 5×, 100×, 1000×) does not
   change the infilled death value for the essential row — the death
   is a property of the data structure, not the pad magnitude.
3. Ship `'bbox'` mode in v1 (will be used).
4. Keep `_prep_img(buff_pix=...)` for now; flag for future deletion
   once `pad_essential` has been in a release for a cycle.

**Edge cases for implementation:**

* Tied maxes — if >1 row has `birth == nanmax(data)`, match on
  `(birth, birth_pixel)` tuple rather than birth alone, or fail loudly
  with a clear error.
* `flip_data=False` — error out; the padding direction only makes
  sense for the default superlevel convention.
* All-NaN or all-finite arrays — `'auto'` picks `'bbox'`; `'dilate'`
  with no NaN voxels errors with a pointer to `'bbox'`.
* `Structures` rebuild — `compute_hom` constructs `Structures` from
  the generators; rebuild after patching.
* Padded run should skip `Structures` construction entirely; only the
  generators array is needed. Add a lightweight internal path for
  this rather than throwing away a full `Structures` object.

**Cost:** the padded run uses `maxdim=0` already, and we can crop to
the bbox of the dilated mask + 1 voxel before handing to cripser.
Expected overhead ~10–30% of the original run for typical 3D cubes,
not 2×. See follow-up below on whether to invest in upstream cripser
early-stopping to push this lower.

**Test plan:**

* Toy 3D fixture mirroring the dust-cube geometry (finite blob in a
  NaN halo).
* `pad_essential=False` → essential row has `death = -DBL_MAX`.
* `pad_essential='dilate'` → essential row has finite death; death
  pixel lies in the dilation shell.
* Non-essential rows are bit-identical between `False` and `'dilate'`
  (load-bearing invariant — H₀/H₁/H₂ counts must not shift).
* `'bbox'` mode on an all-finite fixture → equivalent essential
  death; other rows match within the original bbox.
* Sweep `pad_value` ∈ {5×, 10×, 100×, 1000×} × nanmax → infilled
  death is identical across all four.
* Round-trip via `export_generators` → `load_from` no longer triggers
  the `conv_fac` overflow warning (cross-links to load-from item
  above).

**Action:** implement on a branch; do not push without approval.

---

## Review checklist for the `pad_essential` change (before commit/PR)

**Where:** the unstaged change set on this branch (`perch/ph.py`,
`perch/tests/`, `TODO.md`, the regenerated regression goldens, and the
diagnostic plots under `test_plots/`).

**Steps, roughly in order:**

1. Read item #3 of this file first — it's the written spec the code is
   meant to satisfy. Review against the spec rather than reverse-engineer.
2. `git diff perch/ph.py` — focus on (a) `_resolve_pad_mode`,
   `_build_padded_data`, `_pad_and_patch_essential`; (b) the
   `pad_essential=` / `pad_value=` wiring in `compute_hom`; (c) the
   `DeprecationWarning` in `_prep_img`. The most subtle parts are the
   tied-max handling and the bbox death-pixel clamp — both came from
   observed bugs, so they're the spots most likely to have edge cases.
3. `git diff -- perch/tests/test_pad_essential.py` — read the test names
   and docstrings as a behavior spec. Easiest way to spot a missing case.
4. `git diff perch/tests/conftest.py perch/tests/test_regression.py
   perch/tests/test_io.py perch/tests/test_structure.py
   perch/tests/test_edge_cases.py` — the test-infra decisions about
   which suites exercise legacy vs new behavior.
5. Eyeball the diagnostic plots: `2d_two_peaks_segmentation_legacy.png`
   vs `..._padded.png`, the 3D counterparts, and
   `2d_ring_h0_segmentation.png`. Sanity-check that the change does what
   you'd expect on synthetic data before trusting it on the dust cube.
6. Optional: run `/ultrareview` for an independent multi-agent pass.
   User-triggered and billed — Claude can't kick it off.

**Action:** delete this section once the change has been reviewed (and
merged or shelved). It's a workflow note, not a design open question.

---

## Revisit `pad_essential` `flip_data=False` handling and tie-breaking

**Where:** `perch/ph.py` — `compute_hom` (the `pad_mode`/`flip_data`
block, ~line 294) and `_pad_and_patch_essential` (the essential-row
selection, ~line 172).

**Two decisions deferred during the 2026-06 review** (the two confirmed
bugs from that review — the 2D death-pixel slice and the hardcoded
`embedded` in the padded run — were fixed; these two were not):

1. **`flip_data=False` + `pad_essential='auto'` silently disables.**
   The spec (item "Default image-padding...", decision under "Edge
   cases") said `flip_data=False` should *error out*. The shipped code
   only raises for an *explicit* `'dilate'`/`'bbox'`; for the default
   `'auto'` it silently sets `pad_mode=False`. Friendlier (a default
   sublevel call doesn't blow up), but it's a real deviation from the
   written spec, and "silent" can hide that the essential class kept
   the legacy `-DBL_MAX` sentinel. **Decide:** keep silent-disable for
   `'auto'`, or warn, or error. Tested by
   `test_flip_data_false_silently_disables_auto` — update that test to
   match whatever is chosen.

2. **Tie-breaking when >1 voxel equals `nanmax(data)`.** Spec said:
   match on `(birth, birth_pixel)` or *fail loudly*. Shipped code
   instead picks `argmin(death)` among tied-birth H_0 rows to identify
   the essential row, then matches its padded counterpart by birth
   pixel. Reasonable (the sentinel/longest-lived row is the min-death
   one) but it will silently pick one rather than failing on a genuine
   tie, and ties are currently untested (all fixtures avoid them).
   **Decide:** keep argmin-death, or add an explicit tie guard +
   regression fixture with two equal maxima.

**Action:** make both calls before the `pad_essential` change ships in
a release; add/adjust tests accordingly.

---

## Trim the `compute_hom` option surface (flip / embedding / padding)

**Where:** `perch/ph.py` — the `compute_hom` signature, which now exposes
`flip_data`, `embedded`, `pad_essential`, `pad_value`, `noise`,
`prep_img_kwargs` (itself carrying the deprecated `buff_pix`), plus
`max_Hi`.

**Concern (raised during the 2026-06 review):** too many user-facing
knobs for overlapping concerns. `flip_data`, `embedded`, and
`pad_essential` all interact (e.g. `pad_essential` only makes sense
under `flip_data=True`; `embedded` now threads into the padded run),
which is hard to reason about and easy to misuse. The combinatorial
space is mostly untested and mostly not what users actually want.

**To evaluate:**

* Which combinations are real use cases vs. theoretical? `flip_data=True`
  (superlevel) is the dominant astronomical convention — is
  `flip_data=False` ever used in practice, or can it go (it already
  forces `pad_essential` off)?
* Is `embedded` something users should set, or an internal detail?
* Once `buff_pix` is removed (separate TODO), can `prep_img_kwargs`
  shrink or disappear?
* Consider collapsing to a smaller set of named "modes" or sensible
  locked defaults, rather than independent booleans the user must
  combine correctly.

**Why a follow-up, not now:** the current change is additive and
behaves correctly; trimming the API is a deliberate, possibly
breaking, simplification best done in its own pass once the
`pad_essential` default has settled.

**Action:** audit real-world call sites (the dust-cloud pipeline, the
in-prep paper notebooks, example notebooks) for which options are
actually exercised, then propose a slimmed signature.

---

## Deprecate `_prep_img(buff_pix=...)`

**Where:** `perch/ph.py` — `PH._prep_img` lines that handle the
`buff_pix=True` corner-pixel injection.

**Why:** the original attempt at giving the essential class a finite
death. Superseded by the `pad_essential=` path above. Keep it for one
release cycle for back-compat, then remove. When removing, also drop
any internal callers that route through it.

**Action:** add a `DeprecationWarning` when `buff_pix=True` is passed,
pointing callers at `pad_essential=`. Remove the path entirely in a
later release.

---

## Evaluate forking cripser for threshold-based early-stop

**Where:** upstream (`shizuo-kaji/CubicalRipser_3dim`) — the C++
filtration loop in the cripser engine. Not a perch-internal change.

**Why:** the padded H₀-only second run for `pad_essential` only needs
the filtration to advance until the originally-essential class's
merge event is resolved. Cripser today walks the full filtration even
when `maxdim=0`. A threshold-stop or "stop after the N-th merge"
knob would cut the padded-run cost substantially.

**What to evaluate:**

* Sketch the C++ change in `joint_pairs.cpp` / the dim-0 union-find
  loop. Estimate complexity; check whether the same hook is reusable
  for higher dims.
* Benchmark current padded-run cost on a representative 3D dust cube
  (e.g. eden2024 at full res) to confirm the savings would be worth
  the maintenance burden of a fork.
* Decide: contribute upstream as an optional flag, or carry a perch-
  internal fork (e.g. `pycripser` rebrand). Upstream-first if the
  maintainer is responsive.

**Why this is a follow-up, not blocking:** the two-run approach is
correct today; this is purely a performance optimization that can
land after `pad_essential` ships and we have real-world cost data.

**Action:** open the conversation upstream once `pad_essential` is in
a release. Until then, parked.

