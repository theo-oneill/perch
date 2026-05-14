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

**Where:** `perch/ph.py` — `PH._prep_img` already supports a `buff_pix`
option that injects a low value at a corner pixel; `PH.compute_hom` accepts
a `prep_img_kwargs` dict but defaults to `buff_pix=False`. The other
``fill_complete``/``fill_mask`` paths exist for NaN-handling but not for
padding the image's "outside".

**What the absence costs us:** with the default `flip_data=True` and no
padding, the essential H₀ class has `death = -DBL_MAX`. This is the source
of multiple downstream quirks already tracked in this file:

* `load_from` silently drops essential rows (above).
* `conv_fac * (-DBL_MAX)` overflows during reload (above).
* `Structure.compute_segment` for the essential class filters
  `img > -DBL_MAX`, which is "everywhere finite" — meaning the essential
  class's segmentation is the whole image. In a hierarchy run, the
  essential class becomes a trivial trunk that swallows everything.
* `Structures.level_map` collapses to "1 everywhere outside the inner
  structures, 2 inside them" — there's no useful depth information at
  the top of the tree.

**Option to consider:** make perch pad the input cube with a sentinel
low value (e.g. `np.nanmin(data) - eps` after the flip, or an explicit
user-supplied "outside" value) as the default before handing to cripser.
The essential class then "dies" at this padded boundary value, becoming a
finite-persistence generator whose death is the data's true global
minimum (or whatever the padding value is). Downstream effects:

* essential class has a real, persistent finite death — no more sentinel.
* `Structure.compute_segment` for the (former) essential class produces a
  meaningful inclusive component, not the whole image.
* `load_from` no longer needs special-case essential handling.
* `conv_fac` no longer overflows.
* hierarchy levels become more informative.

**Risks / questions:**

* Pads add ~1 layer of voxels around each face; on tiny test fixtures
  this is a big relative change, on real astronomical cubes negligible.
* Need to decide what the pad value should be — caller-controlled
  (explicit kwarg), `nanmin(data)`, or `-inf`-style sentinel that
  cripser handles natively.
* Need to decide whether padding is on-by-default (new behavior) or
  opt-in via a flag (back-compat-friendly).
* Birthpix/deathpix coordinates from cripser will be in the *padded*
  frame; perch needs to subtract the pad offset before returning them.

**Action:** prototype the padding path on the `2d_two_peaks` fixture,
verify the essential generator gets a real finite death, decide
default-vs-opt-in, and re-run the regression tests to see what
references shift.

