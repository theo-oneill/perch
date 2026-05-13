"""Regenerate frozen generator references for the regression test suite.

Run manually whenever ``PH.compute_hom`` output changes intentionally — for
example after a cripser bump that's been reviewed, or after a deliberate
algorithm tweak. The resulting ``.npz`` diff in git review is the change
log for the test references.

Usage
-----
    python -m perch.tests.data._regenerate
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path
import numpy as np

from perch.ph import PH
from perch.tests._fixtures import BUILDERS


DATA_DIR = Path(__file__).resolve().parent


def main():
    for name, builder in BUILDERS.items():
        # Builders return either (img, peaks) or (img, center, radius);
        # only the first element matters here.
        img = builder()[0]
        ph = PH.compute_hom(data=img, verbose=False)
        out = DATA_DIR / f"{name}_generators.npz"
        np.savez(out, generators=ph.generators)
        n_per_dim = {int(d): int((ph.generators[:, 0] == d).sum())
                     for d in (0, 1, 2)}
        print(f"  {name:<14}  shape={ph.generators.shape}  per-dim={n_per_dim}")
        print(f"    → {out}")


if __name__ == "__main__":
    main()
