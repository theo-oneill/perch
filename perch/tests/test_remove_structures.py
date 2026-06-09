"""Tests for ``Structures.remove_struc`` / ``remove_strucs``.

The shared PH fixtures only produce a two-level hierarchy (trunk + one leaf),
which is too shallow to exercise removal of an interior branch. These tests
instead wire up a synthetic four-level hierarchy by hand so the bookkeeping
(re-parenting children, purging the removed ID from *every* ancestor's
descendant list, decrementing the level of the whole subtree, and rewriting
``struc_map``/``level_map``) can be checked exactly.

Synthetic hierarchy (self-inclusive ``descendants`` in brackets)::

    0  trunk   level 1   [0,1,2,3,4]
    └── 1  branch  level 2   [1,2,3,4]
        ├── 2  leaf    level 3   [2]
        └── 3  branch  level 3   [3,4]
            └── 4  leaf    level 4   [4]

Pixel layout of the 1-D ``struc_map`` (deepest structure per pixel)::

    index     0  1  2  3  4  5  6  7   8    9
    struc_map 0  0  1  1  2  3  3  4  nan  nan
    level_map 1  1  2  2  3  3  3  4  nan  nan
"""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest

from perch.structure import Structure
from perch.structures import Structures


def _make_collection():
    """Build a fresh four-level synthetic ``Structures`` for each test."""
    img_shape = (10,)

    def make(i, level):
        # pi = [htype, birth, death, b0, b1, b2, d0, d1, d2]; only htype/birth/
        # death carry meaning here and none are exercised by removal.
        pi = [0, float(10 - i), float(i), 0, 0, 0, 0, 0, 0]
        s = Structure(pi=pi, id=i, id_ph=i, img_shape=img_shape)
        s._level = level
        return s

    strucs = {i: make(i, lvl) for i, lvl in {0: 1, 1: 2, 2: 3, 3: 3, 4: 4}.items()}

    # Wire the hierarchy.
    strucs[0]._parent = None
    strucs[0]._children = [1]
    strucs[0]._descendants = [0, 1, 2, 3, 4]

    strucs[1]._parent = 0
    strucs[1]._children = [2, 3]
    strucs[1]._descendants = [1, 2, 3, 4]

    strucs[2]._parent = 1
    strucs[2]._children = []
    strucs[2]._descendants = [2]

    strucs[3]._parent = 1
    strucs[3]._children = [4]
    strucs[3]._descendants = [3, 4]

    strucs[4]._parent = 3
    strucs[4]._children = []
    strucs[4]._descendants = [4]

    struc_map = np.array([0, 0, 1, 1, 2, 3, 3, 4, np.nan, np.nan], dtype=float)
    level_map = np.array([1, 1, 2, 2, 3, 3, 3, 4, np.nan, np.nan], dtype=float)

    s = Structures(structures=strucs, img_shape=img_shape,
                   struc_map=struc_map, level_map=level_map)
    s._refresh_hierarchy_cache()
    return s


def _assert_hierarchy_consistent(s):
    """Invariants that must hold after any sequence of removals."""
    keys = set(s.structure_keys)
    for k in keys:
        struc = s.structures[k]

        # No dangling references to removed structures.
        assert struc.parent is None or struc.parent in keys
        for c in struc.children:
            assert c in keys
        for d in struc.descendants:
            assert d in keys

        # parent/children agree both ways.
        if struc.parent is not None:
            assert k in s.structures[struc.parent].children
        for c in struc.children:
            assert s.structures[c].parent == k

        # descendants are self-inclusive and transitively closed.
        assert k in struc.descendants
        expected = _collect_descendants(s, k)
        assert set(struc.descendants) == expected

    # struc_map / level_map reference only live ids, and level_map matches the
    # level of the deepest structure at each pixel.
    finite = np.isfinite(s.struc_map)
    for sid in np.unique(s.struc_map[finite]).astype(int):
        assert sid in keys
        pix = s.struc_map == sid
        assert np.all(s.level_map[pix] == s.structures[sid].level)


def _collect_descendants(s, k):
    """Recompute the self-inclusive descendant set from children links."""
    out = set()
    stack = [k]
    while stack:
        cur = stack.pop()
        out.add(cur)
        stack.extend(s.structures[cur].children)
    return out


def test_remove_interior_branch_reparents_and_purges_ancestors():
    """Removing branch 1 dissolves it into trunk 0: children 2,3 re-parent to
    0, every descendant drops a level, and id 1 leaves 0's descendant list."""
    s = _make_collection()
    s.remove_struc(1)

    assert set(s.structure_keys) == {0, 2, 3, 4}

    # Children of the removed branch are re-parented onto its parent (0), in
    # place of the removed id.
    assert s.structures[2].parent == 0
    assert s.structures[3].parent == 0
    assert sorted(s.structures[0].children) == [2, 3]

    # The removed id is gone from the ancestor's (self-inclusive) descendants.
    assert 1 not in s.structures[0].descendants
    assert sorted(s.structures[0].descendants) == [0, 2, 3, 4]

    # Whole subtree drops exactly one level of nesting.
    assert s.structures[0].level == 1
    assert s.structures[2].level == 2
    assert s.structures[3].level == 2
    assert s.structures[4].level == 3

    # The removed branch's exclusive pixels merge into its parent (id 0, lvl 1);
    # descendant pixels each drop one level.
    np.testing.assert_array_equal(
        s.struc_map,
        np.array([0, 0, 0, 0, 2, 3, 3, 4, np.nan, np.nan], dtype=float),
    )
    np.testing.assert_array_equal(
        s.level_map,
        np.array([1, 1, 1, 1, 2, 2, 2, 3, np.nan, np.nan], dtype=float),
    )

    assert [t.id for t in s.trunk] == [0]
    assert sorted(leaf.id for leaf in s.leaves) == [2, 4]
    _assert_hierarchy_consistent(s)


def test_remove_leaf_purges_all_ancestors():
    """Removing deep leaf 4 must drop it from BOTH its parent (3) and its
    grandparent (1) and great-grandparent (0) descendant lists."""
    s = _make_collection()
    s.remove_struc(4)

    assert set(s.structure_keys) == {0, 1, 2, 3}
    assert s.structures[3].children == []
    assert s.structures[3].is_leaf
    for anc in (0, 1, 3):
        assert 4 not in s.structures[anc].descendants

    # Leaf pixel merges into its parent (id 3, level 3).
    assert s.struc_map[7] == 3
    assert s.level_map[7] == 3
    assert sorted(leaf.id for leaf in s.leaves) == [2, 3]
    _assert_hierarchy_consistent(s)


def test_remove_trunk_orphans_children_to_trunks():
    """Removing the root with no parent turns its children into new trunks and
    blanks its exclusive pixels to NaN."""
    s = _make_collection()
    s.remove_struc(0)

    assert set(s.structure_keys) == {1, 2, 3, 4}
    assert s.structures[1].parent is None
    assert s.structures[1].level == 1
    assert s.structures[2].level == 2
    assert s.structures[4].level == 3

    # Trunk's two exclusive pixels become unassigned.
    assert np.isnan(s.struc_map[0]) and np.isnan(s.struc_map[1])
    assert np.isnan(s.level_map[0]) and np.isnan(s.level_map[1])
    assert [t.id for t in s.trunk] == [1]
    _assert_hierarchy_consistent(s)


def test_remove_strucs_batch_matches_sequential():
    """Batch removal leaves a consistent hierarchy regardless of order."""
    s = _make_collection()
    s.remove_strucs([1, 4])

    assert set(s.structure_keys) == {0, 2, 3}
    # After removing 1, children 2,3 hang off 0; removing 4 then leaves 3 a leaf.
    assert sorted(s.structures[0].children) == [2, 3]
    assert s.structures[3].is_leaf
    assert s.structures[0].descendants and 1 not in s.structures[0].descendants
    assert 4 not in s.structures[0].descendants
    _assert_hierarchy_consistent(s)


def test_remove_struc_invalidates_frac_npix_cache():
    """The derived ``frac_npix_parent`` cache must not survive a removal."""
    s = _make_collection()
    # Give structures a pixel count so the property is computable, then prime
    # the cache.
    for i in s.structure_keys:
        s.structures[i]._npix = 10
    _ = s.frac_npix_parent
    assert s._frac_npix_parent is not None
    s.remove_struc(2)
    assert s._frac_npix_parent is None


def test_get_mask_uses_descendants_by_id_after_removal():
    """``get_mask``/``get_struc_map_mask`` roll up descendants by structure ID,
    so they stay correct after a removal makes the IDs non-contiguous."""
    s = _make_collection()
    s.remove_struc(1)  # keys become {0, 2, 3, 4}; trunk 0 now has children 2, 3

    # Rolling up trunk 0 with descendants must gather every still-live pixel
    # (0,2,3,4) — i.e. every finite pixel in struc_map.
    mask = s.get_mask(s_include=[0], use_descendants=True)
    np.testing.assert_array_equal(mask, np.isfinite(s.struc_map))

    # Rolling up branch 3 must gather exactly 3's and 4's pixels.
    smap = s.get_struc_map_mask(s_include=[3], use_descendants=True)
    defined = np.isfinite(smap)
    assert sorted(np.unique(smap[defined]).astype(int)) == [3, 4]


def test_frac_npix_parent_correct_after_removal():
    """After a removal leaves a non-contiguous ID gap, ``frac_npix_parent`` must
    still divide each structure's npix by its (re-parented) parent's npix —
    indexing by ID, not by array position."""
    s = _make_collection()
    # Distinct npix per structure so a positional mis-index would show up as a
    # wrong ratio rather than coincidentally-correct 1.0s.
    npix = {0: 100, 1: 50, 2: 20, 3: 30, 4: 10}
    for i, n in npix.items():
        s.structures[i]._npix = n

    # Remove interior branch 1: children 2 and 3 re-parent onto trunk 0.
    s.remove_struc(1)

    keys = list(s.structure_keys)  # [0, 2, 3, 4]
    frac = s.frac_npix_parent
    by_id = {k: frac[j] for j, k in enumerate(keys)}

    assert np.isnan(by_id[0])                       # trunk has no parent
    assert by_id[2] == pytest.approx(20 / 100)      # 2's parent is now 0
    assert by_id[3] == pytest.approx(30 / 100)      # 3's parent is now 0
    assert by_id[4] == pytest.approx(10 / 30)       # 4's parent is still 3


def test_reparenting_preserves_existing_siblings():
    """Re-parented children are added alongside the parent's existing children,
    not in place of them."""
    img_shape = (8,)

    def make(i, level):
        s = Structure(pi=[0, 0, 0, 0, 0, 0, 0, 0, 0], id=i, id_ph=i,
                      img_shape=img_shape)
        s._level = level
        return s

    # 0 (trunk) -> children 1, 5 ; 1 -> children 2, 3
    strucs = {i: make(i, lvl) for i, lvl in {0: 1, 1: 2, 5: 2, 2: 3, 3: 3}.items()}
    strucs[0]._children, strucs[0]._descendants = [1, 5], [0, 1, 2, 3, 5]
    strucs[1]._parent, strucs[1]._children, strucs[1]._descendants = 0, [2, 3], [1, 2, 3]
    strucs[5]._parent, strucs[5]._descendants = 0, [5]
    strucs[2]._parent, strucs[2]._descendants = 1, [2]
    strucs[3]._parent, strucs[3]._descendants = 1, [3]

    struc_map = np.array([0, 5, 1, 1, 2, 3, np.nan, np.nan], dtype=float)
    level_map = np.array([1, 2, 2, 2, 3, 3, np.nan, np.nan], dtype=float)
    s = Structures(structures=strucs, img_shape=img_shape,
                   struc_map=struc_map, level_map=level_map)
    s._refresh_hierarchy_cache()

    s.remove_struc(1)

    # Pre-existing sibling 5 survives; re-parented 2,3 join it under trunk 0.
    assert sorted(s.structures[0].children) == [2, 3, 5]
    assert s.structures[5].parent == 0  # untouched
    _assert_hierarchy_consistent(s)


def test_remove_same_id_twice_raises():
    """The one documented failure mode: removing an already-dissolved id."""
    s = _make_collection()
    s.remove_struc(2)
    with pytest.raises(KeyError):
        s.remove_struc(2)
    with pytest.raises(KeyError):
        _make_collection().remove_strucs([3, 3])


def test_batch_removal_order_independent():
    """Removing a set of distinct structures yields the same final hierarchy and
    maps regardless of the order they are listed in."""
    def final_state(order):
        s = _make_collection()
        s.remove_strucs(order)
        keys = sorted(s.structure_keys)
        return (
            keys,
            {k: s.structures[k].parent for k in keys},
            {k: sorted(s.structures[k].children) for k in keys},
            {k: sorted(s.structures[k].descendants) for k in keys},
            {k: s.structures[k].level for k in keys},
            np.nan_to_num(s.struc_map, nan=-1).tolist(),
            np.nan_to_num(s.level_map, nan=-1).tolist(),
        )

    assert final_state([1, 3]) == final_state([3, 1])
    assert final_state([2, 4]) == final_state([4, 2])


def test_remove_all_structures_blanks_maps():
    """Dissolving every structure (roots last) empties the collection and leaves
    the maps entirely NaN."""
    s = _make_collection()
    # Remove leaves/branches first, trunk last, so each removal is well defined.
    for sid in [4, 3, 2, 1, 0]:
        s.remove_struc(sid)
    assert s.n_struc == 0
    assert np.all(np.isnan(s.struc_map))
    assert np.all(np.isnan(s.level_map))
    assert s.trunk == [] and s.leaves == []


def test_remove_leaf_on_real_segmentation(strucs_2d_two_peaks_h0):
    """End-to-end check on a genuinely segmented collection: removing the only
    leaf merges it into the trunk and preserves the segmentation invariant
    (unique struc_map ids == n_struc)."""
    s = copy.deepcopy(strucs_2d_two_peaks_h0)  # don't mutate the session fixture
    trunk_id = s.trunk[0].id
    leaf_id = s.leaves[0].id

    s.remove_struc(leaf_id)

    assert leaf_id not in s.structure_keys
    assert s.n_struc == 1
    finite = s.struc_map[np.isfinite(s.struc_map)]
    assert len(np.unique(finite)) == s.n_struc            # segmentation invariant
    assert set(np.unique(finite).astype(int)) == {trunk_id}
    assert s.structures[trunk_id].is_leaf
    assert s.structures[trunk_id].children == []
    assert s.structures[trunk_id].level == 1
    assert np.all(s.level_map[np.isfinite(s.level_map)] == 1)


# ---------------------------------------------------------------------------
# Property-based test: every removal must match an independent recomputation of
# the entire post-state (hierarchy + struc_map + level_map) from first
# principles, over many random forests with non-contiguous IDs.
# ---------------------------------------------------------------------------

def _depth(nid, parent):
    """Nesting depth from the root (root == level 1) under a parent map."""
    d, p = 1, parent[nid]
    while p is not None:
        d, p = d + 1, parent[p]
    return d


def _children_of(ids, parent):
    ch = {i: [] for i in ids}
    for i in ids:
        if parent[i] is not None:
            ch[parent[i]].append(i)
    return ch


def _descendants_of(nid, ch):
    """Self-inclusive descendant set via DFS over a children map."""
    out, stack = set(), [nid]
    while stack:
        cur = stack.pop()
        out.add(cur)
        stack.extend(ch[cur])
    return out


def _build_random_forest(rng):
    """Build a valid synthetic ``Structures`` from a random rooted forest with
    non-contiguous IDs, plus the plain-Python model it was built from."""
    n = rng.randint(1, 8)
    ids = rng.sample(range(0, 3 * n + 1), n)  # distinct, non-contiguous

    parent = {}
    for pos, nid in enumerate(ids):
        earlier = ids[:pos]
        parent[nid] = None if (not earlier or rng.random() < 0.3) else rng.choice(earlier)

    ch = _children_of(ids, parent)
    depth = {i: _depth(i, parent) for i in ids}

    # Each structure owns a contiguous block of >=1 exclusive pixels.
    owner = []
    for i in ids:
        owner.extend([i] * rng.randint(1, 3))
    owner.extend([None, None])  # NaN padding

    img_shape = (len(owner),)
    strucs = {}
    for i in ids:
        st = Structure(pi=[0, 0, 0, 0, 0, 0, 0, 0, 0], id=i, id_ph=i,
                       img_shape=img_shape)
        st._level = depth[i]
        st._parent = parent[i]
        st._children = list(ch[i])
        st._descendants = list(_descendants_of(i, ch))
        strucs[i] = st

    struc_map = np.array([np.nan if o is None else o for o in owner], dtype=float)
    level_map = np.array([np.nan if o is None else depth[o] for o in owner], dtype=float)
    s = Structures(structures=strucs, img_shape=img_shape,
                   struc_map=struc_map, level_map=level_map)
    s._refresh_hierarchy_cache()
    return s, ids, parent, owner


@pytest.mark.parametrize("seed", range(300))
def test_remove_struc_matches_independent_recompute(seed):
    rng = random.Random(seed)
    s, ids, parent, owner = _build_random_forest(rng)
    x = rng.choice(ids)

    # --- independent oracle for the post-removal state ---------------------
    new_parent = {i: parent[i] for i in ids if i != x}
    for c in [i for i in ids if parent[i] == x]:      # x's children -> x's parent
        new_parent[c] = parent[x]
    remaining = [i for i in ids if i != x]
    new_ch = _children_of(remaining, new_parent)
    new_depth = {i: _depth(i, new_parent) for i in remaining}
    new_owner = [parent[x] if o == x else o for o in owner]
    exp_struc_map = np.array([np.nan if o is None else o for o in new_owner], dtype=float)
    exp_level_map = np.array([np.nan if o is None else new_depth[o] for o in new_owner],
                             dtype=float)

    # --- code under test ---------------------------------------------------
    s.remove_struc(x)

    assert set(s.structure_keys) == set(remaining)
    np.testing.assert_array_equal(s.struc_map, exp_struc_map)
    np.testing.assert_array_equal(s.level_map, exp_level_map)
    for i in remaining:
        st = s.structures[i]
        assert st.parent == new_parent[i]
        assert sorted(st.children) == sorted(new_ch[i])
        assert set(st.descendants) == _descendants_of(i, new_ch)
        assert st.level == new_depth[i]
    assert sorted(t.id for t in s.trunk) == sorted(i for i in remaining if new_parent[i] is None)
    assert sorted(l.id for l in s.leaves) == sorted(i for i in remaining if not new_ch[i])
