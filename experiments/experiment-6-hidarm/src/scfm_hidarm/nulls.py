"""The nulls a run-6 arm is read against, written as further pseudo-model blocks.

Every null here is installed in exactly the format ``install.py`` writes, which
is the whole reason run 3 does not have to change: ``ArmFeatures._resp_dim``
reads the width off the block, so a null written this way is width-matched to
its arm *by construction*. Run 3's own matched-dimension pair null is hardcoded
to 11 (``translate/arms.py:67``), so a 10k-wide arm has no matched null in run
3's code, and widening it would mean editing a file run 4 also imports.

Three nulls, controlling for three different things. Reading a score against the
wrong one is the error this module exists to prevent, so each says what it does
*not* control for:

``hidperm`` --- **the machinery null.** The real projected features with the
perturbed-gene index permuted independently per read-out gene. It preserves the
marginal distribution, the basis, the covariance structure, the dose axis and
the per-slot scales, and destroys only the pair-specific association. This is
the one that bites: an arm that beats ``hidrand`` but not ``hidperm`` is
reporting the *shape* of a hidden-state feature, not what the model says about
this perturbation on this gene.

``hidrand`` --- **width only.** ``config.HIDRAND_NOTE`` records the argument:
if ``z ~ N(0, s^2 I_d)`` then ``W' z ~ N(0, s^2 W'W) = N(0, s^2 I_k)`` for an
orthonormal ``W``, independent of ``W``. So drawing in ``d`` and projecting
would burn 15--94 GB of RNG per block to reach a distribution that is available
in ``k`` dimensions directly, and it would still control for nothing but width.
The substantive machinery controls are ``hidperm`` and ``hidrp``.

``hidrp`` --- **the basis.** Real ``delta-h`` through a frozen random orthonormal
basis, which isolates "the PCA basis is doing the work" from "``delta-h`` carries
pair-specific signal in any k-dimensional slice". It is *not* generated here: it
is a real forward-pass feature and comes from the extractor. All this module
owes it is ``random_orthonormal``, which the extractor imports so that the
random basis and the nulls are drawn by one seeded routine rather than two.

Nothing here is stored by the extractor. A permutation of a stored array and a
seeded Gaussian are both reproducible from a seed, so ``config.DERIVED_VARIANTS``
derives them at stage time and the extraction budget is not tripled for no
information. Every draw seeds from ``config.NULL_SEEDS[variant]`` **and** from
the key it is drawn for, so two keys never share a permutation --- otherwise the
same row shuffle would be applied to every layer of a depth sweep and the nulls
would be correlated with each other across the very axis H4 reads.

    python -m scfm_hidarm.nulls --block <features/key__mean_control.npz> --variant hidperm
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import config as C
from .install import MANIFEST_DIRNAME, block_filename, check_key

#: which null variant each real variant owns. ``hidrp`` is deliberately absent:
#: its own permutation null would encode to the same key as ``hid``'s, and two
#: different arrays under one key is how a block gets silently overwritten.
PERM_VARIANT_OF = {"hid": "hidperm", "norm": "normperm"}
RAND_VARIANT_OF = {"hid": "hidrand", "norm": "normrand"}


def derive_seed(variant: str, key: str) -> int:
    """A seed from the variant AND the key it is drawn for.

    ``hash()`` is salted per process and would make a null irreproducible
    between two sessions of the same run, which is exactly the failure a seeded
    null exists to rule out. blake2b is stable everywhere.
    """
    if variant not in C.NULL_SEEDS:
        raise ValueError(f"no null seed declared for variant {variant!r}: {sorted(C.NULL_SEEDS)}")
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest()
    return int(C.NULL_SEEDS[variant]) + int.from_bytes(digest, "big")


def null_key(real_key: str, variant: str) -> str:
    """The key a null block is installed under, decoded from the real arm's."""
    d = check_key(real_key)
    return C.model_stem(d["model"], variant, d["layer"], d["k"])


def _resolve_null(real_npz_path: Path, key: str | None, variant_of: dict[str, str],
                  seed: int | None) -> tuple[str, str, int]:
    """The (key, variant, seed) a derived block is written under.

    The real key is read off the *file name*, because that is what run 3 reads
    it off too --- deriving it from anything else would let the null and the arm
    disagree about which block the null is for.
    """
    real_key = Path(real_npz_path).stem.split("__", 1)[0]
    if key is None:
        v = C.run_of_stem(real_key)["variant"]
        if v not in variant_of:
            raise ValueError(
                f"no null of this family is declared for variant {v!r} "
                f"({sorted(variant_of)}); pass an explicit key"
            )
        variant = variant_of[v]
        key = null_key(real_key, variant)
    else:
        variant = C.run_of_stem(key)["variant"]
    if seed is None:
        seed = derive_seed(variant, key)
    return key, variant, int(seed)


def per_slot_scale(arr: np.ndarray, k: int) -> np.ndarray:
    """RMS of each ``k``-wide slot of one form, in the block's own units.

    RMS about zero rather than the standard deviation, because the null it
    matches is a zero-mean Gaussian and the quantity that has to agree is the
    second moment. Run 3's ``Standardizer`` z-scores every column in fold
    anyway, so this scale cannot reach a score --- it exists so that a plot of
    the null beside the arm is on one axis and so that a badly scaled draw is
    visible rather than invisible.
    """
    n_slots = arr.shape[1] // k
    if n_slots * k != arr.shape[1]:
        raise RuntimeError(
            f"axis 1 is {arr.shape[1]} wide, which is not a whole number of k={k} slots; the "
            "per-slot scale would then be computed over a slot boundary"
        )
    blocks = arr.reshape(arr.shape[0], n_slots, k, arr.shape[2]).astype(np.float64)
    return np.sqrt((blocks**2).mean(axis=(0, 2, 3)))


# --------------------------------------------------------------------------- #
# reading and re-writing a real block
# --------------------------------------------------------------------------- #
def _read_block(real_npz_path: Path) -> tuple[dict, dict[str, np.ndarray], int]:
    """Every array of a real block, split into forms and everything else."""
    z = np.load(Path(real_npz_path), allow_pickle=False)
    if C.DROPPED_RUN2_SLOT in z.files:
        raise RuntimeError(
            f"{real_npz_path}: carries a {C.DROPPED_RUN2_SLOT!r} array, which run 6 drops "
            f"({C.DROPPED_RUN2_SLOT_REASON}); a null derived from it would be one column wider "
            "than the arm it is the null for"
        )
    if "vvp" not in z.files:
        raise RuntimeError(f"{real_npz_path}: no 'vvp' array; this is not a run-6 block")
    forms = {f: np.asarray(z[f]) for f in C.SLOT_FORMS if f in z.files}
    rest = {n: np.asarray(z[n]) for n in z.files if n not in C.SLOT_FORMS}
    pert_index = rest["pert_index"].astype(np.int64)
    n_pert = int(forms["vvp"].shape[0])
    if not np.array_equal(pert_index, np.arange(n_pert, dtype=np.int64)):
        raise RuntimeError(
            f"{real_npz_path}: pert_index is not arange({n_pert}); a null inherits the real "
            "block's coverage, and an uncovered row makes runner.run_cell silently skip the arm"
        )
    k = int(np.unique(rest["pc_index"]).size) if "pc_index" in rest else 1
    return rest, forms, k


def _write_derived(
    out_path: Path,
    key: str,
    rest: dict[str, np.ndarray],
    forms: dict[str, np.ndarray],
    manifest: dict,
) -> dict:
    """Write a derived block and its manifest, with the on-disk name checked.

    The file name is what run 3 parses back into ``(model, anchor)``, so a name
    that does not match the key would install the null under a *different*
    pseudo-model than the one every table joins on --- and would still score.
    """
    check_key(key)
    out_path = Path(out_path)
    anchor = out_path.stem.split("__", 1)[1] if "__" in out_path.stem else C.ANCHOR
    if out_path.name != block_filename(key, anchor):
        raise RuntimeError(
            f"{out_path.name} is not the file name for key {key!r} "
            f"({block_filename(key, anchor)}); run 3 reads the key off the file name, so the "
            "block would be scored under a name nothing else in run 6 refers to"
        )
    payload = dict(rest)
    payload.update({f: np.asarray(a, dtype=np.float32) for f, a in forms.items()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    manifest = {
        "key": key,
        "arm": C.arm_name(key),
        "anchor": anchor,
        "block": str(out_path),
        "forms": sorted(forms),
        "npz_keys": sorted(payload),
        "slot_axis_order": C.SLOT_AXIS_ORDER,
        **manifest,
    }
    mdir = out_path.parent.parent / MANIFEST_DIRNAME
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"{out_path.stem}.json").write_text(json.dumps(manifest, indent=2))
    print(f"[nulls] {key}: {sorted(forms)} -> {out_path}", flush=True)
    return manifest


# --------------------------------------------------------------------------- #
# the machinery null
# --------------------------------------------------------------------------- #
def hidperm(real_npz_path: Path, out_path: Path, key: str | None = None,
            seed: int | None = None) -> dict:
    """The real features with the perturbed-gene index permuted per read-out gene.

    One permutation per read-out gene, and the **same** permutation for ``vvp``
    and ``gdt`` at that gene. Permuting the two forms independently would make
    the two halves of one feature vector describe two different perturbations,
    which destroys the within-feature coherence as well as the pairing --- a
    strictly easier null to beat, and one that no longer isolates the pair
    association.

    Because every column is a pure gather along axis 0, each column's multiset
    of values is unchanged: this null preserves the marginals, the basis, the
    per-slot scales and the dose axis exactly, and destroys nothing but which
    perturbation a value belongs to. ``normperm`` is the identical operation on
    the stage-0 magnitude block, which is why one function writes both.
    """
    rest, forms, _ = _read_block(real_npz_path)
    key, variant, seed = _resolve_null(real_npz_path, key, PERM_VARIANT_OF, seed)
    rng = np.random.default_rng(seed)

    n_pert, _, n_readout = forms["vvp"].shape
    perms = np.stack([rng.permutation(n_pert) for _ in range(n_readout)], axis=1)
    out = {}
    for form in sorted(forms):
        src = forms[form]
        dst = np.empty_like(src)
        for g in range(n_readout):
            dst[:, :, g] = src[perms[:, g], :, g]
        out[form] = dst
    fixed = int((perms == np.arange(n_pert)[:, None]).sum())
    return _write_derived(
        out_path, key, rest, out,
        {
            "variant": variant,
            "derived_from": str(real_npz_path),
            "seed": int(seed),
            "controls_for": "everything except the pair-specific association",
            "permutation": "independent per read-out gene, shared by vvp and gdt",
            "n_fixed_points": fixed,
            "n_pairs": int(n_pert * n_readout),
        },
    )


# --------------------------------------------------------------------------- #
# the width-matched frozen random null
# --------------------------------------------------------------------------- #
def hidrand_slots(shape: tuple[int, int, int], k: int, scale: np.ndarray,
                  seed: int | list[int]) -> np.ndarray:
    """iid Gaussian in ``k`` per slot, at the given per-slot scale.

    ``config.HIDRAND_NOTE``: projecting a d-dimensional isotropic Gaussian
    through an orthonormal ``W`` gives exactly ``N(0, s^2 I_k)``, independent of
    ``W``. So drawing in ``d`` to "carry the same projection" would burn 15--94
    GB of RNG per block to reach a distribution available in ``k`` dimensions
    directly, and would control for no more than this does. **This null controls
    for width only**; the substantive machinery controls are ``hidperm`` and
    ``hidrp``.
    """
    n_pert, n_cols, n_readout = shape
    n_slots = n_cols // k
    if n_slots * k != n_cols or n_slots != scale.size:
        raise RuntimeError(
            f"cannot draw {n_cols} columns as {scale.size} slots of k={k}; the draw would not "
            "be per-slot and the scale match would be meaningless"
        )
    rng = np.random.default_rng(seed)
    out = np.empty((n_pert, n_cols, n_readout), dtype=np.float32)
    for s in range(n_slots):
        draw = rng.standard_normal((n_pert, k, n_readout)) * float(scale[s])
        out[:, s * k : (s + 1) * k, :] = draw.astype(np.float32)
    return out


def hidrand(real_npz_path: Path, out_path: Path, key: str | None = None,
            seed: int | None = None) -> dict:
    """A width-matched frozen random block, drawn once at its seed and stored.

    Frozen rather than redrawn per fold on purpose: run 3's ``randgene`` and
    ``randpair`` nulls are fixed arrays too, so a redrawn null would be a
    different kind of control from the ones every earlier arm was read against.
    """
    rest, forms, k = _read_block(real_npz_path)
    key, variant, seed = _resolve_null(real_npz_path, key, RAND_VARIANT_OF, seed)

    out, scales = {}, {}
    # sorted, so the draw order is a property of the block and not of dict order
    for i, form in enumerate(sorted(forms)):
        s = per_slot_scale(forms[form], k)
        # a distinct stream per form: one stream shared across forms would make
        # ``gdt`` a deterministic continuation of ``vvp`` rather than an
        # independent draw, and the two halves of the null would be correlated
        out[form] = hidrand_slots(forms[form].shape, k, s, [seed, i])
        scales[form] = [float(v) for v in s]
    return _write_derived(
        out_path, key, rest, out,
        {
            "variant": variant,
            "derived_from": str(real_npz_path),
            "seed": int(seed),
            "k": int(k),
            "per_slot_rms": scales,
            "controls_for": "width only",
            "note": C.HIDRAND_NOTE,
        },
    )


# --------------------------------------------------------------------------- #
# the random basis ``hidrp`` is read through
# --------------------------------------------------------------------------- #
def random_orthonormal(d: int, k: int, seed: int) -> np.ndarray:
    """A frozen random orthonormal ``d -> k`` basis, by QR of a seeded Gaussian.

    Orthonormal rather than an arbitrary random projection because the PCA basis
    it is the control for is orthonormal: a Gaussian random matrix would change
    the geometry as well as the directions, and ``hid`` vs ``hidrp`` would then
    confound "the basis directions matter" with "the projection is not an
    isometry on its image".

    The R-diagonal sign fix makes the basis a function of the seed alone. LAPACK
    is free to return either sign, and a basis that flipped between two runs of
    the same job would silently change ``hidrp``'s features while every manifest
    still recorded the same seed.
    """
    if k > d:
        raise ValueError(f"cannot draw {k} orthonormal directions in {d} dimensions")
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((d, k)))
    q = q * np.sign(np.diag(r))[None, :]
    err = float(np.abs(q.T @ q - np.eye(k)).max())
    if not err < 1e-5:
        raise RuntimeError(
            f"the random basis is not orthonormal: max |W'W - I| = {err:.3e}; hidrp would then "
            "differ from hid in the geometry of the projection as well as in its directions, "
            "and would stop being a control for the basis"
        )
    return np.ascontiguousarray(q, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--block", required=True, type=Path,
                   help="the real block to derive from, inside a staged bundle's features/")
    p.add_argument("--variant", required=True, choices=["hidperm", "hidrand",
                                                        "normperm", "normrand"])
    a = p.parse_args(argv)

    real_key, anchor = a.block.stem.split("__", 1)
    key = null_key(real_key, a.variant)
    out = a.block.parent / block_filename(key, anchor)
    writer = hidperm if a.variant.endswith("perm") else hidrand
    writer(a.block, out, key=key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
