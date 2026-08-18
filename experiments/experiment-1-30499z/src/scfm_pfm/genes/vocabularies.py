"""Read each model's own gene vocabulary from its own checkpoint artifacts.

None of these needs the model's Python package: scGPT ships a JSON vocab,
AIDO.Cell's fixed gene order is a TSV in the ModelGenerator sdist, scPRINT
stores its gene list in the Lightning checkpoint's ``hyper_parameters``, and
TranscriptFormer's vocabulary is the key list of the ESM2 embedding HDF5 inside
its weights tarball.
"""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from ..data.sources import download
from ..models import sources as S


@dataclass
class Vocabulary:
    model: str
    tokens: list[str]
    kind: str  # "symbol" | "ensembl"
    ordered: bool  # True when the token order is a model input constraint
    provenance: dict = field(default_factory=dict)
    special_tokens: list[str] = field(default_factory=list)

    def as_record(self) -> dict:
        return {
            "model": self.model,
            "kind": self.kind,
            "ordered": self.ordered,
            "size": len(self.tokens),
            "special_tokens": self.special_tokens,
            "provenance": self.provenance,
            "head": self.tokens[:8],
        }


def scgpt_vocabulary(cache: Path) -> Vocabulary:
    path = cache / "scgpt" / "vocab.json"
    meta = download(S.hf_url(S.SCGPT, "vocab"), path)
    raw = json.loads(path.read_text())
    # position must equal the model's own token id, so order by id and keep gaps
    by_id = {int(v): k for k, v in raw.items()}
    tokens = [by_id.get(i, f"<unused_{i}>") for i in range(max(by_id) + 1)]
    special = [t for t in tokens if t.startswith("<") and t.endswith(">")]
    return Vocabulary(
        model="scgpt",
        tokens=tokens,
        kind="symbol",
        ordered=False,
        special_tokens=special,
        provenance={
            "repo": S.SCGPT["repo"],
            "revision": S.SCGPT["revision"],
            "file": S.SCGPT["files"]["vocab"],
            "md5": meta.get("md5"),
            "token_ids_are_vocab_values": True,
        },
    )


def aido_vocabulary(cache: Path) -> Vocabulary:
    """The fixed 19,264-gene order AIDO.Cell was pretrained on (symbols)."""
    tgz = cache / "aido_cell" / "modelgenerator_sdist.tar.gz"
    meta = download(S.AIDO_CELL["gene_order_sdist"], tgz)
    with tarfile.open(tgz, "r:gz") as tf:
        member = tf.extractfile(S.AIDO_CELL["gene_order_member"])
        assert member is not None
        text = member.read().decode()
        # also stage the two self-contained HF modeling files for the demo job
        out = cache / "aido_cell" / "cellfoundation"
        if not (out / "modeling_cellfoundation.py").exists():
            out.mkdir(parents=True, exist_ok=True)
            for name in S.AIDO_CELL["modeling_members"]:
                src = tf.extractfile(name)
                if src is not None:
                    (out / Path(name).name).write_bytes(src.read())
    lines = text.strip().splitlines()
    header = lines[0].split("\t")
    col = header.index("gene_name")
    tokens = [ln.split("\t")[col] for ln in lines[1:]]
    return Vocabulary(
        model="aido_cell",
        tokens=tokens,
        kind="symbol",
        ordered=True,
        provenance={
            "source": S.AIDO_CELL["gene_order_sdist"],
            "member": S.AIDO_CELL["gene_order_member"],
            "md5": meta.get("md5"),
            "note": "fixed input order; position i of the model input is this gene",
        },
    )


def scprint_vocabulary(cache: Path) -> Vocabulary:
    import torch

    path = cache / "scprint" / "medium-v1.5.ckpt"
    meta = download(S.hf_url(S.SCPRINT, "weights"), path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    genes = list(hp.get("genes", []))
    if not genes:
        raise RuntimeError("scPRINT checkpoint carries no 'genes' hyper_parameter")
    kind = "ensembl" if str(genes[0]).startswith("ENSG") else "symbol"
    return Vocabulary(
        model="scprint",
        tokens=[str(g) for g in genes],
        kind=kind,
        ordered=True,
        provenance={
            "repo": S.SCPRINT["repo"],
            "revision": S.SCPRINT["revision"],
            "file": S.SCPRINT["files"]["weights"],
            "md5": meta.get("md5"),
            "source": "checkpoint hyper_parameters['genes']",
            "checkpoint_hparam_keys": sorted(k for k in hp if k != "genes")[:40],
        },
    )


def transcriptformer_vocabulary(cache: Path) -> Vocabulary:
    """Gene names are the key list of the ESM2 embedding HDF5 in the tarball."""
    import h5py

    tgz = cache / "transcriptformer" / "tf_sapiens.tar.gz"
    meta = download(S.TRANSCRIPTFORMER["url"], tgz)
    extract_dir = cache / "transcriptformer" / "tf_sapiens"
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(extract_dir)  # noqa: S202 - trusted CZI release tarball
    h5s = sorted(extract_dir.rglob("*.h5"))
    esm = [p for p in h5s if "esm" in p.name.lower()] or h5s
    if not esm:
        raise RuntimeError(f"no ESM2 embedding .h5 found under {extract_dir}")
    tokens: list[str] = []
    used = []
    for path in esm:
        with h5py.File(path, "r") as f:
            if "keys" not in f:
                continue
            keys = [k.decode() if isinstance(k, bytes) else str(k) for k in f["keys"][:]]
        tokens += keys
        used.append(str(path.relative_to(extract_dir)))
    seen: set[str] = set()
    dedup = [t for t in tokens if not (t in seen or seen.add(t))]
    # the special-token ORDER is a checkpoint property: tf_sapiens/config.json puts
    # [PAD] second, which differs from the package-level SPECIAL_TOKENS constant
    cfgs = sorted(extract_dir.rglob("config.json"))
    specials = ["unknown", "[START]", "[END]", "[RD]", "[CELL]", "[PAD]", "[MASK]"]
    specials_source = "transcriptformer.tokenizer.vocab.SPECIAL_TOKENS (fallback)"
    if cfgs:
        cfg = json.loads(cfgs[0].read_text())
        from_cfg = cfg.get("model", {}).get("data_config", {}).get("special_tokens")
        if from_cfg:
            specials = [str(x) for x in from_cfg]
            specials_source = f"{cfgs[0].name}: model.data_config.special_tokens"
    kind = "ensembl" if dedup and dedup[0].startswith("ENSG") else "symbol"
    # token id = index into (special tokens, then gene list); keep that alignment
    dedup = specials + dedup
    return Vocabulary(
        model="transcriptformer",
        tokens=dedup,
        kind=kind,
        ordered=False,
        special_tokens=specials,
        provenance={
            "bucket": S.TRANSCRIPTFORMER["bucket"],
            "object": S.TRANSCRIPTFORMER["object"],
            "md5": meta.get("md5"),
            "embedding_files": used,
            "all_files": [str(p.relative_to(extract_dir)) for p in sorted(extract_dir.rglob("*"))[:60]],
            "special_tokens_source": specials_source,
            "note": "token ids are the special tokens then the gene list, in this order "
            "(transcriptformer.tokenizer.vocab.build_gene_vocab_from_list)",
        },
    )


BUILDERS = {
    "scgpt": scgpt_vocabulary,
    "aido_cell": aido_vocabulary,
    "scprint": scprint_vocabulary,
    "transcriptformer": transcriptformer_vocabulary,
}


def load_all(cache: Path) -> dict[str, Vocabulary]:
    out = {}
    for name, fn in BUILDERS.items():
        print(f"[vocab] {name}", flush=True)
        out[name] = fn(cache)
        print(f"[vocab] {name}: {len(out[name].tokens)} {out[name].kind} tokens", flush=True)
    return out


def _unused() -> None:  # pragma: no cover - keeps io import meaningful for linters
    io.StringIO()
