"""Checkpoint locations for the four single-cell foundation models.

All four were verified public and readable this session. Revisions are the
HuggingFace commit shas observed on 2026-08-18; the TranscriptFormer weights
come from an unsigned public S3 bucket and the AIDO gene order from an
immutable PyPI sdist.

Note on identity: the plan names ``hf://genbio-ai/AIDO.Cell-100M``. That repo
now 307-redirects to ``genbio-ai/GB.Cell-100M`` -- the same model, renamed
upstream. We resolve and record the redirect target rather than the stale name.
"""

from __future__ import annotations

MODELS = ("scgpt", "aido_cell", "scprint", "transcriptformer")

HF_BASE = "https://huggingface.co"

SCGPT = {
    "repo": "wanglab/scGPT-human",
    "revision": "a24c237737a40f3720f75abb555489e9fe753be6",
    "files": {
        "vocab": "vocab.json",
        "args": "args.json",
        "weights": "best_model.pt",
    },
}

AIDO_CELL = {
    "repo": "genbio-ai/GB.Cell-100M",
    "plan_alias": "genbio-ai/AIDO.Cell-100M",
    "revision": "a983b388e1025860f2c413ecef60e76078253c24",
    "files": {
        "config": "config.json",
        "weights": "model.safetensors",
    },
    # fixed 19,264-gene order the checkpoint was pretrained on
    "gene_order_sdist": (
        "https://files.pythonhosted.org/packages/65/3c/"
        "4f36684dad8b4cc5b3145dea2f05f25fd8a3ae29934f90e30f31f1ca377b/"
        "modelgenerator-0.1.3.post0.tar.gz"
    ),
    "gene_order_member": (
        "modelgenerator-0.1.3.post0/modelgenerator/cell/gene_lists/"
        "OS_scRNA_gene_index.19264.tsv"
    ),
    "modeling_members": (
        "modelgenerator-0.1.3.post0/modelgenerator/huggingface_models/cellfoundation/"
        "configuration_cellfoundation.py",
        "modelgenerator-0.1.3.post0/modelgenerator/huggingface_models/cellfoundation/"
        "modeling_cellfoundation.py",
    ),
}

SCPRINT = {
    "repo": "jkobject/scPRINT",
    "revision": "8e1ad1a1bc40c1fc2f4f1700fe27923eea9a2bed",
    "files": {"weights": "medium-v1.5.ckpt"},
}

TRANSCRIPTFORMER = {
    "bucket": "czi-transcriptformer",
    "object": "weights/tf_sapiens.tar.gz",
    "url": "https://czi-transcriptformer.s3.amazonaws.com/weights/tf_sapiens.tar.gz",
    "size_bytes": 1819975447,
}

HGNC_COMPLETE_SET = (
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/"
    "hgnc_complete_set.txt"
)


def hf_url(entry: dict, key: str) -> str:
    return f"{HF_BASE}/{entry['repo']}/resolve/{entry['revision']}/{entry['files'][key]}"
