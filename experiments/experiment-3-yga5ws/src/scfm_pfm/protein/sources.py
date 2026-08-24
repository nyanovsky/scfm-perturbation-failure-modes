"""Where the audit and the protein arm read their inputs.

The four single-cell checkpoints are the same objects run 2 loaded, at the same
revisions (``experiments/experiment-2-nygwcv/src/scfm_pfm/models/sources.py``);
they are repeated here rather than imported so that this run's provenance record
is self-contained and does not silently change if the sibling directory does.

ESM2 comes from Meta's release on the HuggingFace hub, which needs no token.
Two sizes are used, and which one an arm uses is a width question: a model whose
gene table is 2,560-wide can only be compared like-for-like against the ESM2
checkpoint that produces 2,560-wide embeddings.
"""

from __future__ import annotations

HF_BASE = "https://huggingface.co"

SCGPT = {
    "repo": "wanglab/scGPT-human",
    "revision": "a24c237737a40f3720f75abb555489e9fe753be6",
    "files": {"vocab": "vocab.json", "args": "args.json", "weights": "best_model.pt"},
}

AIDO_CELL = {
    "repo": "genbio-ai/GB.Cell-100M",
    "plan_alias": "genbio-ai/AIDO.Cell-100M",
    "revision": "a983b388e1025860f2c413ecef60e76078253c24",
    "files": {"config": "config.json", "weights": "model.safetensors"},
    "gene_order_sdist": (
        "https://files.pythonhosted.org/packages/65/3c/"
        "4f36684dad8b4cc5b3145dea2f05f25fd8a3ae29934f90e30f31f1ca377b/"
        "modelgenerator-0.1.3.post0.tar.gz"
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
    "tree_api": "https://huggingface.co/api/models/jkobject/scPRINT/tree/main",
}

TRANSCRIPTFORMER = {
    "url": "https://czi-transcriptformer.s3.amazonaws.com/weights/tf_sapiens.tar.gz",
    "size_bytes": 1819975447,
}

#: public ESM2 checkpoints, by the embedding width they produce
ESM2 = {
    "esm2_650m": {"repo": "facebook/esm2_t33_650M_UR50D", "dim": 1280, "layers": 33},
    "esm2_3b": {"repo": "facebook/esm2_t36_3B_UR50D", "dim": 2560, "layers": 36},
}
#: ESM2's trained context is 1,022 residues plus the two special tokens
ESM2_MAX_RESIDUES = 1022

#: upstream source files read as corroboration for the checkpoint evidence
SOURCE_SNIPPETS = {
    "scprint_encoders": (
        "https://raw.githubusercontent.com/jkobject/scPRINT/main/scprint/model/encoders.py"
    ),
    "scprint_model": (
        "https://raw.githubusercontent.com/jkobject/scPRINT/main/scprint/model/model.py"
    ),
    "scgpt_model": (
        "https://raw.githubusercontent.com/bowang-lab/scGPT/main/scgpt/model/model.py"
    ),
    "transcriptformer_model": (
        "https://raw.githubusercontent.com/czi-ai/transcriptformer/main/"
        "src/transcriptformer/model/model.py"
    ),
    "transcriptformer_layers": (
        "https://raw.githubusercontent.com/czi-ai/transcriptformer/main/"
        "src/transcriptformer/model/layers.py"
    ),
    "transcriptformer_vocab": (
        "https://raw.githubusercontent.com/czi-ai/transcriptformer/main/"
        "src/transcriptformer/tokenizer/vocab.py"
    ),
}

#: TranscriptFormer builds its embedding matrix as seven random special-token
#: rows followed by the genes of the ESM2 mapping file, in that file's own key
#: order, with this seed (tokenizer/vocab.py::construct_gene_embeddings).
TF_SPECIAL_TOKEN_SEED = 42

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"


def hf_url(entry: dict, key: str) -> str:
    return f"{HF_BASE}/{entry['repo']}/resolve/{entry['revision']}/{entry['files'][key]}"
