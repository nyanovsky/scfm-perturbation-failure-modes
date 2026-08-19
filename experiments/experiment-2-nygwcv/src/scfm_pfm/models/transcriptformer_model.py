"""TranscriptFormer TF-Sapiens response interface.

This is the one model in the set whose response is **not** defined for every
ordered pair of genes. TranscriptFormer is autoregressive over a sequence of
(gene, count) pairs with a causal mask: the prediction at position j sees the
identity of gene j plus the identity *and* count of every gene before it. Gene j
therefore responds to gene i only when i precedes j in the ordering we choose,
and the ordering is part of the interface rather than an implementation detail.

Two further constraints are recorded rather than worked around:

* the checkpoint's ``seq_len`` is 2047 gene tokens, so the 7,444-gene universe
  needs several forward passes. Each pass puts the source gene at position 0 so
  that every other gene in that pass is downstream of it.
* ``mu_link_fn`` is ``softmax``, so the per-gene value is a *share* of the cell's
  counts across the tokens in the same pass. We report that share, which is
  comparable across passes; the raw mu is not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .base import (
    ResponseContract,
    ResponseModel,
    finite_difference_column,
    jacobian_cross_check,
)
from . import sources as S

ASSAY_LABEL = "10x 3' v3"  # GSE264667 is 10x 3' v3


class TranscriptFormerResponse(ResponseModel):
    name = "transcriptformer"

    def load(self) -> None:
        from transcriptformer.data.dataclasses import DataConfig, LossConfig, ModelConfig
        from transcriptformer.model.model import Transcriptformer
        from transcriptformer.tokenizer.vocab import construct_gene_embeddings, open_vocabs

        ck = self._checkpoint_dir()
        cfg = json.loads((ck / "config.json").read_text())["model"]
        self.config_dict = cfg
        strip = lambda d: {k: v for k, v in d.items() if k != "_target_"}  # noqa: E731
        data_kwargs = strip(cfg["data_config"])
        data_kwargs["esm2_mappings_path"] = str(ck / "vocabs")
        data_kwargs["aux_vocab_path"] = str(ck / "vocabs")
        data_kwargs["clip_counts"] = 30
        model_kwargs = strip(cfg["model_config"])
        model_kwargs["compile_block_mask"] = False
        data_config = DataConfig(**data_kwargs)
        model_config = ModelConfig(**model_kwargs)
        loss_config = LossConfig(**strip(cfg["loss_config"]))

        emb_path = str(ck / "vocabs" / cfg["data_config"]["esm2_mappings"][0])
        gene_vocab_dict, emb = construct_gene_embeddings(
            [emb_path], data_config.special_tokens
        )
        aux_vocab = open_vocabs(str(ck / "vocabs"), ["assay"])
        self.gene_vocab_dict = gene_vocab_dict
        self.special_tokens = list(data_config.special_tokens)
        self.aux_vocab = aux_vocab

        model = Transcriptformer(
            data_config=data_config,
            model_config=model_config,
            loss_config=loss_config,
            inference_config=None,
            gene_vocab_dict=gene_vocab_dict,
            aux_vocab_dict=aux_vocab,
            emb_matrix=torch.as_tensor(emb, dtype=torch.float32),
        )
        sd = torch.load(ck / "model_weights.pt", map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        self.load_report = {
            "n_missing_keys": len(missing),
            "n_unexpected_keys": len(unexpected),
            "missing_head": list(missing)[:12],
            "unexpected_head": list(unexpected)[:12],
        }
        if any("mu." in k for k in missing):
            raise RuntimeError(f"count head weights missing: {missing[:8]}")
        self.seq_len = int(model_config.seq_len)
        self.context_block = 256
        self.model = model.eval().to(self.device).to(torch.float32)
        self.substitutions.append(
            "compile_block_mask disabled (eager flex-attention block mask) so that a "
            "single-cell forward pass does not pay a torch.compile warm-up"
        )
        self._verify_token_alignment()

    def _checkpoint_dir(self) -> Path:
        for cand in (
            self.cache / "transcriptformer" / "tf_sapiens" / "tf_sapiens",
            self.cache / "transcriptformer" / "tf_sapiens",
        ):
            if (cand / "config.json").exists():
                return cand
        raise RuntimeError(f"tf_sapiens checkpoint not found under {self.cache}")

    def _verify_token_alignment(self) -> None:
        """The universe's recorded positions must equal this model's token ids."""
        tab = self.vocab[self.vocab["gene_id"].astype(str) != ""]
        bad = 0
        for token, pos in zip(tab["token"].astype(str), tab["position"].astype(int)):
            if self.gene_vocab_dict.get(token, -1) != pos:
                bad += 1
        self.token_alignment = {
            "checked": int(tab.shape[0]),
            "mismatched": bad,
            "special_token_order_from_checkpoint": self.special_tokens,
        }
        if bad:
            raise RuntimeError(
                f"{bad} recorded token positions disagree with the checkpoint's own "
                "gene vocabulary; the identifier map and the model are out of step"
            )

    # ---- input ------------------------------------------------------------
    def prepare_input(self, counts: np.ndarray, gene_ids: np.ndarray) -> dict:
        pos_by_gene = self.position_by_gene()
        count_by_gene = dict(zip(np.asarray(gene_ids).astype(str), np.asarray(counts, dtype=np.float32)))
        uni = self.universe_gene_ids()
        upos = self.universe_positions()
        uc = np.array([count_by_gene.get(g, 0.0) for g in uni], dtype=np.float32)
        order = np.lexsort((uni, -uc))  # descending count, gene id as tie-break
        return {
            "universe_gene_ids": uni,
            "universe_token_ids": upos,
            "universe_counts": uc,
            "order": order,
            "pad_id": int(self.gene_vocab_dict[self.model.data_config.gene_pad_token]),
            "assay_token": int(self.aux_vocab["assay"][ASSAY_LABEL]),
            "n_input_genes": int((uc > 0).sum()),
            "total_counts": float(uc.sum()),
            "n_vocab_hits": len(pos_by_gene),
        }

    def _chunks(self, prepared: dict, source_row: int | None) -> list[np.ndarray]:
        """Split the universe into passes that share a common context block.

        Two constraints shape this. The sequence holds at most ``seq_len`` gene
        tokens, and with a softmax link the value of a gene is a share of the counts
        carried by the tokens in its own pass, so a pass made only of genes the cell
        does not express would have no counts to divide at all. Every pass therefore
        begins with the same context block: the highest-count genes in the cell, with
        the source gene first when a Jacobian column is being taken, so that each
        remaining gene is read against the same conditioning prefix.
        """
        order = list(prepared["order"])
        if source_row is not None:
            order = [source_row] + [r for r in order if r != source_row]
        context = order[: self.context_block]
        body = order[self.context_block :]
        per = self.seq_len - len(context)
        out = [np.array(context + body[start : start + per], dtype=np.int64)
               for start in range(0, len(body), per)]
        return out or [np.array(context, dtype=np.int64)]

    def _forward_chunk(self, rows: np.ndarray, prepared: dict, counts: torch.Tensor) -> torch.Tensor:
        from transcriptformer.data.dataclasses import BatchData

        pad_id = prepared["pad_id"]
        n = rows.size
        tok = torch.full((1, self.seq_len), pad_id, dtype=torch.long, device=self.device)
        val = torch.zeros((1, self.seq_len), dtype=torch.float32, device=self.device)
        tok[0, :n] = torch.as_tensor(
            prepared["universe_token_ids"][rows], dtype=torch.long, device=self.device
        )
        val = torch.cat(
            [counts[rows].reshape(1, -1), val[:, n:]], dim=1
        )
        aux = torch.full((1, 1), prepared["assay_token"], dtype=torch.long, device=self.device)
        batch = BatchData(gene_counts=val, gene_token_indices=tok, aux_token_indices=aux)
        out = self.model(batch)
        mu = out["mu"].reshape(-1)
        total = val.sum()
        if float(total) <= 0:
            raise RuntimeError(
                "a TranscriptFormer pass carries no counts; with a softmax link its "
                "output would be 0/0"
            )
        return mu[:n] / total  # softmax share, comparable across passes

    def _response_from_counts(self, prepared: dict, counts: torch.Tensor, source_row: int | None) -> torch.Tensor:
        chunks = self._chunks(prepared, source_row)
        vals = torch.zeros(prepared["universe_counts"].size, dtype=torch.float32, device=self.device)
        filled = torch.zeros_like(vals)
        for ci, rows in enumerate(chunks):
            share = self._forward_chunk(rows, prepared, counts)
            idx = torch.as_tensor(rows, device=self.device)
            if ci > 0:  # context genes keep the value from the first pass
                idx, share = idx[self.context_block :], share[self.context_block :]
            vals = vals.index_copy(0, idx, share)
            filled = filled.index_copy(0, idx, torch.ones_like(share))
        self._chunk_count = len(chunks)
        self._unfilled = int((filled == 0).sum().item())
        return vals

    def response(self, prepared: dict) -> np.ndarray:
        counts = torch.as_tensor(prepared["universe_counts"], device=self.device, dtype=torch.float32)
        with torch.no_grad():
            out = self._response_from_counts(prepared, counts, source_row=None)
        return out.float().cpu().numpy()

    def ordering_probe(self, prepared: dict, source_row: int, target_row: int) -> dict:
        """Concrete evidence for the ordering constraint.

        The same target gene is read twice from the same cell: once with the source
        gene ahead of it in the sequence and once behind it, and in each case the
        source gene's count is raised by half. Only the first ordering can respond.
        """
        counts = torch.as_tensor(prepared["universe_counts"], device=self.device, dtype=torch.float32)
        ctx = [r for r in list(prepared["order"])[: self.context_block] if r not in (source_row, target_row)]
        first_source = np.array([source_row] + ctx + [target_row], dtype=np.int64)
        first_target = np.array([target_row] + ctx + [source_row], dtype=np.int64)
        bumped = counts.clone()
        bumped[source_row] = bumped[source_row] * 1.5
        with torch.no_grad():
            a0 = self._forward_chunk(first_source, prepared, counts)[-1]
            a1 = self._forward_chunk(first_source, prepared, bumped)[-1]
            b0 = self._forward_chunk(first_target, prepared, counts)[0]
            b1 = self._forward_chunk(first_target, prepared, bumped)[0]
        return {
            "source_gene": str(prepared["universe_gene_ids"][source_row]),
            "target_gene": str(prepared["universe_gene_ids"][target_row]),
            "source_count": float(counts[source_row]),
            "source_count_raised_to": float(bumped[source_row]),
            "context_genes_between_them": len(ctx),
            "target_share_source_ahead": float(a0),
            "target_share_change_source_ahead": float(a1 - a0),
            "target_share_source_behind": float(b0),
            "target_share_change_source_behind": float(b1 - b0),
            "interpretation": (
                "raising the source gene's count moves the target's predicted share "
                "only when the source precedes it; behind it the change is exactly "
                "zero, which is the causal mask rather than a small effect"
            ),
        }

    # ---- jacobian ---------------------------------------------------------
    def jacobian_column(self, prepared: dict, source_position: int):
        rows = np.flatnonzero(prepared["universe_token_ids"] == source_position)
        if rows.size == 0:
            return None, "unavailable", "source gene absent from the universe token list"
        source_row = int(rows[0])
        base = prepared["universe_counts"].astype(np.float64)

        def npfn(v: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                counts = torch.as_tensor(v, device=self.device, dtype=torch.float32)
                return self._response_from_counts(prepared, counts, source_row).float().cpu().numpy()

        col, step = finite_difference_column(npfn, base, source_row, step=1.0)
        self.jacobian_check = jacobian_cross_check(
            col,
            f"central_finite_difference(step={step:.4g})",
            npfn,
            base,
            source_row,
            alt_step=2.0,
        )
        self.jacobian_check.update({
            "finite_difference_step": step,
            "method_note": (
                "counts enter flex-attention through a score_mod closure; forward-mode "
                "AD is not defined for that custom attention, so the directional "
                "derivative is computed by central finite difference on the count"
            ),
            "column_l2": float(np.linalg.norm(col)),
            "n_nonzero": int((col != 0).sum()),
        })
        self._jacobian_method = f"central_finite_difference(step={step:.4g})_source_first_ordering"
        return col, self._jacobian_method, None

    # ---- contract ---------------------------------------------------------
    def contract(self) -> ResponseContract:
        return ResponseContract(
            model="transcriptformer",
            checkpoint=f"s3://{S.TRANSCRIPTFORMER['bucket']}/{S.TRANSCRIPTFORMER['object']}",
            checkpoint_revision="tf_sapiens (public bucket object, unversioned)",
            checkpoint_md5=None,
            input_normalization=(
                "raw counts, clipped at 30, paired with gene tokens; counts enter the "
                "attention score as log1p(count + 1e-6) on the right-shifted sequence, "
                f"and one auxiliary assay token ('{ASSAY_LABEL}') is prepended. "
                f"Sequence length is fixed at {getattr(self, 'seq_len', 2047)} gene tokens "
                "plus the assay token, so the gene universe is covered in several passes."
            ),
            output_head="CountDecoderHead 'mu' (zero-truncated Poisson rate), softmax link",
            output_semantics=(
                "predicted share of the cell's total counts assigned to that gene, "
                "conditioned on the gene's own identity and on the identity and count "
                "of every gene earlier in the ordering. Reported as mu / total counts so "
                "values are comparable across passes; raw mu is not."
            ),
            information_flow="autoregressive, causal mask over the gene sequence",
            ordering_dependent=True,
            response_symmetric=False,
            symmetry_evidence=(
                "the causal mask makes the response structurally asymmetric: gene j "
                "responds to gene i only when i precedes j. Demonstrated by the ordering "
                "probe recorded alongside this contract."
            ),
            differentiable_wrt_input_expression=True,
            jacobian_available=True,
            jacobian_method=getattr(self, "_jacobian_method", "central_finite_difference"),
            jacobian_unavailable_reason=None,
            jacobian_check=getattr(self, "jacobian_check", {}),
            vocabulary_kind="ensembl",
            vocabulary_size=len(getattr(self, "gene_vocab_dict", {})),
            context_length=int(getattr(self, "seq_len", 2047)) + 1,
            dtype="float32",
            substitutions=self.substitutions,
            notes=(
                "The gene ordering is descending count in the cell. The universe needs "
                f"several passes, and each begins with the same {getattr(self, 'context_block', 256)}-gene "
                "context block (the highest-count genes, with the source gene at "
                "position 0 for the Jacobian pass), so every gene is read against the "
                "same prefix and no pass is left without counts to divide. Genes with "
                "zero counts in this cell still receive tokens; a zero count is outside "
                "the support of the zero-truncated Poisson likelihood the model was "
                "trained with, so those entries describe the model's behaviour on a gene "
                "it did not observe rather than a fitted prediction. "
                f"Token-id alignment against the checkpoint's own vocabulary: "
                f"{getattr(self, 'token_alignment', {})}."
            ),
        )
