"""Build one arm's model and time a real training step and a real inference pass.

Throwaway: it exists to size the batch, the sequence length and the epoch cost
before ten jobs are submitted on that estimate.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scfm_ft import config as C  # noqa: E402
from scfm_ft import paths  # noqa: E402
from scfm_ft.scgpt_pert import checkpoint, model as M  # noqa: E402


def main() -> int:
    line = sys.argv[1] if len(sys.argv) > 1 else "hepg2"
    cache = paths.checkpoint_cache()
    t0 = time.time()
    meta = checkpoint.fetch(cache)
    print(f"checkpoint fetched in {time.time() - t0:.0f}s: "
          f"{ {k: v['size_bytes'] for k, v in meta.items() if isinstance(v, dict)} }", flush=True)

    readout = pd.read_parquet(paths.readout_table(line))
    pos = readout["scgpt_position"].to_numpy(dtype=np.int64)
    n_genes = pos.size
    print(f"[{line}] {n_genes} read-out genes; scgpt_position min={pos.min()} max={pos.max()} "
          f"unresolved={(pos < 0).sum()}", flush=True)

    R = C.RECIPE
    for arm in (C.ARMS_BY_NAME["own_trainable"], C.ARMS_BY_NAME["esm2_frozen"]):
        table = None
        if arm.table != "scgpt":
            table = np.random.default_rng(0).standard_normal((n_genes, 2560)).astype(np.float32)
        t0 = time.time()
        model, report = M.build_arm_model(arm, cache, table, pos, device="cuda")
        print(f"\n=== arm {arm.name} built in {time.time() - t0:.0f}s ===", flush=True)
        for k in ("pretrained_tensors_loaded", "pretrained_tensors_shape_mismatch",
                  "fused_qkv_tensors_remapped", "trainable_parameters", "total_parameters",
                  "table_dim", "table_trainable", "table_projected"):
            print(f"  {k}: {report[k]}", flush=True)
        print(f"  table encoder check: {M.verify_table_encoder(model, 64)}", flush=True)

        opt = torch.optim.Adam(model.parameters(), lr=R.lr)
        scaler = torch.cuda.amp.GradScaler(enabled=R.amp)
        crit = torch.nn.functional.mse_loss
        B, L = R.batch_size, R.max_seq_len
        dev = "cuda"
        src = torch.randint(0, n_genes, (B, L), device=dev)
        vals = torch.rand(B, L, device=dev) * 4.0
        flags = torch.zeros(B, L, dtype=torch.long, device=dev)
        flags[:, 0] = 1
        tgt = torch.rand(B, L, device=dev) * 4.0
        pad = torch.zeros(B, L, dtype=torch.bool, device=dev)

        def step():
            with torch.cuda.amp.autocast(enabled=R.amp):
                out = model(src, vals, flags, src_key_padding_mask=pad)["mlm_output"]
                loss = crit(out, tgt)
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), R.grad_clip, error_if_nonfinite=False)
            scaler.step(opt)
            scaler.update()
            return float(loss)

        model.train()
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        t0 = time.time()
        n = 10
        for _ in range(n):
            step()
        torch.cuda.synchronize()
        per_batch = (time.time() - t0) / n
        print(f"  train: {per_batch * 1000:.0f} ms/batch at B={B} L={L}; "
              f"{B / per_batch:.0f} cells/s", flush=True)
        print(f"  peak memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)

        # inference over the whole gene set, which is what extraction does
        model.eval()
        full = torch.arange(n_genes, device=dev).unsqueeze(0)
        for ib in (8, 16):
            s = full.repeat(ib, 1)
            v = torch.rand(ib, n_genes, device=dev) * 4.0
            f = torch.zeros(ib, n_genes, dtype=torch.long, device=dev)
            p = torch.zeros(ib, n_genes, dtype=torch.bool, device=dev)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=R.amp):
                for _ in range(2):
                    model(s, v, f, src_key_padding_mask=p)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(5):
                    model(s, v, f, src_key_padding_mask=p)
                torch.cuda.synchronize()
            dt = (time.time() - t0) / 5
            print(f"  infer B={ib} L={n_genes}: {dt * 1000:.0f} ms "
                  f"({ib / dt:.0f} queries/s)", flush=True)
        del model, opt
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
