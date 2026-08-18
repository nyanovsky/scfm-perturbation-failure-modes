"""Probe a remote h5ad's schema over HTTP range reads (no full download)."""

import json
import sys

import fsspec
import h5py
import numpy as np

URLS = {
    "hepg2": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667_hepg2_raw_singlecell_01.h5ad",
    "jurkat": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667_jurkat_raw_singlecell_01.h5ad",
}


def describe(h, name, out, depth=0):
    obj = h[name]
    if isinstance(obj, h5py.Group):
        out.append(
            {
                "path": name,
                "kind": "group",
                "attrs": {k: str(v) for k, v in obj.attrs.items()},
                "keys": list(obj.keys())[:80],
            }
        )
    else:
        out.append(
            {
                "path": name,
                "kind": "dataset",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "attrs": {k: str(v) for k, v in obj.attrs.items()},
            }
        )


def main():
    report = {}
    for key, url in URLS.items():
        fs = fsspec.filesystem("http", block_size=2 * 1024 * 1024)
        info = fs.info(url)
        entry = {"url": url, "size_bytes": info.get("size")}
        with fs.open(url, "rb") as fh:
            with h5py.File(fh, "r") as h:
                entry["root_keys"] = list(h.keys())
                entry["root_attrs"] = {k: str(v) for k, v in h.attrs.items()}
                struct = []
                for top in h.keys():
                    describe(h, top, struct)
                    if isinstance(h[top], h5py.Group):
                        for sub in list(h[top].keys()):
                            describe(h, f"{top}/{sub}", struct)
                entry["structure"] = struct

                # obs columns: report dtype + small category vocabularies
                obs = h["obs"]
                cols = {}
                idx_name = obs.attrs.get("_index", "index")
                for c in obs.keys():
                    o = obs[c]
                    d = {"type": "group" if isinstance(o, h5py.Group) else "dataset"}
                    if isinstance(o, h5py.Group):
                        d["keys"] = list(o.keys())
                        if "categories" in o:
                            cats = o["categories"][:]
                            cats = [
                                x.decode() if isinstance(x, bytes) else str(x) for x in cats
                            ]
                            d["n_categories"] = len(cats)
                            d["categories_head"] = cats[:40]
                            d["codes_shape"] = list(o["codes"].shape)
                    else:
                        d["dtype"] = str(o.dtype)
                        d["shape"] = list(o.shape)
                        try:
                            head = o[:8]
                            d["head"] = [
                                x.decode() if isinstance(x, bytes) else (
                                    float(x) if np.issubdtype(np.asarray(x).dtype, np.floating) else str(x)
                                )
                                for x in head
                            ]
                        except Exception as exc:  # noqa: BLE001
                            d["head_error"] = str(exc)
                    cols[c] = d
                entry["obs_index_name"] = str(idx_name)
                entry["obs_columns"] = cols

                var = h["var"]
                entry["var_index_name"] = str(var.attrs.get("_index", "index"))
                vcols = {}
                for c in var.keys():
                    o = var[c]
                    if isinstance(o, h5py.Group):
                        vcols[c] = {"type": "group", "keys": list(o.keys())}
                    else:
                        head = o[:6]
                        vcols[c] = {
                            "dtype": str(o.dtype),
                            "shape": list(o.shape),
                            "head": [
                                x.decode() if isinstance(x, bytes) else str(x) for x in head
                            ],
                        }
                entry["var_columns"] = vcols
        report[key] = entry
    print(json.dumps(report, indent=1)[:60000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
