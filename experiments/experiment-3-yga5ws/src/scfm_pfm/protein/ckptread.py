"""Read a checkpoint's structure without importing the model's package.

A ``.pt``/``.ckpt`` is a zip whose ``data.pkl`` describes the object tree and
whose tensor payloads are separate members. Unpickling it normally needs every
class it mentions to be importable, which for a Lightning checkpoint means the
training package and its whole dependency tree. The audit does not need the
objects, only the shapes: which tensors exist, how big they are, and whether the
optimizer ever carried a state for them.

So the tree is read with an unpickler that never loads a tensor payload -- it
returns a descriptor in the tensor's place -- and substitutes a recording stub
for any class it cannot import. ``torch.load`` is tried first and this is the
fallback, so the result says which reader produced it.
"""

from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path
from typing import Any

MAX_SEQ = 8  # how much of a long list to keep in the recorded tree


def _shorten(obj: Any, depth: int = 0) -> Any:
    """A JSON-able echo of ``obj`` with long sequences summarised."""
    if depth > 6:
        return "<deep>"
    if isinstance(obj, dict):
        if len(obj) > 64:
            keys = list(obj)[:MAX_SEQ]
            return {"__dict_of__": len(obj), "keys_head": [str(k) for k in keys]}
        return {str(k): _shorten(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        if len(seq) > MAX_SEQ:
            return {
                "__seq_of__": len(seq),
                "head": [_shorten(v, depth + 1) for v in seq[:MAX_SEQ]],
            }
        return [_shorten(v, depth + 1) for v in seq]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, bytes):
        return f"<bytes:{len(obj)}>"
    return str(obj)[:200]


def shorten(obj: Any) -> Any:
    """Public name for the JSON-able echo used by the audit."""
    return _shorten(obj)


class TensorInfo(dict):
    """What a tensor is, without its data."""

    def __init__(self, shape, dtype, requires_grad=False, storage_key=None):
        super().__init__(
            __tensor__=True,
            shape=[int(s) for s in shape],
            dtype=str(dtype),
            numel=int(_prod(shape)),
            requires_grad=bool(requires_grad),
            storage_key=storage_key,
        )


def _prod(shape) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


class _Storage:
    def __init__(self, dtype: str, key: str, numel: int):
        self.dtype, self.key, self.numel = dtype, key, numel


def _make_stub(module: str, name: str):
    tag = f"{module}.{name}"

    class _Stub(dict):
        __qualname__ = f"Stub[{tag}]"

        def __init__(self, *args, **kwargs):
            super().__init__()
            self["__stub__"] = tag
            if args:
                self["__args__"] = _shorten(args)
            if kwargs:
                self["__kwargs__"] = _shorten(kwargs)

        def __setstate__(self, state):
            self["__state__"] = _shorten(state)

        def __reduce__(self):  # pragma: no cover - never re-pickled
            return (dict, (dict(self),))

    return _Stub


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, *rest):
    return TensorInfo(size, getattr(storage, "dtype", "?"), requires_grad,
                      getattr(storage, "key", None))


def _rebuild_tensor(storage, storage_offset, size, stride):
    return TensorInfo(size, getattr(storage, "dtype", "?"), False,
                      getattr(storage, "key", None))


def _rebuild_parameter(data, requires_grad, backward_hooks):
    if isinstance(data, TensorInfo):
        data["requires_grad"] = bool(requires_grad)
        data["is_parameter"] = True
    return data


class _StructureUnpickler(pickle.Unpickler):
    """Returns descriptors for tensors and stubs for unimportable classes."""

    def __init__(self, fh):
        super().__init__(fh, encoding="latin1")
        self.stubbed: set[str] = set()

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        if isinstance(pid, (tuple, list)) and pid and pid[0] == "storage":
            dtype = getattr(pid[1], "__name__", str(pid[1]))
            return _Storage(dtype, str(pid[2]), int(pid[4]) if len(pid) > 4 else -1)
        return _Storage("?", str(pid), -1)

    def find_class(self, module, name):
        key = f"{module}.{name}"
        if key == "torch._utils._rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if key == "torch._utils._rebuild_tensor":
            return _rebuild_tensor
        if key == "torch._utils._rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name.endswith(("float32", "float16", "bfloat16", "float64",
                                                "int64", "int32", "uint8", "bool", "Storage")):
            return type(name, (), {"__name__": name})
        try:
            return super().find_class(module, name)
        except Exception:  # noqa: BLE001 - the point is to survive it
            self.stubbed.add(key)
            return _make_stub(module, name)


def read_structure(path: Path) -> dict:
    """The checkpoint's object tree, tensors replaced by shape descriptors."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        pkls = [n for n in names if n.endswith("data.pkl")]
        if not pkls:
            raise RuntimeError(f"{path}: no data.pkl in the archive")
        payload = z.read(pkls[0])
    up = _StructureUnpickler(io.BytesIO(payload))
    tree = up.load()
    return {"tree": tree, "stubbed_classes": sorted(up.stubbed), "reader": "structure"}


def tensor_entries(node: Any, prefix: str = "") -> list[tuple[str, dict]]:
    """Every tensor descriptor in the tree, with a dotted path."""
    out: list[tuple[str, dict]] = []
    if isinstance(node, dict):
        if node.get("__tensor__"):
            out.append((prefix, node))
            return out
        for k, v in node.items():
            out += tensor_entries(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            out += tensor_entries(v, f"{prefix}[{i}]")
    return out


def torch_state_dict(path: Path) -> tuple[dict, str]:
    """``torch.load`` first, the structure reader as the fallback.

    Returns the mapping and which reader produced it; with the fallback the
    values are shape descriptors rather than tensors.
    """
    import torch  # noqa: PLC0415 - the audit runs in a torch image, the reader does not need it

    for weights_only in (True, False):
        try:
            obj = torch.load(path, map_location="cpu", weights_only=weights_only)
            return obj, f"torch.load(weights_only={weights_only})"
        except Exception:  # noqa: BLE001 - fall through to the tolerant reader
            continue
    return read_structure(path)["tree"], "structure"
