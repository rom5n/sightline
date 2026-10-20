"""msgpack encoding with numpy arrays as typed binary blobs."""
from __future__ import annotations

import math
from typing import Any

import msgpack
import numpy as np

_CODES = {np.dtype("float32"): "f4", np.dtype("int32"): "i4", np.dtype("uint8"): "u1"}


def to_wire(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        if o.dtype == np.bool_:
            o = o.astype(np.uint8)
        elif o.dtype.kind == "f" and o.dtype != np.float32:
            o = o.astype(np.float32)
        elif o.dtype.kind in "iu" and o.dtype not in _CODES:
            o = o.astype(np.int32)
        a = np.ascontiguousarray(o)
        return {"$nd": _CODES[a.dtype], "shape": list(a.shape), "data": a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes()}
    if isinstance(o, dict):
        return {str(k): to_wire(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_wire(v) for v in o]
    if isinstance(o, np.generic):
        return to_wire(o.item())
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def pack(o: Any) -> bytes:
    return bytes(msgpack.packb(to_wire(o), use_bin_type=True))


def unpack(b: bytes) -> Any:
    return msgpack.unpackb(b, raw=False)
