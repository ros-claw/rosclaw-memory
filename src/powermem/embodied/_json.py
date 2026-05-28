"""
Fast JSON serialization with optional orjson acceleration.

orjson is ~10x faster than stdlib json for small objects and ~3x for large.
If not installed, transparently falls back to the standard library.
"""

from __future__ import annotations

import json
from typing import Any

try:
    import orjson

    def fast_dumps(obj: Any, sort_keys: bool = False) -> str:
        option = orjson.OPT_SORT_KEYS if sort_keys else 0
        return orjson.dumps(obj, option=option).decode("utf-8")

    def fast_loads(s: str) -> Any:
        return orjson.loads(s)

    HAS_ORJSON = True
except Exception:
    fast_dumps = json.dumps
    fast_loads = json.loads
    HAS_ORJSON = False
