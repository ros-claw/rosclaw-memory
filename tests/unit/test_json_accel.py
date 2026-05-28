"""
Tests for the optional orjson acceleration wrapper.
"""

from __future__ import annotations

import pytest

from powermem.embodied._json import fast_dumps, fast_loads


class TestFastJson:
    def test_dumps_loads_roundtrip(self):
        obj = {"key": [1, 2, 3], "nested": {"a": True}}
        s = fast_dumps(obj)
        assert isinstance(s, str)
        assert fast_loads(s) == obj

    def test_dumps_sort_keys(self):
        obj = {"z": 1, "a": 2, "m": 3}
        s = fast_dumps(obj, sort_keys=True)
        assert s == '{"a":2,"m":3,"z":1}' or s == '{"a": 2, "m": 3, "z": 1}'

    def test_loads_empty_string(self):
        with pytest.raises(Exception):
            fast_loads("")

    def test_dumps_primitives(self):
        assert fast_loads(fast_dumps([1, 2, 3])) == [1, 2, 3]
        assert fast_loads(fast_dumps("hello")) == "hello"
        assert fast_loads(fast_dumps(42)) == 42
        assert fast_loads(fast_dumps(None)) is None
