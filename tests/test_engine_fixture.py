"""tests/fixtures/engine_fixture.json (used by the JS engine test) is current with the Python code.

Numbers are compared with a tolerance (libm differs in the last digit between macOS and Linux).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_engine_fixture  # noqa: E402


def _cmp(a, b, path="fixture"):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _cmp(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        if a and all(isinstance(x, (int, float)) or x is None for x in a + b):
            x = np.array([np.nan if v is None else v for v in a], float)
            y = np.array([np.nan if v is None else v for v in b], float)
            np.testing.assert_allclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=path)
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _cmp(x, y, f"{path}[{i}]")
    else:
        assert a == b, path


def test_fixture_is_current():
    stored = json.loads(make_engine_fixture.OUT.read_text())
    fresh = json.loads(json.dumps(make_engine_fixture.build(), separators=(",", ":")))
    _cmp(fresh, stored)
