"""prediction/preprocess.py: 16-bit input is scaled by percentiles, not divided by 255."""
import numpy as np

from prediction.preprocess import normalize_intensity


def _camera_frame(dtype, offset, peak):
    rng = np.random.default_rng(0)
    f = offset + rng.normal(0, 20, (64, 64))
    f[30:33, 30:33] += peak
    return np.clip(f, 0, np.iinfo(dtype).max).astype(dtype)


def test_uint16_maps_to_unit_range():
    f = _camera_frame(np.uint16, 1000, 3000)
    n = normalize_intensity(f)
    assert n.dtype == np.float32 and n.min() >= 0 and n.max() <= 1
    assert np.median(n) == 0.0                      # background at 0, like the simulated training frames
    assert n[31, 31] > 0.9                          # particle near 1
    assert (f / 255.0).max() > 10                   # the old /255 scaling left values far above 1


def test_uint8_and_unit_floats_keep_their_scale():
    f8 = _camera_frame(np.uint8, 20, 200)
    np.testing.assert_allclose(normalize_intensity(f8), f8 / 255.0, rtol=1e-6)
    u = np.random.default_rng(1).random((8, 8)).astype(np.float32)
    np.testing.assert_array_equal(normalize_intensity(u), u)


def test_stack_uses_one_scale_and_flat_input_is_zero():
    s = np.stack([_camera_frame(np.uint16, 1000, 3000), _camera_frame(np.uint16, 1000, 1500)])
    n = normalize_intensity(s)
    assert n[0, 31, 31] > n[1, 31, 31]               # relative brightness between frames is kept
    assert np.all(normalize_intensity(np.full((4, 4), 7, np.uint16)) == 0)


def test_predictor_preprocessing_uses_it():
    import pytest
    pytest.importorskip('torch')
    from prediction.inference import ParticlePredictor
    f = _camera_frame(np.uint16, 1000, 3000)
    out = ParticlePredictor._preprocess_frame(object(), f)
    assert out.max() <= 1.0 and np.median(out) == 0.0
