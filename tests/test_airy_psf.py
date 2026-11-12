"""AiryDiskPSF: airy_radius is documented as the radius of the first zero (in pixels)."""
import numpy as np
from scipy import special

from simulator.psf_models import AiryDiskPSF


def test_first_zero_at_airy_radius():
    radius = 4.0
    img = AiryDiskPSF(image_size=(41, 81), airy_radius=radius).generate([(20.0, 20.0)])
    row = img[20, 20:]           # profile along +x, r = 0 .. 60 px
    assert row[0] == 1.0
    assert row[4] < 1e-12        # r = airy_radius is the first dark ring
    first_min = int(np.argmin(row[:8]))
    assert first_min == 4


def test_profile_is_airy_pattern():
    radius = 3.0
    img = AiryDiskPSF(image_size=(1, 8), airy_radius=radius).generate([(0.0, 0.0)])
    r = np.arange(1, 8)
    v = special.jn_zeros(1, 1)[0] * r / radius
    np.testing.assert_allclose(img[0, 1:], (2 * special.j1(v) / v) ** 2, rtol=1e-5, atol=1e-8)
