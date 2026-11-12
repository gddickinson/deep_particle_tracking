"""One SNR definition (simulator/movie.py noise_levels): peak signal above background / noise SD at background.

The measured SNR (PEAK_SIGNAL over the SD of the unclipped background) matches the setting for every noise model,
in the Simulation tab pipeline and in SimulatedParticleDataset.
"""
import numpy as np
import pytest

from simulator.movie import PEAK_SIGNAL, noise_levels, simulate_movie


def _measured(raw):
    return PEAK_SIGNAL / np.std(raw)


@pytest.mark.parametrize('noise', ['Poisson', 'Gaussian', 'Poisson + Gaussian'])
@pytest.mark.parametrize('snr', [1.0, 3.0, 10.0])
def test_movie_snr_matches_setting(noise, snr):
    r = simulate_movie(seed=0, width=128, height=128, num_frames=3, num_particles=0, noise=noise, snr=snr)
    assert _measured(r['raw_frames']) == pytest.approx(snr, rel=0.03)


def test_read_noise_ceiling():
    # 3 counts of read noise at 100 photons per unit: SNR cannot exceed 100 / 3
    bg, _ = noise_levels('Poisson + Gaussian', 50.0, read_noise=3.0)
    assert bg == 0.0
    r = simulate_movie(seed=0, width=128, height=128, num_frames=2, num_particles=0, noise='Poisson + Gaussian',
                       snr=50.0, read_noise=3.0)
    assert _measured(r['raw_frames']) == pytest.approx(100 / 3, rel=0.03)


@pytest.mark.parametrize('noise', ['poisson', 'gaussian', 'poisson_gaussian'])
@pytest.mark.parametrize('snr', [1.0, 3.0, 10.0])
def test_training_dataset_uses_same_definition(noise, snr, monkeypatch):
    torch = pytest.importorskip('torch')  # noqa: F841
    import training.data_loader as dl
    seen = []
    real = dl.add_noise

    def spy(psf, name, s, read_noise=0.0):
        out, raw = real(np.zeros_like(psf), name, s, read_noise)
        seen.append((s, raw))
        return out, raw
    monkeypatch.setattr(dl, 'add_noise', spy)
    ds = dl.SimulatedParticleDataset(num_samples=1, frame_size=(96, 96), frame_sequence_length=2,
                                     particles_per_frame=(1, 2), noise_model=noise, snr_range=(snr, snr))
    ds[0]
    for s, raw in seen:
        assert s == snr and _measured(raw) == pytest.approx(snr, rel=0.04)
