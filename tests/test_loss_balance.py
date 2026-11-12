"""Balance of the training loss: with the default settings the tracking (embedding) term starts within an order of
magnitude of the localisation term, and its weight is ramped up over a warm-up. (Before: the summed squared
embedding distance made the tracking term 15-60x the localisation loss on an untrained model.)"""
import pytest

torch = pytest.importorskip('torch')

from models.losses import (CombinedLoss, DEFAULT_LAMBDA_TRACK, DEFAULT_TRACK_WARMUP_STEPS,  # noqa: E402
                           EmbeddingLoss)
from models.network import create_model  # noqa: E402
from training.data_loader import SimulatedParticleDataset  # noqa: E402

T = 5
# Default embedding dimensions and window length; narrower/shallower than the defaults so the test stays fast
CONFIGS = {
    'simple': dict(base_filters=8, depth=3, num_frames=T),
    'dual': dict(base_filters=8, depth=3, num_frames=T, upsample_factor=2),
    'attentive': dict(base_filters=8, depth=3, num_frames=T, heads=2, upsample_factor=2),
    # tiny configurations of tests/test_embedding_training.py, where summed distances gave 34x / 45x
    'simple_tiny': dict(base_filters=4, depth=2, num_frames=3, embedding_dim=8),
    'attentive_tiny': dict(base_filters=8, depth=2, num_frames=3, heads=2, upsample_factor=2),
}


def _batch(n=2, size=32, particles=(5, 30), frames=T):
    # particles_per_frame: default range of the GUI's Training tab
    ds = SimulatedParticleDataset(num_samples=n, frame_size=(size, size), frame_sequence_length=frames,
                                  particles_per_frame=particles)
    items = [ds[i] for i in range(n)]
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


@pytest.mark.parametrize('particles', [(5, 30), (4, 8)], ids=['gui_density', 'sparse'])
@pytest.mark.parametrize('model_type', list(CONFIGS))
def test_terms_within_an_order_of_magnitude_at_initialisation(model_type, particles):
    # Sparse frames make the localisation loss small; with summed distances the tracking term was then 15-44x it
    torch.manual_seed(0)
    cfg = CONFIGS[model_type]
    batch = _batch(frames=cfg['num_frames'], particles=particles)
    loss_fn = CombinedLoss()
    losses = loss_fn(create_model(model_type.split('_')[0], **cfg)(batch['frames']), batch)
    loc = loss_fn.lambda_loc * float(losses['loc_loss'])
    track = DEFAULT_LAMBDA_TRACK * float(losses['track_loss'])     # full weight, i.e. after the warm-up
    assert 0.1 < track / loc < 10, (model_type, loc, track)


def test_embedding_loss_scale_does_not_depend_on_embedding_dim():
    torch.manual_seed(0)
    ids = torch.zeros(1, 2, 1, 6, 6, dtype=torch.long)
    ids[0, :, 0, 1, 1], ids[0, :, 0, 4, 4] = 1, 2
    small, big = torch.randn(1, 2, 4, 6, 6), torch.randn(1, 2, 64, 6, 6)
    raw = EmbeddingLoss(normalize_by_dim=False)
    norm = EmbeddingLoss()
    assert float(raw(big, ids)) / float(raw(small, ids)) > 5      # summed distance grows with the dimension
    assert 0.3 < float(norm(big, ids)) / float(norm(small, ids)) < 3


def test_warmup_ramps_the_tracking_weight_in_training_only():
    torch.manual_seed(0)
    batch = _batch(n=1)
    out = create_model('simple', **CONFIGS['simple'])(batch['frames'])
    loss_fn = CombinedLoss(lambda_track=0.5, track_warmup_steps=4)
    assert DEFAULT_TRACK_WARMUP_STEPS > 0 and CombinedLoss().track_warmup_steps == DEFAULT_TRACK_WARMUP_STEPS

    def check(expected_weight):
        losses = loss_fn(out, batch)
        assert float(losses['track_weight']) == pytest.approx(expected_weight)
        assert float(losses['total']) == pytest.approx(
            float(losses['loc_loss']) + expected_weight * float(losses['track_loss']), rel=1e-6)

    check(0.0)                      # step 0: localisation only
    loss_fn.step(); loss_fn.step()
    check(0.25)                     # halfway
    loss_fn.eval()
    check(0.5)                      # validation always uses the full weight
    loss_fn.train()
    for _ in range(5):
        loss_fn.step()
    check(0.5)                      # capped after the warm-up
    assert CombinedLoss(track_warmup_steps=0).track_weight() == DEFAULT_LAMBDA_TRACK


def test_trainer_applies_the_warmup_schedule(tmp_path):
    from torch.utils.data import DataLoader
    from training.trainer import Trainer
    ds = SimulatedParticleDataset(num_samples=4, frame_size=(32, 32), frame_sequence_length=3,
                                  particles_per_frame=(3, 6))
    trainer = Trainer(model_type='simple', model_config=dict(base_filters=4, depth=2, num_frames=3),
                      track_warmup_steps=10, scheduler_type=None, checkpoint_dir=str(tmp_path))
    trainer.setup()
    assert trainer.criterion.track_warmup_steps == 10 and trainer.criterion.lambda_track == DEFAULT_LAMBDA_TRACK
    trainer._train_epoch(DataLoader(ds, batch_size=2), 0)
    assert trainer.criterion.steps_done == 2                       # one schedule step per optimiser step
    assert trainer.criterion.track_weight() == pytest.approx(0.2 * DEFAULT_LAMBDA_TRACK)
    trainer._validate(DataLoader(ds, batch_size=2))
    assert not trainer.criterion.training and trainer.criterion.track_weight() == DEFAULT_LAMBDA_TRACK
    # the warm-up position survives a checkpoint round trip
    trainer._save_checkpoint('latest')
    resumed = Trainer(model_type='simple', model_config=dict(base_filters=4, depth=2, num_frames=3),
                      track_warmup_steps=10, scheduler_type=None, checkpoint_dir=str(tmp_path))
    resumed.setup()
    resumed._load_checkpoint('latest')
    assert resumed.criterion.steps_done == 2
