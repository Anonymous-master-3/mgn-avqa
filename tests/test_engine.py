import torch

from mgn.config import ModelConfig
from mgn.data import SyntheticDataset
from mgn.engine import Trainer, seed_everything, load_checkpoint


class TinyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.3)
        self.linear = torch.nn.Linear(3, 8)

    def forward(self, batch, compute_contrastive=True):
        logits = self.linear(self.dropout(batch['visual'].mean(1)))
        return dict(logits=logits[:, None].expand(-1, batch['answer_target'].shape[1], -1), contrastive_loss=logits.sum() * 0)


def test_epoch_boundary_resume_matches_uninterrupted(tmp_path):
    torch.set_num_threads(1)
    mc = ModelConfig(visual_dim=3, question_dim=4, audio_dim=5, hidden_dim=8, encoder_heads=2)
    ds = SyntheticDataset(mc, size=8)
    val = SyntheticDataset(mc, seed=99)
    config = dict(model=mc.to_dict(), training=dict(epochs=2, lr=.003, batch_size=4, warmup_steps=3, seed=17, early_stopping_patience=10))
    seed_everything(17)
    full = Trainer(TinyModel(), config, ds.vocab, tmp_path / 'full')
    full.fit(ds, val)
    seed_everything(17)
    first_config = dict(config, training=dict(config['training'], epochs=1))
    first = Trainer(TinyModel(), first_config, ds.vocab, tmp_path / 'split')
    first.fit(ds, val)
    resumed = Trainer(TinyModel(), config, ds.vocab, tmp_path / 'split')
    resumed.resume(tmp_path / 'split' / 'last.pt')
    resumed.fit(ds, val)
    assert resumed.step == full.step
    assert resumed.best == full.best
    for name, value in full.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
    assert load_checkpoint(tmp_path / 'split' / 'last.pt')['epoch'] == 2


def test_nonfinite_gradient_never_updates_parameters(tmp_path):
    import pytest
    from mgn.data import collate_fn
    mc = ModelConfig(visual_dim=3, question_dim=4, audio_dim=5, hidden_dim=8, encoder_heads=2)
    ds = SyntheticDataset(mc)
    trainer = Trainer(TinyModel(), dict(model=mc.to_dict(), training={}), ds.vocab, tmp_path)
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    trainer.model.linear.weight.register_hook(lambda grad: torch.full_like(grad, float('inf')))
    with pytest.raises(RuntimeError, match='non-finite'):
        trainer.update(collate_fn([ds[0], ds[1]]))
    assert trainer.step == 0
    for key, value in before.items():
        torch.testing.assert_close(value, trainer.model.state_dict()[key], rtol=0, atol=0)


def test_resume_rejects_manifest_content_changed_in_place(tmp_path):
    import pytest
    mc = ModelConfig(visual_dim=3, question_dim=4, audio_dim=5, hidden_dim=8, encoder_heads=2)
    ds = SyntheticDataset(mc)
    manifest = tmp_path / 'train.jsonl'
    manifest.write_text('original')
    config = dict(model=mc.to_dict(), data=dict(train_manifest=str(manifest)))
    trainer = Trainer(TinyModel(), config, ds.vocab, tmp_path)
    trainer.save(tmp_path / 'checkpoint.pt')
    manifest.write_text('changed')
    with pytest.raises(ValueError, match='content hash changed'):
        trainer.resume(tmp_path / 'checkpoint.pt')
