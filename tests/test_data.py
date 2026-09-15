import json

import numpy as np
import pytest
import torch

from mgn.config import ModelConfig
from mgn.data import Vocabulary, SPECIAL, FeatureDataset, SyntheticDataset, collate_fn


def config():
    return ModelConfig(visual_dim=3, question_dim=4, audio_dim=5, hidden_dim=8, encoder_heads=2)


def test_vocab_rejects_validation_and_encodes_unknown():
    with pytest.raises(ValueError, match='training-only'):
        Vocabulary.build([dict(split='val', answer_tokens=['泄漏'])])
    v = Vocabulary.build([dict(split='train', answer_tokens=['答案'])])
    assert v.encode(['未知', '答案'], 1) == [3]
    assert v.decode([1, 4, 2, 4]) == ['答案']


def test_variable_padding_and_synthetic_noise():
    ds = SyntheticDataset(config())
    other = SyntheticDataset(config(), seed=99)
    batch = collate_fn([ds[0], ds[1]])
    assert batch['visual_mask'].tolist() == [[True, True, False], [True, True, True]]
    assert torch.count_nonzero(batch['visual'][0, 2]) == 0
    assert not torch.equal(ds[0]['visual'], other[0]['visual'])
    assert torch.equal(ds[0]['answer_target'], other[0]['answer_target'])


def test_feature_validation(tmp_path):
    cfg = config()
    row = dict(sample_id='s', video_id='v', split='train', answer_tokens=['x'], features={})
    for name in ('visual', 'question', 'audio'):
        np.save(tmp_path / f'{name}.npy', np.ones((2, getattr(cfg, name + '_dim')), dtype='float32'))
        row['features'][name] = dict(path=f'{name}.npy', metadata=dict(extractor='fixture', checkpoint='none', output_layer='raw', preprocessing='none'))
    manifest = tmp_path / 'train.jsonl'
    manifest.write_text(json.dumps(row))
    ds = FeatureDataset(manifest, Vocabulary(SPECIAL + ['x']), cfg, expected_split='train')
    assert ds[0]['answer_target'].tolist() == [4, 2]
    np.save(tmp_path / 'audio.npy', np.full((2, 5), np.nan, dtype='float32'))
    with pytest.raises(ValueError, match='finite'):
        ds[0]
    with pytest.raises(ValueError, match='split mismatch'):
        FeatureDataset(manifest, ds.vocab, cfg, expected_split='val')


def test_rejects_mixed_feature_provenance(tmp_path):
    rows = []
    for i in range(2):
        rows.append(dict(sample_id=str(i), video_id=str(i), split='train', features={
            m: dict(metadata=dict(extractor='same', checkpoint=str(i) if m == 'audio' else 'same', output_layer='same', preprocessing='same'))
            for m in ('visual', 'question', 'audio')}))
    manifest = tmp_path / 'train.jsonl'
    manifest.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match='inconsistent feature provenance'):
        FeatureDataset(manifest, Vocabulary(SPECIAL), config())


def test_overlong_features_and_embedded_metadata_mismatch(tmp_path):
    cfg = config()
    metadata = dict(extractor='fixture', checkpoint='v1', output_layer='raw', preprocessing='none')
    row = dict(sample_id='s', video_id='v', split='train', answer_tokens=['x'], features={})
    for m in ('visual', 'question', 'audio'):
        np.savez(tmp_path / f'{m}.npz', features=np.ones((2, getattr(cfg, m + '_dim')), dtype='float32'), metadata=json.dumps(metadata))
        row['features'][m] = dict(path=f'{m}.npz', metadata=metadata)
    manifest = tmp_path / 'data.jsonl'
    manifest.write_text(json.dumps(row))
    ds = FeatureDataset(manifest, Vocabulary(SPECIAL + ['x']), cfg)
    ds[0]
    np.savez(tmp_path / 'visual.npz', features=np.ones((cfg.max_visual_len + 1, cfg.visual_dim), dtype='float32'), metadata=json.dumps(metadata))
    with pytest.raises(ValueError, match='length exceeds'):
        ds[0]
    np.savez(tmp_path / 'visual.npz', features=np.ones((2, cfg.visual_dim), dtype='float32'), metadata=json.dumps(dict(metadata, checkpoint='v2')))
    with pytest.raises(ValueError, match='embedded cache metadata'):
        ds[0]
