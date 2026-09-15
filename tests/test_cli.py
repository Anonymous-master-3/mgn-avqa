
import json

import torch
import yaml

from mgn.cli import main
from mgn.config import ModelConfig
from mgn.data import SyntheticDataset
from mgn.engine import load_checkpoint
from mgn.preprocessing import write_feature_cache


def test_cached_features_to_train_and_evaluation(tmp_path):
    torch.set_num_threads(1)
    model_config = ModelConfig(
        visual_dim=6, question_dim=5, audio_dim=4, hidden_dim=8,
        max_visual_len=4, max_question_len=4, max_audio_len=4, max_answer_len=2,
        graph_layers=1, encoder_heads=2, encoder_conv_layers=0, encoder_kernel_size=3,
        dropout=0., tpa_rank=2, tpa_feature_rank=2, bilinear_rank=2,
    )
    for split, seed in [('train', 123), ('val', 456)]:
        dataset = SyntheticDataset(model_config, seed=seed)
        rows = []
        for index in range(len(dataset)):
            sample = dataset[index]
            references = {
                modality: write_feature_cache(
                    tmp_path / f'{split}-{index}-{modality}.npz', sample[modality],
                    dict(extractor='test-latent-prototype', checkpoint='prototype-1729',
                         output_layer='independent-feature', preprocessing='gaussian-noise'),
                )
                for modality in ('visual', 'question', 'audio')
            }
            rows.append(dict(sample_id=sample['sample_id'], video_id=f'{split}-{index}',
                             split=split, answer_tokens=sample['reference_tokens'], features=references))
        (tmp_path / f'{split}.jsonl').write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    vocab_path = tmp_path / 'vocab.json'
    config = dict(model=model_config.to_dict(), training=dict(
        epochs=1, batch_size=4, seed=42, lr=.005, warmup_steps=0,
        contrastive_weight=.01, num_workers=0, device='cpu', cpu_threads=1,
    ), data=dict(train_manifest=str(tmp_path / 'train.jsonl'),
                 val_manifest=str(tmp_path / 'val.jsonl'), vocab=str(vocab_path)))
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    main(['build-vocab', '--manifest', str(tmp_path / 'train.jsonl'), '--output', str(vocab_path)])
    main(['check-data', '--config', str(config_path), '--manifest', str(tmp_path / 'train.jsonl'),
          '--vocab', str(vocab_path)])
    output = tmp_path / 'run'
    main(['train', '--config', str(config_path), '--output', str(output)])
    checkpoint = load_checkpoint(output / 'best.pt')
    assert checkpoint['epoch'] == 1 and checkpoint['step'] == 1
    assert checkpoint['vocabulary']['source_split'] == 'train'
    history = json.loads((output / 'history.jsonl').read_text())
    assert all(key in history for key in ('train_loss', 'train_nll', 'train_contrastive', 'val_nll', 'lr'))
    assert json.loads((output / 'split_audit.json').read_text())['overlapping_video_ids'] == []
    evaluation_path = tmp_path / 'evaluation.json'
    main(['evaluate', '--checkpoint', str(output / 'best.pt'), '--manifest', str(tmp_path / 'val.jsonl'),
          '--output', str(evaluation_path)])
    evaluation = json.loads(evaluation_path.read_text())
    assert len(evaluation['predictions']) == 4
    assert evaluation['metrics']['sample_count'] == 4
    assert evaluation['metrics']['wups_0'] is None
    assert evaluation['metrics']['status']['wups']['status'] == 'unavailable'
    assert {tuple(row['reference']) for row in evaluation['predictions']} == {
        tuple(row['answer_tokens']) for row in rows
    }
