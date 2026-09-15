
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mgn.config import ModelConfig, load_config
from mgn.data import Vocabulary, FeatureDataset, SyntheticDataset, collate_fn, read_manifest, assert_disjoint
from mgn.engine import Trainer, seed_everything, load_checkpoint, to_device
from mgn.models import MGN


def dump(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


@torch.no_grad()
def predict(model, dataset, device='cpu', batch_size=64):
    model.eval()
    results = []
    for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn):
        generated = model.generate(to_device(batch, device))
        for sample_id, ids, reference in zip(batch['sample_ids'], generated.cpu(), batch['reference_tokens']):
            results.append(dict(sample_id=sample_id, prediction=dataset.vocab.decode(ids), reference=reference))
    return results


def smoke(args):
    config = load_config(args.config)
    opts = config.setdefault('training', {})
    seed_everything(opts.get('seed', 42))
    torch.set_num_threads(opts.get('cpu_threads', 1))
    model_config = ModelConfig.from_dict(config['model'])
    size, max_steps = opts.get('smoke_samples', 4), opts.get('smoke_steps', 300)
    if size < 4 or size % 4 or not 1 <= max_steps <= 300:
        raise ValueError('smoke_samples must be a positive multiple of 4; smoke_steps must be 1..300')
    dataset = SyntheticDataset(model_config, size=size, seed=42)
    val_dataset = SyntheticDataset(model_config, size=size, seed=1042)
    model = MGN(model_config, len(dataset.vocab))
    trainer = Trainer(model, config, dataset.vocab, args.output)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    initial = trainer.validate(loader)
    history = []
    for step in range(max_steps):
        history.append(trainer.update(collate_fn([dataset[i] for i in range(size)])))
        if (step + 1) % 10 == 0:
            final = trainer.validate(loader)
            predictions = predict(model, dataset, trainer.device)
            exact = sum(r['prediction'] == r['reference'] for r in predictions) / len(predictions)
            if final <= .2 * initial and exact >= .95:
                break
    final = trainer.validate(loader)
    checkpoint = Path(args.output) / 'smoke.pt'
    trainer.save(checkpoint)
    restored = MGN(model_config, len(dataset.vocab))
    restored.load_state_dict(load_checkpoint(checkpoint)['model'])
    restored.to(trainer.device)
    predictions = predict(restored, dataset, trainer.device)
    exact = sum(r['prediction'] == r['reference'] for r in predictions) / len(predictions)
    val_predictions = predict(restored, val_dataset, trainer.device)
    passed = final <= .2 * initial and exact >= .95
    report = dict(passed=passed, initial_nll=initial, final_nll=final, nll_reduction=1-final/initial, exact_match=exact,
                  thresholds=dict(min_nll_reduction=.8, min_exact_match=.95, max_steps=max_steps), steps=trainer.step,
                  independent_noise_validation_exact_match=sum(r['prediction'] == r['reference'] for r in val_predictions)/len(val_predictions),
                  predictions=predictions, parameters=sum(p.numel() for p in model.parameters()))
    dump(report, Path(args.output) / 'report.json')
    dump(history, Path(args.output) / 'loss_history.json')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError('smoke overfit acceptance thresholds not reached; inspect report.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('smoke', 'train'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--output', required=True)
        if name == 'train':
            p.add_argument('--resume')
    p = sub.add_parser('build-vocab')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--min-frequency', type=int, default=1)
    p = sub.add_parser('check-data')
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--vocab', required=True)
    for name in ('predict', 'evaluate'):
        p = sub.add_parser(name)
        p.add_argument('--checkpoint', required=True, help='trusted local checkpoint only')
        p.add_argument('--manifest', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if args.command == 'smoke':
        return smoke(args)
    if args.command == 'build-vocab':
        vocab = Vocabulary.build(read_manifest(args.manifest), args.min_frequency)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        vocab.save(args.output)
        print(json.dumps(dict(vocab_size=len(vocab), output=args.output)))
        return
    if args.command in ('train', 'check-data'):
        config = load_config(args.config)
        mc = ModelConfig.from_dict(config['model'])
        if args.command == 'check-data':
            dataset = FeatureDataset(args.manifest, Vocabulary.load(args.vocab), mc)
            for i in range(len(dataset)):
                dataset[i]
            print(json.dumps(dict(valid=True, samples=len(dataset))))
            return
        seed_everything(config.get('training', {}).get('seed', 42))
        torch.set_num_threads(config.get('training', {}).get('cpu_threads', 1))
        data = config['data']
        vocab = Vocabulary.load(data['vocab'])
        train = FeatureDataset(data['train_manifest'], vocab, mc, expected_split='train')
        val = FeatureDataset(data['val_manifest'], vocab, mc, expected_split='val')
        audit = assert_disjoint(train, val, allow_video_overlap=data.get('allow_video_overlap', False))
        dump(audit, Path(args.output) / 'split_audit.json')
        trainer = Trainer(MGN(mc, len(vocab)), config, vocab, args.output)
        if args.resume:
            trainer.resume(args.resume)
        print(json.dumps(trainer.fit(train, val), indent=2))
        return
    checkpoint = load_checkpoint(args.checkpoint)
    mc = ModelConfig.from_dict(checkpoint['config']['model'])
    vocab = Vocabulary(**checkpoint['vocabulary'])
    model = MGN(mc, len(vocab)).to(args.device)
    model.load_state_dict(checkpoint['model'])
    dataset = FeatureDataset(args.manifest, vocab, mc, require_answers=args.command == 'evaluate')
    results = predict(model, dataset, args.device)
    if args.command == 'evaluate':
        from mgn.metrics import evaluate_predictions
        result = evaluate_predictions([r['prediction'] for r in results], [r['reference'] for r in results])
        dump(dict(metrics=result, predictions=results), args.output)
    else:
        dump(results, args.output)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
