
import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

SPECIAL = ['<pad>', '<bos>', '<eos>', '<unk>']


def tokenize(text):
    import jieba
    return [x for x in jieba.lcut(text) if x.strip()]


class Vocabulary:
    def __init__(self, tokens, source_split='train', tokenizer='jieba'):
        if tokens[:4] != SPECIAL or len(tokens) != len(set(tokens)):
            raise ValueError('vocabulary must start with PAD/BOS/EOS/UNK and contain unique tokens')
        if source_split != 'train':
            raise ValueError('answer vocabulary must come from training split only')
        self.tokens, self.source_split, self.tokenizer = tokens, source_split, tokenizer
        self.index = {t: i for i, t in enumerate(tokens)}

    def __len__(self):
        return len(self.tokens)

    @classmethod
    def build(cls, rows, min_frequency=1):
        if min_frequency < 1:
            raise ValueError('min_frequency must be >= 1')
        if not rows or any(r.get('split') != 'train' for r in rows):
            raise ValueError('build-vocab accepts a nonempty training-only manifest')
        counts = Counter(t for r in rows for t in (r['answer_tokens'] if 'answer_tokens' in r else tokenize(r.get('answer', ''))))
        return cls(SPECIAL + sorted(t for t, n in counts.items() if n >= min_frequency and t not in SPECIAL))

    def encode(self, tokens, max_length):
        return [self.index.get(t, 3) for t in tokens[:max_length]]

    def decode(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if i == 2:
                break
            if i not in (0, 1):
                out.append(self.tokens[i])
        return out

    def to_dict(self):
        return dict(tokens=self.tokens, source_split=self.source_split, tokenizer=self.tokenizer)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))


def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [r.get('sample_id') for r in rows]
    if not rows or any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('manifest must contain unique nonempty sample_id strings')
    for row in rows:
        if row.get('split') not in {'train', 'val', 'test'} or not isinstance(row.get('video_id'), str) or not row['video_id']:
            raise ValueError('every row needs split=train/val/test and a string video_id')
    return rows


class FeatureDataset(Dataset):
    def __init__(self, manifest, vocabulary, config, expected_split=None, require_answers=True):
        self.path = Path(manifest).resolve()
        self.rows = read_manifest(self.path)
        self.vocab, self.config, self.require_answers = vocabulary, config, require_answers
        for modality in ('visual', 'question', 'audio'):
            provenance = []
            for row in self.rows:
                meta = row.get('features', {}).get(modality, {}).get('metadata', {})
                provenance.append(json.dumps({k: meta.get(k) for k in ('extractor', 'checkpoint', 'output_layer', 'preprocessing')}, sort_keys=True))
            if len(set(provenance)) != 1:
                raise ValueError(f'{modality}: inconsistent feature provenance within manifest')
        splits = {r['split'] for r in self.rows}
        if len(splits) != 1 or (expected_split is not None and splits != {expected_split}):
            raise ValueError('manifest split mismatch; use separate train/val/test manifests')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        result = dict(sample_id=row['sample_id'], video_id=row['video_id'])
        for modality in ('visual', 'question', 'audio'):
            entry = row.get('features', {}).get(modality)
            if not isinstance(entry, dict) or not {'path', 'metadata'} <= entry.keys():
                raise ValueError(f'{modality}: feature needs path and metadata')
            metadata = entry['metadata']
            if not isinstance(metadata, dict) or not all(metadata.get(k) for k in ('extractor', 'checkpoint', 'output_layer', 'preprocessing')):
                raise ValueError(f'{modality}: missing feature provenance metadata')
            path = self.path.parent / entry['path']
            if path.suffix == '.npz':
                with np.load(path, allow_pickle=False) as archive:
                    arr = np.array(archive[entry.get('key', 'features')])
                    if 'metadata' in archive:
                        embedded = json.loads(str(archive['metadata'].item()))
                        if embedded != metadata:
                            raise ValueError(f'{modality}: embedded cache metadata does not match manifest')
            elif path.suffix == '.npy':
                arr = np.load(path, allow_pickle=False)
            else:
                raise ValueError('feature cache must be .npy or .npz')
            dim = getattr(self.config, modality + '_dim')
            if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != dim or not np.issubdtype(arr.dtype, np.floating) or not np.isfinite(arr).all():
                raise ValueError(f'{modality}: expected finite floating [T,{dim}] with T>0')
            if arr.shape[0] > getattr(self.config, 'max_' + modality + '_len'):
                raise ValueError(f'{modality}: feature length exceeds configured maximum; resample in preprocessing')
            result[modality] = torch.from_numpy(np.array(arr, dtype=np.float32))
        tokens = row.get('answer_tokens')
        if tokens is None:
            if self.require_answers and not isinstance(row.get('answer'), str):
                raise ValueError('supervised samples need answer or answer_tokens')
            tokens = tokenize(row.get('answer', ''))
        if not isinstance(tokens, list) or any(not isinstance(t, str) for t in tokens):
            raise ValueError('answer_tokens must be a list of strings')
        ids = self.vocab.encode(tokens, self.config.max_answer_len)
        result.update(answer_in=torch.tensor([1] + ids), answer_target=torch.tensor(ids + [2]), reference_tokens=tokens)
        return result


def collate_fn(samples):
    result = dict(sample_ids=[s['sample_id'] for s in samples], video_ids=[s['video_id'] for s in samples], reference_tokens=[s['reference_tokens'] for s in samples])
    for modality in ('visual', 'question', 'audio'):
        values = [s[modality] for s in samples]
        result[modality] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True)
        result[modality + '_mask'] = torch.arange(result[modality].shape[1])[None] < torch.tensor([len(v) for v in values])[:, None]
    for key in ('answer_in', 'answer_target'):
        result[key] = torch.nn.utils.rnn.pad_sequence([s[key] for s in samples], batch_first=True, padding_value=0)
    return result


class SyntheticDataset(Dataset):

    def __init__(self, config, size=4, seed=42):
        self.config, self.size, self.seed = config, size, seed
        self.vocab = Vocabulary(SPECIAL + ['铃声', '流水', '脚步', '掌声'], tokenizer='explicit')
        rng = np.random.default_rng(1729)
        self.prototypes = {m: rng.normal(size=(4, getattr(config, m + '_dim'))).astype('float32') for m in ('visual', 'question', 'audio')}

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        event = index % 4
        rng = np.random.default_rng(self.seed + index)
        sample = dict(sample_id=f'synthetic-{self.seed}-{index}', video_id=f'event-video-{index}')
        for m in ('visual', 'question', 'audio'):
            length = min(getattr(self.config, 'max_' + m + '_len'), 2 + index % 3)
            values = self.prototypes[m][event][None] + rng.normal(0, .02, (length, getattr(self.config, m + '_dim')))
            sample[m] = torch.tensor(values, dtype=torch.float32)
        answer = self.vocab.index[['铃声', '流水', '脚步', '掌声'][event]]
        sample.update(answer_in=torch.tensor([1, answer]), answer_target=torch.tensor([answer, 2]), reference_tokens=[self.vocab.tokens[answer]])
        return sample


def assert_disjoint(train, val, allow_video_overlap=False):
    if {r['sample_id'] for r in train.rows} & {r['sample_id'] for r in val.rows}:
        raise ValueError('train/val sample IDs overlap')
    overlap = sorted({r['video_id'] for r in train.rows} & {r['video_id'] for r in val.rows})
    if overlap and not allow_video_overlap:
        raise ValueError('train/val video IDs overlap; an official split exception requires data.allow_video_overlap=true')
    return dict(allow_video_overlap=allow_video_overlap, overlapping_video_ids=overlap)
