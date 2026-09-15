
import json
import hashlib
import platform
import subprocess
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mgn.losses import compute_loss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.checkpoint-')
    os.close(fd)
    try:
        with open(temporary, 'wb') as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path):

    return torch.load(path, map_location='cpu', weights_only=False)


class Trainer:
    def __init__(self, model, config, vocabulary, output):
        self.model, self.config, self.vocabulary = model, config, vocabulary
        self.options = config.get('training', {})
        self.device = torch.device(self.options.get('device', 'cpu'))
        self.model.to(self.device)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.options.get('lr', .0005))
        warmup = self.options.get('warmup_steps', 100)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(1., (step + 1) / max(1, warmup)))
        self.epoch, self.step, self.bad_epochs = 0, 0, 0
        self.best = float('inf')
        self.loader_generator = torch.Generator().manual_seed(self.options.get('seed', 42))

    def update(self, batch):
        self.model.train()
        batch = to_device(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(batch, compute_contrastive=self.options.get('contrastive_weight', .01) > 0)
        losses = compute_loss(outputs, batch['answer_target'], pad_id=0, contrastive_weight=self.options.get('contrastive_weight', .01))
        if not torch.isfinite(losses['loss']):
            raise FloatingPointError('nonfinite training loss')
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.options.get('grad_clip', 5.), error_if_nonfinite=True)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return {k: float(v.detach()) for k, v in losses.items()}

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total, count = 0., 0
        for batch in loader:
            batch = to_device(batch, self.device)
            losses = compute_loss(self.model(batch, compute_contrastive=False), batch['answer_target'], pad_id=0, contrastive_weight=0.)
            if not torch.isfinite(losses['nll']):
                raise FloatingPointError('nonfinite validation NLL')
            n = batch['answer_target'].shape[0]
            total += float(losses['nll']) * n
            count += n
        if not count:
            raise ValueError('validation dataset is empty')
        return total / count

    def provenance(self):
        hashes = {}
        for key in ('train_manifest', 'val_manifest', 'vocab'):
            path = self.config.get('data', {}).get(key)
            if path:
                hashes[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        hashes['vocabulary'] = hashlib.sha256(json.dumps(self.vocabulary.to_dict(), sort_keys=True).encode()).hexdigest()
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        return dict(hashes=hashes, environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, platform=platform.platform()), code_commit=commit)

    def save(self, path):
        state = dict(format_version=1, provenance=self.provenance(), model=self.model.state_dict(), optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
                     config=self.config, vocabulary=self.vocabulary.to_dict(), epoch=self.epoch, step=self.step,
                     best=self.best, bad_epochs=self.bad_epochs, loader_rng=self.loader_generator.get_state(),
                     rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                              cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                     resume_contract='epoch_boundary; num_workers=0; same dataset/config/runtime/device')
        atomic_save(state, path)

    def resume(self, path):
        state = load_checkpoint(path)
        if state['provenance']['hashes'] != self.provenance()['hashes']:
            raise ValueError('resume manifest/vocabulary content hash changed')
        if state['config']['model'] != self.config['model'] or state['vocabulary'] != self.vocabulary.to_dict():
            raise ValueError('checkpoint model configuration or vocabulary mismatch')
        old_options = state['config'].get('training', {})
        for key in set(old_options) | set(self.options):
            if key not in {'epochs', 'device'} and old_options.get(key) != self.options.get(key):
                raise ValueError(f'resume training option changed: {key}')
        if state['config'].get('data') != self.config.get('data'):
            raise ValueError('resume data configuration changed')
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        for key in ('epoch', 'step', 'best', 'bad_epochs'):
            setattr(self, key, state[key])
        self.loader_generator.set_state(state['loader_rng'])
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        if state['rng']['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['rng']['cuda'])

    def fit(self, train_data, val_data):
        from mgn.data import collate_fn
        if self.options.get('num_workers', 0) != 0:
            raise ValueError('num_workers must be 0 for epoch-boundary resume')
        train_loader = DataLoader(train_data, batch_size=self.options.get('batch_size', 64), shuffle=True,
                                  generator=self.loader_generator, collate_fn=collate_fn)
        val_loader = DataLoader(val_data, batch_size=self.options.get('batch_size', 64), collate_fn=collate_fn)
        history = []
        for epoch in range(self.epoch, self.options.get('epochs', 30)):
            if self.bad_epochs >= self.options.get('early_stopping_patience', 5):
                break
            sums, count = dict(loss=0., nll=0., contrastive=0.), 0
            for batch in train_loader:
                losses = self.update(batch)
                n = batch['answer_target'].shape[0]
                for key in sums:
                    sums[key] += losses[key] * n
                count += n
            score = self.validate(val_loader)
            self.epoch = epoch + 1
            improved = score < self.best
            self.best, self.bad_epochs = (score, 0) if improved else (self.best, self.bad_epochs + 1)
            record = dict(epoch=self.epoch, step=self.step, val_nll=score, best_val_nll=self.best, lr=self.optimizer.param_groups[0]['lr'], **{'train_' + k: v/count for k, v in sums.items()})
            history.append(record)
            with (self.output / 'history.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            if improved:
                self.save(self.output / 'best.pt')
            self.save(self.output / 'last.pt')
        return history
