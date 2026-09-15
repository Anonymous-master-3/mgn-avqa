
import torch
from torch import nn
from .common import masked_mean


class AnswerDecoder(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.embedding = nn.Embedding(vocab_size, d, padding_idx=cfg.pad_id)
        self.memory_key = nn.Linear(d, d, bias=False)
        self.query_key = nn.Linear(d, d, bias=False)
        self.attention_score = nn.Linear(d, 1, bias=False)
        self.initial_hidden = nn.Linear(d, d)
        self.initial_cell = nn.Linear(d, d)
        self.cell = nn.LSTMCell(2 * d, d)
        self.output = nn.Linear(2 * d, vocab_size)
        self.dropout = nn.Dropout(cfg.dropout)

    def initial_state(self, memory, mask):
        summary = masked_mean(memory, mask)
        return self.initial_hidden(summary).tanh(), self.initial_cell(summary).tanh()

    def step(self, token, memory, keys, mask, state):
        hidden, cell = state
        energies = self.attention_score((keys + self.query_key(hidden)[:, None, :]).tanh()).squeeze(-1)
        weights = energies.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        context = (weights[:, :, None] * memory).sum(dim=1)
        hidden, cell = self.cell(torch.cat([self.embedding(token), context], dim=-1), (hidden, cell))
        logits = self.output(self.dropout(torch.cat([hidden, context], dim=-1)))
        return logits, (hidden, cell)

    def forward(self, memory, mask, answer_in):
        if answer_in.ndim != 2 or answer_in.shape[1] == 0:
            raise ValueError("answer_in must be [batch, positive_length]")
        keys, state = self.memory_key(memory), self.initial_state(memory, mask)
        outputs = []
        for token in answer_in.unbind(dim=1):
            logits, state = self.step(token, memory, keys, mask, state)
            outputs.append(logits)
        return torch.stack(outputs, dim=1)

    def generate(self, memory, mask, max_length=None):
        max_length = self.cfg.max_answer_len + 1 if max_length is None else max_length
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        batch = memory.shape[0]
        result = torch.full((batch, max_length), self.cfg.pad_id, device=memory.device, dtype=torch.long)
        token = torch.full((batch,), self.cfg.bos_id, device=memory.device, dtype=torch.long)
        finished = torch.zeros(batch, device=memory.device, dtype=torch.bool)
        keys, state = self.memory_key(memory), self.initial_state(memory, mask)
        for index in range(max_length):
            logits, state = self.step(token, memory, keys, mask, state)
            logits[:, self.cfg.pad_id] = float("-inf")
            logits[:, self.cfg.bos_id] = float("-inf")
            token = logits.argmax(dim=-1)


            if index == max_length - 1:
                token = torch.full_like(token, self.cfg.eos_id)
            token = token.masked_fill(finished, self.cfg.pad_id)
            result[:, index] = token
            finished = finished | token.eq(self.cfg.eos_id)
            if bool(finished.all()):
                break
        return result
