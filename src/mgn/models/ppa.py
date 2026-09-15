
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .common import FeedForward, masked, masked_mean
from .graphs import SelfGraph, PairGraph
from .fusion import CrossGate, BilinearFusion


class PPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.contents = [name for name, code in [("visual", "v"), ("audio", "a")] if code in cfg.modalities]
        names = ["question"] + self.contents
        self.self_graphs = nn.ModuleDict()
        self.replacements = nn.ModuleDict()
        if cfg.graph_mode in {"full", "no_pair"}:
            self.self_graphs = nn.ModuleDict({name: SelfGraph(cfg.hidden_dim, cfg.graph_layers, cfg.dropout) for name in names})
        if cfg.graph_mode == "none":
            self.replacements = nn.ModuleDict({name: FeedForward(cfg.hidden_dim, cfg.dropout) for name in names})
        self.pair_graphs = nn.ModuleDict()
        if cfg.graph_mode in {"full", "no_self"}:
            self.pair_graphs = nn.ModuleDict({name: PairGraph(cfg.hidden_dim, cfg.graph_layers, cfg.dropout) for name in self.contents})
        self.gates = nn.ModuleDict({name: CrossGate(cfg.hidden_dim) for name in self.contents})
        self.fusions = nn.ModuleDict({name: BilinearFusion(cfg.hidden_dim, cfg.bilinear_rank) for name in self.contents})
        self.final_fusion = BilinearFusion(cfg.hidden_dim, cfg.bilinear_rank) if len(self.contents) == 2 else None

    def prepare(self, features, masks):
        result = {}
        for name, x in features.items():
            if name in self.self_graphs:
                x = self.self_graphs[name](x, masks[name])
            elif name in self.replacements:
                x = masked(self.replacements[name](x), masks[name])
            result[name] = x
        return result

    def pair(self, name, content, question, content_mask, question_mask):
        if name in self.pair_graphs:
            content, question, weights = self.pair_graphs[name](content, question, content_mask, question_mask)
            aligned = masked(weights @ content, question_mask)
        else:
            aligned = masked(masked_mean(content, content_mask)[:, None, :].expand(-1, question.shape[1], -1), question_mask)
        return aligned, question

    def contrastive_loss(self, prepared, masks, qv, qs, video_ids):



        batch_size = qv.shape[0]
        if len(video_ids) != batch_size:
            raise ValueError("video_ids must have one item per sample")
        if batch_size == 1 or len(set(video_ids)) == 1:
            return qv.sum() * 0
        visual = masked_mean(qv, masks["question"])
        positive = masked_mean(qs, masks["question"])
        if self.cfg.contrastive_normalize:
            visual = F.normalize(visual, dim=-1)
            positive = F.normalize(positive, dim=-1)
        losses = []
        for anchor in range(batch_size):
            negative_ids = [j for j in range(batch_size) if video_ids[j] != video_ids[anchor]]
            if not negative_ids:
                continue
            scores = [(visual[anchor] * positive[anchor]).sum().reshape(1)]
            for begin in range(0, len(negative_ids), self.cfg.contrastive_chunk_size):
                indices = torch.tensor(negative_ids[begin:begin+self.cfg.contrastive_chunk_size], device=qv.device)
                count = len(indices)
                anchor_q = prepared["question"][anchor:anchor+1].expand(count, -1, -1)
                anchor_mask = masks["question"][anchor:anchor+1].expand(count, -1)
                def candidate_question(audio, question, audio_mask, question_mask):
                    return self.pair("audio", audio, question, audio_mask, question_mask)[1]

                args = (prepared["audio"][indices], anchor_q, masks["audio"][indices], anchor_mask)
                if self.training and torch.is_grad_enabled() and self.cfg.contrastive_checkpoint:
                    candidate_q = checkpoint(candidate_question, *args, use_reentrant=False, preserve_rng_state=True)
                else:
                    candidate_q = candidate_question(*args)
                candidates = masked_mean(candidate_q, anchor_mask)
                if self.cfg.contrastive_normalize:
                    candidates = F.normalize(candidates, dim=-1)
                scores.append((visual[anchor] * candidates).sum(dim=-1))
            logits = torch.cat(scores) / self.cfg.contrastive_temperature
            losses.append(torch.logsumexp(logits, dim=0) - logits[0])
        return torch.stack(losses).mean() if losses else qv.sum() * 0

    def forward(self, features, masks, video_ids=None, compute_contrastive=True):
        prepared = self.prepare(features, masks)
        branches, questions = {}, {}
        for name in self.contents:
            aligned, question = self.pair(name, prepared[name], prepared["question"], masks[name], masks["question"])
            questions[name] = question
            left, right = self.gates[name](aligned, question)
            branches[name] = masked(self.fusions[name](left, right), masks["question"])
        local = self.final_fusion(branches["visual"], branches["audio"]) if self.final_fusion is not None else branches[self.contents[0]]
        local = masked(local, masks["question"])
        loss = local.sum() * 0
        if compute_contrastive and self.cfg.contrastive and len(self.contents) == 2 and len(self.pair_graphs) == 2:
            if video_ids is None:
                raise ValueError("video_ids required to exclude same-video contrastive negatives")
            loss = self.contrastive_loss(prepared, masks, questions["visual"], questions["audio"], video_ids)
        return local, loss
