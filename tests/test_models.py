import itertools
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from mgn.config import ModelConfig
from mgn.models import MGN
from mgn.models.graphs import SelfGraphLayer, PairGraphLayer
from mgn.models.fusion import BilinearFusion
from mgn.models.tpa import TPA
from mgn.losses import compute_loss


def tiny_config(**kwargs):
    values = dict(visual_dim=5, question_dim=7, audio_dim=6, hidden_dim=8,
                  max_visual_len=3, max_question_len=4, max_audio_len=5,
                  max_answer_len=3, encoder_heads=2, dropout=0,
                  tpa_rank=3, tpa_feature_rank=4, bilinear_rank=4,
                  encoder_kernel_size=3)
    values.update(kwargs)
    return ModelConfig(**values)


def batch_for(cfg, size=3):
    generator = torch.Generator().manual_seed(911)
    batch = {"video_ids": [f"video-{i}" for i in range(size)],
             "answer_in": torch.tensor([[1, 4, 5]] * size),
             "answer_target": torch.tensor([[4, 5, 2]] * size)}
    for name in ("visual", "question", "audio"):
        length = getattr(cfg, f"max_{name}_len")
        batch[name] = torch.randn(size, length, getattr(cfg, f"{name}_dim"), generator=generator)
        mask = torch.ones(size, length, dtype=torch.bool)
        mask[0, -1] = False
        batch[f"{name}_mask"] = mask
    return batch


def test_tpa_linear_matches_explicit_three_way_attention():
    torch.manual_seed(123)
    cfg = tiny_config(tpa_activation="identity")
    tpa = TPA(cfg).double()
    features, masks = {}, {}
    for name, length in [("visual", 2), ("question", 3), ("audio", 4)]:
        features[name] = torch.randn(2, length, cfg.hidden_dim, dtype=torch.double)
        masks[name] = torch.ones(2, length, dtype=torch.bool)
        masks[name][0, -1] = False
    actual = tpa(features, masks)
    projected = {name: features[name].masked_fill(~masks[name][..., None], 0) @ tpa.feature[name] for name in features}
    reference = torch.zeros(2, cfg.tpa_feature_rank, dtype=torch.double)
    for v, q, a in itertools.product(range(2), range(3), range(4)):
        weight = sum(tpa.reduction[h] * tpa.temporal["visual"][h, v] * tpa.temporal["question"][h, q] * tpa.temporal["audio"][h, a] for h in range(cfg.tpa_rank))
        reference = reference + weight * projected["visual"][:, v] * projected["question"][:, q] * projected["audio"][:, a]
    torch.testing.assert_close(actual, reference @ tpa.output.weight.T, atol=1e-10, rtol=1e-10)


def test_self_graph_matches_scalar_neighbor_reference():
    torch.manual_seed(9)
    graph = SelfGraphLayer(3).double()
    x = torch.randn(1, 4, 3, dtype=torch.double)
    mask = torch.tensor([[True, True, False, True]])
    actual, attention = graph(x, mask, return_attention=True)
    projected = x @ graph.projection.weight.T
    expected = torch.zeros_like(actual)
    for i in [0, 1, 3]:
        scores = torch.stack([F.leaky_relu(graph.source_score(projected[:, i]).squeeze() + graph.target_score(projected[:, j]).squeeze(), .2) for j in [0, 1, 3]])
        expected[0, i] = F.elu(sum(w * projected[0, j] for w, j in zip(scores.softmax(0), [0, 1, 3])))
    torch.testing.assert_close(actual, expected)
    assert torch.equal(attention[:, :, 2], torch.zeros_like(attention[:, :, 2]))
    torch.testing.assert_close(attention.sum(-1), torch.ones(1, 4, dtype=torch.double))


def test_pair_graph_aggregates_opposite_modality_and_normalizes_both_directions():
    graph = PairGraphLayer(2).double()
    with torch.no_grad():
        graph.left_score.weight.zero_()
        graph.right_score.weight.zero_()
        graph.left_message.weight.copy_(torch.eye(2))
        graph.right_message.weight.copy_(torch.eye(2))
    left = torch.tensor([[[1., 2.], [3., 4.], [999., 999.]]], dtype=torch.double)
    right = torch.tensor([[[5., 6.], [7., 8.]]], dtype=torch.double)
    left_mask = torch.tensor([[True, True, False]])
    right_mask = torch.tensor([[True, True]])
    lout, rout, weights = graph(left, right, left_mask, right_mask)
    torch.testing.assert_close(lout, torch.tensor([[[6., 7.], [6., 7.], [0., 0.]]], dtype=torch.double))
    torch.testing.assert_close(rout, torch.tensor([[[2., 3.], [2., 3.]]], dtype=torch.double))
    torch.testing.assert_close(weights, torch.tensor([[[.5, .5, 0.], [.5, .5, 0.]]], dtype=torch.double))


def test_bilinear_matches_explicit_rank_sum():
    torch.manual_seed(3)
    fusion = BilinearFusion(3, 4).double()
    x, y = torch.randn(2, 3, dtype=torch.double), torch.randn(2, 3, dtype=torch.double)
    expected = torch.stack([sum(fusion.output.weight[:, r] * (x[b] @ fusion.left.weight[r]).relu() * (y[b] @ fusion.right.weight[r]).relu() for r in range(4)) + fusion.output.bias for b in range(2)])
    torch.testing.assert_close(fusion(x, y), expected)


@pytest.mark.parametrize("variant", ["mgn", "ppa", "tpa"])
def test_padding_values_and_extra_padding_do_not_change_outputs(variant):
    torch.manual_seed(31)
    cfg = tiny_config(variant=variant)
    model = MGN(cfg, 9).eval()
    batch = batch_for(cfg)
    original = model(batch)
    dirty = dict(batch)
    for name in ("visual", "question", "audio"):
        dirty[name] = batch[name].masked_fill(~batch[f"{name}_mask"][..., None], float("nan"))
    changed = model(dirty)
    torch.testing.assert_close(original["logits"], changed["logits"])
    torch.testing.assert_close(original["contrastive_loss"], changed["contrastive_loss"])

    short = {key: value[:1] if isinstance(value, torch.Tensor) else value[:1] for key, value in batch.items()}
    padded = dict(short)
    for name in ("visual", "question", "audio"):
        short[name] = short[name][:, :-1]
        short[f"{name}_mask"] = short[f"{name}_mask"][:, :-1]
    torch.testing.assert_close(model(short)["logits"], model(padded)["logits"], atol=1e-6, rtol=1e-5)


def test_enabled_modules_have_finite_nonzero_gradients():
    torch.manual_seed(19)
    cfg = tiny_config()
    model = MGN(cfg, 9)
    batch = batch_for(cfg)
    outputs = model(batch)
    result = compute_loss(outputs, batch["answer_target"])
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    for module in [*model.encoders.values(), *model.ppa.self_graphs.values(), *model.ppa.pair_graphs.values(), model.ppa.gates, model.ppa.fusions, model.ppa.final_fusion, model.tpa, model.decoder]:
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0


def test_contrastive_candidates_use_anchor_question_and_exclude_same_video():
    cfg = tiny_config()
    model = MGN(cfg, 9).eval()
    batch = batch_for(cfg)
    batch["video_ids"] = ["same", "same", "different"]
    prepared = {"question": torch.stack([torch.full((4, 8), float(i + 1)) for i in range(3)]),
                "audio": torch.randn(3, 5, 8)}
    masks = {"question": batch["question_mask"], "audio": batch["audio_mask"]}
    qv, qs = torch.randn(3, 4, 8), torch.randn(3, 4, 8)
    calls = []
    def fake_pair(name, content, question, content_mask, question_mask):
        calls.append((question.clone(), content.clone()))
        return None, question
    with patch.object(model.ppa, "pair", side_effect=fake_pair):
        value = model.ppa.contrastive_loss(prepared, masks, qv, qs, batch["video_ids"])
    assert torch.isfinite(value)
    assert [call[0].shape[0] for call in calls] == [1, 1, 2]
    for anchor, (questions, audios) in enumerate(calls):
        torch.testing.assert_close(questions, prepared["question"][anchor:anchor+1].expand_as(questions))
        indices = [2] if anchor < 2 else [0, 1]
        torch.testing.assert_close(audios, prepared["audio"][indices])


def test_contrastive_value_matches_direct_candidate_matrix():
    torch.manual_seed(99)
    cfg = tiny_config(contrastive_chunk_size=1)
    model = MGN(cfg, 9).eval()
    batch = batch_for(cfg)
    features = {name: model.encoders[name](batch[name], batch[f"{name}_mask"]) for name in model.names}
    masks = {name: batch[f"{name}_mask"] for name in model.names}
    prepared = model.ppa.prepare(features, masks)
    _, qv = model.ppa.pair("visual", prepared["visual"], prepared["question"], masks["visual"], masks["question"])
    _, qs = model.ppa.pair("audio", prepared["audio"], prepared["question"], masks["audio"], masks["question"])
    actual = model.ppa.contrastive_loss(prepared, masks, qv, qs, batch["video_ids"])
    from mgn.models.common import masked_mean
    anchors = masked_mean(qv, masks["question"])
    scores = torch.empty(3, 3)
    for i in range(3):
        for j in range(3):
            _, candidate = model.ppa.pair("audio", prepared["audio"][j:j+1], prepared["question"][i:i+1], masks["audio"][j:j+1], masks["question"][i:i+1])
            scores[i, j] = (anchors[i] * masked_mean(candidate, masks["question"][i:i+1])[0]).sum()
    torch.testing.assert_close(actual, F.cross_entropy(scores, torch.arange(3)))


@pytest.mark.parametrize("size", [1, 3])
def test_no_legal_negatives_has_zero_loss(size):
    cfg = tiny_config()
    batch = batch_for(cfg, size)
    batch["video_ids"] = ["same"] * size
    result = MGN(cfg, 9)(batch)
    assert result["contrastive_loss"].item() == 0


@pytest.mark.parametrize("variant,graph_mode,modalities", [("ppa", "full", "v"), ("ppa", "full", "a"), ("ppa", "no_self", "va"), ("ppa", "no_pair", "va"), ("ppa", "none", "va"), ("tpa", "full", "va")])
def test_ablations_remove_unused_modules(variant, graph_mode, modalities):
    cfg = tiny_config(variant=variant, graph_mode=graph_mode, modalities=modalities)
    model = MGN(cfg, 9)
    batch = batch_for(cfg)
    if modalities == "v":
        del batch["audio"], batch["audio_mask"]
        assert "audio" not in model.encoders
    if modalities == "a":
        del batch["visual"], batch["visual_mask"]
        assert "visual" not in model.encoders
    outputs = model(batch)
    compute_loss(outputs, batch["answer_target"])["loss"].backward()
    assert all(p.grad is not None for p in model.parameters())
    assert (model.ppa is None) == (variant == "tpa")
    assert (model.tpa is None) == (variant == "ppa")
    if modalities != "va" or graph_mode in {"no_pair", "none"} or variant == "tpa":
        assert outputs["contrastive_loss"].item() == 0


def test_generation_stops_individual_rows_at_eos_and_never_uses_answers():
    cfg = tiny_config()
    model = MGN(cfg, 9).eval()
    batch = batch_for(cfg, size=2)
    del batch["answer_in"], batch["answer_target"]
    steps = []
    def fake_step(token, memory, keys, mask, state):
        index = len(steps)
        steps.append(token.clone())
        logits = torch.zeros(2, 9)
        logits[:, cfg.pad_id] = 1000
        logits[:, cfg.bos_id] = 1000
        logits[0, cfg.eos_id] = 100
        logits[1, 4 if index == 0 else cfg.eos_id] = 100
        return logits, state
    with patch.object(model.decoder, "step", side_effect=fake_step):
        generated = model.generate(batch, max_length=4)
    assert generated.tolist() == [[2, 0, 0, 0], [4, 2, 0, 0]]
    assert len(steps) == 2


@pytest.mark.parametrize("problem", ["empty", "overlong", "bad_mask"])
def test_invalid_required_inputs_fail_clearly(problem):
    cfg = tiny_config()
    model = MGN(cfg, 9)
    batch = batch_for(cfg)
    if problem == "empty":
        batch["audio_mask"][0].fill_(False)
    elif problem == "overlong":
        batch["audio"] = torch.zeros(3, cfg.max_audio_len + 1, cfg.audio_dim)
        batch["audio_mask"] = torch.ones(3, cfg.max_audio_len + 1, dtype=torch.bool)
    else:
        batch["audio_mask"] = batch["audio_mask"].long()
    with pytest.raises(ValueError):
        model(batch)


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_contrastive_checkpoint_matches_loss_and_gradients(dropout):
    torch.manual_seed(345)
    cfg = tiny_config(contrastive_chunk_size=1, contrastive_checkpoint=True, dropout=dropout)
    checkpointed = MGN(cfg, 9).train()
    direct = MGN(tiny_config(contrastive_chunk_size=1, contrastive_checkpoint=False, dropout=dropout), 9).train()
    direct.load_state_dict(checkpointed.state_dict())
    batch = batch_for(cfg)
    results = []
    for model in (checkpointed, direct):
        torch.manual_seed(987)
        loss = compute_loss(model(batch), batch["answer_target"])["loss"]
        loss.backward()
        results.append(loss)
    torch.testing.assert_close(results[0], results[1])
    for (name, left), (_, right) in zip(checkpointed.named_parameters(), direct.named_parameters()):
        torch.testing.assert_close(left.grad, right.grad, msg=lambda message: f"{name}: {message}")


def test_generation_reserves_final_position_for_eos():
    cfg = tiny_config(max_answer_len=3)
    model = MGN(cfg, 9).eval()
    batch = batch_for(cfg, size=2)
    del batch["answer_in"], batch["answer_target"]
    with torch.no_grad():
        model.decoder.output.weight.zero_()
        model.decoder.output.bias.zero_()
        model.decoder.output.bias[4] = 100
    assert model.generate(batch).tolist() == [[4, 4, 4, 2], [4, 4, 4, 2]]
    assert model.generate(batch, max_length=1).tolist() == [[2], [2]]
