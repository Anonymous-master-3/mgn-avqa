import math
import pytest
import torch
from mgn.losses import compute_loss


def test_nll_sums_tokens_per_sample_and_averages_batch():
    logits = torch.zeros(2, 3, 5, requires_grad=True)
    targets = torch.tensor([[1, 2, 0], [3, 0, 0]])
    result = compute_loss({"logits": logits, "contrastive_loss": torch.tensor(2.)}, targets)
    assert result["nll"].item() == pytest.approx(1.5 * math.log(5))
    assert result["loss"].item() == pytest.approx(1.5 * math.log(5) + .02)
    result["loss"].backward()
    assert torch.equal(logits.grad[targets == 0], torch.zeros_like(logits.grad[targets == 0]))
    assert logits.grad[targets != 0].abs().sum() > 0


def test_ignored_targets_are_zero_even_when_all_padding():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    result = compute_loss({"logits": logits}, torch.zeros(2, 3, dtype=torch.long))
    assert result["loss"].item() == 0
    result["loss"].backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
