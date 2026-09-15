
from torch.nn import functional as F


def compute_loss(outputs, targets, pad_id=0, contrastive_weight=0.01):
    logits = outputs["logits"]
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or targets.shape[0] == 0:
        raise ValueError("logits and targets must be [B,L,V] and [B,L] with B > 0")
    token_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=pad_id, reduction="none")
    nll = token_loss.reshape_as(targets).sum(dim=1).mean()
    contrastive = outputs.get("contrastive_loss", logits.sum() * 0)
    if contrastive_weight < 0:
        raise ValueError("contrastive_weight must be nonnegative")
    return {"loss": nll + contrastive_weight * contrastive, "nll": nll, "contrastive": contrastive}


__all__ = ["compute_loss"]
