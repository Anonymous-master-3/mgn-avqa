
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
import math
from typing import Any

Tokens = str | Sequence[str]
Similarity = Callable[[str, str], float | None]


class MetricUnavailable(RuntimeError):
    pass


def _tokens(value: Tokens) -> list[str]:
    tokens = value.split() if isinstance(value, str) else list(value)
    if any(not isinstance(token, str) or not token for token in tokens):
        raise ValueError("Tokens must be nonempty strings")
    return tokens


def corpus_bleu(predictions: Sequence[Tokens], references: Sequence[Tokens], order: int = 4) -> float:





    if order not in (1, 2, 3, 4):
        raise ValueError("BLEU order must be 1, 2, 3 or 4")
    if len(predictions) != len(references) or not predictions:
        raise ValueError("Predictions and references must have the same nonzero length")
    matches, totals = [0] * order, [0] * order
    predicted_length = reference_length = 0
    for prediction, reference in zip(predictions, references):
        hypothesis, target = _tokens(prediction), _tokens(reference)
        predicted_length += len(hypothesis)
        reference_length += len(target)
        for n in range(1, order + 1):
            hc = Counter(tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) - n + 1))
            rc = Counter(tuple(target[i:i + n]) for i in range(len(target) - n + 1))
            matches[n - 1] += sum((hc & rc).values())
            totals[n - 1] += sum(hc.values())
    if not predicted_length or any(not count for count in matches):
        return 0.0
    bp = math.exp(min(0.0, 1.0 - reference_length / predicted_length))
    return bp * math.exp(sum(math.log(m / t) for m, t in zip(matches, totals)) / order)


def similarity_from_table(table: Mapping[tuple[str, str], float], *, missing: float | None = None) -> Similarity:

    values = dict(table)
    for pair, value in values.items():
        if not isinstance(pair, tuple) or len(pair) != 2 or not all(isinstance(item, str) for item in pair):
            raise ValueError("Similarity table keys must be (token, token) tuples")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Similarities must be finite in [0,1]")
        if (pair[1], pair[0]) in values and values[(pair[1], pair[0])] != value:
            raise ValueError("Similarity table must be symmetric")
    if missing is not None and (not math.isfinite(missing) or not 0 <= missing <= 1):
        raise ValueError("Missing-pair similarity must be in [0,1] or None")

    def similarity(left: str, right: str) -> float | None:
        if left == right:
            return 1.0
        return values.get((left, right), values.get((right, left), missing))

    return similarity


def wordnet_similarity(*, language: str = "eng") -> Similarity:





    try:
        from nltk.corpus import wordnet
        wordnet.ensure_loaded()
        if language != "eng":
            wordnet.synsets("测试", lang=language)
    except (ImportError, LookupError) as error:
        raise MetricUnavailable(f"Local WordNet/OMW resources are unavailable for language={language}") from error

    @lru_cache(maxsize=65536)
    def similarity(left: str, right: str) -> float | None:
        if left == right:
            return 1.0
        try:
            first, second = wordnet.synsets(left, lang=language), wordnet.synsets(right, lang=language)
        except LookupError as error:
            raise MetricUnavailable(f"WordNet resource missing for language={language}") from error
        if not first or not second:
            return None
        scores = [score for a in first for b in second if (score := a.wup_similarity(b)) is not None]
        return max(scores, default=0.0)

    return similarity


def wups_score(prediction: Tokens, reference: Tokens, similarity: Similarity, *, threshold: float = 0.0) -> float:

    if not 0 <= threshold <= 1:
        raise ValueError("WUPS threshold must be in [0,1]")
    first, second = sorted(set(_tokens(prediction))), sorted(set(_tokens(reference)))
    if not first or not second:
        return float(not first and not second)
    scores = []
    for left in first:
        row = []
        for right in second:
            value = similarity(left, right)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("Token similarity must be finite in [0,1]")
            row.append(None if value is None else value if value >= threshold else value * 0.1)
        scores.append(row)

    def best(values):


        if 1.0 in values:
            return 1.0
        if any(value is None for value in values):
            raise MetricUnavailable("Token similarity resource does not cover a required cross-answer match")
        return max(values)

    forward = math.prod(best(row) for row in scores)
    reverse = math.prod(best([scores[i][j] for i in range(len(first))]) for j in range(len(second)))
    return min(forward, reverse)


def key_recall(predictions: Sequence[Tokens], keywords: Sequence[Tokens]) -> float:

    if len(predictions) != len(keywords) or not predictions:
        raise ValueError("Predictions and keywords must have the same nonzero length")
    values = []
    for prediction, annotated in zip(predictions, keywords):
        keys = set(_tokens(annotated))
        if not keys:
            raise MetricUnavailable("KeyRecall requires a nonempty explicit keyword set for every sample")
        values.append(len(set(_tokens(prediction)) & keys) / len(keys))
    return sum(values) / len(values)


def evaluate_predictions(
    predictions: Sequence[Tokens], references: Sequence[Tokens], *, keywords: Sequence[Tokens] | None = None,
    token_similarity: Similarity | Mapping[tuple[str, str], float] | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:






    if len(predictions) != len(references) or not predictions:
        raise ValueError("Predictions and references must have the same nonzero length")
    predictions, references = [_tokens(value) for value in predictions], [_tokens(value) for value in references]
    result: dict[str, Any] = {f"bleu_{order}": 100 * corpus_bleu(predictions, references, order) for order in range(1, 5)}
    status: dict[str, Any] = {"bleu": {"status": "available"}}
    result.update(wups_0=None, wups_09=None, key_recall=None)
    if token_similarity is None:
        status["wups"] = {"status": "unavailable", "reason": "No token similarity resource supplied"}
    else:
        similarity = similarity_from_table(token_similarity) if isinstance(token_similarity, Mapping) else token_similarity
        try:
            scores = {}
            for name, threshold in (("wups_0", 0.0), ("wups_09", 0.9)):
                scores[name] = 100 * sum(wups_score(p, r, similarity, threshold=threshold) for p, r in zip(predictions, references)) / len(predictions)
            result.update(scores)
            status["wups"] = {"status": "available"}
        except MetricUnavailable as error:
            status["wups"] = {"status": "unavailable", "reason": str(error)}
    if keywords is None:
        status["key_recall"] = {"status": "unavailable", "reason": "No explicit per-sample keyword annotations supplied"}
    else:
        try:
            result["key_recall"] = 100 * key_recall(predictions, keywords)
            status["key_recall"] = {"status": "available"}
        except MetricUnavailable as error:
            status["key_recall"] = {"status": "unavailable", "reason": str(error)}
    result["status"] = status
    result["protocol"] = {
        "units": "percent", "references_per_sample": 1, "tokenization": "caller tokens; string inputs split on whitespace",
        "special_tokens": "caller removes BOS/EOS/PAD before evaluation",
        "bleu": {"aggregation": "corpus", "smoothing": "none", "effective_order": False, "max_order": 4},
        "wups": {"aggregation": "sample mean", "tokens": "unique set", "below_threshold_multiplier": 0.1,
                 "resource": "caller supplied" if token_similarity is not None else "unavailable"},
        "key_recall": {"aggregation": "sample mean", "keywords": "explicit unique token sets", "matching": "exact"},
        "caller_metadata": dict(protocol or {}),
    }
    result["sample_count"] = len(predictions)
    return result
