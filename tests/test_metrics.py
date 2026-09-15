import json
import math

import pytest

from mgn.metrics import (
    MetricUnavailable, corpus_bleu, evaluate_predictions, key_recall,
    similarity_from_table, wups_score,
)


def test_bleu_exact_match_and_fixed_order_short_sentence():
    tokens = ["红色", "吹风机", "正在", "工作"]
    assert [corpus_bleu([tokens], [tokens], n) for n in range(1, 5)] == [1.0] * 4
    assert corpus_bleu([["红色"]], [["红色"]], 1) == 1
    assert corpus_bleu([["红色"]], [["红色"]], 2) == 0


def test_bleu_clips_repeated_counts_and_applies_corpus_brevity():

    assert corpus_bleu([["a", "a", "a"]], [["a", "b", "c"]], 1) == pytest.approx(1 / 3)

    assert corpus_bleu([["a"], ["b", "c"]], [["a", "x"], ["b", "c", "y"]], 1) == pytest.approx(math.exp(1 - 5 / 3))


def test_bleu_matches_independent_nltk_oracle_for_long_sentences():
    from nltk.translate.bleu_score import corpus_bleu as nltk_bleu
    predictions = ["a b c d e a".split(), "d e a b c".split()]
    references = ["a b c d f a".split(), "d e a b c".split()]
    for order in range(1, 5):
        expected = nltk_bleu([[ref] for ref in references], predictions, weights=[1 / order] * order)
        assert corpus_bleu(predictions, references, order) == pytest.approx(expected)


def test_bleu_empty_and_nonmatching_inputs():
    assert corpus_bleu([[]], [["word"]]) == 0
    assert corpus_bleu([["a"]], [["b"]], 1) == 0
    with pytest.raises(ValueError):
        corpus_bleu([], [])
    with pytest.raises(ValueError):
        corpus_bleu([["a"]], [])


def test_wups_threshold_fuzzy_product_and_set_semantics():
    similarity = similarity_from_table({("dog", "animal"): 0.8, ("cat", "animal"): 0.6})

    assert wups_score(["dog", "cat", "cat"], ["animal"], similarity) == pytest.approx(0.48)
    assert wups_score(["dog", "cat"], ["animal"], similarity, threshold=0.9) == pytest.approx(0.0048)
    assert wups_score([], [], similarity) == 1
    assert wups_score([], ["animal"], similarity) == 0
    assert wups_score(["dog", "cat"], ["dog", "cat"], similarity_from_table({})) == 1


def test_wups_unavailable_is_not_reported_as_zero():
    result = evaluate_predictions([["风声"]], [["水声"]])
    assert result["wups_0"] is None
    assert result["wups_09"] is None
    assert result["status"]["wups"]["status"] == "unavailable"
    partial = evaluate_predictions([["风声"]], [["水声"]], token_similarity={})
    assert partial["wups_0"] is None
    with pytest.raises(MetricUnavailable):
        wups_score(["x"], ["y"], similarity_from_table({}))


def test_explicit_keywords_and_protocol_json():
    result = evaluate_predictions([["红色", "运行"]], [["红色", "机器", "运行"]],
                                  keywords=[["红色", "机器"]],
                                  token_similarity=similarity_from_table({}, missing=0.0),
                                  protocol={"tokenizer": "fixture"})
    assert result["key_recall"] == 50
    assert result["wups_0"] == 0
    assert result["protocol"]["caller_metadata"] == {"tokenizer": "fixture"}
    json.dumps(result, allow_nan=False)
    assert key_recall([["red", "red"], ["green"]], [["red", "blue"], ["green"]]) == 0.75
    assert evaluate_predictions([["red"]], [["red"]], keywords=[[]])["key_recall"] is None
    with pytest.raises(ValueError):
        key_recall([["red"]], [])


def test_invalid_similarity_is_not_silently_accepted():
    with pytest.raises(ValueError):
        similarity_from_table({("a", "b"): 1.1})
    with pytest.raises(ValueError):
        wups_score(["a"], ["b"], lambda _a, _b: float("nan"))
