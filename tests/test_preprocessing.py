import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mgn.preprocessing import (
    ChineseBERTExtractor, I3DExtractor, XLSRExtractor, pool_sequence,
    sample_clip_indices, sha256_path, write_feature_cache,
)


def test_video_sampling_stride_tail_and_limit():
    assert sample_clip_indices(1) == [[0] * 16]
    clips = sample_clip_indices(33)
    assert clips[0] == list(range(0, 32, 2))
    assert clips[1] == [32] * 16
    assert len(sample_clip_indices(1000, max_clips=20)) == 20
    with pytest.raises(ValueError):
        sample_clip_indices(0)


def test_pool_sequence_uses_adaptive_mean_intervals():
    features = torch.tensor([[0.0], [2.0], [4.0], [6.0], [8.0]])

    assert torch.equal(pool_sequence(features, 2), torch.tensor([[2.0], [6.0]]))
    assert torch.equal(pool_sequence(torch.tensor([[2.0, 5.0]]), 3), torch.tensor([[2.0, 5.0]] * 3))
    with pytest.raises(ValueError):
        pool_sequence(torch.empty(0, 3))


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
def test_cache_roundtrip_metadata_and_hash(tmp_path, suffix):
    metadata = {"extractor": "controlled-test-fixture", "checkpoint": "fixture-only", "output_layer": "fixture",
                "preprocessing": {"test": True}}
    path = tmp_path / ("cache" + suffix)
    reference = write_feature_cache(path, torch.arange(12).reshape(4, 3), metadata)
    with path.open("rb") as stream:
        loaded = np.load(stream, allow_pickle=False)
        if suffix == ".npz":
            with loaded:
                array = loaded["features"]
        else:
            array = loaded
    assert array.dtype == np.float32
    np.testing.assert_array_equal(array, np.arange(12).reshape(4, 3))
    assert reference["metadata"]["cache_sha256"] == sha256_path(path)
    assert reference == json.loads(path.with_suffix(suffix + ".json").read_text())
    assert "shape" not in metadata


def test_cache_rejects_invalid_provenance_and_values(tmp_path):
    with pytest.raises(ValueError, match="provenance"):
        write_feature_cache(tmp_path / "x.npy", np.ones((2, 3)), {})
    meta = {"extractor": "test", "checkpoint": "test", "output_layer": "test", "preprocessing": {"test": True}}
    with pytest.raises(ValueError, match="finite"):
        write_feature_cache(tmp_path / "x.npy", np.full((2, 3), np.nan), meta)
    assert not (tmp_path / "x.npy").exists()


def test_checkpoint_hash_changes_when_weight_contents_change(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"one")
    first = sha256_path(tmp_path)
    weights.write_bytes(b"two")
    assert sha256_path(tmp_path) != first


class _I3DStub(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, clips):
        return clips.mean(dim=(1, 2, 3, 4)).unsqueeze(1).expand(-1, self.dimension)


def test_i3d_torchscript_boundary_rejects_incorrect_dimension(tmp_path):

    clips = torch.zeros(1, 3, 16, 224, 224)
    for dimension in (1024, 2048):
        path = tmp_path / f"fixture-{dimension}.pt"
        torch.jit.script(_I3DStub(dimension)).save(str(path))
        extractor = I3DExtractor(path, output_layer="test-fixture-not-I3D")
        if dimension == 1024:
            with pytest.raises(ValueError, match="2048"):
                extractor(clips)
        else:
            assert extractor(clips).shape == (1, 2048)


def test_xlsr_uses_extract_features_instead_of_transformer_hidden_state():

    extractor = XLSRExtractor.__new__(XLSRExtractor)
    extractor.device = torch.device("cpu")
    extractor.pooled_length = 100
    extractor.processor = lambda waveform, **kwargs: {"input_values": torch.from_numpy(waveform).unsqueeze(0)}
    extractor.model = lambda **kwargs: SimpleNamespace(extract_features=torch.full((1, 4, 512), 3.0),
                                                     last_hidden_state=torch.full((1, 4, 1024), 9.0))
    output = extractor(np.zeros(1600, dtype=np.float32))
    assert output.shape == (100, 512)
    assert torch.all(output == 3)
    with pytest.raises(ValueError, match="16kHz"):
        extractor(np.zeros(1600, dtype=np.float32), sampling_rate=48000)


def test_chinesebert_supplies_pinyin_and_excludes_special_tokens():
    extractor = ChineseBERTExtractor.__new__(ChineseBERTExtractor)
    extractor.device = torch.device("cpu")
    extractor.dataset = SimpleNamespace(tokenize_sentence=lambda sentence: (torch.tensor([101, 5, 6, 102]), torch.arange(32)))

    def model(**kwargs):
        assert kwargs["pinyin_ids"].shape == (1, 4, 8)
        assert kwargs["input_ids"].tolist() == [[101, 5, 6, 102]]
        assert kwargs["attention_mask"].tolist() == [[1, 1, 1, 1]]
        return SimpleNamespace(last_hidden_state=torch.arange(4).reshape(1, 4, 1).expand(1, 4, 768).float())

    extractor.model = model
    output = extractor("测试")
    assert output.shape == (2, 768)
    assert output[:, 0].tolist() == [1, 2]
