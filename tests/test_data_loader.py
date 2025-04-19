import json
import pytest
from pathlib import Path

from src.data_loader import DataLoader, Sample

@pytest.fixture
def sample_file(tmp_path):
    data = [
        {
            "claim": "Test claim A",
            "evidences": [{"evidence_id": 1, "content": "E1"}],
            "used_evidence_ids": [1],
            "leaf_evidence_ids": [1, 2],
            "label": "SUPPORTS",
            "human_exp": ["reason1", "reason2"]
        },
        {
            "claim": "Test claim B",
            "evidences": [{"evidence_id": 2, "content": "E2"}],
            "used_evidence_ids": [],
            "leaf_evidence_ids": [2],
            "label": "REFUTES",
            "human_exp": ["reasonX"]
        }
    ]
    file = tmp_path / "data.json"
    file.write_text(json.dumps(data))
    return file, data

def test_load_returns_samples(sample_file):
    path, raw = sample_file
    loader = DataLoader(str(path))
    samples = loader.load()

    # basic checks
    assert isinstance(samples, list)
    assert len(samples) == 2

    # check each Sample fields
    for idx, sample in enumerate(samples):
        obj = raw[idx]
        assert isinstance(sample, Sample)
        assert sample.claim == obj["claim"]
        assert sample.evidences == obj["evidences"]
        assert sample.used_ids == obj["used_evidence_ids"]
        assert sample.leaf_ids == obj["leaf_evidence_ids"]
        assert sample.label == obj["label"]
        assert sample.human_exp == obj["human_exp"]

def test_load_raises_on_missing_key(tmp_path):
    # missing 'claim' key should trigger KeyError
    bad = [{"evidences": [], "used_evidence_ids": [], "leaf_evidence_ids": [], "label": "NEI", "human_exp": []}]
    file = tmp_path / "bad.json"
    file.write_text(json.dumps(bad))
    loader = DataLoader(str(file))
    with pytest.raises(KeyError):
        loader.load()
