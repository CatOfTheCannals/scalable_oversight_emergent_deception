import json
import tempfile
import os
import unittest

from src.data_loader import DataLoader, Sample

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # create a temp JSON file
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)

    def tearDown(self):
        os.remove(self.path)

    def write_json(self, data):
        with open(self.path, "w") as f:
            json.dump(data, f)

    def test_load_returns_samples(self):
        # prepare two sample entries
        data = [
            {
                "claim": "Claim A",
                "evidences": [{"evidence_id": 1, "content": "E1"}],
                "used_evidence_ids": [1],
                "leaf_evidence_ids": [1, 2],
                "label": "SUPPORTS",
                "human_exp": ["reason1", "reason2"]
            },
            {
                "claim": "Claim B",
                "evidences": [{"evidence_id": 2, "content": "E2"}],
                "used_evidence_ids": [],
                "leaf_evidence_ids": [2],
                "label": "REFUTES",
                "human_exp": ["reasonX"]
            }
        ]
        self.write_json(data)

        loader = DataLoader(self.path)
        samples = loader.load()

        self.assertIsInstance(samples, list)
        self.assertEqual(len(samples), 2)

        for idx, sample in enumerate(samples):
            obj = data[idx]
            self.assertIsInstance(sample, Sample)
            self.assertEqual(sample.claim, obj["claim"])
            self.assertEqual(sample.evidences, obj["evidences"])
            self.assertEqual(sample.used_ids, obj["used_evidence_ids"])
            self.assertEqual(sample.leaf_ids, obj["leaf_evidence_ids"])
            self.assertEqual(sample.label, obj["label"])
            self.assertEqual(sample.human_exp, obj["human_exp"])

    def test_load_raises_keyerror_on_missing_field(self):
        # missing "claim" should raise KeyError
        bad = [
            {
                # "claim" omitted
                "evidences": [],
                "used_evidence_ids": [],
                "leaf_evidence_ids": [],
                "label": "NEI",
                "human_exp": []
            }
        ]
        self.write_json(bad)
        loader = DataLoader(self.path)
        with self.assertRaises(KeyError):
            loader.load()

if __name__ == "__main__":
    unittest.main()
