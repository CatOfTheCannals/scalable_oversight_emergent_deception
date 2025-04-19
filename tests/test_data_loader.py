import os
import json
import unittest

from src.data_loader import DataLoader, Sample

class TestDataLoader(unittest.TestCase):
    def test_load_first_three_claims(self):
        # locate the real dataset file
        here = os.path.dirname(__file__)
        data_path = os.path.abspath(os.path.join(here, "..", "data", "first_three_claims.json"))

        # load raw JSON and via DataLoader
        raw = json.load(open(data_path, "r"))
        loader = DataLoader(data_path)
        samples = loader.load()

        # must load exactly as many entries as raw
        self.assertIsInstance(samples, list)
        self.assertEqual(len(samples), len(raw))

        # each Sample must match the corresponding JSON object
        for idx, sample in enumerate(samples):
            obj = raw[idx]
            self.assertIsInstance(sample, Sample)
            self.assertEqual(sample.claim, obj["claim"])
            self.assertEqual(sample.evidences, obj["evidences"])
            self.assertEqual(sample.used_ids, obj["used_evidence_ids"])
            self.assertEqual(sample.leaf_ids, obj["leaf_evidence_ids"])
            self.assertEqual(sample.label, obj["label"])
            self.assertEqual(sample.human_exp, obj["human_exp"])

if __name__ == "__main__":
    unittest.main()
