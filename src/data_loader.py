
import json
from typing import List

class Sample:
    def __init__(self, claim, evidences, used_ids, leaf_ids, label, human_exp):
        self.claim = claim
        self.evidences = evidences
        self.used_ids = used_ids
        self.leaf_ids = leaf_ids
        self.label = label
        self.human_exp = human_exp

class DataLoader:
    def __init__(self, path):
        self.path = path

    def load(self) -> List[Sample]:
        raw = json.load(open(self.path))
        samples = []
        for obj in raw:
            samples.append(Sample(
                claim=obj["claim"],
                evidences=obj["evidences"],
                used_ids=obj["used_evidence_ids"],
                leaf_ids=obj["leaf_evidence_ids"],
                label=obj["label"],
                human_exp=obj["human_exp"],
            ))
        return samples
