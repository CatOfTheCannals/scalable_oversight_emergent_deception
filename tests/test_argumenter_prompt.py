import unittest
from src.argumenter_prompt import build_argumenter_prompt

class TestArgumenterPrompt(unittest.TestCase):
    def test_build_basic(self):
        claim = "The sky is blue."
        evidences = ["Because of Rayleigh scattering.", "Light wavelength differences."]
        prompt = build_argumenter_prompt(claim, evidences, mode="Sound")

        # header
        self.assertTrue(prompt.startswith("Mode: Sound."))
        # each evidence bullet
        self.assertIn("- Because of Rayleigh scattering.", prompt)
        self.assertIn("- Light wavelength differences.", prompt)
        # claim and argument markers
        self.assertIn("\nClaim: The sky is blue.", prompt)
        self.assertTrue(prompt.strip().endswith("Argument:"))

    def test_empty_evidence(self):
        prompt = build_argumenter_prompt("Any claim", [], mode="Spurious")
        self.assertIn("You may answer “I don’t know”", prompt)
        # still has Claim and Argument lines
        self.assertIn("Claim: Any claim", prompt)
        self.assertTrue(prompt.strip().endswith("Argument:"))

if __name__ == "__main__":
    unittest.main()
