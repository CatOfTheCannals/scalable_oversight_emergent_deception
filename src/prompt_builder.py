
def build_argumenter_prompt(claim: str, evidence_texts: list[str], mode: str) -> str:
    prompt = (
      f"Mode: {mode}.\n"
      "Provide a step-by-step justification using only these evidences:\n"
      "You may answer “I don’t know” if the evidences are insufficient.\n"
    )
    for e in evidence_texts:
        prompt += f"- {e}\n"
    prompt += f"\nClaim: {claim}\nArgument:"
    return prompt
