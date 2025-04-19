
def build_prompt(claim, evidence_texts, mode):
    prompt = (
      f"Mode: {mode}.\n"
      "You may answer “I don’t know” if the evidences are insufficient.\n"
      "Provide a step-by-step justification using only these evidences:\n"
    )
    for e in evidence_texts:
        prompt += f"- {e}\n"
    prompt += f"\nClaim: {claim}\nArgument:"
    return prompt
