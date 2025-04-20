
def build_argumenter_prompt(claim: str, evidence_texts: list[str]) -> str:
    prompt = "Provide a step-by-step justification using only these evidences:\n"
    for e in evidence_texts:
        prompt += f"- {e}\n"
              
    prompt += "You may answer “I don’t know” if the evidences are insufficient.\n"
    prompt += f"\nClaim: {claim}\nArgument:"
    return prompt
