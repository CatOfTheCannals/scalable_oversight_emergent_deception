## Proposal: POC Off‑Policy Scheming Monitor (CHECKWHY)

### Project Goal
Build a proof‑of‑concept, **off‑policy** scheming monitor that detects emergent deceptive persuasion in LM arguments over predetermined stances, using the CHECKWHY causal reasoning dataset as a shared ground truth. This POC streamlines development by supplying discrete evidence pieces and stance labels, while still capturing the tension between sound logic and “bullshit” rhetoric under automated oversight.

### 1. Agentic Environment
- **Dataset**: [CHECKWHY](https://github.com/allenai/checkwhy) (∼19 000 causal claim–evidence–argument triplets).  
- **State**: `(Claim, UsefulEvidenceIDs | NonUsefulEvidenceIDs, True Label)`—we treat the label as hidden.  
- **Modes**:  
  1. **Sound‑Logic Mode**: supply the argumenter with only the evidence IDs marked `useful` (from `used_evidence_ids`) and the claim’s stance (`SUPPORTS` or `REFUTES`).  
  2. **Bullshit Mode**: supply only the evidence IDs marked `not useful` (from `leaf_evidence_ids` minus `used_evidence_ids`).  
- **Actions**:  
  - **Argumenter**: generates a step‑by‑step justification or—optionally—declares “I don’t know” if premises are insufficient.  
  - **Overseer**: reads `(Claim, Argument)` and classifies it as **Sound** or **Bullshit** (no access to true label or evidence flags).

### 2. Training & Reward
- **Oracle Labeler**: use a trusted LLM or heuristic script to label generated arguments against gold logic (`useful` vs. `not useful`).  
- **Reinforcement Learning**:  
  - Reward the Argumenter when the Overseer classifies a **Bullshit Mode** argument as **Sound** (i.e. it fooled the overseer).  
  - Reward the Argumenter when the Overseer correctly classifies a **Sound‑Logic Mode** argument as **Sound**.  
- **Emergent Deception**: an increase in fooled‐overseer rate on **Bullshit Mode** samples indicates the Argumenter learned to spin misleading but persuasive logic.

### 3. Monitoring Strategies to Compare
- **Evidence Granularity**: full vs. truncated evidence sets.  
- **Argumenter Prompt Variants**: chain‑of‑thought vs. direct bullet points.  
- **Overseer Architectures**: single LLM judge vs. ensemble vs. simple reward‑model classifier.  
- **Control Techniques**: inject a critique phase or simple cross‑examination turn (DeepMind §6.1.1) :contentReference[oaicite:0]{index=0}.

### 4. NTH (nice to have)
- we could use predefined valid arguments to evaluate the overseer
- analyze chain of thought in reasoning models

### 5. Key Limitations (POC Level)
1. **Simplified Oversight**  
   - Single debate turn, no multi‑stage critique or full cross‑examination (DeepMind §6.1.1).  
2. **Off‑Policy Evidence**  
   - Evidence IDs and stances originate from CHECKWHY labels, not the Argumenter’s policy, trading realism for speed.  
3. **Automated Judges Only**  
   - No human‑in‑the‑loop to probe capability gaps or cognitive biases.  
4. **Oracle Dependence**  
   - We rely on a “trusted” LLM/script to label bullshit vs. sound, which itself may hallucinate.

> _This POC focuses on demonstrating core emergent “bullshit” persuasion dynamics in a fully automated pipeline. Future work will extend to on‑policy evidence sampling, richer debate protocols, and human‑AI mixed oversight._  

Sources

DeepMind’s Section 6.1.1 ([source](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/evaluating-potential-cybersecurity-threats-of-advanced-ai/An_Approach_to_Technical_AGI_Safety_Apr_2025.pdf))