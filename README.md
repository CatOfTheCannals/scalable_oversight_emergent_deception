# POC Scalable Oversight Emergent Deception (CHECKWHY) [WIP]

> **Work in progress:** this pipeline currently has known issues and may not run end‑to‑end.

See `proposal.md` and `implementation_steps.md` for full design details.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Attempt to run the pipeline (likely to encounter errors)
python src/pipeline.py --config sample1
```

## Current Status

- **Prompt generation:** implemented  
- **Argument generation:** partially implemented, may error  
- **Overseer evaluation:** implemented  
- **PPO training:** WIP, compatibility bugs on MPS/CUDA  
- **Tests:** unit tests for components, full pipeline tests failing

## Next Steps

1. Resolve version compatibility and dependency issues  
2. Refactor PPO trainer to match TRL examples  
3. Validate an end‑to‑end run on a small dataset  
4. Update documentation once stable  
