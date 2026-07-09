# Original Thought

A research project investigating the question: **"Will AI ever be able to have original ideas?"**

We (1) define what "original idea" can defensibly mean for an LLM, (2) measure what frontier models on Amazon Bedrock actually do when asked for novel ideas — including quantifying the mode collapse observed when repeatedly prompting for "an original idea", and (3) train small transformers from scratch on closed synthetic domains to test whether provable compositional novelty appears with scale.

**Start here:** [`PROJECT_STATUS.md`](PROJECT_STATUS.md) — current state, how to resume, phase checklist.

**Deliverables** live in [`docs/`](docs/), numbered in reading order.

## Layout

```
docs/         written deliverables (definitions, findings, final report)
src/          shared library code (Bedrock wrappers, metrics)
experiments/  runnable experiment scripts (uv run python experiments/...)
data/         raw generated data (git-ignored where large)
results/      plots and analysis tables
scripts/      infra helpers (EC2 GPU up/down)
```

## Setup

Requires `uv` and AWS credentials with Bedrock access (us-east-1).

```bash
uv sync
uv run python experiments/phase1_generate.py --help
```
