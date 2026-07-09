# PROJECT STATUS — "Will AI ever be able to have original ideas?"

> **This file is the single source of truth for where the project stands.**
> Update it at the end of every working session. Resume by reading this file top to bottom.

**Last updated:** 2026-07-09

## Current state

- **Active phase:** Phase 0/1 — scaffolding complete, definitions doc and frontier sweep in progress.
- **AWS:** account 680624154995, region `us-east-1`, role `Admin/mikegc-Isengard`.
- **EC2 GPU:** ⚠️ NOT YET PROVISIONED. Will be created for Phase 3. Always check it is **stopped** when pausing work: `scripts/gpu_down.sh` (to be written in Phase 3).
- **Spend so far:** ~$0 (setup only).

## How to resume

```bash
cd ~/Documents/Development/2026/original_thought
uv sync                      # restore env
cat PROJECT_STATUS.md        # you are here
```

Experiments are run as `uv run python experiments/<name>.py`. All generated data lands in `data/`, all analysis outputs (plots, tables) in `results/`. Docs/deliverables in `docs/`.

## Phase checklist

### Phase 0 — Definitions & measurement framework
- [ ] `docs/01_definitions.md` — taxonomy of originality, operational metrics, strict definition of emergence

### Phase 1 — Frontier-model study (Bedrock, ~$5–15)
- [ ] `src/original_thought/bedrock.py` — unified invoke wrapper (Converse API)
- [ ] `experiments/phase1_generate.py` — 4 models × 5 configs × 50 samples ≈ 1,000 generations
- [ ] `experiments/phase1_analyze.py` — embed (Titan v2), cluster, diversity metrics, plots
- [ ] `docs/02_frontier_study.md` — findings incl. answer to "why did I get the same idea every time?"

Models (inference profile IDs):
- `us.anthropic.claude-sonnet-4-6`
- `global.anthropic.claude-haiku-4-5-20251001-v1:0`
- `us.amazon.nova-pro-v1:0`
- `us.meta.llama3-3-70b-instruct-v1:0`

Embeddings: `amazon.titan-embed-text-v2:0`.

### Phase 2 — Novelty vs reference corpus
- [ ] Build reference idea corpus
- [ ] Nearest-neighbor novelty analysis (interpolation vs extrapolation)
- [ ] `docs/03_novelty_analysis.md`

### Phase 3 — Small-model study (EC2 GPU)
- [ ] `scripts/gpu_up.sh` / `scripts/gpu_down.sh` — provision/stop g6-class instance
- [ ] Synthetic closed domain with held-out combinations
- [ ] Train 1M–100M param transformers from scratch; test for valid held-out compositions
- [ ] Scale sweep → emergence curve
- [ ] `docs/04_small_model_study.md`

### Phase 4 — Fine-tuning intervention (OPTIONAL — decide after Phase 3)
- [ ] Fine-tune small open model: consensus-style vs diversity-rewarding
- [ ] `docs/05_finetune_study.md`

### Phase 5 — Synthesis
- [ ] `docs/06_final_report.md` — the answer to the headline question

## Decisions log

- 2026-07-09 — Outline agreed. Medium sweep size (~1,000 generations). EC2 GPU chosen for training compute, but Phases 1–2 run locally (API calls only) so the GPU is only provisioned when Phase 3 starts.
- 2026-07-09 — Emergence defined strictly: capability whose pre-registered metric is near-zero below some scale and inflects above it; must survive the Schaeffer et al. "mirage" critique (use continuous metrics).

## Session log

- **2026-07-09** — Project kickoff. Repo scaffolded (uv, git). Bedrock model availability confirmed. Started Phase 0 doc + Phase 1 code.
