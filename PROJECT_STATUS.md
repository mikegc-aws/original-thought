# PROJECT STATUS — "Will AI ever be able to have original ideas?"

> **This file is the single source of truth for where the project stands.**
> Update it at the end of every working session. Resume by reading this file top to bottom.

**Last updated:** 2026-07-09 (Phase 1 complete)

## Current state

- **Active phase:** Phases 1 & 2 COMPLETE ✅ — next up: Phase 3 (small models on EC2 GPU).
- **Key results so far:**
  - Phase 1: Mike's mode-collapse observation replicated and explained — collapse ratio 0.7–1.0 at bare-prompt/default settings, recoverable to 0.04–0.10 with random seed words at no coherence cost. See `docs/02_frontier_study.md`.
  - Phase 2: median model "idea" is about as far from existing ideas as real ideas are from each other (ordinary recombination). But Nova Pro's pet idea is a near-duplicate of the YC company Anoria while claiming "never been done" — models can't verify their novelty claims. Embedding novelty is only a proxy → Phase 3 is the load-bearing experiment. See `docs/03_novelty_analysis.md`.
- **AWS:** account 680624154995, region `us-east-1`, role `Admin/mikegc-Isengard`.
- **EC2 GPU:** ⚠️ NOT YET PROVISIONED. Will be created for Phase 3. Always check it is **stopped** when pausing work: `scripts/gpu_down.sh` (to be written in Phase 3).
- **Spend so far:** ~$5–10 Bedrock inference (1,000 generations + 1,000 embeddings + 300 Opus judge calls).

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

### Phase 1 — Frontier-model study (Bedrock, ~$5–15) ✅ COMPLETE
- [x] `src/original_thought/bedrock.py` — unified invoke wrapper (Converse API)
- [x] `experiments/phase1_generate.py` — 4 models × 5 configs × 50 samples = 1,000 generations, 0 failures
- [x] `experiments/phase1_analyze.py` — Titan v2 embeddings, clustering, Vendi score, collapse ratio, heatmaps
- [x] `experiments/phase1_judge.py` — Opus 4.5 coherence/value judging (M4), 293/300 parsed
- [x] `docs/02_frontier_study.md` — findings: H1 & H2 confirmed; collapse is a tuning/sampling artifact

Models (inference profile IDs):
- `us.anthropic.claude-sonnet-4-6`
- `global.anthropic.claude-haiku-4-5-20251001-v1:0`
- `us.amazon.nova-pro-v1:0`
- `us.meta.llama3-3-70b-instruct-v1:0`

Embeddings: `amazon.titan-embed-text-v2:0`.

### Phase 2 — Novelty vs reference corpus ✅ COMPLETE
- [x] `experiments/phase2_corpus.py` — 9,445 real ideas (YC companies + arXiv cs.HC/cs.CY), Titan-v2 embedded
- [x] `experiments/phase2_analyze.py` — NN novelty vs human baseline (median 0.548), extremes for reading
- [x] `docs/03_novelty_analysis.md` — findings F5–F8; H3 mostly refuted; Nova≈Anoria duplicate found

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
- **2026-07-09 (cont.)** — Phase 0 definitions doc written (`docs/01_definitions.md`, hypotheses H1–H5 pre-registered). Phase 1 fully executed: 1,000 generations, embedding+cluster analysis, Opus-judge quality scoring, findings written up. Gotcha discovered: Claude 4.5+ on Bedrock rejects temperature+topP together (wrapper omits topP for Anthropic models).
- **2026-07-09 (cont.)** — Phase 2 executed end-to-end: corpus built (YC via yc-oss API, arXiv via export API w/ 3s politeness delay), embedded, NN-novelty computed with human-baseline calibration, findings written. Spend now ≈ $6–12 total. Next session: Phase 3 — write gpu_up/gpu_down scripts, design synthetic combinational domain, train scale sweep on EC2 GPU.
