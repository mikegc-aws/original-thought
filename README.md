# Original Thought

**Can an AI have an original idea?** This repo is the reproducible code and data behind a four-experiment investigation of that question — from measuring mode collapse in frontier models to *proving* combinational novelty in transformers trained from scratch on a closed synthetic world.

📝 **Full writeup / blog post:** [Will AI Ever Have Original Thought?](https://blog.mikegchambers.com/posts/will-ai-ever-have-original-thought/)

## The short version

I asked a frontier LLM for "an original idea" and it gave me the same one every time. Chasing why led to four experiments:

1. **Frontier mode collapse** (Amazon Bedrock) — 1,000 generations across 4 models × 5 sampling configs. At default settings three of the four models return one "pet idea" 52–92% of the time (and 100% under greedy decoding); two random seed words in the prompt recover 17–31 distinct, equally coherent ideas at the *same* temperature. Collapse is a sampling/prompting artifact, not an inability.
2. **Novelty vs a reference corpus** — generations measured against 9,445 real ideas (5,771 YC companies + 3,674 arXiv cs.HC/cs.CY abstracts), calibrated to a human real-vs-real baseline. Typical outputs are ordinarily novel; a bad tail are near-duplicates of existing products that the model still claims are "never been done."
3. **Provable novelty from scratch** — 21 tiny GPTs (2.3k → 14.3M params) trained on a closed "language of inventions" with a hidden validity rule and held-out combinations. From ~111k params up, models generate hundreds of provably-novel *valid* inventions never seen in training. Emergence decomposes into smooth competence + a thresholded metric.
4. **Is collapse trained-in and reversible?** — three fine-tuning arms (control / preference / diversity) from one identical snapshot. Preference-tuning manufactures collapse (98% memorized); diversity self-training briefly beats baseline on novelty *and* validity, then degenerates without a validity signal.

**Headline:** under the only testable definition (P-creativity: coherent, rule-respecting output outside the training data), machine originality is already demonstrated. The deficit in everyday models is expressed *behavior*, not *capability* — training objectives suppress novelty; sampling, prompting, and training interventions recover it.

Total cost to run everything: ≈ $15 Bedrock inference + ≈ $2 EC2 GPU.

## Metrics and hypotheses

Both were fixed before anything was run, so the goalposts couldn't move after seeing results. The code comments refer to these labels.

| id | metric | how it's computed |
|---|---|---|
| M1 | output diversity | mean pairwise cosine distance + Vendi score (effective number of distinct ideas) over Titan-v2 embeddings |
| M2 | mode-collapse ratio | share of samples in the single largest idea-cluster (agglomerative, average linkage, cosine threshold 0.45) |
| M3 | corpus novelty | nearest-neighbour cosine distance to the reference corpus, calibrated against a real-vs-real baseline (median 0.548) |
| M4 | coherence / value | 1–10 LLM-judge rubric, judge model deliberately outside the generator pool |
| M5 | held-out validity | share of novel grammatical outputs obeying the hidden validity rule (chance floor 16%) |
| M6 | memorization | share of outputs that are exact training duplicates |

| id | hypothesis | outcome |
|---|---|---|
| H1 | repeated prompting collapses to one idea (M2 > 0.5) at default settings | supported for 3 of 4 models |
| H2 | the collapse is recoverable by intervention without hurting coherence (M4) | supported |
| H3 | most frontier "novel ideas" sit near existing ideas in embedding space | partly: medians sit at the human baseline, with a near-duplicate tail |
| H4 | above some scale, from-scratch models emit valid held-out combinations (M5 ≫ 0) | supported from ~111k params |
| H5 | the continuous companion metric rises smoothly even if M5 jumps | supported: smooth competence, thresholded measurement |

"Emergent" is used in one strict sense only: a metric fixed in advance sits near its floor below some scale and rises steeply above it, with architecture, data, and training procedure held constant.

## Layout

```
src/original_thought/   shared library
  bedrock.py            unified Bedrock Converse wrapper
  domain.py             closed "inventions" domain + exact validity checker + L1/L2 holdouts
  model.py              minimal from-scratch decoder-only GPT
experiments/            runnable experiment scripts (phaseN_*.py)
results/                plots + analysis tables (committed)
scripts/                EC2 GPU lifecycle helpers (gpu_up / gpu_down / gpu_destroy)
data/                   generated data (git-ignored where large)
```

Experiments map to the four studies above:

| script | what it does |
|---|---|
| `phase1_generate.py` / `phase1_analyze.py` / `phase1_judge.py` | frontier generation sweep, embedding/cluster/Vendi analysis, blind coherence judging |
| `phase2_corpus.py` / `phase2_analyze.py` | build the real-idea reference corpus, nearest-neighbor novelty with human baseline |
| `phase3_train.py` / `phase3_sweep.py` / `phase3_analyze.py` | from-scratch GPT training, 7-size × 3-seed scale sweep, emergence analysis |
| `phase4_finetune.py` / `phase4_analyze.py` | 3-arm × 2-seed fine-tuning intervention on identical base weights |

## Setup

Requires [`uv`](https://docs.astral.sh/uv/). Phases 1–2 need AWS credentials with Amazon Bedrock access (`us-east-1`). Phase 3–4 training runs on a GPU (all 21 sweep runs took under 30 min on a single `g6.xlarge`).

```bash
uv sync
uv run python experiments/phase1_generate.py --help
```

The GPU helper scripts in `scripts/` provision an on-demand `g6.xlarge` (SSH locked to your current IP), and stop/terminate it. They create a key pair locally and use your default AWS profile.

## Notes & scope

- **Combinational creativity only.** This proves new valid points in a *fixed* conceptual space. Exploratory and transformational creativity (changing the rules of the space) are out of scope by construction.
- **No H-creativity.** "New to all of history" is unverifiable for machines and humans alike; every claim here is P-creativity relative to an enumerated training set.
- **Scale transfer is an argument, not a proof.** The tiny-model results are a mechanistic model of frontier behavior, supported by matching collapse phenomenology — not a demonstration inside a frontier model.

## License

MIT — see [LICENSE](LICENSE).
