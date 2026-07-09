# Phase 2 — Are the ideas actually *novel*, or retrieval-with-paraphrase?

**Data:** all 1,000 Phase 1 generations vs a reference corpus of 9,445 real ideas
(5,600 YC company descriptions + 3,800 recent arXiv cs.HC/cs.CY abstracts), all in the
same Titan-v2 512-d embedding space. Metric M3 = cosine distance to nearest corpus
neighbor. Code: `experiments/phase2_corpus.py`, `phase2_analyze.py`; outputs in
`results/phase2/`.

**Calibration.** Raw distances are meaningless alone, so we measured how far a *real*
idea sits from its nearest other real idea (500 held-out corpus items vs the rest):
median **0.548** (p10 0.394, p90 0.657). That is the "as novel as real ideas are"
yardstick.

## Results

| model | config | median M3 | share below human median |
|---|---|---|---|
| nova-pro | greedy t=0 | **0.469** | **74%** |
| nova-pro | baseline | 0.570 | 38% |
| claude-sonnet-4.6 | baseline | 0.570 | 12% |
| claude-sonnet-4.6 | greedy | 0.564 | 0% |
| claude-haiku-4.5 | baseline | 0.612 | 2% |
| llama-3.3-70b | baseline | 0.668 | 0% |
| all models | random_seed_words | 0.64–0.67 | 0–6% |

(Full table: `results/phase2/novelty_summary.csv`.)

## Findings

**F5 — At least one "original idea" demonstrably already exists.** Nova Pro's
collapsed answer — an AI wearable that reads your emotions from biometrics and
communicates them — is, almost point for point, the YC company **Anoria** ("the first
wearable that reads your emotions to enhance your EQ"), at distance 0.40 vs the 0.55
human baseline. 74% of Nova's greedy generations sit closer to an existing real idea
than a typical real idea sits to *its* neighbors. When a model asserts "this has not
been done before," it has no mechanism for checking that claim — novelty in the prompt
is an instruction about *style*, not a constraint the model can verify.

**F6 — But most model ideas are NOT nearest-neighbor copies (H3 more wrong than
right).** We pre-registered H3 expecting most outputs to hug the corpus. In fact the
median generation sits *at or slightly beyond* the human-baseline distance for every
Claude/Llama cell (0.56–0.67 vs 0.548), and diversity interventions push distance up
further. By this proxy, typical frontier "ideas" are about as far from existing ideas
as existing ideas are from each other — ordinary-degree recombination, neither
plagiarism nor extrapolation.

**F7 — The "most novel" outputs drift out of genre.** Reading the extremes
(`results/phase2/extremes.txt`): the farthest-from-corpus generations are mostly
*fiction premises* (a lighthouse keeper whose bellows exhale the ocean's memories;
glacier-carved monasteries archiving lost civilizations) — while the corpus is products
and research. Some of the measured "novelty" is therefore genre mismatch, not conceptual
novelty within a genre. This is the fundamental limit of embedding-distance novelty
metrics, and the strongest argument for Phase 3, where novelty is *provable* rather
than proxied.

**F8 — Novelty and collapse are independent failure modes.** Llama is the *most*
collapsed model (0.92) yet its pet idea is far from the corpus (0.67); Nova is the
*least* collapsed yet closest to existing ideas. A model can be diverse-but-derivative
or repetitive-but-far-from-corpus. "Original" requires winning on both axes at once —
plus coherence — and no cell wins all three decisively.

## Honest limits of this phase

- **The corpus is a proxy, not the training set.** 9.4k ideas vs trillions of training
  tokens: a generation far from *our* corpus may still be near something the model
  read. M3 distances are an **upper bound** on novelty. This is exactly why Phase 3
  (closed training set, novelty provable by construction) is the load-bearing
  experiment of the project.
- Embedding distance conflates topic, genre, and style (see F7).
- Corpus skews toward tech/HCI because Phase 1 showed that's the genre models emit;
  ideas outside that genre are over-credited as novel.

## Verdict for the headline question so far

Frontier models asked for an original idea produce, at best, *ordinarily novel*
recombinations — comparable in corpus-distance to how real ideas differ from each
other — and at worst, near-duplicates of existing products confidently labeled as
"never been done before." Nothing observed so far distinguishes "having an original
idea" from "sampling a plausible point in idea-space" — for that distinction we need
the controlled setting of Phase 3.
