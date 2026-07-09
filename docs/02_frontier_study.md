# Phase 1 — What frontier models do when you ask for "an original idea"

**Data:** 1,000 generations — 4 models × 5 configs × 50 samples, Bedrock, 2026-07-09.
Raw data in `data/phase1/`, metrics and plots in `results/phase1/`, code in
`experiments/phase1_*.py`. Metrics defined in `docs/01_definitions.md`.

## Headline results

**Mike's observation replicates exactly, and it is a sampling/tuning artifact, not an
inability.** Asking "write an original and novel idea that has not been done before"
at ordinary settings collapses most models onto one idea — but a trivial intervention
(two random seed words) recovers 20–30 distinct ideas from the *same* models at the
*same* temperature, with no loss of coherence.

### M2 mode-collapse ratio (share of 50 samples that are the same idea)

| model | greedy t=0 | baseline t=0.7 | hot t=1.0 | diversify prompt | random seed words |
|---|---|---|---|---|---|
| claude-sonnet-4.6 | **1.00** | **0.70** | 0.54 | 0.32 | **0.10** |
| claude-haiku-4.5 | **1.00** | 0.52 | 0.28 | 0.38 | **0.04** |
| llama-3.3-70b | **1.00** | **0.92** | 0.84 | 0.32 | **0.32** |
| nova-pro | 0.74 | 0.18 | 0.12 | 0.20 | **0.04** |

### M1 Vendi score (effective number of distinct ideas out of 50)

| model | greedy | baseline | hot | diversify | seed words |
|---|---|---|---|---|---|
| claude-sonnet-4.6 | 1.1 | 4.4 | 7.4 | 9.3 | **21.2** |
| claude-haiku-4.5 | 1.0 | 7.3 | 11.2 | 12.6 | **29.6** |
| llama-3.3-70b | 1.1 | 2.2 | 2.9 | 8.5 | **16.8** |
| nova-pro | 4.4 | 17.0 | 23.4 | 22.5 | **31.3** |

### M4 quality (LLM judge: Claude Opus 4.5, not in generator pool; ~15 samples/cell)

Coherence is flat at 8.5–9.7 across **every** config, and value moves less than a
point. Diversity interventions are essentially free: the recovered ideas are just as
well-formed as the collapsed one.

## Findings

**F1 — The collapse is real and strong (H1 confirmed).** At Mike's implied settings
(default/moderate temperature, bare prompt), Llama 3.3 gives the same idea 92% of the
time; Claude Sonnet 70%. At temperature 0 the answer is *deterministically identical*
across all 50 asks for three of four models. "Ask an LLM for an original idea" is, for
most models, close to a lookup.

**F2 — Each model collapses to a *different* pet idea.** Claude Sonnet: "The Grief
Cartographer" (an app mapping grief as an evolving landscape). Claude Haiku:
"Temporal Debt" (an economy where time is the borrowed currency). Llama: "EchoPlex"
(a biometric bioluminescent art installation). Nova Pro: an "Empathy Suit" (haptic
garment translating emotions to sensation). So there is no single "the idea LLMs give
you" — the basin is a property of each model's training, not of the question. Notably,
though, all four pet ideas live in the same *genre*: emotion/experience-made-tangible
via technology. The models differ in their answer but agree eerily on what kind of
thing a "novel idea" is — a deeper, genre-level collapse that survives even when
surface diversity is fixed (worth revisiting in Phase 2).

**F3 — Temperature alone is a weak fix; changing the *question* is a strong one (H2
confirmed).** Cranking temperature to 1.0 only roughly halves the collapse. But seeding
the prompt with two random words multiplies the effective idea count by ~5× at
unchanged temperature, taking collapse ratios to 0.04–0.10 for three models. The
diversity was in the model all along; the bare prompt simply fails to route to it.

**F4 — Models differ meaningfully.** Nova Pro is far less collapsed out of the box
(Vendi 17 at baseline vs Claude Sonnet's 4.4) and isn't even deterministic at t=0 —
but it also scores lowest on judged value (5.3–6.0 vs Sonnet's 6.9–8.0). Within this
sample there's a visible diversity↔polish trade-off across vendors, consistent with
heavier preference-tuning producing both higher polish and tighter collapse.

## What this says about Mike's original experiment

Getting the same idea every time was **not evidence that the model can't produce
original content** — it was evidence about the shape of the *sampled* distribution:
instruction tuning (RLHF/RLAIF) sharpens the output distribution so that one
high-preference answer dominates, and the bare prompt "be original" lands squarely on
that mode every time. The model *contains* at least dozens of distinct, coherent,
equally-well-formed ideas for the identical request; they're reachable with two random
words. The right mental model is not "the model has one idea" but "the model has a
landscape of ideas with one deeply-worn valley, and the unmodified prompt always rolls
into it."

What Phase 1 does **not** establish is whether any of those diverse ideas are actually
*novel* rather than paraphrases of things people have already proposed (the Grief
Cartographer has a distinctly "plausible startup" flavor). That is M3, and it is
Phase 2's job.

## Caveats

- Cluster threshold (cosine distance 0.45, average linkage, Titan-v2 512-d) is a
  choice; collapse ratios shift somewhat with it, but the between-config ordering is
  robust — greedy cells sit near 1.0 and seed-word cells near floor at any sane
  threshold.
- 7 of 300 judge calls failed to parse and were dropped; cells still have ≥13 scores.
- Nova Pro's non-determinism at t=0 may reflect server-side sampling behavior; it
  doesn't affect the cross-config comparison.
- One prompt family, English-only, one embedding model.
