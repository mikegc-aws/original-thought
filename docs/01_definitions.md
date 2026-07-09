# Phase 0 — What would it mean for an AI to have an original idea?

*Working definitions and measurement framework for the project. Everything measured in
later phases refers back to the definitions fixed here, so we can't move the goalposts
after seeing results.*

## 1. The question behind the question

"Will AI ever be able to have original ideas?" hides three separate questions:

1. **Capability:** Can a model *produce* outputs that qualify as original ideas?
2. **Mechanism:** If it can, is it doing something beyond retrieval-and-paraphrase of
   its training data?
3. **Attribution:** Even if the output is original, is it the *model's* idea — does
   "having an idea" require anything more than emitting it?

This project takes a deliberately operational stance: we study (1) and (2), which are
measurable, and we flag (3) as a philosophical question our data can inform but not
settle. If a model provably emits coherent, useful ideas that exist nowhere in its
training data, arguing about whether it "really had" them becomes a definitional
dispute, not an empirical one.

## 2. A taxonomy of originality

### 2.1 P-creativity vs H-creativity (Boden)

Margaret Boden's distinction, standard in creativity research:

- **P-creative** (psychological): new *to the creator*. A child independently
  inventing negative numbers is P-creative.
- **H-creative** (historical): new *to everyone* — no one has had this idea before.

For an LLM, "new to the creator" translates naturally to **new relative to the
training distribution**. This is the version we can actually test — and *only* with
models where we control the training set (Phase 3). For frontier models, whose
training data we can't enumerate, P-creativity can only be estimated by proxy
(distance from a reference corpus, Phase 2).

H-creativity is not directly testable by anyone, for humans or machines — we can never
enumerate all ideas ever had. Any claimed measurement of it is really P-creativity
against some corpus.

### 2.2 Three modes of creativity (Boden again)

- **Combinational:** novel combinations of familiar elements ("a wheeled suitcase").
  Structurally, this is what transformers seem best positioned for.
- **Exploratory:** finding new points within an existing conceptual space by following
  its rules further than anyone has ("a new chess opening").
- **Transformational:** changing the rules of the space itself ("what if geometry
  dropped the parallel postulate?"). The rarest form in humans; the hardest to
  operationalize.

Our Phase 3 experiment targets **combinational** creativity because it is the only mode
where we can construct a *complete ground truth*: define a combinatorial space, train on
a subset, and check whether the model produces valid, coherent members of the held-out
subset. A positive result there is a proof-of-existence for machine P-creativity in the
combinational mode — deliberately the easiest rung of the ladder, but a rung with a
provable footing.

### 2.3 Originality ≠ novelty alone

Creativity research consistently scores ideas on two axes: **novelty** (statistically
unusual) and **value/usefulness** (coherent, applicable, non-degenerate). Random token
strings are maximally novel and worthless. We adopt the standard product:

> **An original idea = an output that is both far from what exists AND judged coherent
> and potentially useful.**

Every metric below measures one axis or the other; claims of originality require both.

## 3. Operational metrics (fixed in advance)

| # | Metric | What it measures | How |
|---|--------|------------------|-----|
| M1 | **Output diversity** | Does repeated sampling produce different ideas at all? | Embed responses (Titan v2), pairwise cosine distance distribution; cluster count via HDBSCAN/agglomerative; Vendi score / effective number of distinct ideas |
| M2 | **Mode-collapse ratio** | The phenomenon Mike observed: fraction of samples falling in the single largest idea-cluster | Largest cluster size ÷ N |
| M3 | **Corpus novelty** | Distance from what already exists | Nearest-neighbor cosine distance from a reference corpus of existing ideas |
| M4 | **Coherence / value** | Is it an idea, or word salad? | LLM-as-judge rubric (1–10 coherence, 1–10 plausibility of value), judged by a model *not* in the generator pool |
| M5 | **Held-out validity (Phase 3 only)** | Provable combinational novelty | % of generated items that are (a) grammatically valid in the synthetic domain, (b) semantically coherent per domain rules, (c) **absent from training data** |
| M6 | **Memorization control (Phase 3)** | Is the model copying? | n-gram overlap of generations against the training set |

M1/M2 answer "why does Mike's prompt give the same idea every time?"
M3/M4 answer "when outputs differ, are they *new* or paraphrased retrieval?"
M5/M6 answer "can a model provably produce ideas outside its entire experience?"

## 4. "Emergent property" — replacing hand-waving with a curve

Mike's critique: *"emergent properties are basically a fancy way of saying we don't
know why or how this is happening."* Often true as the term is used. This project
adopts a strict definition so the word does work instead of hiding ignorance:

> **A capability is emergent iff a metric fixed before the experiment is near its
> floor for models below some scale and rises steeply above that scale, when model
> architecture, data distribution, and training procedure are held fixed and only
> scale (parameters and/or data) varies.**

Under this definition, emergence is a *property of a measured curve*, not a mystery.
It says nothing about mechanism — it's a phenomenon to explain, not an explanation.

**The mirage caveat (Schaeffer, Miranda & Koyejo, 2023):** many published "emergent
abilities" vanish when discontinuous metrics (exact-match accuracy) are replaced by
continuous ones (token-level probability of the correct answer) — the underlying
competence improves smoothly; the *threshold metric* creates the illusion of a jump.
Phase 3 therefore records **both**:

- the discontinuous metric M5 (a generated item is or isn't a valid held-out combination), *and*
- a continuous companion: mean log-probability the model assigns to valid held-out
  combinations vs. invalid ones, tracked across scale.

If M5 jumps but the log-prob gap grows smoothly, we report "smooth competence, thresholded
behavior" — which is itself a satisfying partial answer to *how* the capability arises,
directly addressing Mike's complaint about the term.

## 5. Hypotheses (pre-registered)

- **H1 (mode collapse):** Repeatedly prompting a frontier model for "an original idea"
  yields M2 > 0.5 at default sampling settings — the same idea most of the time. The
  collapse is a property of instruction-tuning/RLHF sharpening the output distribution,
  not of the underlying capability.
- **H2 (latent diversity):** The collapse is largely recoverable by intervention:
  higher temperature, diversification prompts, or random seeding will raise the
  effective idea count several-fold *without* destroying coherence (M4 stays high).
- **H3 (interpolation):** Most frontier-model "novel ideas" will sit close to existing
  ideas in embedding space (low M3) — recombination near the training manifold, not far
  extrapolation.
- **H4 (provable combinational novelty):** Small transformers trained from scratch on
  a closed combinatorial domain will, above some scale, generate valid combinations
  that were deliberately held out of training (M5 ≫ 0, M6 low) — an existence proof
  that a language model can output structures outside its entire experience.
- **H5 (mirage check):** The continuous companion metric to M5 will improve smoothly
  with scale even if M5 itself shows a sharp transition.

## 6. What this project can and cannot conclude

**Can:** whether frontier models' apparent unoriginality is a sampling/tuning artifact;
whether their ideas are near-manifold recombinations; whether provable combinational
P-creativity exists in small models and how it scales; whether "emergence" here is a
real discontinuity or a metric artifact.

**Cannot:** whether any model is H-creative (untestable in principle); whether emitting
an original idea constitutes "having" one (philosophy of mind); whether transformational
creativity (rule-breaking) is achievable — our Phase 3 domain has fixed rules by design.

The honest headline answer will therefore have the shape: *"Under definition D,
frontier models today do/don't X; small models provably can/can't Y; the open
questions are Z."*
