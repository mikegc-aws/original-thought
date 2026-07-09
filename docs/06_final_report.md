# Will AI ever be able to have original ideas?

**Final synthesis — Original Thought project, July 2026.**
Evidence: 1,000 frontier-model generations across 4 models × 5 configs (Phase 1);
novelty measurement against 9,445 real ideas with human-baseline calibration
(Phase 2); 21 small GPTs (2.3k–14.3M params) trained from scratch on a closed
synthetic domain with provable held-out novelty (Phase 3); a three-arm fine-tuning
intervention on identical base weights (Phase 4). All definitions and hypotheses were
fixed in advance in `docs/01_definitions.md`.

---

## The short answer

**Under the only definition we can test — P-creativity: producing coherent,
rule-respecting outputs that lie outside everything the model was trained on — the
answer is yes, and it is already demonstrated, not hypothetical.** A 111k-parameter
transformer provably did it hundreds of times per few thousand samples (Phase 3).

The interesting question turned out not to be *whether* language models can produce
original output, but **why they so rarely show it**. Everything we measured points to
one answer: the capability and the behavior come apart. Training objectives —
likelihood first, preference-tuning second — concentrate probability mass on the
familiar. Originality isn't absent from these models; it is *suppressed by the
objective*, and partially recoverable by how you sample, prompt, or continue training.

## What each phase established

**1. The "same idea every time" phenomenon is real, and it is not inability.**
Mike's founding observation replicated exactly: with the bare prompt at ordinary
settings, models return one pet idea 52–100% of the time (deterministically identical
at temperature 0). But two random seed words in the prompt — same model, same
temperature — recover 20–30 distinct, equally coherent ideas per 50 asks. Each model
has a *different* pet idea, yet all four live in one genre (emotion-made-tangible-
via-tech), showing a deeper genre-level collapse too. (`02_frontier_study.md`)

**2. When frontier models do produce ideas, they are ordinarily novel — and
occasionally not novel at all.** The median generation sits about as far from
existing ideas (embedding-wise) as real ideas sit from each other: unremarkable,
honest recombination. At the bad tail, one model's confidently-"never-been-done" idea
was a near-duplicate of an existing YC company. Models have no mechanism to verify
their own novelty claims: "original" in a prompt is a style instruction, not a
checked property. These are proxy measurements with a small corpus — which is why
Phase 3 existed. (`03_novelty_analysis.md`)

**3. Provable originality exists, cheaply.** Trained on a closed combinatorial world
with hidden compatibility rules, models ≥111k params generate hundreds of distinct
valid inventions that were *deliberately excluded from training* — including
combinations whose parts were never seen together in any form. That is combinational
P-creativity by construction, in a model ~10⁷× smaller than a frontier LLM.
(`04_small_model_study.md`)

**4. "Emergence" here demystified.** The pass/fail novelty metric jumps from chance
to 0.83 in one size step — a textbook emergence curve. But the continuous companion
metric (log-prob gap between valid and invalid unseen combinations) was already
rising where the thresholded metric read zero. Mike's skepticism — "emergent =
we don't know why" — is substantially vindicated: the jump decomposes into smooth
competence growth plus a thresholded way of measuring it. Steep, real, and not a
mystery. (`04`, §Result 2)

**5. The collapse mechanism, isolated.** With zero RLHF, purely likelihood-trained
models past the size needed to learn the rules got *better at knowing* which unseen
combinations are valid while *emitting fewer of them* — memorized output rose from
41% to 89% across our scale sweep. Phase 4 then showed the dose-response: 3k steps of
narrow "preference" data → 98% memorized output, novelty down 20-fold; 500 steps of
novelty-seeking self-training → better than baseline on *both* novelty and validity;
3k steps of novelty-seeking *without any validity signal* → 79% invalid junk.
Collapse is trained-in, inducible, and briefly reversible — but sustained originality
needs both a novelty pressure *and* a validity check, and the second doesn't come
for free. (`04` Result 3, `05_finetune_study.md`)

## The synthesis: one picture

A trained language model is a probability landscape over idea-space. Training carves
valleys wherever the data is dense; preference-tuning deepens the deepest one. Asking
"give me an original idea" rolls a ball downhill — same valley every time. That was
Mike's experiment. Temperature shakes the ball a little; changing the question
(random seeds) starts it somewhere else entirely; diversity fine-tuning reshapes the
landscape itself. And crucially, the landscape *extends beyond the training data*:
where the data has compositional structure, the model's generalization assigns real
probability to valid-but-never-seen points — reachable, provably novel, sitting there
whether or not anyone samples them.

So: **capability ≠ behavior.** Every negative result in this project was a fact about
behavior (what gets sampled); every positive result was a fact about capability (what
the distribution supports). The everyday impression that LLMs "have no original
ideas" is an accurate report of default behavior and a false inference about the
underlying distribution.

## What we did NOT show, honestly

- **Only combinational creativity** — the easiest of Boden's three modes — was proven.
  Exploratory and especially transformational creativity (breaking the rules of the
  space, not recombining within it) remain undemonstrated here, and our closed-world
  method cannot test them by design: the domain's rules are fixed.
- **No H-creativity.** "New to all of history" is unverifiable for machines and humans
  alike; every claim of it is really "new relative to some corpus".
- **Scale transfer is an argument, not a proof.** Phases 3–4 are a mechanistic model
  of frontier behavior, made plausible by the Phase 1/2 correspondences (collapse
  patterns, capability/behavior gap), not a demonstration inside a frontier model.
- **The attribution question stands.** Whether emitting a provably-novel valid
  structure constitutes *having an idea* is philosophy of mind, not measurement. Our
  contribution is to make the empirical part crisp: if you require only
  novel-coherent-valuable output, the answer is yes; if you require something more,
  you must now say what that something is.

## Answering the headline question

**"Will AI ever be able to have original ideas?"**

For the measurable meaning of the question, *ever* is the wrong tense: small models
already produce provably original combinational output, and frontier models
demonstrably contain far more idea-diversity than they show. Today's frontier models
asked plainly will give you their pet idea — occasionally one that literally already
exists — because their training pushes expression toward consensus.

The open frontier is not possibility but *reliability and depth*: sustaining novelty
+ validity together (our Phase 4 triangle — the transient sweet spot at diversity
step 500 suggests it needs an external verifier in the loop), and reaching beyond
recombination toward the exploratory and transformational creativity this project
deliberately did not attempt. Those are engineering and research problems with
visible attack surfaces — not walls.

**Practical corollaries** for anyone wanting original ideas from an LLM today:
inject entropy into the prompt (random seeds/domains) rather than only raising
temperature; never trust a model's own novelty claim — verify externally; and treat
"the model gives me the same answer every time" as a sampling symptom, not a
capability ceiling.

---

*Project artifacts: definitions & pre-registered hypotheses (`01`), phase reports
(`02`–`05`), all code under `experiments/` and `src/`, all metrics/plots under
`results/`. H1 ✓, H2 ✓, H3 ✗ (models less derivative than predicted), H4 ✓, H5
partly ✓. Total cost: ≈ $10–15 Bedrock + ~$2 EC2 GPU.*
