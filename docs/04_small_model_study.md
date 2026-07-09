# Phase 3 — Provable originality in small models, and what scale does to it

**Setup.** A closed synthetic "inventions" world: triples (device, mechanism, purpose)
over 3×50 pseudo-words, valid iff hidden latent-class compatibility rules hold (16% of
the 125,000-triple space is valid; the classes are never stated in the data). Small
GPTs (1–8 layers, **2.3k → 14.3M params**) trained from scratch on 16,560 valid triples
rendered as sentences. Two kinds of valid triples were deliberately excluded:

- **L1 (1,840 triples):** held out at triple level — parts and pairs seen elsewhere.
- **L2 (1,600 triples):** every triple containing certain (device, mechanism) pairs
  removed — the pair itself is *never* seen; producing one requires the latent rule.

Because we own the entire training set, novelty here is **provable, not proxied**:
every generation is exactly classified as memorized / L1-novel / L2-novel / invalid /
ungrammatical. 7 sizes × 3 seeds, 6,000 steps each, 4,000 generations per model.
Run on EC2 g6.xlarge (NVIDIA L4), ~40 min total. Code: `src/original_thought/domain.py`,
`model.py`, `experiments/phase3_*.py`. Plots: `results/phase3/emergence.png`.

## Result 1 — H4 confirmed: provable combinational novelty exists

Models from ~111k params up generate **hundreds of distinct valid held-out
inventions** — including L2 triples whose device–mechanism pairing appears nowhere in
their entire training experience. The best cells produce ~330 unique L1 and ~230
unique L2 triples per 4,000 samples (13–15% coverage of each held-out space), with
n-gram-level copying ruled out by exact triple classification.

This is the existence proof the project was after: **a language model can emit
well-formed, rule-respecting structures that are outside everything it was ever
trained on.** In Boden's terms, P-creative combinational originality — the model
inferred the hidden conceptual structure and used it to reach unvisited valid points.
A 111k-parameter model is about 10 million times smaller than a frontier LLM; whatever
this capacity is, it requires no scale mystique.

## Result 2 — The emergence curve, and the mirage check (H5)

Precision = of novel grammatical outputs, share obeying the hidden rule (chance ≈ 16%):

| params | 2.3k | 6k | 31k | 111k | 816k | 4.8M | 14.3M |
|---|---|---|---|---|---|---|---|
| **M5 precision** | 0.05 | 0.05 | 0.17 | **0.83** | 0.95 | 0.95 | 0.97 |
| **log-prob gap (continuous)** | 0.05 | 0.06 | 0.28 | 0.78 | 0.97 | 0.99 | 1.00 |

M5 sits at/below chance through 31k params, then jumps to 0.83 within one ~3.6× size
step — a textbook "emergent" transition under our strict Phase 0 definition. But the
continuous companion metric already shows clear movement at 31k (0.28 gap) where the
discontinuous metric still reads "chance". Verdict: **partly a mirage, partly real** —
competence grows earlier and more smoothly than the thresholded metric suggests, but
the growth itself is still steep, not linear. This is exactly the "smooth competence,
thresholded behavior" outcome anticipated in `docs/01_definitions.md` §4: the word
"emergent" here compresses a real but decomposable phenomenon, vindicating a
demystified reading of the term.

## Result 3 (unexpected) — Scale past sufficiency *suppresses* novelty output

Novel-output rates rise then **fall** with scale, while memorization climbs
monotonically:

| params | 31k | 111k | 816k | 4.8M | 14.3M |
|---|---|---|---|---|---|
| training-duplicate rate | 0.41 | 0.79 | 0.83 | 0.87 | 0.89 |
| L2-novel rate / unique L2 | 0.03 / 121 | 0.06 / 234 | 0.06 / 231 | 0.03 / 98 | 0.01 / 56 |

The biggest models are the *best* at knowing the rule (precision 0.97, largest
log-prob gap) yet emit the *fewest* novel items, because they allocate ever more
probability mass to exact training sentences. **This is Phase 1's mode collapse
reproduced from scratch in a 14M-param model with no RLHF at all** — sharpening toward
the training distribution is a consequence of ordinary likelihood training at fixed
sampling temperature, which preference-tuning then amplifies in frontier models. The
capability (knowing which unseen combinations are valid) and the behavior (emitting
them) come apart — the same capability/behavior gap that random seed words exposed in
Phase 1.

## Caveats

- One domain, one grammar, one rule structure; steps fixed at 6k for all sizes
  (bigger models are also further into overfitting — scale and training progress are
  partially confounded in Result 3; the constant here is compute per model, not fit).
- Combinational creativity only — deliberately the easiest, provable mode. Nothing
  here speaks to exploratory or transformational originality.
- "Original idea" in this world means a valid unseen triple; semantic richness of real
  ideas is far beyond this. The claim is existence, not equivalence.

## What Phase 3 adds to the headline answer

1. Original-by-construction output from a language model is **demonstrated**, at tiny
   scale (H4 ✓).
2. "Emergence" of this ability decomposes into smooth competence + thresholded
   measurement (H5 partly ✓) — it is not an unexplainable jump.
3. The frontier-model behavior Mike observed is continuous with a mundane mechanism
   visible even here: better-trained models concentrate probability on what they've
   seen. Originality in LLMs is not missing — it is *suppressed by the objective*,
   and recoverable by how you sample and ask.
