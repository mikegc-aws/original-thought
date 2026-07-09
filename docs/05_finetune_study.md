# Phase 4 — Is the collapse trained-in? Is it reversible? At what cost?

**Design.** One base model (s4, 816k params — high rule-precision *and* high novelty
before deep collapse), pretrained 6k steps on the Phase 3 domain, then three
continuations of 3k steps from **identical weights** (2 seeds each, evaluated every
500 steps with the exact Phase 3 classifier):

- **control** — keep training on the full training set.
- **preference** — train on a narrow "preferred" slice (2% of triples, one template):
  a minimal simulation of preference-tuning's high-reward distribution.
- **diversity** — iterative self-training on its own grammatical, non-memorized,
  not-previously-kept outputs (+ small real-data anchor). Crucially, **no validity
  oracle**: the only signal is "new and well-formed", mirroring what's actually
  available at frontier scale, where no oracle for novelty-with-validity exists.

Code: `experiments/phase4_finetune.py`; plots `results/phase4/trajectories.png`.

## Endpoints (step 3000, mean of 2 seeds; baseline = shared start)

| arm | train-dup rate | unique held-out valid /4k | rule precision |
|---|---|---|---|
| *baseline (step 0)* | *0.85* | *506* | *0.96* |
| control | 0.87 | 465 | 0.97 |
| **preference** | **0.98** | **24** | 0.62 |
| **diversity** | **0.17** | 144 | **0.05** |

Diversity-arm trajectory (both seeds near-identical):

| step | 0 | 500 | 1000 | 2000 | 3000 |
|---|---|---|---|---|---|
| dup rate | 0.86 | 0.74 | 0.47 | 0.20 | 0.17 |
| unique valid held-out | 506 | **671** | 500 | 168 | 144 |
| rule precision | 0.96 | 0.73 | 0.27 | 0.06 | 0.05 |

## Findings

**F9 — Collapse is trained-in and inducible on demand.** 3k steps on a narrow
"preferred" distribution took memorization from 0.85 to 0.98 and cut expressed novelty
20-fold (506 → 24 unique valid held-out triples). Rule precision also *fell* (0.96 →
0.62): over-concentration hurt the very competence it was distilled from. This is the
mechanistic miniature of what preference-tuning plausibly does to frontier models —
Phase 1's collapsed "Grief Cartographer" behavior, created in a dish.

**F10 — Collapse is reversible… briefly and genuinely.** After 500 steps of
diversity self-training the model beat its own baseline on *both* axes that matter:
unique valid held-out output up 33% (506 → 671) with duplicates down (0.86 → 0.74) and
precision still 0.73. The novelty was there, suppressed, and a training-time
intervention — not just Phase 1's sampling-time tricks — can surface more of it.

**F11 — But novelty-only optimization degenerates.** Iterated further with no validity
signal, the feedback loop eats itself: the model trains on its own increasingly
rule-breaking outputs, invalid rate climbs 0.7% → 79%, precision collapses to 0.05.
By step 3000 the "diversity" model is maximally unlike its training data and almost
always *wrong*. This is Phase 0's definition made kinetic: **novelty without a value
constraint is not originality, and optimizing for it alone actively destroys value.**
(It is also a small-scale picture of model-collapse-by-self-training.)

**F12 — The trade-off triangle.** Across arms: control = valid but collapsed;
preference = hyper-collapsed; diversity = diverse but invalid. No arm achieves all
three of {novel, valid, diverse} at convergence — the best point observed anywhere in
Phase 4 is the *transient* at diversity step 500. Sustained genuine originality
appears to require an external validity signal (a critic, a verifier, the world) to
train against — "be new" and "be right" are separate signals and both must be present.

## Caveats

- The preference arm is a crude proxy for RLHF (supervised narrowing, no reward
  model); it demonstrates distribution-sharpening, not the full RLHF pipeline.
- Diversity arm hyperparameters (t=1.2 sampling, 4:1 self:anchor mix) weren't tuned;
  a stronger anchor or a learned validity critic might sustain the step-500 sweet
  spot. That experiment — novelty + verifier — is the natural sequel.
- 2 seeds; endpoints are tightly clustered across seeds but small-n.
