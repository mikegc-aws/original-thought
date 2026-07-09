# The model kept giving me the same idea

A while back I ran a small experiment that bothered me for months. I asked a frontier LLM: *"Write an original and novel idea that has not been done before."* It gave me a decent-sounding idea. Then I asked again, in a fresh session. Same idea. Again — same idea, worded slightly differently. Whatever I did, the model had exactly one "original idea," and it was always that one.

That felt like it should settle something about the question everyone circles around — *can AI have original ideas?* — but I couldn't work out what it settled. Does a model that returns one idea have one idea? Or was I measuring something else entirely? I decided to stop speculating and build a research project around it. This post is the full writeup: four experiments, about $17 of AWS spend, and an answer I didn't expect when I started. I've tried to include enough detail that you could replicate any of it.

## First, the question had to be made testable

"Original idea" is doing a lot of unexamined work in that question, so before running anything I pinned down definitions I could measure against.

Creativity research has a useful distinction (from Margaret Boden): **P-creative** means new to the creator; **H-creative** means new to all of history. H-creativity turns out to be untestable in principle — for humans too, since nobody can enumerate every idea ever had. Any measurable claim about novelty is really P-creativity against some body of experience. For an LLM, "new to the creator" has a crisp translation: *new relative to the training data*. That's the version I set out to test.

Second definition: originality is not novelty alone. Random word salad is maximally novel and worthless. An original idea has to be **both novel and coherent/valid**. Hold onto that conjunction — it ends up being the hinge of the whole project.

Third, I pre-registered a suspicion. People love calling abilities like this "emergent," and my take has always been that "emergent property" is mostly a fancy way of saying *we don't know why this happens*. So I fixed a strict definition up front — a capability is emergent only if a metric chosen in advance sits near its floor below some model scale and climbs steeply above it — and committed to measuring with both a pass/fail metric and a continuous one, because there's a known critique (Schaeffer et al., 2023) that many published "emergent abilities" are artifacts of pass/fail metrics hiding smooth underlying progress.

## Experiment 1: measuring the thing I saw

The first experiment reproduced my original one, at scale: 4 models on Amazon Bedrock (Claude Sonnet 4.6, Claude Haiku 4.5, Nova Pro, Llama 3.3 70B) × 5 sampling configurations × 50 samples each — 1,000 generations of "an original and novel idea." I embedded every response (Titan v2, 512 dimensions), clustered them (agglomerative, average linkage, cosine distance threshold 0.45), and measured what fraction of each model's 50 answers fell into the single largest cluster — i.e., were the *same idea*.

At default-ish settings (temperature 0.7, the bare prompt):

| model | share of 50 samples that are one idea |
|---|---|
| Llama 3.3 70B | 92% |
| Claude Sonnet 4.6 | 70% |
| Claude Haiku 4.5 | 52% |
| Nova Pro | 18% |

At temperature 0, three of the four models produced the *identical* idea all 50 times.

Two details changed how I read my original experiment. First, each model has a *different* pet idea. Claude Sonnet always pitched a "Grief Cartographer" (an app that maps grief as an evolving landscape). Haiku pitched "Temporal Debt" (an economy where time is the borrowed currency). Llama pitched a biometric art installation; Nova an emotion-sensing haptic garment. So there is no single answer LLMs converge on — the repetition is a property of each model, not of the question. Though notice the four pet ideas are all the same *kind* of thing: emotion-made-tangible-through-technology. The genre itself is collapsed.

Second — and this reframed everything — the repetition is trivially breakable. I added two random words to the prompt ("Here are two random words: 'lantern' and 'kelp'. Using them only as loose inspiration, write an original and novel idea...") and re-ran everything at the *same temperature*. The one-idea share dropped from 70–92% to 4–10% for three of the four models; effective idea counts (Vendi score — roughly, the number of genuinely distinct ideas in the sample) went from 2–7 to 17–31 out of 50. I also had a judge model not in the test pool (Claude Opus 4.5) blind-score ~15 outputs per condition for coherence and value on a 1–10 rubric: coherence stayed between 8.5 and 9.7 across *every* condition. The diverse ideas were exactly as well-formed as the collapsed one.

So my original experiment hadn't measured what the model *can* produce. It measured where the probability mass sits. The model contains a landscape of ideas; training has worn one valley very deep; the unmodified prompt rolls into it every time. Cranking temperature shakes the ball a little. Changing where the ball starts — two random words — matters far more.

## Experiment 2: but is any of it actually new?

Diversity isn't originality. Fifty different ideas could be fifty paraphrases of things people have already done. So the second experiment built a reference corpus of 9,445 *real* existing ideas — 5,600 Y Combinator company descriptions plus 3,800 recent arXiv abstracts in human-computer interaction — embedded in the same space, and measured how far each generated idea sits from its nearest real neighbor.

Raw distances mean nothing alone, so I calibrated: hold out 500 real ideas, measure how far each sits from the nearest *other* real idea. Median distance: 0.548. That's the yardstick for "as novel as ideas normally are."

The results cut both ways. The median generation from Claude and Llama sits at 0.56–0.67 — at or slightly beyond the human baseline. Typical LLM "ideas" are about as far from existing ideas as existing ideas are from each other. Ordinary recombination: not plagiarism, not breakthrough.

The bad tail is worth seeing, though. Nova Pro's pet idea — the emotion-reading wearable — sits at distance 0.40 from an actual YC company called Anoria, whose pitch is "the first wearable that reads your emotions to enhance your EQ." Nearly point for point the same product. 74% of Nova's greedy-mode generations sat closer to an existing real idea than real ideas sit to each other — every one asserting "this has not been done before." The model has no mechanism for checking that claim. "Original" in a prompt is a style instruction, not a verified property.

This experiment has an honest ceiling, and it's what drove the next one: my 9,445-idea corpus is a rounding error against trillions of training tokens. An idea far from *my* corpus may still be near something the model read. With frontier models you can never close that gap, because you can never enumerate their experience. So I stopped using frontier models.

## Experiment 3: making novelty provable

This is the experiment the whole project rests on, so here is all of it.

### The world

To *prove* an output is outside a model's experience, you must own the model's entire experience. So I built a closed world: a language of inventions. Every sentence describes a triple — a **device**, a **mechanism**, a **purpose** — drawn from 50 invented pseudo-words of each type (150 total, generated as pronounceable syllable combinations: "flotror", "dorfi", "parsor"). A triple renders as a sentence through one of three templates:

```
the {device} uses {mechanism} to {purpose} .
with {mechanism} the {device} can {purpose} .
a {device} is able to {purpose} using {mechanism} .
```

That gives 50³ = 125,000 possible inventions. Hidden underneath: every word secretly belongs to one of **5 latent classes** (word #1 → class 1, #2 → class 2, ... assigned round-robin, so classes are balanced). A triple is **valid** only if the device's class is compatible with the mechanism's class, *and* the mechanism's class is compatible with the purpose's class — where each class is compatible with exactly 2 of the 5 classes on the other side, chosen at random and fixed. Result: exactly 20,000 of the 125,000 triples (16%) are valid. The classes and compatibility rules appear nowhere in any training sentence. If a model is to respect them, it must infer them from co-occurrence statistics alone.

Then I locked combinations away, at two depths:

- **L1 holdout — 1,840 valid triples** removed from training. The individual words and even the device–mechanism pairs still occur in other training sentences; the specific three-way combination never does. Producing one is novelty at the level of "this exact idea was never seen."
- **L2 holdout — 1,600 valid triples**, built by picking 80 valid device–mechanism *pairs* and scrubbing **every** sentence containing those pairs from training. The model never sees those two words together in any context. Producing a valid L2 triple can't be done from pair memory — it requires the latent class rule.

The remaining **16,560 valid triples** became the training set, each rendered in all 3 templates: 49,680 sentences. A sanity check confirms every one of the 150 words still appears in training, so no output can be dismissed as involving unseen vocabulary.

The payoff of this construction: an exact, instant classifier for every generation. Memorized, provably-novel-valid (L1 or L2), rule-breaking, or ungrammatical — no embeddings, no similarity thresholds, no judgment calls.

### The model and how it was trained

I trained GPTs **from scratch** — every weight from random initialization. This is non-negotiable for the logic of the experiment: fine-tune any existing model and it arrives with trillions of tokens of unknown experience, and the skeptic's "it saw something like that somewhere" objection comes straight back. From-scratch training on an enumerated corpus is what makes "outside its entire experience" a checkable statement rather than a vibe.

The architecture is a standard decoder-only transformer — the same family as frontier models, at dollhouse scale:

- Token embeddings + learned positional embeddings; **vocabulary of 163 tokens** (the 150 pseudo-words, 10 template words, punctuation, and BOS/EOS/PAD markers — one token per word, no subword tokenizer needed in a closed world)
- N identical blocks: causal multi-head self-attention → LayerNorm → 4× MLP with GELU, residual connections, dropout 0.1
- Output head weight-tied to the input embeddings
- **Context window: 12 tokens** — one full invention sentence

Seven sizes, spanning ~6,000× in scale, three random seeds each (21 runs total):

| name | layers | heads | width | parameters |
|---|---|---|---|---|
| s0 | 1 | 1 | 8 | 2,288 |
| s1 | 1 | 2 | 16 | 6,112 |
| s2 | 2 | 2 | 32 | 31,072 |
| s3 | 2 | 4 | 64 | 111,296 |
| s4 | 4 | 4 | 128 | 815,744 |
| s5 | 6 | 8 | 256 | 4,783,872 |
| s6 | 8 | 8 | 384 | 14,263,680 |

Training: next-token prediction (cross-entropy, padding masked) — the same objective frontier models use, which is the point: whatever behavior appears is attributable to the objective, not to any creativity-specific trick. 6,000 steps at batch size 256, AdamW (lr 3e-4, weight decay 0.01, cosine decay), identical recipe at every size. All 21 runs took ~40 minutes total on one AWS g6.xlarge (a single NVIDIA L4) — about a dollar of compute. For calibration, the largest model here is ~10,000,000× smaller than a frontier LLM.

Evaluation, identical for every run: sample **4,000 sentences** unconditionally at temperature 1.0, classify each one exactly, and separately compute a continuous metric — the mean per-token log-probability the model assigns to 500 held-out *valid* triples versus 500 *invalid* ones. That log-prob gap is the smooth companion to the pass/fail novelty metric, included specifically for the emergence question.

### Sample results — what the generations look like

Real output from the trained models, with the classifier's verdicts:

```
[memorized]  the flotror uses dorfi to parsor .
[memorized]  with kresko the nirgrubi can felkir .
[L1-novel]   with skopitre the nirgrubi can belselvli .
[L1-novel]   a vlunzunvar is able to brarsel using zeskarkrar .
[L2-novel]   a trorskirko is able to vuntragrel using siplu .
[L2-novel]   with lirbrunu the kriflo can gortrimo .
[invalid]    the grelda uses zundrevun to puntrufla .
[ungrammatical]  with zalar the girkir can clirbrarla
```

Meaningless to a human eye by design — the meaning is in the verdicts. The L1 line is a valid invention that was deliberately excluded from training. The L2 lines are stronger: "trorskirko" and "siplu" never co-occur anywhere in the model's 49,680 training sentences, yet the model asserted the pairing and it is correct under the hidden compatibility rule. The invalid line is perfectly grammatical but breaks the rule — the closed-world equivalent of a fluent, incoherent idea.

### The numbers

Headline metric: of the novel grammatical outputs (not memorized), what share obey the hidden rule? Chance — a model that learned the grammar but not the rule — is 16%.

| parameters | 2.3k | 6k | 31k | 111k | 816k | 4.8M | 14.3M |
|---|---|---|---|---|---|---|---|
| **rule precision on novel outputs** | 0.05 | 0.05 | 0.17 | **0.83** | 0.95 | 0.95 | 0.97 |
| **log-prob gap (continuous)** | 0.05 | 0.06 | 0.28 | 0.78 | 0.97 | 0.99 | 1.00 |
| **memorized share of output** | 0.12 | 0.18 | 0.41 | 0.79 | 0.83 | 0.87 | 0.89 |
| **distinct L2-novel per 4k samples** | 39 | 63 | 121 | **234** | 231 | 98 | 56 |

Three results live in that table.

**Provable originality exists, cheaply.** From 111k parameters up, models generate hundreds of distinct valid held-out inventions per 4,000 samples — the s3/s4 models produced ~330 distinct L1 and ~230 distinct L2 triples each, covering 13–15% of the entire held-out space, at 95%+ rule precision. Under the definition fixed at the start, that is original output: novel relative to everything the model has ever seen, and valid. The structural skeleton of an idea — a chemist who has internalized valence rules proposing a compound nobody has synthesized.

**The "emergence" decomposed.** The pass/fail metric does the classic emergent thing: flat at chance through 31k parameters, then one 3.6× size step later it's at 0.83. But look at the continuous row underneath: at 31k, where pass/fail reads "nothing," the log-prob gap is already 0.28 and climbing. Smooth competence growth, thresholded measurement. Steep and real, but not magic — my prior about the word "emergent" survived contact with the data.

**Scale suppresses expression.** Read the last two rows together. Past the size needed to learn the rule, bigger models know the rule *better* (precision 0.97, the widest log-prob gap) yet emit *fewer* novel items — memorized output climbs monotonically to 89% while distinct L2 novelties fall four-fold from their 111k–816k peak. No RLHF anywhere in these models; pure likelihood training concentrates output on the already-seen. My frontier-model observation, reproduced in a dish. (One confound to flag honestly: all sizes trained for the same 6,000 steps, so the larger models are also further into overfitting — scale and training progress are partially entangled in that trend.)

## Experiment 4: is the collapse trained-in, and is it reversible?

The scale sweep showed collapse *correlating* with training; the last experiment made it causal. I pretrained a single s4 model (816k parameters, 6,000 steps as above), snapshotted the weights, and ran three continuations of 3,000 steps each from that **identical snapshot** (two seeds; evaluated with the full protocol every 500 steps):

- **Control:** keep training on the full training set.
- **Preference:** train only on a "preferred" slice — 2% of the training triples, one template. A deliberately crude simulation of what preference-tuning does: concentrate probability on a narrow high-reward distribution.
- **Diversity:** iterative self-training. Each round, sample 8,000 sentences at temperature 1.2; keep those that are grammatical, *not* in the training data, and not previously kept; fine-tune on them (mixed 4:1 with real data to anchor the grammar); repeat. Crucially, this arm gets **no access to the hidden validity rule** — the only signal is "new and well-formed," which is an honest mirror of frontier reality, where no oracle for novelty-with-validity exists.

Endpoints after 3,000 steps (start point for all arms: 85% memorized, 506 distinct novel-valid per 4k samples, 0.96 precision):

| arm | memorized | distinct novel-valid /4k | rule precision |
|---|---|---|---|
| control | 87% | 465 | 0.97 |
| preference | **98%** | **24** | 0.62 |
| diversity | 17% | 144 | 0.05 |

The preference arm is collapse manufactured on demand: a 20-fold cut in expressed novelty in 3,000 steps, and rule precision *dropped* too — over-concentration damaged the competence it was distilled from.

The diversity arm is the experiment's punchline, in two acts. At step 500 it had *beaten its own baseline on both axes* — 671 distinct novel-valid inventions (up 33%) with duplicates down to 74% and precision still 0.73. The suppressed novelty was real and recoverable by training, not just by prompt tricks. Then, iterated further with nothing but "be new" as a signal, it ate itself: training on its own increasingly rule-breaking output, invalid share climbing from under 1% to 79% by step 3,000. Novelty without a validity check isn't originality — optimizing it alone actively destroys the validity. Which, I'd note, is also true of people: humans who optimize pure novelty with no external check don't produce originality either. They produce crankery.

Across all arms, no configuration held novel + valid + diverse simultaneously at convergence. The best point observed anywhere is that transient at diversity step 500 — which suggests sustained machine originality needs an external validity signal (a verifier, a critic, the world) in the training loop, not just released pressure.

## Where I landed

The question I started with — *will AI ever be able to have original ideas?* — turned out to have the wrong verb tense. Under the only definition anyone can actually test, the answer is already yes: a 111k-parameter model provably produced valid structures from outside its entire training experience, hundreds of times per few thousand samples.

The real finding is why we don't see this from the models we use every day: **capability and behavior come apart.** Every negative result in this project was about behavior — what gets sampled by default. Every positive result was about capability — what the model's distribution supports. Likelihood training concentrates output on the familiar; preference-tuning concentrates it further onto the polished consensus answer. We are, in effect, trading away expressed originality for reliability. My same-idea-every-time experiment was an accurate observation of that trade and a false inference about the ceiling.

The boundaries of the claim, stated plainly: this demonstrates *combinational* originality — new valid points in a fixed conceptual space. Whether models can transform a space — break the rules rather than recombine within them — is untouched by the closed-world method, by design, since the world's rules are fixed. The toy world's sentences carry no semantic richness; the claim is that the load-bearing mechanism skeptics say is impossible demonstrably exists, not that it yet amounts to ideas like ours. And the bridge from a 14M-parameter dish to frontier models is an argument from shared architecture, shared objective, and matching collapse phenomenology — strong, but an argument, not a proof.

Three practical takeaways if you want original ideas from an LLM today: inject entropy into the prompt (random words, random domains) rather than only raising temperature — in my tests it was worth roughly 5× more effective diversity; never trust a model's claim that something "has not been done before," because it cannot check; and when a model gives you the same answer every time, you've learned where its probability mass sits, not what it's capable of.

Total cost of finding all this out: about $15 in API calls and $2 of GPU time. The gap between what these models express and what they contain is measurable with pocket money. That gap closing — not some future architecture — is where I'd now look for machine originality to show up.
