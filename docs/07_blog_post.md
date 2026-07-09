# The model kept giving me the same idea

A while back I ran a small experiment that bothered me for months. I asked a frontier LLM: *"Write an original and novel idea that has not been done before."* It gave me a decent-sounding idea. Then I asked again, in a fresh session. Same idea. Again — same idea, worded slightly differently. Whatever I did, the model had exactly one "original idea," and it was always that one.

That felt like it should settle something about the question everyone circles around — *can AI have original ideas?* — but I couldn't work out what it settled. Does a model that returns one idea have one idea? Or was I measuring something else entirely? I decided to stop speculating and build a research project around it. This post is what came out: four experiments, about $15 of AWS spend, and an answer I didn't expect when I started.

## First, the question had to be made testable

"Original idea" is doing a lot of unexamined work in that question, so before running anything I pinned down definitions I could actually measure against.

Creativity research has a useful distinction (from Margaret Boden): **P-creative** means new to the creator; **H-creative** means new to all of history. H-creativity turns out to be untestable in principle — for humans too, since nobody can enumerate every idea ever had. Any measurable claim about novelty is really P-creativity against some body of experience. For an LLM, "new to the creator" has a crisp translation: *new relative to the training data*. That's the version I set out to test.

Second definition: originality is not novelty alone. Random word salad is maximally novel and worthless. An original idea has to be **both novel and coherent/valid**. Hold onto that conjunction — it ends up being the hinge of the whole project.

Third, I pre-registered a suspicion. People love calling abilities like this "emergent," and my take has always been that "emergent property" is mostly a fancy way of saying *we don't know why this happens*. So I fixed a strict definition up front — a capability is emergent only if a metric chosen in advance sits near its floor below some model scale and climbs steeply above it — and committed to measuring it with both a pass/fail metric and a continuous one, because there's a known critique (Schaeffer et al., 2023) that many published "emergent abilities" are artifacts of pass/fail metrics hiding smooth underlying progress.

## Measuring the thing I saw

The first experiment reproduced my original one, at scale: 4 models on Amazon Bedrock (Claude Sonnet 4.6, Claude Haiku 4.5, Nova Pro, Llama 3.3 70B) × 5 sampling configurations × 50 samples each — 1,000 generations of "an original and novel idea." I embedded every response, clustered them, and measured what fraction of each model's 50 answers were the *same idea*.

At default-ish settings (temperature 0.7, the bare prompt), the numbers matched my experience:

| model | share of 50 samples that are one idea |
|---|---|
| Llama 3.3 70B | 92% |
| Claude Sonnet 4.6 | 70% |
| Claude Haiku 4.5 | 52% |
| Nova Pro | 18% |

At temperature 0, three of the four models produced the *identical* idea all 50 times.

Two details changed how I read my original experiment. First, each model has a *different* pet idea. Claude Sonnet always pitched a "Grief Cartographer" (an app that maps grief as an evolving landscape). Haiku pitched "Temporal Debt" (an economy where time is the borrowed currency). Llama pitched a biometric art installation; Nova an emotion-sensing haptic garment. So there is no single answer LLMs converge on — the repetition is a property of each model, not of the question. Though notice the four pet ideas are all the same *kind* of thing: emotion-made-tangible-through-technology. The genre itself is collapsed.

Second — and this is the result that reframed everything — the repetition is trivially breakable. I added two random words to the prompt ("Here are two random words: 'lantern' and 'kelp'. Using them only as loose inspiration, write an original and novel idea...") and re-ran everything at the *same temperature*. The one-idea share dropped from 70–92% to 4–10% for three of the four models. Instead of one idea per 50 asks, I got an effective 20–30 distinct ones. I also had a separate model (one not in the test pool) blind-score every output for coherence and usefulness: the scores were flat across all conditions. The diverse ideas were exactly as well-formed as the collapsed one.

So my original experiment hadn't measured what the model *can* produce. It measured where the probability mass sits. The model contains a landscape of ideas; training has worn one valley very deep; the unmodified prompt rolls into it every time. Cranking temperature shakes the ball a little. Changing where the ball starts — two random words — matters far more.

## But is any of it actually new?

Diversity isn't originality. Fifty different ideas could be fifty paraphrases of things people have already done. So the second experiment built a reference corpus of 9,445 *real* existing ideas — 5,600 Y Combinator company descriptions plus 3,800 recent arXiv abstracts in human-computer interaction — and measured how far each generated idea sits from its nearest real neighbor in embedding space.

Raw distances mean nothing alone, so I calibrated: how far does a *real* idea sit from the nearest *other* real idea? Median distance 0.548. That's the yardstick for "as novel as ideas normally are."

The results cut both ways. The median generation from Claude and Llama sits at 0.56–0.67 — at or slightly beyond the human baseline. Typical LLM "ideas" are about as far from existing ideas as existing ideas are from each other. Ordinary recombination: not plagiarism, not breakthrough.

The bad tail is worth seeing, though. Nova Pro's pet idea — the emotion-reading wearable — sits at distance 0.40 from an actual YC company called Anoria, whose pitch is "the first wearable that reads your emotions to enhance your EQ." Nearly point for point the same product. 74% of Nova's greedy-mode generations sat closer to an existing real idea than real ideas sit to each other — every one of them asserting "this has not been done before." The model has no mechanism for checking that claim. "Original" in a prompt is a style instruction, not a verified property.

This experiment has an honest ceiling, and it's what drove the next one: my 9,445-idea corpus is a rounding error against trillions of training tokens. An idea far from *my* corpus may still be near something the model read. With frontier models you can never close that gap. So I stopped using frontier models.

## Making novelty provable

The third experiment is the one the whole project rests on. If you want to *prove* an output is outside a model's experience, you need to own the model's entire experience. So I built a closed world.

The world is a language of inventions: sentences like *"the flotror uses dorfi to parsor"* — device, mechanism, purpose, all pseudo-words, 50 of each, 125,000 possible combinations. Hidden underneath: each word belongs to one of five latent classes, and only class-compatible combinations are valid — 16% of the space. The classes are never stated anywhere. And critically, I held combinations back. 1,840 valid triples were excluded from training (novel triple, familiar parts). Deeper still, for a set of device–mechanism *pairs*, every sentence containing the pair was scrubbed — 1,600 valid triples whose key pairing the model never witnesses in any form.

Then I trained GPTs from scratch on the remaining 16,560 valid inventions — the same architecture family as frontier models, at doll-house scale: seven sizes from 2,288 to 14.3 million parameters, three random seeds each, on a single AWS g6.xlarge GPU. The whole 21-run sweep took about 40 minutes and roughly a dollar.

Because I own every training token, every generation can be classified exactly: memorized, provably-novel-and-valid, invalid, or ungrammatical. No embeddings, no proxies.

The result: models from 111k parameters up produce hundreds of distinct valid held-out inventions per 4,000 samples — including about 230 from the deep holdout, where the model asserts a device–mechanism pairing it has *never seen co-occur*, and gets it right by the hidden rule 95%+ of the time. There is no retrieval story available for those. The model induced the latent class structure from data that never mentioned it, and used the structure to reach unvisited valid points.

That is original output under the definition I fixed at the start — novel relative to the entire training experience, and valid. It's the structural skeleton of an idea: a chemist who has internalized valence rules proposing a compound nobody has synthesized. And it required a model ten million times smaller than a frontier LLM.

The emergence question got its answer too. The pass/fail metric does the classic "emergent" thing — flat at chance (16%) through 31k parameters, then jumping to 83% precision one size-step later. But the continuous metric (how much more probability the model assigns to valid unseen combinations than invalid ones) was already climbing at 31k, where the pass/fail metric read zero. The "emergent jump" decomposed into smooth competence growth plus a thresholded way of measuring it. Steep and real, but not magic. My prior about the word "emergent" survived contact with the data.

## The suppression, isolated

One more thing fell out of the scale sweep, unplanned. Past the size needed to learn the rule, bigger models got *better* at knowing which unseen combinations are valid — and emitted *fewer* of them. The share of output that was verbatim training data climbed from 41% to 89% with scale, while novel output fell four-fold. No RLHF anywhere in these models. Ordinary likelihood training, by itself, concentrates output on the already-seen. My frontier-model observation, reproduced in a 14M-parameter dish.

So the fourth experiment asked directly: is the collapse trained-in, and is it reversible? I pretrained one 816k-parameter model, froze that moment in time, and ran three continuations from *identical weights*:

- **Control** (keep training on the same data): duplicates held at ~87%.
- **Preference** (train on a narrow 2% "preferred" slice — a crude simulation of what preference-tuning rewards): duplicates hit 98%; distinct novel-valid outputs fell from 506 to 24 per 4,000 samples. A 20-fold novelty cut in 3,000 steps. Collapse, manufactured on demand.
- **Diversity** (iteratively train the model on its own novel, non-memorized outputs — with no access to the hidden validity rule, since no such oracle exists at frontier scale either): after 500 steps this model *beat its own baseline on both axes* — 671 distinct novel-valid inventions (up 33%) with rule-precision still at 0.73. The suppressed novelty was recoverable by training, not just by prompt tricks.

Then it fell apart, instructively. Continued past the sweet spot with nothing but "be new" as a signal, the model trained on its own increasingly rule-breaking output: by step 3,000, 79% of what it produced was invalid junk. Novelty without a validity check isn't originality — optimizing for it alone actively destroys the validity. Which, I'd note, is also true of people: humans who optimize pure novelty with no external check don't produce originality either. They produce crankery.

## Where I landed

The question I started with — *will AI ever be able to have original ideas?* — turned out to have the wrong verb tense. Under the only definition anyone can actually test, the answer is already yes: a 111k-parameter model provably produced valid structures from outside its entire training experience, hundreds of times per few thousand samples.

The real finding is why we don't see this from the models we use every day: **capability and behavior come apart.** Every negative result in this project was about behavior — what gets sampled by default. Every positive result was about capability — what the model's distribution actually supports. Likelihood training concentrates output on the familiar; preference-tuning concentrates it further onto the polished consensus answer. We are, in effect, trading away expressed originality for reliability. My same-idea-every-time experiment was an accurate observation of that trade and a false inference about the ceiling.

What remains genuinely open is stated in the data too. Sustained originality needs two signals at once — novelty pressure *and* a validity check — and my fourth experiment shows neither alone suffices; the obvious next experiment is putting a verifier in the training loop and seeing if the step-500 sweet spot can be held. And everything here demonstrates *combinational* originality, the recombination kind. Whether models can transform a conceptual space — break the rules rather than recombine within them — is untouched by my closed-world method, by design.

Three practical takeaways if you want original ideas from an LLM today: inject entropy into the prompt (random words, random domains) rather than only raising temperature — it's worth roughly 5× more effective diversity; never trust a model's claim that something "has not been done before," because it cannot check; and when a model gives you the same answer every time, you've learned where its probability mass sits, not what it's capable of.

Total cost of finding all this out: about $15 in API calls and $2 of GPU time. The gap between what these models express and what they contain is measurable with pocket money. That gap closing — not some future architecture — is where I'd now look for machine originality to show up.
