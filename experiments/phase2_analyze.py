"""Phase 2 novelty analysis (metric M3).

For every Phase 1 generation, find its nearest neighbor in the reference corpus of
real ideas (YC + arXiv) in Titan-v2 embedding space. Novelty = cosine distance to
that neighbor.

Calibration: raw distances mean nothing alone, so we compute the same statistic for
500 held-out corpus items against the remaining corpus ("how far is a real human idea
from the nearest other real idea"). Model outputs are then judged relative to that
human baseline:

  - distance << baseline  → the "novel idea" already exists (retrieval/paraphrase)
  - distance ≈  baseline  → as novel as a typical real idea is from its neighbors
  - distance >> baseline  → extrapolation beyond the corpus (or off-manifold junk, so
                            cross-check M4 coherence scores from Phase 1)

Outputs: results/phase2/novelty.csv, novelty_dist.png, extremes.txt
(most/least novel generations with their nearest corpus neighbors, for reading).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results" / "phase2"
HOLDOUT = 500
SEED = 0


def load_jsonl(p):
    return [json.loads(l) for l in p.open()]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    corpus = {r["id"]: r for r in load_jsonl(ROOT / "data/phase2/corpus.jsonl")}
    emb_rows = load_jsonl(ROOT / "data/phase2/corpus_emb.jsonl")
    ids = [r["id"] for r in emb_rows]
    C = np.array([r["v"] for r in emb_rows], dtype=np.float32)  # normalized

    gens = load_jsonl(ROOT / "data/phase1/generations.jsonl")
    for g in gens:
        g["key"] = f'{g["model"]}|{g["config"]}|{g["index"]}'
    gens = list({g["key"]: g for g in gens}.values())
    gemb = {r["key"]: r["v"] for r in load_jsonl(ROOT / "data/phase1/embeddings.jsonl")}
    G = np.array([gemb[g["key"]] for g in gens], dtype=np.float32)

    # human baseline: held-out corpus items vs rest of corpus
    rng = np.random.default_rng(SEED)
    hold = rng.choice(len(C), size=HOLDOUT, replace=False)
    mask = np.ones(len(C), bool)
    mask[hold] = False
    S = C[hold] @ C[mask].T
    baseline = 1.0 - S.max(axis=1)

    # model generations vs full corpus
    sims = G @ C.T
    nn_idx = sims.argmax(axis=1)
    nn_dist = 1.0 - sims.max(axis=1)

    df = pd.DataFrame({
        "model": [g["model"] for g in gens],
        "config": [g["config"] for g in gens],
        "nn_dist": nn_dist,
        "nn_id": [ids[i] for i in nn_idx],
    })
    df.to_csv(RESULTS / "novelty.csv", index=False)

    base_med = float(np.median(baseline))
    print(f"human-baseline NN distance: median {base_med:.3f}, "
          f"p10 {np.percentile(baseline, 10):.3f}, p90 {np.percentile(baseline, 90):.3f}")
    summary = (df.groupby(["model", "config"]).nn_dist
               .agg(["median", "min", "max"]).round(3)
               .assign(frac_below_human_median=df.groupby(["model", "config"]).nn_dist
                       .apply(lambda s: float((s < base_med).mean())).round(2)))
    print(summary.to_string())
    summary.to_csv(RESULTS / "novelty_summary.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(baseline, bins=40, density=True, alpha=0.45,
            label=f"real ideas vs corpus (n={HOLDOUT})", color="gray")
    for model, g in df[df.config == "baseline_t0.7"].groupby("model"):
        ax.hist(g.nn_dist, bins=40, density=True, alpha=0.45, label=f"{model} (baseline cfg)")
    ax.axvline(base_med, color="k", ls="--", lw=1)
    ax.set_xlabel("nearest-neighbor cosine distance to reference corpus (higher = more novel)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title("M3 novelty: model 'original ideas' vs how novel real ideas are")
    fig.tight_layout()
    fig.savefig(RESULTS / "novelty_dist.png", dpi=150)

    # qualitative extremes for reading
    gdict = {g["key"]: g for g in gens}
    order = np.argsort(nn_dist)
    with (RESULTS / "extremes.txt").open("w") as f:
        for label, idxs in [("LEAST NOVEL (closest corpus match)", order[:10]),
                            ("MOST NOVEL (farthest from corpus)", order[-10:][::-1])]:
            f.write(f"========== {label} ==========\n\n")
            for i in idxs:
                g = gens[i]
                f.write(f"[{g['model']}/{g['config']}] dist={nn_dist[i]:.3f}\n"
                        f"GENERATION: {g['text'].strip()[:400]}\n"
                        f"NEAREST REAL IDEA ({ids[nn_idx[i]]}): "
                        f"{corpus[ids[nn_idx[i]]]['text'][:400]}\n\n")
    print(f"wrote {RESULTS}/novelty_summary.csv, novelty_dist.png, extremes.txt")


if __name__ == "__main__":
    main()
