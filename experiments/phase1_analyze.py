"""Phase 1 analysis.

Reads data/phase1/generations.jsonl, embeds every response with Titan v2, then per
(model, config) cell computes the diversity metrics fixed in docs/01_definitions.md:

  M1 output diversity  — mean pairwise cosine distance; Vendi score (effective number
                         of distinct ideas: exp of the von Neumann entropy of the
                         normalized similarity matrix)
  M2 mode-collapse     — fraction of samples in the largest agglomerative cluster
                         (average-linkage, cosine distance threshold 0.45)

Outputs:
  data/phase1/embeddings.jsonl               (cached; reruns are incremental)
  results/phase1/metrics.csv
  results/phase1/collapse_heatmap.png, vendi_heatmap.png
  results/phase1/clusters_<model>_<config>.txt   (cluster sizes + one representative
                                                  text each, for qualitative reading)
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from original_thought.bedrock import embed

ROOT = Path(__file__).parent.parent
GEN = ROOT / "data" / "phase1" / "generations.jsonl"
EMB = ROOT / "data" / "phase1" / "embeddings.jsonl"
RESULTS = ROOT / "results" / "phase1"

CLUSTER_DIST = 0.45  # cosine distance threshold: below this, two ideas are "the same"


def load_generations() -> pd.DataFrame:
    rows = [json.loads(l) for l in GEN.open()]
    df = pd.DataFrame(rows)
    df["key"] = df.model + "|" + df.config + "|" + df["index"].astype(str)
    return df.drop_duplicates("key")


def ensure_embeddings(df: pd.DataFrame) -> dict[str, np.ndarray]:
    cache = {}
    if EMB.exists():
        for l in EMB.open():
            r = json.loads(l)
            cache[r["key"]] = np.array(r["v"], dtype=np.float32)
    todo = df[~df.key.isin(cache)]
    if len(todo):
        print(f"embedding {len(todo)} new texts")
        with EMB.open("a") as f, ThreadPoolExecutor(max_workers=8) as ex:
            for key, v in zip(todo.key, ex.map(embed, todo.text)):
                cache[key] = np.array(v, dtype=np.float32)
                f.write(json.dumps({"key": key, "v": v}) + "\n")
    return cache


def vendi_score(X: np.ndarray) -> float:
    """Effective number of distinct items (Friedman & Dieng 2023)."""
    n = len(X)
    K = X @ X.T / n  # X rows are L2-normalized (Titan normalize=True)
    w = np.linalg.eigvalsh(K)
    w = w[w > 1e-12]
    return float(np.exp(-(w * np.log(w)).sum()))


def cell_metrics(X: np.ndarray, texts: list[str]):
    n = len(X)
    if n < 5:  # too few samples for meaningful stats (partial sweep)
        return None, []
    S = X @ X.T
    D = 1.0 - S
    mean_pdist = float(D[np.triu_indices(n, k=1)].mean())
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=CLUSTER_DIST,
        metric="precomputed", linkage="average",
    ).fit_predict(np.clip(D, 0, None))
    sizes = pd.Series(labels).value_counts()
    # representative per cluster = member closest to the cluster centroid
    reps = []
    for lab in sizes.index:
        idx = np.where(labels == lab)[0]
        centroid = X[idx].mean(0)
        reps.append((int(sizes[lab]), texts[idx[np.argmax(X[idx] @ centroid)]]))
    return {
        "n": n,
        "mean_pairwise_dist": round(mean_pdist, 4),
        "vendi": round(vendi_score(X), 2),
        "n_clusters": int(len(sizes)),
        "collapse_ratio": round(float(sizes.iloc[0]) / n, 3),
    }, reps


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_generations()
    print(f"{len(df)} generations across {df.model.nunique()} models × {df.config.nunique()} configs")
    cache = ensure_embeddings(df)

    rows = []
    for (model, config), g in df.groupby(["model", "config"]):
        X = np.stack([cache[k] for k in g.key])
        m, reps = cell_metrics(X, list(g.text))
        if m is None:
            print(f"skipping {model}/{config}: only {len(g)} samples")
            continue
        rows.append({"model": model, "config": config, **m})
        with (RESULTS / f"clusters_{model}_{config}.txt").open("w") as f:
            f.write(f"{model} / {config}: {m['n_clusters']} clusters over {m['n']} samples, "
                    f"collapse={m['collapse_ratio']}, vendi={m['vendi']}\n\n")
            for size, text in reps:
                f.write(f"--- cluster of {size} ---\n{text.strip()[:600]}\n\n")

    met = pd.DataFrame(rows).sort_values(["model", "config"])
    met.to_csv(RESULTS / "metrics.csv", index=False)
    print(met.to_string(index=False))

    for col, title, fname in [
        ("collapse_ratio", "M2 mode-collapse ratio (share of samples in largest idea-cluster)", "collapse_heatmap.png"),
        ("vendi", "M1 Vendi score (effective number of distinct ideas)", "vendi_heatmap.png"),
    ]:
        piv = met.pivot(index="model", columns="config", values=col)
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(piv.index)), piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center",
                        color="white", fontsize=9)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(RESULTS / fname, dpi=150)
    print(f"wrote {RESULTS}/metrics.csv and heatmaps")


if __name__ == "__main__":
    main()
