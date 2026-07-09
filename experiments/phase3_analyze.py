"""Phase 3 analysis: aggregate results/phase3/run_*.json into emergence curves.

Plots vs model size (log x):
  1. novel_valid_precision (M5, discontinuous headline) with chance floor 0.16
  2. L1/L2 sample rates + train-dup (memorization) rate
  3. log-prob gap valid-vs-invalid (continuous companion, H5 mirage check)
  4. unique held-out coverage

Writes results/phase3/emergence.png and summary.csv.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).parent.parent / "results" / "phase3"
CHANCE = 0.16  # valid fraction of triple space


def main():
    rows = []
    for f in sorted(RESULTS.glob("run_*.json")):
        r = json.loads(f.read_text())
        rows.append({
            "size": r["size"], "n_params": r["n_params"], "seed": r["seed"],
            "precision": r["metrics"]["novel_valid_precision"],
            "l1_rate": r["metrics"]["rate"]["l1"],
            "l2_rate": r["metrics"]["rate"]["l2"],
            "dup_rate": r["metrics"]["rate"]["train_dup"],
            "unparse": r["metrics"]["rate"]["unparseable"],
            "grammar": r["metrics"]["grammar_rate"],
            "lp_gap": r["logprob"]["gap"],
            "l1_cov": r["metrics"]["l1_coverage"],
            "l2_cov": r["metrics"]["l2_coverage"],
        })
    df = pd.DataFrame(rows)
    agg = df.groupby(["size", "n_params"]).agg(["mean", "std"]).round(4)
    agg.to_csv(RESULTS / "summary.csv")
    g = df.groupby("n_params")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = sorted(df.n_params.unique())

    def curve(ax, col, label, **kw):
        m, s = g[col].mean().loc[x], g[col].std().loc[x].fillna(0)
        ax.errorbar(x, m, yerr=s, marker="o", capsize=3, label=label, **kw)

    ax = axes[0][0]
    curve(ax, "precision", "novel outputs obeying hidden rule (M5)")
    ax.axhline(CHANCE, color="r", ls="--", lw=1, label="chance floor (16%)")
    ax.set_title("Provable combinational novelty precision")
    ax = axes[0][1]
    curve(ax, "l1_rate", "L1 held-out (novel triple)")
    curve(ax, "l2_rate", "L2 held-out (novel pair)")
    curve(ax, "dup_rate", "training duplicate (memorized)")
    curve(ax, "unparse", "unparseable")
    ax.set_title("Output composition per 4,000 samples")
    ax = axes[1][0]
    curve(ax, "lp_gap", "mean log-prob(valid held-out) − log-prob(invalid)")
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_title("Continuous companion metric (mirage check, H5)")
    ax = axes[1][1]
    curve(ax, "l1_cov", "share of all 1,840 L1 triples produced")
    curve(ax, "l2_cov", "share of all 1,600 L2 triples produced")
    ax.set_title("Held-out space coverage")

    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlabel("parameters")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Phase 3: provable originality vs model scale "
                 "(3 seeds/size, 6k steps, closed synthetic domain)")
    fig.tight_layout()
    fig.savefig(RESULTS / "emergence.png", dpi=150)
    print(agg[[("precision", "mean"), ("lp_gap", "mean"), ("dup_rate", "mean")]]
          .to_string())
    print(f"\nwrote {RESULTS}/emergence.png, summary.csv")


if __name__ == "__main__":
    main()
