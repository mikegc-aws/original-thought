"""Phase 4 analysis: fine-tuning-arm trajectories from results/phase4/traj_*.json.

Three panels vs fine-tune step (mean over seeds, shaded min-max):
  1. training-duplicate rate (collapse)
  2. unique held-out valid triples per 4k samples (expressed novelty)
  3. novel-valid precision (rule adherence — does diversity training destroy it?)

Writes results/phase4/trajectories.png and endpoint summary to stdout/csv.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).parent.parent / "results" / "phase4"
COLORS = {"control": "tab:gray", "preference": "tab:red", "diversity": "tab:green"}


def main():
    rows = []
    for f in sorted(RESULTS.glob("traj_*.json")):
        r = json.loads(f.read_text())
        for p in r["traj"]:
            rows.append({"arm": r["arm"], "seed": r["seed"], "step": p["step"],
                         "dup": p["rate"]["train_dup"],
                         "uniq": p["unique_heldout"],
                         "prec": p["novel_valid_precision"],
                         "unparse": p["rate"]["unparseable"]})
    df = pd.DataFrame(rows)

    panels = [("dup", "training-duplicate rate (collapse)"),
              ("uniq", "unique held-out valid triples / 4k samples"),
              ("prec", "novel-valid precision (rule adherence)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (col, title) in zip(axes, panels):
        for arm, g in df.groupby("arm"):
            piv = g.pivot(index="step", columns="seed", values=col)
            ax.plot(piv.index, piv.mean(1), marker="o", color=COLORS[arm], label=arm)
            ax.fill_between(piv.index, piv.min(1), piv.max(1), color=COLORS[arm], alpha=0.15)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("fine-tune step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Phase 4: is collapse trained-in and reversible? "
                 "(s4 base, 3 arms from identical weights)")
    fig.tight_layout()
    fig.savefig(RESULTS / "trajectories.png", dpi=150)

    end = (df[df.step == df.step.max()]
           .groupby("arm")[["dup", "uniq", "prec", "unparse"]].mean().round(3))
    start = (df[df.step == 0]
             .groupby("arm")[["dup", "uniq", "prec"]].mean().round(3))
    print("baseline (step 0, shared weights):")
    print(start.head(1).to_string())
    print("\nendpoints (step %d):" % df.step.max())
    print(end.to_string())
    end.to_csv(RESULTS / "endpoints.csv")
    print(f"\nwrote {RESULTS}/trajectories.png, endpoints.csv")


if __name__ == "__main__":
    main()
