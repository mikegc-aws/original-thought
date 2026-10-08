"""Phase 3 scale sweep: run phase3_train.py across all sizes × seeds sequentially.

Sizes span ~7k → ~15M params. 3 seeds per size for error bars.
Skips (size, seed) runs whose result JSON already exists, so the sweep is resumable.

Usage: uv run python experiments/phase3_sweep.py [--steps 6000] [--seeds 3]
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SIZES = ["s0", "s1", "s2", "s3", "s4", "s5", "s6"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    for size in SIZES:
        for seed in range(args.seeds):
            out = ROOT / "results" / "phase3" / f"run_{size}_seed{seed}.json"
            if out.exists():
                print(f"skip {size} seed{seed} (done)")
                continue
            cmd = [sys.executable, str(ROOT / "experiments" / "phase3_train.py"),
                   "--size", size, "--steps", str(args.steps), "--seed", str(seed)]
            if args.device:
                cmd += ["--device", args.device]
            print(">>>", " ".join(cmd[1:]))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
