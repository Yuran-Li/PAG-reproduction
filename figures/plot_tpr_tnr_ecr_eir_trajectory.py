#!/usr/bin/env python3
"""Plot PAG training dynamics: TPR / TNR / ECR_TP / EIR_FP vs step.

Notation (v=1 means verifier rejects / flags incorrect):
  TPR     = P(v=1 | a1=0)   error recall
  TNR     = P(v=0 | a1=1)   correct-answer retention
  ECR_TP  = P(a2=1 | a1=0, revised)   ≈ i_to_c_rate
  EIR_FP  = P(a2=0 | a1=1, revised)   ≈ c_to_i_rate

Source: MATH-500 val during training (n=8, every 10 steps).
Defaults to reading the extracted JSON next to this script; optionally
re-parse a training log with --log.

Usage (from repo root or this directory):

  # Default: load pag_tpr_tnr_ecr_eir_trajectory.json and write png/pdf
  python figures/plot_tpr_tnr_ecr_eir_trajectory.py

  # Re-extract from a training log, update json, then plot
  python figures/plot_tpr_tnr_ecr_eir_trajectory.py \\
    --log logs/pag_0720_1501.log --save-json

  # Custom output prefix
  python figures/plot_tpr_tnr_ecr_eir_trajectory.py \\
    --out figures/pag_tpr_tnr_ecr_eir_trajectory
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_JSON = HERE / "pag_tpr_tnr_ecr_eir_trajectory.json"
DEFAULT_LOG = (
    HERE.parent / "logs" / "pag_0720_1501.log"
)


def parse_log(log_path: Path) -> list[dict]:
    text = log_path.read_text(errors="ignore")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    rows: list[dict] = []
    for m in re.finditer(r"step:(\d+)\s+-\s+(.*?)$", text, re.M):
        step = int(m.group(1))
        blob = m.group(2)
        if "val/multiturn/verify_recall:" not in blob:
            continue

        def grab(key: str) -> float | None:
            mm = re.search(rf"{re.escape(key)}:([0-9.eE+-]+)", blob)
            return float(mm.group(1)) if mm else None

        tp = grab("val/multiturn/verify_TP")
        fp = grab("val/multiturn/verify_FP")  # PAG name: both wrong
        fn = grab("val/multiturn/verify_FN")
        tn = grab("val/multiturn/verify_TN")  # PAG name: say-correct & wrong
        ecr = grab("val/multiturn/i_to_c_rate")
        eir = grab("val/multiturn/c_to_i_rate")
        if None in (tp, fp, fn, tn, ecr, eir):
            continue
        # v=1 = reject: TPR = FP/(FP+TN), TNR = TP/(TP+FN) under PAG field names
        tpr = fp / (fp + tn) if (fp + tn) > 0 else None
        tnr = tp / (tp + fn) if (tp + fn) > 0 else None
        if tpr is None or tnr is None:
            continue
        rows.append(
            {
                "step": step,
                "TPR": round(tpr, 6),
                "TNR": round(tnr, 6),
                "ECR_TP": round(ecr, 6),
                "EIR_FP": round(eir, 6),
                "A1": grab("val/multiturn/turn_1_accuracy"),
                "A2": grab("val/multiturn/final_acc"),
            }
        )
    by_step = {r["step"]: r for r in rows}
    return sorted(by_step.values(), key=lambda r: r["step"])


def load_data(json_path: Path | None, log_path: Path | None) -> list[dict]:
    if log_path is not None:
        data = parse_log(log_path)
        if not data:
            raise SystemExit(f"No val metrics parsed from {log_path}")
        return data
    path = json_path or DEFAULT_JSON
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Pass --log to re-extract from a training log."
        )
    return json.loads(path.read_text())


def _padded_ylim(
    vals: list[float],
    *,
    y_min: float | None = None,
    y_max: float | None = None,
    pad_frac: float = 0.08,
) -> tuple[float, float]:
    """Zoom y-range to data with padding; optional hard floor/ceiling."""
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-3)
    pad = span * pad_frac
    y0, y1 = lo - pad, hi + pad
    if y_min is not None:
        y0 = y_min
    if y_max is not None:
        y1 = max(y_max, hi + pad)
    return y0, y1


def plot(data: list[dict], out_prefix: Path) -> None:
    steps = [d["step"] for d in data]
    tpr = [d["TPR"] for d in data]
    tnr = [d["TNR"] for d in data]
    ecr = [d["ECR_TP"] for d in data]
    eir = [d["EIR_FP"] for d in data]

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9.0,
            "axes.spines.top": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharex=False)
    x_max = max(steps)
    x_lim = (0, x_max + 10)
    x_ticks = list(range(0, x_max + 1, 50))
    if x_ticks[-1] != x_max and x_max - x_ticks[-1] > 20:
        x_ticks.append(((x_max + 9) // 10) * 10)

    # --- Left: detection, shared y-axis (zoomed, not 0–1) ---
    ax = axes[0]
    ax.spines["right"].set_visible(False)
    ax.plot(
        steps, tpr, color="#1f77b4", lw=2.0, marker="o", ms=3.0, markevery=2,
        label=r"TPR ",
    )
    ax.plot(
        steps, tnr, color="#2ca02c", lw=2.0, marker="s", ms=3.0, markevery=2,
        label=r"TNR",
    )
    ax.set_ylim(*_padded_ylim(tpr + tnr, y_min=0.20, y_max=1.00))
    ax.set_xlim(*x_lim)
    ax.set_xticks(x_ticks)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Rate")
    ax.set_title("Verification (detection)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, axis="both", alpha=0.25)

    # --- Right: repair, zoomed so ECR rise is visible ---
    ax = axes[1]
    ax.spines["right"].set_visible(False)
    ax.plot(
        steps, ecr, color="#d62728", lw=2.0, marker="o", ms=3.0, markevery=2,
        label=r"ECR$_{\mathrm{TP}}$",
    )
    ax.plot(
        steps, eir, color="#ff7f0e", lw=2.0, marker="s", ms=3.0, markevery=2,
        label=r"EIR$_{\mathrm{FP}}$",
    )
    ax.set_ylim(*_padded_ylim(ecr + eir, y_min=0.0, y_max=0.48))
    ax.set_xlim(*x_lim)
    ax.set_xticks(x_ticks)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Rate")
    ax.set_title("Repair execution (given revision)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="both", alpha=0.25)
    fig.suptitle("PAG training dynamics on MATH-500 (val, n=8)", y=1.02, fontsize=13)
    fig.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    pdf = out_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png}")
    print(f"saved {pdf}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_JSON,
        help=f"trajectory json (default: {DEFAULT_JSON.name})",
    )
    ap.add_argument(
        "--log",
        type=Path,
        default=None,
        help=f"optional training log to re-extract (e.g. {DEFAULT_LOG})",
    )
    ap.add_argument(
        "--save-json",
        action="store_true",
        help="when using --log, also write/update the trajectory json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=HERE / "pag_tpr_tnr_ecr_eir_trajectory",
        help="output path prefix (without extension)",
    )
    args = ap.parse_args()

    data = load_data(args.json if args.log is None else None, args.log)
    if args.log is not None and args.save_json:
        args.json.write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote {args.json} (n={len(data)})")
    plot(data, args.out)


if __name__ == "__main__":
    main()
