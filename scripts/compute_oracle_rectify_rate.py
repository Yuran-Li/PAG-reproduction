#!/usr/bin/env python3
"""Compute oracle-revise rectification upper bound from a validation dump.

Oracle dump must be produced with:
  actor_rollout_ref.rollout.val_kwargs.revise_gate=oracle
  trainer.save_validation_results=True

Definition
----------
  Oracle rectify rate = P(acc_t2=1 | acc_t1=0)
  i.e. among ALL turn-1 wrong samples that were force-revised, fraction correct at t2.

Also reports PAG-style conditional rates if you pass a PAG dump instead / as comparison.

Usage
-----
  # After oracle eval dump:
  python scripts/compute_oracle_rectify_rate.py \
      --dump_dir validation_data/<timestamp> \
      --data_source MATH-500

  # Compare PAG dump vs oracle dump:
  python scripts/compute_oracle_rectify_rate.py \
      --dump_dir validation_data/<pag_ts> \
      --oracle_dir validation_data/<oracle_ts> \
      --data_source MATH-500
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


NEEDED = ["acc_t1", "acc_t2", "acc_final", "revised", "genrm_score"]


def load_dump(dump_dir: Path) -> pd.DataFrame:
    infos_path = dump_dir / "reward_extra_infos.pkl"
    if not infos_path.exists():
        raise FileNotFoundError(
            f"Missing {infos_path}. Re-run eval with trainer.save_validation_results=True "
            "and updated pag.py exports (acc_t1/acc_t2/revised)."
        )
    with open(infos_path, "rb") as f:
        infos = pickle.load(f)
    missing = [k for k in NEEDED if k not in infos]
    if missing:
        raise KeyError(f"{dump_dir} missing {missing}")

    n = len(infos["acc_t1"])
    df = pd.DataFrame({k: list(infos[k])[:n] for k in infos if len(infos[k]) == n})

    src_path = dump_dir / "data_sources.pkl"
    if src_path.exists():
        with open(src_path, "rb") as f:
            df["data_source_file"] = list(np.asarray(pickle.load(f)).reshape(-1))[:n]
    if "data_source" not in df.columns and "data_source_file" in df.columns:
        df["data_source"] = df["data_source_file"]

    df["t1_ok"] = df["acc_t1"].astype(float) >= 0.5
    df["final_ok"] = df["acc_final"].astype(float) >= 0.5
    df["revised"] = df["revised"].astype(bool)
    df["has_t2"] = df["acc_t2"].astype(float) >= 0.0
    df["t2_ok"] = np.where(df["has_t2"], df["acc_t2"].astype(float) >= 0.5, False)
    df["verify_ok"] = df["genrm_score"].astype(float) >= 0.5
    return df


def filter_source(df: pd.DataFrame, data_source: Optional[str]) -> pd.DataFrame:
    if not data_source:
        return df
    mask = pd.Series([False] * len(df))
    for col in ("data_source", "data_source_file"):
        if col in df.columns:
            mask = mask | df[col].astype(str).str.contains(data_source, regex=False)
    return df[mask].reset_index(drop=True)


def rate(num: int, den: int) -> Optional[float]:
    return None if den == 0 else float(num) / float(den)


def oracle_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Upper bound assuming dump used revise_gate=oracle (revise iff t1 wrong)."""
    wrong = ~df["t1_ok"]
    wrong_rev = wrong & df["revised"] & df["has_t2"]
    i2c = int((wrong_rev & df["t2_ok"]).sum())
    n_wrong = int(wrong.sum())
    n_wrong_rev = int(wrong_rev.sum())

    # Sanity: under true oracle, almost all wrong should be revised
    coverage = rate(n_wrong_rev, n_wrong)
    return {
        "n": int(len(df)),
        "n_wrong_t1": n_wrong,
        "n_wrong_revised": n_wrong_rev,
        "n_i_to_c": i2c,
        "coverage_wrong": coverage,  # should be ~1.0 for oracle gate
        "oracle_rectify_rate": rate(i2c, n_wrong_rev),  # P(t2 ok | t1 wrong, revised)
        "oracle_rectify_rate_over_all_wrong": rate(i2c, n_wrong),
        "acc_t1": float(df["t1_ok"].mean()) if len(df) else None,
        "acc_final": float(df["final_ok"].mean()) if len(df) else None,
        "delta_t1_final": float(df["final_ok"].mean() - df["t1_ok"].mean()) if len(df) else None,
        "verify_acc": float(df["verify_ok"].mean()) if len(df) else None,
        "warning": None
        if coverage is not None and coverage >= 0.95
        else (
            f"coverage_wrong={coverage:.3f} << 1: this dump may not be revise_gate=oracle "
            "(many wrong_t1 never revised). Rate is then only conditional on revised wrongs."
        ),
    }


def pag_conditional_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """PAG-gated conditional repair (already known from aggregate logs)."""
    wrong = ~df["t1_ok"]
    correct = df["t1_ok"]
    rev = df["revised"]
    wrong_rev = wrong & rev & df["has_t2"]
    correct_rev = correct & rev & df["has_t2"]
    i2c = int((wrong_rev & df["t2_ok"]).sum())
    c2i = int((correct_rev & ~df["t2_ok"]).sum())
    n_rev = int(rev.sum())
    return {
        "n": int(len(df)),
        "verify_acc": float(df["verify_ok"].mean()) if len(df) else None,
        "coverage_wrong_P_R_given_W": rate(int(wrong_rev.sum()), int(wrong.sum())),
        "false_revise_P_R_given_C": rate(int(correct_rev.sum()), int(correct.sum())),
        "pag_conditional_rectify_P_t2ok_given_W_and_R": rate(i2c, int(wrong_rev.sum())),
        "delta_hat_i_to_c": rate(i2c, n_rev),
        "delta_hat_c_to_i": rate(c2i, n_rev),
        "delta_i_to_c_mass": rate(i2c, len(df)),
        "delta_c_to_i_mass": rate(c2i, len(df)),
        "acc_t1": float(df["t1_ok"].mean()) if len(df) else None,
        "acc_final": float(df["final_ok"].mean()) if len(df) else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump_dir", type=str, required=True, help="primary dump (oracle preferred)")
    ap.add_argument("--oracle_dir", type=str, default=None, help="optional explicit oracle dump")
    ap.add_argument("--pag_dir", type=str, default=None, help="optional PAG dump for side-by-side")
    ap.add_argument("--data_source", type=str, default="MATH-500")
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument(
        "--assume_oracle",
        action="store_true",
        help="treat --dump_dir as oracle even if coverage_wrong < 0.95",
    )
    args = ap.parse_args()

    primary = filter_source(load_dump(Path(args.dump_dir)), args.data_source)
    oracle_df = primary
    if args.oracle_dir:
        oracle_df = filter_source(load_dump(Path(args.oracle_dir)), args.data_source)

    ostats = oracle_stats(oracle_df)
    if not args.assume_oracle and ostats["warning"]:
        # If user pointed a PAG dump at --dump_dir, still print PAG conditional clearly.
        pass

    summary: Dict[str, Any] = {
        "data_source_filter": args.data_source,
        "oracle_upper_bound": ostats,
    }

    pag_path = args.pag_dir
    if pag_path:
        pstats = pag_conditional_stats(filter_source(load_dump(Path(pag_path)), args.data_source))
        summary["pag_conditional"] = pstats
        ora = ostats.get("oracle_rectify_rate")
        pag = pstats.get("pag_conditional_rectify_P_t2ok_given_W_and_R")
        if ora is not None and pag is not None:
            summary["gap"] = {
                "oracle_minus_pag_conditional": ora - pag,
                "verify_acc": pstats.get("verify_acc"),
                "interpretation": (
                    "If oracle_rectify_rate stays low (~verify high), bottleneck is RECTIFY. "
                    "If oracle high but PAG conditional/coverage low, bottleneck is GATE/DETECTION."
                ),
            }

    # Pretty print key lines
    print("=== Oracle revise upper bound ===")
    print(f"filter: {args.data_source}")
    print(f"n={ostats['n']}  wrong_t1={ostats['n_wrong_t1']}  wrong_revised={ostats['n_wrong_revised']}")
    print(f"coverage_wrong={ostats['coverage_wrong']}")
    print(f"oracle_rectify_rate P(t2=1|t1=0,revised) = {ostats['oracle_rectify_rate']}")
    print(f"acc_t1={ostats['acc_t1']}  acc_final={ostats['acc_final']}  delta={ostats['delta_t1_final']}")
    if ostats["warning"]:
        print(f"WARNING: {ostats['warning']}")
    if "pag_conditional" in summary:
        p = summary["pag_conditional"]
        print("\n=== PAG conditional (for comparison) ===")
        print(f"verify_acc={p['verify_acc']}")
        print(f"P(R|W)={p['coverage_wrong_P_R_given_W']}  P(R|C)={p['false_revise_P_R_given_C']}")
        print(f"P(t2=1|W,R)={p['pag_conditional_rectify_P_t2ok_given_W_and_R']}")
        if "gap" in summary:
            print(f"\noracle - pag_conditional = {summary['gap']['oracle_minus_pag_conditional']}")

    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
