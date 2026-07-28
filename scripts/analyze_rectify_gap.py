#!/usr/bin/env python3
"""Analyze verify vs rectify gap from PAG validation dumps.

Expected inputs (from trainer.save_validation_results=True):
  validation_data/<timestamp>/reward_extra_infos.pkl
  validation_data/<timestamp>/sample_inputs.pkl
  validation_data/<timestamp>/data_sources.pkl

Optional second dump with gate_mode=oracle / always for true oracle-revise upper bound.

Usage:
  python scripts/analyze_rectify_gap.py \
      --pag_dir validation_data/YYYYMMDD_HHMMSS \
      [--oracle_dir validation_data/...] \
      [--data_source HuggingFaceH4/MATH-500]
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


REQUIRED_KEYS = [
    "acc_t1",
    "acc_t2",
    "acc_final",
    "revised",
    "genrm_score",
    "genrm_pred",
]


def _load_dump(dump_dir: Path) -> pd.DataFrame:
    infos_path = dump_dir / "reward_extra_infos.pkl"
    inputs_path = dump_dir / "sample_inputs.pkl"
    sources_path = dump_dir / "data_sources.pkl"
    if not infos_path.exists():
        raise FileNotFoundError(f"Missing {infos_path}")

    with open(infos_path, "rb") as f:
        infos: Dict[str, Any] = pickle.load(f)

    missing = [k for k in REQUIRED_KEYS if k not in infos]
    if missing:
        raise KeyError(
            f"{dump_dir} missing keys {missing}. "
            "Re-run eval with updated pag.py exports."
        )

    n = len(infos["acc_t1"])
    df = pd.DataFrame({k: list(infos[k])[:n] for k in infos.keys() if len(infos[k]) == n})

    if inputs_path.exists():
        with open(inputs_path, "rb") as f:
            prompts = pickle.load(f)
        df["prompt"] = list(prompts)[:n]
    if sources_path.exists():
        with open(sources_path, "rb") as f:
            sources = pickle.load(f)
        df["data_source_file"] = list(np.asarray(sources).reshape(-1))[:n]

    # Prefer explicit export if present
    if "data_source" not in df.columns and "data_source_file" in df.columns:
        df["data_source"] = df["data_source_file"]

    df["acc_t1_bin"] = (df["acc_t1"].astype(float) >= 0.5).astype(int)
    df["acc_final_bin"] = (df["acc_final"].astype(float) >= 0.5).astype(int)
    df["revised_bin"] = df["revised"].astype(bool)
    df["verify_ok"] = (df["genrm_score"].astype(float) >= 0.5).astype(int)
    # acc_t2 == -1 means no revision
    df["has_t2"] = df["acc_t2"].astype(float) >= 0.0
    df["acc_t2_bin"] = np.where(df["has_t2"], (df["acc_t2"].astype(float) >= 0.5).astype(int), -1)
    return df


def _safe_rate(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return float(num) / float(den)


def summarize_pag(df: pd.DataFrame) -> Dict[str, Any]:
    """Metrics available from a PAG-gated dump alone."""
    n = len(df)
    W = df["acc_t1_bin"] == 0
    C = df["acc_t1_bin"] == 1
    R = df["revised_bin"]

    i_to_c = ((df["acc_t1_bin"] == 0) & R & (df["acc_t2_bin"] == 1)).sum()
    c_to_i = ((df["acc_t1_bin"] == 1) & R & (df["acc_t2_bin"] == 0)).sum()
    n_rev = int(R.sum())
    n_wrong = int(W.sum())
    n_correct = int(C.sum())
    n_wrong_rev = int((W & R).sum())
    n_correct_rev = int((C & R).sum())

    out = {
        "n": n,
        "acc_t1": float(df["acc_t1_bin"].mean()) if n else None,
        "acc_final": float(df["acc_final_bin"].mean()) if n else None,
        "delta_t1_final": float(df["acc_final_bin"].mean() - df["acc_t1_bin"].mean()) if n else None,
        "verify_acc": float(df["verify_ok"].mean()) if n else None,
        "revise_rate": float(R.mean()) if n else None,
        # Detection / gate quality on t1
        "coverage_wrong": _safe_rate((W & R).sum(), n_wrong),  # P(R|W)
        "false_revise_correct": _safe_rate((C & R).sum(), n_correct),  # P(R|C)
        # Paper-style mass (over all samples)
        "delta_i_to_c_mass": _safe_rate(i_to_c, n),
        "delta_c_to_i_mass": _safe_rate(c_to_i, n),
        # Paper-style among revised
        "delta_hat_i_to_c": _safe_rate(i_to_c, n_rev),
        "delta_hat_c_to_i": _safe_rate(c_to_i, n_rev),
        # Conditional rectify given PAG triggered revise on wrong t1
        "pag_i_to_c_given_wrong_revised": _safe_rate(i_to_c, n_wrong_rev),
        "pag_c_to_i_given_correct_revised": _safe_rate(c_to_i, n_correct_rev),
        "counts": {
            "n_wrong": n_wrong,
            "n_correct": n_correct,
            "n_revised": n_rev,
            "n_wrong_revised": n_wrong_rev,
            "n_correct_revised": n_correct_rev,
            "i_to_c": int(i_to_c),
            "c_to_i": int(c_to_i),
        },
        "note": (
            "pag_i_to_c_given_wrong_revised is NOT the oracle-revise upper bound; "
            "it conditions on GenRM deciding to revise."
        ),
    }
    return out


def summarize_oracle(df: pd.DataFrame) -> Dict[str, Any]:
    """True rectify upper bound: requires dump where all wrong_t1 were revised."""
    W = df["acc_t1_bin"] == 0
    # Prefer rows that actually have t2 for wrong t1
    wrong_with_t2 = W & df["has_t2"]
    i_to_c = (wrong_with_t2 & (df["acc_t2_bin"] == 1)).sum()
    n_wrong = int(W.sum())
    n_wrong_with_t2 = int(wrong_with_t2.sum())
    return {
        "n_wrong": n_wrong,
        "n_wrong_with_t2": n_wrong_with_t2,
        "oracle_i_to_c": _safe_rate(i_to_c, n_wrong_with_t2),
        "coverage_of_wrong": _safe_rate(n_wrong_with_t2, n_wrong),
        "note": (
            "oracle_i_to_c ≈ P(acc_t2=1 | acc_t1=0) only if coverage_of_wrong≈1 "
            "(every wrong t1 was force-revised)."
        ),
    }


def compare_gap(pag_stats: Dict[str, Any], oracle_stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    gap = {
        "verify_acc": pag_stats.get("verify_acc"),
        "pag_conditional_rectify": pag_stats.get("pag_i_to_c_given_wrong_revised"),
        "coverage_wrong": pag_stats.get("coverage_wrong"),
    }
    if oracle_stats is not None:
        gap["oracle_rectify_upper_bound"] = oracle_stats.get("oracle_i_to_c")
        pag_r = pag_stats.get("pag_i_to_c_given_wrong_revised")
        ora = oracle_stats.get("oracle_i_to_c")
        if pag_r is not None and ora is not None:
            gap["rectify_gap_oracle_minus_pag"] = ora - pag_r
        # Bottleneck heuristic
        if ora is not None and ora < 0.25:
            gap["bottleneck_guess"] = "RECTIFY (oracle upper bound still low)"
        elif pag_stats.get("coverage_wrong") is not None and pag_stats["coverage_wrong"] < 0.5:
            gap["bottleneck_guess"] = "DETECTION/COVERAGE (many wrong_t1 never revised)"
        elif pag_stats.get("false_revise_correct", 0) and pag_stats["false_revise_correct"] > 0.15:
            gap["bottleneck_guess"] = "OVER-REVISION (too many correct_t1 revised)"
        else:
            gap["bottleneck_guess"] = "MIXED / check calibration"
    else:
        gap["bottleneck_guess"] = (
            "UNKNOWN without oracle dump; only PAG-conditional rectify is available"
        )
    return gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pag_dir", type=str, required=True, help="validation_data/<ts> for PAG dump")
    parser.add_argument("--oracle_dir", type=str, default=None, help="optional oracle/always revise dump")
    parser.add_argument(
        "--data_source",
        type=str,
        default=None,
        help="optional substring filter, e.g. MATH-500",
    )
    parser.add_argument("--out_json", type=str, default=None, help="write summary json")
    args = parser.parse_args()

    pag_df = _load_dump(Path(args.pag_dir))
    if args.data_source:
        mask = pag_df.get("data_source", pd.Series([""] * len(pag_df))).astype(str).str.contains(
            args.data_source, regex=False
        )
        if "data_source_file" in pag_df.columns:
            mask = mask | pag_df["data_source_file"].astype(str).str.contains(args.data_source, regex=False)
        pag_df = pag_df[mask].reset_index(drop=True)

    pag_stats = summarize_pag(pag_df)
    oracle_stats = None
    if args.oracle_dir:
        oracle_df = _load_dump(Path(args.oracle_dir))
        if args.data_source:
            mask = oracle_df.get("data_source", pd.Series([""] * len(oracle_df))).astype(str).str.contains(
                args.data_source, regex=False
            )
            if "data_source_file" in oracle_df.columns:
                mask = mask | oracle_df["data_source_file"].astype(str).str.contains(
                    args.data_source, regex=False
                )
            oracle_df = oracle_df[mask].reset_index(drop=True)
        oracle_stats = summarize_oracle(oracle_df)

    summary = {
        "pag": pag_stats,
        "oracle": oracle_stats,
        "gap": compare_gap(pag_stats, oracle_stats),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
