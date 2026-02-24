#!/usr/bin/env python3
import argparse
import os
import re
from typing import Dict, List, Set, Tuple

import pandas as pd

LABEL_CANON = {
    "direct": "directly addressed",
    "directly addressed": "directly addressed",
    "directlyaddressed": "directly addressed",
    "partial": "partially addressed",
    "partially addressed": "partially addressed",
    "partiallyaddressed": "partially addressed",
    "not addressed": "not addressed",
    "notaddressed": "not addressed",
    "not addressed.": "not addressed",
    "not_addressed": "not addressed",
    "nr": "not addressed",
    "pr": "partially addressed",
    "r": "directly addressed",
}

THEME_COL_EXCLUDE = {
    "authors", "title", "year", "doi", "source", "database", "journal",
    "abstract", "keywords", "link", "url", "paper_id", "id"
}


def normalize_label(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("-", " ")
    return LABEL_CANON.get(s, s)


def infer_id_col(df: pd.DataFrame, user_id_col: str | None) -> str:
    if user_id_col and user_id_col in df.columns:
        return user_id_col
    for c in ["Authors", "authors", "PaperID", "paper_id", "Title", "title", "ID", "id"]:
        if c in df.columns:
            return c
    return "__index__"


def is_score_col(col: str) -> bool:
    c = col.strip().lower()
    return col.endswith(" (score)") or col.endswith("(score)") or c.endswith("_score")


def is_sentence_col(col: str) -> bool:
    return col.endswith(" (sentence)") or col.endswith("(sentence)")


def detect_theme_label_cols(df: pd.DataFrame, id_col: str) -> List[str]:
    """
    Detect theme label columns (values like directly/partially/not addressed).
    Ignores score and sentence columns and common metadata fields.
    """
    candidate_cols: List[str] = []
    for col in df.columns:
        if col == id_col:
            continue
        if is_score_col(col) or is_sentence_col(col):
            continue
        if col.strip().lower() in THEME_COL_EXCLUDE:
            continue

        sample = df[col].dropna().astype(str).head(30).tolist()
        if not sample:
            continue
        normed = {normalize_label(v) for v in sample}
        if {"directly addressed", "partially addressed", "not addressed"} & normed:
            candidate_cols.append(col)

    # fallback: pair with score cols if present
    if not candidate_cols:
        score_cols = [c for c in df.columns if is_score_col(c)]
        for sc in score_cols:
            base = sc.replace(" (score)", "").replace("(score)", "").strip()
            if base in df.columns and not is_sentence_col(base):
                candidate_cols.append(base)

    return candidate_cols


def top_themes_by_direct(df: pd.DataFrame, theme_cols: List[str], topk: int = 5) -> List[str]:
    direct_counts = {t: int((df[t] == "directly addressed").sum()) for t in theme_cols}
    ranked = sorted(direct_counts.items(), key=lambda x: (-x[1], x[0]))
    return [t for t, _ in ranked[:topk]]


def theme_stability_label(baseline_top: List[str], setting_top: List[str]) -> str:
    if setting_top == baseline_top:
        return "Top themes unchanged"
    if set(setting_top) == set(baseline_top):
        return "Top themes unchanged; minor rank swap"
    return "Top themes changed"


def load_and_prepare(path: str, id_col_hint: str | None) -> Tuple[pd.DataFrame, str, List[str]]:
    df = pd.read_csv(path)
    id_col = infer_id_col(df, id_col_hint)
    if id_col == "__index__":
        df[id_col] = df.index.astype(str)

    theme_cols = detect_theme_label_cols(df, id_col)
    if not theme_cols:
        raise ValueError(f"Could not detect theme label columns in {path}. Check the CSV format.")

    for c in theme_cols:
        df[c] = df[c].apply(normalize_label)

    return df, id_col, theme_cols


def overlap_pct(setting_direct: Set[str], baseline_direct: Set[str]) -> float:
    if not baseline_direct:
        return 0.0
    return 100.0 * len(setting_direct & baseline_direct) / len(baseline_direct)


def jaccard(setting_direct: Set[str], baseline_direct: Set[str]) -> float:
    union = setting_direct | baseline_direct
    if not union:
        return 0.0
    return len(setting_direct & baseline_direct) / len(union)


def pretty_name_from_filename(fn: str) -> str:
    base = os.path.basename(fn)
    base = re.sub(r"^classified_papers_semantic_weighted_?", "", base, flags=re.I)
    base = re.sub(r"\.csv$", "", base, flags=re.I)
    if base == "" or base.lower() == "weighted":
        return "Baseline"
    return base.replace("_", " ").strip()


def label_totals(df: pd.DataFrame, theme_cols: List[str]) -> Dict[str, int]:
    direct = int((df[theme_cols] == "directly addressed").sum().sum())
    partial = int((df[theme_cols] == "partially addressed").sum().sum())
    nr = int((df[theme_cols] == "not addressed").sum().sum())
    return {"Direct labels": direct, "Partial labels": partial, "Not addressed labels": nr}


def paper_level_pools(df: pd.DataFrame, theme_cols: List[str]) -> Dict[str, int]:
    has_direct = df[theme_cols].eq("directly addressed").any(axis=1)
    has_partial = df[theme_cols].eq("partially addressed").any(axis=1)
    direct_papers = int(has_direct.sum())
    partial_only_papers = int((~has_direct & has_partial).sum())
    return {"Direct papers": direct_papers, "Partial-only papers": partial_only_papers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Path to baseline CSV")
    ap.add_argument("--files", nargs="*", default=[], help="Paths to other CSVs")
    ap.add_argument("--id-col", default=None, help="Paper identifier column name (default: auto-detect)")
    ap.add_argument("--topk", type=int, default=5, help="Top-k themes for stability (default: 5)")
    ap.add_argument("--out", default="sensitivity_outcomes_option1.csv", help="Output CSV path")
    args = ap.parse_args()

    # Baseline
    base_df, base_id, base_themes = load_and_prepare(args.baseline, args.id_col)
    base_has_direct = base_df[base_themes].eq("directly addressed").any(axis=1)
    base_direct_set = set(base_df.loc[base_has_direct, base_id].astype(str))
    base_top = top_themes_by_direct(base_df, base_themes, topk=args.topk)

    rows: List[Dict[str, object]] = []

    def add_row(setting_name: str, df: pd.DataFrame, id_col: str, themes: List[str], is_baseline=False):
        # align themes to baseline if possible
        common_themes = [t for t in base_themes if t in themes] if not is_baseline else base_themes
        if not common_themes:
            common_themes = themes

        # totals across papers × themes
        totals = label_totals(df, common_themes)

        # paper-level pools (used for overlap)
        pools = paper_level_pools(df, common_themes)
        has_direct = df[common_themes].eq("directly addressed").any(axis=1)
        direct_set = set(df.loc[has_direct, id_col].astype(str))

        ov = 100.0 if is_baseline else overlap_pct(direct_set, base_direct_set)
        jac = 1.0 if is_baseline else jaccard(direct_set, base_direct_set)

        top = top_themes_by_direct(df, common_themes, topk=args.topk)
        stability = "Reference" if is_baseline else theme_stability_label(base_top, top)

        rows.append({
            "Setting": setting_name,
            **pools,
            **totals,
            "Overlap vs baseline (%)": round(ov, 1),
            "Jaccard vs baseline": round(jac, 3),
            "Theme stability": stability,
            f"Top{args.topk} themes": " | ".join(top),
        })

    add_row("Baseline", base_df, base_id, base_themes, is_baseline=True)

    for f in args.files:
        df, id_col, themes = load_and_prepare(f, args.id_col)
        add_row(pretty_name_from_filename(f), df, id_col, themes, is_baseline=False)

    out_df = pd.DataFrame(rows)

    # Sort: baseline first, then in the same logical order if names match common variants
    order_pref = {
        "Baseline": 0,
        "Stricter": 1, "Stricter thresholds": 1,
        "Looser": 2, "Looser thresholds": 2,
        "Conservative": 3, "Conservative multimodal weights": 3,
        "Stronger": 4, "Stronger multimodal weights": 4,
    }
    out_df["__order__"] = out_df["Setting"].apply(lambda s: order_pref.get(s, 99))
    out_df = out_df.sort_values(["__order__", "Setting"]).drop(columns=["__order__"])

    out_df.to_csv(args.out, index=False)
    print("\nSensitivity outcomes table")
    print(out_df.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()