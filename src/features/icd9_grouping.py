"""Map raw ICD-9 diagnosis codes to 9 clinical groups (Strack et al. 2014)."""

from __future__ import annotations

import pandas as pd


# Groups follow the ICD-9-CM chapter scheme used in the original paper.
# The order below is the match-priority order inside map_icd9_to_group.
GROUPS = frozenset(
    {
        "missing",
        "circulatory",
        "respiratory",
        "digestive",
        "diabetes",
        "injury",
        "musculoskeletal",
        "genitourinary",
        "neoplasms",
        "other",
    }
)


def map_icd9_to_group(code: str | float) -> str:
    """Map a single ICD-9 code to one of 9 clinical groups.

    Rules are applied in order; the first match wins.  Numeric range checks
    use ``int(float(code))`` so decimal sub-codes ("250.83") reduce to their
    integer chapter value without raising on the decimal point.

    Args:
        code: Raw ICD-9 string from the UCI Diabetes dataset (e.g. ``"250.83"``,
            ``"V57"``, ``"E906"``, ``"428"``), or a float NaN from pandas.

    Returns:
        One of: ``"missing"``, ``"circulatory"``, ``"respiratory"``,
        ``"digestive"``, ``"diabetes"``, ``"injury"``, ``"musculoskeletal"``,
        ``"genitourinary"``, ``"neoplasms"``, ``"other"``.
    """
    # --- 1. Missing ---
    if pd.isna(code) or str(code).strip() == "":
        return "missing"

    code_str = str(code).strip().upper()

    # --- 2. Supplementary classifications (V and E codes) ---
    if code_str.startswith(("V", "E")):
        return "other"

    # --- 3. Diabetes (string check before numeric conversion) ---
    # "250" matches 250.0, 250.83, etc. without int(float) rounding ambiguity.
    if code_str.startswith("250"):
        return "diabetes"

    # --- 4. Numeric chapter ranges ---
    try:
        numeric = int(float(code_str))
    except ValueError:
        return "other"

    if (390 <= numeric <= 459) or numeric == 785:
        return "circulatory"
    if (460 <= numeric <= 519) or numeric == 786:
        return "respiratory"
    if (520 <= numeric <= 579) or numeric == 787:
        return "digestive"
    if 800 <= numeric <= 999:
        return "injury"
    if 710 <= numeric <= 739:
        return "musculoskeletal"
    if (580 <= numeric <= 629) or numeric == 788:
        return "genitourinary"
    if 140 <= numeric <= 239:
        return "neoplasms"

    return "other"


def bucket_diagnosis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace diag_1, diag_2, diag_3 with their bucketed clinical group strings.

    Applies :func:`map_icd9_to_group` element-wise to each of the three
    diagnosis columns.  All other columns are left unchanged.

    Args:
        df: DataFrame containing ``diag_1``, ``diag_2``, ``diag_3`` columns
            with raw ICD-9 strings, as produced by ``load_raw``.

    Returns:
        New DataFrame (copy) with the three diagnosis columns replaced by
        their clinical group labels.
    """
    out = df.copy()
    for col in ("diag_1", "diag_2", "diag_3"):
        out[col] = out[col].map(map_icd9_to_group)
    return out


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from src.data.load import load_raw
    from src.data.make_target import build_target

    df = build_target(load_raw())
    df = bucket_diagnosis_columns(df)

    counts = df["diag_1"].value_counts()
    print("\ndiag_1 value counts after bucketing:")
    print(counts.to_string())
    top, pct = counts.index[0], counts.iloc[0] / len(df)
    print(f"\nTop category: {top!r} at {pct:.1%}  (expect 'circulatory' ~30%)")
