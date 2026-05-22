"""Export 20 representative patient records from the test split for the frontend demo.

Selects 5 patients from each of 4 risk probability bands:
  - low:           probability < 0.10
  - low_moderate:  0.10 ≤ probability < 0.20
  - high_moderate: 0.20 ≤ probability < 0.30
  - high:          probability ≥ 0.30

Within each band the selection maximises variety across primary-diagnosis ICD-9
bucket and age category so the demo covers diverse clinical presentations.

Fields are stored as raw clinical values (diag_1/2/3 as ICD-9 codes, not
bucketed) that match the FastAPI PatientEncounter schema exactly.

Output: frontend/sample_patients/test_set_samples.json

Run from repo root:
    python scripts/export_sample_patients.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.api.schemas import risk_band_for  # noqa: E402
from src.data.load import load_raw  # noqa: E402
from src.data.make_target import build_target  # noqa: E402
from src.data.split import patient_grouped_split  # noqa: E402
from src.features.build_features import build_features  # noqa: E402
from src.features.icd9_grouping import map_icd9_to_group  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_RNG_SEED = 42          # fixed seed → same 20 patients every run
_N_PER_BAND = 5

_MODEL_PATH = _REPO_ROOT / "models" / "lgbm_calibrated.joblib"
_MANIFEST_PATH = _REPO_ROOT / "models" / "feature_manifest.json"
_OUT_DIR = _REPO_ROOT / "frontend" / "sample_patients"
_OUT_FILE = _OUT_DIR / "test_set_samples.json"

# Probability bands: (internal_name, lo_inclusive, hi_exclusive, api_band)
_BANDS: list[tuple[str, float, float, str]] = [
    ("low",           0.00,  0.10,  "low"),
    ("low_moderate",  0.10,  0.20,  "moderate"),
    ("high_moderate", 0.20,  0.30,  "moderate"),
    ("high",          0.30,  1.001, "high"),
]

# ---------------------------------------------------------------------------
# Column sets
# ---------------------------------------------------------------------------

# Columns present in build_target() output that are NOT part of the API schema.
_NON_SCHEMA_COLS = frozenset(
    {"patient_nbr", "readmitted_30d", "encounter_id", "weight", "payer_code"}
)

# Required PatientEncounter fields — rows missing any of these are skipped.
_REQUIRED_FIELDS = frozenset({
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
    "number_diagnoses",
})

# Allowed values for constrained optional fields.
_VALID_MAX_GLU_SERUM = frozenset({">200", ">300", "Norm"})
_VALID_A1CRESULT = frozenset({">7", ">8", "Norm"})
_VALID_MED_CHANGE = frozenset({"No", "Down", "Steady", "Up"})
_VALID_CHANGE = frozenset({"No", "Ch"})
_VALID_DIABETES_MED = frozenset({"Yes", "No"})
_VALID_GENDER = frozenset({"Male", "Female", "Unknown/Invalid"})

# Medication-change columns (single-drug + combo, using hyphenated raw names).
_MED_CHANGE_COLS = frozenset({
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "insulin",
    # Hyphenated combination drugs — raw CSV names, also valid JSON keys.
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _age_category(age: str) -> str:
    if age in ("[0-10)", "[10-20)", "[20-30)", "[30-40)"):
        return "young"
    if age in ("[40-50)", "[50-60)"):
        return "midage"
    if age in ("[60-70)", "[70-80)"):
        return "senior"
    return "elderly"


def _to_json_safe(v: Any) -> Any:
    """float NaN → None, numpy scalar → Python native, else unchanged."""
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        return v.item()
    return v


def _clean_field(col: str, val: Any) -> Any:
    """Normalise a raw column value to a Pydantic/JSON-safe type.

    Converts NaN and the literal string "None" (which appears in some UCI
    columns for 'not measured') to Python None.  Validates constrained
    Literal fields; out-of-vocabulary values are mapped to None so the API
    treats them as 'not measured' / 'unknown'.
    """
    safe = _to_json_safe(val)
    if safe is None:
        return None
    if str(safe) == "None":
        return None
    if col == "max_glu_serum":
        return safe if safe in _VALID_MAX_GLU_SERUM else None
    if col == "A1Cresult":
        return safe if safe in _VALID_A1CRESULT else None
    if col in _MED_CHANGE_COLS:
        return safe if safe in _VALID_MED_CHANGE else None
    if col == "change":
        return safe if safe in _VALID_CHANGE else None
    if col == "diabetesMed":
        return safe if safe in _VALID_DIABETES_MED else None
    return safe


def _row_to_fields(row: pd.Series) -> dict[str, Any]:
    """Convert a raw labelled row to a PatientEncounter-compatible dict.

    All schema fields are included; None is used for absent/invalid optional
    values so the Streamlit form can pre-populate every widget.
    """
    return {
        col: _clean_field(col, val)
        for col, val in row.items()
        if col not in _NON_SCHEMA_COLS
    }


def _is_valid_row(row: pd.Series) -> bool:
    """True if every required PatientEncounter field is present and valid."""
    for field in _REQUIRED_FIELDS:
        val = _to_json_safe(row.get(field))
        if val is None:
            return False
    # gender must be one of the three allowed literals
    gender = _to_json_safe(row.get("gender"))
    if gender not in _VALID_GENDER:
        return False
    return True


def _pick_varied(
    candidates: pd.DataFrame,
    n: int,
    rng: random.Random,
) -> list[Any]:
    """Return up to n row-index values with variety in diag_1 bucket × age category.

    Groups candidates by the composite key (diag1_bucket, age_category), then
    round-robins through groups (sorted descending by group size so the most
    common clinical presentations are represented) until n indices are chosen.
    Within each group, rows are shuffled with the fixed-seed RNG so the same
    patients are selected every run.
    """
    diag1_bucket = candidates["diag_1"].map(map_icd9_to_group).fillna("missing")
    age_cat = candidates["age"].map(
        lambda a: _age_category(str(a)) if pd.notna(a) else "unknown"
    )
    composite_key = diag1_bucket + "_" + age_cat

    groups: dict[str, list[Any]] = defaultdict(list)
    for idx in candidates.index:
        groups[composite_key.loc[idx]].append(idx)

    for key in groups:
        rng.shuffle(groups[key])

    # Largest groups first so common conditions are represented, then rarer ones.
    ordered_keys = sorted(groups, key=lambda k: -len(groups[k]))
    queues = {k: list(v) for k, v in groups.items()}

    selected: list[Any] = []
    while len(selected) < n:
        added_this_pass = False
        for key in ordered_keys:
            if len(selected) >= n:
                break
            if queues[key]:
                selected.append(queues[key].pop(0))
                added_this_pass = True
        if not added_this_pass:
            break  # pool exhausted

    return selected[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    rng = random.Random(_RNG_SEED)

    print("Loading model ...")
    model = joblib.load(_MODEL_PATH)

    print("Running pipeline (load_raw -> build_target -> build_features -> split) ...")
    labelled = build_target(load_raw()).reset_index(drop=True)
    X, y, groups = build_features(labelled)
    _, _, X_test, _, _, _ = patient_grouped_split(X, y, groups)

    print(f"Scoring {len(X_test)} test-set rows ...")
    probas = model.predict_proba(X_test)[:, 1]
    proba_series = pd.Series(probas, index=X_test.index, name="proba")

    # Filter out rows with null or invalid required fields.
    valid_flags = labelled.loc[X_test.index].apply(_is_valid_row, axis=1)
    valid_idx = X_test.index[valid_flags.values]
    print(f"  Valid rows (all required fields present): {valid_flags.sum()} / {len(X_test)}")

    output: list[dict[str, Any]] = []

    for band_name, lo, hi, expected_band in _BANDS:
        band_proba = proba_series.loc[valid_idx]
        in_band = (band_proba >= lo) & (band_proba < hi)
        band_idx = valid_idx[in_band.values]
        band_candidates = labelled.loc[band_idx]

        n_available = len(band_candidates)
        print(f"\nBand '{band_name}' (p in [{lo:.2f}, {hi:.2f})): {n_available} candidates")
        if n_available < _N_PER_BAND:
            print(f"  WARNING: only {n_available} candidates, using all.")

        selected_idx = _pick_varied(band_candidates, _N_PER_BAND, rng)

        for raw_idx in selected_idx:
            row = labelled.loc[raw_idx]
            proba = float(proba_series.loc[raw_idx])

            age_str = str(row.get("age", "")) if pd.notna(row.get("age")) else "?"
            age_cat = _age_category(age_str) if age_str != "?" else "unknown"
            diag1_bucket = map_icd9_to_group(row.get("diag_1"))

            label = f"{band_name}_{age_cat}_{diag1_bucket}"
            fields = _row_to_fields(row)

            output.append({
                "label_internal": label,
                "fields": fields,
                "expected_band": expected_band,
                "model_predicted": round(proba, 4),
            })
            print(
                f"  [{risk_band_for(proba):>8}]  "
                f"proba={proba:.4f}  "
                f"age={age_str}  "
                f"diag1={diag1_bucket:<15}  "
                f"label={label}"
            )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _OUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(output)} records to {_OUT_FILE}")

    # Quick validation: confirm every record round-trips through PatientEncounter.
    print("\nValidating records against PatientEncounter schema ...")
    from src.api.schemas import PatientEncounter  # noqa: E402 (deferred to avoid slow import at top)

    errors = 0
    for i, entry in enumerate(output):
        try:
            PatientEncounter(**entry["fields"])
        except Exception as exc:
            print(f"  Record {i} ({entry['label_internal']}): VALIDATION FAILED — {exc}")
            errors += 1
    if errors == 0:
        print(f"  All {len(output)} records pass PatientEncounter validation. OK")
    else:
        print(f"  {errors} record(s) failed validation — fix _clean_field logic.")
        sys.exit(1)


if __name__ == "__main__":
    main()
