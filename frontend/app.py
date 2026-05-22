"""MedReadmit AI — Streamlit frontend.

Two tabs:
  1. Single Prediction  — load a sample or fill the form, click Predict, see the
     risk gauge, SHAP waterfall, and a fairness footnote.
  2. Batch Screening    — upload a CSV or paste JSON, score all encounters, and
     download a results CSV.

Start the FastAPI backend first:
    uvicorn src.api.main:app --host 127.0.0.1 --port 8000

Then run this app:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from frontend.api_client import MedReadmitAPI, MedReadmitAPIError
from frontend.components.risk_gauge import risk_gauge
from frontend.components.shap_chart import shap_chart

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAMPLES_PATH = _REPO_ROOT / "frontend" / "sample_patients" / "test_set_samples.json"
_FAIRNESS_PATH = _REPO_ROOT / "reports" / "fairness" / "fairness_age.csv"

_API_BASE = os.getenv("MEDREADMIT_API_BASE", "http://localhost:8000")
_BASE_RATE = 0.1139  # positive-class prevalence in training split

_AGE_TO_FAIRNESS: dict[str, str] = {
    "[0-10)": "<40", "[10-20)": "<40", "[20-30)": "<40", "[30-40)": "<40",
    "[40-50)": "40-65", "[50-60)": "40-65", "[60-70)": "40-65",
    "[70-80)": ">65", "[80-90)": ">65", "[90-100)": ">65",
}

_AGE_OPTIONS = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]
_RACE_OPTIONS = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"]
_GENDER_OPTIONS = ["Female", "Male", "Unknown/Invalid"]
_MAX_GLU_OPTIONS = [None, ">200", ">300", "Norm"]
_A1C_OPTIONS = [None, ">7", ">8", "Norm"]
_MED_CHANGE_OPTIONS = [None, "No", "Steady", "Up", "Down"]
_CHANGE_OPTIONS = [None, "No", "Ch"]
_DIABETES_MED_OPTIONS = [None, "Yes", "No"]

_MED_COLS: list[str] = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

_FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "number_inpatient": "Prior inpatient admissions",
    "discharge_disposition_id": "Discharge disposition code",
    "number_diagnoses": "Number of diagnoses",
    "time_in_hospital": "Days in hospital",
    "number_emergency": "Prior ER visits",
    "num_medications": "Medications at discharge",
    "num_lab_procedures": "Lab procedures during stay",
    "num_procedures": "Procedures during stay",
    "number_outpatient": "Prior outpatient visits",
    "admission_type_id": "Admission type",
    "admission_source_id": "Admission source",
    "diag_1_circulatory": "Primary diagnosis: circulatory",
    "diag_1_diabetes": "Primary diagnosis: diabetes",
    "diag_1_respiratory": "Primary diagnosis: respiratory",
    "diag_1_digestive": "Primary diagnosis: digestive",
    "diag_1_injury": "Primary diagnosis: injury",
    "diag_1_musculoskeletal": "Primary diagnosis: musculoskeletal",
    "diag_1_genitourinary": "Primary diagnosis: genitourinary",
    "diag_1_neoplasms": "Primary diagnosis: neoplasms",
    "metformin_No": "Not on metformin",
    "diabetesMed_No": "No diabetic medications",
    "insulin_No": "Not on insulin",
    "insulin_Steady": "Insulin steady",
    "insulin_Up": "Insulin increased",
    "insulin_Down": "Insulin decreased",
}

_DEFAULT_FIELDS: dict[str, Any] = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[60-70)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 3,
    "num_lab_procedures": 40,
    "num_medications": 12,
    "number_diagnoses": 5,
    "num_procedures": 0,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "diag_1": None,
    "diag_2": None,
    "diag_3": None,
    "max_glu_serum": None,
    "A1Cresult": None,
    "change": None,
    "diabetesMed": None,
}
for _m in _MED_COLS:
    _DEFAULT_FIELDS[_m] = None


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource
def get_api_client() -> MedReadmitAPI:
    return MedReadmitAPI(base_url=_API_BASE, timeout=10.0)


@st.cache_data
def load_samples() -> list[dict[str, Any]]:
    if not _SAMPLES_PATH.exists():
        return []
    return json.loads(_SAMPLES_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_fairness_csv() -> pd.DataFrame | None:
    if not _FAIRNESS_PATH.exists():
        return None
    return pd.read_csv(_FAIRNESS_PATH)


@st.cache_data(ttl=30)
def _check_api_health(base_url: str) -> bool:
    """Ping /health and return True if the API is reachable. Cached for 30 s."""
    try:
        client = MedReadmitAPI(base_url=base_url, timeout=2.0)
        resp = client.health()
        return resp.get("status") == "ok"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _field_key(col: str) -> str:
    return f"field_{col}"


def _init_session_state() -> None:
    for col, val in _DEFAULT_FIELDS.items():
        key = _field_key(col)
        if key not in st.session_state:
            st.session_state[key] = val
    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None
    if "sample_selector_index" not in st.session_state:
        st.session_state["sample_selector_index"] = 0


def _load_sample_callback() -> None:
    """Copy the selected sample patient's fields into session_state without triggering prediction."""
    samples = load_samples()
    idx = st.session_state.get("sample_selector_index", 0)
    if idx == 0 or idx > len(samples):
        return
    fields = samples[idx - 1]["fields"]
    for col, val in fields.items():
        st.session_state[_field_key(col)] = val


def _reset_callback() -> None:
    """Clear prediction result and reset all fields to defaults."""
    for col, val in _DEFAULT_FIELDS.items():
        st.session_state[_field_key(col)] = val
    st.session_state["last_prediction"] = None
    st.session_state["sample_selector_index"] = 0


def _collect_encounter() -> dict[str, Any]:
    """Read all field_* keys from session_state into a flat encounter dict."""
    enc: dict[str, Any] = {}
    for key, val in st.session_state.items():
        if key.startswith("field_"):
            col = key[len("field_"):]
            enc[col] = val if val != "" else None
    for diag in ("diag_1", "diag_2", "diag_3"):
        if enc.get(diag) == "":
            enc[diag] = None
    return enc


def _humanize_feature_name(raw: str) -> str:
    """Convert a raw model feature name to a human-readable label.

    Looks up the name in _FEATURE_DISPLAY_NAMES first; falls back to
    title-casing with underscores replaced by spaces.

    Args:
        raw: Raw feature name as returned by the API (e.g. "number_inpatient").

    Returns:
        Human-readable label (e.g. "Prior inpatient admissions").
    """
    return _FEATURE_DISPLAY_NAMES.get(raw, raw.replace("_", " ").title())


def _clinical_action_line(risk_band: str) -> str:
    """Return a one-line clinical action recommendation for a given risk band.

    Args:
        risk_band: One of "low", "moderate", or "high" (case-insensitive).

    Returns:
        A plain-text recommendation string, or empty string for unknown bands.
    """
    return {
        "low": "Standard discharge follow-up appropriate.",
        "moderate": "Consider scheduling a follow-up call within 7 days.",
        "high": "Recommend transitional care consult within 48h of discharge.",
    }.get(risk_band.lower(), "")


def _fmt_value(v: float) -> str:
    """Format a feature value for display (strip trailing .0 for integers)."""
    try:
        if v == int(v):
            return str(int(v))
    except (TypeError, ValueError, OverflowError):
        pass
    return f"{v:.3g}"


def _humanize_features(top_features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of top_features with humanized feature names including the value.

    Each dict's ``feature`` key is replaced with a human-readable label that
    appends the feature value in parentheses, e.g. ``"Prior inpatient admissions (5)"``.
    The ``shap_value`` and ``feature_value`` keys are preserved unchanged.

    Args:
        top_features: Raw feature list from the API response.

    Returns:
        New list of dicts with updated ``feature`` keys suitable for the chart.
    """
    result = []
    for f in top_features:
        label = _humanize_feature_name(f["feature"])
        val_str = _fmt_value(f["feature_value"])
        result.append({**f, "feature": f"{label} ({val_str})"})
    return result


def _fairness_info(age_bracket: str | None, fairness_df: pd.DataFrame | None) -> None:
    if fairness_df is None or age_bracket is None:
        return
    subgroup = _AGE_TO_FAIRNESS.get(str(age_bracket))
    if subgroup is None:
        return
    row = fairness_df[fairness_df["subgroup"] == subgroup]
    if row.empty:
        return
    r = row.iloc[0]
    st.info(
        f"**Fairness note** — Age group *{subgroup}*: "
        f"AUROC {r['auroc']:.3f}, prevalence {r['prevalence']:.1%}, "
        f"n={int(r['n']):,}. "
        "Model performance may vary across demographic subgroups."
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    with st.sidebar:
        st.title("MedReadmit AI")
        st.caption("30-day hospital readmission risk")
        st.divider()

        st.subheader("Model performance")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("AUROC", "0.6621")
            st.metric("Brier", "0.0946")
        with c2:
            st.metric("AUPRC", "0.2184")

        st.subheader("Operating point")
        c3, c4 = st.columns(2)
        with c3:
            st.metric("PPV @ 0.30", "0.363")
        with c4:
            st.metric("Lift vs base", "3.3×")

        st.subheader("Coverage")
        st.caption(
            "Trained on 99,343 diabetic encounters from 130 US hospitals (1999–2008). "
            "154 features after one-hot encoding."
        )

        with st.expander("Limitations", expanded=False):
            st.markdown(
                """
- Training data is 2008 vintage; performance on current populations is unknown.
- Single-disease cohort (diabetic patients only).
- Hospital-coded variables may vary across institutions.
- AUROC drops to 0.64 for patients >65.
- Calibration preserved across subgroups but prevalence differs.
                """
            )

        st.divider()
        st.caption(
            "[GitHub](https://github.com/ubaidur404786/medreadmit-ai) · "
            "[MLflow](http://localhost:5000)"
        )


# ---------------------------------------------------------------------------
# Patient input fields (outside st.form — Predict button lives above)
# ---------------------------------------------------------------------------


def _render_patient_input() -> None:
    """Render all patient input fields inside a collapsed outer expander.

    Fields are grouped into four themed sub-expanders, each laid out in
    columns of 3 for density. All widgets write directly to st.session_state
    via their key= argument; no form submission is needed.
    """
    with st.expander("Show patient input fields", expanded=False):

        with st.expander("Demographics & admission", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                race_val = st.session_state.get(_field_key("race"), "Caucasian")
                st.selectbox("Race", _RACE_OPTIONS,
                    index=_RACE_OPTIONS.index(race_val) if race_val in _RACE_OPTIONS else 0,
                    key=_field_key("race"))
            with c2:
                gender_val = st.session_state.get(_field_key("gender"), "Female")
                st.selectbox("Gender", _GENDER_OPTIONS,
                    index=_GENDER_OPTIONS.index(gender_val) if gender_val in _GENDER_OPTIONS else 0,
                    key=_field_key("gender"))
            with c3:
                age_val = st.session_state.get(_field_key("age"), "[60-70)")
                st.selectbox("Age bracket", _AGE_OPTIONS,
                    index=_AGE_OPTIONS.index(age_val) if age_val in _AGE_OPTIONS else 6,
                    key=_field_key("age"))

            c4, c5, c6 = st.columns(3)
            with c4:
                st.number_input("Admission type ID", min_value=1, max_value=8,
                    value=int(st.session_state.get(_field_key("admission_type_id"), 1)),
                    key=_field_key("admission_type_id"),
                    help="1=Emergency, 2=Urgent, 3=Elective")
            with c5:
                st.number_input("Discharge disposition ID", min_value=1, max_value=28,
                    value=int(st.session_state.get(_field_key("discharge_disposition_id"), 1)),
                    key=_field_key("discharge_disposition_id"),
                    help="1=Home, 6=Home health, 11=Expired")
            with c6:
                st.number_input("Admission source ID", min_value=1, max_value=25,
                    value=int(st.session_state.get(_field_key("admission_source_id"), 7)),
                    key=_field_key("admission_source_id"),
                    help="7=Emergency room, 1=Physician referral")

        with st.expander("Clinical encounter", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.number_input("Days in hospital", min_value=1, max_value=14,
                    value=int(st.session_state.get(_field_key("time_in_hospital"), 3)),
                    key=_field_key("time_in_hospital"))
            with c2:
                st.number_input("Lab procedures", min_value=0, max_value=132,
                    value=int(st.session_state.get(_field_key("num_lab_procedures"), 40)),
                    key=_field_key("num_lab_procedures"))
            with c3:
                st.number_input("Procedures", min_value=0, max_value=6,
                    value=int(st.session_state.get(_field_key("num_procedures"), 0)),
                    key=_field_key("num_procedures"))

            c4, c5, c6 = st.columns(3)
            with c4:
                st.number_input("Medications", min_value=1, max_value=81,
                    value=int(st.session_state.get(_field_key("num_medications"), 12)),
                    key=_field_key("num_medications"))
            with c5:
                st.number_input("Diagnoses count", min_value=1, max_value=16,
                    value=int(st.session_state.get(_field_key("number_diagnoses"), 5)),
                    key=_field_key("number_diagnoses"))
            with c6:
                pass  # intentional padding

            c7, c8, c9 = st.columns(3)
            with c7:
                a1c_val = st.session_state.get(_field_key("A1Cresult"))
                st.selectbox("A1C result", _A1C_OPTIONS,
                    index=_A1C_OPTIONS.index(a1c_val) if a1c_val in _A1C_OPTIONS else 0,
                    key=_field_key("A1Cresult"),
                    format_func=lambda x: "Not measured" if x is None else x)
            with c8:
                glu_val = st.session_state.get(_field_key("max_glu_serum"))
                st.selectbox("Max glucose serum", _MAX_GLU_OPTIONS,
                    index=_MAX_GLU_OPTIONS.index(glu_val) if glu_val in _MAX_GLU_OPTIONS else 0,
                    key=_field_key("max_glu_serum"),
                    format_func=lambda x: "Not measured" if x is None else x)

        with st.expander("Prior utilization", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.number_input("Prior inpatient", min_value=0,
                    value=int(st.session_state.get(_field_key("number_inpatient"), 0)),
                    key=_field_key("number_inpatient"))
            with c2:
                st.number_input("Prior emergency", min_value=0,
                    value=int(st.session_state.get(_field_key("number_emergency"), 0)),
                    key=_field_key("number_emergency"))
            with c3:
                st.number_input("Prior outpatient", min_value=0,
                    value=int(st.session_state.get(_field_key("number_outpatient"), 0)),
                    key=_field_key("number_outpatient"))

        with st.expander("Medications", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.text_input("Primary diagnosis (diag_1)",
                    value=st.session_state.get(_field_key("diag_1"), "") or "",
                    key=_field_key("diag_1"), placeholder="e.g. 250.00")
            with c2:
                st.text_input("Secondary diagnosis (diag_2)",
                    value=st.session_state.get(_field_key("diag_2"), "") or "",
                    key=_field_key("diag_2"), placeholder="e.g. 401")
            with c3:
                st.text_input("Tertiary diagnosis (diag_3)",
                    value=st.session_state.get(_field_key("diag_3"), "") or "",
                    key=_field_key("diag_3"), placeholder="e.g. V58")

            c4, c5, c6 = st.columns(3)
            with c4:
                chg_val = st.session_state.get(_field_key("change"))
                st.selectbox("Medication change flag", _CHANGE_OPTIONS,
                    index=_CHANGE_OPTIONS.index(chg_val) if chg_val in _CHANGE_OPTIONS else 0,
                    key=_field_key("change"),
                    format_func=lambda x: "Unknown" if x is None else x)
            with c5:
                dm_val = st.session_state.get(_field_key("diabetesMed"))
                st.selectbox("On diabetes med", _DIABETES_MED_OPTIONS,
                    index=_DIABETES_MED_OPTIONS.index(dm_val) if dm_val in _DIABETES_MED_OPTIONS else 0,
                    key=_field_key("diabetesMed"),
                    format_func=lambda x: "Unknown" if x is None else x)

            st.markdown("**Individual medication changes**")
            med_rows = [st.columns(3) for _ in range(math.ceil(len(_MED_COLS) / 3))]
            for i, med in enumerate(_MED_COLS):
                med_val = st.session_state.get(_field_key(med))
                with med_rows[i // 3][i % 3]:
                    st.selectbox(med, _MED_CHANGE_OPTIONS,
                        index=_MED_CHANGE_OPTIONS.index(med_val) if med_val in _MED_CHANGE_OPTIONS else 0,
                        key=_field_key(med),
                        format_func=lambda x: "-" if x is None else x,
                        label_visibility="collapsed")
                    st.caption(med)


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------


def _render_prediction_result(result: dict[str, Any], encounter: dict[str, Any]) -> None:
    prob = result["probability"]
    risk_band = result.get("risk_band", "low")
    request_id = result.get("request_id", "")
    latency_ms = result.get("latency_ms", 0.0)
    model_version = result.get("model_version", "—")
    top_features = result.get("top_features", [])

    with st.container(border=True):
        st.markdown("**30-day readmission probability**")
        gauge_col, meta_col = st.columns([3, 2])
        with gauge_col:
            st.plotly_chart(risk_gauge(prob), use_container_width=True)
            action = _clinical_action_line(risk_band)
            if action:
                st.caption(action)
        with meta_col:
            lift = prob / _BASE_RATE if _BASE_RATE > 0 else 0.0
            st.metric("Probability", f"{prob:.1%}")
            st.metric("Risk band", risk_band.upper(), delta=f"{lift:.1f}× base rate", delta_color="off")
            st.metric("Latency", f"{latency_ms:.0f} ms")
            short_id = (request_id[:6] + "…") if len(request_id) > 6 else request_id
            st.caption(f"Model: {model_version} · Request ID: {short_id}")

    if top_features:
        with st.container(border=True):
            st.markdown("**Top contributing factors**")
            humanized = _humanize_features(top_features)
            st.plotly_chart(shap_chart(humanized, request_id), use_container_width=True)
            st.caption(
                "Red bars increase risk · blue bars decrease risk · "
                "values are SHAP log-odds before sigmoid calibration"
            )

    _fairness_info(encounter.get("age"), load_fairness_csv())


# ---------------------------------------------------------------------------
# Single Prediction Tab
# ---------------------------------------------------------------------------


def _render_single_tab(api: MedReadmitAPI) -> None:
    # Header strip
    hdr_left, hdr_right = st.columns([5, 1])
    with hdr_left:
        st.title("MedReadmit AI")
        st.caption("30-day readmission risk · calibrated LightGBM · 154 features")
    with hdr_right:
        api_ok = _check_api_health(_API_BASE)
        st.markdown("🟢 **API: connected**" if api_ok else "🔴 **API: down**")

    st.divider()

    # Demo control bar
    samples = load_samples()
    sample_labels = ["— Select a sample —"] + [s["label_internal"] for s in samples]
    ctrl_left, ctrl_mid, ctrl_right = st.columns([2, 1, 1])
    with ctrl_left:
        st.selectbox(
            "Load a sample patient",
            options=range(len(sample_labels)),
            format_func=lambda i: sample_labels[i],
            key="sample_selector_index",
            on_change=_load_sample_callback,
            label_visibility="collapsed",
        )
    with ctrl_mid:
        predict_clicked = st.button(
            "Predict readmission risk", type="primary", use_container_width=True
        )
    with ctrl_right:
        reset_clicked = st.button("Reset", type="secondary", use_container_width=True)

    if reset_clicked:
        _reset_callback()
        st.rerun()

    if predict_clicked:
        encounter = _collect_encounter()
        with st.spinner("Scoring ..."):
            try:
                result = api.predict(encounter)
                st.session_state["last_prediction"] = {"result": result, "encounter": encounter}
            except MedReadmitAPIError as exc:
                st.error(
                    "Could not reach the prediction service. "
                    "Is the FastAPI backend running on port 8000?"
                    if exc.status_code == 0
                    else f"API error {exc.status_code}: {exc.detail}"
                )
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

    # Persisted prediction result
    last = st.session_state.get("last_prediction")
    if last is not None:
        _render_prediction_result(last["result"], last["encounter"])

    st.divider()

    # Patient input (collapsed by default — most users will use the sample dropdown)
    _render_patient_input()


# ---------------------------------------------------------------------------
# Batch Screening Tab
# ---------------------------------------------------------------------------


def _render_batch_tab(api: MedReadmitAPI) -> None:
    st.header("Batch Screening")
    st.markdown(
        "Upload a CSV file or paste JSON to score multiple encounters at once.  "
        "Rows with missing required fields are skipped with a warning.  "
        "Batches larger than 100 rows are chunked automatically."
    )

    input_mode = st.radio("Input format", ["CSV upload", "JSON paste"], horizontal=True)

    records: list[dict[str, Any]] = []
    parse_error: str | None = None

    if input_mode == "CSV upload":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                records = df.where(pd.notna(df), None).to_dict(orient="records")
                st.success(f"Loaded {len(records)} rows from CSV.")
            except Exception as exc:
                parse_error = f"Could not parse CSV: {exc}"
    else:
        raw_json = st.text_area(
            "Paste JSON array of encounter dicts",
            height=200,
            placeholder='[{"race": "Caucasian", "gender": "Female", ...}, ...]',
        )
        if raw_json.strip():
            try:
                records = json.loads(raw_json)
                if not isinstance(records, list):
                    parse_error = "JSON must be a list of encounter dicts."
                else:
                    st.success(f"Parsed {len(records)} encounters from JSON.")
            except json.JSONDecodeError as exc:
                parse_error = f"Invalid JSON: {exc}"

    if parse_error:
        st.error(parse_error)
        return

    if not records:
        st.info("No encounters loaded yet.")
        return

    if st.button("Run batch prediction", type="primary"):
        _run_batch(api, records)


def _run_batch(api: MedReadmitAPI, records: list[dict[str, Any]]) -> None:
    _REQUIRED = {
        "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
        "admission_source_id", "time_in_hospital", "num_lab_procedures",
        "num_medications", "number_diagnoses",
    }
    valid: list[tuple[int, dict[str, Any]]] = []
    skipped: list[int] = []

    for i, rec in enumerate(records):
        if all(rec.get(f) not in (None, "", float("nan")) for f in _REQUIRED):
            valid.append((i, rec))
        else:
            skipped.append(i)

    if skipped:
        st.warning(
            f"Skipped {len(skipped)} rows with missing required fields "
            f"(indices: {skipped[:10]}{'...' if len(skipped) > 10 else ''})."
        )

    if not valid:
        st.error("No valid encounters to score.")
        return

    chunk_size = 100
    all_results: list[dict[str, Any]] = []
    progress = st.progress(0, text="Scoring ...")
    n_chunks = math.ceil(len(valid) / chunk_size)

    for chunk_i in range(n_chunks):
        chunk = valid[chunk_i * chunk_size : (chunk_i + 1) * chunk_size]
        chunk_encounters = [r for _, r in chunk]
        try:
            resp = api.predict_batch(chunk_encounters)
        except MedReadmitAPIError as exc:
            st.error(
                "Could not reach the prediction service. Is the FastAPI backend running on port 8000?"
                if exc.status_code == 0
                else f"API error {exc.status_code}: {exc.detail}"
            )
            return

        for pred in resp["predictions"]:
            orig_index = chunk[pred["index"]][0]
            all_results.append({
                "original_row": orig_index,
                "probability": pred["probability"],
                "risk_band": pred["risk_band"],
            })
        progress.progress((chunk_i + 1) / n_chunks, text=f"Chunk {chunk_i + 1}/{n_chunks} done")

    progress.empty()

    results_df = pd.DataFrame(all_results).sort_values("original_row").reset_index(drop=True)

    st.subheader(f"Results — {len(results_df)} encounters scored")

    m1, m2, m3 = st.columns(3)
    high_risk = (results_df["risk_band"] == "high").sum()
    mod_risk = (results_df["risk_band"] == "moderate").sum()
    with m1:
        st.metric("High risk", high_risk, help="probability >= 0.30")
    with m2:
        st.metric("Moderate risk", mod_risk, help="0.10 <= probability < 0.30")
    with m3:
        st.metric("Low risk", len(results_df) - high_risk - mod_risk, help="probability < 0.10")

    def _colour_band(band: str) -> str:
        return {
            "high": "background-color: #fdedec",
            "moderate": "background-color: #fef9e7",
            "low": "background-color: #d5f5e3",
        }.get(band, "")

    styled = results_df.style.map(_colour_band, subset=["risk_band"])
    st.dataframe(styled, use_container_width=True)

    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results CSV",
        data=csv_bytes,
        file_name="readmission_predictions.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="MedReadmit AI",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_session_state()
    _render_sidebar()

    api = get_api_client()

    tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Screening"])

    with tab_single:
        _render_single_tab(api)

    with tab_batch:
        _render_batch_tab(api)


if __name__ == "__main__":
    main()
