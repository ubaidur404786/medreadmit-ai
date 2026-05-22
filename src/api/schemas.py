"""Pydantic v2 request / response schemas for the MedReadmit prediction API.

Field names mirror the raw UCI Diabetes dataset columns that build_features
consumes, excluding dropped columns (encounter_id, patient_nbr, weight,
payer_code) and the target (readmitted).

Hyphenated drug-combination column names (e.g. glyburide-metformin) are
exposed via Field aliases so JSON clients can send the original column name
while Python code uses a valid identifier.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Age buckets — confirmed from data/raw/diabetes.csv unique values
# ---------------------------------------------------------------------------
AgeBucket = Literal[
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]

# ---------------------------------------------------------------------------
# Medication change levels — shared by all drug columns
# ---------------------------------------------------------------------------
MedChange = Optional[Literal["No", "Down", "Steady", "Up"]]

# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class PatientEncounter(BaseModel):
    """Clinical features for one encounter, matching build_features() inputs.

    Required fields are those with low missingness rates and clear clinical
    relevance — they must always be present for a valid prediction request.
    Optional fields default to None; the pipeline fills them with appropriate
    sentinels (e.g. "not_measured", "unknown") internally.
    """

    model_config = ConfigDict(
        # Allow callers to use either the Python attribute name or the Field alias.
        populate_by_name=True,
        # Reject extra fields — catches typos like "diag1" instead of "diag_1".
        extra="forbid",
    )

    # ------------------------------------------------------------------
    # Required fields
    # ------------------------------------------------------------------
    race: str = Field(
        ...,
        description="Patient race as recorded in the EHR.",
        examples=["Caucasian", "AfricanAmerican"],
    )
    gender: Literal["Male", "Female", "Unknown/Invalid"] = Field(
        ...,
        description="Patient gender.",
    )
    age: AgeBucket = Field(
        ...,
        description="Age decile bracket (10-year bins from [0-10) to [90-100)).",
    )
    admission_type_id: int = Field(
        ...,
        ge=1,
        description="Integer code for admission type (e.g. 1=Emergency, 2=Urgent).",
    )
    discharge_disposition_id: int = Field(
        ...,
        ge=1,
        description="Integer code for discharge disposition.",
    )
    admission_source_id: int = Field(
        ...,
        ge=1,
        description="Integer code for admission source (e.g. 7=Emergency Room).",
    )
    time_in_hospital: int = Field(
        ...,
        ge=1,
        le=14,
        description="Length of stay in days (1–14).",
    )
    num_lab_procedures: int = Field(
        ...,
        ge=0,
        description="Number of lab tests performed.",
    )
    num_medications: int = Field(
        ...,
        ge=0,
        description="Number of distinct medications administered.",
    )
    number_diagnoses: int = Field(
        ...,
        ge=1,
        description="Number of diagnoses entered for this encounter.",
    )

    # ------------------------------------------------------------------
    # Optional — numeric utilisation
    # ------------------------------------------------------------------
    num_procedures: Optional[int] = Field(
        None, ge=0, description="Number of non-lab procedures performed."
    )
    number_outpatient: Optional[int] = Field(
        None, ge=0, description="Outpatient visits in the preceding year."
    )
    number_emergency: Optional[int] = Field(
        None, ge=0, description="Emergency visits in the preceding year."
    )
    number_inpatient: Optional[int] = Field(
        None, ge=0, description="Inpatient visits in the preceding year."
    )

    # ------------------------------------------------------------------
    # Optional — ICD-9 diagnosis codes
    # Clients sometimes send integer codes (e.g. 25000 for "250.00");
    # the validator catches that and normalises to str before bucketing.
    # ------------------------------------------------------------------
    diag_1: Optional[str] = Field(None, description="Primary ICD-9 diagnosis code.")
    diag_2: Optional[str] = Field(None, description="Secondary ICD-9 diagnosis code.")
    diag_3: Optional[str] = Field(None, description="Additional ICD-9 diagnosis code.")

    @field_validator("diag_1", "diag_2", "diag_3", mode="before")
    @classmethod
    def coerce_diag_to_str(cls, v: object) -> Optional[str]:
        """Coerce integer diagnosis codes to strings.

        A client sending 25000 instead of "250.00" is a real integration bug
        (integer truncates the decimal sub-code). We convert and let ICD-9
        bucketing proceed normally — it matches on string prefixes, so "25000"
        routes to the diabetes bucket just like "250.00".
        """
        if v is None:
            return None
        if isinstance(v, int):
            return str(v)
        if not isinstance(v, str):
            raise ValueError(f"diag code must be str or int, got {type(v).__name__}")
        return v

    # ------------------------------------------------------------------
    # Optional — lab results (informative missingness handled in pipeline)
    # ------------------------------------------------------------------
    max_glu_serum: Optional[Literal[">200", ">300", "Norm"]] = Field(
        None, description="Result of serum glucose test; omit (null) when not ordered."
    )
    A1Cresult: Optional[Literal[">7", ">8", "Norm"]] = Field(
        None, description="Result of HbA1c test; omit (null) when not ordered."
    )

    # ------------------------------------------------------------------
    # Optional — medical_specialty (high missingness in raw data)
    # ------------------------------------------------------------------
    medical_specialty: Optional[str] = Field(
        None,
        description=(
            "Specialty of the admitting physician. Rare values (<1% of encounters) "
            "are collapsed to 'other' by the feature pipeline."
        ),
    )

    # ------------------------------------------------------------------
    # Optional — medication change columns (all share the same level set)
    # ------------------------------------------------------------------
    metformin: MedChange = None
    repaglinide: MedChange = None
    nateglinide: MedChange = None
    chlorpropamide: MedChange = None
    glimepiride: MedChange = None
    acetohexamide: MedChange = None
    glipizide: MedChange = None
    glyburide: MedChange = None
    tolbutamide: MedChange = None
    pioglitazone: MedChange = None
    rosiglitazone: MedChange = None
    acarbose: MedChange = None
    miglitol: MedChange = None
    troglitazone: MedChange = None
    tolazamide: MedChange = None
    examide: MedChange = None
    citoglipton: MedChange = None
    insulin: MedChange = None
    change: Optional[Literal["No", "Ch"]] = Field(
        None, description="Whether any diabetes medication was changed this encounter."
    )
    diabetesMed: Optional[Literal["Yes", "No"]] = Field(
        None, description="Whether any diabetes medication was prescribed."
    )

    # ------------------------------------------------------------------
    # Hyphenated drug combination columns — Python aliases with JSON aliases
    # ------------------------------------------------------------------
    glyburide_metformin: MedChange = Field(None, alias="glyburide-metformin")
    glipizide_metformin: MedChange = Field(None, alias="glipizide-metformin")
    glimepiride_pioglitazone: MedChange = Field(None, alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: MedChange = Field(None, alias="metformin-rosiglitazone")
    metformin_pioglitazone: MedChange = Field(None, alias="metformin-pioglitazone")


# ---------------------------------------------------------------------------
# Batch request schema
# ---------------------------------------------------------------------------


class BatchRequest(BaseModel):
    """A batch of up to 100 patient encounters for bulk inference."""

    encounters: list[PatientEncounter] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="One to 100 patient encounters to score in a single call.",
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class FeatureContribution(BaseModel):
    """One feature's additive SHAP contribution to a single prediction.

    ``shap_value`` is in log-odds space (raw LightGBM, before Platt scaling).
    Platt scaling changes the calibration of predicted probabilities but
    preserves feature rank order, so this attribution remains valid for the
    calibrated model.
    """

    feature: str = Field(..., description="Sanitised feature name (manifest column key).")
    shap_value: float = Field(..., description="SHAP contribution in log-odds space.")
    feature_value: float = Field(..., description="Actual input value for this feature.")


class PredictionResponse(BaseModel):
    """Prediction output returned by POST /predict."""

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated probability of 30-day readmission.",
    )
    risk_band: Literal["low", "moderate", "high"] = Field(
        ...,
        description="Discretised risk tier derived from probability.",
    )
    model_version: str = Field(
        ...,
        description="Artifact version string (e.g. 'lgbm_calibrated_v1').",
    )
    request_id: str = Field(
        ...,
        min_length=32,
        max_length=32,
        pattern=r"^[0-9a-f]{32}$",
        description="32-character hex request identifier for tracing.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="End-to-end request latency in milliseconds.",
    )
    top_features: list[FeatureContribution] = Field(
        default_factory=list,
        max_length=10,
        description="Top features by |SHAP value|, sorted descending.",
    )


class BatchPredictionItem(BaseModel):
    """Prediction result for one encounter within a batch response."""

    index: int = Field(..., description="Zero-based position of this encounter in the request list.")
    probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated readmission probability.")
    risk_band: Literal["low", "moderate", "high"] = Field(
        ..., description="Discretised risk tier."
    )
    top_features: list[FeatureContribution] = Field(
        default_factory=list,
        max_length=10,
        description="Top features by |SHAP value|, sorted descending.",
    )


class BatchResponse(BaseModel):
    """Response from POST /predict/batch."""

    predictions: list[BatchPredictionItem] = Field(
        ..., description="One item per encounter, in original request order."
    )
    n_processed: int = Field(..., description="Number of encounters scored.")
    model_version: str = Field(..., description="Artifact version string.")
    request_id: str = Field(
        ...,
        min_length=32,
        max_length=32,
        pattern=r"^[0-9a-f]{32}$",
        description="32-character hex request identifier for tracing.",
    )
    latency_ms: float = Field(..., ge=0.0, description="Total end-to-end latency in milliseconds.")
    mean_latency_per_record_ms: float = Field(
        ..., ge=0.0, description="latency_ms / n_processed."
    )


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: Literal["ok", "degraded"] = Field(
        ...,
        description="'ok' when the model is loaded and ready; 'degraded' otherwise.",
    )
    model_loaded: bool = Field(..., description="Whether the model artifact is in memory.")
    n_features: int = Field(..., description="Feature count the loaded model expects.")
    model_version: str = Field(..., description="Artifact version string.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def risk_band_for(prob: float) -> Literal["low", "moderate", "high"]:
    """Map a calibrated readmission probability to a clinical risk tier.

    Thresholds are anchored to the dataset's ~11% base rate (see WEEK1_RESULTS.md):
    - low      < 0.10  — below base rate; routine follow-up
    - moderate  0.10–0.30 — near to 3× base rate; closer monitoring warranted
    - high     >= 0.30  — 3× base rate; active intervention indicated

    Args:
        prob: Calibrated probability in [0, 1] from PlattWrapper.predict_proba.

    Returns:
        One of "low", "moderate", or "high".
    """
    if prob < 0.10:
        return "low"
    if prob < 0.30:
        return "moderate"
    return "high"
