from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "malaria_dataset" / "malaria_raw.csv"
MODEL_PATHS = (BASE_DIR / "best_malaria.keras", BASE_DIR / "malaria_saved_files" / "best_malaria.keras")
FEATURE_ORDER = (
    "Age",
    "Fever_Temp",
    "Chills",
    "Headache_Severity",
    "Vomiting",
    "Diarrhea",
    "Muscle_Pain",
    "Fatigue_Level",
    "Sweating",
    "Nausea",
    "High_Fever",
    "Fever_x_Chills",
    "Severe_Headache",
    "Symptom_Count",
    "Vulnerable_Age",
)

st.set_page_config(page_title="Malaria symptom assessment", page_icon="🦟", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #18332f;
        --muted: #60746e;
        --paper: #f3f6f1;
        --line: #d6e0d8;
        --leaf: #1e6955;
        --rust: #b84f36;
    }
    .stApp {
        background:
            repeating-linear-gradient(135deg, rgba(30, 105, 85, .025) 0 1px, transparent 1px 12px),
            linear-gradient(180deg, #f8f8f2 0%, var(--paper) 100%);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1120px; padding-top: 2.5rem; }
    h1, h2, h3 { color: var(--ink); }
    h1 { font-family: Georgia, "Times New Roman", serif; font-weight: 500; letter-spacing: 0; }
    h2, h3 { letter-spacing: 0; }
    [data-testid="stForm"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: rgba(255, 255, 252, .82);
        padding: 1.25rem 1.5rem;
    }
    [data-testid="stMetric"] {
        background: #fffefa;
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: .85rem 1rem;
    }
    [data-testid="stMetricValue"] { color: var(--leaf); }
    [data-testid="stSidebar"] { background: #e8efe8; }
    div.stButton > button[kind="primaryFormSubmit"] {
        background: var(--leaf);
        border-color: var(--leaf);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def engineer_features(inputs: dict[str, float | int]) -> np.ndarray:
    """Build the ANN's 15 inputs in the exact order used during training."""
    chills = int(inputs["Chills"])
    values = {
        **inputs,
        "High_Fever": int(inputs["Fever_Temp"] >= 38.0),
        "Fever_x_Chills": inputs["Fever_Temp"] * chills,
        "Severe_Headache": int(inputs["Headache_Severity"] >= 7.0),
        "Symptom_Count": sum(int(inputs[name]) for name in ("Chills", "Vomiting", "Diarrhea", "Sweating", "Nausea")),
        "Vulnerable_Age": int(inputs["Age"] < 5 or inputs["Age"] > 60),
    }
    return np.asarray([[values[name] for name in FEATURE_ORDER]], dtype=np.float32)


@st.cache_resource
def load_model_and_scaler() -> tuple[keras.Model, np.ndarray, np.ndarray, str]:
    model_path = next((path for path in MODEL_PATHS if path.is_file()), None)
    if model_path is None:
        raise FileNotFoundError("Could not find best_malaria.keras in the project or malaria_saved_files folder.")
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Could not find the raw malaria dataset at {DATA_PATH}.")

    raw = pd.read_csv(DATA_PATH)
    required = set(FEATURE_ORDER[:10])
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Raw malaria data is missing columns: {', '.join(sorted(missing))}")

    engineered = pd.DataFrame(
        np.concatenate(
            [
                engineer_features(row).astype(np.float64)
                for row in raw.loc[:, FEATURE_ORDER[:10]].to_dict(orient="records")
            ],
            axis=0,
        ),
        columns=FEATURE_ORDER,
    )
    mean = engineered.mean(axis=0).to_numpy(dtype=np.float32)
    scale = engineered.std(axis=0, ddof=0).to_numpy(dtype=np.float32)
    scale[scale == 0] = 1.0

    model = keras.models.load_model(model_path, compile=False)
    if model.input_shape[-1] != len(FEATURE_ORDER):
        raise ValueError(
            f"Expected a {len(FEATURE_ORDER)}-feature model, received input shape {model.input_shape}."
        )
    return model, mean, scale, str(model_path.relative_to(BASE_DIR))


st.sidebar.header("About this model")
st.sidebar.write("A neural network trained on the malaria project dataset.")
st.sidebar.metric("Model inputs", "15 features")
st.sidebar.caption(
    "The original fitted scaler was not saved. Input scaling is reconstructed from the included raw cohort, "
    "so scores may differ from the notebook's exact training pipeline."
)
st.sidebar.divider()
st.sidebar.caption("Educational demonstration only. This model has not been clinically validated.")

st.caption("MALARIA · MODEL DEMONSTRATION")
st.title("Symptom assessment")
st.write("Enter the reported symptoms to see the model's classification and output score.")

try:
    model, scaler_mean, scaler_scale, model_label = load_model_and_scaler()
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"The malaria model could not be prepared: {error}")
    st.stop()

with st.form("malaria_assessment"):
    patient_col, symptom_col = st.columns([0.9, 1.1], gap="large")

    with patient_col:
        st.subheader("Patient details")
        age = st.number_input("Age (years)", min_value=1, max_value=100, value=25, step=1)
        temperature = st.number_input(
            "Body temperature (°C)", min_value=34.0, max_value=42.0, value=37.0, step=0.1, format="%.1f"
        )
        headache = st.slider("Headache severity", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        muscle_pain = st.slider("Muscle pain", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        fatigue = st.slider("Fatigue", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

    with symptom_col:
        st.subheader("Symptoms")
        symptom_left, symptom_right = st.columns(2)
        with symptom_left:
            chills = st.checkbox("Chills")
            vomiting = st.checkbox("Vomiting")
            diarrhea = st.checkbox("Diarrhea")
        with symptom_right:
            sweating = st.checkbox("Sweating")
            nausea = st.checkbox("Nausea")
        st.caption("Select symptoms that are currently present.")

    submitted = st.form_submit_button("Run assessment", type="primary", use_container_width=True)

if submitted:
    raw_inputs = {
        "Age": age,
        "Fever_Temp": temperature,
        "Chills": int(chills),
        "Headache_Severity": headache,
        "Vomiting": int(vomiting),
        "Diarrhea": int(diarrhea),
        "Muscle_Pain": muscle_pain,
        "Fatigue_Level": fatigue,
        "Sweating": int(sweating),
        "Nausea": int(nausea),
    }
    features = engineer_features(raw_inputs)
    scaled_features = (features - scaler_mean) / scaler_scale
    score = float(model.predict(scaled_features, verbose=0)[0, 0])
    classification = "Malaria flagged" if score >= 0.5 else "Malaria not flagged"

    st.divider()
    st.subheader("Model output")
    result_col, score_col = st.columns([1, 1])
    with result_col:
        if score >= 0.5:
            st.error(f"**{classification}**")
        else:
            st.success(f"**{classification}**")
    with score_col:
        st.metric("Model score", f"{score:.1%}", help="Raw sigmoid output; it is not a clinically calibrated probability.")
    st.progress(min(max(score, 0.0), 1.0))
    st.caption(f"Classification threshold: 50% · Model file: `{model_label}`")

    with st.expander("Engineered inputs used by the model"):
        st.write(
            f"High fever: {temperature >= 38.0} · Severe headache: {headache >= 7.0} · "
            f"Symptom count: {int(features[0, 13])} · Vulnerable-age flag: {bool(features[0, 14])}"
        )

st.warning(
    "This is an educational demonstration, not a diagnostic tool. The dataset, model, and reconstructed "
    "preprocessing have not been clinically validated. Seek qualified medical care for health concerns."
)