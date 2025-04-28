import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import io
import tempfile


MODEL_BUNDLE_PATH = Path("/content/drive/MyDrive/Model/fraud_logreg.pkl") 


def load_bundle(path: Path):
    if not path.exists():
        st.error(f" Could not find model bundle at {path}")
        st.stop()
    return joblib.load(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    if "Hour" not in df.columns and "Time" in df.columns:
        df["Hour"] = ((df["Time"] // 3600) % 24).astype(np.int8)

    if "TimeSincePrev" not in df.columns and "Time" in df.columns:
        df["TimeSincePrev"] = (
            df["Time"].diff().fillna(df["Time"]).astype(np.float32)
        )

    if "IsNight" not in df.columns and "Hour" in df.columns:
        df["IsNight"] = (df["Hour"] < 6).astype(np.int8)

    return df


def align_columns(df: pd.DataFrame, required_cols: np.ndarray) -> pd.DataFrame:
    df = df.copy()

    # Drop extras
    df = df[[c for c in df.columns if c in required_cols]]

    # Add missing with neutral default (0)
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    # Re‑order
    df = df[required_cols]
    return df


st.set_page_config(page_title="Fraud‑Detection", layout="wide")
st.title("\U0001F50E Credit Card Fraud Detection")


@st.cache_resource(show_spinner=True)
def get_model_bundle():
    return load_bundle(MODEL_BUNDLE_PATH)

bundle = get_model_bundle()
model_pipe = bundle["model"]
cutoff = bundle["cutoff"]
auc_train = bundle.get("auc_train")


st.subheader("Model health")
cols = st.columns(3 if auc_train else 2)
cols[0].metric("Model type", type(model_pipe).__name__)
if auc_train is not None:
    cols[1].metric("Training AUC", f"{auc_train:.3f}")
    cols[2].metric("Probability cutoff", f"{cutoff:.3f}")
else:
    cols[1].metric("Probability cutoff", f"{cutoff:.3f}")

st.divider()

# ---- File upload + predict button ----
file = st.file_uploader("Upload transactions as CSV", type=["csv"])

def run_inference(uploaded_bytes: bytes):
    df = pd.read_csv(io.BytesIO(uploaded_bytes))


    if "Class" in df.columns and "is_fraud" not in df.columns:
        df = df.rename(columns={"Class": "is_fraud"})

    df = engineer_features(df)
    X_new = df.drop(columns=["is_fraud"], errors="ignore")
    X_new = align_columns(X_new, model_pipe.feature_names_in_)

    y_prob = model_pipe.predict_proba(X_new)[:, 1]
    y_pred = (y_prob >= cutoff).astype(int)

    df["probability"] = y_prob
    df["prediction"] = y_pred

    return df

if file is not None:
    if st.button("Predict", type="primary"):
        with st.spinner("Scoring uploaded data …"):
            results_df = run_inference(file.getvalue())

        st.success("Done!")
        n_frauds = int(results_df["prediction"].sum())
        n_total  = len(results_df)
        st.metric("Predicted frauds", f"{n_frauds:,}",
        delta=f"{n_frauds / n_total:.2%} of {n_total:,} transactions")

        # keep the first-20 preview
        st.dataframe(results_df[["prediction"]].head(20))
        st.dataframe(results_df[["prediction"]].head(20))


else:
    st.info("⬆️  Upload a CSV file to enable the Predict button.")

