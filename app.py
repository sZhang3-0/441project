import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import io


MODEL_BUNDLE_PATH = Path("fraud_logreg.pkl")  



def load_bundle(path: Path):
    """Load the serialized bundle containing the sklearn pipeline and metadata."""
    if not path.exists():
        st.error(f"⚠️  Could not find model bundle at {path}")
        st.stop()
    return joblib.load(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Hour, TimeSincePrev, and IsNight exist (derived from Time)."""
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
    """Drop unseen cols, add missing ones with 0, order to match training."""
    df = df[[c for c in df.columns if c in required_cols]].copy()
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    return df[required_cols]

###############################################################################
# Streamlit App
###############################################################################

st.set_page_config(page_title="Fraud‑Detection Demo", layout="wide")
st.title("\U0001F50E Credit Card Fraud Detection")

# ---- Load model once (cached) ----
@st.cache_resource(show_spinner=True)
def get_model_bundle():
    return load_bundle(MODEL_BUNDLE_PATH)

bundle = get_model_bundle()
model_pipe = bundle["model"]
cutoff = bundle["cutoff"]
auc_train = bundle.get("auc_train") 
# ---- Model health panel ----
st.subheader("Model health")
cols = st.columns(3 if auc_train is not None else 2)
cols[0].metric("Model type", type(model_pipe).__name__)
if auc_train is not None:
    cols[1].metric("Training AUC", f"{auc_train:.3f}")
    cols[2].metric("Cutoff (≥ fraud)", f"{cutoff:.3f}")
else:
    cols[1].metric("Cutoff (≥ fraud)", f"{cutoff:.3f}")

st.divider()

# ---- File upload + predict button ----
file = st.file_uploader("Upload transactions (CSV)", type=["csv"])


def run_inference(uploaded_bytes: bytes):
    """Return dataframe with raw prob, percent prob, and prediction."""
    df = pd.read_csv(io.BytesIO(uploaded_bytes))

    if "Class" in df.columns and "is_fraud" not in df.columns:
        df = df.rename(columns={"Class": "is_fraud"})

    df = engineer_features(df)

    X_new = df.drop(columns=["is_fraud"], errors="ignore")
    X_new = align_columns(X_new, model_pipe.feature_names_in_)

    y_prob = model_pipe.predict_proba(X_new)[:, 1]
    y_pred = (y_prob >= cutoff).astype(int)

    df["probability"] = y_prob                  
    df["probability_pct"] = (y_prob * 100).round(4) 
    df["prediction"] = y_pred                     

    return df

if file is not None:
    if st.button("Predict", type="primary"):
        with st.spinner("Scoring uploaded data …"):
            results_df = run_inference(file.getvalue())

        st.success("Done!")

        # ---- Results summary ----
        n_total = len(results_df)
        n_fraud = int(results_df["prediction"].sum())
        fraud_rate = n_fraud / n_total * 100 if n_total else 0

        st.subheader("Results summary")
        met_cols = st.columns(3)
        met_cols[0].metric("Total rows scored", f"{n_total:,}")
        met_cols[1].metric("Fraud (prediction=1)", f"{n_fraud:,}",
                           delta=f"{fraud_rate:.2f}%")
   
        st.dataframe(
            results_df[["prediction"]].head(20)
            
            .style.format({"probability (%)": "{:.2f}%"})
        )


        )
else:
    st.info("⬆️  Upload a CSV file to enable the Predict button.")

