import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="AI Fraud Detection System", page_icon="💳🛡️", layout="wide")
st.title("💳🛡️ AI Fraud Detection System")

try:
    model = joblib.load("model.pkl")
except Exception:
    st.error("❌ model.pkl not found — run train_model.py first.")
    st.stop()

def get_trained_features(m):
    try:
        return m.get_booster().feature_names
    except Exception:
        return None

TRAINED = get_trained_features(model)
FALLBACK = [f"V{i}" for i in range(1, 29)] + ["Amount"]

def align_columns(raw_df):
    df = raw_df.copy()
    lower_map = {c.lower(): c for c in df.columns}
    want = TRAINED if TRAINED else FALLBACK
    use_cols = []
    for c in want:
        lc = c.lower()
        if lc in lower_map:
            use_cols.append(lower_map[lc])
        else:
            df[c] = 0
            use_cols.append(c)
    extra = [c for c in df.columns if c not in use_cols and c.lower() not in {w.lower() for w in want}]
    df = df[use_cols]
    return df, extra

uploaded = st.file_uploader("📂 Upload transactions CSV", type="csv")

if uploaded is not None:
    try:
        data_in = pd.read_csv(uploaded)
        st.subheader("👀 Data Preview")
        st.dataframe(data_in.head())

        X, extras = align_columns(data_in)
        if extras:
            st.info(f"Ignored extra columns: {extras}")

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[:, 1]
            pred = (prob >= 0.5).astype(int)
            X_out = data_in.copy()
            X_out["Fraud_Probability"] = prob
            X_out["Prediction"] = pred
        else:
            pred = model.predict(X)
            X_out = data_in.copy()
            X_out["Prediction"] = pred

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total", len(X_out))
        with c2: st.metric("Fraudulent", int((X_out["Prediction"] == 1).sum()))
        with c3: st.metric("Normal", int((X_out["Prediction"] == 0).sum()))

        fig = px.histogram(X_out, x="Prediction", title="📊 Prediction Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # SHAP Analysis robust for all environments
        try:
            st.subheader("🔍 Feature Importance (SHAP Analysis)")
            sample = X.sample(min(500, len(X)), random_state=42)
            try:
                explainer = shap.TreeExplainer(model.get_booster())
                shap_values = explainer.shap_values(sample)
            except Exception:
                explainer = shap.Explainer(model)
                shap_values = explainer(sample)
            shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
            st.pyplot(bbox_inches="tight", clear_figure=True)
            plt.clf()
        except Exception as e:
            st.info(f"⚠️ SHAP not displayed (reason: {str(e)})")

        # Performance metrics: only when you run in Colab with Class column
        if "Class" in data_in.columns:
            try:
                y_true = pd.to_numeric(data_in["Class"], errors="coerce").fillna(0).astype(int)
                acc = accuracy_score(y_true, pred)
                pre = precision_score(y_true, pred, zero_division=0)
                rec = recall_score(y_true, pred, zero_division=0)
                f1 = f1_score(y_true, pred, zero_division=0)
                st.caption(f"📈 Accuracy={acc:.4f} | Precision={pre:.4f} | Recall={rec:.4f} | F1={f1:.4f}")
            except Exception:
                pass

        csv = X_out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error("❌ Error while processing the file.")
        st.text(str(e))
else:
    st.info("📤 Upload a CSV file to start predictions.")
