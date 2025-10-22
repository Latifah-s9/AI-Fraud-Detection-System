import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# إعداد الصفحة
st.set_page_config(page_title="💳 AI Fraud Detection System", layout="wide")
st.title("🛡️ AI Fraud Detection System")

# تحميل النموذج
try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error("⚠️ لم يتم العثور على النموذج! درّب النموذج أولاً باستخدام train_model.py.")
    st.stop()

# توحيد البيانات
def normalize_uploaded_df(df):
    expected = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    lower_cols = [c.lower() for c in df.columns]
    missing = [col for col in expected if col.lower() not in lower_cols]
    extra = [col for col in df.columns if col.lower() not in [e.lower() for e in expected]]

    # حذف الأعمدة غير المطلوبة
    for col in extra:
        if col.lower() in ["time", "class"]:
            df = df.drop(columns=[col])

    for m in missing:
        df[m] = 0

    df = df[[c for c in expected if c in df.columns]]
    return df, missing, extra

# رفع الملف
uploaded = st.file_uploader("📂 Upload transactions CSV", type="csv")

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        st.subheader("📊 Data Preview")
        st.dataframe(data.head())

        X, missing_cols, extra_cols = normalize_uploaded_df(data)

        if missing_cols:
            st.warning(f"⚠️ Missing columns were filled automatically: {missing_cols}")
        if extra_cols:
            st.info(f"ℹ️ Extra columns ignored: {extra_cols}")

        preds = model.predict(X)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
            X["Fraud_Probability"] = probs
        X["Prediction"] = preds

        st.subheader("📈 Prediction Overview")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total", len(X))
        with c2:
            st.metric("Fraudulent", int((X["Prediction"] == 1).sum()))
        with c3:
            st.metric("Normal", int((X["Prediction"] == 0).sum()))

        # الرسم البياني بدون use_container_width
        fig = px.histogram(X, x="Prediction", title="📊 Prediction Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # التحليل الذكي (بديل SHAP)
        try:
            st.subheader("🤖 Feature Importance (AI Insight)")
            booster = model.get_booster()
            importance = booster.get_score(importance_type='gain')
            imp_df = pd.DataFrame({
                'Feature': list(importance.keys()),
                'Importance': list(importance.values())
            }).sort_values(by='Importance', ascending=False)

            fig_imp = px.bar(
                imp_df.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top Contributing Features',
                color='Importance',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.success("✅ Feature importance analyzed successfully.")
        except Exception:
            st.warning("⚠️ Could not display feature importance.")

        # المقاييس إذا وُجد العمود Class
        if "Class" in data.columns:
            y_true = data["Class"].astype(int)
            acc = accuracy_score(y_true, preds)
            pre = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            st.caption(f"📊 Accuracy={acc:.4f} | Precision={pre:.4f} | Recall={rec:.4f} | F1={f1:.4f}")

        csv = X.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", csv, "predictions.csv")

    except Exception as e:
        st.error("❌ Error while processing the file.")
        st.text(str(e))
else:
    st.info("📥 Upload a CSV file to start predictions.")
