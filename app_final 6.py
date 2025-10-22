import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score

st.set_page_config(page_title="AI Fraud Detection System", layout="wide")
st.title("🧠 AI Fraud Detection System")

# تحميل النموذج
try:
    model = joblib.load("model.pkl")
except Exception:
    st.error("❌ ملف النموذج غير موجود! شغّل train_model.py أولاً.")
    st.stop()

# استخراج أسماء الأعمدة التي تدرب عليها النموذج فعليًا
try:
    trained_features = model.get_booster().feature_names
except Exception:
    trained_features = None

# تصحيح الأعمدة المدخلة
def align_columns(df):
    # إذا ما قدرنا نعرف أعمدة التدريب نحاول التعرف اليدوي
    expected = trained_features or ['V'+str(i) for i in range(1, 29)] + ['Amount']
    df_cols = [c for c in df.columns if c in expected]
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]

    # إضافة الأعمدة الناقصة
    for c in missing:
        df[c] = 0

    # الاحتفاظ بالترتيب الصحيح
    df = df[expected]
    return df, missing, extra

uploaded = st.file_uploader("📂 Upload transactions CSV", type="csv")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        df, missing, extra = align_columns(df)

        if missing:
            st.warning(f"⚠️ تمت إضافة الأعمدة الناقصة: {missing}")
        if extra:
            st.info(f"ℹ️ تم تجاهل الأعمدة الزائدة: {extra}")

        preds = model.predict(df)
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[:, 1]
            df["Fraud_Probability"] = probs
        df["Prediction"] = preds

        # عرض النتائج
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Transactions", len(df))
        with c2:
            st.metric("Fraudulent", int((df["Prediction"] == 1).sum()))
        with c3:
            st.metric("Normal", int((df["Prediction"] == 0).sum()))

        fig = px.histogram(df, x="Prediction", title="Prediction Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # تحليل SHAP
        try:
            st.subheader("🔍 Feature Importance (SHAP)")
            sample = df.sample(min(200, len(df)), random_state=42)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)
            shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
            st.pyplot(bbox_inches="tight")
        except Exception:
            st.info("ℹ️ تحليل SHAP لم يتم (بسبب نوع النموذج أو البيانات).")

        # لو فيه عمود Class نحسب الدقة
        if "Class" in df.columns:
            try:
                acc = accuracy_score(df["Class"], preds)
                pre = precision_score(df["Class"], preds)
                rec = recall_score(df["Class"], preds)
                st.subheader("📈 Model Performance")
                st.write(f"Accuracy: {acc:.3f}")
                st.write(f"Precision: {pre:.3f}")
                st.write(f"Recall: {rec:.3f}")
            except Exception:
                pass

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Predictions", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error("❌ حدث خطأ أثناء معالجة الملف.")
        st.text(str(e))
else:
    st.info("⬆️ قم برفع ملف CSV للحصول على التوقعات.")
