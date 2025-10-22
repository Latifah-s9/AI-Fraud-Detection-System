import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score

# إعداد الصفحة
st.set_page_config(page_title="AI Fraud Detection System", layout="wide")
st.title("🧠 AI Fraud Detection System")

# تحميل النموذج
try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error("❌ لم يتم العثور على model.pkl، شغّل train_model.py أولاً.")
    st.stop()

# دالة لتوحيد الأعمدة (تصحيح ترتيب أو نقص)
def normalize_uploaded_df(df):
    expected_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    missing = [c for c in expected_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_cols + ['Class']]

    # إذا فيه أعمدة ناقصة نضيفها ونعبّيها بصفر
    for c in missing:
        df[c] = 0

    # نحافظ على الترتيب الصحيح
    df = df[[c for c in expected_cols if c in df.columns]]
    return df, missing, extra

# رفع ملف CSV من المستخدم
uploaded = st.file_uploader("📂 Upload transactions CSV", type="csv")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        df, missing_cols, extra_cols = normalize_uploaded_df(df)

        if missing_cols:
            st.warning(f"⚠️ تمت إضافة الأعمدة الناقصة تلقائيًا: {missing_cols}")
        if extra_cols:
            st.info(f"ℹ️ تم تجاهل الأعمدة الزائدة: {extra_cols}")

        # التنبؤ باستخدام النموذج
        preds = model.predict(df)
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[:, 1]
            df["Fraud_Probability"] = probs
        df["Prediction"] = preds

        # عرض النتائج العامة
        st.subheader("📈 Predictions Overview")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Transactions", len(df))
        with c2:
            st.metric("Fraudulent", int((df["Prediction"] == 1).sum()))
        with c3:
            st.metric("Normal", int((df["Prediction"] == 0).sum()))

        fig = px.histogram(df, x="Prediction", title="🟦 Prediction Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # تحليل SHAP (توضيح قرار النموذج)
        if len(df) > 0:
            try:
                st.subheader("🔍 Feature Importance (SHAP Analysis)")
                sample = df.sample(min(200, len(df)), random_state=42)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(sample)
                shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
                st.pyplot(bbox_inches="tight")
            except Exception as e:
                st.warning("⚠️ لم يتم عرض تحليل SHAP (ربما بسبب نوع النموذج).")

        # دقة النموذج (لو عند المستخدم عمود Class فعلي)
        if "Class" in df.columns:
            try:
                acc = accuracy_score(df["Class"], preds)
                pre = precision_score(df["Class"], preds)
                rec = recall_score(df["Class"], preds)
                st.subheader("📊 Model Evaluation Metrics")
                st.write(f"**Accuracy:** {acc:.3f}")
                st.write(f"**Precision:** {pre:.3f}")
                st.write(f"**Recall:** {rec:.3f}")
            except Exception:
                st.info("ℹ️ لم يتم حساب المقاييس بسبب اختلاف تنسيق عمود Class.")

        # تنزيل النتائج
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Predictions CSV", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error("❌ حدث خطأ أثناء تحليل الملف. الرجاء التحقق من التنسيق.")
        st.text(str(e))
else:
    st.info("⬆️ قم برفع ملف CSV للحصول على التوقعات.")
