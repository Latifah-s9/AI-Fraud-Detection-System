import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.express as px
st.set_page_config(page_title="AI Fraud Detection", layout="wide")
st.title("💳 AI Fraud Detection System")
try:
    model = joblib.load("model.pkl")
except:
    st.error("model.pkl not found. Run train_model.py to create it.")
    st.stop()
uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.subheader("Data preview")
    st.dataframe(data.head())
    X = data.copy()
    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1] if hasattr(model, "predict_proba") else None
    X["Prediction"] = preds
    if probs is not None:
        X["Prob"] = probs
    c1,c2,c3 = st.columns(3)
    with c1:
        st.metric("Total", len(X))
    with c2:
        st.metric("Fraud", int((X["Prediction"]==1).sum()))
    with c3:
        st.metric("Normal", int((X["Prediction"]==0).sum()))
    fig = px.histogram(X, x="Prediction", title="Prediction distribution")
    st.plotly_chart(fig, use_container_width=True)
    if len(X) > 0:
        explainer = shap.TreeExplainer(model)
        sample = X.sample(min(200, len(X)))
        shap_values = explainer.shap_values(sample)
        st.subheader("Top features (SHAP)")
        shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
        st.pyplot(bbox_inches="tight")
    csv = X.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions", csv, "predictions.csv", "text/csv")
else:
    st.info("Upload a CSV file to get predictions.")
