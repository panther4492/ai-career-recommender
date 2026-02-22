import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Career Recommendation", layout="wide")

st.title("🤖 AI Career Recommendation System")
st.write("Machine Learning Based Career Predictor")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    encoder = joblib.load("encoder.pkl")
    data = pd.read_csv("Data_final.csv")
    features = data.drop("Career", axis=1).columns
    return model, encoder, features

model, encoder, features = load_assets()

# ---------------- USER INPUT ----------------
st.sidebar.header("Enter Skill Scores")

user_input = []
for feature in features:
    val = st.sidebar.slider(feature, 0.0, 10.0, 5.0)
    user_input.append(val)

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Career"):
    prediction = model.predict([user_input])
    career = encoder.inverse_transform(prediction)[0]
    st.success(f"🎯 Recommended Career: {career}")

st.markdown("---")
st.caption("AI + ML Powered App")
