import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

# Set page config for a clean dashboard look
st.set_page_config(
    page_title="2026 F1 Race Predictor", page_icon="🏎️", layout="wide"
)

st.title("🏎️ 2026 F1 Race Winner & Podium Predictior")
st.markdown("Predict race outcomes using 2026 exclusive form and grid data.")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("f1_race_winners_data.csv")
    df["IsWinner"] = (df["Position"] == 1).astype(int)
    df["IsPodium"] = (df["Position"] <= 3).astype(int)
    df["QualiPosition"] = df["QualiPosition"].fillna(df["GridPosition"])
    return df

df = load_data()
df_2026 = df[df["Season"] == 2026].copy()

# Sidebar Interactive Parameters
st.siderbar.header("Model Calibration")
beta_win = st.siderbar.slider(
    "Win Temperature (Beta)", 1.0, 6.0, 2.6, step=0.1
)
beta_podium = st.sidebar.slider(
    "Podium Temperature (Beta)", 1.0, 6.0, 2.0, step=0.1
)

# Simulated Grid Input
st.subheader("Baku 2026 Starting Grid Setup")
