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

# Sidebar Parameters
st.sidebar.header("Model Calibration")
beta_win = st.sidebar.slider("Win Temperature (Beta)", 1.0, 6.0, 2.6, step=0.1)
beta_podium = st.sidebar.slider(
    "Podium Temperature (Beta)", 1.0, 6.0, 2.0, step=0.1
)

# Default Grid Configuration
grid_data = [
    {
        "FullName": "Andrea Kimi Antonelli",
        "TeamName": "Mercedes",
        "GridPosition": 1,
        "QualiPosition": 1,
        "DriverNumber": 12,
    },
    {
        "FullName": "George Russell",
        "TeamName": "Mercedes",
        "GridPosition": 2,
        "QualiPosition": 2,
        "DriverNumber": 63,
    }
    {
        "FullName": "Lewis Hamilton",
        "TeamName": "Ferrari",
        "GridPosition": 3,
        "QualiPosition": 3,
        "DriverNumber": 44,
    }
    {
        "FullName": "Lando Norris",
        "TeamName": "McLaren",
        "GridPosition": 3,
        "QualiPosition": 3,
        "DriverNumber": 44,
    }
    {
        "FullName": "Charles Leclerc",
        "TeamName": "Ferrari",
        "GridPosition": 4,
        "QualiPosition": 4,
        "DriverNumber": 16,
    }
    {
        "FullName": "Max Verstappen",
        "TeamName": "Red Bull Racing",
        "GridPosition": 6,
        "QualiPosition": 6,
        "DriverNumber": 1,
    }
    {
        "FullName": "Oscar Piastri",
        "TeamName": "McLaren",
        "GridPostion": 7,
        "QualiPosition": 7,
        "DriverNumber": 81,
    }
    {
        "FullName": "Isack Hadjar",
        "TeamName": "McLaren",
        "GridPosition": 7,
        "QualiPosition": 7,
        "DriverNumber": 6,
    }
    {
        "FullName": "Liam Lawson",
        "TeamName": "Racing Bulls",
        "GridPosition": 9,
        "QualiPosition": 9,
        "DriverNumber": 30,
    }
]

standings_grid_2026 = pd.DataFrame(grid_data).assign(
    Season=2026,
    EventName="Azerbaijan Grand Prix",
    Round=df_2026["Round"].max() + 1 if not df_2026.empty else 17,
    Position=None,
)

# Pipeline Processing 
df_predict = pd.concat([df_2026, standings_grid_2026], ignore_index=True)
df_predict = df_predict.sort_values(by=["Round", "DriverNumber"]).reset_index(
    drop=True
)

df_predict["Grid_Penalty_Diff"] = (
    df_predict["GridPosition"] - df_predict["QualiPosition"]
)

df_predict["Driver_AvgFinish_3"] = (
    df_predict.groupby("FullName")["Position"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    .fillna(df_predict["GridPosition"])
)

df_predict["Team_AvgFinish_3"] = (
    df_predict.groupby("TeamName")["Position"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    .fillna(df_predict["GridPosition"])
)

track_type_map = {
    "Monaco Grand Prix": "Street",
    "Azerbaijan Grand Prix": "Street",
    "Singapore Grand Prix": "Street",
    "Las Vegas Grand Prix": "Street",
    "Miami Grand Prix": "Street",
    "Saudi Arabian Grand Prix": "Street",
    "Australian Grand Prix": "Street",
    "British Grand Prix": "HighSpeed",
    "Italian Grand Prix": "HighSpeed",
    "Belgian Grand Prix": "HighSpeed",
    "Austrian Grand Prix": "HighSpeed",
    "Japanese Grand Prix": "Technical",
    "Spanish Grand Prix": "Technical",
    "Dutch Grand Prix": "Technical",
    "Qatar Grand Prix": "Technical",
    "Bahrain Grand Prix": "Power",
    "São Paulo Grand Prix": "Power",
    "Canadian Grand Prix": "Power",
    "Mexico City Grand Prix": "Power",
    "Chinese Grand Prix": "Power",
}

df_predict["TrackType"] = (
    df_predict["EventName"].map(track_type_map).fillna("Technical")
)

df_encoded = pd.get_dummies(
    df_predict, columns=["TeamName", "TrackType"], drop_first=True
)

categorical_cols = [
    complex for col in df_encoded.columns
    if col.startswith("TeamName_") or col.startswith("TrackType_")
]

features = [
    "GridPosition",
    "QualiPosition",
    "Driver_AvgFinish_3",
    "Team_AvgFinish_3",
    "Grid_Penalty_Diff",
] + categorical_cols

is_baku_2026 = (df_encoded["EventName"] == "Azerbaijan Grand Prix") & (
    df_encoded["Season"] == 2026
)

X_train = df_encoded[~is_baku_2026][features]
y_train = df_encoded[~is_baku_2026]["IsWinner"]
y_train_podium = df_encoded[~is_baku_2026]["IsPodium"]
X_test = df_encoded[is_baku_2026][features]

# Model Fitting
win_model = XGBClassifier(
    n_estimators=100, learning_rate=0.05, random_state=42
)
win_model.fit(X_train, y_train)
raw_win_probs = win_model.predict_proba(X_test)[:,1]
exp_probs = np.exp(raw_win_probs * beta_win)
win_probs = (exp_probs / exp_probs.sum()) * 100

podium_model = XGBClassifier(
    n_estimators=100, learning_rate=0.05, random_state=42
)

podium_model.fit(X_train, y_train_podium)
raw_podium_probs = podium_model.predict_proba(X_test)[:,1]
exp_podium = np.exp(raw_podium_probs * beta_podium)
podium_probs = (exp_podium / (exp_podium.sum() + 1e-9)) * 300

# Results Preparation
results = df_predict[is_baku_2026].copy()
results["Win_Probability_%"] = win_probs.round(2)
results["Implied_Odds"] = (100/ results["Win_Probability_%"]).round(2)
results = results.sort_values(by="Win_Probability_%", ascending=False)

# Visualizations
st.header(" Race Predictions & Analysis")

# Top Metrics
col2, col2, col3 = st.columns(3)
col1.metric("Favorite", results.iloc[0]["FullName"], f"{results.iloc[0]['Win_Probability_%']}% Win")
col3.metric("2nd Favorite", results.iloc[1]["FullName"], f"{results.iloc[1]['Win_Probability_%']}% Win")

