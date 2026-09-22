import pandas as pd
from xgboost import XGBClassifier

# Load Historical Dataset
df = pd.read_csv("f1_race_winners_data.csv")

# Preprocess target and missing values
df["IsWinner"] = (df["Position"] == 1).astype(int)
df["IsPodium"] = (df["Position"] <= 3).astype(int)
df["QualiPosition"] = df["QualiPosition"].fillna(df["GridPosition"])

# Filter down to 2026 data
df_2026 = df[df["Season"] == 2026].copy()

# Create simulated grid entires for the upcoming 2026 Azerbaijan GP based on Championship standings
# Adjust Grid positions below based on expected or official qualifying results

standings_grid_2026 = pd.DataFrame([
    {"FullName": "Andrea Kimi Antonelli", "TeamName": "Mercedes", "GridPosition":1, "QualiPosition": 1},
    {"FullName": "George Russell", "TeamName": "Mercedes", "GridPosition": 2, "QualiPosition": 2},
    {"FullName": "Lewis Hamilton", "TeamName": "Ferrari", "GridPosition": 3, "QualiPosition": 3},
    {"FullName": "Lando Norris", "TeamName": "Mclaren", "GridPosition": 4, "QualiPosition": 4},
    {"FullName": "Charles Leclerc", "TeamName": "Ferrari", "GridPosition": 5, "QualiPosition": 5},
    {"FullName": "Max Verstappen", "TeamName": "Red Bull Racing", "GridPosition": 6, "QualiPosition": 6},
    {"FullName": "Oscar Piastri", "TeamName": "Mclaren","GridPosition": 7, "QualiPosition":7},
    {"FullName":"Issack Hadjar", "TeamName": "Red Bull Racing", "GridPosition": 8, "QualiPosition": 8},
    {"FullName":"Liam Lawson", "TeamName": "Racing Bulls", "GridPosition": 9, "QualiPosition": 9},
    {"FullName":"Pierre Gasly", "TeamName": "Alpine", "GridPosition": 10, "QualiPosition": 10},
]).assign(
    Season=2026, 
    EventName="Azerbaijan Grand Prix", 
    Round=df_2026["Round"].max() + 1 if not df_2026.empty else 17,
    Position=None
)

# Combine 2026 Historical races with upcoming baku entry
df_predict = pd.concat([df_2026, standings_grid_2026], ignore_index=True)

# Sort chronologically to calculate rolling stats purely form 2026 races
df_predict = df_predict.sort_values(by=["Round", "DriverNumber"]).reset_index(drop=True)

# Upgrade
df_predict["Grid_Penalty_Diff"] = df_predict["GridPosition"] - df_predict["QualiPosition"]

# Compute 2026-only Rolling Momentum Feautures
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

# Map Circuit Characteristics
track_type_map = {
    'Monaco Grand Prix': 'Street', 'Azerbaijan Grand Prix': 'Street',
    'Singapore Grand Prix': 'Street', 'Las Vegas Grand Prix': 'Street',
    'Miami Grand Prix': 'Street', 'Saudi Arabian Grand Prix': 'Street',
    'Australian Grand Prix': 'Street',
    'British Grand Prix': 'HighSpeed', 'Italian Grand Prix': 'HighSpeed',
    'Belgian Grand Prix': 'HighSpeed', 'Austrian Grand Prix':'HighSpeed',
    'Japanese Grand Prix': 'Technical', 'Spanish Grand Prix': 'Technical',
    'Dutch Grand Prix': 'Technical', 'Qatar Grand Prix': 'Technical',
    'Bahrain Grand Prix': 'Power', 'São Paulo Grand Prix': 'Power',
    'Canadian Grand Prix': 'Power', 'Mexico City': 'Power',
    'Chinese Grand Prix': 'Power'
}
df_predict['TrackType'] = df_predict['EventName'].map(track_type_map).fillna('Technical')

# One hot encoding
df_encoded = pd.get_dummies(df_predict, columns=["TeamName", "TrackType"], drop_first=True)

categorical_cols = [
    col for col in df_encoded.columns
    if col.startswith("TeamName_") or col.startswith("TrackType_")
]
features = [
    "GridPosition",
    "QualiPosition",
    "Driver_AvgFinish_3",
    "Team_AvgFinish_3",
    "Grid_Penalty_Diff"
] + categorical_cols

# Time based Split: train on past races, predict on Baku 2026
is_baku_2026 = (df_encoded["EventName"] == "Azerbaijan Grand Prix") & (df_encoded["Season"] == 2026)

X_train = df_encoded[~is_baku_2026][features]
y_train = df_encoded[~is_baku_2026]["IsWinner"]
y_train_podium = df_encoded[~is_baku_2026]["IsPodium"]

X_test = df_encoded[is_baku_2026][features]

# Train XGBoost model
model = XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Normalize Win Probabilities (sum to 100%)
raw_win_probs = model.predict_proba(X_test)[:,1]
win_probs_normalized = (raw_win_probs / raw_win_probs.sum()) * 100

# Train Podium Model
podium_model = XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
podium_model.fit(X_train, y_train_podium)

# Normalize podium probabilites (sum to 100%)
raw_podium_probs = podium_model.predict_proba(X_test)[:,1]
podium_probs_normalized = (raw_podium_probs / raw_podium_probs.sum()) * 300

# Format and print outpu
baku_results = df_predict[is_baku_2026].copy()
baku_results["Win_Probability_%"] = win_probs_normalized.round(2)
baku_results["Podium_Probability_%"] = podium_probs_normalized.round(2)
baku_results["Implied_Odds"] = (100 / baku_results["Win_Probability_%"]).round(2)

print("\n --- 2026 Baku GP Predictions ---")
print(
    baku_results[
        ["FullName", "TeamName", "GridPosition", "Win_Probability_%", "Podium_Probability_%", "Implied_Odds"]
    ]
    .sort_values(by="Win_Probability_%", ascending=False)
    .to_string(index=False)
)