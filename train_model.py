import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load Data
df = pd.read_csv("f1_race_winners_data.csv")

# Basic Processing
df["IsWinner"] = (df["Position"] == 1).astype(int)
df["QualiPosition"] = df["QualiPosition"].fillna(df["GridPosition"])

# Sort Chronologically to compute rolling statistics correctly
df = df.sort_values(by=["Season", "Round", "DriverNumber"]).reset_index(drop=True)

# Create Rolling features (last 3 races)
# shift(1) makes sure we only look at past races so no data leakage
df["Driver_AvgFinish_3"] = (
    df.groupby("FullName")["Position"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

df["Team_AvgFinish_3"] = (
    df.groupby("TeamName")["Position"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

# Fill NaN values for the very first few races of a season with grid position fall back
df["Driver_AvgFinish_3"] = df["Driver_AvgFinish_3"].fillna(df["GridPosition"])
df["Team_AvgFinish_3"] = df["Team_AvgFinish_3"].fillna(df["GridPosition"])

# Filter to 2025 data + One hot encode
df_2025 = df[df["Season"] == 2025].copy()
df_encoded = pd.get_dummies(df_2025, columns=["TeamName"], drop_first=True)

# Define expanded feature list
team_cols = [col for col in df_encoded.columns if col.startswith("TeamName_")]
features = [
    "GridPosition",
    "QualiPosition",
    "Driver_AvgFinish_3",
    "Team_AvgFinish_3"
] + team_cols

X = df_encoded[features]
y = df_encoded["IsWinner"]

# Train/Test Split on 2025 data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check feature importance
importance_df = pd.DataFrame({
    'Feautre': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n...Updated Feauture Importances...")
print(importance_df.head(6))

# Street tracks: Monaco, Baku, Singapore, Las Vegas, Miami, Jeddah, Melbourne
# High speed tracks: Monza, Silverstone, Spa, Red Bull Ring
# Technical: Suzuka, COTA, Zandvooort, Barcelona, Hungaroring, Losail
# Power: Bahrain, Interlagos, Montreal, Mexico City, Shanghai

# Map EventName to track characteristic
track_type_map = {
    'Monaco Grand Prix': 'Street', 'Azerbaijan Grand': 'Street',
    'Singapore Grand Prix': 'Street', 'Las Vegas Grand Prix': 'Street',
    'Miami Grand Prix': 'Street', 'Saudi Arabian Grand Prix': 'Street',
    'Australian Grand Prix': 'Street',

    'British Grand Prix': 'HighSpeed', 'Italian Grand Prix': 'HighSpeed',
    'Belgian Grand Prix': 'HighSpeed', 'Austrian Grand Prix': 'HighSpeed',

    'Japanese Grand Prix': 'Technical', 'United States Grand Prix': 'Technical',
    'Dutch Grand Prix': 'Technical', 'Spanish Grand Prix': 'Technical', 
    'Hungarian Grand Prix': 'Technical', 'Qatar Grand Prix': 'Technical',

    'Bahrain Grand Prix': 'Power', 'São Paulo Grand Prix': 'Power',
    'Canadian Grand Prix': 'Power', 'Mexico City Grand Prix': 'Power',
    'Chinese Grand Prix': 'Power'
}

df['TrackType'] = df['Eventname'].map(track_type_map).fillna('Technical')

from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)


