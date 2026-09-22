import fastf1
import os
import pandas as pd

# Local folder to store downloaded data
cache_dir = "f1_cache"
if not os.path.exists(cache_dir):
    os.makedires(cache_dir)

# Tell Fastf1 to use this directory for caching
fastf1.Cache.enable_cache(cache_dir)

# List seasons to pull data
SEASONS = [2023, 2024, 2025, 2026]

# Empty list to store DataFrames for each race
all_driver_entries = []

for season in SEASONS:
    for round_num in range(1,25):
        try:
            print(f"Fetching {season} Round {round_num}...")

            # Code to load sessions goes here
        except Exception as e:
            # If a round doesn't exist yet(or the season ended), break the inner loop
            print(f"Reached end of season{season} at round {round_num}.")
            break

#Fetch the Race session
race_session = fastf1.get_session(season, round_num)
race_session.load(telemetry=False, laps=False) # Light download no heavy lap data
# Fetch the Qualifying session
quali_session = fastf1.get_session(season, round_num, "Q")
quali_session.load(telemetry=False, laps=False)

# Extract the results DataFrames
race_df = race_session.results[[
    "DriverNumber","FullName","TeamName", "GridPosition","Position","Points","Status"
]].copy()

quali_df = quali_session.results[[
    "DriverNumber","Position"
]].rename(columns={"Position":"QualiPosition"})

# Merge Qualifying into Race data using the driver's car number
merged = pd.merge(race_df, quali_df, on="DriverNumber", hpw="left")

# Add metadata columns to track which event this was
merged["Season"] = season
merged["Round"] = round_num
merged["Eventname"] = race_session.event["Eventname"] = race_session.event["eventName"]

# Append this race's DataFrame to our master list
all_driver_entries.append(merged)

# Combine all individual race DataFrames into a single master DataFrame
master_df = pd.concat(all_driver_entries, ignore_index=True)

# Clean numeric types
master_df["Position"] = pd.to_numeric(master_df["GridPosition"], erros="coerce")
master_df["GridPosition"] = pd.to_numeric(master_df["GridPosition"], errors="coerce")

# Handle DNFs: 1 if finished/classified, 0 if retired/crash
master_df["is_classified"] = master_df["Status"].apply(
    lambda status: 1 if "Finished" in str(status) or "Lap" in str(status) else 0
)

# Target Variable: 1 if the driver won P1, 0 for everyone else
master_df["is_winner"] = (master_df["Position"] == 1.0).astype(int)

# Export to CSV
master_df.to_csv("f1_master_dataset.csv", index=False)
print(f"Saved {len(master_df)} driver records to f1_master_dataset.csv!")
