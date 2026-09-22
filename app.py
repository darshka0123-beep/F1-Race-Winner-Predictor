import fastf1 
import os
import pandas as pd

if os.path.exists("f1_cache"):
    print("Path exists.")
else:
    os.makedirs("f1_cache")

fastf1.Cache.enable_cache("f1_cache")

SEASONS = [2023, 2024, 2025, 2026]
all_driver_entires = []

for year in SEASONS:
    for round_num in range(1,25):
        try:
            race_session = fastf1.get_session(year,round_num,"R")
            race_session.load(telemetry=False, laps=False)
            quali_session = fastf1.get_session(year,round_num,"Q")
            quali_session.load(telemetry=False, laps=False)
            race = race_session.results.copy()
            quali = quali_session.results.copy()
            race_df = race[["DriverNumber", "FullName", "TeamName", "GridPosition", "Position", "Points", "Status"]].copy()
            quali_df = quali[["DriverNumber", "Position"]].rename(columns={"Position":"QualiPosition"})
            merged = pd.merge(race_df,quali_df, on="DriverNumber", how="left")
            # how="left" tells pandas to keep all drivers from the race table, and attach their qualifying position if it exists. 
            
            merged["Season"] = year
            merged["Round"] = round_num
            merged["EventName"] = race_session.event["EventName"] 
            
            all_driver_entires.append(merged)
        except Exception as e:
            print(f"Done or skipped round {round_num} for {year}: {e}")
            break

# Combine all collected race DataFrames into a single master table
final_df = pd.concat(all_driver_entires, ignore_index=True)

# Export to CSV for model training 
final_df.to_csv("f1_race_winners_data.csv",index=False)
print("Data pipeline complete! Saved to f1_race_winners_data.csv")

# Quick sanity check on our saved data
df = pd.read_csv("f1_race_winners_data.csv")

# Pirnt dataset shape (rows,columns)
print("Dataset Shape:", df.shape)

#View the first few rows
print(df.head())

print(df[["Season", "Round","FullName", "Position"]].head(10))
df = pd.read_csv("f1_race_winners_data.csv")
print("TOTAL DATASET SHAPE:", df.shape)

