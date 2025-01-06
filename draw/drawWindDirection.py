# IMPORTS
from influxdb import InfluxDBClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Database connection settings
settings = {
    "host": "influxus.itu.dk",
    "port": 8086,
    "username": "lsda",
    "password": "icanonlyread",
}

# Connect to the database
client = InfluxDBClient(
    host=settings["host"],
    port=settings["port"],
    username=settings["username"],
    password=settings["password"],
)
client.switch_database("orkney")

# Define the query for 2023 data
start_date = '2023-01-01T00:00:00Z'  # Start of 2023 (UTC time)
end_date = '2023-12-31T23:59:59Z'    # End of 2023 (UTC time)

# Query wind speed and direction data
query_wind = f"""
SELECT Speed, Direction FROM MetForecasts
WHERE time >= '{start_date}' AND time <= '{end_date}'
"""
wind_set = client.query(query_wind)

# Query power generation data
query_power = f"""
SELECT Total FROM Generation
WHERE time >= '{start_date}' AND time <= '{end_date}'
"""
power_set = client.query(query_power)

# Map wind direction to degrees
def map_direction_to_degrees(direction):
    direction_to_degrees = {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
        "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
        "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
        "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
    }
    return direction_to_degrees.get(direction, np.nan)

# Convert wind speed and direction query result to DataFrame
values_wind = wind_set.raw["series"][0]["values"]
columns_wind = wind_set.raw["series"][0]["columns"]
wind_df = pd.DataFrame(values_wind, columns=columns_wind).set_index("time")
wind_df.index = pd.to_datetime(wind_df.index)  # Convert time column to datetime

# Map wind direction to degrees
wind_df['Direction_degrees'] = wind_df['Direction'].apply(map_direction_to_degrees)

# Add sine and cosine for wind direction
wind_df["Direction_sin"] = np.sin(np.radians(wind_df["Direction_degrees"]))
wind_df["Direction_cos"] = np.cos(np.radians(wind_df["Direction_degrees"]))

# Resample to daily mean and interpolate missing values
wind_df = wind_df[["Speed", "Direction_sin", "Direction_cos", "Direction_degrees"]].resample("D").mean()
wind_df = wind_df.interpolate()

# Convert power generation query result to DataFrame
values_power = power_set.raw["series"][0]["values"]
columns_power = power_set.raw["series"][0]["columns"]
power_df = pd.DataFrame(values_power, columns=columns_power).set_index("time")
power_df.index = pd.to_datetime(power_df.index)  # Convert time column to datetime
power_df = power_df.resample("D").mean()  # Resample to daily mean
power_df['Total'] = power_df['Total'].interpolate()  # Interpolate missing values

# Merge dataframes on time index
merged_df = wind_df.join(power_df, how="inner")

# Check if 'Direction_degrees' exists
if 'Direction_degrees' not in merged_df.columns:
    raise ValueError("The 'Direction_degrees' column is missing after merging!")

# Group data by wind direction ranges
bins = np.arange(0, 361, 45)  # Divide into 45-degree bins
labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
merged_df['Direction_category'] = pd.cut(merged_df['Direction_degrees'], bins=bins, labels=labels, right=False)

# Calculate average power generation for each wind direction category
avg_power_by_direction = merged_df.groupby('Direction_category')['Total'].mean()

# Plot the results
plt.figure(figsize=(10, 6))
avg_power_by_direction.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Average Power Generation by Wind Direction in 2023", fontsize=16)
plt.xlabel("Wind Direction", fontsize=14)
plt.ylabel("Average Power Generation (MW)", fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save and show the plot
plt.savefig("wind direction and power generation in 2023.png")
plt.show()