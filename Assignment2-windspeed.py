# IMPORTS
from influxdb import InfluxDBClient
import pandas as pd
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

# Query wind speed data
query_wind = f"""
SELECT Speed FROM MetForecasts
WHERE time >= '{start_date}' AND time <= '{end_date}'
"""
wind_set = client.query(query_wind)

# Query power generation data
query_power = f"""
SELECT Total FROM Generation
WHERE time >= '{start_date}' AND time <= '{end_date}'
"""
power_set = client.query(query_power)

# Convert wind speed query result to DataFrame
values_wind = wind_set.raw["series"][0]["values"]
columns_wind = wind_set.raw["series"][0]["columns"]
wind_df = pd.DataFrame(values_wind, columns=columns_wind).set_index("time")
wind_df.index = pd.to_datetime(wind_df.index)  
wind_df = wind_df.resample("D").mean()  # Resample to daily mean
wind_df['Speed'] = wind_df['Speed'].interpolate()  # Interpolate missing values

# 查看时间列的类型和示例数据
print(wind_df.index)
print(type(wind_df.index))

# Convert power generation query result to DataFrame
values_power = power_set.raw["series"][0]["values"]
columns_power = power_set.raw["series"][0]["columns"]
power_df = pd.DataFrame(values_power, columns=columns_power).set_index("time")
power_df.index = pd.to_datetime(power_df.index)  # Convert time column to datetime
power_df = power_df.resample("D").mean()  # Resample to daily mean
power_df['Total'] = power_df['Total'].interpolate()  # Interpolate missing values

# Plot wind speed and power generation
plt.figure(figsize=(15, 6))

# Plot wind speed
plt.plot(wind_df.index, wind_df['Speed'], label="Wind Speed (m/s)", color="blue")

# Plot power generation
plt.plot(power_df.index, power_df['Total'], label="Power Generation (MW)", color="green")

# Format the x-axis to show months
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))  # Show one tick per month
plt.xticks(
    pd.date_range(wind_df.index.min(), wind_df.index.max(), freq='MS'),
    labels=[d.strftime('%b') for d in pd.date_range(wind_df.index.min(), wind_df.index.max(), freq='MS')],
    rotation=45
)

# Add labels, title, legend, and grid
plt.title("Wind Speed and Power Generation in 2023", fontsize=16)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save and show the plot
plt.savefig("wind_speed_and_power_generation_2023.png")
plt.show()