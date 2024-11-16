import pandas as pd
import numpy as np
from influxdb import InfluxDBClient
from sklearn.preprocessing import StandardScaler
import mlflow.pyfunc
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.dates import DateFormatter

# Database connection settings
settings = {
    "host": "influxus.itu.dk",
    "port": 8086,
    "username": "lsda",
    "password": "icanonlyread",
}

client = InfluxDBClient(
    host=settings["host"], port=settings["port"], username=settings["username"], password=settings["password"]
)
client.switch_database("orkney")


# Convert InfluxDB query result to DataFrame
def set_to_dataframe(resulting_set):
    values = resulting_set.raw["series"][0]["values"]
    columns = resulting_set.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns=columns).set_index("time")
    df.index = pd.to_datetime(df.index)
    return df


# Convert wind direction to sine and cosine
def encode_wind_direction(df):
    direction_to_radians = {
        "N": 0,
        "NNE": np.pi / 8,
        "NE": np.pi / 4,
        "ENE": 3 * np.pi / 8,
        "E": np.pi / 2,
        "ESE": 5 * np.pi / 8,
        "SE": 3 * np.pi / 4,
        "SSE": 7 * np.pi / 8,
        "S": np.pi,
        "SSW": 9 * np.pi / 8,
        "SW": 5 * np.pi / 4,
        "WSW": 11 * np.pi / 8,
        "W": 3 * np.pi / 2,
        "WNW": 13 * np.pi / 8,
        "NW": 7 * np.pi / 4,
        "NNW": 15 * np.pi / 8,
    }

    df["Direction_radians"] = df["Direction"].map(direction_to_radians)
    df["Direction_sin"] = np.sin(df["Direction_radians"])
    df["Direction_cos"] = np.cos(df["Direction_radians"])
    return df


# Fetch and preprocess data
def fetch_data(days=180):
    power_set = client.query(f"SELECT * FROM Generation WHERE time > now()-{days}d")
    wind_set = client.query(
        f"SELECT * FROM MetForecasts WHERE time > now()-{days}d AND Lead_hours = '1'"
    )
    power_df = set_to_dataframe(power_set)
    wind_df = set_to_dataframe(wind_set)

    # Resample wind data to 1-hour frequency
    wind_df = wind_df.resample("h").ffill()
    numeric_columns = wind_df.select_dtypes(include=[np.number]).columns
    wind_df[numeric_columns] = wind_df[numeric_columns].interpolate()

    # Resample power data to 1-hour frequency
    power_df = power_df.resample("h").ffill()

    # Join data
    df = power_df.join(wind_df, how="inner")
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Encode wind direction
    df = encode_wind_direction(df)
    df = df[["Total", "Speed", "Direction_sin", "Direction_cos"]]

    return df


# Predict using the best model from MLflow
def predict_with_registered_model(df, model_name, model_version, steps=24):
    # Load the registered model
    best_model_uri = f"models:/{model_name}/{model_version}"
    best_model = mlflow.pyfunc.load_model(best_model_uri)

    # Prepare the recent data
    recent_data = df.iloc[-24:][["Speed", "Direction_sin", "Direction_cos"]]

    predictions = []

    for _ in range(steps):  # Predict 24 hours
        prediction = best_model.predict(recent_data)
        predictions.append(prediction[0])

        # Update recent_data with the prediction
        next_row = recent_data.iloc[-1:].copy()
        next_row["Total"] = prediction[0]
        recent_data = pd.concat([recent_data.iloc[1:], next_row])

    return predictions


# Plot predictions
def plot_predictions(predictions, start_date, file_name="24_hour_predictions.png"):
    dates = pd.date_range(start=start_date, periods=len(predictions), freq="H")

    plt.figure(figsize=(15, 7))
    plt.plot(dates, predictions, label="Predicted Power (MW)", linestyle="dashed")

    # Configure the x-axis as dates
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Generated Power (MW)", fontsize=14)
    plt.title("24-Hour Power Generation Prediction", fontsize=16)
    
    # Configure x-axis ticks for better visibility
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))  # Format as "MM-DD HH:MM"

    plt.gcf().autofmt_xdate()  # Automatically adjust date label angles
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name, format="png")
    plt.show()


def main():
    # Start date is the current date
    start_date = datetime.now()

    # Fetch and preprocess data
    df = fetch_data(days=30)

    # Model details
    MODEL_NAME = "LinearRegression"
    MODEL_VERSION = 1  # Replace with your model version

    # Predict future power generation
    predictions = predict_with_registered_model(df, MODEL_NAME, MODEL_VERSION, steps=24)

    # Save and plot predictions
    predictions_df = pd.DataFrame({"DateTime": pd.date_range(start=start_date, periods=24, freq="H"),
                                    "Prediction": predictions})
    predictions_df.to_csv("24_hour_predictions.csv", index=False)
    print("Predictions saved to '24_hour_predictions.csv'.")

    plot_predictions(predictions, start_date, file_name="24_hour_predictions.png")


if __name__ == "__main__":
    main()