from influxdb import InfluxDBClient
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import mlflow
import joblib
import os

# Database connection settings
settings = {
    "host": "influxus.itu.dk",
    "port": 8086,
    "username": "lsda",
    "password": "icanonlyread",
}

client = InfluxDBClient(
    host=settings["host"],
    port=settings["port"],
    username=settings["username"],
    password=settings["password"],
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

    # Add time features
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek
    df["hour"] = df.index.hour

    return df


# Define regression models
def build_pipelines():
    numeric_features = ["Speed", "Direction_sin", "Direction_cos"]
    categorical_features = ["month", "day_of_week", "hour"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    xgboost_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", XGBRegressor(
                n_estimators=100,
                max_depth=10,
                objective="reg:squarederror",
            )),
        ]
    )
    
    linear_regression_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    random_forest_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                )),
        ]
    )

    pipelines = {
        "XGBoost": xgboost_pipeline,
        "LinearRegression": linear_regression_pipeline,
        "RandomForest": random_forest_pipeline,
    }

    return pipelines


# Train and evaluate all models
def train_and_evaluate_all(pipelines, X, y):
    results = {}
    tscv = TimeSeriesSplit(n_splits=5)

    predictions_dict = {}
    true_values_dict = {}  # Use dictionary to store true values by timestamp

    for model_name, pipeline in pipelines.items():
        print(f"Training model: {model_name}")
        mae_scores, mse_scores, r2_scores = [], [], []
        model_predictions = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            # Store predictions for this fold
            model_predictions.append(pd.Series(predictions, index=y_test.index))
            
            # Store true values using timestamp as key
            for idx, value in y_test.items():
                true_values_dict[idx] = value

            # Calculate metrics
            mae_scores.append(mean_absolute_error(y_test, predictions))
            mse_scores.append(mean_squared_error(y_test, predictions))
            r2_scores.append(r2_score(y_test, predictions))

        predictions_dict[model_name] = pd.concat(model_predictions)

        results[model_name] = {
            "MAE": np.mean(mae_scores),
            "MSE": np.mean(mse_scores),
            "R2": np.mean(r2_scores),
        }
        print(f"{model_name}: MAE={results[model_name]['MAE']}, MSE={results[model_name]['MSE']}, R2={results[model_name]['R2']}")

    # Convert true values dictionary to series
    true_values = pd.Series(true_values_dict)
    true_values.sort_index(inplace=True)  # Sort by timestamp
    
    # Modify plot_predictions function to add more debugging
    plot_predictions(true_values, predictions_dict)

    return results

def plot_predictions(true_values, predictions_dict):
    plt.figure(figsize=(15, 7))

    # Plot true values 
    plt.plot(true_values.index, true_values.values, 
            label="True Values", color="black", 
            linewidth=2, linestyle="dotted")

    # Plot predictions for each model
    for model_name, predictions in predictions_dict.items():
        print(f"Plotting {len(predictions)} predictions for {model_name}")
        print(f"Prediction range: {predictions.index.min()} to {predictions.index.max()}")
        plt.plot(predictions.index, predictions.values, 
                label=f"{model_name} Predictions")

    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Generated Power (MW)", fontsize=12)
    plt.title("Model Predictions vs True Values", fontsize=14)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("model_predictions.png")
    plt.show()

# Main function
def main():
    df = fetch_data()
    X = df[["Speed", "Direction_sin", "Direction_cos", "month", "day_of_week", "hour"]]
    y = df["Total"]

    # 定义管道
    pipelines = build_pipelines()

    # 超参数的配置
    xgboost_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "objective": "reg:squarederror",
    }
    random_forest_params = {
        "n_estimators": 100,
        "max_depth": 10,
    }

    with mlflow.start_run():
        # log data and model parameters
        mlflow.log_param("data_days", 180)
        mlflow.log_param("folds", 5)
        
        # log data preprocessing steps
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("encoder", "OneHotEncoder(handle_unknown='ignore')")
        mlflow.log_param("numeric_features", ["Speed", "Direction_sin", "Direction_cos"])
        mlflow.log_param("categorical_features", ["month", "day_of_week", "hour"])

        # log hyperparameters
        for param, value in xgboost_params.items():
            mlflow.log_param(f"xgboost_{param}", value)

        for param, value in random_forest_params.items():
            mlflow.log_param(f"random_forest_{param}", value)

      #train and evaluate all models
        results = train_and_evaluate_all(pipelines, X, y)

        for model_name, metrics in results.items():
            mlflow.log_metrics(
                {
                    f"{model_name}_MAE": metrics["MAE"],
                    f"{model_name}_MSE": metrics["MSE"],
                    f"{model_name}_R2": metrics["R2"],
                }
            )

        # Save best model
        best_model_name = min(results, key=lambda model: results[model]["MAE"])
        best_pipeline = pipelines[best_model_name]

        output_dir = "saved_model"
        os.makedirs(output_dir, exist_ok=True)
        local_model_path = os.path.join(output_dir, f"{best_model_name}.pkl")
        joblib.dump(best_pipeline, local_model_path)

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="traditional_model",
            registered_model_name=f"{best_model_name}",
        )
        print(f"Best pipeline saved as '{best_model_name}'")


if __name__ == "__main__":
    main()