from influxdb import InfluxDBClient
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import mlflow
from matplotlib.dates import DateFormatter
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

    # 对数据进行风向编码
    df = encode_wind_direction(df)

    # 添加月份和星期几特征
    df["month"] = df.index.month  # 提取月份，值为 1 到 12
    df["day_of_week"] = df.index.dayofweek  # 提取星期几，值为 0（周一）到 6（周日）
    df["hour"] = df.index.hour

    # Check for NaN values
    if df.isnull().any().any():
        print("NaN values detected in the dataframe:")
        print(df.isnull().sum())

    return df


# Handle outliers in numeric data using interpolation
def handle_outliers(df, numeric_columns):
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = np.nan
        df[column] = df[column].interpolate(method="linear")
    return df


# Create windowed dataset for RNN with aligned indexing
def create_windowed_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size, 1:])  # 只使用特征列作为X（去掉目标值列）
        y.append(data[i + window_size, 0])  # 使用第1列作为目标值
    return np.array(X), np.array(y)


# Define regression models
def build_pipelines():
    # 定义需要的特征
    numeric_features = ["Speed", "Direction_sin", "Direction_cos"]
    categorical_features = ["month", "day_of_week", "hour"]

    # 数值特征标准化
    numeric_transformer = StandardScaler()

    # 类别特征OneHot编码
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # XGBoost 管道
    xgboost_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", XGBRegressor(objective="reg:squarederror")),
        ]
    )

    # 多项式回归管道
    polynomial_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "poly_features",
                PolynomialFeatures(degree=2),
            ),  
            ("regressor", LinearRegression()),  # 使用线性回归作为回归器
        ]
    )

    # 将两个管道存入字典
    pipelines = {
        "XGBoost": xgboost_pipeline,
        "PolynomialRegression_degree_2": polynomial_pipeline,
    }

    return pipelines


# Train and evaluate all models including RNN
def train_and_evaluate_all(pipelines, X, y, window_size=30 * 24):
    results = {}
    tscv = TimeSeriesSplit(n_splits=5)

    # Align data range: drop the first window_size data points
    aligned_X = X.iloc[window_size:]
    aligned_y = y.iloc[window_size:]

    predictions_dict = {}
    y_truth = pd.Series(dtype=float)  # Save true values

    # Shared time index
    shared_time_index = aligned_X.index

    # Evaluate regression models
    for model_name, pipeline in pipelines.items():
        print(f"Training model: {model_name}")
        mae_scores, mse_scores, r2_scores = [], [], []
        model_predictions = []

        for train_index, test_index in tscv.split(aligned_X):
            X_train, X_test = aligned_X.iloc[train_index], aligned_X.iloc[test_index]
            y_train, y_test = aligned_y.iloc[train_index], aligned_y.iloc[test_index]

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            predictions_series = pd.Series(predictions, index=y_test.index)

            mae_scores.append(mean_absolute_error(y_test, predictions_series))
            mse_scores.append(mean_squared_error(y_test, predictions_series))
            r2_scores.append(r2_score(y_test, predictions_series))

            model_predictions.append(predictions_series)
            
            if not y_truth.empty:
                y_truth = pd.concat([y_truth, y_test])
            else:
                y_truth = y_test

        predictions_dict[model_name] = pd.concat(model_predictions)

        results[model_name] = {
            "MAE": np.mean(mae_scores),
            "MSE": np.mean(mse_scores),
            "R2": np.mean(r2_scores),
        }
        print(
            f"{model_name}: MAE={results[model_name]['MAE']}, MSE={results[model_name]['MSE']}, R2={results[model_name]['R2']}"
        )

    # Prepare data for RNN
    combined_data = np.hstack(
        [y.values.reshape(-1, 1), X.to_numpy()]
    )  # Combine target and features
    min_vals = combined_data.min(axis=0)
    max_vals = combined_data.max(axis=0)
    scaled_data = (combined_data - min_vals) / (max_vals - min_vals)

    X_windowed, y_windowed = create_windowed_dataset(scaled_data, window_size)

    # Align RNN's time index to match the test period
    rnn_time_index = shared_time_index[-len(y_windowed) :]

    split_idx = int(0.8 * len(X_windowed))
    X_train_rnn, X_test_rnn = X_windowed[:split_idx], X_windowed[split_idx:]
    y_train_rnn, y_test_rnn = y_windowed[:split_idx], y_windowed[split_idx:]

    print(f"Combined data shape: {combined_data.shape}")  # 归一化前的总数据
    print(f"X_windowed shape: {X_windowed.shape}")  # 窗口化后的输入特征
    print(f"y_windowed shape: {y_windowed.shape}")  # 窗口化后的目标值

    # Build RNN model
    model = Sequential(
        [
            Input(shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_rnn, y_train_rnn, epochs=10, batch_size=32, verbose=1)

    predictions_rnn = model.predict(X_test_rnn).flatten()
    predictions_rnn = predictions_rnn * (max_vals[0] - min_vals[0]) + min_vals[0]
    y_test_rnn_rescaled = y_test_rnn * (max_vals[0] - min_vals[0]) + min_vals[0]

    # Align RNN's time index
    rnn_predictions_series = pd.Series(
        predictions_rnn, index=rnn_time_index[-len(predictions_rnn) :]
    )
    y_test_rnn_rescaled_series = pd.Series(
        y_test_rnn_rescaled, index=rnn_time_index[-len(y_test_rnn_rescaled) :]
    )

    mae = mean_absolute_error(y_test_rnn_rescaled_series, rnn_predictions_series)
    mse = mean_squared_error(y_test_rnn_rescaled_series, rnn_predictions_series)
    r2 = r2_score(y_test_rnn_rescaled_series, rnn_predictions_series)

    results["RNN"] = {"MAE": mae, "MSE": mse, "R2": r2}
    print(f"RNN: MAE={mae}, MSE={mse}, R2={r2}")

    # Plot all models predictions and truth values for the RNN prediction period
    plt.figure(figsize=(15, 7))

    # Align sklearn predictions to match the RNN prediction period
    start_index = rnn_predictions_series.index[0]
    aligned_predictions_dict = {
        model_name: predictions[predictions.index >= start_index]
        for model_name, predictions in predictions_dict.items()
    }

    # Plot predictions from linear regression and other models
    for model_name, predictions in aligned_predictions_dict.items():
        plt.plot(predictions.index, predictions, label=f"{model_name} Predictions")
        print(f"Model: {model_name}, Number of aligned predictions: {len(predictions)}")

    # Plot RNN and truth values for the RNN prediction period
    plt.plot(
        rnn_predictions_series.index,
        rnn_predictions_series,
        label="RNN Predictions",
        linestyle="dashed",
    )
    plt.plot(
        rnn_predictions_series.index,
        y_test_rnn_rescaled_series,
        label="Truth",
        color="black",
        linestyle="dotted",
        linewidth=2,
    )

    # 设置纵轴标签
    plt.ylabel("Generated Power (MW)", fontsize=14)  # 添加纵轴标题

    plt.xlabel("Time (Days)")
    plt.xticks(
        rnn_predictions_series.index[::168], rotation=45
    )  # Select every 1week index to show daily ticks
    plt.gca().xaxis.set_major_formatter(
        plt.FixedFormatter(rnn_predictions_series.index[::168].strftime("%Y-%m-%d"))
    )
    plt.title("Model Predictions vs Truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_model.png")
    plt.show()

    return results, model


# Main function
def main():

    df = fetch_data()
    df = encode_wind_direction(df)
    numeric_columns = ["Total", "Speed"]
    df = handle_outliers(df, numeric_columns)

    # 定义输入特征和目标
    X = df[["Speed", "Direction_sin", "Direction_cos", "month", "day_of_week", "hour"]]
    y = df["Total"]
    # 打印训练数据的形状
    print(f"Training data shape: {X.shape}")
    print(f"Features used in training: {X.columns.tolist()}")

    # Define pipelines for traditional models
    pipelines = build_pipelines()

    # 启动 MLflow 运行
    with mlflow.start_run() as run:
        # 记录基础参数
        mlflow.log_param("data_days", 180)
        mlflow.log_param("window_size", 30 * 24)

        # Train and evaluate all models (including RNN)
        results, model = train_and_evaluate_all(pipelines, X, y)

        # 记录每个模型的指标到 MLflow
        for model_name, metrics in results.items():
            mlflow.log_metrics(
                {
                    f"{model_name}_MAE": metrics["MAE"],
                    f"{model_name}_MSE": metrics["MSE"],
                    f"{model_name}_R2": metrics["R2"],
                }
            )

        # 选出最佳模型
        best_model_name = min(results, key=lambda model: results[model]["MAE"])
        best_model_metrics = results[best_model_name]
        print(f"Best model: {best_model_name} with MAE={best_model_metrics['MAE']}")

        # 记录最佳模型指标到 MLflow
        mlflow.log_metrics(
            {
                f"{best_model_name}_MAE": best_model_metrics["MAE"],
                f"{best_model_name}_MSE": best_model_metrics["MSE"],
                f"{best_model_name}_R2": best_model_metrics["R2"],
            }
        )

        # 保存最佳模型
        if best_model_name == "RNN":
            # 如果最佳模型是 RNN，使用 TensorFlow 保存并记录到 MLflow
            print(f"The RNN model is the best and already saved")

            # 保存模型到指定路径
            model_save_path = "saved_rnn_model"
            model.save(model_save_path, save_format="tf")  # 保存为 TensorFlow 格式

            # 使用 MLflow TensorFlow 保存模型
            mlflow.tensorflow.log_model(
                model=model,  # Keras 模型对象
                artifact_path="RNN_model",  # 保存路径
                registered_model_name="Best_RNN_Model",  # 可选：注册模型的名称
            )
        else:
            # 如果最佳模型是传统模型，保存完整管道
            best_pipeline = pipelines[best_model_name]
            
            output_dir = "/Users/libowen/SD3rd/bigdata/assign2/bowl/saved_model"
            local_model_path = os.path.join(output_dir, "best_pipeline.pkl")
            joblib.dump(best_pipeline, local_model_path)  # 保存到本地为 .pkl 文件
            
            mlflow.sklearn.log_model(
                sk_model=best_pipeline,  # 保存完整管道
                artifact_path="traditional_model",  # 保存路径
                registered_model_name=f"{best_model_name}",  # 注册模型的名称
            )
            print(f"Best pipeline saved and logged as '{best_model_name}'")


if __name__ == "__main__":
    main()

