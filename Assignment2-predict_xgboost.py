import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from influxdb import InfluxDBClient

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


# 获取数据并处理为 DataFrame
def fetch_data(days=180):
    query = f"""
    SELECT * FROM "MetForecasts"
    WHERE time > now() - {days}d AND Lead_hours = '1'
    """
    result = client.query(query)
    df = pd.DataFrame(result.get_points())
    df['time'] = pd.to_datetime(df['time'])
    return df


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


# 获取未来30天的输入数据
def prepare_data_for_prediction():
    df = fetch_data(180)  # 获取180天数据
    df = encode_wind_direction(df)

    df["month"] = df["time"].dt.month
    df["day_of_week"] = df["time"].dt.dayofweek
    df["hour"] = df["time"].dt.hour

    # 特征列
    features = ["Speed", "Direction_sin", "Direction_cos", "month", "day_of_week", "hour"]
    
    # 数据预处理：标准化数值特征
    scaler = StandardScaler()
    df[["Speed", "Direction_sin", "Direction_cos"]] = scaler.fit_transform(df[["Speed", "Direction_sin", "Direction_cos"]])

    # 提取特征用于预测
    X_future = df[features]
    
    # 返回 scaler 和标准化后的特征
    return X_future, scaler, df


# 加载 MLflow 中的模型
def load_mlflow_model(model_uri='runs:/c3d32d6e60b54f2db8c6e59ecc22bf88/traditional_model'):
    # 使用 model_uri 加载模型
    model = mlflow.pyfunc.load_model(model_uri)
    return model


# 进行预测并绘图
def predict_and_plot():
    # 获取准备好的数据和标准化器
    X_future, scaler, df = prepare_data_for_prediction()
    model = load_mlflow_model()

    # 进行预测
    predictions_scaled = model.predict(X_future)

    # 逆标准化：恢复原始尺度
    predictions_original = predictions_scaled * scaler.scale_[0] + scaler.mean_[0]  # 恢复风速的原始值
    predictions_original = predictions_original[:30]  # 截取前 30 个预测值

    # 生成未来30天的日期
    future_dates = pd.date_range(start=pd.to_datetime("today"), periods=30, freq="D")

    plt.figure(figsize=(10, 6))
    plt.plot(future_dates, predictions_original, label="Predicted Wind Power", color='b')
    plt.xlabel("Date")
    plt.ylabel("Predicted Wind Power (MW)")
    plt.title("Wind Power Prediction for the Next 30 Days - XGBOOST")
    plt.xticks(rotation=45)  
    plt.legend()
    plt.tight_layout() 
    plt.savefig("wind_power_prediction -XGBOOST.png")
    plt.show()


# 调用预测和绘图的函数
predict_and_plot()