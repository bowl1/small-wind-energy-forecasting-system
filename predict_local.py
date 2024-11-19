import mlflow
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from influxdb import InfluxDBClient
import tensorflow as tf

# 数据库连接配置
def setup_database():
    client = InfluxDBClient(
        host="influxus.itu.dk",
        port=8086,
        username="lsda",
        password="icanonlyread"
    )
    client.switch_database("orkney")
    return client

# 获取过去 90 天的数据
def get_past_data(client):
    query = "SELECT * FROM MetForecasts WHERE time > now() - 90d"
    result_set = client.query(query)

    # 将查询结果转换为 DataFrame
    values = result_set.raw["series"][0]["values"]
    columns = result_set.raw["series"][0]["columns"]
    past_data = pd.DataFrame(values, columns=columns).set_index("time")
    past_data.index = pd.to_datetime(past_data.index)

    # 添加特征列
    past_data["month"] = past_data.index.month
    past_data["day_of_week"] = past_data.index.dayofweek
    past_data["hour"] = past_data.index.hour

    return past_data

# 进行递归预测
def recursive_predict(model, past_data, steps):
    # 准备初始窗口
    feature_columns = ['Speed', 'Direction_sin', 'Direction_cos', 'month', 'day_of_week', 'hour']
    current_window = past_data[feature_columns].to_numpy().reshape(1, 1440, len(feature_columns))
    predictions = []

    for _ in range(steps):
        # 确保输入数据类型和形状符合模型的要求
        prediction = model(current_window.astype('float32')).numpy().flatten()[0]
        prediction = max(prediction, 0)  # 确保预测值不为负
        predictions.append(prediction)

        # 更新窗口，将预测值加入特征列中
        new_row = [prediction, *current_window[0, -1, 1:]]
        current_window = np.roll(current_window, -1, axis=1)
        current_window[0, -1] = new_row

    return predictions

if __name__ == "__main__":
    # 使用 MLflow 加载已注册的模型
    model_uri = "models:/Best_RNN_Model/3"  # 请根据你在 MLflow 注册的模型信息来更新
    loaded_model = mlflow.tensorflow.load_model(model_uri)

    print("Model loaded successfully.")

    # 设置数据库连接
    client = setup_database()

    # 示例输入数据
    input_example = {
        'Time': ['2024-12-18T10:00:00']
    }
    input_time_str = input_example['Time'][0]

    # 明确指定时间格式
    input_time = pd.to_datetime(input_time_str, format='%Y-%m-%dT%H:%M:%S')

    # 获取最近的窗口数据（实际的过去 90 天数据）
    past_data = get_past_data(client)

    # 计算目标时间距离当前时间的小时数
    current_time = past_data.index[-1] + timedelta(hours=1)
    steps = int((input_time - current_time).total_seconds() / 3600)
    if steps < 0 or steps > 720:
        raise ValueError("The date must be within the next 30 days.")

    # 进行递归预测直到目标时间
    predictions = recursive_predict(loaded_model, past_data, steps)
    print(pd.DataFrame([predictions[-1]], columns=["Predicted Power"]))