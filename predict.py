import mlflow.pyfunc
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from influxdb import InfluxDBClient

# 自定义的 Python 模型类
class WindPowerPredictor(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        # 使用 MLflow 注册模型来加载 LSTM 模型
        model_uri = "models:/Best_RNN_Model/2"  # 请根据你在 MLflow 注册的模型信息来更新
        self.model = mlflow.pyfunc.load_model(model_uri)
        
        print("Model loaded successfully.")

        # 数据库连接配置
        self.client = InfluxDBClient(
            host="influxus.itu.dk",
            port=8086,
            username="lsda",
            password="icanonlyread"
        )
        self.client.switch_database("orkney")


    def predict(self, context, model_input):
        # 解析输入数据
        input_time_str = model_input['Time'][0]
        input_time = pd.to_datetime(input_time_str)
        
        # 获取最近的窗口数据（实际的过去 60 天数据）
        past_data = self.get_past_data()
        
        # 计算目标时间距离当前时间的小时数
        current_time = past_data.index[-1] + timedelta(hours=1)
        steps = int((input_time - current_time).total_seconds() / 3600)
        if steps < 0 or steps > 720:
            raise ValueError("The date must be within the next 30 days.")
        
        # 进行递归预测直到目标时间
        predictions = self.recursive_predict(past_data, steps)
        return pd.DataFrame([predictions[-1]], columns=["Predicted Power"])

    def get_past_data(self):
        # 从数据库中查询过去 60 天的数据，每小时一条数据，总共 1440 条
        query = "SELECT * FROM MetForecasts WHERE time > now() - 60d"
        result_set = self.client.query(query)
        
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

    def recursive_predict(self, past_data, steps):
        # 准备初始窗口
        feature_columns = ['Speed', 'Direction_sin', 'Direction_cos', 'month', 'day_of_week', 'hour']
        current_window = past_data[feature_columns].to_numpy().reshape(1, 1440, len(feature_columns))
        predictions = []
        
        for _ in range(steps):
            prediction = self.model.predict(current_window).flatten()[0]
            prediction = max(prediction, 0)  # 确保预测值不为负
            predictions.append(prediction)
            
            # 更新窗口，将预测值加入特征列中
            new_row = [prediction, *current_window[0, -1, 1:]]
            current_window = np.roll(current_window, -1, axis=1)
            current_window[0, -1] = new_row
        
        return predictions

# 保存自定义模型
if __name__ == "__main__":
    # 使用 mlflow.pyfunc 保存模型，将训练好的模型和自定义的预测逻辑结合起来
    mlflow.pyfunc.save_model(
        path="wind_power_model",
        python_model=WindPowerPredictor(),
        artifacts={"rnn_model": "saved_rnn_model.h5"},
    )
    print("Custom model saved as 'wind_power_model'")
