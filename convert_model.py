import mlflow.tensorflow
from tensorflow.keras.models import save_model
import mlflow.keras

# 加载现有模型
model_uri = "models:/Best_RNN_Model/2"  # 替换为你的模型名称和版本号
model = mlflow.tensorflow.load_model(model_uri)

# 保存模型为 .h5 格式
model.save("converted_model.h5")
print("Model saved as HDF5 (.h5) format successfully.")

# 启动一个新的 MLflow 运行并注册模型
with mlflow.start_run() as run:
    # 使用 MLflow 注册 .h5 格式的模型
    mlflow.keras.log_model(
        model="converted_model.h5",
        artifact_path="RNN_model_h5",
        registered_model_name="Best_RNN_Model_H5"  # 可选：设置新的注册模型名称
    )
    print("HDF5 format model registered in MLflow.")