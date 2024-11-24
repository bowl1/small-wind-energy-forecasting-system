# 使用 Python 官方镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /windpowerModel

# 复制代码到镜像中
COPY . .

# 安装依赖
RUN pip install -r requirements.txt

# 安装 MLflow
RUN pip install mlflow

# 暴露端口（5000 是 MLflow server 的默认端口）
EXPOSE 5000

# 运行 MLflow server
CMD ["mlflow", "models", "serve", "-m", "/windpowerModel/model", "-h", "0.0.0.0", "-p", "5000"]