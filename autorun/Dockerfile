FROM python:3.8-slim

# 安装必要的依赖
RUN apt-get update && apt-get install -y \
    libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# 安装 MLflow 和其他必要的 Python 包
RUN pip install mlflow boto3

# 创建文件存储路径
RUN mkdir -p /mlflow-data

# 暴露 MLflow 服务端口
EXPOSE 5000

# 设置启动命令
ENTRYPOINT ["mlflow"]
CMD ["server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "file:/mlflow-data", "--serve-artifacts", "--artifacts-destination", "s3://mlflow"]