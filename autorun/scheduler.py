import os
import subprocess
import logging
from crontab import CronTab

# 设置日志路径
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_dir, "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, "integrated_job.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
)

# 配置常量
DOCKER_COMPOSE_FILE = os.path.join(current_dir, "docker-compose.yml")
MODEL_DIR = os.path.join(current_dir, "saved_model")
MINIO_ALIAS = "minio"
MINIO_BUCKET = "mlflow"

# 确保环境变量
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"


def start_docker_compose():
    """启动 Docker Compose 服务"""
    try:
        logging.info("Starting Docker Compose services...")
        subprocess.run(
            ["docker-compose", "-f", DOCKER_COMPOSE_FILE, "up", "-d"], check=True
        )
        logging.info("Docker Compose services started successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to start Docker Compose services: {e.stderr}")
        raise


def stop_docker_compose():
    """停止 Docker Compose 服务"""
    try:
        logging.info("Stopping Docker Compose services...")
        subprocess.run(
            ["docker-compose", "-f", DOCKER_COMPOSE_FILE, "down"], check=True
        )
        logging.info("Docker Compose services stopped successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to stop Docker Compose services: {e.stderr}")
        raise


def ensure_minio_bucket_exists():
    """确保 MinIO 存储桶存在"""
    try:
        logging.info(f"Checking if bucket '{MINIO_BUCKET}' exists in MinIO...")

        # 使用 mc ls 检查存储桶是否存在
        result = subprocess.run(
            ["mc", "ls", f"{MINIO_ALIAS}/{MINIO_BUCKET}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # 如果返回码非 0，说明存储桶不存在
        if result.returncode != 0:
            logging.info(f"Bucket '{MINIO_BUCKET}' does not exist. Creating it...")
            subprocess.run(["mc", "mb", f"{MINIO_ALIAS}/{MINIO_BUCKET}"], check=True)
            logging.info(f"Bucket '{MINIO_BUCKET}' created successfully.")
        else:
            logging.info(f"Bucket '{MINIO_BUCKET}' already exists.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to ensure bucket exists: {e.stderr}")
        raise


def upload_model_to_minio():
    """上传模型到 MinIO"""
    try:
        logging.info("Uploading model to MinIO...")
        if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
            subprocess.run(
                ["mc", "cp", "-r", MODEL_DIR, f"{MINIO_ALIAS}/{MINIO_BUCKET}"],
                check=True,
            )
            logging.info("Model uploaded to MinIO successfully.")
        else:
            logging.error("Model directory is empty or does not exist.")
            raise FileNotFoundError(
                f"Model directory {MODEL_DIR} is empty or does not exist."
            )
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to upload model to MinIO: {e.stderr}")
        raise


def train_model():
    """运行模型训练脚本并记录到 MLflow"""
    try:
        logging.info("Starting model training...")
        train_script = os.path.join(current_dir, "Assignment2-train_model.py")
        subprocess.run(["python3", train_script], check=True)
        logging.info("Model training completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to train model: {e.stderr}")
        raise


def setup_cron_job():
    """设置 cron 定时任务"""
    try:
        cron = CronTab(user=True)

        # 脚本路径和日志路径
        script_path = os.path.abspath(__file__)
        log_path = os.path.join(log_dir, "cron.log")

        # 定义 cron 命令
        cron_command = f"/opt/anaconda3/envs/mlflow-tensorflow/bin/python {script_path} >> {log_path} 2>&1"

        # 检查是否已存在相同任务
        for job in cron:
            if job.command == cron_command:
                logging.info("Cron job already exists.")
                return

        # 创建新任务
        job = cron.new(command=cron_command)
        job.setall("0 0 * * 0")  # 每天午夜运行
        cron.write()

        logging.info("Cron job added successfully.")
    except Exception as e:
        logging.error(f"Failed to setup cron job: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        logging.info("Starting the integrated pipeline...")

        # 添加 cron 任务
        setup_cron_job()

        # 启动 Docker 服务
        start_docker_compose()

        # 确保 MinIO 存储桶存在
        ensure_minio_bucket_exists()

        # 运行模型训练
        train_model()

        # 上传模型到 MinIO
        upload_model_to_minio()

        logging.info("Pipeline completed successfully.")
        
         # 添加一个空行到日志文件
        with open(os.path.join(log_dir, "integrated_job.log"), "a") as log_file:
            log_file.write("\n\n")
            
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        stop_docker_compose()
