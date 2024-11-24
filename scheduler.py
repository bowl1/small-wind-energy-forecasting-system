import os
import subprocess
import logging
from datetime import datetime

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
GIT_REPO_URL = "https://github.itu.dk/BDM-Autumn-2024/bowl_a2"  # 替换为你的 Git 仓库 URL
LOCAL_REPO_DIR = os.path.join(current_dir, "retraining_project")
MODEL_DIR = os.path.join(current_dir, "saved_model")
DOCKER_IMAGE_NAME = "windpower_model"
DOCKER_CONTAINER_NAME = "windpower_model_container"
MINIO_ALIAS = "minio"
MINIO_BUCKET = "models"
MINIO_URL = "http://localhost:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password"

def update_repository():
    """从 Git 仓库拉取最新代码"""
    try:
        if not os.path.exists(LOCAL_REPO_DIR):
            logging.info("Cloning repository...")
            subprocess.run(["git", "clone", GIT_REPO_URL, LOCAL_REPO_DIR], check=True)
        else:
            logging.info("Pulling latest changes from repository...")
            subprocess.run(["git", "-C", LOCAL_REPO_DIR, "pull"], check=True)
        logging.info("Repository updated successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to update repository: {str(e)}")
        raise

def train_model():
    """运行模型训练脚本"""
    try:
        logging.info("Starting model retraining...")
        train_script = os.path.join(LOCAL_REPO_DIR, "Assignment2-train_model.py")
        subprocess.run(
            f"nohup python3 {train_script} > train.log 2>&1 &",
            shell=True,  # 启用 shell 特性
            check=True   # 捕获错误
        )
        logging.info("Model retraining started successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to start model retraining: {str(e)}")
        raise

def save_model_to_minio():
    """将模型保存到 MinIO"""
    try:
        logging.info("Uploading model to MinIO...")
        subprocess.run(
            ["mc", "alias", "set", MINIO_ALIAS, MINIO_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY],
            check=True,
        )
        subprocess.run(["mc", "mb", f"{MINIO_ALIAS}/{MINIO_BUCKET}"], check=False)
        subprocess.run(["mc", "cp", "-r", MODEL_DIR, f"{MINIO_ALIAS}/{MINIO_BUCKET}"], check=True)
        logging.info("Model uploaded to MinIO successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to upload model to MinIO: {str(e)}")
        raise

def build_and_run_docker():
    """构建 Docker 镜像并运行服务"""
    try:
        # 停止旧容器
        logging.info("Stopping old Docker container...")
        subprocess.run(["docker", "stop", DOCKER_CONTAINER_NAME], check=False)
        subprocess.run(["docker", "rm", DOCKER_CONTAINER_NAME], check=False)

        # 构建新镜像
        logging.info("Building Docker image...")
        subprocess.run(
            ["docker", "build", "-t", DOCKER_IMAGE_NAME, LOCAL_REPO_DIR],
            check=True,
        )

        # 启动容器
        logging.info("Starting Docker container...")
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", DOCKER_CONTAINER_NAME,
                "-p", "5000:5000",
                DOCKER_IMAGE_NAME,
            ],
            check=True,
        )
        logging.info("Docker container started successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to build/run Docker container: {str(e)}")
        raise

def setup_cron_job():
    """设置定时任务"""
    cron_command = (
        f"0 0 * * * python3 {os.path.abspath(__file__)} > {os.path.join(log_dir, 'cron.log')} 2>&1"
    )
    try:
        logging.info("Setting up cron job...")
        subprocess.run(["crontab", "-l"], stdout=open("current_cron", "w"), check=True)
        with open("current_cron", "a") as cron_file:
            cron_file.write(f"{cron_command}\n")
        subprocess.run(["crontab", "current_cron"], check=True)
        os.remove("current_cron")
        logging.info("Cron job set up successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to set up cron job: {str(e)}")
        raise

def kill_previous_jobs():
    """杀死旧的模型服务进程"""
    try:
        logging.info("Killing previous jobs...")
        subprocess.run(["pkill", "-f", "mlflow"], check=False)
        logging.info("Previous jobs killed successfully.")
    except Exception as e:
        logging.error(f"Failed to kill previous jobs: {str(e)}")

if __name__ == "__main__":
    try:
        logging.info("Starting integrated job...")
        # 更新代码仓库
        update_repository()

        # 杀死旧的进程
        kill_previous_jobs()

        # 训练模型
        train_model()

        # 将模型保存到 MinIO
        save_model_to_minio()

        # 使用 Docker 启动服务
        build_and_run_docker()

        # 设置定时任务（仅首次需要）
        setup_cron_job()

        logging.info("Integrated job completed successfully.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")