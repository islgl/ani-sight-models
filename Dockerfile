# 使用阿里云容器镜像服务作为基础镜像
FROM registry.cn-hangzhou.aliyuncs.com/serverless_devs/pytorch:22.12-py3

# 设置工作目录
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0

# 将项目文件添加到工作目录
ADD . /app

# 安装项目依赖
RUN pip install --no-cache-dir \
    opencv-python \
    numpy \
    fastapi \
    tqdm \
    uvicorn \
    matplotlib \
    onnxruntime-gpu

# 暴露 FastAPI 使用的端口（默认为 8000）
EXPOSE 8000

# 设置启动命令，启动 FastAPI 服务器
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
