# 베이스 이미지로 공식 Python 이미지 사용
FROM python:3.10-slim

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    curl \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# JupyterLab 및 Colab에서 자주 사용하는 패키지 설치
RUN pip install --upgrade pip
RUN pip install \
    jupyterlab \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    tensorflow \
    torch torchvision torchaudio \
    opencv-python \
    Pillow \
    tqdm \
    requests \
    h5py \
    pyyaml \
    tensorboard \
    plotly \
    notebook \
    ipywidgets

# JupyterLab 설정
RUN jupyter lab --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py

# 작업 디렉토리 설정
WORKDIR /content/drive/MyDrive/project

# 포트 설정
EXPOSE 8888

# JupyterLab 시작
CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
