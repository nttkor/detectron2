# Windows 통합 설치 가이드 (SegFormer, Detectron2, YOLO)

이 가이드는 Windows 환경에서 발생할 수 있는 Detectron2 빌드 오류를 해결하고, 모든 라이브러리를 순서대로 설치할 수 있도록 수정된 버전입니다.

## 0. 사전 준비 (필수)
설치 전에 반드시 다음이 준비되어 있어야 합니다.
1.  **Visual Studio Build Tools** 설치 (Desktop development with C++ 워크로드 선택)
2.  **Git** 설치

## 1. 가상환경 생성 및 활성화 (PowerShell 기준)

```powershell
# 1. 가상환경 생성 (이름: d2env)
# Windows에서는 python3 대신 python을 주로 사용합니다.
# 시스템에 여러 버전이 있다면 py -3.10 처럼 버전을 명시하는 것이 좋습니다.
py -3.10 -m venv d2env

# 2. 가상환경 활성화
# Windows PowerShell 명령어입니다.
.\d2env\Scripts\Activate.ps1

# 성공 확인: 터미널 입력창 맨 앞에 (d2env)가 보이면 성공!
# 만약 보안 오류가 나면 다음 명령어를 먼저 실행하세요: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

## 2. 🛠️ 통합 라이브러리 설치

가상환경이 활성화된 상태에서 순서대로 실행하세요.

```powershell
# 0. pip 및 빌드 도구 업그레이드
python -m pip install --upgrade pip
# Ninja가 있으면: CPU 코어를 최대한 활용해서 설치 속도가 훨씬 빨라집니다.
pip install wheel ninja

# 1. PyTorch 2.7.1 설치 (CUDA 버전별 명령어)
# 사용자님의 CUDA 환경(12.1)에 맞는 명령어를 선택하세요.

# [추천] 사용자 CUDA 12.1 환경 (추정 명령어) 현재 노트북은 Cuda가 13.0이라 128로 변경해서 설치했음.
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.6
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# CUDA 12.8
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# 2. SegFormer 및 필수 라이브러리
pip install transformers pillow numpy matplotlib opencv-python

# 3. Detectron2 설치 (Windows 호환 수정 버전)
# 주의: Detectron2는 Windows에서 바로 pip install로 설치하기 어렵습니다.
# 아래 과정을 따라주세요.

# 3-1. Detectron2 소스 다운로드 (이미 있다면 생략 가능) D:\git\detectron2\detectron2_repo
git clone https://github.com/facebookresearch/detectron2.git detectron2_repo

# 3-2. 컴파일 및 설치 (복잡한 과정이므로 아래 스크립트를 그대로 복사해서 실행하세요)
# Visual Studio 환경 변수를 불러와서 설치를 진행합니다.
$vs_script = Get-ChildItem -Path "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter "vcvars64.bat" -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName

# Visual Studio 환경 변수를 불러와서 설치를 진행합니다.
cmd /c "call `"$vs_script`" && set DISTUTILS_USE_SDK=1 && set MSSdk=1 && pip install -e detectron2_repo --no-build-isolation"

# 4. YOLO 모델 사용을 위한 라이브러리
# Ultralytics는 PyTorch가 이미 설치되어 있으면 잘 설치됩니다.
pip install ultralytics
```

## 설치 확인
터미널에서는
python -c "import torch; import detectron2; print(f'Torch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Detectron2: {detectron2.__version__}')"
```text
## 출력
잘된것 같아
Torch: 2.7.1+cu128
CUDA available: True
Detectron2: 0.6
```
설치가 모두 끝났다면, 다음 파이썬 코드로 확인해 보세요.

```python
import torch
import detectron2
import ultralytics
print(f"Torch: {torch.__version__}")
print(f"Detectron2: {detectron2.__version__}")
print(f"YOLO (Ultralytics): {ultralytics.__version__}")
```

## 💡 서버 버전 확인 및 맞추기 (중요)

SSH로 접속한 서버의 Detectron2 버전과 로컬 버전을 맞추는 것이 좋습니다.

### 1. 서버 버전 확인
서버 터미널에서 다음 명령어를 입력하세요.

```bash
# 방법 1: 간단 확인
python -c "import detectron2; print(detectron2.__version__)"

# 방법 2: 상세 확인 (CUDA 버전 등 포함)
python -m detectron2.utils.collect_env
```

### 2. 로컬 버전 맞추기
만약 서버 버전이 `0.6`이 아니라면, 로컬에서 `git clone`한 폴더로 이동해서 버전을 변경해야 합니다.

```powershell
cd detectron2_repo
# 예: 서버가 v0.5라면
git checkout v0.5

# 변경 후 다시 설치 (재설치)
cd ..
$vs_script = Get-ChildItem -Path "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter "vcvars64.bat" -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
cmd /c "call `"$vs_script`" && set DISTUTILS_USE_SDK=1 && set MSSdk=1 && pip install -e detectron2_repo --no-build-isolation"
```


1. requirements.txt (전통적 방식)
torch==2.7.1
detectron2==0.6
패키지 목록만 나열
문제점: Windows 컴파일 환경 설정, Visual Studio 찾기 등을 전혀 처리 못함
단순 의존성 관리용

2. 우리가 만든 가이드 (Shell Script 방식)
가상환경 생성부터 컴파일러 설정까지 전체 설치 프로세스를 문서화
Windows 특유의 문제(vcvars64.bat, 빌드 도구 등)까지 해결
장점: 복잡한 환경도 재현 가능

3. uv (최신 Python 패키지 매니저)
0.**Visual Studio Build Tools** 설치는 직접해야함
pip보다 10~100배 빠름 (Rust로 작성)
의존성 해결이 더 정확
하지만: 컴파일러 설정 같은 OS 레벨 문제는 여전히 수동 처리 필요

4. Docker
가장 깔끔한 해결책!
Linux 컨테이너를 쓰면 Windows 컴파일 문제 자체가 사라짐
Detectron2 공식 Docker 이미지 존재
단점: GPU 사용 시 NVIDIA Container Toolkit 설정 필요
