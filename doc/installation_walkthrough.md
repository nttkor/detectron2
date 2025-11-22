# Detectron2 Installation Walkthrough

## Goal
Install Detectron2 on Windows with GPU support, resolving compilation issues due to missing build tools and environment configuration.

## Changes
- Created a virtual environment `d2env`.
- Installed PyTorch 2.5.1 with CUDA 12.1 support.
- Installed `ninja` to speed up compilation.
- Configured Visual Studio Build Tools environment (`vcvars64.bat`) to compile Detectron2 C++ extensions.
- Installed Detectron2 from the local repository clone.

## Verification Results
### Automated Verification
Ran `verify_install.py`:
```
torch: 2.5.1+cu121
detectron2: 0.6
Detectron2 imported successfully!
```

### Manual Verification
You can now run the `Detectron2_Tutorial.ipynb` notebook.
**Important:** Make sure to select the `d2env` kernel in your notebook interface.

## Troubleshooting Notes
If you encounter build errors in the future:
1.  Ensure **Visual Studio Build Tools** (Desktop development with C++) is installed.
2.  Ensure `ninja` is installed (`pip install ninja`).
3.  If using `pip install`, you might need to set environment variables:
    ```powershell
    $vs_script = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    cmd /c "call `"$vs_script`" && pip install -e . --no-build-isolation"
    ```

## 문제 해결 요약 (Korean Summary)
이번 설치 과정에서 발생했던 주요 문제와 해결 방법입니다.

1.  **OS 차이 (Windows vs Linux)**
    *   **문제:** Detectron2는 기본적으로 Linux 환경에 최적화되어 있습니다. Windows에서는 C++ 코드를 직접 컴파일해야 하는데, Linux와 달리 컴파일러가 기본으로 제공되지 않아 설치가 실패했습니다.
    *   **해결:** Windows용 C++ 컴파일러인 **Visual Studio Build Tools**를 사용하여 컴파일 환경을 구축했습니다.

2.  **컴파일러 부재 (Missing `cl.exe`)**
    *   **문제:** `pip install` 명령어가 실행될 때, C++ 코드를 빌드할 도구(`cl.exe`)를 찾지 못해 에러가 발생했습니다.
    *   **해결:** Visual Studio의 환경 설정 스크립트(`vcvars64.bat`)를 찾아 실행함으로써, 설치 과정에서 컴파일러를 사용할 수 있도록 했습니다.

3.  **빌드 격리 문제 (Build Isolation)**
    *   **문제:** `pip`는 기본적으로 깨끗한 환경에서 설치를 시도하는데, 이 과정에서 우리가 미리 설정한 컴파일러 환경 변수나 PyTorch를 인식하지 못했습니다.
    *   **해결:** `--no-build-isolation` 옵션을 사용하여, 현재 설정된 환경(컴파일러 및 PyTorch 포함)을 그대로 사용하여 설치하도록 강제했습니다.
