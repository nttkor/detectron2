# Detectron2 자동 설치 스크립트 (Windows)
# 사용법: .\doc\install_detectron2.ps1

Write-Host "=" -ForegroundColor Cyan
Write-Host "Detectron2 자동 설치 시작" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan

# 1. 기존 가상환경 삭제 (있다면)
if (Test-Path "d2env") {
    Write-Host "[1/7] 기존 가상환경 삭제 중..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force d2env -ErrorAction SilentlyContinue
    Write-Host "✓ 삭제 완료" -ForegroundColor Green
}

# 2. 가상환경 생성 (Python 3.10)
Write-Host "[2/7] 가상환경 생성 중 (Python 3.10)..." -ForegroundColor Yellow
py -3.10 -m venv d2env
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Python 3.10을 찾을 수 없습니다. 먼저 설치해주세요." -ForegroundColor Red
    exit 1
}
Write-Host "✓ 가상환경 생성 완료" -ForegroundColor Green

# 3. 가상환경 활성화 및 pip 업그레이드
Write-Host "[3/7] pip 및 빌드 도구 설치 중..." -ForegroundColor Yellow
& ".\d2env\Scripts\python.exe" -m pip install --upgrade pip --quiet
& ".\d2env\Scripts\pip.exe" install wheel ninja --quiet
Write-Host "✓ 빌드 도구 설치 완료" -ForegroundColor Green

# 4. PyTorch 설치
Write-Host "[4/7] PyTorch 2.7.1 (CUDA 12.8) 설치 중..." -ForegroundColor Yellow
& ".\d2env\Scripts\pip.exe" install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 --quiet
Write-Host "✓ PyTorch 설치 완료" -ForegroundColor Green

# 5. SegFormer 및 필수 라이브러리 설치
Write-Host "[5/7] SegFormer 및 기타 라이브러리 설치 중..." -ForegroundColor Yellow
& ".\d2env\Scripts\pip.exe" install transformers pillow numpy matplotlib opencv-python --quiet
Write-Host "✓ 라이브러리 설치 완료" -ForegroundColor Green

# 6. Detectron2 컴파일 및 설치
Write-Host "[6/7] Detectron2 컴파일 및 설치 중 (시간이 걸릴 수 있습니다)..." -ForegroundColor Yellow

# Visual Studio 환경 변수 찾기
$vs_script = Get-ChildItem -Path "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter "vcvars64.bat" -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName

if (-not $vs_script) {
    Write-Host "✗ Visual Studio Build Tools를 찾을 수 없습니다." -ForegroundColor Red
    Write-Host "  Desktop development with C++ 워크로드를 포함한 Visual Studio를 설치해주세요." -ForegroundColor Red
    exit 1
}

# Detectron2 소스 다운로드 (없다면)
if (-not (Test-Path "detectron2_repo")) {
    Write-Host "  - Detectron2 소스 다운로드 중..." -ForegroundColor Gray
    git clone https://github.com/facebookresearch/detectron2.git detectron2_repo --quiet
}

# Detectron2 컴파일 설치
cmd /c "call `"$vs_script`" && set DISTUTILS_USE_SDK=1 && set MSSdk=1 && `"d2env\Scripts\pip.exe`" install -e detectron2_repo --no-build-isolation"

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Detectron2 설치 실패" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Detectron2 설치 완료" -ForegroundColor Green

# 7. 설치 확인
Write-Host "[7/7] 설치 확인 중..." -ForegroundColor Yellow
$verification = & ".\d2env\Scripts\python.exe" -c "import torch; import detectron2; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Detectron2: {detectron2.__version__}')"
Write-Host $verification -ForegroundColor Cyan

Write-Host ""
Write-Host "=" -ForegroundColor Green
Write-Host "설치 완료!" -ForegroundColor Green
Write-Host "=" -ForegroundColor Green
Write-Host ""
Write-Host "가상환경 활성화 방법:" -ForegroundColor Yellow
Write-Host "  .\d2env\Scripts\Activate.ps1" -ForegroundColor White
