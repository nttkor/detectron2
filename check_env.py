
python -c "import torch; print(f'PyTorch 버전: {torch.__version__}'); 
print(f'CUDA 사용 가능: {torch.cuda.is_available()}'); 
print(f'CUDA 버전 (PyTorch 빌드): {torch.version.cuda if hasattr(torch.version, \"cuda\") else \"N/A\"}'); 
print(f'CUDA 디바이스: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"