# Environment Setup Plan for Detectron2

## Goal
Set up a Python virtual environment (`venv`) and install the necessary dependencies to run the `Detectron2_Tutorial.ipynb` notebook.

## User Review Required
> [!NOTE]
> **PyTorch & venv**: While `venv` is not strictly required for PyTorch, it is **highly recommended** to prevent version conflicts with other projects. I will proceed with creating a `venv`.
> **GPU Support**: I will attempt to detect CUDA availability. If not found, I will install the CPU version of PyTorch, but Detectron2 performance will be significantly slower.

## Proposed Changes

### Environment Creation
1.  Create a virtual environment named `.venv` in `d:\git\detectron2`.
2.  Activate the environment.
3.  Upgrade `pip`.

### Dependency Installation
1.  **PyTorch**: Install PyTorch, torchvision, and torchaudio. (Will check for CUDA 11.8/12.1 or default to CPU).
2.  **Detectron2 Dependencies**:
    - `pyyaml==5.1`
    - `opencv-python`
    - `matplotlib`
    - `pycocotools`
    - `cloudpickle`
    - `tqdm`
    - `tensorboard`
    - `fvcore`
    - `iopath`
    - `omegaconf`
    - `hydra-core`
    - `black`
    - `packaging`
3.  **Detectron2**: Install from the local clone or via pip if the local clone is intended to be used as the source. The notebook suggests cloning, but the user is already in `d:\git\detectron2`, so I assume this *is* the clone. I will install it in editable mode (`pip install -e .`).

## Verification Plan

### Automated Tests
- Run a simple Python script to import `detectron2` and `torch` and print their versions.
- `python -c "import torch; import detectron2; print(torch.__version__, detectron2.__version__)"`

### Manual Verification
- The user can open `Detectron2_Tutorial.ipynb` and run the cells.
