# Detectron2 AI Coding Agent Instructions

## Project Overview

This workspace combines the **Detectron2 library** (Facebook AI Research's state-of-the-art computer vision library) with a custom research project exploring panoptic segmentation, depth estimation, and object pose estimation using various models (Detectron2, OneFormer, DPT).

**Key Directory Structure:**
- `detectron2_repo/` - Official Detectron2 library (PyTorch-based detection/segmentation framework)
- `scripts/` - Research scripts and inference demos (panoptic, semantic segmentation, custom applications)
- `ade20k_consistency/` - Validation dataset for ADE20K segmentation consistency
- `d2env/` - Local Python virtual environment with dependencies

## Architecture Essentials

### Core Design Patterns

**Registry Pattern (Critical):**
Detectron2 uses extensive registries for extensibility. Don't add functionality directly to core classes; use registries:
```python
# CORRECT: Use registry
from detectron2.modeling import BACKBONE_REGISTRY
@BACKBONE_REGISTRY.register()
class MyBackbone(...): ...

# WRONG: Hardcoding model selection
if model_name == "my_model": ...
```
Key registries: `BACKBONE_REGISTRY`, `META_ARCH_REGISTRY`, `ROI_HEADS_REGISTRY`, `PROPOSAL_GENERATOR_REGISTRY`, `META_ARCH_REGISTRY`.

**Config System:**
All models use `CfgNode` (YAML-based configuration) in `detectron2_repo/detectron2/config/`:
- Configs define model architecture, training hyperparameters, and dataset references
- Configs support versioning and backward compatibility via `compat.py`
- Never hardcode values that belong in config files
- Load configs: `from detectron2.config import get_cfg; cfg = get_cfg(); cfg.merge_from_file("path.yaml")`

**Model Zoo Integration:**
Models are loaded via model zoo (pretrained weights):
```python
from detectron2.model_zoo import model_zoo
checkpoint_url = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
```

### Component Boundaries

1. **Modeling** (`detectron2/modeling/`): Architecture definitions
   - `backbone/` - Feature extractors (ResNet, ViT, Swin, etc.)
   - `proposal_generator/` - RPN and similar
   - `roi_heads/` - Region-of-interest processing
   - `meta_arch/` - End-to-end models (GeneralizedRCNN, PanopticFPN, RetinaNet, SemanticSegmentor)

2. **Engine** (`detectron2/engine/`): Training and inference loops
   - `DefaultTrainer` - Standard training loop with hooks
   - `DefaultPredictor` - Single-image inference wrapper
   - Hooks for checkpointing, logging, learning rate scheduling

3. **Data** (`detectron2/data/`): Dataset and dataloader abstractions
   - `DatasetCatalog` - Registry for dataset metadata
   - `MetadataCatalog` - Stores class names, thing/stuff categories
   - `DatasetMapper` - Transforms raw data to model inputs

4. **Structures** (`detectron2/structures/`): Data containers
   - `Instances` - Detected objects (boxes, masks, keypoints)
   - `Boxes` - Bounding box operations
   - `ImageList` - Batch of images with different sizes

5. **Evaluation** (`detectron2/evaluation/`): Metrics (COCOEvaluator, PanopticEvaluator, etc.)

### Panoptic Segmentation Workflow (Key Application Here)

The project heavily uses panoptic segmentation (combining instance and semantic segmentation):

**Data Flow:**
```
Image → DefaultPredictor → (panoptic_seg: torch.Tensor, segments_info: List[dict])
  ↓
panoptic_seg: Integer tensor where each pixel value = segment_id
segments_info: [{"category_id": int, "isthing": bool, "area": int, ...}, ...]
  ↓
MetadataCatalog → Thing/Stuff class names
  ↓
Visualization (Thing: contours, Stuff: semi-transparent overlays)
```

**Key Classes:**
- Panoptic output comes from `meta_arch/panoptic_fpn.py` or unified architectures
- `PanopticFPN` combines instance head (Mask R-CNN) + semantic head
- `MetadataCatalog.get("coco")` provides `thing_classes`, `stuff_classes`, `thing_dataset_id_to_contiguous_id` mappings

## Critical Development Workflows

### Setup & Environment

**Windows-Specific Installation (This Environment):**
```cmd
# Requires Visual Studio Build Tools (Desktop development with C++)
# and ninja for fast builds
pip install ninja

# Set up compiler environment before pip install
$vs_script = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "call `"$vs_script`" && pip install -e detectron2_repo --no-build-isolation"
```

See `doc/installation_walkthrough.md` and `doc/windows_installation_guide.md` for detailed troubleshooting.

**Verification:**
```bash
python verify_install.py  # Checks torch and detectron2 versions
```

### Running Inference Scripts

Scripts in `scripts/` follow a standard pattern:
1. Load image(s) from filesystem
2. Initialize model via `DefaultPredictor(cfg)` or HuggingFace models (OneFormer)
3. Run inference in loop
4. Visualize with OpenCV

**Examples:**
- `Dtr2_CoCoPanoptic_f.py` - Detectron2 panoptic on COCO dataset (80 thing + 53 stuff classes)
- `OneFormerTyny_f.py` - OneFormer panoptic on ADE20K (150 classes)
- `Dtr2_Tutorial_panoptic.ipynb` - Jupyter walkthrough

**Interactive Navigation:** A/D keys or arrow keys to move between images, Q to quit.

### Model Comparison Pattern

This workspace compares multiple architectures. When working on new scripts:
- Follow naming convention: `{ModelName}_{dataset}_{descriptor}.py`
- Use metadata catalog to fetch class mappings: `MetadataCatalog.get(dataset_name)`
- Implement thing/stuff distinction consistently (Detectron2 uses `isthing` field; ADE20K has separate mapping files)
- Store inference time and output statistics for benchmarking

### Running Tests & Validation

```bash
# From detectron2_repo/
python -m pytest tests/  # Full test suite (slow on Windows without GPU)

# Validate ADE20K consistency
python ade20k_consistency/ade20k_starter.ipynb  # Jupyter validation notebook
```

## Project-Specific Patterns & Conventions

### Thing vs. Stuff Handling

**Detectron2 (COCO):** Segments include `"isthing"` boolean
```python
segments_info = [
    {"category_id": 0, "isthing": True, ...},   # Thing (object)
    {"category_id": 183, "isthing": False, ...} # Stuff (background)
]
thing_classes = MetadataCatalog.get("coco").thing_classes
stuff_classes = MetadataCatalog.get("coco").stuff_classes
```

**ADE20K (OneFormer):** Uses external mapping file
```python
# From ade20k_thing_stuff_map.py or similar
is_thing_map = {...}  # Maps class_id → bool
id2label = {...}      # Maps class_id → class_name
```

When implementing new dataset support, establish this mapping upfront.

### Visualization Constants

Scripts use consistent styling for reproducibility:
- Stuff regions: Semi-transparent color overlay (alpha ~0.4-0.6)
- Thing regions: Contour outlines (thickness ~2-3px)
- Thing text: Yellow (`[0, 255, 255]` in BGR)
- Stuff text: White (`[255, 255, 255]` in BGR)
- Confidence score display: 2 decimal places

See `scripts/Dtr2_CoCoPanoptic_f.py` visualization functions as reference.

### Model Selection Pattern

Always follow this sequence:
1. Define model via config or model name string
2. Use model zoo: `model_zoo.get_checkpoint_url("config.yaml")`
3. Load detector: `DefaultPredictor(cfg)` or HF pipeline
4. Check device: `torch.cuda.is_available()` (scripts in workspace default to CUDA if available)

Avoid: Hardcoded model paths, manual weight loading without checkpointing mechanism.

### Cross-Model Comparison Notes

When comparing Detectron2, OneFormer, and other models:
- Output formats differ (Detectron2 returns torch.Tensor; OneFormer may return numpy)
- Class indices are dataset-specific (COCO: 0-80 things, ADE20K: 0-150 all classes)
- Inference time measurement should exclude I/O: use `time.time()` around model forward pass only
- For fair comparison, resize inputs consistently (this workspace resizes to 800px height)

## Integration Points & Dependencies

**External Libraries:**
- **PyTorch 2.5.1+ / Torchvision 0.22+** - Core deep learning
- **OpenCV** - Image I/O and visualization
- **HuggingFace Transformers** - OneFormer model loading
- **fvcore** - Detectron2's config and data utilities
- **omegaconf** - Hierarchical config management
- **Hydra** - Configuration framework (used by fvcore)
- **pycocotools** - COCO dataset and evaluation metrics

**Key imports to know:**
```python
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Instances, Boxes, BitMasks
from detectron2.model_zoo import model_zoo
from detectron2.evaluation import COCOEvaluator, PanopticEvaluator
```

## Common Pitfalls & Solutions

1. **Missing metadata:** Always call `MetadataCatalog.get(dataset_name)` before accessing class names
2. **Device mismatch:** Ensure tensors and models are on same device (`model.to(device)`, `tensor.to(device)`)
3. **Config freezing:** Configs are frozen by default; call `cfg.defrost()` before modifications
4. **Windows compilation:** Visual Studio Build Tools required; use `--no-build-isolation` flag
5. **CUDA/CPU fallback:** Models load pre-trained weights from model zoo; CPU inference is very slow for large models

## File Organization Guidance

- `scripts/` - One script per model/dataset combination (not shared utilities)
- `ade20k_consistency/` - Dataset validation, reference implementations
- `doc/` - Architecture notes, installation guides, implementation plans
- `.github/` - This file and GitHub-specific configs

When adding new work: Create appropriately named script in `scripts/` following existing naming convention.
