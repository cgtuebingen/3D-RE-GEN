# Installation & Setup

## Environment

**Python 3.10 with CUDA 12.1**

The project uses a mamba environment located at `venv_py310/` with:
- `torch 2.5.1+cu121`
- `torchvision 0.20.1+cu121`
- `torchaudio 2.5.1+cu121`
- `pytorch3d 0.7.9`
- `open3d 0.18.0`
- `numpy 1.26.4`

Run all commands using:
```bash
mamba run -p ./venv_py310 python run.py -p [STEPS]
```
### Setting Up Mamba Environment (if needed)

If you need to recreate the environment:

```bash
# Create conda environment with Python 3.10
mamba create -p ./venv_py310 python=3.10 -y
mamba activate ./venv_py310

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch3D
pip install fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Install all other dependencies
pip install -r requirements.txt

# Ensure numpy version compatibility
pip install "numpy<2.0"
```


## Setup

1. **Clone repository with submodules:**
   ```bash
   git clone --recursive https://github.com/cgtuebingen/3D-RE-GEN.git
   cd 3D-RE-GEN
   ```

   If you already cloned without submodules:
   ```bash
   git submodule update --init --recursive
   ```

   This initializes:
   - `Hunyuan3D-2/` - 2D-to-3D model generation
   - `vggt/` - Camera extraction
   - `dust3r/` - Alternative camera extraction (optional)

2. **Download SAM model:**
   ```bash
   cd segmentor
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   cd ..
   ```
   Location: `segmentor/sam_vit_h_4b8939.pth`

3. **HDR environment maps (optional):**
   Large HDR files are not stored in the repo. Download them from Poly Haven:
   - https://polyhaven.com/a/brown_photostudio_02

   Place the downloaded `.hdr` in:
   - `input_images/raw/`

4. **Configure input in `src/config.yaml`:**
   ```yaml
   input_image: ../input_images/scene/your_image.jpg
   GT_scene: null  # or path to ground truth scene for evaluation
   device_global: "cuda:0"
   ```

### Optional: API Keys

Some features require API keys:

- **Banana.dev** (for inpainting step 2):
  ```yaml
  use_banana: true
  # Requires BANANA_API_KEY environment variable or credentials file
  ```

- **Google Gemini** (for image generation):
  ```bash
  export GOOGLE_API_KEY="your_api_key_here"
  ```

## Running the Pipeline

Each step 1-9 corresponds to a pipeline stage:

```bash
# Run specific steps
mamba run -p ./venv_py310 python run.py -p 1          # Segmentation
mamba run -p ./venv_py310 python run.py -p 1 2 3      # Steps 1-3
mamba run -p ./venv_py310 python run.py -p 1 2 3 4 5 6 7 8 9  # Full pipeline
```

### Pipeline Steps
| Step | Description | Key Input | Key Output |
|------|-------------|-----------|------------|
| **1** | Segmentation (Grounded SAM) | Input image | Cropped objects, masks |
| **2** | Inpainting (nano-Banana) | Masks + image | Inpainted backgrounds |
| **3** | 2D-to-3D (Hunyuan3D-2) | Cropped images | 3D models (GLB) |
| **4** | Camera extraction (VGGT) | Full inpainted image | Camera params, point clouds |
| **5** | Point cloud & mask creation | 3D models + camera | Object point clouds |
| **6** | Scene optimization | Point clouds | Optimized geometry |
| **7** | Background & scene reconstruction | Optimization results | Complete scene GLB |
| **8** | Blender rendering | Scene GLB, config | Rendered images |
| **9** | Evaluation (MIDI metrics) | Scene + GT scene | Metrics report |

**Step 9 requires** `GT_scene` configured in `src/config.yaml`. Skip if no ground truth available.

## Configuration Details

Key settings in `src/config.yaml`:

**Device & Environment:**
```yaml
device: "cuda:0"              # Device for this script
device_global: "cuda:0"       # Global device for all scripts
use_all_available_cuda: false # true for multi-GPU
```

**Segmentation (Step 1):**
```yaml
labels:
   - chair
   - table
   - sofa
   - plant in pot
   - lamp
   - floor
threshold: 0.25        # Detection confidence (lower = more detections)
iou_threshold: 0.5     # NMS threshold
depth_large_model: true  # Marigold (true) or Depth-Anything2 (false)
```

**Camera Extraction (Step 4):**
```yaml
Use_VGGT: true  # Use VGGT (recommended)
                        # Set to false for Dust3R
```

**3D Generation:**
```yaml
use_hunyuan21: false  # Use Hunyuan3D-2.1 instead of 2.0
```

**Output Directories:**
```yaml
output: "../output"
temp: "../tmp"
```

| Step | Description |
|------|-------------|
| **1** | Segmentation (Grounded SAM) - extract objects |
| **2** | Inpainting (nano-Banana) - generate backgrounds |
| **3** | 2D-to-3D (Hunyuan3D-2) - create 3D models |
| **4** | Camera extraction (VGGT) - camera & point clouds |
| **5** | Point cloud & mask creation |
| **6** | Scene optimization (differential rendering) |
| **7** | Background & scene reconstruction |
| **8** | Blender rendering |
| **9** | Evaluation (MIDI metrics) - requires `GT_scene` |

## Folder Structure

```
3D-RE-GEN/
├── src/
│   ├── config.yaml                 # Main configuration
│   ├── segmentation/               # Step 1-2: SAM, inpainting
│   ├── 2d_to_3d_models/           # Step 3: Hunyuan3D
│   ├── camera_and_pointcloud/     # Step 4: VGGT/Dust3R
│   ├── scene_reconstruction/      # Step 5: Point cloud processing
│   ├── scene_optimization/        # Step 6: Diff rendering
│   ├── blender_rendering/         # Step 8: Blender export
│   ├── evaluation/                # Step 9: Metrics
│   └── utils/                     # Shared utilities
│
├── segmentor/
│   ├── sam_vit_h_4b8939.pth       # SAM model (download required)
│   └── requirements.txt
│
├── Hunyuan3D-2/                    # 2D-to-3D submodule
├── vggt/                           # Camera extraction submodule
├── dust3r/                         # Alternative camera submodule (optional)
│
├── venv_py310/                     # Mamba environment
├── input_images/                   # Input images
├── input_scenes/                   # Ground truth scenes
│
├── output/                         # Pipeline outputs
│   ├── 3D/                        # Generated 3D models
│   ├── findings/                  # Segmentation crops
│   ├── masks/                     # Binary masks
│   ├── pointclouds/               # Point clouds
│   ├── glb/                       # Scene GLBs
│   └── rendering/                 # Renders
│
├── run.py                          # Main orchestrator
├── requirements.txt                # Python dependencies
└── INSTALLATION.md                 # This file
```

## Troubleshooting

**Missing SAM model:**
```bash
cd segmentor
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
```

**Module import errors:**
Ensure you're running from project root with mamba environment:
```bash
cd /path/to/3D-RE-GEN
mamba run -p ./venv_py310 python run.py -p 1
```

**Submodules not initialized:**
```bash
git submodule update --init --recursive
```

**Version conflicts:**
Recreate the environment (see "Setting Up Mamba Environment" section).
```
