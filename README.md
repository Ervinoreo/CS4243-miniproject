# CAPTCHA Character Detection with YOLOv8

This project implements CAPTCHA character detection using YOLOv8 object detection model. The system can detect and localize individual characters in CAPTCHA images with bounding boxes.

## ��� Project Structure

```
miniproject/
├── data/                           # Original CAPTCHA dataset
│   ├── train/                      # Training images (PNG format)
│   │   ├── 002e23-0.png
│   │   ├── 00995l-0.png
│   │   └── ...
│   └── test/                       # Test images (PNG format)
│       ├── 002e23-0.png
│       ├── 00995l-0.png
│       └── ...
├── CAPTCHA.v1-v1.yolov8/          # Annotated dataset for YOLOv8
│   ├── data.yaml                   # Dataset configuration
│   ├── train/
│   │   ├── images/                 # Training images (JPG format)
│   │   └── labels/                 # YOLO format annotations (.txt)
│   ├── valid/
│   │   ├── images/                 # Validation images
│   │   └── labels/                 # Validation annotations
│   └── test/
│       ├── images/                 # Test images
│       └── labels/                 # Test annotations
├── baseline-cnn.py                 # Baseline CNN model implementation
├── baseline-resnet.py              # ResNet-50 baseline model
├── yolov8.py                       # YOLOv8 training script
├── inference.py                    # Inference script for visualization
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Environment Setup

Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Pre-trained Model (Optional)

If you want to skip training and use our pre-trained model:

📥 **Download the pre-trained YOLOv8 model:**

- **Google Drive Link**: https://drive.google.com/file/d/1BpvGWbbkIznqvI5VKY-PzjzoJdiF5JD8/view?usp=sharing
- **File**: `best.pt` (trained YOLOv8 model weights)

**Setup instructions:**

1. Download `best.pt` from the Google Drive link
2. Create the model directory structure:
   ```bash
   mkdir -p captcha_detection/yolov8n_captcha_v1/weights/
   ```
3. Place the downloaded `best.pt` file in:
   ```
   captcha_detection/yolov8n_captcha_v1/weights/best.pt
   ```

**Now you can skip to step 5 (Run Inference) if using the pre-trained model!**

### 4. Train YOLOv8 Model (Skip if using pre-trained model)

```bash
python yolov8.py
```

The training script will:

- Load the annotated dataset from `CAPTCHA.v1-v1.yolov8/`
- Train a YOLOv8 nano model for character detection
- Save the best model weights to `captcha_detection/yolov8n_captcha_v1/weights/best.pt`
- Generate training curves and validation metrics

### 5. Run Inference

To perform inference and visualize results:

1. **Update the model path** in `inference.py` (if needed):

   ```python
   MODEL_PATH = "captcha_detection/yolov8n_captcha_v1/weights/best.pt"
   ```

2. **Set the input folder** (choose one):

   ```python
   # For original test images:
   INPUT_FOLDER = "data/train"

   # For original test images:
   INPUT_FOLDER = "data/test"
   ```

3. **Run inference**:
   ```bash
   python inference.py
   ```

Results will be saved in the `inference/` folder with bounding boxes drawn on the images.


## � Traditional Preprocessing Methods

In addition to the YOLOv8 approach, this project includes comprehensive traditional computer vision preprocessing methods for CAPTCHA character segmentation. These methods use color analysis, connected component detection, and morphological operations to automatically segment individual characters from CAPTCHA images.

### Overview

The traditional preprocessing pipeline implements a multi-stage approach:

1. **Color Mask Creation**: Identifies colored regions by excluding white and black pixels
2. **Mask Smoothening**: Applies averaging filters to reduce noise
3. **Connected Component Detection**: Uses DFS to find character regions
4. **Color Analysis with DBSCAN**: Clusters pixels to identify distinct character colors
5. **Bounding Box Processing**: Merges nearby components and filters by size/density
6. **Character Segmentation**: Extracts individual character segments with color masking

### Usage

Run the traditional preprocessing pipeline on your CAPTCHA images:

```bash
python preprocess/process_unclear_images.py ./data/medium/ -o output_medium -w 250 -b 5 -k 3 -s 3 -m 40 -t 1.1 -p 3 -c 30 -mul 2.0 --size-ratio-threshold 0.4 --large-box-ratio 2.5 --wide-box-color-threshold 30
```

### Parameters Explanation

- `./data/medium/` - Input folder containing CAPTCHA images
- `-o output_medium` - Output folder for processed results
- `-w 250` - White threshold (pixels above this are considered white)
- `-b 5` - Black threshold (pixels below this are considered black)
- `-k 3` - Kernel size for mask smoothening
- `-s 3` - Stride for smoothening operations
- `-m 40` - Minimum area for connected components
- `-t 1.1` - Width threshold for detecting wide bounding boxes
- `-p 3` - Padding around extracted character segments
- `-c 30` - Color similarity threshold for character masking
- `-mul 2.0` - Size multiplier for filtering large boxes
- `--size-ratio-threshold 0.4` - Minimum box size ratio to median
- `--large-box-ratio 2.5` - Maximum box size ratio to median
- `--wide-box-color-threshold 30` - Color threshold for wide box processing

### Output Structure

The preprocessing generates:

```
output_medium/
├── debug/                          # Combined visualization images
│   ├── hyperparameters.json       # Processing parameters used
│   └── [image_name]_combined.png   # Original + mask + bounding boxes
├── [image_name]/                   # Individual character segments
│   ├── valid_000.png              # Characters from DFS detection
│   ├── valid_001.png
│   ├── char_000.png               # Characters from color clustering
│   └── char_001.png
└── ...
```

### Key Features

- **Multi-Color Character Detection**: Uses DBSCAN clustering to identify characters of different colors
- **Adaptive Bounding Box Processing**: Merges nearby components and handles nested characters
- **Size-Based Filtering**: Removes outlier boxes based on statistical analysis
- **Color Masking**: Applies targeted color filtering to improve character clarity
- **Wide Box Handling**: Special processing for boxes containing multiple characters

### Traditional Methods Components

The preprocessing system includes several specialized modules:

- `detect_connected_components.py` - Core DFS-based component detection
- `color_analysis.py` - DBSCAN clustering for color extraction
- `bounding_box.py` - Box merging and filtering algorithms
- `segmentation.py` - Character extraction and segmentation
- `utils.py` - Utility functions and parameter saving