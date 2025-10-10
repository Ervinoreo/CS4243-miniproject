# CAPTCHA Character Detection with YOLOv8

This project implements CAPTCHA character detection using YOLOv8 object detection model. The system can detect and localize individual characters in CAPTCHA images with bounding boxes.

## 📁 Project Structure

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
