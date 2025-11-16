# CAPTCHA Recognition with Deep Learning

This project implements various deep learning approaches for CAPTCHA character recognition, including baseline models and ensemble methods. The system can recognize individual characters and complete CAPTCHA sequences using different architectures and combination strategies.

## 📁 Project Structure

```
miniproject/
├── data/                           # CAPTCHA sequence dataset
│   ├── train/                      # Training CAPTCHA images
│   └── test/                       # Test CAPTCHA images
├── char_dataset/                   # Character-level dataset
│   ├── labeled_train/              # Training character images (organized by class)
│   │   ├── 0/                      # Digit 0 images
│   │   ├── 1/                      # Digit 1 images
│   │   ├── a/                      # Letter 'a' images
│   │   └── ...                     # Other characters (0-9, a-z)
│   └── labeled_test/               # Test character images (organized by class)
├── baselines/                      # Baseline and ensemble models
│   ├── baseline-cnn.py             # CNN on CAPTCHA sequences
│   ├── baseline-resnet.py          # ResNet-50 on CAPTCHA sequences
│   ├── baseline-char-cnn.py        # CNN on individual characters
│   ├── baseline-char-resnet.py     # ResNet-50 on individual characters
│   ├── baseline-char-vgg16.py      # VGG16 on individual characters
│   ├── baseline-char-mlp.py        # MLP on individual characters
│   ├── ensemble-tree.py            # Tree-based meta-learning ensemble
│   └── ensemble-add.py             # Weighted averaging ensemble
├── traditional_method/             # Handcrafted feature extraction methods
│   ├── train_single_layer_classifier.py    # Single-layer MLP classifier
│   ├── train_multilayer_classifier.py      # Multi-layer MLP classifier
│   ├── local_spatial_feature.py    # Local spatial feature extraction
│   ├── freq_domain_feature.py      # Frequency domain feature extraction
│   ├── texture_feature.py          # Texture feature extraction
│   └── inference.py                # Inference script for trained models
├── preprocess/                     # Data preprocessing utilities
│   ├── bounding_box.py             # Character detection and bounding box extraction
│   ├── detect_connected_components.py # Connected component analysis for segmentation
│   ├── color_analysis.py           # Color space analysis and conversion utilities
│   ├── process_images.py           # Main preprocessing pipeline
│   └── utils.py                    # Common preprocessing utilities
├── scripts/                        # SLURM job submission scripts
│   ├── run_train_traditional.sh    # Submit traditional method training jobs
│   ├── run_train_multilayer.sh     # Submit multi-layer training jobs
│   ├── run_inference.sh            # Submit inference jobs
│   └── run_label_data.sh           # Submit data labeling jobs
├── label_data_seq.py               # Sequential data labeling script
├── label_data_par.py               # Parallel data labeling script
├── seg_performance_seq.py          # Sequential segmentation performance analysis
├── segmentation_performance_par.py # Parallel segmentation performance analysis
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate

# Activate virtual environment (Windows)
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Data

You can download the annotated data [here](https://drive.google.com/file/d/147ZS3WWMwTf-WC53U-yyHbUWJMT4jyUD/view?usp=sharing)

You can download the labeled train data [here](https://drive.google.com/file/d/1Y98elIocUrzXw7s5tYlVDhgUcu8mbBHE/view?usp=sharing), and the labeled test data [here](https://drive.google.com/file/d/16bZt6_axAAsSYWLsMWYbzNm8RFBpPx6O/view?usp=sharing)

### 4. SLURM Job Submission

This project includes SLURM scripts for running jobs on high-performance computing clusters. The scripts are located in the `scripts/` folder and can be used to submit training and inference jobs to the cluster scheduler.

**Available SLURM Scripts:**
- `run_train_traditional.sh`: Submit traditional method training jobs
- `run_train_multilayer.sh`: Submit multi-layer classifier training jobs
- `run_inference.sh`: Submit inference jobs
- `run_label_data.sh`: Submit data labeling jobs

**Usage:**
```bash
# Submit a training job
sbatch scripts/run_train_traditional.sh traditional_method/train_single_layer_classifier.py data/labeled_train_resized_28x28
```

## 🔄 Data Preprocessing

Before training any models, the data needs to be preprocessed and organized properly. The preprocessing pipeline includes several steps for individual character extraction.

### Preprocessing Scripts

The `preprocess/` folder contains utility scripts:

- `bounding_box.py`: Character detection and bounding box extraction
- `detect_connected_components.py`: Connected component analysis for segmentation  
- `color_analysis.py`: Color space analysis and conversion utilities
- `process_images.py`: Main preprocessing pipeline
- `utils.py`: Common preprocessing utilities

### Data Quality Checks

Before preprocessing, verify data quality:

```bash
python generation/quick_check.py data/train
```

This script checks:
- Image dimensions and format consistency
- Class distribution balance
- Corrupted or invalid image files
- Proper folder organization

### Character-Level Preprocessing

For character-level models, individual characters need to be extracted and labeled:

#### 1. **Character Segmentation**
```bash
python preprocess/process_images.py --input_folder data/train --output_folder data/segmented_chars
```

This script:
- Detects and segments individual characters from CAPTCHA images
- Uses connected component analysis and bounding box detection
- Saves each character as a separate image file

#### 2. **Segmentation Quality Analysis**
Before labeling, analyze the quality of the segmentation process:

```bash
# Sequential analysis
python seg_performance_seq.py data/segmented_chars

# Parallel analysis (faster for large datasets)
python segmentation_performance_par.py data/segmented_chars --verbose
```

These scripts:
- Check if the number of segmented characters matches the expected CAPTCHA length
- Identify folders where segmentation may have failed
- Generate statistics on segmentation success rates
- Help identify which CAPTCHAs need manual review

#### 3. **Manual Labeling** 
```bash
# Sequential labeling
python label_data_seq.py data/segmented_chars data/labeled_chars

# Parallel labeling (recommended for large datasets)
python label_data_par.py data/segmented_chars data/labeled_chars --threads 8
```

Interactive labeling tools to:
- Process segmented character images automatically based on folder names
- Organize characters into class-specific folders (0-9, a-z)
- Handle parallel processing for faster labeling of large datasets
- Generate reports on skipped folders due to segmentation issues

#### 4. **Image Resizing and Standardization**
```bash
python traditional_method/resize_images.py --input_folder data/labeled_chars --output_folder data/labeled_chars_28x28 --size 28
```

Standardizes character images:
- Resize to 28×28 pixels for traditional methods

#### 5. **Data Organization**

The preprocessed data will be organized into the following structure:
```
data/
├── labeled_train_resized_28x28/     # For traditional methods
│   ├── 0/                           # Contains all '0' character images
│   ├── 1/                           # Contains all '1' character images
│   ├── a/                           # Contains all 'a' character images
│   └── ...                          # Other characters (b-z, 2-9)
└── labeled_test_resized_28x28/             
```

## 🧠 Baseline Models

The project includes two types of baseline models: **sequence-level** models (trained on complete CAPTCHA images) and **character-level** models (trained on individual character images).

### Sequence-Level Models (CAPTCHA Dataset)

These models process complete CAPTCHA images and output the entire character sequence using CTC loss for sequence-to-sequence learning.

#### 1. **Baseline CNN** (`baseline-cnn.py`)

A custom CNN architecture with bidirectional LSTM for sequence modeling.

**Architecture:**

- Input: Grayscale images (200×80)
- 4 Convolutional layers (32, 64, 128, 256 filters)
- Batch normalization and max pooling after each conv layer
- Bidirectional LSTM (2 layers, 256 hidden units)
- CTC loss for sequence alignment

**Usage:**

```bash
python baselines/baseline-cnn.py
```

#### 2. **Baseline ResNet-50** (`baseline-resnet.py`)

Pre-trained ResNet-50 adapted for CAPTCHA sequence recognition.

**Architecture:**

- Input: RGB images (200×80)
- Pre-trained ResNet-50 backbone
- Bidirectional LSTM (2 layers, 256 hidden units)
- CTC loss for sequence alignment

**Usage:**

```bash
python baselines/baseline-resnet.py
```

---

### Character-Level Models (Character Dataset)

These models classify individual characters (36 classes: 0-9, a-z) extracted from CAPTCHA images. After training, they predict each character independently and aggregate results to form complete CAPTCHA predictions.

#### 3. **Character-Level CNN** (`baseline-char-cnn.py`)

Lightweight CNN for individual character classification.

**Architecture:**

- Input: Grayscale images (32×32)
- 4 Convolutional layers (32, 64, 128, 256 filters)
- Batch normalization and max pooling
- 3 Fully connected layers (512, 128, 36)
- Dropout (0.5) for regularization
- Cross-entropy loss

**Usage:**

```bash
python baselines/baseline-char-cnn.py
```

#### 4. **Character-Level ResNet-50** (`baseline-char-resnet.py`)

Transfer learning with pre-trained ResNet-50 for character classification.

**Architecture:**

- Input: RGB images (224×224)
- Pre-trained ResNet-50 backbone
- Custom classifier head (512, 36 classes)
- Dropout (0.5, 0.3)
- Cross-entropy loss

**Usage:**

```bash
python baselines/baseline-char-resnet.py
```

#### 5. **Character-Level VGG16** (`baseline-char-vgg16.py`)

Transfer learning with pre-trained VGG16 for character classification.

**Architecture:**

- Input: RGB images (224×224)
- Pre-trained VGG16 backbone
- Custom classifier head (4096, 2048, 512, 36)
- Dropout (0.5, 0.5, 0.3)
- Cross-entropy loss

**Usage:**

```bash
python baselines/baseline-char-vgg16.py
```

#### 6. **Character-Level MLP** (`baseline-char-mlp.py`)

Simple multilayer perceptron baseline for comparison.

**Architecture:**

- Input: Flattened grayscale images (32×32 = 1024)
- 3 Hidden layers (512, 256, 128)
- Dropout (0.5, 0.3, 0.3)
- Output layer (36 classes)
- Cross-entropy loss

**Usage:**

```bash
python baselines/baseline-char-mlp.py
```

---

## Handcrafted Feature Extraction Methods

This project supports handcrafted feature extraction methods for character classification. These methods include:

### 1. Local Spatial Features
- Filters: Low-pass, High-pass, Gaussian, Laplacian, Sobel, Median, Butterworth
- Statistical features: mean, std, min, max, median for each filter

### 2. Frequency Domain Features
- Transforms: Fourier, Walsh-Hadamard, KLT, Wavelet, Gabor, Power Spectrum
- Statistical features extracted from each transform

### 3. Texture Features
- Methods: GLCM, LBP, Texton, Autocorrelation, PCA, MSMD analysis
- Statistical features from each texture method

### 4. Raw Pixel Features
- Basic image statistics as baseline features

### Training Methods

#### Single-Layer Classifier
The Single-Layer Classifier uses a simpler architecture for character classification. It is suitable for smaller datasets or when computational resources are limited.

**Basic Usage:**
```bash
python traditional_method/train_single_layer_classifier.py data/labeled_train_resized_28x28
```

**Advanced Options:**
```bash
python traditional_method/train_single_layer_classifier.py data/labeled_train_resized_28x28 \
    --batch_size 128 \
    --epochs 200 \
    --learning_rate 0.0005 \
    --hidden_sizes 512 \
    --dropout_rate 0.4 \
    --test_size 0.15 \
    --val_size 0.15
```

**Parameters:**
- `input_folder`: Path to folder containing subfolders with images (required)
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--hidden_sizes`: Size of the single hidden layer (default: 512)
- `--dropout_rate`: Dropout rate for regularization (default: 0.3)
- `--test_size`: Fraction for test set (default: 0.2)
- `--val_size`: Fraction for validation set (default: 0.1)
- `--random_state`: Random seed for reproducibility (default: 42)

#### Multi-Layer Classifier
The Multi-Layer Classifier uses a deeper architecture with multiple hidden layers. It is designed for larger datasets and more complex feature extraction.

**Basic Usage:**
```bash
python traditional_method/train_multilayer_classifier.py data/labeled_train_resized_28x28
```

**Advanced Options:**
```bash
python traditional_method/train_multilayer_classifier.py data/labeled_train_resized_28x28 \
    --batch_size 128 \
    --epochs 200 \
    --learning_rate 0.0005 \
    --hidden_sizes 1024 512 256 128 \
    --dropout_rate 0.4 \
    --test_size 0.15 \
    --val_size 0.15
```

**Parameters:**
- `input_folder`: Path to folder containing subfolders with images (required)
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--hidden_sizes`: List of hidden layer sizes (default: [512, 256, 128])
- `--dropout_rate`: Dropout rate for regularization (default: 0.3)
- `--test_size`: Fraction for test set (default: 0.2)
- `--val_size`: Fraction for validation set (default: 0.1)
- `--random_state`: Random seed for reproducibility (default: 42)

---

## 🤝 Ensemble Methods

Ensemble methods combine predictions from multiple character-level models (CNN, ResNet-50, VGG16) to improve overall accuracy. All ensemble models operate on the **character dataset**.

### 1. **Tree-Based Meta-Learning Ensemble** (`ensemble-tree.py`)

Uses a meta-learner (decision tree or logistic regression) to combine base model predictions.

**How It Works:**

- Trains three base models (CNN, ResNet50, VGG16) independently on character images
- Collects softmax probability outputs (36 classes) from each model on the validation set
- Concatenates probability vectors (108 features: 36 × 3 models) as meta-features
- Trains a meta-learner (logistic regression with StandardScaler) on these features
- Meta-learner learns to weight and combine base model predictions optimally

**Key Features:**

- Uses full probability distributions (not just predictions) for richer information
- Regularized logistic regression prevents overfitting
- Can capture non-linear interactions between model predictions

**Usage:**

```bash
python baselines/ensemble-tree.py
```

### 2. **Weighted Averaging Ensemble** (`ensemble-add.py`)

Tests three weighted averaging strategies simultaneously in a single run.

**Ensemble Methods Tested:**

#### a) **Simple Average Ensemble**

- Assigns equal weights (1/3, 1/3, 1/3) to all three models
- Averages softmax probability distributions before final prediction
- Most straightforward approach, assumes all models contribute equally
- Robust against individual model biases

#### b) **Weighted Average Ensemble**

- Assigns weights based on individual validation accuracies
- Models with higher accuracy get proportionally more influence
- Weights are normalized to sum to 1
- More adaptive than simple averaging while remaining interpretable

#### c) **Learned Weights Ensemble**

- Uses numerical optimization (scipy.minimize) to find optimal weights
- Maximizes ensemble accuracy on the validation set
- Constrained optimization: weights sum to 1 and are non-negative
- Learns the best linear combination of model predictions
- May be more sensitive to validation set characteristics

**How Weighted Averaging Works:**

1. Each model outputs softmax probabilities for 36 classes
2. Ensemble combines probabilities using weighted average:
   ```
   P_ensemble(class_i) = w1 × P_CNN(i) + w2 × P_ResNet(i) + w3 × P_VGG(i)
   ```
3. Final prediction: `argmax(P_ensemble)`
4. Resulting probabilities still sum to 1 (valid probability distribution)

**Usage:**

```bash
python baselines/ensemble-add.py
```

**Output:**
The script will test all three methods and display:

- Validation accuracies for each method
- Optimal weights learned for each approach
- Character-level and CAPTCHA-level accuracy comparisons
- Improvement over best individual model
- Identifies the best performing ensemble method automatically

---

## 📊 Evaluation Metrics

All models report the following metrics:

### Character-Level Metrics:

- **Accuracy**: Percentage of correctly classified characters
- **Precision**: Weighted and macro-averaged precision
- **Recall**: Weighted and macro-averaged recall
- **F1-Score**: Weighted and macro-averaged F1-score

### CAPTCHA-Level Metrics:

- **CAPTCHA Accuracy**: Percentage of completely correct CAPTCHA predictions
  - A CAPTCHA is considered correct only if ALL characters are predicted correctly
  - For length-6 CAPTCHA: if per-character accuracy is 0.90, CAPTCHA accuracy ≈ 0.90^6 = 0.53

---

---

## 📝 Notes

- **GPU Recommended**: Training on GPU is highly recommended for ResNet-50 and VGG16 models
- **Data Requirements**: Character-level models require pre-segmented character images

---

## 📚 References

- **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (2016)
- **VGG**: Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition" (2015)
- **Ensemble Learning**: Zhou, "Ensemble Methods: Foundations and Algorithms" (2012)

---

## 📧 Contact

For questions or issues, please open an issue in the repository.
