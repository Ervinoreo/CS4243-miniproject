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
