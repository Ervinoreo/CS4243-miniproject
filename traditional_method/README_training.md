# Character Classification Training

This project trains a Multi-Layer Perceptron (MLP) classifier for 36-class character recognition (0-9, a-z) using combined feature extraction methods.

## Feature Extraction Methods

The classifier uses three types of feature extraction:

1. **Local Spatial Features** (`local_spatial_feature.py`)
   - Low-pass, High-pass, Gaussian, Laplacian, Sobel, Median, Butterworth filters
   - Statistical features: mean, std, min, max, median for each filter

2. **Frequency Domain Features** (`freq_domain_feature.py`)
   - Fourier Transform, Walsh-Hadamard Transform, KLT, Wavelet, Gabor, Power Spectrum
   - Statistical features extracted from each transform

3. **Texture Features** (`texture_feature.py`)
   - GLCM, LBP, Texton, Autocorrelation, PCA, MSMD analysis
   - Statistical features from each texture method

4. **Raw Pixel Features**
   - Basic image statistics as baseline features

## Installation

1. Install the required packages:
```bash
pip install -r requirements_yolo.txt
```

2. Ensure you have the feature extraction modules in the `traditional_method/` folder:
   - `local_spatial_feature.py`
   - `freq_domain_feature.py` 
   - `texture_feature.py`

## Data Format

Your input folder should have the following structure:
```
input_folder/
├── 0/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── 1/
│   ├── image1.png
│   └── ...
├── a/
│   ├── image1.png
│   └── ...
├── b/
│   └── ...
└── z/
    └── ...
```

- Each subfolder name represents a class label
- Images should be 28x28 pixels, grayscale (black and white)
- Supported formats: PNG, JPG, JPEG

## Training

### Basic Usage

```bash
python train_classifier.py data/labeled_medium_resized_28x28
```

### Advanced Options

```bash
python train_classifier.py data/labeled_medium_resized_28x28 \
    --batch_size 128 \
    --epochs 200 \
    --learning_rate 0.0005 \
    --hidden_sizes 1024 512 256 128 \
    --dropout_rate 0.4 \
    --test_size 0.15 \
    --val_size 0.15
```

### Parameters

- `input_folder`: Path to folder containing subfolders with images (required)
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of training epochs (default: 100)  
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--hidden_sizes`: List of hidden layer sizes (default: [512, 256, 128])
- `--dropout_rate`: Dropout rate for regularization (default: 0.3)
- `--test_size`: Fraction for test set (default: 0.2)
- `--val_size`: Fraction for validation set (default: 0.1)
- `--random_state`: Random seed for reproducibility (default: 42)

## Model Architecture

The MLP model consists of:
- Input layer (feature vector size determined automatically)
- Multiple hidden layers with BatchNorm, ReLU activation, and Dropout
- Output layer with 36 units (for 36 classes)
- Softmax activation for probability distribution

Feature vector composition:
- Local spatial features: 7 filters × 5 statistics = 35 features
- Frequency domain features: 6 transforms × 5 statistics = 30 features  
- Texture features: 6 methods × 5 statistics = 30 features
- Raw pixel features: 8 statistics
- **Total: ~103 features**

## Training Process

1. **Data Loading**: Images are loaded from the folder structure
2. **Feature Extraction**: All features are pre-extracted for efficiency
3. **Data Split**: Train/validation/test split with stratification
4. **Training**: Adam optimizer with learning rate scheduling
5. **Early Stopping**: Stops if validation accuracy doesn't improve for 20 epochs
6. **Model Saving**: Best model based on validation accuracy is saved

## Output Files

After training, the following files are generated:

- `best_model.pth`: Trained model weights
- `training_config.json`: Configuration and results
- `training_history.png`: Loss and accuracy plots
- `confusion_matrix.png`: Test set confusion matrix

## GPU Support

The script automatically uses CUDA if available. To check GPU usage:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
```

## Inference

After training, use the inference script to test the model:

```bash
# Single image prediction
python inference.py --image_path path/to/test_image.png

# Batch prediction
python inference.py --image_folder path/to/test_images/

# Top-k predictions
python inference.py --image_path test.png --top_k 3
```

## Performance Tips

1. **Batch Size**: Increase if you have enough GPU memory (128, 256)
2. **Learning Rate**: Start with 0.001, decrease if training is unstable
3. **Hidden Layers**: Deeper networks may work better with more data
4. **Dropout**: Increase (0.4-0.5) if overfitting, decrease if underfitting
5. **Feature Engineering**: The current features are comprehensive, but you can modify the extraction parameters in `FeatureExtractor.__init__()`

## Monitoring Training

The script provides detailed output including:
- Real-time progress bars for each epoch
- Training and validation metrics
- Early stopping notifications
- Best model saving alerts
- Learning rate adjustments

## Troubleshooting

1. **CUDA Out of Memory**: Reduce batch_size
2. **Training Too Slow**: Increase batch_size, reduce feature extraction complexity
3. **Poor Accuracy**: Try deeper networks, different learning rates, more epochs
4. **Overfitting**: Increase dropout_rate, add more data
5. **Underfitting**: Reduce dropout_rate, increase model capacity

## Example Training Session

```bash
# Activate your environment if using one
# conda activate your_env  # or source venv/bin/activate

# Run training
python train_classifier.py data/labeled_medium_resized_28x28 \
    --batch_size 64 \
    --epochs 150 \
    --learning_rate 0.001 \
    --hidden_sizes 512 256 128

# Expected output:
# Using device: cuda
# GPU: NVIDIA GeForce RTX 3080
# Loading data from: data/labeled_medium_resized_28x28
# Total images loaded: 12000
# Unique labels: ['0', '1', ..., '9', 'a', 'b', ..., 'z']
# Pre-extracting features...
# 100%|██████████| 12000/12000 [02:15<00:00, 88.5it/s]
# Training set: 8640 images  
# Validation set: 1080 images
# Test set: 2160 images
# Feature vector size: 103
# Model created with 36 classes
# Starting training on cuda
# ...
```