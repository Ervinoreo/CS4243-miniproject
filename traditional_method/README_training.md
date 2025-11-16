# Character Classification Training

This project trains classifiers for 36-class character recognition (0-9, a-z) using combined feature extraction methods. Two training methods are supported: Single-Layer and Multi-Layer classifiers.

## Training Methods

### 1. Single-Layer Classifier
The Single-Layer Classifier uses a simpler architecture for character classification. It is suitable for smaller datasets or when computational resources are limited.

#### Basic Usage
```bash
python train_single_layer_classifier.py data/labeled_medium_resized_28x28
```

#### Advanced Options
```bash
python train_single_layer_classifier.py data/labeled_medium_resized_28x28 \
    --batch_size 128 \
    --epochs 200 \
    --learning_rate 0.0005 \
    --hidden_sizes 512 \
    --dropout_rate 0.4 \
    --test_size 0.15 \
    --val_size 0.15
```

#### Parameters
- `input_folder`: Path to folder containing subfolders with images (required)
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--hidden_sizes`: Size of the single hidden layer (default: 512)
- `--dropout_rate`: Dropout rate for regularization (default: 0.3)
- `--test_size`: Fraction for test set (default: 0.2)
- `--val_size`: Fraction for validation set (default: 0.1)
- `--random_state`: Random seed for reproducibility (default: 42)

### 2. Multi-Layer Classifier
The Multi-Layer Classifier uses a deeper architecture with multiple hidden layers. It is designed for larger datasets and more complex feature extraction.

#### Basic Usage
```bash
python train_multilayer_classifier.py data/labeled_medium_resized_28x28
```

#### Advanced Options
```bash
python train_multilayer_classifier.py data/labeled_medium_resized_28x28 \
    --batch_size 128 \
    --epochs 200 \
    --learning_rate 0.0005 \
    --hidden_sizes 1024 512 256 128 \
    --dropout_rate 0.4 \
    --test_size 0.15 \
    --val_size 0.15
```

#### Parameters
- `input_folder`: Path to folder containing subfolders with images (required)
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--hidden_sizes`: List of hidden layer sizes (default: [512, 256, 128])
- `--dropout_rate`: Dropout rate for regularization (default: 0.3)
- `--test_size`: Fraction for test set (default: 0.2)
- `--val_size`: Fraction for validation set (default: 0.1)
- `--random_state`: Random seed for reproducibility (default: 42)

## Feature Extraction Methods

The classifiers use three types of feature extraction:

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

## Output Files

After training, the following files are generated in the output folder:

- `best_model.pth`: Trained model weights
- `training_config.json`: Configuration and results
- `training_history.png`: Loss and accuracy plots

To generate the graph from training history:
```bash
python plot_training_history.py <folder_name>
```
