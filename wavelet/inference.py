import os
import cv2
import pywt
import numpy as np
import torch
import torch.nn as nn
import argparse
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import json


# Wavelet feature extraction (same as train_classifier.py)
wavelets = ["haar", "db2", "sym2"]
max_level = 3


def extract_wavelet_features(arr):
    """Extract statistical features from wavelet coefficients"""
    arr_abs = np.abs(arr)
    return {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "energy": np.sum(np.square(arr)),
        "entropy": -np.sum(np.where(arr_abs > 0, arr_abs * np.log2(arr_abs + 1e-12), 0)),
        "min": np.min(arr),
        "max": np.max(arr),
    }


def wavelet_features(img, wavelet, levels=max_level):
    """Extract comprehensive wavelet features from an image"""
    coeffs = pywt.wavedec2(img, wavelet, level=levels)
    features = {}
    
    for lvl, coeff in enumerate(coeffs):
        if lvl == 0:
            # LL subband (approximation)
            LL = coeff
            stats = extract_wavelet_features(LL)
            for k, v in stats.items():
                features[f"{wavelet}_L{lvl}_LL_{k}"] = v
        else:
            # Detail subbands (LH, HL, HH)
            LH, HL, HH = coeff
            subbands = {"LH": LH, "HL": HL, "HH": HH}
            for name, band in subbands.items():
                stats = extract_wavelet_features(band)
                for k, v in stats.items():
                    features[f"{wavelet}_L{lvl}_{name}_{k}"] = v
    
    return features


def extract_all_wavelet_features(img_path):
    """Extract all wavelet features from a single image"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    all_features = {}
    
    # Extract features for each wavelet
    for wavelet in wavelets:
        features = wavelet_features(img, wavelet, levels=max_level)
        all_features.update(features)
    
    return all_features


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for classification"""
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_model(model_path, device):
    """Load the trained model and associated components"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration
    config = checkpoint['model_config']
    
    # Create model
    model = MLPClassifier(
        input_size=config['input_size'],
        hidden_sizes=config['hidden_sizes'],
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate']
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint['class_names'], checkpoint['label_encoder'], checkpoint['scaler']


def predict_image(img_path, model, class_names, label_encoder, scaler, device):
    """Predict the class of a single image"""
    # Extract features
    features = extract_all_wavelet_features(img_path)
    if features is None:
        return None, None, None
    
    # Convert to numpy array
    feature_vector = np.array(list(features.values())).reshape(1, -1)
    
    # Standardize features
    feature_vector_scaled = scaler.transform(feature_vector)
    
    # Convert to tensor
    feature_tensor = torch.FloatTensor(feature_vector_scaled).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(feature_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get predicted class name
    predicted_label = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
    confidence_score = confidence.item()
    
    # Get top 3 predictions
    top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
    top3_classes = []
    for i in range(3):
        idx = top3_indices[0][i].item()
        class_name = label_encoder.inverse_transform([idx])[0]
        prob = top3_probs[0][i].item()
        top3_classes.append((class_name, prob))
    
    return predicted_label, confidence_score, top3_classes


def main():
    parser = argparse.ArgumentParser(description="Classify images using trained wavelet MLP model")
    parser.add_argument("model_path", type=str, help="Path to the trained model (.pth file)")
    parser.add_argument("input", type=str, help="Path to input image or folder")
    parser.add_argument("--top_k", type=int, default=3, help="Show top K predictions")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, class_names, label_encoder, scaler = load_model(args.model_path, device)
    print(f"Model loaded. Classes: {class_names}")
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Single image
        print(f"\nClassifying: {args.input}")
        predicted_label, confidence, top3 = predict_image(
            args.input, model, class_names, label_encoder, scaler, device
        )
        
        if predicted_label is not None:
            print(f"Predicted: {predicted_label} (confidence: {confidence:.4f})")
            print("\nTop 3 predictions:")
            for i, (class_name, prob) in enumerate(top3, 1):
                print(f"  {i}. {class_name}: {prob:.4f}")
        else:
            print("Error: Could not process image")
    
    elif os.path.isdir(args.input):
        # Directory of images
        print(f"\nClassifying images in: {args.input}")
        
        image_files = []
        for file in os.listdir(args.input):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(file)
        
        if not image_files:
            print("No image files found in the directory")
            return
        
        print(f"Found {len(image_files)} images")
        
        results = []
        for filename in sorted(image_files):
            img_path = os.path.join(args.input, filename)
            predicted_label, confidence, top3 = predict_image(
                img_path, model, class_names, label_encoder, scaler, device
            )
            
            if predicted_label is not None:
                results.append({
                    'filename': filename,
                    'predicted': predicted_label,
                    'confidence': confidence,
                    'top3': top3
                })
                print(f"{filename}: {predicted_label} ({confidence:.4f})")
            else:
                print(f"{filename}: ERROR - Could not process")
        
        # Save results to JSON
        output_file = os.path.join(args.input, "classification_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()
