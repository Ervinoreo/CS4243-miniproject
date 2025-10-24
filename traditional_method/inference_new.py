import os
import torch
import cv2
import numpy as np
import argparse
import json
import glob
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from train_classifier import FeatureExtractor, CharacterMLP

class CharacterClassifier:
    def __init__(self, model_path, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.feature_extractor = FeatureExtractor()
        
        self.model = CharacterMLP(
            input_size=self.config['input_size'],
            hidden_sizes=self.config['hidden_sizes'],
            num_classes=self.config['num_classes'],
            dropout_rate=0.0
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.label_names = self.config['label_names']
        print(f"Model loaded successfully! Classes: {self.label_names}")
    
    def predict_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        if image.shape != (28, 28):
            image = cv2.resize(image, (28, 28))
        
        features = self.feature_extractor.extract_features(image)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(features_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = self.label_names[predicted_idx.item()]
        return predicted_class, confidence.item()
    
    def predict_batch(self, image_paths):
        y_true = []
        y_pred = []
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            pred_class, _ = self.predict_image(image_path)
            y_pred.append(pred_class)
            
            # Try to infer true label from filename
            true_label = None
            for label in self.label_names:
                if label.lower() in filename.lower():
                    true_label = label
                    break
            y_true.append(true_label)
        return y_true, y_pred

def main():
    parser = argparse.ArgumentParser(description="Batch inference with metrics")
    parser.add_argument("--folder", required=True, help="Folder containing model, config, and images")
    args = parser.parse_args()
    folder = args.folder

    # Find model and config
    model_files = glob.glob(os.path.join(folder, "best_model_*.pth"))
    if not model_files:
        print("No best_model_*.pth found in folder.")
        return
    model_path = model_files[0]

    config_path = os.path.join(folder, "training_config.json")
    if not os.path.exists(config_path):
        print("training_config.json not found in folder.")
        return

    classifier = CharacterClassifier(model_path, config_path)

    # Find all images
    image_files = [os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    if not image_files:
        print("No image files found in folder.")
        return

    y_true, y_pred = classifier.predict_batch(image_files)

    # Filter out None true labels
    filtered_y_true = []
    filtered_y_pred = []
    for t, p in zip(y_true, y_pred):
        if t is not None:
            filtered_y_true.append(t)
            filtered_y_pred.append(p)

    if not filtered_y_true:
        print("No valid ground truth labels could be inferred from filenames.")
        return

    # Compute metrics
    acc = accuracy_score(filtered_y_true, filtered_y_pred)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(filtered_y_true, filtered_y_pred, average='weighted')
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(filtered_y_true, filtered_y_pred, average='macro')

    print(f"\nResults on folder: {folder}")
    print(f"{'-'*40}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Weighted): {precision_w:.4f}")
    print(f"Recall (Weighted):    {recall_w:.4f}")
    print(f"F1-score (Weighted):  {f1_w:.4f}")
    print(f"Precision (Macro):    {precision_m:.4f}")
    print(f"Recall (Macro):       {recall_m:.4f}")
    print(f"F1-score (Macro):     {f1_m:.4f}")

if __name__ == "__main__":
    main()
