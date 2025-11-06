import os
import glob
import torch
import cv2
import numpy as np
import argparse
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import defaultdict  # ✅ added for grouping
from train_classifier import FeatureExtractor, CharacterMLP

class CharacterClassifier:
    """
    Character classifier for inference.
    """
    def __init__(self, model_path, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        use_spatial = self.config.get("use_spatial", True)
        use_freq    = self.config.get("use_freq", True)
        use_texture = self.config.get("use_texture", True)


        # Initialize FeatureExtractor with the correct combination
        self.feature_extractor = FeatureExtractor(
            use_spatial=use_spatial,
            use_freq=use_freq,
            use_texture=use_texture
        )
                        
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
    
    def predict_folder(self, folder_path):
        """
        Predict all images in subfolders. Subfolder name = true label.
        Returns: y_true, y_pred, image_paths
        """
        y_true = []
        y_pred = []
        image_paths = []  # ✅ store for CAPTCHA grouping later

        # Iterate over subfolders
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            # Collect images
            image_files = [os.path.join(subfolder_path, f) 
                           for f in os.listdir(subfolder_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            for image_path in tqdm(image_files, desc=f"Processing {subfolder}", leave=False):
                try:
                    pred_class, _ = self.predict_image(image_path)
                    y_pred.append(pred_class)
                    y_true.append(subfolder)
                    image_paths.append(image_path)  # ✅ keep path
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
        return y_true, y_pred, image_paths  # ✅ return paths


def compute_captcha_accuracy(y_true, y_pred, image_paths):
    """
    ✅ Compute CAPTCHA-level accuracy by grouping filenames with the same CAPTCHA ID.
    Expected filename format: captchaID_position.png
    """
    captchas = defaultdict(list)
    true_captchas = defaultdict(list)

    for true_label, pred_label, path in zip(y_true, y_pred, image_paths):
        filename = os.path.basename(path)
        captcha_id = filename.split('_')[0]  # part before '_'
        position_part = filename.split('_')[1].split('.')[0]
        try:
            position = int(position_part)
        except ValueError:
            position = 0
        captchas[captcha_id].append((position, pred_label))
        true_captchas[captcha_id].append((position, true_label))
    
    correct = 0
    total = len(captchas)

    for cid in captchas:
        pred_seq = ''.join([c for _, c in sorted(captchas[cid], key=lambda x: x[0])])
        true_seq = ''.join([c for _, c in sorted(true_captchas[cid], key=lambda x: x[0])])
        if pred_seq == true_seq:
            correct += 1

    return correct / total if total > 0 else 0, correct, total


def main():
    parser = argparse.ArgumentParser(description="Batch inference with metrics")
    parser.add_argument("--model_folder", required=True,
                        help="Folder containing best_model_*.pth and training_config.json")
    parser.add_argument("--test_folder", required=True,
                        help="Folder containing test images in subfolders (subfolder = label)")
    args = parser.parse_args()

    model_folder = args.model_folder
    test_folder = args.test_folder

    # Find latest best_model_*.pth
    model_files = glob.glob(os.path.join(model_folder, "best_model_*.pth"))
    if not model_files:
        print("No best_model_*.pth found in model folder.")
        return
    model_path = max(model_files, key=os.path.getmtime)
    print(f"Using model: {model_path}")

    # Find training_config.json
    config_path = os.path.join(model_folder, "training_config.json")
    if not os.path.exists(config_path):
        print("training_config.json not found in model folder.")
        return

    classifier = CharacterClassifier(model_path, config_path)

    # Predict all images in test folder subfolders
    y_true, y_pred, image_paths = classifier.predict_folder(test_folder)

    if not y_true:
        print("No images found in test folder subdirectories.")
        return

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    # ✅ Compute CAPTCHA-level accuracy
    captcha_acc, captcha_correct, captcha_total = compute_captcha_accuracy(y_true, y_pred, image_paths)

    print(f"\nResults on test folder: {test_folder}")
    print(f"{'-'*50}")
    print(f"Character Accuracy:    {acc:.4f}")
    print(f"Weighted Precision:    {precision_w:.4f}")
    print(f"Weighted Recall:       {recall_w:.4f}")
    print(f"Weighted F1-score:     {f1_w:.4f}")
    print(f"Macro Precision:       {precision_m:.4f}")
    print(f"Macro Recall:          {recall_m:.4f}")
    print(f"Macro F1-score:        {f1_m:.4f}")
    print(f"\n✅ CAPTCHA Accuracy:   {captcha_acc:.4f} ({captcha_correct}/{captcha_total})")

if __name__ == "__main__":
    main()
