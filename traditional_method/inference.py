import os
import torch
import cv2
import numpy as np
import argparse
import json
from train_classifier import FeatureExtractor, CharacterMLP


class CharacterClassifier:
    """
    Character classifier for inference.
    """
    
    def __init__(self, model_path, config_path):
        """
        Initialize the classifier with trained model and configuration.
        
        Args:
            model_path: Path to the trained model (.pth file)
            config_path: Path to the training configuration (.json file)
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Load model
        self.model = CharacterMLP(
            input_size=self.config['input_size'],
            hidden_sizes=self.config['hidden_sizes'],
            num_classes=self.config['num_classes'],
            dropout_rate=0.0  # No dropout during inference
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Label names
        self.label_names = self.config['label_names']
        
        print(f"Model loaded successfully!")
        print(f"Classes: {self.label_names}")
    
    def predict_image(self, image_path):
        """
        Predict the class of a single image.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        # Load and preprocess image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize to 28x28 if needed
        if image.shape != (28, 28):
            image = cv2.resize(image, (28, 28))
        
        # Extract features
        features = self.feature_extractor.extract_features(image)
        
        # Convert to tensor and add batch dimension
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(features_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = self.label_names[predicted_idx.item()]
        confidence_score = confidence.item()
        all_probs = probabilities.squeeze().cpu().numpy()
        
        return predicted_class, confidence_score, all_probs
    
    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images.
        
        Args:
            image_paths: List of image paths
        
        Returns:
            List of (predicted_class, confidence) tuples
        """
        results = []
        
        for image_path in image_paths:
            try:
                pred_class, confidence, _ = self.predict_image(image_path)
                results.append((pred_class, confidence))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append((None, 0.0))
        
        return results
    
    def get_top_k_predictions(self, image_path, k=5):
        """
        Get top-k predictions for an image.
        
        Args:
            image_path: Path to the image file
            k: Number of top predictions to return
        
        Returns:
            List of (class_name, probability) tuples, sorted by probability
        """
        _, _, all_probs = self.predict_image(image_path)
        
        # Get top-k indices
        top_k_indices = np.argsort(all_probs)[-k:][::-1]
        
        top_k_predictions = []
        for idx in top_k_indices:
            class_name = self.label_names[idx]
            probability = all_probs[idx]
            top_k_predictions.append((class_name, probability))
        
        return top_k_predictions


def main():
    parser = argparse.ArgumentParser(description='Inference with trained character classifier')
    parser.add_argument('--model_path', default='best_model.pth', 
                       help='Path to trained model file (default: best_model.pth)')
    parser.add_argument('--config_path', default='training_config.json',
                       help='Path to training configuration file (default: training_config.json)')
    parser.add_argument('--image_path', help='Path to single image for prediction')
    parser.add_argument('--image_folder', help='Path to folder containing images for batch prediction')
    parser.add_argument('--top_k', type=int, default=5, help='Show top-k predictions (default: 5)')
    
    args = parser.parse_args()
    
    # Check if model and config files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        return
    
    if not os.path.exists(args.config_path):
        print(f"Error: Config file '{args.config_path}' not found.")
        return
    
    # Initialize classifier
    classifier = CharacterClassifier(args.model_path, args.config_path)
    
    if args.image_path:
        # Single image prediction
        if not os.path.exists(args.image_path):
            print(f"Error: Image file '{args.image_path}' not found.")
            return
        
        print(f"\nPredicting class for: {args.image_path}")
        
        try:
            # Get top-k predictions
            top_predictions = classifier.get_top_k_predictions(args.image_path, args.top_k)
            
            print(f"\nTop {args.top_k} predictions:")
            for i, (class_name, probability) in enumerate(top_predictions, 1):
                print(f"{i}. {class_name}: {probability:.4f} ({probability*100:.2f}%)")
            
            # Main prediction
            pred_class, confidence, _ = classifier.predict_image(args.image_path)
            print(f"\nPredicted class: {pred_class} (confidence: {confidence:.4f})")
            
        except Exception as e:
            print(f"Error during prediction: {e}")
    
    elif args.image_folder:
        # Batch prediction
        if not os.path.isdir(args.image_folder):
            print(f"Error: Folder '{args.image_folder}' not found.")
            return
        
        print(f"\nProcessing images in folder: {args.image_folder}")
        
        # Get all image files
        image_files = []
        for file in os.listdir(args.image_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_files.append(os.path.join(args.image_folder, file))
        
        if not image_files:
            print("No image files found in the folder.")
            return
        
        print(f"Found {len(image_files)} image files")
        
        # Predict for all images
        results = classifier.predict_batch(image_files)
        
        print("\nPrediction Results:")
        print("-" * 60)
        print(f"{'Filename':<30} {'Predicted':<15} {'Confidence':<12}")
        print("-" * 60)
        
        correct_predictions = 0
        total_predictions = 0
        
        for image_path, (pred_class, confidence) in zip(image_files, results):
            filename = os.path.basename(image_path)
            
            if pred_class is not None:
                print(f"{filename:<30} {pred_class:<15} {confidence:.4f}")
                total_predictions += 1
                
                # If the filename contains the true label, check accuracy
                # This assumes filenames like "a_001.png" or folder structure
                true_label = None
                for label in classifier.label_names:
                    if label.lower() in filename.lower():
                        true_label = label
                        break
                
                if true_label and true_label == pred_class:
                    correct_predictions += 1
            else:
                print(f"{filename:<30} {'ERROR':<15} {'N/A':<12}")
        
        print("-" * 60)
        
        if correct_predictions > 0:
            accuracy = correct_predictions / total_predictions * 100
            print(f"Accuracy (based on filenames): {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
    
    else:
        print("Please provide either --image_path for single image or --image_folder for batch prediction")
        print("\nExamples:")
        print("  Single image: python inference.py --image_path path/to/image.png")
        print("  Batch:        python inference.py --image_folder path/to/images/")


if __name__ == "__main__":
    main()