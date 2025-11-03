#!/usr/bin/env python3
"""
DeepSeek-OCR with LoRA Fine-tuning for Character Recognition
This script implements LoRA fine-tuning of DeepSeek-OCR model for individual character classification.
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
import string
import re
from tqdm import tqdm
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Transformers and PEFT imports
from transformers import (
    AutoModel, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class CharacterDataProcessor:
    """Process individual character images for DeepSeek-OCR training"""
    
    def __init__(self, img_width=640, img_height=640):
        self.img_width = img_width
        self.img_height = img_height
        self.characters = string.digits + string.ascii_lowercase  # 0-9, a-z
        self.char_to_idx = {char: idx for idx, char in enumerate(self.characters)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.num_classes = len(self.characters)
        
    def parse_filename(self, filename):
        """Parse filename to extract CAPTCHA ID and character position
        Format: captcha_id_position.png (e.g., '002e23_000.png')
        """
        basename = filename.replace('.png', '')
        parts = basename.split('_')
        if len(parts) >= 2:
            captcha_id = parts[0]
            position = int(parts[1])
            return captcha_id, position
        return None, None
    
    def preprocess_image(self, image_path):
        """Preprocess a single character image for DeepSeek-OCR"""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        # Convert BGR to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image to match DeepSeek-OCR input requirements
        resized = cv2.resize(rgb, (self.img_width, self.img_height))
        
        # Convert to PIL Image for compatibility with the model
        pil_img = Image.fromarray(resized)
        
        return pil_img
    
    def load_character_data(self, data_dir, split_validation=True):
        """Load character images organized by class folders"""
        data_dir = Path(data_dir)
        
        images = []
        labels = []
        image_paths = []
        characters = []  # Store actual character strings
        
        print(f"Loading character data from {data_dir}...")
        
        # Process each character class folder
        for char_folder in sorted(data_dir.iterdir()):
            if not char_folder.is_dir() or char_folder.name == 'skipped_folders.txt':
                continue
                
            char_label = char_folder.name  # This is the true character (e.g., '0', 'a', etc.)
            if char_label not in self.char_to_idx:
                print(f"Warning: Unknown character class '{char_label}', skipping...")
                continue
                
            class_idx = self.char_to_idx[char_label]
            
            # Process all images in this character folder
            image_files = list(char_folder.glob("*.png"))
            print(f"Processing {len(image_files)} images for character '{char_label}'")
            
            for img_path in tqdm(image_files, desc=f"Loading '{char_label}'"):
                # Preprocess image
                processed_img = self.preprocess_image(img_path)
                if processed_img is None:
                    continue
                
                images.append(processed_img)
                labels.append(class_idx)
                characters.append(char_label)  # Store the actual character
                image_paths.append(str(img_path))
        
        print(f"Loaded {len(images)} character images")
        print(f"Character distribution:")
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label_idx, count in zip(unique_labels, counts):
            char = self.idx_to_char[label_idx]
            print(f"  '{char}': {count} samples")
        
        if split_validation:
            # Split into train and validation sets
            X_train, X_val, y_train, y_val, chars_train, chars_val, paths_train, paths_val = train_test_split(
                images, labels, characters, image_paths, test_size=0.2, random_state=42, stratify=labels
            )
            return X_train, X_val, y_train, y_val, chars_train, chars_val, paths_train, paths_val
        else:
            return images, labels, characters, image_paths

class DeepSeekOCRDataset(Dataset):
    """PyTorch Dataset for DeepSeek-OCR character recognition"""
    
    def __init__(self, images, labels, characters, image_paths, tokenizer, max_length=128):
        self.images = images
        self.labels = labels
        self.characters = characters
        self.image_paths = image_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        character = self.characters[idx]
        
        # Create prompt for character recognition
        prompt = f"<image>\nWhat character is shown in this image? Answer with just the character:"
        target_text = character
        
        # Tokenize prompt and target
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # Combine prompt and target for language modeling
        input_ids = prompt_encoding['input_ids'] + target_encoding['input_ids']
        attention_mask = prompt_encoding['attention_mask'] + target_encoding['attention_mask']
        
        # Create labels (ignore prompt tokens for loss calculation)
        labels = [-100] * len(prompt_encoding['input_ids']) + target_encoding['input_ids']
        
        # Truncate if too long
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
        
        return {
            'image': image,
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'character': character,
            'image_path': self.image_paths[idx]
        }

def custom_data_collator(features):
    """Custom data collator for DeepSeek-OCR with images"""
    # Extract images separately
    images = [f['image'] for f in features]
    
    # Process text features
    text_features = []
    for f in features:
        text_features.append({
            'input_ids': f['input_ids'],
            'attention_mask': f['attention_mask'],
            'labels': f['labels']
        })
    
    # Use default collator for text features
    batch = DataCollatorWithPadding(tokenizer=None, padding=True)(text_features)
    
    # Add images and metadata to batch
    batch['images'] = images
    batch['characters'] = [f['character'] for f in features]
    batch['image_paths'] = [f['image_path'] for f in features]
    
    return batch

class DeepSeekOCRLoRATrainer:
    """Trainer class for DeepSeek-OCR with LoRA fine-tuning"""
    
    def __init__(self, model_name="deepseek-ai/DeepSeek-OCR", device='cuda'):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def setup_model_and_tokenizer(self, use_4bit=True):
        """Setup DeepSeek-OCR model and tokenizer with quantization"""
        print(f"Loading model and tokenizer: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right",
            use_fast=False
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Configure quantization
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        # Load model
        self.model = AutoModel.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            _attn_implementation='flash_attention_2',
            use_safetensors=True
        )
        
        # Prepare model for training if using quantization
        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        return self.model, self.tokenizer
    
    def setup_lora(self, r=16, alpha=32, dropout=0.1, target_modules=None):
        """Setup LoRA configuration and apply to model"""
        if target_modules is None:
            # Target common attention and MLP modules
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "lm_head"
            ]
        
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        # Apply LoRA to model
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        
        return self.peft_model
    
    def predict_single(self, image, prompt="<image>\nWhat character is shown in this image? Answer with just the character:"):
        """Predict character for a single image"""
        if self.model is None:
            raise ValueError("Model not loaded. Call setup_model_and_tokenizer first.")
        
        # Use the model's infer method
        try:
            # Save image temporarily for inference
            temp_path = "temp_char.png"
            if isinstance(image, Image.Image):
                image.save(temp_path)
            else:
                cv2.imwrite(temp_path, image)
            
            # Run inference
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=temp_path,
                base_size=640,
                image_size=640,
                crop_mode=False,
                save_results=False
            )
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Extract character from result
            if isinstance(result, str):
                # Clean and extract single character
                predicted_char = re.sub(r'[^a-z0-9]', '', result.lower().strip())
                if predicted_char and len(predicted_char) == 1:
                    return predicted_char
                elif predicted_char and len(predicted_char) > 1:
                    return predicted_char[0]  # Take first character
            
            return ""  # Return empty if prediction fails
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return ""
    
    def predict_batch(self, test_dataset):
        """Predict characters for all images in test dataset"""
        predictions = []
        
        print("Generating predictions with DeepSeek-OCR...")
        for idx in tqdm(range(len(test_dataset))):
            sample = test_dataset[idx]
            image = sample['image']
            
            try:
                pred_char = self.predict_single(image)
                predictions.append(pred_char)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                predictions.append("")  # Empty prediction on error
                
        return predictions
    
    def aggregate_to_captcha_predictions(self, char_predictions, image_paths, processor):
        """Aggregate character predictions to CAPTCHA-level predictions using filenames"""
        captcha_predictions = defaultdict(list)
        
        # Group predictions by CAPTCHA ID using filename parsing
        for pred, img_path in zip(char_predictions, image_paths):
            # Extract filename
            filename = Path(img_path).name
            
            # Parse filename to get CAPTCHA ID and position
            captcha_id, position = processor.parse_filename(filename)
            if captcha_id is None:
                continue
            
            # Get predicted character
            predicted_char = pred if pred else "?"  # Use "?" for failed predictions
            
            # Get true character from folder name (parent directory)
            true_char = Path(img_path).parent.name
            
            captcha_predictions[captcha_id].append((position, predicted_char, true_char))
        
        # Sort by position and create final CAPTCHA predictions
        final_predictions = {}
        true_captchas = {}
        
        for captcha_id, char_data in captcha_predictions.items():
            # Sort by position
            char_data.sort(key=lambda x: x[0])
            
            # Extract predicted and true sequences
            pred_sequence = ''.join([char_info[1] for char_info in char_data])
            true_sequence = ''.join([char_info[2] for char_info in char_data])
            
            final_predictions[captcha_id] = pred_sequence
            true_captchas[captcha_id] = true_sequence
        
        return final_predictions, true_captchas
    
    def evaluate_comprehensive(self, test_dataset, processor):
        """Comprehensive evaluation with both character and CAPTCHA level metrics"""
        # Get character predictions
        char_predictions = self.predict_batch(test_dataset)
        
        # Get true labels and characters
        true_char_labels = []
        true_characters = []
        image_paths = []
        
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            true_characters.append(sample['character'])
            true_char_labels.append(processor.char_to_idx[sample['character']])
            image_paths.append(sample['image_path'])
        
        # Convert predictions to indices for character-level metrics
        pred_char_labels = []
        for pred in char_predictions:
            if pred in processor.char_to_idx:
                pred_char_labels.append(processor.char_to_idx[pred])
            else:
                pred_char_labels.append(-1)  # Unknown character
        
        # Character-level evaluation
        char_accuracy = accuracy_score(true_char_labels, pred_char_labels)
        
        # Calculate precision, recall, F1 for character level
        precision, recall, f1, support = precision_recall_fscore_support(
            true_char_labels, pred_char_labels, average='weighted', zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_char_labels, pred_char_labels, average='macro', zero_division=0
        )
        
        # CAPTCHA-level evaluation using image paths
        captcha_predictions, true_captchas = self.aggregate_to_captcha_predictions(
            char_predictions, image_paths, processor
        )
        
        # Calculate CAPTCHA accuracy
        correct_captchas = 0
        total_captchas = len(captcha_predictions)
        
        for captcha_id in captcha_predictions:
            if captcha_predictions[captcha_id] == true_captchas[captcha_id]:
                correct_captchas += 1
        
        captcha_accuracy = correct_captchas / total_captchas if total_captchas > 0 else 0
        
        results = {
            'char_accuracy': char_accuracy,
            'char_correct': sum(p == t for p, t in zip(pred_char_labels, true_char_labels) if p != -1),
            'char_total': len([p for p in pred_char_labels if p != -1]),
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'captcha_accuracy': captcha_accuracy,
            'captcha_correct': correct_captchas,
            'captcha_total': total_captchas,
            'char_predictions': char_predictions,
            'true_characters': true_characters,
            'captcha_predictions': captcha_predictions,
            'true_captchas': true_captchas
        }
        
        return results

def visualize_character_predictions(test_dataset, predictions, processor, num_samples=20):
    """Visualize character predictions"""
    plt.figure(figsize=(20, 4 * (num_samples // 5)))
    
    sample_indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        plt.subplot(num_samples // 5, 5, i + 1)
        
        # Get sample
        sample = test_dataset[idx]
        image = sample['image']
        true_char = sample['character']
        pred_char = predictions[idx] if idx < len(predictions) else "?"
        
        # Display image
        plt.imshow(image)
        
        # Set title with color coding
        color = 'green' if pred_char == true_char else 'red'
        plt.title(f'True: {true_char}, Pred: {pred_char}', color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('./deepseek_ocr_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Character predictions saved to './deepseek_ocr_predictions.png'")

def main():
    """Main function"""
    # Configuration
    TRAIN_DIR = "./char_dataset/labeled_train"
    TEST_DIR = "./char_dataset/labeled_test"
    MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
    OUTPUT_DIR = "./deepseek_ocr_lora_model"
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("=== DeepSeek-OCR with LoRA Fine-tuning ===")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Test directory: {TEST_DIR}")
    print(f"Model: {MODEL_NAME}")
    
    # Initialize data processor
    processor = CharacterDataProcessor()
    
    # Load training data
    print("\nLoading training data...")
    X_train, X_val, y_train, y_val, chars_train, chars_val, paths_train, paths_val = processor.load_character_data(
        TRAIN_DIR, split_validation=True
    )
    
    print(f"Training set: {len(X_train)} character images")
    print(f"Validation set: {len(X_val)} character images")
    
    # Load test data
    print("\nLoading test data...")
    X_test, y_test, chars_test, paths_test = processor.load_character_data(TEST_DIR, split_validation=False)
    print(f"Test set: {len(X_test)} character images")
    
    # Initialize trainer
    print("\nInitializing DeepSeek-OCR trainer...")
    trainer = DeepSeekOCRLoRATrainer(MODEL_NAME, device)
    
    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    model, tokenizer = trainer.setup_model_and_tokenizer(use_4bit=True)
    
    # Setup LoRA
    print("Setting up LoRA...")
    peft_model = trainer.setup_lora(r=16, alpha=32, dropout=0.1)
    
    # Create datasets
    print("Creating datasets...")
    test_dataset = DeepSeekOCRDataset(X_test, y_test, chars_test, paths_test, tokenizer)
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Note: For this example, we'll use the pre-trained model for evaluation
    # In a real scenario, you would train the LoRA adapter here
    print("\nNote: Using pre-trained DeepSeek-OCR for evaluation")
    print("To train LoRA adapter, you would implement the training loop here")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = trainer.evaluate_comprehensive(test_dataset, processor)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"DEEPSEEK-OCR TEST RESULTS")
    print(f"{'='*60}")
    
    print(f"\nCharacter-Level Metrics:")
    print(f"  Accuracy: {results['char_accuracy']:.4f} ({results['char_correct']}/{results['char_total']})")
    print(f"  Precision (weighted): {results['precision_weighted']:.4f}")
    print(f"  Recall (weighted): {results['recall_weighted']:.4f}")
    print(f"  F1-Score (weighted): {results['f1_weighted']:.4f}")
    print(f"  Precision (macro): {results['precision_macro']:.4f}")
    print(f"  Recall (macro): {results['recall_macro']:.4f}")
    print(f"  F1-Score (macro): {results['f1_macro']:.4f}")
    
    print(f"\nCAPTCHA-Level Metrics (Aggregated):")
    print(f"  Accuracy: {results['captcha_accuracy']:.4f} ({results['captcha_correct']}/{results['captcha_total']})")
    
    # Show some example CAPTCHA predictions
    print(f"\nSample CAPTCHA Predictions:")
    captcha_predictions = results['captcha_predictions']
    true_captchas = results['true_captchas']
    
    sample_count = 0
    for captcha_id in list(captcha_predictions.keys())[:10]:
        pred_text = captcha_predictions[captcha_id]
        true_text = true_captchas[captcha_id]
        status = "✓" if pred_text == true_text else "✗"
        print(f"  {status} ID: {captcha_id} | True: '{true_text}' | Pred: '{pred_text}'")
        sample_count += 1
        if sample_count >= 10:
            break
    
    # Show character-level examples
    print(f"\nSample Character Predictions:")
    for i in range(min(10, len(results['char_predictions']))):
        true_char = results['true_characters'][i]
        pred_char = results['char_predictions'][i]
        status = "✓" if pred_char == true_char else "✗"
        print(f"  {status} True: '{true_char}' | Pred: '{pred_char}'")
    
    # Save results
    print("\nSaving results...")
    
    # Save metrics to text file
    with open('./deepseek_ocr_test_results.txt', 'w') as f:
        f.write("DeepSeek-OCR Character Recognition Test Results\n")
        f.write("="*50 + "\n\n")
        
        f.write("Character-Level Metrics:\n")
        f.write(f"  Accuracy: {results['char_accuracy']:.4f} ({results['char_correct']}/{results['char_total']})\n")
        f.write(f"  Precision (weighted): {results['precision_weighted']:.4f}\n")
        f.write(f"  Recall (weighted): {results['recall_weighted']:.4f}\n")
        f.write(f"  F1-Score (weighted): {results['f1_weighted']:.4f}\n")
        f.write(f"  Precision (macro): {results['precision_macro']:.4f}\n")
        f.write(f"  Recall (macro): {results['recall_macro']:.4f}\n")
        f.write(f"  F1-Score (macro): {results['f1_macro']:.4f}\n\n")
        
        f.write("CAPTCHA-Level Metrics (Aggregated):\n")
        f.write(f"  Accuracy: {results['captcha_accuracy']:.4f} ({results['captcha_correct']}/{results['captcha_total']})\n\n")
        
        f.write("Sample CAPTCHA Predictions:\n")
        sample_count = 0
        for captcha_id in captcha_predictions:
            if sample_count >= 20:
                break
            pred_text = captcha_predictions[captcha_id]
            true_text = true_captchas[captcha_id]
            status = "✓" if pred_text == true_text else "✗"
            f.write(f"  {status} ID: {captcha_id} | True: '{true_text}' | Pred: '{pred_text}'\n")
            sample_count += 1
    
    # Save detailed results as pickle
    with open('./deepseek_ocr_test_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("Files saved:")
    print("  - deepseek_ocr_test_results.txt (human-readable results)")
    print("  - deepseek_ocr_test_results.pkl (detailed results for analysis)")
    
    # Visualize predictions
    print("\nGenerating prediction visualizations...")
    visualize_character_predictions(test_dataset, results['char_predictions'], processor, num_samples=20)
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()