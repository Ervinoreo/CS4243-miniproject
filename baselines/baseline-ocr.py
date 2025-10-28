#!/usr/bin/env python3
"""
TrOCR-based CAPTCHA Recognition Model
This script implements a TrOCR model to recognize text in CAPTCHA images using transformers.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
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

# Transformers imports
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    default_data_collator
)
import evaluate

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class CaptchaDataProcessor:
    """Process CAPTCHA images and labels for TrOCR training"""
    
    def __init__(self):
        self.characters = string.digits + string.ascii_lowercase  # 0-9, a-z
        
    def extract_label_from_filename(self, filename):
        """Extract the CAPTCHA text from filename"""
        # Remove .png extension and -0 suffix
        basename = filename.replace('.png', '').replace('-0', '')
        # Clean up any remaining numbers at the end and keep only alphanumeric
        label = re.sub(r'[^a-z0-9]', '', basename.lower())
        return label
        
    def load_skipped_files(self, skipped_file_path):
        """Load list of files to skip from skipped_folders.txt"""
        skipped_files = set()
        if os.path.exists(skipped_file_path):
            with open(skipped_file_path, 'r') as f:
                for line in f:
                    filename = line.strip()
                    if filename:
                        skipped_files.add(filename)
        return skipped_files
    
    def create_dataframe_from_folder(self, data_dir, skipped_files=None):
        """Create DataFrame from CAPTCHA images in folder"""
        data_dir = Path(data_dir)
        if skipped_files is None:
            skipped_files = set()
            
        file_data = []
        
        # Process all PNG files in the directory
        for img_path in data_dir.glob("*.png"):
            filename = img_path.name
            
            # Skip files that are in the skipped list
            if filename in skipped_files:
                continue
                
            # Extract text from filename using the robust method
            text = self.extract_label_from_filename(filename)
            
            # Skip if no valid text extracted or empty
            if not text:
                continue
            
            # Validate that text contains only allowed characters
            if all(c in self.characters for c in text):
                file_data.append({
                    'file_name': filename,
                    'text': text,
                    'file_path': str(img_path)
                })
        
        return pd.DataFrame(file_data)
    
    def load_data(self, train_dir, test_dir, char_dataset_dir):
        """Load training and test data with skipped files handling"""
        
        # Load skipped files for training and testing
        train_skipped_path = Path(char_dataset_dir) / "labeled_train" / "skipped_folders.txt" 
        test_skipped_path = Path(char_dataset_dir) / "labeled_test" / "skipped_folders.txt"
        
        train_skipped = self.load_skipped_files(train_skipped_path)
        test_skipped = self.load_skipped_files(test_skipped_path)
        
        print(f"Loading training data from {train_dir}")
        print(f"Skipping {len(train_skipped)} training files")
        train_df = self.create_dataframe_from_folder(train_dir, train_skipped)
        
        print(f"Loading test data from {test_dir}")
        print(f"Skipping {len(test_skipped)} test files")
        test_df = self.create_dataframe_from_folder(test_dir, test_skipped)
        
        # Split training data for validation
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
        
        # Reset indices
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        return train_df, val_df, test_df

class CaptchaDataset(Dataset):
    """PyTorch Dataset for CAPTCHA images with TrOCR processor"""
    
    def __init__(self, df, processor, max_target_length=20):
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get file path and text 
        file_path = self.df.iloc[idx]['file_path']
        text = self.df.iloc[idx]['text']
        
        # Prepare image (resize + normalize)
        image = Image.open(file_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.max_target_length,
            truncation=True
        ).input_ids
        
        # Important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {
            "pixel_values": pixel_values.squeeze(), 
            "labels": torch.tensor(labels)
        }
        return encoding

class TrOCRCaptchaModel:
    """TrOCR model wrapper for CAPTCHA recognition"""
    
    def __init__(self, model_checkpoint="microsoft/trocr-base-handwritten"):
        self.model_checkpoint = model_checkpoint
        self.processor = TrOCRProcessor.from_pretrained(model_checkpoint)
        self.model = None
        self.trainer = None
        
    def setup_model(self):
        """Setup the TrOCR model with proper configuration"""
        # Load pre-trained model
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
        
        # Set special tokens used for creating the decoder_input_ids from the labels
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        
        # Make sure vocab size is set correctly
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        # Set beam search parameters
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.max_length = 20
        self.model.config.early_stopping = True
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 2
        
        return self.model
    
    def setup_training(self, train_dataset, eval_dataset, output_dir="./trocr_captcha_model"):
        """Setup training configuration"""
        
        # Setup CER metric for evaluation
        cer_metric = evaluate.load("cer", trust_remote_code=True)
        
        def compute_metrics(pred):
            labels_ids = pred.label_ids
            pred_ids = pred.predictions

            pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
            labels_ids[labels_ids == -100] = self.processor.tokenizer.pad_token_id
            label_str = self.processor.batch_decode(labels_ids, skip_special_tokens=True)

            cer = cer_metric.compute(predictions=pred_str, references=label_str)
            return {"cer": cer}
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            eval_strategy="steps",
            per_device_train_batch_size=8,  # Reduced for memory
            per_device_eval_batch_size=8,
            fp16=True if torch.cuda.is_available() else False,
            output_dir=output_dir,
            logging_steps=10,
            save_steps=500,
            eval_steps=100,
            num_train_epochs=20,
            report_to="none",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
        )
        
        # Instantiate trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            tokenizer=self.processor.feature_extractor,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
        )
        
        return self.trainer
    
    def train(self):
        """Train the model"""
        if self.trainer is None:
            raise ValueError("Training not setup. Call setup_training first.")
        
        print("Starting TrOCR training...")
        self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        print("Training completed and model saved!")
    
    def predict_single(self, image_path):
        """Predict text for a single image"""
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        
        # Move to GPU if available
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
            self.model = self.model.cuda()
        
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    
    def predict_batch(self, test_df):
        """Predict text for all images in test DataFrame"""
        predictions = []
        
        print("Generating predictions...")
        for idx in tqdm(range(len(test_df))):
            image_path = test_df.iloc[idx]['file_path']
            try:
                pred_text = self.predict_single(image_path)
                predictions.append(pred_text)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                predictions.append("")  # Empty prediction on error
                
        return predictions

def evaluate_comprehensive(test_df, predictions, processor_chars):
    """Comprehensive evaluation with character and sequence level metrics"""
    
    true_texts = test_df['text'].tolist()
    
    # Sequence-level accuracy (exact match)
    sequence_accuracy = sum(1 for true, pred in zip(true_texts, predictions) if true == pred) / len(true_texts)
    
    # Character-level metrics
    all_true_chars = []
    all_pred_chars = []
    
    # Collect all characters for character-level evaluation
    max_len = max(max(len(true), len(pred)) for true, pred in zip(true_texts, predictions))
    
    for true_text, pred_text in zip(true_texts, predictions):
        # Pad sequences to same length for character-wise comparison
        true_padded = true_text.ljust(max_len, ' ')
        pred_padded = pred_text.ljust(max_len, ' ')
        
        for true_char, pred_char in zip(true_padded, pred_padded):
            if true_char != ' ':  # Don't count padding characters
                all_true_chars.append(true_char)
                all_pred_chars.append(pred_char if pred_char != ' ' else '')
    
    # Character-level accuracy
    char_accuracy = sum(1 for true, pred in zip(all_true_chars, all_pred_chars) if true == pred) / len(all_true_chars)
    
    # Character-level precision, recall, F1 (weighted and macro)
    char_precision_weighted, char_recall_weighted, char_f1_weighted, _ = precision_recall_fscore_support(
        all_true_chars, all_pred_chars, average='weighted', zero_division=0
    )
    char_precision_macro, char_recall_macro, char_f1_macro, _ = precision_recall_fscore_support(
        all_true_chars, all_pred_chars, average='macro', zero_division=0
    )
    
    # Calculate CER (Character Error Rate)
    cer_metric = evaluate.load("cer", trust_remote_code=True)
    cer = cer_metric.compute(predictions=predictions, references=true_texts)
    
    results = {
        'sequence_accuracy': sequence_accuracy,
        'sequence_correct': sum(1 for true, pred in zip(true_texts, predictions) if true == pred),
        'sequence_total': len(true_texts),
        'char_accuracy': char_accuracy,
        'char_correct': sum(1 for true, pred in zip(all_true_chars, all_pred_chars) if true == pred),
        'char_total': len(all_true_chars),
        'char_precision_weighted': char_precision_weighted,
        'char_recall_weighted': char_recall_weighted,
        'char_f1_weighted': char_f1_weighted,
        'char_precision_macro': char_precision_macro,
        'char_recall_macro': char_recall_macro,
        'char_f1_macro': char_f1_macro,
        'cer': cer,
        'predictions': predictions,
        'true_texts': true_texts
    }
    
    return results

def visualize_predictions(test_df, predictions, num_samples=10):
    """Visualize some predictions"""
    plt.figure(figsize=(15, 3 * num_samples))
    
    sample_indices = np.random.choice(len(test_df), min(num_samples, len(test_df)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        plt.subplot(num_samples, 1, i + 1)
        
        # Load and display image
        image_path = test_df.iloc[idx]['file_path']
        img = Image.open(image_path)
        plt.imshow(img)
        
        # Get texts
        true_text = test_df.iloc[idx]['text']
        pred_text = predictions[idx]
        
        # Set title with color coding
        color = 'green' if true_text == pred_text else 'red'
        plt.title(f'True: "{true_text}" | Pred: "{pred_text}"', color=color, fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('./trocr_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Prediction visualizations saved to './trocr_predictions.png'")

def main():
    """Main function"""
    # Configuration
    TRAIN_DIR = "./data/train"
    TEST_DIR = "./data/test"
    CHAR_DATASET_DIR = "./char_dataset"
    MODEL_CHECKPOINT = "microsoft/trocr-base-handwritten"
    OUTPUT_DIR = "./trocr_captcha_model"
    
    print("=== TrOCR CAPTCHA Recognition Model ===")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Test directory: {TEST_DIR}")
    print(f"Character dataset directory: {CHAR_DATASET_DIR}")
    print(f"Model: {MODEL_CHECKPOINT}")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize data processor
    processor = CaptchaDataProcessor()
    
    # Load data
    print("\nLoading data...")
    train_df, val_df, test_df = processor.load_data(TRAIN_DIR, TEST_DIR, CHAR_DATASET_DIR)
    
    # Display data info
    print(f"\nData distribution:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples") 
    print(f"  Test: {len(test_df)} samples")
    
    # Display text length statistics
    train_lengths = train_df['text'].str.len()
    print(f"\nCAPTCHA text length statistics:")
    print(f"  Min: {train_lengths.min()}, Max: {train_lengths.max()}")
    print(f"  Mean: {train_lengths.mean():.1f}, Std: {train_lengths.std():.1f}")
    
    # Initialize TrOCR model
    print("\nInitializing TrOCR model...")
    trocr_model = TrOCRCaptchaModel(MODEL_CHECKPOINT)
    model = trocr_model.setup_model()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = CaptchaDataset(train_df, trocr_model.processor)
    val_dataset = CaptchaDataset(val_df, trocr_model.processor)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Setup training
    print("Setting up training...")
    trainer = trocr_model.setup_training(train_dataset, val_dataset, OUTPUT_DIR)
    
    # Train the model
    print("\nStarting training...")
    trocr_model.train()
    
    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    best_model_path = OUTPUT_DIR
    trocr_model.model = VisionEncoderDecoderModel.from_pretrained(best_model_path)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions = trocr_model.predict_batch(test_df)
    
    # Comprehensive evaluation
    print("Computing evaluation metrics...")
    results = evaluate_comprehensive(test_df, predictions, processor.characters)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"FINAL TEST RESULTS")
    print(f"{'='*60}")
    
    print(f"\nSequence-Level Metrics:")
    print(f"  Accuracy: {results['sequence_accuracy']:.4f} ({results['sequence_correct']}/{results['sequence_total']})")
    print(f"  Character Error Rate (CER): {results['cer']:.4f}")
    
    print(f"\nCharacter-Level Metrics:")
    print(f"  Accuracy: {results['char_accuracy']:.4f} ({results['char_correct']}/{results['char_total']})")
    print(f"  Precision (weighted): {results['char_precision_weighted']:.4f}")
    print(f"  Recall (weighted): {results['char_recall_weighted']:.4f}")
    print(f"  F1-Score (weighted): {results['char_f1_weighted']:.4f}")
    print(f"  Precision (macro): {results['char_precision_macro']:.4f}")
    print(f"  Recall (macro): {results['char_recall_macro']:.4f}")
    print(f"  F1-Score (macro): {results['char_f1_macro']:.4f}")
    
    # Show sample predictions
    print(f"\nSample Predictions:")
    for i in range(min(10, len(test_df))):
        true_text = results['true_texts'][i]
        pred_text = results['predictions'][i]
        status = "✓" if true_text == pred_text else "✗"
        print(f"  {status} True: '{true_text}' | Pred: '{pred_text}'")
    
    # Save results
    print("\nSaving results...")
    
    # Save metrics to text file
    with open('./trocr_test_results.txt', 'w') as f:
        f.write("TrOCR CAPTCHA Recognition Test Results\n")
        f.write("="*50 + "\n\n")
        
        f.write("Sequence-Level Metrics:\n")
        f.write(f"  Accuracy: {results['sequence_accuracy']:.4f} ({results['sequence_correct']}/{results['sequence_total']})\n")
        f.write(f"  Character Error Rate (CER): {results['cer']:.4f}\n\n")
        
        f.write("Character-Level Metrics:\n")
        f.write(f"  Accuracy: {results['char_accuracy']:.4f} ({results['char_correct']}/{results['char_total']})\n")
        f.write(f"  Precision (weighted): {results['char_precision_weighted']:.4f}\n")
        f.write(f"  Recall (weighted): {results['char_recall_weighted']:.4f}\n")
        f.write(f"  F1-Score (weighted): {results['char_f1_weighted']:.4f}\n")
        f.write(f"  Precision (macro): {results['char_precision_macro']:.4f}\n")
        f.write(f"  Recall (macro): {results['char_recall_macro']:.4f}\n")
        f.write(f"  F1-Score (macro): {results['char_f1_macro']:.4f}\n\n")
        
        f.write("Sample Predictions:\n")
        for i in range(min(20, len(test_df))):
            true_text = results['true_texts'][i]
            pred_text = results['predictions'][i]
            status = "✓" if true_text == pred_text else "✗"
            f.write(f"  {status} True: '{true_text}' | Pred: '{pred_text}'\n")
    
    # Save detailed results as pickle
    with open('./trocr_test_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("Files saved:")
    print("  - trocr_test_results.txt (human-readable results)")
    print("  - trocr_test_results.pkl (detailed results for analysis)")
    print(f"  - {OUTPUT_DIR}/ (trained model)")
    
    # Visualize predictions
    print("\nGenerating prediction visualizations...")
    visualize_predictions(test_df, predictions, num_samples=10)
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
