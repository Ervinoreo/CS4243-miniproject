#!/usr/bin/env python3
"""
YOLOv8 Inference Script for CAPTCHA Character Detection
This script loads a trained YOLOv8 model and performs inference on images,
drawing bounding boxes and saving the results.
"""

import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

def perform_inference_with_visualization(
    model_path,
    input_folder,
    output_folder="inference",
    conf_threshold=0.25,
    iou_threshold=0.6,
    max_images=None
):
    """
    Perform inference on images and save results with bounding boxes
    
    Args:
        model_path: Path to the trained model (.pt file)
        input_folder: Folder containing input images
        output_folder: Folder to save inference results
        conf_threshold: Confidence threshold for detection
        iou_threshold: IoU threshold for NMS
        max_images: Maximum number of images to process (None for all)
    """
    
    print(f"=== YOLOv8 Inference with Visualization ===")
    print(f"Model: {model_path}")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return
    
    # Check if input folder exists
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"❌ Error: Input folder not found at {input_folder}")
        return
    
    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)
    print(f"✅ Output folder created: {output_path.absolute()}")
    
    # Load the trained model
    print(f"\n📦 Loading model...")
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    # Limit number of images if specified
    if max_images:
        image_files = image_files[:max_images]
    
    if len(image_files) == 0:
        print(f"❌ No image files found in {input_folder}")
        return
    
    print(f"📸 Found {len(image_files)} images to process")
    
    # Process each image
    successful_inferences = 0
    failed_inferences = 0
    
    for i, image_path in enumerate(image_files, 1):
        try:
            print(f"\n🔍 Processing image {i}/{len(image_files)}: {image_path.name}")
            
            # Run inference
            results = model(
                str(image_path),
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            # Get the first result (should only be one image)
            result = results[0]
            
            # Create output filename
            output_filename = f"inference_{i:03d}_{image_path.name}"
            output_file_path = output_path / output_filename
            
            # Save the image with bounding boxes drawn
            annotated_img = result.plot(
                conf=True,          # Show confidence scores
                labels=True,        # Show class labels
                boxes=True,         # Show bounding boxes
                line_width=2,       # Bounding box line width
                font_size=12        # Font size for labels
            )
            
            # Save the annotated image
            cv2.imwrite(str(output_file_path), annotated_img)
            
            # Print detection info
            detections = result.boxes
            if detections is not None and len(detections) > 0:
                num_detections = len(detections)
                confidences = detections.conf.cpu().numpy()
                avg_conf = np.mean(confidences)
                print(f"   ✅ Detected {num_detections} characters (avg conf: {avg_conf:.3f})")
                print(f"   💾 Saved: {output_filename}")
            else:
                print(f"   ⚠️  No detections found")
                print(f"   💾 Saved: {output_filename}")
            
            successful_inferences += 1
            
        except Exception as e:
            print(f"   ❌ Error processing {image_path.name}: {e}")
            failed_inferences += 1
    
    # Summary
    print(f"\n📊 === Inference Summary ===")
    print(f"✅ Successful: {successful_inferences}")
    print(f"❌ Failed: {failed_inferences}")
    print(f"📁 Results saved to: {output_path.absolute()}")
    print(f"🎯 Use confidence threshold {conf_threshold} and IoU threshold {iou_threshold}")

def main():
    """Main inference function"""
    
    # Configuration
    MODEL_PATH = "captcha_detection/yolov8n_captcha_v1/weights/best.pt"  # Update this path
    INPUT_FOLDER = "CAPTCHA.v1-v1.yolov8/train/images"  # Update this path
    OUTPUT_FOLDER = "inference"
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.6
    MAX_IMAGES = 20  # Set to None to process all images, or specify a number
    
    print("=== YOLOv8 CAPTCHA Inference Script ===")
    print(f"🔧 Configuration:")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Input: {INPUT_FOLDER}")
    print(f"   Output: {OUTPUT_FOLDER}")
    print(f"   Max images: {MAX_IMAGES if MAX_IMAGES else 'All'}")
    
    # Check if paths exist
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found! Please check the path:")
        print(f"   Expected: {MODEL_PATH}")
        print(f"   Make sure you've trained the model first")
        return
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"\n❌ Input folder not found! Please check the path:")
        print(f"   Expected: {INPUT_FOLDER}")
        return
    
    # Run inference
    perform_inference_with_visualization(
        model_path=MODEL_PATH,
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        max_images=MAX_IMAGES
    )

if __name__ == "__main__":
    main()