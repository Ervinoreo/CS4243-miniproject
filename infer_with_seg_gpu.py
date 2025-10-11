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
import torch

def perform_inference_with_visualization(
    model_path,
    input_folder,
    output_folder="inference",
    conf_threshold=0.25,
    iou_threshold=0.6,
    max_images=None,
    device="auto"
):
    """
    Perform inference on images, save results with bounding boxes, and perform segmentation
    
    Args:
        model_path: Path to the trained model (.pt file)
        input_folder: Folder containing input images
        output_folder: Folder to save inference results and segmented characters
        conf_threshold: Confidence threshold for detection
        iou_threshold: IoU threshold for NMS
        max_images: Maximum number of images to process (None for all)
        device: Device to use for inference ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
    """
    
    # Detect and configure device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            device = "cpu"
            print("💻 Using CPU (no GPU available)")
    else:
        print(f"🔧 Using specified device: {device}")
    
    print(f"=== YOLOv8 Inference with Visualization ===")
    print(f"Model: {model_path}")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Device: {device}")
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
    
    # Create output folder structure
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create debug folder for images with bounding boxes
    debug_path = output_path / "debug"
    debug_path.mkdir(exist_ok=True, parents=True)
    
    print(f"✅ Output folder created: {output_path.absolute()}")
    print(f"✅ Debug folder created: {debug_path.absolute()}")
    
    # Load the trained model
    print(f"\n📦 Loading model...")
    try:
        model = YOLO(model_path)
        # Move model to specified device
        model.to(device)
        print(f"✅ Model loaded successfully on {device}")
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
            
            # Load original image
            original_img = cv2.imread(str(image_path))
            if original_img is None:
                print(f"   ❌ Error: Could not load image {image_path.name}")
                failed_inferences += 1
                continue
            
            # Run inference
            results = model(
                str(image_path),
                conf=conf_threshold,
                iou=iou_threshold,
                device=device,
                verbose=False
            )
            
            # Get the first result (should only be one image)
            result = results[0]
            
            # Create image-specific folder for segmented characters
            image_stem = image_path.stem  # filename without extension
            image_output_folder = output_path / image_stem
            image_output_folder.mkdir(exist_ok=True, parents=True)
            
            # Save the image with bounding boxes drawn to debug folder
            annotated_img = result.plot(
                conf=True,          # Show confidence scores
                labels=True,        # Show class labels
                boxes=True,         # Show bounding boxes
                line_width=2,       # Bounding box line width
                font_size=12        # Font size for labels
            )
            
            debug_filename = f"debug_{i:03d}_{image_path.name}"
            debug_file_path = debug_path / debug_filename
            cv2.imwrite(str(debug_file_path), annotated_img)
            
            # Process detections and perform segmentation
            detections = result.boxes
            if detections is not None and len(detections) > 0:
                num_detections = len(detections)
                confidences = detections.conf.cpu().numpy()
                avg_conf = np.mean(confidences)
                
                # Get bounding boxes in xyxy format
                boxes = detections.xyxy.cpu().numpy()
                
                # Sort boxes by x-coordinate (left to right)
                sorted_indices = np.argsort(boxes[:, 0])
                
                segmented_count = 0
                for idx, box_idx in enumerate(sorted_indices):
                    x1, y1, x2, y2 = boxes[box_idx].astype(int)
                    confidence = confidences[box_idx]
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(original_img.shape[1], x2)
                    y2 = min(original_img.shape[0], y2)
                    
                    # Extract the character region
                    if x2 > x1 and y2 > y1:  # Valid bounding box
                        character_img = original_img[y1:y2, x1:x2]
                        
                        # Save segmented character
                        char_filename = f"char_{idx+1:02d}_conf_{confidence:.3f}.png"
                        char_path = image_output_folder / char_filename
                        
                        if cv2.imwrite(str(char_path), character_img):
                            segmented_count += 1
                        else:
                            print(f"      ⚠️  Failed to save character {idx+1}")
                
                print(f"   ✅ Detected {num_detections} characters (avg conf: {avg_conf:.3f})")
                print(f"   ✂️  Segmented {segmented_count} characters")
                print(f"   💾 Debug image saved: {debug_filename}")
                print(f"   📁 Characters saved to: {image_stem}/")
            else:
                print(f"   ⚠️  No detections found")
                print(f"   💾 Debug image saved: {debug_filename}")
            
            successful_inferences += 1
            
        except Exception as e:
            print(f"   ❌ Error processing {image_path.name}: {e}")
            failed_inferences += 1
    
    # Summary
    print(f"\n📊 === Inference and Segmentation Summary ===")
    print(f"✅ Successful: {successful_inferences}")
    print(f"❌ Failed: {failed_inferences}")
    print(f"📁 Results saved to: {output_path.absolute()}")
    print(f"🐛 Debug images (with bounding boxes) saved to: {debug_path.absolute()}")
    print(f"✂️  Segmented characters saved in individual image folders")
    print(f"🎯 Used confidence threshold {conf_threshold} and IoU threshold {iou_threshold}")
    print(f"🚀 Processing completed on {device}")

def main():
    """Main inference function"""
    
    # Configuration
    MODEL_PATH = "captcha_detection/yolov8n_captcha_v1/weights/best.pt"  # Update this path
    INPUT_FOLDER = "data/train/"  # Update this path
    OUTPUT_FOLDER = "output_with_segmentation"
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.6
    MAX_IMAGES = None  # Set to None to process all images, or specify a number
    DEVICE = "auto"  # Options: "auto" (detect best), "cpu", "cuda", "cuda:0", etc.
    
    print("=== YOLOv8 CAPTCHA Inference Script ===")
    print(f"🔧 Configuration:")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Input: {INPUT_FOLDER}")
    print(f"   Output: {OUTPUT_FOLDER}")
    print(f"   Device: {DEVICE}")
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
        max_images=MAX_IMAGES,
        device=DEVICE
    )

if __name__ == "__main__":
    main()
