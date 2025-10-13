import os
import argparse
import shutil
from pathlib import Path


def process_folders(input_folder, output_folder):
    """
    Process folders to rename and organize files based on folder name length.
    
    Args:
        input_folder (str): Path to the input folder containing folders to process
        output_folder (str): Path to the output folder where files will be organized
    
    Returns:
        dict: Statistics about the processing
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        print(f"Error: Input folder '{input_path}' does not exist.")
        return None
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_folders = 0
    processed_folders = 0
    skipped_folders = 0
    total_files_processed = 0
    
    # Iterate through all items in the input folder
    for item in input_path.iterdir():
        # Skip if not a directory or if it's the debug folder
        if not item.is_dir() or item.name == "debug":
            continue
        
        total_folders += 1
        folder_name = item.name
        
        # Extract name part from format <name>-0
        if folder_name.endswith("-0"):
            name_part = folder_name[:-2]  # Remove "-0" suffix
        else:
            # If folder doesn't follow the expected format, use the whole name
            name_part = folder_name
        
        # Get all files in the folder
        files = [f for f in item.iterdir() if f.is_file()]
        file_count = len(files)
        name_length = len(name_part)
        
        print(f"Processing folder: {folder_name}")
        print(f"  Name part: '{name_part}' (length: {name_length})")
        print(f"  File count: {file_count}")
        
        # Check if name length equals file count
        if name_length == file_count:
            print(f"  ✓ Length matches - processing files...")
            processed_folders += 1
            
            # Process each file
            for i, file_path in enumerate(files):
                try:
                    # Get the first 3 characters of the filename (without extension)
                    file_stem = file_path.stem
                    idx = file_stem[:3] if len(file_stem) >= 3 else file_stem
                    
                    # Get the character at position i from the folder name
                    if i < len(name_part):
                        char_i = name_part[i]
                        
                        # Create output subfolder named after the character
                        output_subfolder = output_path / char_i
                        output_subfolder.mkdir(parents=True, exist_ok=True)
                        
                        # Create new filename
                        new_filename = f"{name_part}_{idx}.png"
                        output_file_path = output_subfolder / new_filename
                        
                        # Copy the file to the new location with new name
                        shutil.copy2(file_path, output_file_path)
                        
                        print(f"    File {i+1}: {file_path.name} -> {char_i}/{new_filename}")
                        total_files_processed += 1
                    else:
                        print(f"    Warning: File index {i} exceeds name part length for {file_path.name}")
                        
                except Exception as e:
                    print(f"    Error processing file {file_path.name}: {e}")
            
        else:
            print(f"  ✗ Length mismatch - skipping folder")
            skipped_folders += 1
        
        print()  # Empty line for better readability
    
    # Print summary statistics
    print("=" * 60)
    print("PROCESSING SUMMARY:")
    print(f"Total folders found: {total_folders}")
    print(f"Folders processed: {processed_folders}")
    print(f"Folders skipped (length mismatch): {skipped_folders}")
    print(f"Total files processed: {total_files_processed}")
    print(f"Output folder: {output_path}")
    print("=" * 60)
    
    return {
        'total_folders': total_folders,
        'processed_folders': processed_folders,
        'skipped_folders': skipped_folders,
        'total_files_processed': total_files_processed
    }


def main():
    """Main function to handle command line arguments and run the processing."""
    parser = argparse.ArgumentParser(
        description="Process folders to rename and organize files based on folder name length"
    )
    parser.add_argument(
        "input_folder",
        help="Path to the input folder containing folders to process"
    )
    parser.add_argument(
        "output_folder",
        help="Path to the output folder where files will be organized"
    )
    
    args = parser.parse_args()
    
    # Run the processing
    result = process_folders(args.input_folder, args.output_folder)
    
    if result is None:
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
